use std::{
    ffi::{CStr, CString, NulError},
    ptr,
    string::FromUtf8Error,
};

use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase, TensorMut, TensorRef, TensorView,
    ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType},
    metadata::Dynamic,
    versioned,
};
use onyxrt_sys as sys;
use snafu::{Snafu, ensure};

use crate::{
    ffi::{
        self, OutputOwner, ShapeInfoGuard, TypeInfoGuard, ValueGuard, check_status, cstring,
        default_allocator, require_const_non_null, require_non_null,
    },
    model::Environment,
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Ffi {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(display("invalid DLPack tensor: {source}"))]
    DlpackTensor { source: dlpark::tensor::Error },

    #[snafu(display("could not create DLPack output metadata: {source}"))]
    DlpackMetadata { source: dlpark::metadata::Error },

    #[snafu(display(
        "DLPack device {:?}:{} is unsupported; expected CPU:0",
        device_type,
        device_id
    ))]
    UnsupportedDevice {
        device_type: DLDeviceType,
        device_id: i32,
    },

    #[snafu(display(
        "DLPack device {:?}:{} is unsupported for I/O Binding; expected CPU:0 or CUDA:<non-negative>",
        device_type,
        device_id
    ))]
    UnsupportedIoBindingDevice {
        device_type: DLDeviceType,
        device_id: i32,
    },

    #[snafu(display("ONNX Runtime tensor memory `{name}` on device {device_id} is unsupported"))]
    UnsupportedOrtMemory { name: String, device_id: i32 },

    #[snafu(display(
        "unsupported DLPack dtype (code={}, bits={}, lanes={})",
        dtype.code.0,
        dtype.bits,
        dtype.lanes
    ))]
    UnsupportedDlpackType { dtype: DLDataType },

    #[snafu(display(
        "unsupported DLPack flags {flags:?} for dtype (code={}, bits={}, lanes={})",
        dtype.code.0,
        dtype.bits,
        dtype.lanes
    ))]
    UnsupportedDlpackFlags {
        flags: DlpackFlags,
        dtype: DLDataType,
    },

    #[snafu(display("unsupported ONNX tensor element type {element_type:?}"))]
    UnsupportedOnnxType {
        element_type: sys::ONNXTensorElementDataType,
    },

    #[snafu(display("unsupported ONNX value type {value_type:?}; expected a tensor"))]
    UnsupportedValueType { value_type: sys::ONNXType },

    #[snafu(display("ONNX Runtime returned null for {what}"))]
    NullPointer { what: &'static str },

    #[snafu(display("string tensor shape dimension {axis} is negative: {value}"))]
    InvalidStringDimension { axis: usize, value: i64 },

    #[snafu(display("string tensor shape requires {expected} values, got {actual}"))]
    StringElementCount { expected: usize, actual: usize },

    #[snafu(display("string tensor element {index} contains an interior NUL byte"))]
    InvalidStringElement { index: usize, source: NulError },

    #[snafu(display("string tensor element count overflows usize"))]
    StringElementCountOverflow,

    #[snafu(display("string tensor element {index} is not valid UTF-8"))]
    InvalidStringUtf8 { index: usize, source: FromUtf8Error },

    #[snafu(display(
        "string tensor element {index} has invalid byte range {start}..{end} for {data_len} bytes"
    ))]
    InvalidStringOffsets {
        index: usize,
        start: usize,
        end: usize,
        data_len: usize,
    },

    #[snafu(display("{kind} value contained {actual} children; expected {expected}"))]
    ContainerValueCount {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum IoKind {
    Input,
    Output,
    Initializer,
}

/// A named DLPack output returned by [`crate::session::Session::run_all`].
pub struct Output {
    pub(super) name: String,
    pub(super) tensor: versioned::Dlpack,
}

/// A string tensor owned by Rust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringTensor {
    shape: Vec<i64>,
    values: Vec<String>,
}

impl StringTensor {
    /// Returns the tensor shape.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Returns the flattened row-major string values.
    pub fn values(&self) -> &[String] {
        &self.values
    }

    /// Consumes the tensor and returns its flattened values.
    pub fn into_values(self) -> Vec<String> {
        self.values
    }
}

/// A runtime output value supported by the high-level API.
pub enum Value {
    /// A numeric tensor exported through owning DLPack storage.
    Tensor(versioned::Dlpack),
    /// A string tensor copied into Rust-owned UTF-8 strings.
    Strings(StringTensor),
    /// A heterogeneous ONNX sequence.
    Sequence(Vec<Value>),
    /// An ONNX map represented by its key and value tensors.
    Map {
        /// The map key tensor.
        keys: Box<Value>,
        /// The map value tensor or sequence.
        values: Box<Value>,
    },
    /// An ONNX optional value.
    Optional(Option<Box<Value>>),
}

/// A named runtime value returned by [`crate::session::Session::run_values`].
pub struct ValueOutput {
    pub(super) name: String,
    pub(super) value: Value,
}

impl ValueOutput {
    /// Returns the model output name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrows the output value.
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Consumes the named output.
    pub fn into_value(self) -> Value {
        self.value
    }
}

/// An owned string tensor input.
pub struct StringInput {
    name: CString,
    shape: Vec<i64>,
    values: Vec<CString>,
}

/// An unnamed runtime input used as a child of a sequence or map.
pub enum InputValue<'a> {
    /// A borrowed compact CPU DLPack tensor.
    Tensor(TensorRef<'a>),
    /// An owned string tensor.
    Strings(StringValueInput),
    /// A sequence of values with a common ONNX type.
    Sequence(Vec<InputValue<'a>>),
    /// A map represented by key and value tensors or containers.
    Map {
        /// Scalar-key tensor value.
        keys: Box<InputValue<'a>>,
        /// Map values.
        values: Box<InputValue<'a>>,
    },
}

/// An unnamed owned string tensor used inside a container input.
pub struct StringValueInput {
    shape: Vec<i64>,
    values: Vec<CString>,
}

/// A named sequence input.
pub struct SequenceInput<'a> {
    name: CString,
    pub(super) values: Vec<InputValue<'a>>,
}

/// A named map input.
pub struct MapInput<'a> {
    name: CString,
    pub(super) keys: InputValue<'a>,
    pub(super) values: InputValue<'a>,
}

impl StringInput {
    /// Creates a compact string tensor input from flattened row-major values.
    pub fn new<I, S>(name: &str, shape: impl Into<Vec<i64>>, values: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let shape = shape.into();
        let values = string_input_values(&shape, values)?;
        Ok(Self {
            name: cstring(name, "input")?,
            shape,
            values,
        })
    }

    /// Returns the input name.
    pub fn name(&self) -> &CStr {
        &self.name
    }

    /// Returns the tensor shape.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
}

impl<'a> InputValue<'a> {
    /// Borrows a validated DLPack tensor as a container child.
    pub fn from_dlpack<M>(tensor: &'a Managed<M>) -> Result<Self>
    where
        M: ManagedTensorBase,
    {
        let flags = tensor.flags();
        let tensor = tensor
            .validate()
            .map_err(|source| Error::DlpackTensor { source })?;
        validate_cpu_compact(&tensor)?;
        validate_dlpack_flags(flags, tensor.dtype())?;
        dlpack_to_onnx(tensor.dtype())?;
        Ok(Self::Tensor(tensor))
    }

    /// Creates an owned string tensor container child.
    pub fn strings<I, S>(shape: impl Into<Vec<i64>>, values: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        StringValueInput::new(shape, values).map(Self::Strings)
    }

    /// Creates a sequence container child.
    pub fn sequence(values: impl IntoIterator<Item = InputValue<'a>>) -> Self {
        Self::Sequence(values.into_iter().collect())
    }

    /// Creates a map container child.
    pub fn map(keys: InputValue<'a>, values: InputValue<'a>) -> Self {
        Self::Map {
            keys: Box::new(keys),
            values: Box::new(values),
        }
    }
}

impl StringValueInput {
    /// Creates an unnamed compact string tensor.
    pub fn new<I, S>(shape: impl Into<Vec<i64>>, values: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let shape = shape.into();
        let values = string_input_values(&shape, values)?;
        Ok(Self { shape, values })
    }

    /// Returns the tensor shape.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
}

impl<'a> SequenceInput<'a> {
    /// Creates a named sequence input.
    pub fn new(name: &str, values: impl IntoIterator<Item = InputValue<'a>>) -> Result<Self> {
        Ok(Self {
            name: cstring(name, "input")?,
            values: values.into_iter().collect(),
        })
    }

    /// Returns the input name.
    pub fn name(&self) -> &CStr {
        &self.name
    }
}

impl<'a> MapInput<'a> {
    /// Creates a named map input from its key and value components.
    pub fn new(name: &str, keys: InputValue<'a>, values: InputValue<'a>) -> Result<Self> {
        Ok(Self {
            name: cstring(name, "input")?,
            keys,
            values,
        })
    }

    /// Returns the input name.
    pub fn name(&self) -> &CStr {
        &self.name
    }
}

/// A heterogeneous model input accepted by [`crate::session::Session::run_values`].
pub enum ValueInput<'a> {
    /// A borrowed compact CPU DLPack tensor.
    Tensor(Input<'a>),
    /// An owned string tensor.
    Strings(StringInput),
    /// A named sequence.
    Sequence(SequenceInput<'a>),
    /// A named map.
    Map(MapInput<'a>),
}

impl ValueInput<'_> {
    pub(super) fn name(&self) -> &CStr {
        match self {
            Self::Tensor(input) => input.name(),
            Self::Strings(input) => input.name(),
            Self::Sequence(input) => input.name(),
            Self::Map(input) => input.name(),
        }
    }
}

impl Output {
    /// Returns the model output name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrows the owning DLPack handle.
    pub fn tensor(&self) -> &versioned::Dlpack {
        &self.tensor
    }

    /// Consumes the named output and returns its DLPack handle.
    pub fn into_dlpack(self) -> versioned::Dlpack {
        self.tensor
    }
}

/// A named, validated DLPack input borrowed by a session run.
pub struct Input<'a> {
    pub(super) name: CString,
    pub(super) tensor: TensorRef<'a>,
}

/// A named caller-owned DLPack tensor used as an inference output buffer.
pub struct OutputBuffer<'a> {
    pub(super) name: CString,
    pub(super) tensor: TensorMut<'a>,
}

/// A named compact CPU or CUDA DLPack input for ORT I/O Binding.
pub struct IoBindingInput<'a> {
    pub(super) name: CString,
    pub(super) tensor: TensorRef<'a>,
}

/// A named writable compact CPU or CUDA DLPack output for ORT I/O Binding.
pub struct IoBindingOutput<'a> {
    pub(super) name: CString,
    pub(super) tensor: TensorMut<'a>,
}

/// A named output whose allocation and dynamic shape are selected by ORT.
#[derive(Debug, Clone)]
pub struct IoBindingDeviceOutput {
    pub(super) name: CString,
    pub(super) device: DLDevice,
}

impl IoBindingDeviceOutput {
    /// Requests an ORT-allocated CUDA output on `device_id`.
    pub fn cuda(name: &str, device_id: i32) -> Result<Self> {
        ensure!(
            device_id >= 0,
            UnsupportedIoBindingDeviceSnafu {
                device_type: DLDeviceType::CUDA,
                device_id,
            }
        );
        Ok(Self {
            name: cstring(name, "I/O Binding device output")?,
            device: DLDevice {
                device_type: DLDeviceType::CUDA,
                device_id,
            },
        })
    }

    /// Requests an ORT-allocated CPU output.
    pub fn cpu(name: &str) -> Result<Self> {
        Ok(Self {
            name: cstring(name, "I/O Binding device output")?,
            device: DLDevice::CPU,
        })
    }

    /// Returns the output name.
    pub fn name(&self) -> &CStr {
        &self.name
    }

    /// Returns the requested DLPack device.
    pub fn device(&self) -> DLDevice {
        self.device
    }
}

impl<'a> IoBindingInput<'a> {
    /// Validates and borrows a CPU or CUDA DLPack tensor without copying it.
    pub fn from_dlpack<M>(name: &str, tensor: &'a Managed<M>) -> Result<Self>
    where
        M: ManagedTensorBase,
    {
        let flags = tensor.flags();
        let tensor = tensor
            .validate()
            .map_err(|source| Error::DlpackTensor { source })?;
        validate_iobinding_compact(&tensor)?;
        validate_dlpack_flags(flags, tensor.dtype())?;
        dlpack_to_onnx(tensor.dtype())?;
        Ok(Self {
            name: cstring(name, "I/O Binding input")?,
            tensor,
        })
    }

    /// Returns the input name.
    pub fn name(&self) -> &CStr {
        &self.name
    }

    /// Returns the DLPack device of the bound input.
    pub fn device(&self) -> DLDevice {
        self.tensor.device()
    }
}

impl<'a> IoBindingOutput<'a> {
    /// Exclusively borrows a writable CPU or CUDA DLPack output without copying it.
    pub fn from_dlpack<M>(name: &str, tensor: &'a mut Managed<M>) -> Result<Self>
    where
        M: ManagedTensorBase,
    {
        let flags = tensor.flags();
        let tensor = tensor
            .validate_mut()
            .map_err(|source| Error::DlpackTensor { source })?;
        validate_iobinding_compact(&tensor)?;
        validate_dlpack_flags(flags, tensor.dtype())?;
        dlpack_to_onnx(tensor.dtype())?;
        Ok(Self {
            name: cstring(name, "I/O Binding output")?,
            tensor,
        })
    }

    /// Returns the output name.
    pub fn name(&self) -> &CStr {
        &self.name
    }

    /// Returns the DLPack device of the bound output.
    pub fn device(&self) -> DLDevice {
        self.tensor.device()
    }
}

impl<'a> OutputBuffer<'a> {
    /// Exclusively borrows and validates a writable compact CPU DLPack tensor.
    pub fn from_dlpack<M>(name: &str, tensor: &'a mut Managed<M>) -> Result<Self>
    where
        M: ManagedTensorBase,
    {
        let flags = tensor.flags();
        let tensor = tensor
            .validate_mut()
            .map_err(|source| Error::DlpackTensor { source })?;
        validate_cpu_compact(&tensor)?;
        validate_dlpack_flags(flags, tensor.dtype())?;
        dlpack_to_onnx(tensor.dtype())?;
        Ok(Self {
            name: cstring(name, "output")?,
            tensor,
        })
    }

    /// Returns the output name.
    pub fn name(&self) -> &CStr {
        &self.name
    }
}

impl<'a> Input<'a> {
    /// Validates and borrows a legacy or versioned DLPack tensor.
    pub fn from_dlpack<M>(name: &str, tensor: &'a Managed<M>) -> Result<Self>
    where
        M: ManagedTensorBase,
    {
        let flags = tensor.flags();
        let tensor = tensor
            .validate()
            .map_err(|source| Error::DlpackTensor { source })?;
        validate_cpu_compact(&tensor)?;
        validate_dlpack_flags(flags, tensor.dtype())?;
        dlpack_to_onnx(tensor.dtype())?;
        Ok(Self {
            name: cstring(name, "input")?,
            tensor,
        })
    }

    /// Returns the input name.
    pub fn name(&self) -> &CStr {
        &self.name
    }
}

pub(super) fn validate_cpu_compact(tensor: &impl TensorView) -> Result<()> {
    let tensor = tensor.as_raw_tensor();
    let device = tensor.device();
    ensure!(
        device.device_type == DLDeviceType::CPU && device.device_id == 0,
        UnsupportedDeviceSnafu {
            device_type: device.device_type,
            device_id: device.device_id,
        }
    );
    if !tensor
        .is_compact()
        .map_err(|source| Error::DlpackTensor { source })?
    {
        return Err(Error::DlpackTensor {
            source: dlpark::tensor::Error::NonCompactStrides,
        });
    }
    Ok(())
}

pub(super) fn validate_iobinding_compact(tensor: &impl TensorView) -> Result<()> {
    let tensor = tensor.as_raw_tensor();
    let device = tensor.device();
    let supported = (device.device_type == DLDeviceType::CPU && device.device_id == 0)
        || (device.device_type == DLDeviceType::CUDA && device.device_id >= 0);
    ensure!(
        supported,
        UnsupportedIoBindingDeviceSnafu {
            device_type: device.device_type,
            device_id: device.device_id,
        }
    );
    if !tensor
        .is_compact()
        .map_err(|source| Error::DlpackTensor { source })?
    {
        return Err(Error::DlpackTensor {
            source: dlpark::tensor::Error::NonCompactStrides,
        });
    }
    Ok(())
}

pub(super) fn create_dlpack_memory_info(
    api: &sys::Api,
    device: DLDevice,
) -> Result<*mut sys::OrtMemoryInfo> {
    let mut memory_info = ptr::null_mut();
    if device.device_type == DLDeviceType::CPU && device.device_id == 0 {
        // SAFETY: output points to writable storage and enum values are valid.
        check_status(api, unsafe {
            (api.table().CreateCpuMemoryInfo)(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        })?;
    } else if device.device_type == DLDeviceType::CUDA && device.device_id >= 0 {
        // SAFETY: `Cuda` is the allocator name required by ONNX Runtime for
        // CUDA device allocations; output points to writable storage.
        check_status(api, unsafe {
            (api.table().CreateMemoryInfo)(
                c"Cuda".as_ptr(),
                sys::OrtAllocatorType::OrtDeviceAllocator,
                device.device_id,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        })?;
    } else {
        return Err(Error::UnsupportedIoBindingDevice {
            device_type: device.device_type,
            device_id: device.device_id,
        });
    }
    Ok(require_non_null(memory_info, "DLPack memory info")?)
}

pub(super) fn create_external_tensor(
    api: &sys::Api,
    memory_info: *const sys::OrtMemoryInfo,
    tensor: &impl TensorView,
) -> Result<*mut sys::OrtValue> {
    let tensor = tensor.as_raw_tensor();
    let mut value = ptr::null_mut();
    // SAFETY: the descriptor was validated as compact and its byte-offset
    // adjusted address is used only as an opaque pointer for its declared
    // device. Rust never dereferences a CUDA pointer.
    let data =
        unsafe { tensor.offset_bytes_ptr() }.map_err(|source| Error::DlpackTensor { source })?;
    let shape = tensor.shape();
    // SAFETY: memory_info and shape are live, data describes tensor.num_bytes()
    // external bytes, and ORT does not take ownership of the allocation.
    check_status(api, unsafe {
        (api.table().CreateTensorWithDataAsOrtValue)(
            memory_info,
            data.cast_mut().cast(),
            tensor.num_bytes(),
            shape.as_ptr(),
            shape.len(),
            dlpack_to_onnx(tensor.dtype())?,
            &mut value,
        )
    })?;
    Ok(require_non_null(value, "I/O Binding tensor value")?)
}

pub(super) fn validate_dlpack_flags(flags: DlpackFlags, dtype: DLDataType) -> Result<()> {
    ensure!(
        !flags.contains(DlpackFlags::IS_SUBBYTE_TYPE_PADDED),
        UnsupportedDlpackFlagsSnafu { flags, dtype }
    );
    Ok(())
}

pub(super) fn string_element_count(shape: &[i64]) -> Result<usize> {
    shape
        .iter()
        .copied()
        .enumerate()
        .try_fold(1_usize, |count, (axis, value)| {
            let dimension = usize::try_from(value)
                .map_err(|_| Error::InvalidStringDimension { axis, value })?;
            count
                .checked_mul(dimension)
                .ok_or(Error::StringElementCountOverflow)
        })
}

pub(super) fn string_input_values<I, S>(shape: &[i64], values: I) -> Result<Vec<CString>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let expected = string_element_count(shape)?;
    let values = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            CString::new(value.as_ref())
                .map_err(|source| Error::InvalidStringElement { index, source })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        values.len() == expected,
        StringElementCountSnafu {
            expected,
            actual: values.len(),
        }
    );
    Ok(values)
}

pub(super) fn dlpack_to_onnx(dtype: DLDataType) -> Result<sys::ONNXTensorElementDataType> {
    use sys::ONNXTensorElementDataType as O;
    ensure!(dtype.lanes == 1, UnsupportedDlpackTypeSnafu { dtype });
    let value = match (dtype.code, dtype.bits) {
        (DLDataTypeCode::FLOAT, 16) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        (DLDataTypeCode::FLOAT, 32) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        (DLDataTypeCode::FLOAT, 64) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        (DLDataTypeCode::BFLOAT, 16) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        (DLDataTypeCode::INT, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        (DLDataTypeCode::INT, 16) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        (DLDataTypeCode::INT, 32) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        (DLDataTypeCode::INT, 64) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        (DLDataTypeCode::UINT, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        (DLDataTypeCode::UINT, 16) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        (DLDataTypeCode::UINT, 32) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        (DLDataTypeCode::UINT, 64) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        (DLDataTypeCode::BOOL, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        (DLDataTypeCode::COMPLEX, 64) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
        (DLDataTypeCode::COMPLEX, 128) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
        (DLDataTypeCode::FLOAT8_E4M3FN, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
        (DLDataTypeCode::FLOAT8_E4M3FNUZ, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
        (DLDataTypeCode::FLOAT8_E5M2, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
        (DLDataTypeCode::FLOAT8_E5M2FNUZ, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ,
        (DLDataTypeCode::UINT, 4) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
        (DLDataTypeCode::INT, 4) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
        (DLDataTypeCode::FLOAT4_E2M1FN, 4) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1,
        (DLDataTypeCode::UINT, 2) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2,
        (DLDataTypeCode::INT, 2) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2,
        (DLDataTypeCode::FLOAT8_E8M0FNU, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0,
        _ => return Err(Error::UnsupportedDlpackType { dtype }),
    };
    Ok(value)
}

pub(super) fn onnx_to_dlpack(element_type: sys::ONNXTensorElementDataType) -> Result<DLDataType> {
    use sys::ONNXTensorElementDataType as O;
    let (code, bits) = match element_type {
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => (DLDataTypeCode::FLOAT, 16),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => (DLDataTypeCode::FLOAT, 32),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => (DLDataTypeCode::FLOAT, 64),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => (DLDataTypeCode::BFLOAT, 16),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => (DLDataTypeCode::INT, 8),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => (DLDataTypeCode::INT, 16),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => (DLDataTypeCode::INT, 32),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => (DLDataTypeCode::INT, 64),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => (DLDataTypeCode::UINT, 8),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => (DLDataTypeCode::UINT, 16),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => (DLDataTypeCode::UINT, 32),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => (DLDataTypeCode::UINT, 64),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => (DLDataTypeCode::BOOL, 8),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 => (DLDataTypeCode::COMPLEX, 64),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 => (DLDataTypeCode::COMPLEX, 128),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN => (DLDataTypeCode::FLOAT8_E4M3FN, 8),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ => (DLDataTypeCode::FLOAT8_E4M3FNUZ, 8),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 => (DLDataTypeCode::FLOAT8_E5M2, 8),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ => (DLDataTypeCode::FLOAT8_E5M2FNUZ, 8),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 => (DLDataTypeCode::UINT, 4),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 => (DLDataTypeCode::INT, 4),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1 => (DLDataTypeCode::FLOAT4_E2M1FN, 4),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2 => (DLDataTypeCode::UINT, 2),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2 => (DLDataTypeCode::INT, 2),
        O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0 => (DLDataTypeCode::FLOAT8_E8M0FNU, 8),
        _ => return Err(Error::UnsupportedOnnxType { element_type }),
    };
    Ok(DLDataType {
        code,
        bits,
        lanes: 1,
    })
}

pub(super) fn create_string_tensor(
    api: &sys::Api,
    allocator: *mut sys::OrtAllocator,
    input: &StringInput,
) -> Result<*mut sys::OrtValue> {
    create_string_tensor_parts(api, allocator, &input.shape, &input.values)
}

pub(super) fn create_string_tensor_parts(
    api: &sys::Api,
    allocator: *mut sys::OrtAllocator,
    shape: &[i64],
    strings: &[CString],
) -> Result<*mut sys::OrtValue> {
    let mut value = ptr::null_mut();
    // SAFETY: allocator and shape are live; value is writable output storage.
    check_status(api, unsafe {
        (api.table().CreateTensorAsOrtValue)(
            allocator,
            shape.as_ptr(),
            shape.len(),
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
            &mut value,
        )
    })?;
    let value = require_non_null(value, "string input value")?;
    let mut guard = ValueGuard::new(api, value);
    let pointers = strings
        .iter()
        .map(|value| value.as_ptr())
        .collect::<Vec<_>>();
    let pointers_ptr = if pointers.is_empty() {
        ptr::null()
    } else {
        pointers.as_ptr()
    };
    // SAFETY: every pointer is NUL-terminated and remains live for the call;
    // a zero-length tensor uses a null array pointer.
    check_status(api, unsafe {
        (api.table().FillStringTensor)(value, pointers_ptr, pointers.len())
    })?;
    guard.disarm();
    Ok(value)
}

pub(super) fn create_input_value(
    api: &sys::Api,
    memory_info: *const sys::OrtMemoryInfo,
    allocator: *mut sys::OrtAllocator,
    input: &InputValue<'_>,
) -> Result<*mut sys::OrtValue> {
    match input {
        InputValue::Tensor(tensor) => create_external_tensor(api, memory_info, tensor),
        InputValue::Strings(input) => {
            create_string_tensor_parts(api, allocator, &input.shape, &input.values)
        }
        InputValue::Sequence(values) => create_sequence_input(api, memory_info, allocator, values),
        InputValue::Map { keys, values } => {
            create_map_input(api, memory_info, allocator, keys, values)
        }
    }
}

pub(super) fn create_sequence_input(
    api: &sys::Api,
    memory_info: *const sys::OrtMemoryInfo,
    allocator: *mut sys::OrtAllocator,
    inputs: &[InputValue<'_>],
) -> Result<*mut sys::OrtValue> {
    let inputs = inputs.iter().collect::<Vec<_>>();
    create_container_input(
        api,
        memory_info,
        allocator,
        &inputs,
        sys::ONNXType::ONNX_TYPE_SEQUENCE,
        "sequence input value",
    )
}

pub(super) fn create_map_input(
    api: &sys::Api,
    memory_info: *const sys::OrtMemoryInfo,
    allocator: *mut sys::OrtAllocator,
    keys: &InputValue<'_>,
    values: &InputValue<'_>,
) -> Result<*mut sys::OrtValue> {
    create_container_input(
        api,
        memory_info,
        allocator,
        &[keys, values],
        sys::ONNXType::ONNX_TYPE_MAP,
        "map input value",
    )
}

pub(super) fn create_container_input(
    api: &sys::Api,
    memory_info: *const sys::OrtMemoryInfo,
    allocator: *mut sys::OrtAllocator,
    inputs: &[&InputValue<'_>],
    value_type: sys::ONNXType,
    description: &'static str,
) -> Result<*mut sys::OrtValue> {
    let mut children = Vec::with_capacity(inputs.len());
    for input in inputs {
        let value = create_input_value(api, memory_info, allocator, input)?;
        children.push(ValueGuard::new(api, value));
    }
    let pointers = children
        .iter()
        .map(|value| value.ptr.cast_const())
        .collect::<Vec<_>>();
    let pointers = if pointers.is_empty() {
        ptr::null()
    } else {
        pointers.as_ptr()
    };
    let mut value = ptr::null_mut();
    // SAFETY: child pointers are live for the call. CreateValue retains the
    // supplied values, so the temporary child guards may be dropped afterward.
    check_status(api, unsafe {
        (api.table().CreateValue)(pointers, children.len(), value_type, &mut value)
    })?;
    Ok(require_non_null(value, description)?)
}

pub(super) fn output_to_value(
    environment: Environment,
    value: *mut sys::OrtValue,
) -> Result<Value> {
    let api = &environment.inner.api;
    let mut value_guard = ValueGuard::new(api, value);
    let mut is_tensor = 0;
    // SAFETY: value is a live ORT output and is_tensor is writable.
    check_status(api, unsafe {
        (api.table().IsTensor)(value.cast_const(), &mut is_tensor)
    })?;
    if is_tensor == 0 {
        let mut info = ptr::null_mut();
        // SAFETY: value is live and info is writable output storage.
        check_status(api, unsafe {
            (api.table().GetTypeInfo)(value.cast_const(), &mut info)
        })?;
        let info = require_non_null(info, "output type info")?;
        let _guard = TypeInfoGuard {
            api,
            type_info: info,
        };
        let mut value_type = sys::ONNXType::ONNX_TYPE_UNKNOWN;
        // SAFETY: info is live and value_type is writable.
        check_status(api, unsafe {
            (api.table().GetOnnxTypeFromTypeInfo)(info, &mut value_type)
        })?;
        return match value_type {
            sys::ONNXType::ONNX_TYPE_SEQUENCE
            | sys::ONNXType::ONNX_TYPE_MAP
            | sys::ONNXType::ONNX_TYPE_OPTIONAL => {
                container_output(&environment, value, value_type)
            }
            _ => Err(Error::UnsupportedValueType { value_type }),
        };
    }

    let mut info = ptr::null_mut();
    // SAFETY: value is a live tensor and info is writable output storage.
    check_status(api, unsafe {
        (api.table().GetTensorTypeAndShape)(value, &mut info)
    })?;
    let info = require_non_null(info, "output tensor shape info")?;
    let info_guard = ShapeInfoGuard { api, info };
    let mut element_type = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    // SAFETY: info is live and element_type is writable.
    check_status(api, unsafe {
        (api.table().GetTensorElementType)(info, &mut element_type)
    })?;
    drop(info_guard);

    if element_type == sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING {
        value_guard.disarm();
        return string_output(api, value).map(Value::Strings);
    }
    value_guard.disarm();
    drop(value_guard);
    output_to_dlpack(environment, value).map(Value::Tensor)
}

pub(super) fn container_output(
    environment: &Environment,
    value: *mut sys::OrtValue,
    value_type: sys::ONNXType,
) -> Result<Value> {
    let api = &environment.inner.api;
    let allocator = default_allocator(api)?;
    let mut count = 0;
    // SAFETY: value is a live container and count is writable output storage.
    check_status(api, unsafe {
        (api.table().GetValueCount)(value.cast_const(), &mut count)
    })?;
    let mut children = Vec::with_capacity(count);
    for index in 0..count {
        let mut child = ptr::null_mut();
        // SAFETY: index is within the queried child count and child is writable.
        check_status(api, unsafe {
            (api.table().GetValue)(value.cast_const(), index as i32, allocator, &mut child)
        })?;
        let child = require_non_null(child, "container child value")?;
        children.push(output_to_value(environment.clone(), child)?);
    }
    match value_type {
        sys::ONNXType::ONNX_TYPE_SEQUENCE => Ok(Value::Sequence(children)),
        sys::ONNXType::ONNX_TYPE_MAP => {
            if children.len() != 2 {
                return Err(Error::ContainerValueCount {
                    kind: "map",
                    expected: 2,
                    actual: children.len(),
                });
            }
            let mut children = children.into_iter();
            Ok(Value::Map {
                keys: Box::new(children.next().expect("map child count checked")),
                values: Box::new(children.next().expect("map child count checked")),
            })
        }
        sys::ONNXType::ONNX_TYPE_OPTIONAL => match children.len() {
            0 => Ok(Value::Optional(None)),
            1 => Ok(Value::Optional(Some(Box::new(
                children.pop().expect("optional child count checked"),
            )))),
            actual => Err(Error::ContainerValueCount {
                kind: "optional",
                expected: 1,
                actual,
            }),
        },
        _ => Err(Error::UnsupportedValueType { value_type }),
    }
}

pub(super) fn string_output(api: &sys::Api, value: *mut sys::OrtValue) -> Result<StringTensor> {
    let value_guard = ValueGuard::new(api, value);
    let mut info = ptr::null_mut();
    // SAFETY: value is live and info is writable output storage.
    check_status(api, unsafe {
        (api.table().GetTensorTypeAndShape)(value, &mut info)
    })?;
    let info = require_non_null(info, "string tensor shape info")?;
    let info_guard = ShapeInfoGuard { api, info };
    let mut rank = 0;
    // SAFETY: info is live and rank is writable.
    check_status(api, unsafe {
        (api.table().GetDimensionsCount)(info, &mut rank)
    })?;
    let mut shape = vec![0_i64; rank];
    // SAFETY: shape contains rank writable entries.
    check_status(api, unsafe {
        (api.table().GetDimensions)(info, shape.as_mut_ptr(), rank)
    })?;
    let mut element_count = 0;
    // SAFETY: info is live and element_count is writable.
    check_status(api, unsafe {
        (api.table().GetTensorShapeElementCount)(info, &mut element_count)
    })?;
    drop(info_guard);

    let mut data_len = 0;
    // SAFETY: value is a live string tensor and data_len is writable.
    check_status(api, unsafe {
        (api.table().GetStringTensorDataLength)(value.cast_const(), &mut data_len)
    })?;
    let mut data = vec![0_u8; data_len];
    let mut offsets = vec![0_usize; element_count];
    let data_ptr = if data.is_empty() {
        ptr::null_mut()
    } else {
        data.as_mut_ptr().cast()
    };
    let offsets_ptr = if offsets.is_empty() {
        ptr::null_mut()
    } else {
        offsets.as_mut_ptr()
    };
    // SAFETY: buffers match the exact sizes queried from the tensor.
    check_status(api, unsafe {
        (api.table().GetStringTensorContent)(
            value.cast_const(),
            data_ptr,
            data.len(),
            offsets_ptr,
            offsets.len(),
        )
    })?;
    let mut values = Vec::with_capacity(element_count);
    for (index, start) in offsets.iter().copied().enumerate() {
        let end = offsets.get(index + 1).copied().unwrap_or(data.len());
        if start > end || end > data.len() {
            return Err(Error::InvalidStringOffsets {
                index,
                start,
                end,
                data_len: data.len(),
            });
        }
        let bytes = data[start..end].to_vec();
        values.push(
            String::from_utf8(bytes)
                .map_err(|source| Error::InvalidStringUtf8 { index, source })?,
        );
    }
    drop(value_guard);
    Ok(StringTensor { shape, values })
}

pub(super) fn output_to_dlpack(
    environment: Environment,
    value: *mut sys::OrtValue,
) -> Result<versioned::Dlpack> {
    let api = environment.inner.api.clone();
    let mut value_guard = ValueGuard::new(&api, value);
    let mut info = ptr::null_mut();
    // SAFETY: value is a live ORT output and info is writable output storage.
    check_status(&api, unsafe {
        (api.table().GetTensorTypeAndShape)(value, &mut info)
    })?;
    let info = require_non_null(info, "tensor shape info")?;
    let info_guard = ShapeInfoGuard { api: &api, info };

    let mut element_type = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    // SAFETY: info remains live until info_guard is dropped.
    check_status(&api, unsafe {
        (api.table().GetTensorElementType)(info, &mut element_type)
    })?;
    let dtype = onnx_to_dlpack(element_type)?;
    let mut memory_info = ptr::null();
    // SAFETY: value is live and memory_info is writable borrowed output storage.
    check_status(&api, unsafe {
        (api.table().GetTensorMemoryInfo)(value, &mut memory_info)
    })?;
    let memory_info = require_const_non_null(memory_info, "tensor memory info")?;
    let mut memory_name = ptr::null();
    let mut device_id = 0;
    // SAFETY: memory_info is borrowed from the live tensor value.
    check_status(&api, unsafe {
        (api.table().MemoryInfoGetName)(memory_info, &mut memory_name)
    })?;
    // SAFETY: memory_info is live and device_id is writable.
    check_status(&api, unsafe {
        (api.table().MemoryInfoGetId)(memory_info, &mut device_id)
    })?;
    let memory_name = require_const_non_null(memory_name, "tensor memory name")?;
    // SAFETY: ORT memory-info names are NUL-terminated for the value lifetime.
    let memory_name = unsafe { CStr::from_ptr(memory_name) }
        .to_string_lossy()
        .into_owned();
    let device = match memory_name.as_str() {
        "Cpu" | "CPU" => DLDevice::CPU,
        "Cuda" | "CUDA" if device_id >= 0 => DLDevice {
            device_type: DLDeviceType::CUDA,
            device_id,
        },
        "CudaPinned" | "CUDA_PINNED" if device_id >= 0 => DLDevice {
            device_type: DLDeviceType::CUDAHOST,
            device_id,
        },
        _ => {
            return Err(Error::UnsupportedOrtMemory {
                name: memory_name,
                device_id,
            });
        }
    };
    let mut rank = 0;
    // SAFETY: info is valid and rank is writable.
    check_status(&api, unsafe {
        (api.table().GetDimensionsCount)(info, &mut rank)
    })?;
    let mut shape = vec![0_i64; rank];
    // SAFETY: shape has exactly rank writable elements.
    check_status(&api, unsafe {
        (api.table().GetDimensions)(info, shape.as_mut_ptr(), rank)
    })?;
    let mut data = ptr::null_mut();
    // SAFETY: value is a live tensor value and data is writable output storage.
    check_status(&api, unsafe {
        (api.table().GetTensorMutableData)(value, &mut data)
    })?;
    let mut element_count = 0;
    // SAFETY: info remains live and element_count is writable.
    check_status(&api, unsafe {
        (api.table().GetTensorShapeElementCount)(info, &mut element_count)
    })?;
    if data.is_null() && element_count != 0 {
        return Err(Error::NullPointer {
            what: "tensor data",
        });
    }
    drop(info_guard);

    let owner = Box::new(OutputOwner {
        environment,
        value: value as usize,
    });
    value_guard.disarm();
    let mut initialized = Dynamic::compact(shape.as_slice())
        .initialize(owner)
        .map_err(|source| Error::DlpackMetadata { source })?;
    initialized
        .set_data(data)
        .set_device(device)
        .set_dtype(dtype)
        .set_byte_offset(0);
    // SAFETY: the OutputOwner retains the OrtValue allocation, metadata is
    // owned by dlpark, and all scalar fields were populated above.
    Ok(unsafe { initialized.finish() })
}
