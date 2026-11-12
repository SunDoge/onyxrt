//! Low-level RAII wrappers for ONNX Runtime values and tensor memory.
//!
//! This module works directly with `OrtValue` and `OrtMemoryInfo`. It is the
//! escape hatch for integrations that do not use DLPack and the foundation on
//! which higher-level tensor adapters can be built.

use std::{
    ffi::{CString, c_void},
    marker::PhantomData,
    mem::{align_of, size_of_val},
    ptr::{self, NonNull},
    slice,
};

use onyxrt_sys as sys;
use snafu::{Snafu, ensure};

use crate::{
    ffi::{
        self, ShapeInfoGuard, api_function, check_status, cstring, require_non_null, status_call,
    },
    model::{ElementType, Environment},
    session::{RunOptions, Session},
};

/// Result type for low-level value and execution operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors produced by the low-level ONNX Runtime API.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Ffi {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(display("object belongs to a different ONNX Runtime environment"))]
    ForeignEnvironment,

    #[snafu(display("tensor shape dimension {axis} is negative: {value}"))]
    NegativeDimension { axis: usize, value: i64 },

    #[snafu(display("tensor element count overflows usize"))]
    ElementCountOverflow,

    #[snafu(display("tensor shape requires {expected} elements, got {actual}"))]
    ElementCount { expected: usize, actual: usize },

    #[snafu(display("OrtValue is not a dense tensor"))]
    NotTensor,

    #[snafu(display("tensor has element type {actual:?}, expected {expected:?}"))]
    ElementTypeMismatch {
        expected: ElementType,
        actual: ElementType,
    },

    #[snafu(display("tensor memory is not CPU-accessible"))]
    NonCpuMemory,

    #[snafu(display("tensor data pointer is not aligned for the requested Rust type"))]
    MisalignedData,

    #[snafu(display("ONNX Runtime replaced preallocated raw output `{name}`"))]
    ReplacedOutput { name: String },
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
    }
}

/// ONNX Runtime memory placement information.
pub struct MemoryInfo {
    environment: Environment,
    memory_info: NonNull<sys::OrtMemoryInfo>,
    cpu_accessible: bool,
}

// SAFETY: OrtMemoryInfo is immutable after construction and contains no Rust
// references. Its environment retains the originating API table.
unsafe impl Send for MemoryInfo {}
// SAFETY: all exposed operations on memory info are read-only.
unsafe impl Sync for MemoryInfo {}

impl MemoryInfo {
    /// Creates the standard CPU arena memory descriptor.
    pub fn cpu(environment: Environment) -> Result<Self> {
        let api = &environment.inner.api;
        let mut memory_info = ptr::null_mut();
        // SAFETY: output storage is writable and both enum values are valid.
        status_call!(
            api,
            CreateCpuMemoryInfo(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        )?;
        Ok(Self {
            environment,
            memory_info: NonNull::new(require_non_null(memory_info, "memory info")?)
                .expect("require_non_null returned a non-null pointer"),
            cpu_accessible: true,
        })
    }

    /// Creates a CUDA device-memory descriptor.
    pub fn cuda(environment: Environment, device_id: i32) -> Result<Self> {
        let api = &environment.inner.api;
        let mut memory_info = ptr::null_mut();
        // SAFETY: the allocator name is static and output storage is writable.
        status_call!(
            api,
            CreateMemoryInfo(
                c"Cuda".as_ptr(),
                sys::OrtAllocatorType::OrtDeviceAllocator,
                device_id,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        )?;
        Ok(Self {
            environment,
            memory_info: NonNull::new(require_non_null(memory_info, "memory info")?)
                .expect("require_non_null returned a non-null pointer"),
            cpu_accessible: false,
        })
    }

    /// Returns the native memory-info pointer.
    pub fn as_ptr(&self) -> *const sys::OrtMemoryInfo {
        self.memory_info.as_ptr()
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the live memory-info object.
        unsafe {
            api_function!(self.environment.inner.api, ReleaseMemoryInfo)(self.memory_info.as_ptr())
        };
    }
}

/// Rust element types that have an unambiguous ONNX tensor representation.
pub trait TensorElement: private::Sealed + 'static {
    /// Corresponding ONNX tensor element type.
    const ELEMENT_TYPE: ElementType;
}

macro_rules! tensor_elements {
    ($($rust:ty => $element:ident),+ $(,)?) => {
        $(
            impl private::Sealed for $rust {}
            impl TensorElement for $rust {
                const ELEMENT_TYPE: ElementType = ElementType::$element;
            }
        )+
    };
}

tensor_elements! {
    f32 => Float32,
    f64 => Float64,
    i8 => Int8,
    i16 => Int16,
    i32 => Int32,
    i64 => Int64,
    u8 => Uint8,
    u16 => Uint16,
    u32 => Uint32,
    u64 => Uint64,
}

mod private {
    pub trait Sealed {}
}

/// An owned `OrtValue` that may borrow externally managed tensor storage.
pub struct Value<'buffer> {
    environment: Environment,
    value: NonNull<sys::OrtValue>,
    _buffer: PhantomData<&'buffer mut [u8]>,
}

impl<'buffer> Value<'buffer> {
    /// Creates a tensor backed directly by a mutable Rust slice.
    pub fn tensor<T: TensorElement>(
        environment: Environment,
        memory_info: &MemoryInfo,
        data: &'buffer mut [T],
        shape: impl AsRef<[i64]>,
    ) -> Result<Self> {
        ensure!(
            Environment::same_runtime(&environment, &memory_info.environment),
            ForeignEnvironmentSnafu
        );
        ensure!(memory_info.cpu_accessible, NonCpuMemorySnafu);
        let shape = shape.as_ref();
        let expected = element_count(shape)?;
        ensure!(
            expected == data.len(),
            ElementCountSnafu {
                expected,
                actual: data.len()
            }
        );
        let byte_len = size_of_val(data);
        let data =
            NonNull::new(data.as_mut_ptr().cast::<c_void>()).unwrap_or_else(NonNull::dangling);
        // SAFETY: the mutable slice is retained by the returned lifetime, has
        // exactly the shape's element count, and its byte size is correct.
        unsafe {
            Self::from_raw_parts(
                environment,
                memory_info,
                data,
                byte_len,
                shape,
                T::ELEMENT_TYPE,
            )
        }
    }

    /// Creates a tensor over arbitrary external storage.
    ///
    /// # Safety
    ///
    /// `data` must remain valid, correctly aligned, and writable for
    /// `byte_len` bytes for the returned lifetime. The allocation must reside
    /// in the device described by `memory_info`, and `shape`/`element_type`
    /// must accurately describe the storage.
    pub unsafe fn from_raw_parts(
        environment: Environment,
        memory_info: &MemoryInfo,
        data: NonNull<c_void>,
        byte_len: usize,
        shape: &[i64],
        element_type: ElementType,
    ) -> Result<Self> {
        ensure!(
            Environment::same_runtime(&environment, &memory_info.environment),
            ForeignEnvironmentSnafu
        );
        let api = &memory_info.environment.inner.api;
        let mut value = ptr::null_mut();
        // SAFETY: upheld by this function's caller; all descriptor arrays stay
        // live for the duration of the native constructor call.
        status_call!(
            api,
            CreateTensorWithDataAsOrtValue(
                memory_info.as_ptr(),
                data.as_ptr(),
                byte_len,
                shape.as_ptr(),
                shape.len(),
                to_native_element_type(element_type),
                &mut value,
            )
        )?;
        Ok(Self {
            environment,
            value: NonNull::new(require_non_null(value, "tensor value")?)
                .expect("require_non_null returned a non-null pointer"),
            _buffer: PhantomData,
        })
    }

    /// Adopts an owned native value returned by the same environment.
    ///
    /// # Safety
    ///
    /// `value` must be a uniquely owned live `OrtValue` created by
    /// `environment`. Ownership is transferred to the returned wrapper.
    pub unsafe fn from_raw_owned(environment: &Environment, value: NonNull<sys::OrtValue>) -> Self {
        Self {
            environment: environment.clone(),
            value,
            _buffer: PhantomData,
        }
    }

    /// Returns the native value pointer for read-only FFI calls.
    pub fn as_ptr(&self) -> *const sys::OrtValue {
        self.value.as_ptr()
    }

    /// Returns the native value pointer for calls that may mutate the value.
    pub fn as_mut_ptr(&mut self) -> *mut sys::OrtValue {
        self.value.as_ptr()
    }

    /// Returns whether this value is a dense tensor.
    pub fn is_tensor(&self) -> Result<bool> {
        let api = &self.environment.inner.api;
        let mut is_tensor = 0;
        // SAFETY: the value is live and output storage is writable.
        status_call!(api, IsTensor(self.as_ptr(), &mut is_tensor))?;
        Ok(is_tensor != 0)
    }

    /// Returns the tensor element type and shape.
    pub fn tensor_info(&self) -> Result<TensorInfo> {
        ensure!(self.is_tensor()?, NotTensorSnafu);
        let api = &self.environment.inner.api;
        let mut info = ptr::null_mut();
        // SAFETY: the value is a live tensor and output storage is writable.
        status_call!(api, GetTensorTypeAndShape(self.as_ptr(), &mut info))?;
        let info = require_non_null(info, "tensor shape info")?;
        let guard = ShapeInfoGuard { api, info };

        let mut element_type =
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        // SAFETY: the shape info is live and output storage is writable.
        status_call!(api, GetTensorElementType(guard.info, &mut element_type))?;
        let mut rank = 0;
        // SAFETY: the shape info is live and output storage is writable.
        status_call!(api, GetDimensionsCount(guard.info, &mut rank))?;
        let mut shape = vec![0_i64; rank];
        // SAFETY: shape has exactly rank writable entries.
        status_call!(
            api,
            GetDimensions(guard.info, shape.as_mut_ptr(), shape.len())
        )?;
        Ok(TensorInfo {
            element_type: element_type.into(),
            shape,
        })
    }

    /// Borrows CPU tensor storage as a typed slice.
    pub fn as_slice<T: TensorElement>(&self) -> Result<&[T]> {
        let (pointer, length) = self.typed_data::<T>()?;
        // SAFETY: typed_data validated CPU placement, type, alignment, and
        // element count. The borrow is tied to this live value.
        Ok(unsafe { slice::from_raw_parts(pointer.cast_const(), length) })
    }

    /// Exclusively borrows CPU tensor storage as a typed mutable slice.
    pub fn as_slice_mut<T: TensorElement>(&mut self) -> Result<&mut [T]> {
        let (pointer, length) = self.typed_data::<T>()?;
        // SAFETY: &mut self ensures exclusive access for the returned borrow.
        Ok(unsafe { slice::from_raw_parts_mut(pointer, length) })
    }

    fn typed_data<T: TensorElement>(&self) -> Result<(*mut T, usize)> {
        let info = self.tensor_info()?;
        ensure!(
            info.element_type == T::ELEMENT_TYPE,
            ElementTypeMismatchSnafu {
                expected: T::ELEMENT_TYPE,
                actual: info.element_type
            }
        );
        ensure!(self.is_cpu_tensor()?, NonCpuMemorySnafu);
        let mut data = ptr::null_mut();
        let api = &self.environment.inner.api;
        // SAFETY: the value is a dense tensor and output storage is writable.
        status_call!(api, GetTensorMutableData(self.value.as_ptr(), &mut data))?;
        let data = require_non_null(data.cast::<T>(), "tensor data")?;
        ensure!(data.align_offset(align_of::<T>()) == 0, MisalignedDataSnafu);
        Ok((data, element_count(&info.shape)?))
    }

    fn is_cpu_tensor(&self) -> Result<bool> {
        let api = &self.environment.inner.api;
        let mut memory_info = ptr::null();
        // SAFETY: the value is live and output storage is writable.
        status_call!(api, GetTensorMemoryInfo(self.as_ptr(), &mut memory_info))?;
        let memory_info = require_non_null(memory_info.cast_mut(), "tensor memory info")?;
        let mut device_type = sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU;
        // SAFETY: borrowed memory info remains valid for the value lifetime.
        unsafe { api_function!(api, MemoryInfoGetDeviceType)(memory_info, &mut device_type) };
        Ok(device_type == sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU)
    }

    /// Releases Rust ownership and returns the native pointer.
    pub fn into_raw(self) -> *mut sys::OrtValue {
        let pointer = self.value.as_ptr();
        std::mem::forget(self);
        pointer
    }
}

impl Drop for Value<'_> {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the live OrtValue.
        unsafe { api_function!(self.environment.inner.api, ReleaseValue)(self.value.as_ptr()) };
    }
}

/// Tensor metadata queried from a low-level `OrtValue`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor element type.
    pub element_type: ElementType,
    /// Concrete tensor dimensions.
    pub shape: Vec<i64>,
}

/// A named borrowed value supplied to [`Session::run_raw`].
pub struct Input<'value> {
    name: CString,
    value: NonNull<sys::OrtValue>,
    environment: &'value Environment,
}

/// A named caller-owned value used as a preallocated session output.
pub struct Output<'value> {
    name: CString,
    value: NonNull<sys::OrtValue>,
    environment: &'value Environment,
    _borrow: PhantomData<&'value mut sys::OrtValue>,
}

impl<'value> Output<'value> {
    /// Creates a named preallocated output.
    pub fn new<'buffer>(name: &str, value: &'value mut Value<'buffer>) -> Result<Self> {
        Ok(Self {
            name: cstring(name, "raw output")?,
            value: value.value,
            environment: &value.environment,
            _borrow: PhantomData,
        })
    }

    /// Returns the output name.
    pub fn name(&self) -> &std::ffi::CStr {
        &self.name
    }

    /// Returns the borrowed native value pointer.
    pub fn value_ptr(&self) -> *mut sys::OrtValue {
        self.value.as_ptr()
    }
}

impl<'value> Input<'value> {
    /// Creates a named raw input without copying its value.
    pub fn new<'buffer>(name: &str, value: &'value Value<'buffer>) -> Result<Self> {
        Ok(Self {
            name: cstring(name, "raw input")?,
            value: value.value,
            environment: &value.environment,
        })
    }

    /// Returns the input name.
    pub fn name(&self) -> &std::ffi::CStr {
        &self.name
    }

    /// Returns the borrowed native value pointer.
    pub fn value_ptr(&self) -> *const sys::OrtValue {
        self.value.as_ptr()
    }
}

impl Session {
    /// Returns the borrowed native session pointer.
    pub fn as_ptr(&self) -> *const sys::OrtSession {
        self.session as *const sys::OrtSession
    }

    /// Runs the session with low-level `OrtValue` inputs.
    pub fn run_raw(
        &self,
        inputs: &[Input<'_>],
        output_names: &[&str],
    ) -> Result<Vec<Value<'static>>> {
        self.run_raw_impl(inputs, output_names, None)
    }

    /// Runs low-level values with reusable run options.
    pub fn run_raw_with_options(
        &self,
        inputs: &[Input<'_>],
        output_names: &[&str],
        options: &RunOptions,
    ) -> Result<Vec<Value<'static>>> {
        ensure!(options.belongs_to(self), ForeignEnvironmentSnafu);
        self.run_raw_impl(inputs, output_names, Some(options))
    }

    /// Runs with caller-owned, preallocated low-level outputs.
    pub fn run_raw_into(&self, inputs: &[Input<'_>], outputs: &mut [Output<'_>]) -> Result<()> {
        self.run_raw_into_impl(inputs, outputs, None)
    }

    /// Runs with preallocated outputs and reusable run options.
    pub fn run_raw_into_with_options(
        &self,
        inputs: &[Input<'_>],
        outputs: &mut [Output<'_>],
        options: &RunOptions,
    ) -> Result<()> {
        ensure!(options.belongs_to(self), ForeignEnvironmentSnafu);
        self.run_raw_into_impl(inputs, outputs, Some(options))
    }

    fn run_raw_impl(
        &self,
        inputs: &[Input<'_>],
        output_names: &[&str],
        options: Option<&RunOptions>,
    ) -> Result<Vec<Value<'static>>> {
        for input in inputs {
            ensure!(
                Environment::same_runtime(&self.environment, input.environment),
                ForeignEnvironmentSnafu
            );
        }
        let output_names = output_names
            .iter()
            .map(|name| cstring(name, "raw output").map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        let input_names = inputs
            .iter()
            .map(|input| input.name.as_ptr())
            .collect::<Vec<_>>();
        let input_values = inputs.iter().map(Input::value_ptr).collect::<Vec<_>>();
        let output_name_ptrs = output_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut outputs = vec![ptr::null_mut(); output_names.len()];
        let api = &self.environment.inner.api;
        // SAFETY: pointer arrays have matching lengths and all referenced
        // values and strings remain live for this synchronous call.
        let status = unsafe {
            api_function!(api, Run)(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                input_names.as_ptr(),
                input_values.as_ptr(),
                input_values.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                outputs.as_mut_ptr(),
            )
        };
        if let Err(error) = check_status(api, status) {
            release_values(api, outputs);
            return Err(error.into());
        }
        if outputs.iter().any(|value| value.is_null()) {
            release_values(api, outputs);
            return Err(ffi::Error::NullPointer {
                what: "raw output value",
            }
            .into());
        }
        outputs
            .into_iter()
            .map(|value| {
                let value = NonNull::new(value).expect("output null pointers were rejected");
                // SAFETY: successful Run transfers each output to the caller.
                Ok(unsafe { Value::from_raw_owned(&self.environment, value) })
            })
            .collect()
    }

    fn run_raw_into_impl(
        &self,
        inputs: &[Input<'_>],
        outputs: &mut [Output<'_>],
        options: Option<&RunOptions>,
    ) -> Result<()> {
        for environment in inputs
            .iter()
            .map(|input| input.environment)
            .chain(outputs.iter().map(|output| output.environment))
        {
            ensure!(
                Environment::same_runtime(&self.environment, environment),
                ForeignEnvironmentSnafu
            );
        }
        let input_names = inputs
            .iter()
            .map(|input| input.name.as_ptr())
            .collect::<Vec<_>>();
        let input_values = inputs.iter().map(Input::value_ptr).collect::<Vec<_>>();
        let output_names = outputs
            .iter()
            .map(|output| output.name.as_ptr())
            .collect::<Vec<_>>();
        let original_outputs = outputs.iter().map(Output::value_ptr).collect::<Vec<_>>();
        let mut returned_outputs = original_outputs.clone();
        let api = &self.environment.inner.api;
        // SAFETY: all arrays have matching lengths and every borrowed value is
        // retained exclusively where ORT may write to it.
        let status = unsafe {
            api_function!(api, Run)(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                input_names.as_ptr(),
                input_values.as_ptr(),
                input_values.len(),
                output_names.as_ptr(),
                output_names.len(),
                returned_outputs.as_mut_ptr(),
            )
        };
        let mut replaced = None;
        for ((returned, original), output) in returned_outputs
            .into_iter()
            .zip(original_outputs)
            .zip(outputs.iter())
        {
            if returned != original {
                if !returned.is_null() {
                    // SAFETY: a replacement value is newly owned by this call.
                    unsafe { api_function!(api, ReleaseValue)(returned) };
                }
                replaced.get_or_insert_with(|| output.name.to_string_lossy().into_owned());
            }
        }
        check_status(api, status)?;
        if let Some(name) = replaced {
            return Err(Error::ReplacedOutput { name });
        }
        Ok(())
    }
}

impl Environment {
    /// Returns the loaded ONNX Runtime C API table.
    pub fn raw_api(&self) -> &sys::Api {
        &self.inner.api
    }
}

fn release_values(api: &sys::Api, values: Vec<*mut sys::OrtValue>) {
    for value in values.into_iter().filter(|value| !value.is_null()) {
        // SAFETY: each non-null output is uniquely owned by the failed run.
        unsafe { api_function!(api, ReleaseValue)(value) };
    }
}

fn element_count(shape: &[i64]) -> Result<usize> {
    shape
        .iter()
        .copied()
        .enumerate()
        .try_fold(1_usize, |count, (axis, value)| {
            let dimension =
                usize::try_from(value).map_err(|_| Error::NegativeDimension { axis, value })?;
            count
                .checked_mul(dimension)
                .ok_or(Error::ElementCountOverflow)
        })
}

fn to_native_element_type(element_type: ElementType) -> sys::ONNXTensorElementDataType {
    use sys::ONNXTensorElementDataType as O;
    match element_type {
        ElementType::Undefined => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        ElementType::Float32 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ElementType::Uint8 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        ElementType::Int8 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ElementType::Uint16 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        ElementType::Int16 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        ElementType::Int32 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        ElementType::Int64 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        ElementType::String => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
        ElementType::Bool => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        ElementType::Float16 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        ElementType::Float64 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        ElementType::Uint32 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        ElementType::Uint64 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        ElementType::Complex64 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
        ElementType::Complex128 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
        ElementType::BFloat16 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        ElementType::Float8E4M3Fn => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
        ElementType::Float8E4M3Fnuz => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
        ElementType::Float8E5M2 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
        ElementType::Float8E5M2Fnuz => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ,
        ElementType::Uint4 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
        ElementType::Int4 => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
        ElementType::Float4E2M1 => sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1,
        ElementType::Uint2 => sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2,
        ElementType::Int2 => sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2,
        ElementType::Float8E8M0 => sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0,
    }
}
