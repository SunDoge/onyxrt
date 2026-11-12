//! Low-level RAII wrappers for ONNX Runtime values and tensor memory.
//!
//! This module works directly with `OrtValue` and `OrtMemoryInfo`. It is the
//! escape hatch for integrations that do not use DLPack and the foundation on
//! which higher-level tensor adapters can be built.

use std::{
    ffi::{CStr, CString, c_void},
    future::Future,
    marker::PhantomData,
    mem::{align_of, size_of_val},
    pin::Pin,
    ptr::{self, NonNull},
    slice,
    sync::Arc,
    task::{Context, Poll},
};

use onyxrt_sys as sys;
use snafu::{Snafu, ensure};

use crate::{
    ffi::{
        self, ShapeInfoGuard, api_function, check_status, cstring, require_non_null, status_call,
    },
    io_binding::{AsyncCompletion, async_run_callback},
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

    #[snafu(transparent)]
    Session { source: crate::session::Error },

    #[snafu(transparent)]
    Async { source: crate::io_binding::Error },

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

    #[snafu(display("ONNX Runtime returned unknown allocator type {value}"))]
    UnknownAllocatorType { value: i32 },

    #[snafu(display("ONNX Runtime returned unknown memory type {value}"))]
    UnknownMemoryType { value: i32 },

    #[snafu(display("container index {index} exceeds ONNX Runtime's i32 index range"))]
    ContainerIndexOverflow { index: usize },

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
}

// SAFETY: OrtMemoryInfo is immutable after construction and contains no Rust
// references. Its environment retains the originating API table.
unsafe impl Send for MemoryInfo {}
// SAFETY: all exposed operations on memory info are read-only.
unsafe impl Sync for MemoryInfo {}

impl MemoryInfo {
    /// Creates a memory descriptor for an execution-provider allocator.
    pub fn new(
        environment: Environment,
        name: &str,
        allocator_type: AllocatorType,
        device_id: i32,
        memory_type: MemoryType,
    ) -> Result<Self> {
        let name = cstring(name, "memory info name")?;
        let api = &environment.inner.api;
        let mut memory_info = ptr::null_mut();
        // SAFETY: the name and output storage remain live for the call and all
        // enum values have valid ONNX Runtime representations.
        status_call!(
            api,
            CreateMemoryInfo(
                name.as_ptr(),
                allocator_type.into(),
                device_id,
                memory_type.into(),
                &mut memory_info,
            )
        )?;
        Ok(Self {
            environment,
            memory_info: NonNull::new(require_non_null(memory_info, "memory info")?)
                .expect("require_non_null returned a non-null pointer"),
        })
    }

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
        })
    }

    /// Creates a CUDA device-memory descriptor.
    pub fn cuda(environment: Environment, device_id: i32) -> Result<Self> {
        Self::new(
            environment,
            "Cuda",
            AllocatorType::Device,
            device_id,
            MemoryType::Default,
        )
    }

    /// Returns the native memory-info pointer.
    pub fn as_ptr(&self) -> *const sys::OrtMemoryInfo {
        self.memory_info.as_ptr()
    }

    /// Returns the allocator name retained by ONNX Runtime.
    pub fn name(&self) -> Result<&CStr> {
        let api = &self.environment.inner.api;
        let mut name = ptr::null();
        // SAFETY: the descriptor is live and output storage is writable.
        status_call!(api, MemoryInfoGetName(self.as_ptr(), &mut name))?;
        let name = require_non_null(name.cast_mut(), "memory info name")?;
        // SAFETY: ONNX Runtime documents this string as valid for the lifetime
        // of the descriptor, which is borrowed by the returned reference.
        Ok(unsafe { CStr::from_ptr(name) })
    }

    /// Returns the allocator kind.
    pub fn allocator_type(&self) -> Result<AllocatorType> {
        memory_info_allocator_type(&self.environment.inner.api, self.as_ptr())
    }

    /// Returns the execution-provider device identifier.
    pub fn device_id(&self) -> Result<i32> {
        memory_info_device_id(&self.environment.inner.api, self.as_ptr())
    }

    /// Returns the execution-provider memory kind.
    pub fn memory_type(&self) -> Result<MemoryType> {
        memory_info_memory_type(&self.environment.inner.api, self.as_ptr())
    }

    /// Returns the broad device category.
    pub fn device_type(&self) -> DeviceType {
        memory_info_device_type(&self.environment.inner.api, self.as_ptr())
    }

    /// Returns a copied snapshot of all descriptor fields.
    pub fn details(&self) -> Result<MemoryInfoDetails> {
        memory_info_details(&self.environment.inner.api, self.as_ptr())
    }

    /// Compares two descriptors using ONNX Runtime's native semantics.
    pub fn is_equal(&self, other: &Self) -> Result<bool> {
        ensure!(
            Environment::same_runtime(&self.environment, &other.environment),
            ForeignEnvironmentSnafu
        );
        let api = &self.environment.inner.api;
        let mut comparison = -1;
        // SAFETY: both descriptors are live and the output storage is writable.
        status_call!(
            api,
            CompareMemoryInfo(self.as_ptr(), other.as_ptr(), &mut comparison)
        )?;
        Ok(comparison == 0)
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

/// A session allocator for a specific memory descriptor.
///
/// Values allocated through this object borrow it because ONNX Runtime keeps
/// calling the native allocator until the value is released.
pub struct Allocator<'session> {
    session: &'session Session,
    allocator: NonNull<sys::OrtAllocator>,
}

impl Allocator<'_> {
    /// Returns the native allocator pointer.
    pub fn as_ptr(&self) -> *mut sys::OrtAllocator {
        self.allocator.as_ptr()
    }

    /// Returns a copied snapshot of the allocator's memory descriptor.
    pub fn memory_info(&self) -> Result<MemoryInfoDetails> {
        let api = &self.session.environment.inner.api;
        let mut memory_info = ptr::null();
        // SAFETY: the allocator is live and output storage is writable.
        status_call!(api, AllocatorGetInfo(self.as_ptr(), &mut memory_info))?;
        let memory_info = require_non_null(memory_info.cast_mut(), "allocator memory info")?;
        memory_info_details(api, memory_info)
    }
}

impl Drop for Allocator<'_> {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the live session allocator.
        unsafe {
            api_function!(self.session.environment.inner.api, ReleaseAllocator)(self.as_ptr())
        };
    }
}

/// Native allocator strategy recorded in [`MemoryInfo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorType {
    /// Allocations are made directly by the device allocator.
    Device,
    /// Allocations are served from an arena.
    Arena,
}

impl From<AllocatorType> for sys::OrtAllocatorType {
    fn from(value: AllocatorType) -> Self {
        match value {
            AllocatorType::Device => Self::OrtDeviceAllocator,
            AllocatorType::Arena => Self::OrtArenaAllocator,
        }
    }
}

/// Execution-provider memory usage recorded in [`MemoryInfo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// CPU memory consumed by a non-CPU execution provider.
    CpuInput,
    /// CPU-accessible memory produced by a non-CPU execution provider.
    CpuOutput,
    /// The execution provider's default memory.
    Default,
}

impl From<MemoryType> for sys::OrtMemType {
    fn from(value: MemoryType) -> Self {
        match value {
            MemoryType::CpuInput => Self::OrtMemTypeCPUInput,
            MemoryType::CpuOutput => Self::OrtMemTypeCPUOutput,
            MemoryType::Default => Self::OrtMemTypeDefault,
        }
    }
}

/// Broad device category associated with tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceType {
    /// Host CPU memory.
    Cpu,
    /// GPU memory.
    Gpu,
    /// FPGA memory.
    Fpga,
    /// A device category introduced by a newer runtime.
    Unknown(u32),
}

/// Copied fields from an ONNX Runtime memory descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryInfoDetails {
    /// Allocator or execution-provider name.
    pub name: String,
    /// Native allocation strategy.
    pub allocator_type: AllocatorType,
    /// Execution-provider device identifier.
    pub device_id: i32,
    /// Execution-provider memory usage.
    pub memory_type: MemoryType,
    /// Broad device category.
    pub device_type: DeviceType,
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
        ensure!(
            memory_info.device_type() == DeviceType::Cpu,
            NonCpuMemorySnafu
        );
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

    /// Allocates a tensor from a session allocator.
    ///
    /// The returned value borrows `allocator`; the allocator must remain live
    /// because ONNX Runtime uses it to release the tensor storage.
    pub fn allocate_tensor_in<T: TensorElement>(
        environment: Environment,
        allocator: &'buffer Allocator<'_>,
        shape: impl AsRef<[i64]>,
    ) -> Result<Self> {
        Self::allocate_tensor_in_with_type(environment, allocator, shape, T::ELEMENT_TYPE)
    }

    /// Allocates a tensor from a session allocator with an explicit type.
    pub fn allocate_tensor_in_with_type(
        environment: Environment,
        allocator: &'buffer Allocator<'_>,
        shape: impl AsRef<[i64]>,
        element_type: ElementType,
    ) -> Result<Self> {
        ensure!(
            Environment::same_runtime(&environment, &allocator.session.environment),
            ForeignEnvironmentSnafu
        );
        let shape = shape.as_ref();
        element_count(shape)?;
        let api = &environment.inner.api;
        let mut value = ptr::null_mut();
        // SAFETY: allocator is retained through the returned lifetime, shape
        // remains live for the call, and output storage is writable.
        status_call!(
            api,
            CreateTensorAsOrtValue(
                allocator.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                to_native_element_type(element_type),
                &mut value,
            )
        )?;
        Ok(Self {
            environment,
            value: NonNull::new(require_non_null(value, "allocated tensor value")?)
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
    pub unsafe fn from_raw_owned(environment: Environment, value: NonNull<sys::OrtValue>) -> Self {
        Self {
            environment,
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

    /// Returns the ONNX value category.
    pub fn value_type(&self) -> Result<ValueType> {
        let api = &self.environment.inner.api;
        let mut value_type = sys::ONNXType::ONNX_TYPE_UNKNOWN;
        // SAFETY: the value is live and output storage is writable.
        status_call!(api, GetValueType(self.as_ptr(), &mut value_type))?;
        Ok(value_type.into())
    }

    /// Returns the child count of a sequence, map, or optional value.
    pub fn value_count(&self) -> Result<usize> {
        let api = &self.environment.inner.api;
        let mut count = 0;
        // SAFETY: the value is live and output storage is writable. ORT
        // validates that this value category supports child access.
        status_call!(api, GetValueCount(self.as_ptr(), &mut count))?;
        Ok(count)
    }

    /// Copies one child from a sequence, map, or optional value.
    ///
    /// ONNX Runtime uses its process-lifetime default CPU allocator for the
    /// returned value, so the child does not borrow its parent.
    pub fn value_at(&self, index: usize) -> Result<Value<'static>> {
        let index = i32::try_from(index).map_err(|_| Error::ContainerIndexOverflow { index })?;
        let api = &self.environment.inner.api;
        let allocator = ffi::default_allocator(api)?;
        let mut value = ptr::null_mut();
        // SAFETY: the parent and allocator are live and output storage is
        // writable. ORT validates the index against the container.
        status_call!(api, GetValue(self.as_ptr(), index, allocator, &mut value))?;
        let value = NonNull::new(require_non_null(value, "container child value")?)
            .expect("require_non_null returned a non-null pointer");
        // SAFETY: GetValue transfers unique ownership of the copied child.
        Ok(unsafe { Value::from_raw_owned(self.environment.clone(), value) })
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

    /// Returns a copied snapshot of this tensor's memory descriptor.
    pub fn memory_info(&self) -> Result<MemoryInfoDetails> {
        ensure!(self.is_tensor()?, NotTensorSnafu);
        let api = &self.environment.inner.api;
        let mut memory_info = ptr::null();
        // SAFETY: the value is live and output storage is writable.
        status_call!(api, GetTensorMemoryInfo(self.as_ptr(), &mut memory_info))?;
        let memory_info = require_non_null(memory_info.cast_mut(), "tensor memory info")?;
        memory_info_details(api, memory_info)
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

impl Value<'static> {
    /// Allocates a tensor owned by ONNX Runtime using its default CPU allocator.
    pub fn allocate_tensor<T: TensorElement>(
        environment: Environment,
        shape: impl AsRef<[i64]>,
    ) -> Result<Self> {
        Self::allocate_tensor_with_type(environment, shape, T::ELEMENT_TYPE)
    }

    /// Allocates a tensor with an explicitly selected ONNX element type.
    pub fn allocate_tensor_with_type(
        environment: Environment,
        shape: impl AsRef<[i64]>,
        element_type: ElementType,
    ) -> Result<Self> {
        let shape = shape.as_ref();
        element_count(shape)?;
        let api = &environment.inner.api;
        let allocator = ffi::default_allocator(api)?;
        let mut value = ptr::null_mut();
        // SAFETY: the default allocator has process lifetime, shape remains
        // live for the call, and output storage is writable.
        status_call!(
            api,
            CreateTensorAsOrtValue(
                allocator,
                shape.as_ptr(),
                shape.len(),
                to_native_element_type(element_type),
                &mut value,
            )
        )?;
        let value = NonNull::new(require_non_null(value, "allocated tensor value")?)
            .expect("require_non_null returned a non-null pointer");
        // SAFETY: CreateTensorAsOrtValue transfers unique ownership to us.
        Ok(unsafe { Self::from_raw_owned(environment, value) })
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

/// ONNX category stored in a low-level [`Value`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValueType {
    /// The runtime did not identify the category.
    Unknown,
    /// A dense tensor.
    Tensor,
    /// A sequence of values.
    Sequence,
    /// A key/value map represented by two child values.
    Map,
    /// A runtime-specific opaque value.
    Opaque,
    /// A sparse tensor.
    SparseTensor,
    /// An optional value with zero or one child.
    Optional,
    /// A category introduced by a newer runtime.
    Other(u32),
}

impl From<sys::ONNXType> for ValueType {
    fn from(value: sys::ONNXType) -> Self {
        match value {
            sys::ONNXType::ONNX_TYPE_UNKNOWN => Self::Unknown,
            sys::ONNXType::ONNX_TYPE_TENSOR => Self::Tensor,
            sys::ONNXType::ONNX_TYPE_SEQUENCE => Self::Sequence,
            sys::ONNXType::ONNX_TYPE_MAP => Self::Map,
            sys::ONNXType::ONNX_TYPE_OPAQUE => Self::Opaque,
            sys::ONNXType::ONNX_TYPE_SPARSETENSOR => Self::SparseTensor,
            sys::ONNXType::ONNX_TYPE_OPTIONAL => Self::Optional,
            value => Self::Other(value.0),
        }
    }
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

    /// Creates a session allocator for the requested memory location.
    pub fn allocator<'session>(
        &'session self,
        memory_info: &MemoryInfo,
    ) -> Result<Allocator<'session>> {
        ensure!(
            Environment::same_runtime(&self.environment, &memory_info.environment),
            ForeignEnvironmentSnafu
        );
        let api = &self.environment.inner.api;
        let mut allocator = ptr::null_mut();
        // SAFETY: the session and descriptor are live and output storage is writable.
        status_call!(
            api,
            CreateAllocator(self.as_ptr(), memory_info.as_ptr(), &mut allocator)
        )?;
        Ok(Allocator {
            session: self,
            allocator: NonNull::new(require_non_null(allocator, "session allocator")?)
                .expect("require_non_null returned a non-null pointer"),
        })
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

    /// Submits a low-level run to ONNX Runtime's native asynchronous API.
    ///
    /// Dropping an unfinished future requests cancellation and waits for the
    /// native callback before releasing any borrowed input value.
    pub fn run_raw_async<'a>(
        &'a self,
        inputs: &'a [Input<'a>],
        output_names: &[&str],
    ) -> Result<RawRunFuture<'a>> {
        let options = self.run_options()?;
        self.run_raw_async_with_options(inputs, output_names, options)
    }

    /// Submits a native asynchronous run with owned reusable run options.
    pub fn run_raw_async_with_options<'a>(
        &'a self,
        inputs: &'a [Input<'a>],
        output_names: &[&str],
        options: RunOptions,
    ) -> Result<RawRunFuture<'a>> {
        ensure!(options.belongs_to(self), ForeignEnvironmentSnafu);
        for input in inputs {
            ensure!(
                Environment::same_runtime(&self.environment, input.environment),
                ForeignEnvironmentSnafu
            );
        }

        let input_names = inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<Vec<_>>();
        let input_name_ptrs = input_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let input_value_ptrs = inputs.iter().map(Input::value_ptr).collect::<Vec<_>>();
        let output_names = output_names
            .iter()
            .map(|name| cstring(name, "raw output").map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        let output_name_ptrs = output_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let output_slots = vec![ptr::null_mut(); output_names.len()];
        let api = &self.environment.inner.api;
        let completion = Arc::new(AsyncCompletion::new(api.clone(), output_names.len()));
        let mut future = RawRunFuture {
            session: self,
            environment: self.environment.clone(),
            completion,
            run_options: options,
            _input_names: input_names,
            input_name_ptrs,
            input_value_ptrs,
            _output_names: output_names,
            output_name_ptrs,
            output_slots,
            _input_borrow: PhantomData,
            finished: false,
        };

        let callback_state = Arc::into_raw(future.completion.clone());
        // SAFETY: every pointer array is retained in the future, the input
        // borrow lasts until callback completion, and callback_state contains
        // exactly one strong Arc reference reclaimed by async_run_callback.
        let status = unsafe {
            api_function!(api, RunAsync)(
                self.session as *mut sys::OrtSession,
                future.run_options.as_ptr(),
                future.input_name_ptrs.as_ptr(),
                future.input_value_ptrs.as_ptr(),
                future.input_value_ptrs.len(),
                future.output_name_ptrs.as_ptr(),
                future.output_name_ptrs.len(),
                future.output_slots.as_mut_ptr(),
                Some(async_run_callback),
                callback_state.cast_mut().cast(),
            )
        };
        if !status.is_null() {
            // SAFETY: submission failed, so ORT will not invoke the callback.
            unsafe { drop(Arc::from_raw(callback_state)) };
            future.finished = true;
            check_status(api, status)?;
        }
        Ok(future)
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
                Ok(unsafe { Value::from_raw_owned(self.environment.clone(), value) })
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

/// Future returned by [`Session::run_raw_async`].
#[must_use = "futures do nothing unless awaited or polled"]
pub struct RawRunFuture<'a> {
    session: &'a Session,
    environment: Environment,
    completion: Arc<AsyncCompletion>,
    run_options: RunOptions,
    _input_names: Vec<CString>,
    input_name_ptrs: Vec<*const std::ffi::c_char>,
    input_value_ptrs: Vec<*const sys::OrtValue>,
    _output_names: Vec<CString>,
    output_name_ptrs: Vec<*const std::ffi::c_char>,
    output_slots: Vec<*mut sys::OrtValue>,
    _input_borrow: PhantomData<&'a [Input<'a>]>,
    finished: bool,
}

impl Future for RawRunFuture<'_> {
    type Output = Result<Vec<Value<'static>>>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        assert!(!self.finished, "RawRunFuture polled after completion");
        let Some(result) = self.completion.poll_result(context.waker()) else {
            return Poll::Pending;
        };
        self.finished = true;
        Poll::Ready(match result {
            Ok(values) => Ok(values
                .into_iter()
                .map(|value| {
                    let value = NonNull::new(value as *mut sys::OrtValue)
                        .expect("async completion rejected null outputs");
                    // SAFETY: callback completion transfers unique ownership.
                    unsafe { Value::from_raw_owned(self.environment.clone(), value) }
                })
                .collect()),
            Err(source) => Err(Error::Async { source }),
        })
    }
}

impl RawRunFuture<'_> {
    /// Requests cooperative cancellation of the native run.
    pub fn cancel(&self) -> Result<()> {
        self.run_options.terminate()?;
        Ok(())
    }
}

impl Drop for RawRunFuture<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let _ = self.run_options.terminate();
        if let Ok(values) = self.completion.wait_result() {
            for value in values {
                // SAFETY: successful callback outputs not consumed by poll are
                // uniquely owned by this future.
                unsafe {
                    api_function!(self.session.environment.inner.api, ReleaseValue)(
                        value as *mut sys::OrtValue,
                    )
                };
            }
        }
        self.finished = true;
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

fn memory_info_name(api: &sys::Api, memory_info: *const sys::OrtMemoryInfo) -> Result<String> {
    let mut name = ptr::null();
    // SAFETY: memory_info is live and output storage is writable. The returned
    // string is owned by memory_info and callers tie the borrow accordingly.
    status_call!(api, MemoryInfoGetName(memory_info, &mut name))?;
    let name = require_non_null(name.cast_mut(), "memory info name")?;
    // SAFETY: ONNX Runtime documents this as a NUL-terminated string valid for
    // the lifetime of memory_info. Copy it before returning.
    Ok(unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned())
}

fn memory_info_allocator_type(
    api: &sys::Api,
    memory_info: *const sys::OrtMemoryInfo,
) -> Result<AllocatorType> {
    let mut allocator_type = sys::OrtAllocatorType::OrtInvalidAllocator;
    // SAFETY: memory_info is live and output storage is writable.
    status_call!(api, MemoryInfoGetType(memory_info, &mut allocator_type))?;
    match allocator_type {
        sys::OrtAllocatorType::OrtDeviceAllocator => Ok(AllocatorType::Device),
        sys::OrtAllocatorType::OrtArenaAllocator => Ok(AllocatorType::Arena),
        value => Err(Error::UnknownAllocatorType { value: value.0 }),
    }
}

fn memory_info_device_id(api: &sys::Api, memory_info: *const sys::OrtMemoryInfo) -> Result<i32> {
    let mut device_id = 0;
    // SAFETY: memory_info is live and output storage is writable.
    status_call!(api, MemoryInfoGetId(memory_info, &mut device_id))?;
    Ok(device_id)
}

fn memory_info_memory_type(
    api: &sys::Api,
    memory_info: *const sys::OrtMemoryInfo,
) -> Result<MemoryType> {
    let mut memory_type = sys::OrtMemType::OrtMemTypeDefault;
    // SAFETY: memory_info is live and output storage is writable.
    status_call!(api, MemoryInfoGetMemType(memory_info, &mut memory_type))?;
    match memory_type {
        sys::OrtMemType::OrtMemTypeCPUInput => Ok(MemoryType::CpuInput),
        sys::OrtMemType::OrtMemTypeCPUOutput => Ok(MemoryType::CpuOutput),
        sys::OrtMemType::OrtMemTypeDefault => Ok(MemoryType::Default),
        value => Err(Error::UnknownMemoryType { value: value.0 }),
    }
}

fn memory_info_device_type(api: &sys::Api, memory_info: *const sys::OrtMemoryInfo) -> DeviceType {
    let mut device_type = sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU;
    // SAFETY: memory_info is live and output storage is writable.
    unsafe { api_function!(api, MemoryInfoGetDeviceType)(memory_info, &mut device_type) };
    match device_type {
        sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU => DeviceType::Cpu,
        sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU => DeviceType::Gpu,
        sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_FPGA => DeviceType::Fpga,
        value => DeviceType::Unknown(value.0),
    }
}

fn memory_info_details(
    api: &sys::Api,
    memory_info: *const sys::OrtMemoryInfo,
) -> Result<MemoryInfoDetails> {
    Ok(MemoryInfoDetails {
        name: memory_info_name(api, memory_info)?,
        allocator_type: memory_info_allocator_type(api, memory_info)?,
        device_id: memory_info_device_id(api, memory_info)?,
        memory_type: memory_info_memory_type(api, memory_info)?,
        device_type: memory_info_device_type(api, memory_info),
    })
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
