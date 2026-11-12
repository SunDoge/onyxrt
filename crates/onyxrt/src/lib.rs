//! Safe, CPU-first ONNX Runtime bindings with DLPack interoperability.

use std::{
    ffi::{CStr, CString, NulError, OsStr},
    future::Future,
    marker::PhantomData,
    path::{Path, PathBuf},
    pin::Pin,
    ptr,
    string::FromUtf8Error,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Waker},
};

use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase, TensorMut, TensorRef, TensorView,
    ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType},
    metadata::Dynamic,
    versioned,
};
use onyxrt_sys as sys;
use snafu::Snafu;

/// Result type used by this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors returned by the safe ONNX Runtime binding.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    /// Loading or negotiating with the ONNX Runtime dynamic library failed.
    #[snafu(display("failed to initialize ONNX Runtime"))]
    LoadRuntime { source: sys::Error },

    /// ONNX Runtime returned a status error.
    #[snafu(display("ONNX Runtime error {code:?}: {message}"))]
    Runtime {
        /// Native error code.
        code: sys::OrtErrorCode,
        /// Native diagnostic message.
        message: String,
    },

    /// A native API reported success but returned a null pointer.
    #[snafu(display("ONNX Runtime returned null for {what}"))]
    NullPointer {
        /// Name of the expected value.
        what: &'static str,
    },

    /// A name contained an interior NUL byte.
    #[snafu(display("{kind} name contains an interior NUL byte"))]
    InvalidName {
        /// Kind of name being converted.
        kind: &'static str,
        /// Conversion error.
        source: NulError,
    },

    /// A model path cannot be represented by the platform ORT path type.
    #[snafu(display("model path `{}` contains an interior NUL", path.display()))]
    InvalidModelPath {
        /// Invalid model path.
        path: PathBuf,
    },

    /// DLPack descriptor validation failed.
    #[snafu(display("invalid DLPack tensor: {source}"))]
    DlpackTensor { source: dlpark::tensor::Error },

    /// DLPack metadata allocation failed.
    #[snafu(display("could not create DLPack output metadata: {source}"))]
    DlpackMetadata { source: dlpark::metadata::Error },

    /// Only CPU tensors are supported by the current provider implementation.
    #[snafu(display(
        "DLPack device {:?}:{} is unsupported; expected CPU:0",
        device_type,
        device_id
    ))]
    UnsupportedDevice {
        /// DLPack device type.
        device_type: DLDeviceType,
        /// DLPack device index.
        device_id: i32,
    },

    /// An I/O Binding tensor uses a device this binding cannot describe.
    #[snafu(display(
        "DLPack device {:?}:{} is unsupported for I/O Binding; expected CPU:0 or CUDA:<non-negative>",
        device_type,
        device_id
    ))]
    UnsupportedIoBindingDevice {
        /// DLPack device type.
        device_type: DLDeviceType,
        /// Device index.
        device_id: i32,
    },

    /// ORT returned tensor memory that cannot be represented as DLPack here.
    #[snafu(display("unsupported ONNX Runtime tensor memory `{name}` on device {device_id}"))]
    UnsupportedOrtMemory {
        /// ORT allocator or device name.
        name: String,
        /// ORT device index.
        device_id: i32,
    },

    /// A DLPack element type has no ONNX Runtime mapping yet.
    #[snafu(display(
        "unsupported DLPack dtype (code={}, bits={}, lanes={})",
        dtype.code.0,
        dtype.bits,
        dtype.lanes
    ))]
    UnsupportedDlpackType {
        /// Unsupported descriptor.
        dtype: DLDataType,
    },

    /// A DLPack storage-layout flag is incompatible with ONNX Runtime.
    #[snafu(display(
        "unsupported DLPack flags {flags:?} for dtype (code={}, bits={}, lanes={})",
        dtype.code.0,
        dtype.bits,
        dtype.lanes
    ))]
    UnsupportedDlpackFlags {
        /// Incompatible DLPack flags.
        flags: DlpackFlags,
        /// Tensor element descriptor.
        dtype: DLDataType,
    },

    /// An ONNX Runtime output type has no DLPack mapping yet.
    #[snafu(display("unsupported ONNX tensor element type {element_type:?}"))]
    UnsupportedOnnxType {
        /// Unsupported ONNX type.
        element_type: sys::ONNXTensorElementDataType,
    },

    /// A model input or output is not a tensor.
    #[snafu(display("unsupported ONNX value type {value_type:?}; expected a tensor"))]
    UnsupportedValueType {
        /// Model value kind.
        value_type: sys::ONNXType,
    },

    /// ORT unexpectedly replaced a caller-provided output value.
    #[snafu(display("ONNX Runtime replaced preallocated output `{name}`"))]
    ReplacedOutput {
        /// Output name whose value pointer changed.
        name: String,
    },

    /// ONNX Runtime returned an invalid provider count.
    #[snafu(display("ONNX Runtime returned invalid provider count {count}"))]
    InvalidProviderCount {
        /// Negative provider count returned by the native API.
        count: i32,
    },

    /// The native asynchronous callback returned an unexpected output count.
    #[snafu(display("ONNX Runtime async callback returned {actual} outputs; expected {expected}"))]
    AsyncOutputCount {
        /// Expected number of outputs.
        expected: usize,
        /// Count supplied to the callback.
        actual: usize,
    },

    /// I/O Binding returned a different number of outputs than requested.
    #[snafu(display("ONNX Runtime I/O Binding returned {actual} outputs; expected {expected}"))]
    BoundOutputCount {
        /// Requested number of outputs.
        expected: usize,
        /// Number returned by ORT.
        actual: usize,
    },

    /// Internal Rust callback processing panicked.
    #[snafu(display("ONNX Runtime async callback processing panicked"))]
    AsyncCallbackPanicked,

    /// Run options were created by a different ONNX Runtime environment.
    #[snafu(display("run options belong to a different ONNX Runtime environment"))]
    ForeignRunOptions,

    /// A string tensor shape contained a negative dimension.
    #[snafu(display("string tensor shape dimension {axis} is negative: {value}"))]
    InvalidStringDimension { axis: usize, value: i64 },

    /// A string tensor's shape and number of values disagree.
    #[snafu(display("string tensor shape requires {expected} values, got {actual}"))]
    StringElementCount { expected: usize, actual: usize },

    /// A string tensor element contained an interior NUL byte.
    #[snafu(display("string tensor element {index} contains an interior NUL byte"))]
    InvalidStringElement { index: usize, source: NulError },

    /// Computing a string tensor's element count overflowed `usize`.
    #[snafu(display("string tensor element count overflows usize"))]
    StringElementCountOverflow,

    /// ONNX Runtime returned a string tensor element that was not valid UTF-8.
    #[snafu(display("string tensor element {index} is not valid UTF-8"))]
    InvalidStringUtf8 { index: usize, source: FromUtf8Error },

    /// ONNX Runtime returned invalid offsets for string tensor storage.
    #[snafu(display(
        "string tensor element {index} has invalid byte range {start}..{end} for {data_len} bytes"
    ))]
    InvalidStringOffsets {
        index: usize,
        start: usize,
        end: usize,
        data_len: usize,
    },

    /// A container value returned an unexpected number of children.
    #[snafu(display("{kind} value contained {actual} children; expected {expected}"))]
    ContainerValueCount {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
}

/// An ONNX tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ElementType {
    /// Element type is not specified.
    Undefined,
    /// IEEE 754 binary32.
    Float32,
    /// Unsigned 8-bit integer.
    Uint8,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 16-bit integer.
    Uint16,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// UTF-8 string.
    String,
    /// Boolean.
    Bool,
    /// IEEE 754 binary16.
    Float16,
    /// IEEE 754 binary64.
    Float64,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Unsigned 64-bit integer.
    Uint64,
    /// Complex value composed of two binary32 values.
    Complex64,
    /// Complex value composed of two binary64 values.
    Complex128,
    /// Brain floating point with 16 bits.
    BFloat16,
    /// Float8 E4M3FN.
    Float8E4M3Fn,
    /// Float8 E4M3FNUZ.
    Float8E4M3Fnuz,
    /// Float8 E5M2.
    Float8E5M2,
    /// Float8 E5M2FNUZ.
    Float8E5M2Fnuz,
    /// Unsigned packed 4-bit integer.
    Uint4,
    /// Signed packed 4-bit integer.
    Int4,
    /// Float4 E2M1.
    Float4E2M1,
    /// Unsigned packed 2-bit integer.
    Uint2,
    /// Signed packed 2-bit integer.
    Int2,
    /// Float8 E8M0.
    Float8E8M0,
}

impl From<sys::ONNXTensorElementDataType> for ElementType {
    fn from(value: sys::ONNXTensorElementDataType) -> Self {
        use sys::ONNXTensorElementDataType as O;
        match value {
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED => Self::Undefined,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Self::Float32,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Self::Uint8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => Self::Int8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => Self::Uint16,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => Self::Int16,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Self::Int32,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Self::Int64,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => Self::String,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Self::Bool,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Self::Float16,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => Self::Float64,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => Self::Uint32,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => Self::Uint64,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 => Self::Complex64,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 => Self::Complex128,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => Self::BFloat16,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN => Self::Float8E4M3Fn,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ => Self::Float8E4M3Fnuz,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 => Self::Float8E5M2,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ => Self::Float8E5M2Fnuz,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 => Self::Uint4,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 => Self::Int4,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1 => Self::Float4E2M1,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2 => Self::Uint2,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2 => Self::Int2,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0 => Self::Float8E8M0,
            _ => Self::Undefined,
        }
    }
}

/// A model tensor dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    /// A statically known non-negative extent.
    Fixed(u64),
    /// An unknown extent identified by a model symbol such as `batch`.
    Symbolic(String),
    /// An unknown extent without a symbolic name.
    Dynamic,
}

/// Metadata for one model tensor input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    element_type: ElementType,
    dimensions: Vec<Dimension>,
}

/// Tensor type metadata used inside recursive model value descriptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorTypeInfo {
    element_type: ElementType,
    dimensions: Vec<Dimension>,
}

/// Recursive ONNX model value type metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueTypeInfo {
    /// A dense tensor.
    Tensor(TensorTypeInfo),
    /// A sequence whose elements share one ONNX type.
    Sequence(Box<ValueTypeInfo>),
    /// A map with scalar tensor keys and recursively described values.
    Map {
        /// Scalar element type used by the key tensor.
        key: ElementType,
        /// Type of each map value.
        value: Box<ValueTypeInfo>,
    },
    /// An optional value and its contained type.
    Optional(Box<ValueTypeInfo>),
    /// A sparse tensor. ORT's model-reflection API does not expose its shape
    /// through the dense tensor type-info cast.
    SparseTensor,
    /// An implementation-defined opaque value.
    Opaque,
    /// A value whose ONNX type is unknown.
    Unknown,
}

/// Metadata for one named model input, output, or overridable initializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueInfo {
    name: String,
    value_type: ValueTypeInfo,
}

/// Read-only metadata embedded in an ONNX model.
pub struct ModelMetadata {
    environment: Environment,
    metadata: usize,
}

// SAFETY: model metadata is immutable after session creation and all access is
// through ORT's read-only query functions.
unsafe impl Send for ModelMetadata {}
// SAFETY: queries do not mutate the native metadata object.
unsafe impl Sync for ModelMetadata {}

impl ModelMetadata {
    /// Returns the producer name embedded in the model.
    pub fn producer_name(&self) -> Result<String> {
        self.string(MetadataStringKind::Producer)
    }

    /// Returns the model graph name.
    pub fn graph_name(&self) -> Result<String> {
        self.string(MetadataStringKind::GraphName)
    }

    /// Returns the model domain.
    pub fn domain(&self) -> Result<String> {
        self.string(MetadataStringKind::Domain)
    }

    /// Returns the model description.
    pub fn description(&self) -> Result<String> {
        self.string(MetadataStringKind::Description)
    }

    /// Returns the graph description.
    pub fn graph_description(&self) -> Result<String> {
        self.string(MetadataStringKind::GraphDescription)
    }

    /// Returns the model version.
    pub fn version(&self) -> Result<i64> {
        let api = &self.environment.inner.api;
        let mut version = 0;
        // SAFETY: metadata is live and version is writable output storage.
        check_status(api, unsafe {
            (api.table().ModelMetadataGetVersion)(self.as_ptr(), &mut version)
        })?;
        Ok(version)
    }

    /// Looks up a custom metadata value. Missing keys return `None`.
    pub fn custom(&self, key: &str) -> Result<Option<String>> {
        let key = cstring(key, "model metadata key")?;
        let api = &self.environment.inner.api;
        let allocator = default_allocator(api)?;
        let mut value = ptr::null_mut();
        // SAFETY: metadata, allocator, and key are live; value is writable.
        check_status(api, unsafe {
            (api.table().ModelMetadataLookupCustomMetadataMap)(
                self.as_ptr(),
                allocator,
                key.as_ptr(),
                &mut value,
            )
        })?;
        if value.is_null() {
            return Ok(None);
        }
        let guard = AllocatedBufferGuard {
            api,
            allocator,
            allocation: value.cast(),
        };
        // SAFETY: ORT returned an allocator-owned NUL-terminated string.
        let value = unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned();
        drop(guard);
        Ok(Some(value))
    }

    fn string(&self, kind: MetadataStringKind) -> Result<String> {
        let api = &self.environment.inner.api;
        let allocator = default_allocator(api)?;
        let mut value = ptr::null_mut();
        // SAFETY: metadata and allocator are live; value is writable output.
        let status = unsafe {
            match kind {
                MetadataStringKind::Producer => {
                    (api.table().ModelMetadataGetProducerName)(self.as_ptr(), allocator, &mut value)
                }
                MetadataStringKind::GraphName => {
                    (api.table().ModelMetadataGetGraphName)(self.as_ptr(), allocator, &mut value)
                }
                MetadataStringKind::Domain => {
                    (api.table().ModelMetadataGetDomain)(self.as_ptr(), allocator, &mut value)
                }
                MetadataStringKind::Description => {
                    (api.table().ModelMetadataGetDescription)(self.as_ptr(), allocator, &mut value)
                }
                MetadataStringKind::GraphDescription => (api
                    .table()
                    .ModelMetadataGetGraphDescription)(
                    self.as_ptr(), allocator, &mut value
                ),
            }
        };
        check_status(api, status)?;
        let value = require_non_null(value, "model metadata string")?;
        let guard = AllocatedBufferGuard {
            api,
            allocator,
            allocation: value.cast(),
        };
        // SAFETY: ORT returned an allocator-owned NUL-terminated string.
        let value = unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned();
        drop(guard);
        Ok(value)
    }

    fn as_ptr(&self) -> *const sys::OrtModelMetadata {
        self.metadata as *const sys::OrtModelMetadata
    }
}

impl Drop for ModelMetadata {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns the native metadata handle.
        unsafe {
            (self.environment.inner.api.table().ReleaseModelMetadata)(
                self.metadata as *mut sys::OrtModelMetadata,
            )
        };
    }
}

#[derive(Clone, Copy)]
enum MetadataStringKind {
    Producer,
    GraphName,
    Domain,
    Description,
    GraphDescription,
}

impl TensorInfo {
    /// Returns the model value name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tensor element type.
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Returns the tensor dimensions in model order.
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }
}

impl TensorTypeInfo {
    /// Returns the tensor element type.
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Returns the tensor dimensions in model order.
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }
}

impl ValueInfo {
    /// Returns the model value name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the recursive ONNX value type.
    pub fn value_type(&self) -> &ValueTypeInfo {
        &self.value_type
    }
}

/// A loaded ONNX Runtime environment.
#[derive(Clone)]
pub struct Environment {
    inner: Arc<EnvironmentInner>,
}

struct EnvironmentInner {
    api: sys::Api,
    env: usize,
}

// SAFETY: ORT environments are process-wide configuration objects documented
// for use by sessions on multiple threads. The pointer is only released in Drop.
unsafe impl Send for EnvironmentInner {}
// SAFETY: EnvironmentInner exposes no mutation and ORT permits concurrent use.
unsafe impl Sync for EnvironmentInner {}

impl Drop for EnvironmentInner {
    fn drop(&mut self) {
        // SAFETY: env was created by this exact API table and is released once.
        unsafe { (self.api.table().ReleaseEnv)(self.env as *mut sys::OrtEnv) };
    }
}

impl Environment {
    /// Starts configuring an environment.
    pub fn builder() -> EnvironmentBuilder {
        EnvironmentBuilder::default()
    }

    /// Returns the loaded ONNX Runtime version string.
    pub fn runtime_version(&self) -> &str {
        self.inner.api.version()
    }

    /// Returns the linked runtime or dynamic-library origin.
    pub fn runtime_origin(&self) -> &str {
        self.inner.api.origin()
    }

    /// Returns execution providers compiled into the loaded runtime.
    ///
    /// Presence in this list does not guarantee that a provider's external
    /// dependencies are installed or that it can create a working session.
    pub fn available_providers(&self) -> Result<Vec<String>> {
        let api = &self.inner.api;
        let mut providers = ptr::null_mut();
        let mut count = 0_i32;
        // SAFETY: both outputs point to writable storage. On success, ownership
        // of the returned array is transferred to AvailableProvidersGuard.
        check_status(api, unsafe {
            (api.table().GetAvailableProviders)(&mut providers, &mut count)
        })?;
        if count < 0 {
            if !providers.is_null() {
                // SAFETY: the pointer came from GetAvailableProviders and must
                // be paired with the exact count returned by that call.
                discard_provider_array(api, providers, count);
            }
            return Err(Error::InvalidProviderCount { count });
        }
        let guard = AvailableProvidersGuard {
            api,
            providers,
            count,
        };
        if count == 0 {
            return Ok(Vec::new());
        }
        let providers = require_non_null(providers, "available provider array")?;
        let count = count as usize;
        // SAFETY: ORT returned an array containing exactly count pointers.
        let pointers = unsafe { std::slice::from_raw_parts(providers, count) };
        let mut names = Vec::with_capacity(count);
        for pointer in pointers {
            let pointer = require_const_non_null(*pointer, "available provider name")?;
            // SAFETY: every provider entry is documented as NUL-terminated.
            names.push(
                unsafe { CStr::from_ptr(pointer) }
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        drop(guard);
        Ok(names)
    }

    /// Starts configuring a CPU session.
    pub fn session_builder(&self) -> SessionBuilder {
        SessionBuilder {
            environment: self.clone(),
            intra_threads: None,
            inter_threads: None,
            optimization: GraphOptimizationLevel::All,
            execution_mode: ExecutionMode::Sequential,
            cpu_provider: CpuExecutionProvider::default(),
            execution_providers: Vec::new(),
            memory_pattern: true,
            profiling_prefix: None,
            optimized_model_path: None,
            log_id: None,
            log_level: None,
            log_verbosity: None,
            deterministic_compute: None,
            config_entries: Vec::new(),
            custom_ops_libraries: Vec::new(),
        }
    }
}

/// Builder for [`Environment`].
pub struct EnvironmentBuilder {
    name: String,
    severity: LogLevel,
    library_path: Option<PathBuf>,
}

impl Default for EnvironmentBuilder {
    fn default() -> Self {
        Self {
            name: "onyxrt".to_owned(),
            severity: LogLevel::Warning,
            library_path: None,
        }
    }
}

impl EnvironmentBuilder {
    /// Sets the identifier included in runtime logs.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the minimum runtime log severity.
    pub fn log_level(mut self, severity: LogLevel) -> Self {
        self.severity = severity;
        self
    }

    /// Uses an exact ONNX Runtime dynamic library path.
    ///
    /// Building with `link-static` makes [`Self::build`] return
    /// [`Error::LoadRuntime`] when this option is set.
    pub fn library_path(mut self, path: impl AsRef<OsStr>) -> Self {
        self.library_path = Some(PathBuf::from(path.as_ref()));
        self
    }

    /// Loads ONNX Runtime and creates the environment.
    pub fn build(self) -> Result<Environment> {
        let api = match self.library_path {
            Some(path) => sys::Api::load(path),
            None => sys::Api::load_default(),
        }
        .map_err(|source| Error::LoadRuntime { source })?;
        let name = cstring(&self.name, "environment")?;
        let mut env = ptr::null_mut();
        // SAFETY: arguments are valid and output points to writable storage.
        let status =
            unsafe { (api.table().CreateEnv)(self.severity.into(), name.as_ptr(), &mut env) };
        check_status(&api, status)?;
        let env = require_non_null(env, "environment")?;
        Ok(Environment {
            inner: Arc::new(EnvironmentInner {
                api,
                env: env as usize,
            }),
        })
    }
}

/// ONNX Runtime logging threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Verbose diagnostic messages.
    Verbose,
    /// Informational messages.
    Info,
    /// Warnings and errors.
    Warning,
    /// Errors only.
    Error,
    /// Fatal errors only.
    Fatal,
}

impl From<LogLevel> for sys::OrtLoggingLevel {
    fn from(value: LogLevel) -> Self {
        match value {
            LogLevel::Verbose => Self::ORT_LOGGING_LEVEL_VERBOSE,
            LogLevel::Info => Self::ORT_LOGGING_LEVEL_INFO,
            LogLevel::Warning => Self::ORT_LOGGING_LEVEL_WARNING,
            LogLevel::Error => Self::ORT_LOGGING_LEVEL_ERROR,
            LogLevel::Fatal => Self::ORT_LOGGING_LEVEL_FATAL,
        }
    }
}

/// Session graph optimization setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOptimizationLevel {
    /// Disable graph rewrites.
    Disabled,
    /// Safe/basic graph rewrites.
    Basic,
    /// Basic plus extended rewrites.
    Extended,
    /// All available rewrites.
    All,
}

impl From<GraphOptimizationLevel> for sys::GraphOptimizationLevel {
    fn from(value: GraphOptimizationLevel) -> Self {
        match value {
            GraphOptimizationLevel::Disabled => Self::ORT_DISABLE_ALL,
            GraphOptimizationLevel::Basic => Self::ORT_ENABLE_BASIC,
            GraphOptimizationLevel::Extended => Self::ORT_ENABLE_EXTENDED,
            GraphOptimizationLevel::All => Self::ORT_ENABLE_ALL,
        }
    }
}

/// Session graph execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Executes graph nodes sequentially. This is the CPU-friendly default.
    Sequential,
    /// Allows independent graph branches to execute in parallel.
    Parallel,
}

impl From<ExecutionMode> for sys::ExecutionMode {
    fn from(value: ExecutionMode) -> Self {
        match value {
            ExecutionMode::Sequential => Self::ORT_SEQUENTIAL,
            ExecutionMode::Parallel => Self::ORT_PARALLEL,
        }
    }
}

/// Configuration for ONNX Runtime's built-in CPU execution provider.
///
/// The CPU provider is always registered by ONNX Runtime and is the fallback
/// provider even when accelerator providers are added later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuExecutionProvider {
    arena: bool,
}

impl Default for CpuExecutionProvider {
    fn default() -> Self {
        Self { arena: true }
    }
}

impl CpuExecutionProvider {
    /// Creates the default CPU provider configuration.
    pub const fn new() -> Self {
        Self { arena: true }
    }

    /// Enables or disables the CPU memory arena.
    ///
    /// The arena generally improves repeated-inference throughput at the cost
    /// of retaining allocations for reuse.
    pub const fn with_arena(mut self, enabled: bool) -> Self {
        self.arena = enabled;
        self
    }

    /// Returns whether the CPU memory arena is enabled.
    pub const fn arena_enabled(self) -> bool {
        self.arena
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ProviderOptions(Vec<(String, String)>);

impl ProviderOptions {
    fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some((_, current)) = self.0.iter_mut().find(|(current, _)| current == &key) {
            *current = value;
        } else {
            self.0.push((key, value));
        }
    }

    fn encode(&self, kind: &'static str) -> Result<EncodedProviderOptions> {
        let mut keys = Vec::with_capacity(self.0.len());
        let mut values = Vec::with_capacity(self.0.len());
        for (key, value) in &self.0 {
            keys.push(cstring(key, kind)?);
            values.push(cstring(value, kind)?);
        }
        let key_pointers = keys.iter().map(|value| value.as_ptr()).collect();
        let value_pointers = values.iter().map(|value| value.as_ptr()).collect();
        Ok(EncodedProviderOptions {
            _keys: keys,
            _values: values,
            key_pointers,
            value_pointers,
        })
    }
}

struct EncodedProviderOptions {
    _keys: Vec<CString>,
    _values: Vec<CString>,
    key_pointers: Vec<*const std::ffi::c_char>,
    value_pointers: Vec<*const std::ffi::c_char>,
}

impl EncodedProviderOptions {
    fn len(&self) -> usize {
        self.key_pointers.len()
    }
}

/// CUDA memory-arena growth policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaArenaExtendStrategy {
    /// Grow allocations to the next power of two.
    NextPowerOfTwo,
    /// Grow by exactly the requested allocation size.
    SameAsRequested,
}

impl CudaArenaExtendStrategy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::NextPowerOfTwo => "kNextPowerOfTwo",
            Self::SameAsRequested => "kSameAsRequested",
        }
    }
}

/// cuDNN convolution-algorithm selection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnConvAlgoSearch {
    /// Benchmark all candidate algorithms.
    Exhaustive,
    /// Use cuDNN's heuristic selection.
    Heuristic,
    /// Use the default algorithm.
    Default,
}

impl CudnnConvAlgoSearch {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Exhaustive => "EXHAUSTIVE",
            Self::Heuristic => "HEURISTIC",
            Self::Default => "DEFAULT",
        }
    }
}

/// Configuration for the CUDA execution provider.
///
/// ONNX Runtime loads `libonnxruntime_providers_cuda` when this provider is
/// appended. The provider library and its matching CUDA/cuDNN dependencies
/// must be discoverable by the platform loader.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaExecutionProvider {
    options: ProviderOptions,
}

impl CudaExecutionProvider {
    /// Creates a CUDA provider using ONNX Runtime defaults (device 0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects the CUDA device ordinal.
    pub fn with_device_id(mut self, device_id: i32) -> Self {
        self.options.insert("device_id", device_id.to_string());
        self
    }

    /// Limits the CUDA memory arena in bytes.
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.options.insert("gpu_mem_limit", bytes.to_string());
        self
    }

    /// Selects how the CUDA memory arena grows.
    pub fn with_arena_extend_strategy(mut self, strategy: CudaArenaExtendStrategy) -> Self {
        self.options
            .insert("arena_extend_strategy", strategy.as_str());
        self
    }

    /// Selects how cuDNN convolution algorithms are chosen.
    pub fn with_cudnn_conv_algo_search(mut self, search: CudnnConvAlgoSearch) -> Self {
        self.options
            .insert("cudnn_conv_algo_search", search.as_str());
        self
    }

    /// Controls whether copies use the provider's default compute stream.
    ///
    /// ONNX Runtime recommends leaving this enabled; disabling it can cause
    /// data races for some models.
    pub fn with_copy_in_default_stream(mut self, enabled: bool) -> Self {
        self.options
            .insert("do_copy_in_default_stream", bool_option(enabled));
        self
    }

    /// Allows cuDNN convolution search to use its maximum workspace.
    pub fn with_cudnn_max_workspace(mut self, enabled: bool) -> Self {
        self.options
            .insert("cudnn_conv_use_max_workspace", bool_option(enabled));
        self
    }

    /// Enables or disables TensorFloat-32 math where supported.
    pub fn with_tf32(mut self, enabled: bool) -> Self {
        self.options.insert("use_tf32", bool_option(enabled));
        self
    }

    /// Enables CUDA graph capture for compatible workloads.
    pub fn with_cuda_graph(mut self, enabled: bool) -> Self {
        self.options
            .insert("enable_cuda_graph", bool_option(enabled));
        self
    }

    /// Sets an ONNX Runtime CUDA provider option not covered by typed methods.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key, value);
        self
    }
}

/// Configuration for the TensorRT execution provider.
///
/// The GPU distribution commonly ships the TensorRT provider shared library,
/// but NVIDIA TensorRT and compatible CUDA libraries are still required.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorRtExecutionProvider {
    options: ProviderOptions,
}

impl TensorRtExecutionProvider {
    /// Creates a TensorRT provider using ONNX Runtime defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects the CUDA device ordinal used by TensorRT.
    pub fn with_device_id(mut self, device_id: i32) -> Self {
        self.options.insert("device_id", device_id.to_string());
        self
    }

    /// Enables or disables FP16 TensorRT kernels.
    pub fn with_fp16(mut self, enabled: bool) -> Self {
        self.options.insert("trt_fp16_enable", bool_option(enabled));
        self
    }

    /// Enables or disables the TensorRT engine cache.
    pub fn with_engine_cache(mut self, enabled: bool) -> Self {
        self.options
            .insert("trt_engine_cache_enable", bool_option(enabled));
        self
    }

    /// Sets the directory used for TensorRT engine caching.
    pub fn with_engine_cache_path(mut self, path: impl AsRef<Path>) -> Self {
        self.options.insert(
            "trt_engine_cache_path",
            path.as_ref().as_os_str().to_string_lossy(),
        );
        self
    }

    /// Sets an ONNX Runtime TensorRT provider option.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key, value);
        self
    }
}

/// Configuration for the legacy in-tree OpenVINO execution provider.
///
/// New deployments should prefer OpenVINO's standalone plugin EP when using
/// an ONNX Runtime release that supports plugin registration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenVinoExecutionProvider {
    options: ProviderOptions,
}

impl OpenVinoExecutionProvider {
    /// Creates an OpenVINO provider using ONNX Runtime defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects an OpenVINO device such as `CPU`, `GPU`, or `NPU`.
    pub fn with_device_type(mut self, device_type: impl Into<String>) -> Self {
        self.options.insert("device_type", device_type);
        self
    }

    /// Sets an ONNX Runtime OpenVINO provider option.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key, value);
        self
    }
}

/// Configuration for the Intel oneDNN (formerly DNNL) execution provider.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DnnlExecutionProvider {
    options: ProviderOptions,
}

impl DnnlExecutionProvider {
    /// Creates a oneDNN provider using ONNX Runtime defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an ONNX Runtime oneDNN provider option.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key, value);
        self
    }
}

/// A provider configured through ONNX Runtime's generic name/options API.
///
/// C API v21 supports providers including `QNN`, `SNPE`, `XNNPACK`, `CoreML`,
/// and `AZURE` through this entry point. CUDA, TensorRT, and OpenVINO require
/// their dedicated builder methods instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedExecutionProvider {
    name: String,
    options: ProviderOptions,
}

impl NamedExecutionProvider {
    /// Creates a named provider configuration.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            options: ProviderOptions::default(),
        }
    }

    /// Sets a provider-specific string option.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key, value);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExecutionProvider {
    Cuda(CudaExecutionProvider),
    TensorRt(TensorRtExecutionProvider),
    OpenVino(OpenVinoExecutionProvider),
    Dnnl(DnnlExecutionProvider),
    Named(NamedExecutionProvider),
}

const fn bool_option(enabled: bool) -> &'static str {
    if enabled { "1" } else { "0" }
}

#[cfg(not(target_os = "windows"))]
struct NativeModelPath(CString);

#[cfg(not(target_os = "windows"))]
impl NativeModelPath {
    fn new(path: &Path) -> Result<Self> {
        #[cfg(unix)]
        use std::os::unix::ffi::OsStrExt;

        #[cfg(unix)]
        let bytes = path.as_os_str().as_bytes();
        #[cfg(not(unix))]
        let fallback = path.to_string_lossy();
        #[cfg(not(unix))]
        let bytes = fallback.as_bytes();

        CString::new(bytes)
            .map(Self)
            .map_err(|_| Error::InvalidModelPath {
                path: path.to_owned(),
            })
    }

    fn as_ptr(&self) -> *const sys::os_char {
        self.0.as_ptr()
    }
}

#[cfg(target_os = "windows")]
struct NativeModelPath(Vec<u16>);

#[cfg(target_os = "windows")]
impl NativeModelPath {
    fn new(path: &Path) -> Result<Self> {
        use std::os::windows::ffi::OsStrExt;

        let mut encoded = path.as_os_str().encode_wide().collect::<Vec<_>>();
        if encoded.contains(&0) {
            return Err(Error::InvalidModelPath {
                path: path.to_owned(),
            });
        }
        encoded.push(0);
        Ok(Self(encoded))
    }

    fn as_ptr(&self) -> *const sys::os_char {
        self.0.as_ptr()
    }
}

/// Builder for an ONNX Runtime inference session.
pub struct SessionBuilder {
    environment: Environment,
    intra_threads: Option<i32>,
    inter_threads: Option<i32>,
    optimization: GraphOptimizationLevel,
    execution_mode: ExecutionMode,
    cpu_provider: CpuExecutionProvider,
    execution_providers: Vec<ExecutionProvider>,
    memory_pattern: bool,
    profiling_prefix: Option<PathBuf>,
    optimized_model_path: Option<PathBuf>,
    log_id: Option<String>,
    log_level: Option<LogLevel>,
    log_verbosity: Option<i32>,
    deterministic_compute: Option<bool>,
    config_entries: Vec<(String, String)>,
    custom_ops_libraries: Vec<PathBuf>,
}

impl SessionBuilder {
    /// Sets the intra-operator thread count.
    pub fn intra_threads(mut self, threads: i32) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    /// Sets the inter-operator thread count.
    pub fn inter_threads(mut self, threads: i32) -> Self {
        self.inter_threads = Some(threads);
        self
    }

    /// Sets graph optimization behavior.
    pub fn graph_optimization(mut self, level: GraphOptimizationLevel) -> Self {
        self.optimization = level;
        self
    }

    /// Sets sequential or parallel graph execution.
    pub fn execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Enables or disables ORT's memory-pattern optimization.
    pub fn memory_pattern(mut self, enabled: bool) -> Self {
        self.memory_pattern = enabled;
        self
    }

    /// Enables session profiling with the given platform-native file prefix.
    pub fn profiling(mut self, file_prefix: impl AsRef<Path>) -> Self {
        self.profiling_prefix = Some(file_prefix.as_ref().to_owned());
        self
    }

    /// Writes the optimized model produced during session creation.
    pub fn optimized_model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.optimized_model_path = Some(path.as_ref().to_owned());
        self
    }

    /// Sets the identifier included in this session's runtime logs.
    pub fn log_id(mut self, id: impl Into<String>) -> Self {
        self.log_id = Some(id.into());
        self
    }

    /// Overrides the minimum log severity for this session.
    pub fn log_level(mut self, level: LogLevel) -> Self {
        self.log_level = Some(level);
        self
    }

    /// Sets the verbose logging level for this session.
    pub fn log_verbosity(mut self, verbosity: i32) -> Self {
        self.log_verbosity = Some(verbosity);
        self
    }

    /// Enables or disables deterministic compute where supported by ORT.
    pub fn deterministic_compute(mut self, enabled: bool) -> Self {
        self.deterministic_compute = Some(enabled);
        self
    }

    /// Adds a low-level ONNX Runtime session configuration entry.
    pub fn config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let key = key.into();
        let value = value.into();
        if let Some(existing) = self
            .config_entries
            .iter_mut()
            .find(|(existing, _)| existing == &key)
        {
            existing.1 = value;
        } else {
            self.config_entries.push((key, value));
        }
        self
    }

    /// Registers a shared library that exports ONNX Runtime custom operators.
    pub fn custom_ops_library(mut self, path: impl AsRef<Path>) -> Self {
        self.custom_ops_libraries.push(path.as_ref().to_owned());
        self
    }

    /// Configures the built-in CPU execution provider.
    pub fn cpu_provider(mut self, provider: CpuExecutionProvider) -> Self {
        self.cpu_provider = provider;
        self
    }

    /// Appends CUDA with the next available provider priority.
    ///
    /// Providers are tried in the order their builder methods are called. The
    /// built-in CPU provider remains available as the final fallback.
    pub fn cuda_provider(mut self, provider: CudaExecutionProvider) -> Self {
        self.execution_providers
            .push(ExecutionProvider::Cuda(provider));
        self
    }

    /// Appends TensorRT with the next available provider priority.
    pub fn tensorrt_provider(mut self, provider: TensorRtExecutionProvider) -> Self {
        self.execution_providers
            .push(ExecutionProvider::TensorRt(provider));
        self
    }

    /// Appends the legacy in-tree OpenVINO provider.
    pub fn openvino_provider(mut self, provider: OpenVinoExecutionProvider) -> Self {
        self.execution_providers
            .push(ExecutionProvider::OpenVino(provider));
        self
    }

    /// Appends the Intel oneDNN provider.
    pub fn dnnl_provider(mut self, provider: DnnlExecutionProvider) -> Self {
        self.execution_providers
            .push(ExecutionProvider::Dnnl(provider));
        self
    }

    /// Appends a provider supported by ONNX Runtime's generic provider API.
    pub fn named_provider(mut self, provider: NamedExecutionProvider) -> Self {
        self.execution_providers
            .push(ExecutionProvider::Named(provider));
        self
    }

    /// Builds a session directly from an ONNX model path.
    ///
    /// The native path API preserves the model directory, allowing ONNX
    /// Runtime to resolve relative external-data tensor files correctly.
    pub fn build_from_file(self, path: impl AsRef<Path>) -> Result<Session> {
        let path = path.as_ref();
        let native_path = NativeModelPath::new(path)?;
        let options = self.create_options()?;
        let api = &self.environment.inner.api;
        let mut session = ptr::null_mut();
        // SAFETY: env, native_path, and options remain live through the call;
        // session points to writable output storage.
        let status = unsafe {
            (api.table().CreateSession)(
                self.environment.inner.env as *const sys::OrtEnv,
                native_path.as_ptr(),
                options.options,
                &mut session,
            )
        };
        check_status(api, status)?;
        let session = require_non_null(session, "session")?;
        drop(options);
        Ok(Session {
            environment: self.environment,
            session: session as usize,
        })
    }

    /// Builds a session from an in-memory ONNX model.
    pub fn build_from_bytes(self, model: &[u8]) -> Result<Session> {
        let options = self.create_options()?;
        let api = &self.environment.inner.api;
        let mut session = ptr::null_mut();
        // SAFETY: model and options remain live through the call; env is valid.
        let status = unsafe {
            (api.table().CreateSessionFromArray)(
                self.environment.inner.env as *const sys::OrtEnv,
                model.as_ptr().cast(),
                model.len(),
                options.options,
                &mut session,
            )
        };
        check_status(api, status)?;
        let session = require_non_null(session, "session")?;
        drop(options);
        Ok(Session {
            environment: self.environment,
            session: session as usize,
        })
    }

    fn create_options(&self) -> Result<SessionOptionsGuard<'_>> {
        let api = &self.environment.inner.api;
        let mut options = ptr::null_mut();
        // SAFETY: output points to writable storage.
        check_status(api, unsafe {
            (api.table().CreateSessionOptions)(&mut options)
        })?;
        let options = require_non_null(options, "session options")?;
        let guard = SessionOptionsGuard { api, options };

        // SAFETY: options is a live, uniquely configured options object.
        check_status(api, unsafe {
            (api.table().SetSessionGraphOptimizationLevel)(options, self.optimization.into())
        })?;
        // SAFETY: options is live and the execution-mode enum is valid.
        check_status(api, unsafe {
            (api.table().SetSessionExecutionMode)(options, self.execution_mode.into())
        })?;
        // SAFETY: options is live and uniquely configured before session creation.
        let arena_status = unsafe {
            if self.cpu_provider.arena_enabled() {
                (api.table().EnableCpuMemArena)(options)
            } else {
                (api.table().DisableCpuMemArena)(options)
            }
        };
        check_status(api, arena_status)?;
        // SAFETY: options is live and uniquely configured before session creation.
        let memory_pattern_status = unsafe {
            if self.memory_pattern {
                (api.table().EnableMemPattern)(options)
            } else {
                (api.table().DisableMemPattern)(options)
            }
        };
        check_status(api, memory_pattern_status)?;
        if let Some(prefix) = &self.profiling_prefix {
            let prefix = NativeModelPath::new(prefix)?;
            // SAFETY: prefix uses ORT's platform path representation and stays
            // live through the synchronous configuration call.
            check_status(api, unsafe {
                (api.table().EnableProfiling)(options, prefix.as_ptr())
            })?;
        }
        if let Some(path) = &self.optimized_model_path {
            let path = NativeModelPath::new(path)?;
            // SAFETY: path is platform-native and live for the call.
            check_status(api, unsafe {
                (api.table().SetOptimizedModelFilePath)(options, path.as_ptr())
            })?;
        }
        if let Some(id) = &self.log_id {
            let id = cstring(id, "session log id")?;
            // SAFETY: id is NUL-terminated and live for the call.
            check_status(api, unsafe {
                (api.table().SetSessionLogId)(options, id.as_ptr())
            })?;
        }
        if let Some(level) = self.log_level {
            let level: sys::OrtLoggingLevel = level.into();
            // SAFETY: the logging level is a valid ORT enum value.
            check_status(api, unsafe {
                (api.table().SetSessionLogSeverityLevel)(options, level.0 as i32)
            })?;
        }
        if let Some(verbosity) = self.log_verbosity {
            // SAFETY: options is live and uniquely configured.
            check_status(api, unsafe {
                (api.table().SetSessionLogVerbosityLevel)(options, verbosity)
            })?;
        }
        if let Some(enabled) = self.deterministic_compute {
            // SAFETY: options is live and uniquely configured.
            check_status(api, unsafe {
                (api.table().SetDeterministicCompute)(options, enabled)
            })?;
        }
        for (key, value) in &self.config_entries {
            let key = cstring(key, "session configuration key")?;
            let value = cstring(value, "session configuration value")?;
            // SAFETY: both strings are NUL-terminated and live for the call.
            check_status(api, unsafe {
                (api.table().AddSessionConfigEntry)(options, key.as_ptr(), value.as_ptr())
            })?;
        }
        for path in &self.custom_ops_libraries {
            let path = NativeModelPath::new(path)?;
            // SAFETY: path uses ORT's native representation and remains live
            // through the synchronous registration call.
            check_status(api, unsafe {
                (api.table().RegisterCustomOpsLibrary_V2)(options, path.as_ptr())
            })?;
        }
        if let Some(threads) = self.intra_threads {
            // SAFETY: options remains valid and uniquely borrowed here.
            check_status(api, unsafe {
                (api.table().SetIntraOpNumThreads)(options, threads)
            })?;
        }
        if let Some(threads) = self.inter_threads {
            // SAFETY: options remains valid and uniquely borrowed here.
            check_status(api, unsafe {
                (api.table().SetInterOpNumThreads)(options, threads)
            })?;
        }

        for provider in &self.execution_providers {
            provider.append(api, options)?;
        }

        Ok(guard)
    }
}

impl ExecutionProvider {
    fn append(&self, api: &sys::Api, options: *mut sys::OrtSessionOptions) -> Result<()> {
        match self {
            Self::Cuda(provider) => append_cuda_provider(api, options, provider),
            Self::TensorRt(provider) => append_tensorrt_provider(api, options, provider),
            Self::OpenVino(provider) => append_openvino_provider(api, options, provider),
            Self::Dnnl(provider) => append_dnnl_provider(api, options, provider),
            Self::Named(provider) => append_named_provider(api, options, provider),
        }
    }
}

fn append_cuda_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &CudaExecutionProvider,
) -> Result<()> {
    let encoded = provider.options.encode("CUDA provider option")?;
    let mut options = ptr::null_mut();
    // SAFETY: output points to writable storage owned by the guard below.
    check_status(api, unsafe {
        (api.table().CreateCUDAProviderOptions)(&mut options)
    })?;
    let options = require_non_null(options, "CUDA provider options")?;
    let guard = CudaProviderOptionsGuard { api, options };
    if encoded.len() != 0 {
        // SAFETY: the key/value pointer arrays have matching lengths and all
        // C strings remain live through the synchronous call.
        check_status(api, unsafe {
            (api.table().UpdateCUDAProviderOptions)(
                options,
                encoded.key_pointers.as_ptr(),
                encoded.value_pointers.as_ptr(),
                encoded.len(),
            )
        })?;
    }
    // SAFETY: both option objects are live for the duration of the call.
    check_status(api, unsafe {
        (api.table().SessionOptionsAppendExecutionProvider_CUDA_V2)(session_options, options)
    })?;
    drop(guard);
    Ok(())
}

fn append_tensorrt_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &TensorRtExecutionProvider,
) -> Result<()> {
    let encoded = provider.options.encode("TensorRT provider option")?;
    let mut options = ptr::null_mut();
    // SAFETY: output points to writable storage owned by the guard below.
    check_status(api, unsafe {
        (api.table().CreateTensorRTProviderOptions)(&mut options)
    })?;
    let options = require_non_null(options, "TensorRT provider options")?;
    let guard = TensorRtProviderOptionsGuard { api, options };
    if encoded.len() != 0 {
        // SAFETY: the key/value pointer arrays have matching lengths and all
        // C strings remain live through the synchronous call.
        check_status(api, unsafe {
            (api.table().UpdateTensorRTProviderOptions)(
                options,
                encoded.key_pointers.as_ptr(),
                encoded.value_pointers.as_ptr(),
                encoded.len(),
            )
        })?;
    }
    // SAFETY: both option objects are live for the duration of the call.
    check_status(api, unsafe {
        (api.table()
            .SessionOptionsAppendExecutionProvider_TensorRT_V2)(session_options, options)
    })?;
    drop(guard);
    Ok(())
}

fn append_openvino_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &OpenVinoExecutionProvider,
) -> Result<()> {
    let encoded = provider.options.encode("OpenVINO provider option")?;
    // SAFETY: the key/value pointer arrays have matching lengths and all C
    // strings remain live through the synchronous call.
    check_status(api, unsafe {
        (api.table()
            .SessionOptionsAppendExecutionProvider_OpenVINO_V2)(
            session_options,
            encoded.key_pointers.as_ptr(),
            encoded.value_pointers.as_ptr(),
            encoded.len(),
        )
    })
}

fn append_dnnl_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &DnnlExecutionProvider,
) -> Result<()> {
    let encoded = provider.options.encode("oneDNN provider option")?;
    let mut options = ptr::null_mut();
    // SAFETY: output points to writable storage owned by the guard below.
    check_status(api, unsafe {
        (api.table().CreateDnnlProviderOptions)(&mut options)
    })?;
    let options = require_non_null(options, "oneDNN provider options")?;
    let guard = DnnlProviderOptionsGuard { api, options };
    if encoded.len() != 0 {
        // SAFETY: the key/value pointer arrays have matching lengths and all
        // C strings remain live through the synchronous call.
        check_status(api, unsafe {
            (api.table().UpdateDnnlProviderOptions)(
                options,
                encoded.key_pointers.as_ptr(),
                encoded.value_pointers.as_ptr(),
                encoded.len(),
            )
        })?;
    }
    // SAFETY: both option objects are live for the duration of the call.
    check_status(api, unsafe {
        (api.table().SessionOptionsAppendExecutionProvider_Dnnl)(session_options, options)
    })?;
    drop(guard);
    Ok(())
}

fn append_named_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &NamedExecutionProvider,
) -> Result<()> {
    let name = cstring(&provider.name, "execution provider")?;
    let encoded = provider.options.encode("execution provider option")?;
    // SAFETY: provider name, key/value arrays, and their C strings remain live
    // through the synchronous call.
    check_status(api, unsafe {
        (api.table().SessionOptionsAppendExecutionProvider)(
            session_options,
            name.as_ptr(),
            encoded.key_pointers.as_ptr(),
            encoded.value_pointers.as_ptr(),
            encoded.len(),
        )
    })
}

struct SessionOptionsGuard<'a> {
    api: &'a sys::Api,
    options: *mut sys::OrtSessionOptions,
}

impl Drop for SessionOptionsGuard<'_> {
    fn drop(&mut self) {
        if !self.options.is_null() {
            // SAFETY: this guard uniquely owns the options pointer.
            unsafe { (self.api.table().ReleaseSessionOptions)(self.options) };
        }
    }
}

struct CudaProviderOptionsGuard<'a> {
    api: &'a sys::Api,
    options: *mut sys::OrtCUDAProviderOptionsV2,
}

impl Drop for CudaProviderOptionsGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the options pointer.
        unsafe { (self.api.table().ReleaseCUDAProviderOptions)(self.options) };
    }
}

struct TensorRtProviderOptionsGuard<'a> {
    api: &'a sys::Api,
    options: *mut sys::OrtTensorRTProviderOptionsV2,
}

struct DnnlProviderOptionsGuard<'a> {
    api: &'a sys::Api,
    options: *mut sys::OrtDnnlProviderOptions,
}

impl Drop for DnnlProviderOptionsGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the options pointer.
        unsafe { (self.api.table().ReleaseDnnlProviderOptions)(self.options) };
    }
}

impl Drop for TensorRtProviderOptionsGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the options pointer.
        unsafe { (self.api.table().ReleaseTensorRTProviderOptions)(self.options) };
    }
}

/// Reusable options applied to an ONNX Runtime inference run.
///
/// Create this with [`Session::run_options`]. Configuration methods require
/// exclusive access, while [`Self::terminate`] may be called through shared
/// access to cooperatively cancel an active run.
pub struct RunOptions {
    environment: Environment,
    run_options: usize,
}

// SAFETY: ORT run options may be moved between threads. Configuration requires
// &mut self, while the terminate operations are specifically provided by ORT
// for cooperative cancellation of an active run.
unsafe impl Send for RunOptions {}
// SAFETY: shared access only exposes ORT's thread-safe termination operations
// and immutable configuration queries.
unsafe impl Sync for RunOptions {}

impl RunOptions {
    fn new(environment: Environment) -> Result<Self> {
        let api = &environment.inner.api;
        let mut run_options = ptr::null_mut();
        // SAFETY: output points to writable storage.
        check_status(api, unsafe {
            (api.table().CreateRunOptions)(&mut run_options)
        })?;
        let run_options = require_non_null(run_options, "run options")?;
        Ok(Self {
            environment,
            run_options: run_options as usize,
        })
    }

    /// Sets the run tag included in runtime diagnostics and profiling data.
    pub fn set_tag(&mut self, tag: &str) -> Result<&mut Self> {
        let tag = cstring(tag, "run tag")?;
        let api = &self.environment.inner.api;
        // SAFETY: the options are live and tag is NUL-terminated for the call.
        check_status(api, unsafe {
            (api.table().RunOptionsSetRunTag)(self.as_mut_ptr(), tag.as_ptr())
        })?;
        Ok(self)
    }

    /// Sets the minimum log severity for this run.
    pub fn set_log_level(&mut self, level: LogLevel) -> Result<&mut Self> {
        let level: sys::OrtLoggingLevel = level.into();
        let api = &self.environment.inner.api;
        // SAFETY: the options are live and the logging enum value is valid.
        check_status(api, unsafe {
            (api.table().RunOptionsSetRunLogSeverityLevel)(self.as_mut_ptr(), level.0 as i32)
        })?;
        Ok(self)
    }

    /// Sets the run-specific verbose logging level.
    pub fn set_log_verbosity(&mut self, verbosity: i32) -> Result<&mut Self> {
        let api = &self.environment.inner.api;
        // SAFETY: the options are live and uniquely configured through &mut self.
        check_status(api, unsafe {
            (api.table().RunOptionsSetRunLogVerbosityLevel)(self.as_mut_ptr(), verbosity)
        })?;
        Ok(self)
    }

    /// Adds a provider/runtime-specific run configuration entry.
    pub fn add_config_entry(&mut self, key: &str, value: &str) -> Result<&mut Self> {
        let key = cstring(key, "run configuration key")?;
        let value = cstring(value, "run configuration value")?;
        let api = &self.environment.inner.api;
        // SAFETY: options and both NUL-terminated strings are live for the call.
        check_status(api, unsafe {
            (api.table().AddRunConfigEntry)(self.as_mut_ptr(), key.as_ptr(), value.as_ptr())
        })?;
        Ok(self)
    }

    /// Requests cooperative termination of a run using these options.
    pub fn terminate(&self) -> Result<()> {
        let api = &self.environment.inner.api;
        // SAFETY: ORT permits termination to be requested while a run is active.
        check_status(api, unsafe {
            (api.table().RunOptionsSetTerminate)(self.as_mut_ptr())
        })
    }

    /// Clears a previous cooperative termination request.
    pub fn reset_terminate(&self) -> Result<()> {
        let api = &self.environment.inner.api;
        // SAFETY: the options are live and ORT synchronizes its terminate state.
        check_status(api, unsafe {
            (api.table().RunOptionsUnsetTerminate)(self.as_mut_ptr())
        })
    }

    fn ensure_session(&self, session: &Session) -> Result<()> {
        if Arc::ptr_eq(&self.environment.inner, &session.environment.inner) {
            Ok(())
        } else {
            Err(Error::ForeignRunOptions)
        }
    }

    fn as_ptr(&self) -> *const sys::OrtRunOptions {
        self.run_options as *const sys::OrtRunOptions
    }

    fn as_mut_ptr(&self) -> *mut sys::OrtRunOptions {
        self.run_options as *mut sys::OrtRunOptions
    }
}

impl Drop for RunOptions {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns the live options pointer.
        unsafe { (self.environment.inner.api.table().ReleaseRunOptions)(self.as_mut_ptr()) };
    }
}

/// An ONNX Runtime inference session.
pub struct Session {
    environment: Environment,
    session: usize,
}

// SAFETY: ORT sessions support concurrent Run calls and the wrapper holds no
// thread-affine or mutable Rust state.
unsafe impl Send for Session {}
// SAFETY: the native session API supports concurrent Run calls.
unsafe impl Sync for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: session was created by this environment and is released once.
        unsafe {
            (self.environment.inner.api.table().ReleaseSession)(
                self.session as *mut sys::OrtSession,
            )
        };
    }
}

impl Session {
    /// Creates reusable options associated with this session's runtime.
    pub fn run_options(&self) -> Result<RunOptions> {
        RunOptions::new(self.environment.clone())
    }

    /// Creates a reusable native I/O Binding object for this session.
    pub fn io_binding(&self) -> Result<IoBinding<'_>> {
        let api = &self.environment.inner.api;
        let mut binding = ptr::null_mut();
        // SAFETY: the session is live and binding is writable output storage.
        check_status(api, unsafe {
            (api.table().CreateIoBinding)(self.session as *mut sys::OrtSession, &mut binding)
        })?;
        Ok(IoBinding {
            session: self,
            binding: require_non_null(binding, "I/O binding")?,
        })
    }

    /// Returns metadata for all model inputs in graph order.
    pub fn inputs(&self) -> Result<Vec<TensorInfo>> {
        self.tensor_infos(IoKind::Input)
    }

    /// Returns recursive type metadata for all model inputs in graph order.
    ///
    /// Unlike [`Self::inputs`], this supports sequence, map, optional, sparse,
    /// and opaque model values in addition to dense tensors.
    pub fn input_values(&self) -> Result<Vec<ValueInfo>> {
        self.value_infos(IoKind::Input)
    }

    /// Returns metadata for all model outputs in graph order.
    pub fn outputs(&self) -> Result<Vec<TensorInfo>> {
        self.tensor_infos(IoKind::Output)
    }

    /// Returns recursive type metadata for all model outputs in graph order.
    pub fn output_values(&self) -> Result<Vec<ValueInfo>> {
        self.value_infos(IoKind::Output)
    }

    /// Returns tensor metadata for overridable model initializers.
    pub fn overridable_initializers(&self) -> Result<Vec<TensorInfo>> {
        self.tensor_infos(IoKind::Initializer)
    }

    /// Returns recursive type metadata for overridable model initializers.
    pub fn overridable_initializer_values(&self) -> Result<Vec<ValueInfo>> {
        self.value_infos(IoKind::Initializer)
    }

    /// Returns the immutable metadata embedded in the model.
    pub fn metadata(&self) -> Result<ModelMetadata> {
        let api = &self.environment.inner.api;
        let mut metadata = ptr::null_mut();
        // SAFETY: the session is live and metadata is writable output storage.
        check_status(api, unsafe {
            (api.table().SessionGetModelMetadata)(
                self.session as *const sys::OrtSession,
                &mut metadata,
            )
        })?;
        Ok(ModelMetadata {
            environment: self.environment.clone(),
            metadata: require_non_null(metadata, "model metadata")? as usize,
        })
    }

    /// Returns the profiling start timestamp in nanoseconds.
    pub fn profiling_start_time_ns(&self) -> Result<u64> {
        let api = &self.environment.inner.api;
        let mut timestamp = 0;
        // SAFETY: the session is live and timestamp is writable output storage.
        check_status(api, unsafe {
            (api.table().SessionGetProfilingStartTimeNs)(
                self.session as *const sys::OrtSession,
                &mut timestamp,
            )
        })?;
        Ok(timestamp)
    }

    /// Ends profiling and returns the generated profile file path.
    ///
    /// Profiling must have been enabled through [`SessionBuilder::profiling`].
    pub fn end_profiling(&mut self) -> Result<PathBuf> {
        let api = &self.environment.inner.api;
        let allocator = default_allocator(api)?;
        let mut path = ptr::null_mut();
        // SAFETY: the session is exclusively borrowed, allocator is live, and
        // path is writable output storage.
        check_status(api, unsafe {
            (api.table().SessionEndProfiling)(
                self.session as *mut sys::OrtSession,
                allocator,
                &mut path,
            )
        })?;
        let path = require_non_null(path, "profile output path")?;
        let guard = AllocatedBufferGuard {
            api,
            allocator,
            allocation: path.cast(),
        };
        // SAFETY: ORT returned an allocator-owned NUL-terminated UTF-8 path.
        let path = PathBuf::from(
            unsafe { CStr::from_ptr(path) }
                .to_string_lossy()
                .into_owned(),
        );
        drop(guard);
        Ok(path)
    }

    /// Runs the model and requests every declared output in graph order.
    pub fn run_all(&self, inputs: &[Input<'_>]) -> Result<Vec<Output>> {
        let infos = self.outputs()?;
        let names = infos.iter().map(TensorInfo::name).collect::<Vec<_>>();
        let tensors = self.run(inputs, &names)?;
        Ok(infos
            .into_iter()
            .zip(tensors)
            .map(|(info, tensor)| Output {
                name: info.name,
                tensor,
            })
            .collect())
    }

    /// Runs every declared output using explicit reusable run options.
    pub fn run_all_with_options(
        &self,
        inputs: &[Input<'_>],
        options: &RunOptions,
    ) -> Result<Vec<Output>> {
        let infos = self.outputs()?;
        let names = infos.iter().map(TensorInfo::name).collect::<Vec<_>>();
        let tensors = self.run_with_options(inputs, &names, options)?;
        Ok(infos
            .into_iter()
            .zip(tensors)
            .map(|(info, tensor)| Output {
                name: info.name,
                tensor,
            })
            .collect())
    }

    /// Submits inference to ONNX Runtime's intra-op thread pool and returns a
    /// Future completed by the native `RunAsync` callback.
    ///
    /// Dropping an unfinished future waits for the callback before releasing
    /// borrowed input storage. This makes cancellation memory-safe, although
    /// dropping can therefore block.
    pub fn run_all_async<'a>(&'a self, inputs: &'a [Input<'a>]) -> Result<RunFuture<'a>> {
        let options = self.run_options()?;
        self.run_all_async_with_options(inputs, options)
    }

    /// Submits inference with explicitly configured run options.
    ///
    /// The returned future owns the options so cancellation remains available
    /// for the complete native callback lifetime.
    pub fn run_all_async_with_options<'a>(
        &'a self,
        inputs: &'a [Input<'a>],
        options: RunOptions,
    ) -> Result<RunFuture<'a>> {
        options.ensure_session(self)?;
        let api = &self.environment.inner.api;
        let infos = self.outputs()?;
        let output_names = infos
            .iter()
            .map(|info| cstring(info.name(), "output"))
            .collect::<Result<Vec<_>>>()?;
        let names = infos.into_iter().map(|info| info.name).collect::<Vec<_>>();

        let mut memory_info = ptr::null_mut();
        // SAFETY: output storage is writable and enum values are valid.
        check_status(api, unsafe {
            (api.table().CreateCpuMemoryInfo)(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        })?;
        let memory_info = require_non_null(memory_info, "CPU memory info")?;
        let memory_guard = MemoryInfoGuard { api, memory_info };

        let mut input_values = Vec::with_capacity(inputs.len());
        for input in inputs {
            let mut value = ptr::null_mut();
            let shape = input.tensor.shape();
            // SAFETY: Input holds validated compact CPU storage and the Future
            // retains the borrow until the native callback finishes.
            let bytes = unsafe { input.tensor.cpu_bytes() }
                .map_err(|source| Error::DlpackTensor { source })?;
            // SAFETY: data, shape, and memory info remain live until callback.
            check_status(api, unsafe {
                (api.table().CreateTensorWithDataAsOrtValue)(
                    memory_info,
                    bytes.as_ptr().cast_mut().cast(),
                    bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    dlpack_to_onnx(input.tensor.dtype())?,
                    &mut value,
                )
            })?;
            input_values.push(ValueGuard::new(
                api,
                require_non_null(value, "async input value")?,
            ));
        }

        let input_names = inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<Vec<_>>();
        let input_name_ptrs = input_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let input_value_ptrs = input_values
            .iter()
            .map(|value| value.ptr.cast_const())
            .collect::<Vec<_>>();
        let output_name_ptrs = output_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let output_slots = vec![ptr::null_mut(); names.len()];
        let completion = Arc::new(AsyncCompletion::new(api.clone(), names.len()));
        let mut future = RunFuture {
            session: self,
            environment: self.environment.clone(),
            completion,
            names,
            _memory_info: memory_guard,
            _input_values: input_values,
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
        // SAFETY: every pointer array is backed by heap allocations retained in
        // the Future. callback_state is one strong Arc reference reclaimed by
        // async_run_callback. Drop waits if the Future is cancelled.
        let status = unsafe {
            (api.table().RunAsync)(
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
        if !status.0.is_null() {
            // SAFETY: submission failed, so ORT did not take ownership of the
            // callback reference and will not invoke the callback.
            unsafe { drop(Arc::from_raw(callback_state)) };
            // Prevent Drop from waiting for a callback that will not occur.
            future.finished = true;
            check_status(api, status)?;
        }
        Ok(future)
    }

    /// Executes heterogeneous tensor inputs and returns typed runtime values.
    ///
    /// Numeric tensors use zero-copy DLPack input storage. String tensors are
    /// copied into ORT-owned storage and string outputs are copied back to Rust.
    pub fn run_values(
        &self,
        inputs: &[ValueInput<'_>],
        output_names: &[&str],
    ) -> Result<Vec<ValueOutput>> {
        self.run_values_impl(inputs, output_names, None)
    }

    /// Executes heterogeneous values with explicit reusable run options.
    pub fn run_values_with_options(
        &self,
        inputs: &[ValueInput<'_>],
        output_names: &[&str],
        options: &RunOptions,
    ) -> Result<Vec<ValueOutput>> {
        options.ensure_session(self)?;
        self.run_values_impl(inputs, output_names, Some(options))
    }

    fn run_values_impl(
        &self,
        inputs: &[ValueInput<'_>],
        output_names: &[&str],
        options: Option<&RunOptions>,
    ) -> Result<Vec<ValueOutput>> {
        let api = &self.environment.inner.api;
        let mut memory_info = ptr::null_mut();
        // SAFETY: output storage is writable and enum values are valid.
        check_status(api, unsafe {
            (api.table().CreateCpuMemoryInfo)(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        })?;
        let memory_info = require_non_null(memory_info, "CPU memory info")?;
        let _memory_guard = MemoryInfoGuard { api, memory_info };
        let allocator = default_allocator(api)?;

        let mut input_values = Vec::with_capacity(inputs.len());
        for input in inputs {
            let value = match input {
                ValueInput::Tensor(input) => {
                    create_external_tensor(api, memory_info, &input.tensor)?
                }
                ValueInput::Strings(input) => create_string_tensor(api, allocator, input)?,
                ValueInput::Sequence(input) => {
                    create_sequence_input(api, memory_info, allocator, &input.values)?
                }
                ValueInput::Map(input) => {
                    create_map_input(api, memory_info, allocator, &input.keys, &input.values)?
                }
            };
            input_values.push(ValueGuard::new(api, value));
        }

        let output_cstrings = output_names
            .iter()
            .map(|name| cstring(name, "output"))
            .collect::<Result<Vec<_>>>()?;
        let input_name_ptrs = inputs
            .iter()
            .map(|input| input.name().as_ptr())
            .collect::<Vec<_>>();
        let input_value_ptrs = input_values
            .iter()
            .map(|value| value.ptr.cast_const())
            .collect::<Vec<_>>();
        let output_name_ptrs = output_cstrings
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut output_values = vec![ptr::null_mut(); output_names.len()];

        // SAFETY: all pointer arrays have matching lengths and all values stay
        // live for the synchronous native run.
        let status = unsafe {
            (api.table().Run)(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                input_value_ptrs.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                output_values.as_mut_ptr(),
            )
        };
        if let Err(error) = check_status(api, status) {
            for value in output_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: ORT initialized this caller-owned output.
                unsafe { (api.table().ReleaseValue)(value) };
            }
            return Err(error);
        }
        if output_values.iter().any(|value| value.is_null()) {
            for value in output_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: every non-null output is owned by this call.
                unsafe { (api.table().ReleaseValue)(value) };
            }
            return Err(Error::NullPointer {
                what: "output value",
            });
        }

        output_names
            .iter()
            .zip(output_values)
            .map(|(name, value)| {
                Ok(ValueOutput {
                    name: (*name).to_owned(),
                    value: output_to_value(self.environment.clone(), value)?,
                })
            })
            .collect()
    }

    /// Executes inference using named DLPack inputs and returns versioned DLPack outputs.
    ///
    /// CPU input storage is shared with ONNX Runtime without copying and is
    /// borrowed for the duration of this call. Output storage remains owned by
    /// ONNX Runtime until each returned DLPack handle is dropped.
    pub fn run(
        &self,
        inputs: &[Input<'_>],
        output_names: &[&str],
    ) -> Result<Vec<versioned::Dlpack>> {
        self.run_impl(inputs, output_names, None)
    }

    /// Executes selected outputs with explicit reusable run options.
    pub fn run_with_options(
        &self,
        inputs: &[Input<'_>],
        output_names: &[&str],
        options: &RunOptions,
    ) -> Result<Vec<versioned::Dlpack>> {
        options.ensure_session(self)?;
        self.run_impl(inputs, output_names, Some(options))
    }

    fn run_impl(
        &self,
        inputs: &[Input<'_>],
        output_names: &[&str],
        options: Option<&RunOptions>,
    ) -> Result<Vec<versioned::Dlpack>> {
        let api = &self.environment.inner.api;
        let mut memory_info = ptr::null_mut();
        // SAFETY: output storage is writable and enum values are valid.
        check_status(api, unsafe {
            (api.table().CreateCpuMemoryInfo)(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        })?;
        let memory_info = require_non_null(memory_info, "CPU memory info")?;
        let memory_guard = MemoryInfoGuard { api, memory_info };

        let mut input_values = Vec::with_capacity(inputs.len());
        for input in inputs {
            let mut value = ptr::null_mut();
            let shape = input.tensor.shape();
            // SAFETY: Input owns a validated compact CPU descriptor for this
            // lifetime. The returned slice applies DLPack byte_offset and uses
            // the packed whole-tensor byte count for sub-byte dtypes.
            let bytes = unsafe { input.tensor.cpu_bytes() }
                .map_err(|source| Error::DlpackTensor { source })?;
            // SAFETY: Input validation established compact CPU storage; its
            // data and shape remain live for the duration of this Run call.
            // ORT treats Run inputs as const despite the legacy creation API's
            // mutable pointer parameter.
            check_status(api, unsafe {
                (api.table().CreateTensorWithDataAsOrtValue)(
                    memory_info,
                    bytes.as_ptr().cast_mut().cast(),
                    bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    dlpack_to_onnx(input.tensor.dtype())?,
                    &mut value,
                )
            })?;
            input_values.push(ValueGuard::new(
                api,
                require_non_null(value, "input value")?,
            ));
        }

        let output_cstrings = output_names
            .iter()
            .map(|name| cstring(name, "output"))
            .collect::<Result<Vec<_>>>()?;
        let input_name_ptrs = inputs
            .iter()
            .map(|input| input.name.as_ptr())
            .collect::<Vec<_>>();
        let input_value_ptrs = input_values
            .iter()
            .map(|value| value.ptr.cast_const())
            .collect::<Vec<_>>();
        let output_name_ptrs = output_cstrings
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut output_values = vec![ptr::null_mut(); output_names.len()];

        // SAFETY: all name/value pointer arrays have matching lengths and stay
        // live through the synchronous call; output storage is writable.
        let status = unsafe {
            (api.table().Run)(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                input_value_ptrs.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                output_values.as_mut_ptr(),
            )
        };
        if let Err(error) = check_status(api, status) {
            for value in output_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: ORT initialized this output and ownership was not transferred.
                unsafe { (api.table().ReleaseValue)(value) };
            }
            return Err(error);
        }

        drop(memory_guard);
        if output_values.iter().any(|value| value.is_null()) {
            for value in output_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: every non-null entry is an owned ORT output.
                unsafe { (api.table().ReleaseValue)(value) };
            }
            return Err(Error::NullPointer {
                what: "output value",
            });
        }

        // Guard every raw output before conversion. If one conversion fails,
        // the IntoIter drops all remaining guards and previously converted
        // DLPack handles release themselves normally.
        let output_guards = output_values
            .into_iter()
            .map(|value| ValueGuard::new(api, value))
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(output_guards.len());
        for mut guard in output_guards {
            let value = guard.ptr;
            guard.disarm();
            outputs.push(output_to_dlpack(self.environment.clone(), value)?);
        }
        Ok(outputs)
    }

    /// Runs inference directly into caller-owned writable DLPack tensors.
    ///
    /// Output storage is never copied. Each output buffer is exclusively
    /// borrowed for the duration of the binding and must have the exact shape
    /// and element type required by the model.
    pub fn run_into(&self, inputs: &[Input<'_>], outputs: &mut [OutputBuffer<'_>]) -> Result<()> {
        self.run_into_impl(inputs, outputs, None)
    }

    /// Runs into caller-owned buffers with explicit reusable run options.
    pub fn run_into_with_options(
        &self,
        inputs: &[Input<'_>],
        outputs: &mut [OutputBuffer<'_>],
        options: &RunOptions,
    ) -> Result<()> {
        options.ensure_session(self)?;
        self.run_into_impl(inputs, outputs, Some(options))
    }

    fn run_into_impl(
        &self,
        inputs: &[Input<'_>],
        outputs: &mut [OutputBuffer<'_>],
        options: Option<&RunOptions>,
    ) -> Result<()> {
        let api = &self.environment.inner.api;
        let mut memory_info = ptr::null_mut();
        // SAFETY: output storage is writable and enum values are valid.
        check_status(api, unsafe {
            (api.table().CreateCpuMemoryInfo)(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        })?;
        let memory_info = require_non_null(memory_info, "CPU memory info")?;
        let _memory_guard = MemoryInfoGuard { api, memory_info };

        let mut input_values = Vec::with_capacity(inputs.len());
        for input in inputs {
            let mut value = ptr::null_mut();
            let shape = input.tensor.shape();
            // SAFETY: the validated CPU descriptor and adjusted byte view stay
            // live through the synchronous Run call.
            let bytes = unsafe { input.tensor.cpu_bytes() }
                .map_err(|source| Error::DlpackTensor { source })?;
            // SAFETY: ORT treats input tensors as const during Run.
            check_status(api, unsafe {
                (api.table().CreateTensorWithDataAsOrtValue)(
                    memory_info,
                    bytes.as_ptr().cast_mut().cast(),
                    bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    dlpack_to_onnx(input.tensor.dtype())?,
                    &mut value,
                )
            })?;
            input_values.push(ValueGuard::new(
                api,
                require_non_null(value, "input value")?,
            ));
        }

        let mut output_values = Vec::with_capacity(outputs.len());
        for output in outputs.iter_mut() {
            let mut value = ptr::null_mut();
            let shape = output.tensor.shape().to_vec();
            let dtype = output.tensor.dtype();
            // SAFETY: OutputBuffer holds exclusive access to validated compact
            // CPU storage, including the adjusted byte_offset.
            let bytes = unsafe { output.tensor.cpu_bytes_mut() }
                .map_err(|source| Error::DlpackTensor { source })?;
            // SAFETY: the mutable byte storage, shape, and memory info remain
            // live until Run completes; the OrtValue does not own the bytes.
            check_status(api, unsafe {
                (api.table().CreateTensorWithDataAsOrtValue)(
                    memory_info,
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    dlpack_to_onnx(dtype)?,
                    &mut value,
                )
            })?;
            output_values.push(ValueGuard::new(
                api,
                require_non_null(value, "preallocated output value")?,
            ));
        }

        let input_name_ptrs = inputs
            .iter()
            .map(|input| input.name.as_ptr())
            .collect::<Vec<_>>();
        let input_value_ptrs = input_values
            .iter()
            .map(|value| value.ptr.cast_const())
            .collect::<Vec<_>>();
        let output_name_ptrs = outputs
            .iter()
            .map(|output| output.name.as_ptr())
            .collect::<Vec<_>>();
        let mut returned_outputs = output_values
            .iter()
            .map(|value| value.ptr)
            .collect::<Vec<_>>();

        // SAFETY: pointer arrays and all external data buffers remain live and
        // have matching lengths through this synchronous call.
        let status = unsafe {
            (api.table().Run)(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                input_value_ptrs.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                returned_outputs.as_mut_ptr(),
            )
        };

        let mut replaced = None;
        for ((returned, original), output) in returned_outputs
            .into_iter()
            .zip(output_values.iter())
            .zip(outputs.iter())
        {
            if returned != original.ptr {
                if !returned.is_null() {
                    // SAFETY: a different non-null pointer was newly returned
                    // by ORT and is not owned by the original guard.
                    unsafe { (api.table().ReleaseValue)(returned) };
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

    /// Runs inference with preallocated CPU or CUDA DLPack buffers through
    /// ONNX Runtime I/O Binding.
    ///
    /// No tensor data is copied by this binding. CUDA pointers are passed to
    /// ONNX Runtime as device pointers and are never dereferenced by Rust.
    /// Inputs and outputs are synchronized through the provider-specific ORT
    /// synchronization APIs around the run. Every output must have the exact
    /// shape and element type required by the model.
    pub fn run_with_iobinding(
        &self,
        inputs: &[IoBindingInput<'_>],
        outputs: &mut [IoBindingOutput<'_>],
    ) -> Result<()> {
        self.run_with_iobinding_impl(inputs, outputs, None)
    }

    /// Runs preallocated I/O Binding buffers with explicit run options.
    pub fn run_with_iobinding_options(
        &self,
        inputs: &[IoBindingInput<'_>],
        outputs: &mut [IoBindingOutput<'_>],
        options: &RunOptions,
    ) -> Result<()> {
        options.ensure_session(self)?;
        self.run_with_iobinding_impl(inputs, outputs, Some(options))
    }

    fn run_with_iobinding_impl(
        &self,
        inputs: &[IoBindingInput<'_>],
        outputs: &mut [IoBindingOutput<'_>],
        options: Option<&RunOptions>,
    ) -> Result<()> {
        let api = &self.environment.inner.api;
        let mut binding = ptr::null_mut();
        // SAFETY: the session is live and binding is writable output storage.
        check_status(api, unsafe {
            (api.table().CreateIoBinding)(self.session as *mut sys::OrtSession, &mut binding)
        })?;
        let binding = require_non_null(binding, "I/O binding")?;
        let binding_guard = IoBindingGuard { api, binding };

        let mut memory_infos = Vec::with_capacity(inputs.len() + outputs.len());
        let mut values = Vec::with_capacity(inputs.len() + outputs.len());

        for input in inputs {
            let memory_info = create_dlpack_memory_info(api, input.tensor.device())?;
            let memory_guard = MemoryInfoGuard { api, memory_info };
            let value = create_external_tensor(api, memory_info, &input.tensor)?;
            let value_guard = ValueGuard::new(api, value);
            // SAFETY: binding, name, and value are live. The input value and
            // its external DLPack allocation remain live through the run.
            check_status(api, unsafe {
                (api.table().BindInput)(binding, input.name.as_ptr(), value.cast_const())
            })?;
            memory_infos.push(memory_guard);
            values.push(value_guard);
        }

        for output in outputs.iter_mut() {
            let memory_info = create_dlpack_memory_info(api, output.tensor.device())?;
            let memory_guard = MemoryInfoGuard { api, memory_info };
            let value = create_external_tensor(api, memory_info, &output.tensor)?;
            let value_guard = ValueGuard::new(api, value);
            // SAFETY: binding, name, and value are live. IoBindingOutput keeps
            // exclusive access to the external allocation through the run.
            check_status(api, unsafe {
                (api.table().BindOutput)(binding, output.name.as_ptr(), value.cast_const())
            })?;
            memory_infos.push(memory_guard);
            values.push(value_guard);
        }

        // SAFETY: binding owns valid bound values and the provider may use this
        // call to synchronize a producer stream before inference.
        check_status(api, unsafe {
            (api.table().SynchronizeBoundInputs)(binding)
        })?;
        // SAFETY: the session, binding, OrtValues, and all external buffers
        // remain live for the synchronous run.
        check_status(api, unsafe {
            (api.table().RunWithBinding)(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                binding,
            )
        })?;
        // SAFETY: binding remains live and this blocks until provider writes to
        // the bound output buffers have completed.
        check_status(api, unsafe {
            (api.table().SynchronizeBoundOutputs)(binding)
        })?;

        // Release the binding while every OrtValue and MemoryInfo it references
        // is still alive.
        drop(binding_guard);
        drop(values);
        drop(memory_infos);
        Ok(())
    }

    /// Runs with pre-bound inputs while letting ORT allocate outputs on the
    /// requested CPU or CUDA devices.
    ///
    /// This supports data-dependent and otherwise dynamic output shapes. Each
    /// returned DLPack handle owns the corresponding ORT allocation and keeps
    /// the runtime loaded until the handle is dropped.
    pub fn run_with_iobinding_outputs(
        &self,
        inputs: &[IoBindingInput<'_>],
        outputs: &[IoBindingDeviceOutput],
    ) -> Result<Vec<Output>> {
        self.run_with_iobinding_outputs_impl(inputs, outputs, None)
    }

    /// Runs ORT-allocated bound outputs with explicit run options.
    pub fn run_with_iobinding_outputs_options(
        &self,
        inputs: &[IoBindingInput<'_>],
        outputs: &[IoBindingDeviceOutput],
        options: &RunOptions,
    ) -> Result<Vec<Output>> {
        options.ensure_session(self)?;
        self.run_with_iobinding_outputs_impl(inputs, outputs, Some(options))
    }

    fn run_with_iobinding_outputs_impl(
        &self,
        inputs: &[IoBindingInput<'_>],
        outputs: &[IoBindingDeviceOutput],
        options: Option<&RunOptions>,
    ) -> Result<Vec<Output>> {
        let api = &self.environment.inner.api;
        let mut binding = ptr::null_mut();
        // SAFETY: the session is live and binding is writable output storage.
        check_status(api, unsafe {
            (api.table().CreateIoBinding)(self.session as *mut sys::OrtSession, &mut binding)
        })?;
        let binding = require_non_null(binding, "dynamic-output I/O binding")?;
        let binding_guard = IoBindingGuard { api, binding };
        let mut memory_infos = Vec::with_capacity(inputs.len() + outputs.len());
        let mut input_values = Vec::with_capacity(inputs.len());

        for input in inputs {
            let memory_info = create_dlpack_memory_info(api, input.tensor.device())?;
            let memory_guard = MemoryInfoGuard { api, memory_info };
            let value = create_external_tensor(api, memory_info, &input.tensor)?;
            let value_guard = ValueGuard::new(api, value);
            // SAFETY: all input objects remain live through the bound run.
            check_status(api, unsafe {
                (api.table().BindInput)(binding, input.name.as_ptr(), value.cast_const())
            })?;
            memory_infos.push(memory_guard);
            input_values.push(value_guard);
        }
        for output in outputs {
            let memory_info = create_dlpack_memory_info(api, output.device)?;
            let memory_guard = MemoryInfoGuard { api, memory_info };
            // SAFETY: binding, name, and memory info are live. ORT allocates an
            // output with the runtime shape on the requested device.
            check_status(api, unsafe {
                (api.table().BindOutputToDevice)(binding, output.name.as_ptr(), memory_info)
            })?;
            memory_infos.push(memory_guard);
        }

        // SAFETY: provider-specific synchronization makes bound inputs visible.
        check_status(api, unsafe {
            (api.table().SynchronizeBoundInputs)(binding)
        })?;
        // SAFETY: all binding resources remain live through the run.
        check_status(api, unsafe {
            (api.table().RunWithBinding)(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                binding,
            )
        })?;
        // SAFETY: wait until device output allocations are ready for consumers.
        check_status(api, unsafe {
            (api.table().SynchronizeBoundOutputs)(binding)
        })?;

        let mut allocator = ptr::null_mut();
        // SAFETY: allocator is writable output storage; the default allocator
        // has process lifetime and is not released by callers.
        check_status(api, unsafe {
            (api.table().GetAllocatorWithDefaultOptions)(&mut allocator)
        })?;
        let allocator = require_non_null(allocator, "default allocator")?;
        let mut raw_values = ptr::null_mut();
        let mut value_count = 0;
        // SAFETY: all output pointers are writable and binding is live.
        check_status(api, unsafe {
            (api.table().GetBoundOutputValues)(
                binding,
                allocator,
                &mut raw_values,
                &mut value_count,
            )
        })?;
        let allocation_guard = AllocatedBufferGuard {
            api,
            allocator,
            allocation: raw_values.cast(),
        };
        let raw_values = if value_count == 0 {
            Vec::new()
        } else {
            let raw_values = require_non_null(raw_values, "bound output value array")?;
            // SAFETY: ORT returned value_count entries allocated as one array.
            unsafe { std::slice::from_raw_parts(raw_values, value_count) }.to_vec()
        };
        // ORT may return a non-null zero-length allocation; free it too.
        drop(allocation_guard);
        if value_count != outputs.len() {
            for value in raw_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: each returned output is owned by the caller.
                unsafe { (api.table().ReleaseValue)(value) };
            }
            return Err(Error::BoundOutputCount {
                expected: outputs.len(),
                actual: value_count,
            });
        }
        if raw_values.iter().any(|value| value.is_null()) {
            for value in raw_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: each non-null returned output is caller-owned.
                unsafe { (api.table().ReleaseValue)(value) };
            }
            return Err(Error::NullPointer {
                what: "bound output value",
            });
        }

        // Release binding references before transferring output ownership into
        // DLPack handles.
        drop(binding_guard);
        drop(input_values);
        drop(memory_infos);

        let guards = raw_values
            .into_iter()
            .map(|value| ValueGuard::new(api, value))
            .collect::<Vec<_>>();
        let mut result = Vec::with_capacity(guards.len());
        for (output, mut guard) in outputs.iter().zip(guards) {
            let value = guard.ptr;
            guard.disarm();
            result.push(Output {
                name: output.name.to_string_lossy().into_owned(),
                tensor: output_to_dlpack(self.environment.clone(), value)?,
            });
        }
        Ok(result)
    }

    /// Submits a preallocated CUDA I/O Binding run without synchronizing
    /// execution providers at the end of `RunWithBinding`.
    ///
    /// The returned guard retains all DLPack borrows and native objects until
    /// [`PendingIoBindingRun::wait`] or Drop synchronizes the bound outputs.
    /// CPU fallback, cross-device copies, or provider behavior may still make
    /// submission block internally.
    pub fn submit_with_iobinding<'a>(
        &'a self,
        inputs: &'a [IoBindingInput<'_>],
        outputs: &'a mut [IoBindingOutput<'_>],
    ) -> Result<PendingIoBindingRun<'a>> {
        let options = self.run_options()?;
        self.submit_with_iobinding_options(inputs, outputs, options)
    }

    /// Submits preallocated I/O Binding buffers with caller-configured options
    /// and defers provider output synchronization to the returned guard.
    pub fn submit_with_iobinding_options<'a>(
        &'a self,
        inputs: &'a [IoBindingInput<'_>],
        outputs: &'a mut [IoBindingOutput<'_>],
        mut options: RunOptions,
    ) -> Result<PendingIoBindingRun<'a>> {
        options.ensure_session(self)?;
        let api = &self.environment.inner.api;
        let mut binding = ptr::null_mut();
        // SAFETY: the session is live and binding is writable output storage.
        check_status(api, unsafe {
            (api.table().CreateIoBinding)(self.session as *mut sys::OrtSession, &mut binding)
        })?;
        let binding = require_non_null(binding, "async I/O binding")?;
        let binding_guard = IoBindingGuard { api, binding };
        let mut memory_infos = Vec::with_capacity(inputs.len() + outputs.len());
        let mut values = Vec::with_capacity(inputs.len() + outputs.len());

        for input in inputs {
            let memory_info = create_dlpack_memory_info(api, input.tensor.device())?;
            let memory_guard = MemoryInfoGuard { api, memory_info };
            let value = create_external_tensor(api, memory_info, &input.tensor)?;
            let value_guard = ValueGuard::new(api, value);
            // SAFETY: all bound objects and external storage are retained by
            // the pending guard until synchronization.
            check_status(api, unsafe {
                (api.table().BindInput)(binding, input.name.as_ptr(), value.cast_const())
            })?;
            memory_infos.push(memory_guard);
            values.push(value_guard);
        }
        for output in outputs.iter_mut() {
            let memory_info = create_dlpack_memory_info(api, output.tensor.device())?;
            let memory_guard = MemoryInfoGuard { api, memory_info };
            let value = create_external_tensor(api, memory_info, &output.tensor)?;
            let value_guard = ValueGuard::new(api, value);
            // SAFETY: the pending guard retains exclusive output storage and
            // every native object until provider synchronization completes.
            check_status(api, unsafe {
                (api.table().BindOutput)(binding, output.name.as_ptr(), value.cast_const())
            })?;
            memory_infos.push(memory_guard);
            values.push(value_guard);
        }

        // SAFETY: binding is live; provider-specific synchronization makes
        // producer work visible before inference starts.
        check_status(api, unsafe {
            (api.table().SynchronizeBoundInputs)(binding)
        })?;

        options.add_config_entry("disable_synchronize_execution_providers", "1")?;
        // SAFETY: all bindings and external buffers remain live in the guard
        // returned below. This call submits work without the final EP sync.
        check_status(api, unsafe {
            (api.table().RunWithBinding)(
                self.session as *mut sys::OrtSession,
                options.as_ptr(),
                binding,
            )
        })?;

        Ok(PendingIoBindingRun {
            session: self,
            binding: Some(binding_guard),
            _values: values,
            _memory_infos: memory_infos,
            _borrows: PhantomData,
            synchronized: false,
        })
    }

    fn tensor_infos(&self, kind: IoKind) -> Result<Vec<TensorInfo>> {
        self.value_infos(kind)?
            .into_iter()
            .map(|info| match info.value_type {
                ValueTypeInfo::Tensor(tensor) => Ok(TensorInfo {
                    name: info.name,
                    element_type: tensor.element_type,
                    dimensions: tensor.dimensions,
                }),
                other => Err(Error::UnsupportedValueType {
                    value_type: value_type_kind(&other),
                }),
            })
            .collect()
    }

    fn value_infos(&self, kind: IoKind) -> Result<Vec<ValueInfo>> {
        let api = &self.environment.inner.api;
        let session = self.session as *const sys::OrtSession;
        let mut count = 0;
        // SAFETY: session is live and count is writable output storage.
        let status = unsafe {
            match kind {
                IoKind::Input => (api.table().SessionGetInputCount)(session, &mut count),
                IoKind::Output => (api.table().SessionGetOutputCount)(session, &mut count),
                IoKind::Initializer => {
                    (api.table().SessionGetOverridableInitializerCount)(session, &mut count)
                }
            }
        };
        check_status(api, status)?;

        let mut allocator = ptr::null_mut();
        // SAFETY: allocator is writable output storage; the returned default
        // allocator has process lifetime and must not be released by callers.
        check_status(api, unsafe {
            (api.table().GetAllocatorWithDefaultOptions)(&mut allocator)
        })?;
        let allocator = require_non_null(allocator, "default allocator")?;

        (0..count)
            .map(|index| self.value_info(kind, index, allocator))
            .collect()
    }

    fn value_info(
        &self,
        kind: IoKind,
        index: usize,
        allocator: *mut sys::OrtAllocator,
    ) -> Result<ValueInfo> {
        let api = &self.environment.inner.api;
        let session = self.session as *const sys::OrtSession;
        let mut raw_name = ptr::null_mut();
        // SAFETY: session and allocator are live; raw_name is writable output storage.
        let status = unsafe {
            match kind {
                IoKind::Input => {
                    (api.table().SessionGetInputName)(session, index, allocator, &mut raw_name)
                }
                IoKind::Output => {
                    (api.table().SessionGetOutputName)(session, index, allocator, &mut raw_name)
                }
                IoKind::Initializer => (api.table().SessionGetOverridableInitializerName)(
                    session,
                    index,
                    allocator,
                    &mut raw_name,
                ),
            }
        };
        check_status(api, status)?;
        let raw_name = require_non_null(raw_name, "model value name")?;
        // SAFETY: ORT returns a NUL-terminated allocator-owned name.
        let name = unsafe { CStr::from_ptr(raw_name) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: raw_name was allocated by this allocator and is freed once.
        check_status(api, unsafe {
            (api.table().AllocatorFree)(allocator, raw_name.cast())
        })?;

        let mut type_info = ptr::null_mut();
        // SAFETY: session is live and type_info is writable output storage.
        let status = unsafe {
            match kind {
                IoKind::Input => {
                    (api.table().SessionGetInputTypeInfo)(session, index, &mut type_info)
                }
                IoKind::Output => {
                    (api.table().SessionGetOutputTypeInfo)(session, index, &mut type_info)
                }
                IoKind::Initializer => (api.table().SessionGetOverridableInitializerTypeInfo)(
                    session,
                    index,
                    &mut type_info,
                ),
            }
        };
        check_status(api, status)?;
        let type_info = require_non_null(type_info, "model value type info")?;
        let _type_guard = TypeInfoGuard { api, type_info };
        let value_type = read_value_type(api, type_info)?;

        Ok(ValueInfo { name, value_type })
    }
}

fn value_type_kind(value_type: &ValueTypeInfo) -> sys::ONNXType {
    match value_type {
        ValueTypeInfo::Tensor(_) => sys::ONNXType::ONNX_TYPE_TENSOR,
        ValueTypeInfo::Sequence(_) => sys::ONNXType::ONNX_TYPE_SEQUENCE,
        ValueTypeInfo::Map { .. } => sys::ONNXType::ONNX_TYPE_MAP,
        ValueTypeInfo::Optional(_) => sys::ONNXType::ONNX_TYPE_OPTIONAL,
        ValueTypeInfo::SparseTensor => sys::ONNXType::ONNX_TYPE_SPARSETENSOR,
        ValueTypeInfo::Opaque => sys::ONNXType::ONNX_TYPE_OPAQUE,
        ValueTypeInfo::Unknown => sys::ONNXType::ONNX_TYPE_UNKNOWN,
    }
}

fn read_value_type(api: &sys::Api, type_info: *mut sys::OrtTypeInfo) -> Result<ValueTypeInfo> {
    let mut value_type = sys::ONNXType::ONNX_TYPE_UNKNOWN;
    // SAFETY: type_info is live and value_type is writable.
    check_status(api, unsafe {
        (api.table().GetOnnxTypeFromTypeInfo)(type_info, &mut value_type)
    })?;
    match value_type {
        sys::ONNXType::ONNX_TYPE_TENSOR => {
            read_tensor_type(api, type_info).map(ValueTypeInfo::Tensor)
        }
        sys::ONNXType::ONNX_TYPE_SEQUENCE => {
            let mut sequence_info = ptr::null();
            // SAFETY: type_info is a live sequence description and owns the
            // borrowed sequence-info view.
            check_status(api, unsafe {
                (api.table().CastTypeInfoToSequenceTypeInfo)(type_info, &mut sequence_info)
            })?;
            let sequence_info = require_const_non_null(sequence_info, "sequence type info")?;
            let mut element = ptr::null_mut();
            // SAFETY: sequence_info is live and element is writable output.
            check_status(api, unsafe {
                (api.table().GetSequenceElementType)(sequence_info, &mut element)
            })?;
            let element = require_non_null(element, "sequence element type info")?;
            let _guard = TypeInfoGuard {
                api,
                type_info: element,
            };
            read_value_type(api, element).map(|element| ValueTypeInfo::Sequence(Box::new(element)))
        }
        sys::ONNXType::ONNX_TYPE_MAP => {
            let mut map_info = ptr::null();
            // SAFETY: type_info is a live map description and owns the borrowed
            // map-info view.
            check_status(api, unsafe {
                (api.table().CastTypeInfoToMapTypeInfo)(type_info, &mut map_info)
            })?;
            let map_info = require_const_non_null(map_info, "map type info")?;
            let mut key = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            // SAFETY: map_info is live and key is writable.
            check_status(api, unsafe {
                (api.table().GetMapKeyType)(map_info, &mut key)
            })?;
            let mut value = ptr::null_mut();
            // SAFETY: map_info is live and value is writable output.
            check_status(api, unsafe {
                (api.table().GetMapValueType)(map_info, &mut value)
            })?;
            let value = require_non_null(value, "map value type info")?;
            let _guard = TypeInfoGuard {
                api,
                type_info: value,
            };
            Ok(ValueTypeInfo::Map {
                key: key.into(),
                value: Box::new(read_value_type(api, value)?),
            })
        }
        sys::ONNXType::ONNX_TYPE_OPTIONAL => {
            let mut optional_info = ptr::null();
            // SAFETY: type_info is a live optional description and owns the
            // borrowed optional-info view.
            check_status(api, unsafe {
                (api.table().CastTypeInfoToOptionalTypeInfo)(type_info, &mut optional_info)
            })?;
            let optional_info = require_const_non_null(optional_info, "optional type info")?;
            let mut contained = ptr::null_mut();
            // SAFETY: optional_info is live and contained is writable output.
            check_status(api, unsafe {
                (api.table().GetOptionalContainedTypeInfo)(optional_info, &mut contained)
            })?;
            let contained = require_non_null(contained, "optional contained type info")?;
            let _guard = TypeInfoGuard {
                api,
                type_info: contained,
            };
            read_value_type(api, contained)
                .map(|contained| ValueTypeInfo::Optional(Box::new(contained)))
        }
        sys::ONNXType::ONNX_TYPE_SPARSETENSOR => Ok(ValueTypeInfo::SparseTensor),
        sys::ONNXType::ONNX_TYPE_OPAQUE => Ok(ValueTypeInfo::Opaque),
        sys::ONNXType::ONNX_TYPE_UNKNOWN => Ok(ValueTypeInfo::Unknown),
        _ => Ok(ValueTypeInfo::Unknown),
    }
}

fn read_tensor_type(api: &sys::Api, type_info: *mut sys::OrtTypeInfo) -> Result<TensorTypeInfo> {
    let mut tensor_info = ptr::null();
    // SAFETY: type_info is a tensor type and owns the returned borrowed view.
    check_status(api, unsafe {
        (api.table().CastTypeInfoToTensorInfo)(type_info, &mut tensor_info)
    })?;
    let tensor_info = require_const_non_null(tensor_info, "tensor type info")?;
    let mut element_type = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    // SAFETY: tensor_info remains live with its owning type_info.
    check_status(api, unsafe {
        (api.table().GetTensorElementType)(tensor_info, &mut element_type)
    })?;
    let mut rank = 0;
    // SAFETY: tensor_info is live and rank is writable.
    check_status(api, unsafe {
        (api.table().GetDimensionsCount)(tensor_info, &mut rank)
    })?;
    let mut dimensions = vec![0_i64; rank];
    // SAFETY: dimensions contains rank writable entries.
    check_status(api, unsafe {
        (api.table().GetDimensions)(tensor_info, dimensions.as_mut_ptr(), rank)
    })?;
    let mut symbols = vec![ptr::null(); rank];
    // SAFETY: symbols contains rank writable entries and borrowed strings
    // remain valid while type_info is live.
    check_status(api, unsafe {
        (api.table().GetSymbolicDimensions)(tensor_info, symbols.as_mut_ptr(), rank)
    })?;
    let dimensions = dimensions
        .into_iter()
        .zip(symbols)
        .map(|(dimension, symbol)| {
            if dimension >= 0 {
                Dimension::Fixed(dimension as u64)
            } else if symbol.is_null() {
                Dimension::Dynamic
            } else {
                // SAFETY: non-null symbolic names are NUL-terminated and
                // remain live with the owning type_info.
                Dimension::Symbolic(
                    unsafe { CStr::from_ptr(symbol) }
                        .to_string_lossy()
                        .into_owned(),
                )
            }
        })
        .collect();
    Ok(TensorTypeInfo {
        element_type: element_type.into(),
        dimensions,
    })
}

/// A reusable ONNX Runtime I/O Binding object associated with one session.
///
/// Call [`Self::bind`] to create a scoped set of external DLPack bindings. The
/// native object itself can be rebound after the scoped binding is dropped.
pub struct IoBinding<'session> {
    session: &'session Session,
    binding: *mut sys::OrtIoBinding,
}

// SAFETY: the binding is uniquely owned and may be moved, while operations
// require &mut self. Its session supports concurrent runs.
unsafe impl Send for IoBinding<'_> {}

impl<'session> IoBinding<'session> {
    /// Binds preallocated CPU or CUDA inputs and outputs for repeated runs.
    pub fn bind<'binding>(
        &'binding mut self,
        inputs: &'binding [IoBindingInput<'binding>],
        outputs: &'binding mut [IoBindingOutput<'binding>],
    ) -> Result<BoundIoBinding<'binding, 'session>> {
        self.clear();
        let api = &self.session.environment.inner.api;
        let mut memory_infos = Vec::with_capacity(inputs.len() + outputs.len());
        let mut values = Vec::with_capacity(inputs.len() + outputs.len());

        let result = (|| {
            for input in inputs {
                let memory_info = create_dlpack_memory_info(api, input.tensor.device())?;
                let memory_guard = MemoryInfoGuard { api, memory_info };
                let value = create_external_tensor(api, memory_info, &input.tensor)?;
                let value_guard = ValueGuard::new(api, value);
                // SAFETY: the scoped BoundIoBinding retains the value and its
                // external DLPack allocation until the native binding clears it.
                check_status(api, unsafe {
                    (api.table().BindInput)(self.binding, input.name.as_ptr(), value.cast_const())
                })?;
                memory_infos.push(memory_guard);
                values.push(value_guard);
            }
            for output in outputs.iter_mut() {
                let memory_info = create_dlpack_memory_info(api, output.tensor.device())?;
                let memory_guard = MemoryInfoGuard { api, memory_info };
                let value = create_external_tensor(api, memory_info, &output.tensor)?;
                let value_guard = ValueGuard::new(api, value);
                // SAFETY: output storage is exclusively borrowed for the full
                // BoundIoBinding lifetime.
                check_status(api, unsafe {
                    (api.table().BindOutput)(self.binding, output.name.as_ptr(), value.cast_const())
                })?;
                memory_infos.push(memory_guard);
                values.push(value_guard);
            }
            Ok(())
        })();

        if let Err(error) = result {
            self.clear();
            return Err(error);
        }

        Ok(BoundIoBinding {
            binding: self,
            _values: values,
            _memory_infos: memory_infos,
            _borrows: PhantomData,
        })
    }

    /// Clears all native input and output bindings.
    pub fn clear(&mut self) {
        let api = &self.session.environment.inner.api;
        // SAFETY: the binding is live and uniquely accessed through &mut self.
        unsafe {
            (api.table().ClearBoundInputs)(self.binding);
            (api.table().ClearBoundOutputs)(self.binding);
        }
    }
}

impl Drop for IoBinding<'_> {
    fn drop(&mut self) {
        self.clear();
        // SAFETY: this object uniquely owns the live native binding.
        unsafe { (self.session.environment.inner.api.table().ReleaseIoBinding)(self.binding) };
    }
}

/// A scoped set of preallocated DLPack values attached to an [`IoBinding`].
///
/// It may execute repeatedly. Dropping it clears the native binding before
/// releasing any OrtValue or DLPack borrow it references.
pub struct BoundIoBinding<'binding, 'session> {
    binding: &'binding mut IoBinding<'session>,
    _values: Vec<ValueGuard<'binding>>,
    _memory_infos: Vec<MemoryInfoGuard<'binding>>,
    _borrows: PhantomData<&'binding mut ()>,
}

impl BoundIoBinding<'_, '_> {
    /// Executes and synchronizes this set of bindings once.
    pub fn run(&mut self) -> Result<()> {
        self.run_impl(None)
    }

    /// Executes and synchronizes with explicit reusable run options.
    pub fn run_with_options(&mut self, options: &RunOptions) -> Result<()> {
        options.ensure_session(self.binding.session)?;
        self.run_impl(Some(options))
    }

    fn run_impl(&mut self, options: Option<&RunOptions>) -> Result<()> {
        let session = self.binding.session;
        let api = &session.environment.inner.api;
        // SAFETY: all external values remain retained by self.
        check_status(api, unsafe {
            (api.table().SynchronizeBoundInputs)(self.binding.binding)
        })?;
        // SAFETY: the session, native binding, options, and external values live
        // through the synchronous call.
        check_status(api, unsafe {
            (api.table().RunWithBinding)(
                session.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                self.binding.binding,
            )
        })?;
        // SAFETY: wait until all provider writes to external outputs complete.
        check_status(api, unsafe {
            (api.table().SynchronizeBoundOutputs)(self.binding.binding)
        })
    }
}

impl Drop for BoundIoBinding<'_, '_> {
    fn drop(&mut self) {
        self.binding.clear();
    }
}

/// A submitted I/O Binding run that retains all bound DLPack storage.
///
/// Call [`Self::wait`] to report synchronization errors. Dropping an
/// unfinished value also synchronizes, but cannot report an error.
#[must_use = "the pending run must be waited on or dropped to synchronize outputs"]
pub struct PendingIoBindingRun<'a> {
    session: &'a Session,
    // Declared before values and memory info so normal field destruction
    // releases binding-held references first.
    binding: Option<IoBindingGuard<'a>>,
    _values: Vec<ValueGuard<'a>>,
    _memory_infos: Vec<MemoryInfoGuard<'a>>,
    _borrows: PhantomData<&'a mut ()>,
    synchronized: bool,
}

// SAFETY: Session is Sync, bound inputs are shared Send + Sync borrows, bound
// outputs are exclusive Send borrows, and all native objects are retained until
// provider synchronization completes.
unsafe impl Send for PendingIoBindingRun<'_> {}

impl PendingIoBindingRun<'_> {
    /// Blocks until bound provider output work has completed.
    pub fn wait(mut self) -> Result<()> {
        self.synchronize()?;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<()> {
        if self.synchronized {
            return Ok(());
        }
        let binding = self
            .binding
            .as_ref()
            .expect("binding remains present until field destruction")
            .binding;
        let api = &self.session.environment.inner.api;
        // SAFETY: the binding and all bound storage remain live in self.
        check_status(api, unsafe {
            (api.table().SynchronizeBoundOutputs)(binding)
        })?;
        self.synchronized = true;
        Ok(())
    }
}

impl Drop for PendingIoBindingRun<'_> {
    fn drop(&mut self) {
        if self.synchronized {
            return;
        }
        let binding = self
            .binding
            .as_ref()
            .expect("binding remains present until field destruction")
            .binding;
        let api = &self.session.environment.inner.api;
        // SAFETY: all bound storage remains live until after this Drop body.
        let status = unsafe { (api.table().SynchronizeBoundOutputs)(binding) };
        if !status.0.is_null() {
            // Drop cannot report synchronization errors, but must release the
            // owned status object.
            // SAFETY: a non-null status is owned by the caller.
            unsafe { (api.table().ReleaseStatus)(status.0) };
        }
        self.synchronized = true;
    }
}

struct AsyncCompletion {
    api: sys::Api,
    expected_outputs: usize,
    inner: Mutex<AsyncCompletionInner>,
    completed: Condvar,
}

struct AsyncCompletionInner {
    result: Option<Result<Vec<usize>>>,
    waker: Option<Waker>,
}

impl AsyncCompletion {
    fn new(api: sys::Api, expected_outputs: usize) -> Self {
        Self {
            api,
            expected_outputs,
            inner: Mutex::new(AsyncCompletionInner {
                result: None,
                waker: None,
            }),
            completed: Condvar::new(),
        }
    }

    fn complete(&self, result: Result<Vec<usize>>) {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        inner.result = Some(result);
        let waker = inner.waker.take();
        drop(inner);
        self.completed.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn poll_result(&self, waker: &Waker) -> Option<Result<Vec<usize>>> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        if inner.result.is_some() {
            inner.result.take()
        } else {
            let replace = inner
                .waker
                .as_ref()
                .is_none_or(|current| !current.will_wake(waker));
            if replace {
                inner.waker = Some(waker.clone());
            }
            None
        }
    }

    fn wait_result(&self) -> Result<Vec<usize>> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        while inner.result.is_none() {
            inner = self
                .completed
                .wait(inner)
                .unwrap_or_else(|error| error.into_inner());
        }
        inner.result.take().expect("result checked above")
    }
}

// SAFETY: callback completion is protected by a mutex and the stored usize
// values are opaque owned OrtValue addresses, never dereferenced without the
// retained API table.
unsafe impl Send for AsyncCompletion {}
// SAFETY: all shared mutation is synchronized by the mutex and condition variable.
unsafe impl Sync for AsyncCompletion {}

unsafe extern "system" fn async_run_callback(
    user_data: *mut std::ffi::c_void,
    outputs: *mut *mut sys::OrtValue,
    num_outputs: usize,
    status: sys::OrtStatusPtr,
) {
    // SAFETY: RunAsync received exactly one strong Arc reference as user_data;
    // this callback is invoked at most once and reclaims that reference.
    let completion = unsafe { Arc::from_raw(user_data.cast::<AsyncCompletion>()) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        async_callback_result(&completion, outputs, num_outputs, status)
    }))
    .unwrap_or(Err(Error::AsyncCallbackPanicked));
    completion.complete(result);
}

fn async_callback_result(
    completion: &AsyncCompletion,
    outputs: *mut *mut sys::OrtValue,
    num_outputs: usize,
    status: sys::OrtStatusPtr,
) -> Result<Vec<usize>> {
    if !status.0.is_null() {
        release_callback_outputs(&completion.api, outputs, num_outputs);
        return check_status(&completion.api, status).map(|()| Vec::new());
    }
    if num_outputs != completion.expected_outputs {
        release_callback_outputs(&completion.api, outputs, num_outputs);
        return Err(Error::AsyncOutputCount {
            expected: completion.expected_outputs,
            actual: num_outputs,
        });
    }
    let result = collect_callback_outputs(outputs, num_outputs);
    if result.is_err() {
        release_callback_outputs(&completion.api, outputs, num_outputs);
    }
    result
}

fn collect_callback_outputs(
    outputs: *mut *mut sys::OrtValue,
    num_outputs: usize,
) -> Result<Vec<usize>> {
    if num_outputs == 0 {
        return Ok(Vec::new());
    }
    if outputs.is_null() {
        return Err(Error::NullPointer {
            what: "async output array",
        });
    }
    // SAFETY: ORT supplies num_outputs entries on a successful callback.
    let values = unsafe { std::slice::from_raw_parts(outputs, num_outputs) };
    if values.iter().any(|value| value.is_null()) {
        return Err(Error::NullPointer {
            what: "async output value",
        });
    }
    Ok(values.iter().map(|value| *value as usize).collect())
}

fn release_callback_outputs(api: &sys::Api, outputs: *mut *mut sys::OrtValue, num_outputs: usize) {
    if outputs.is_null() {
        return;
    }
    // SAFETY: callback supplied an array of num_outputs entries.
    let values = unsafe { std::slice::from_raw_parts(outputs, num_outputs) };
    for value in values.iter().copied().filter(|value| !value.is_null()) {
        // SAFETY: callback output ownership belongs to the caller.
        unsafe { (api.table().ReleaseValue)(value) };
    }
}

/// Future returned by [`Session::run_all_async`].
#[must_use = "futures do nothing unless awaited or polled"]
pub struct RunFuture<'a> {
    session: &'a Session,
    environment: Environment,
    completion: Arc<AsyncCompletion>,
    names: Vec<String>,
    _input_values: Vec<ValueGuard<'a>>,
    _memory_info: MemoryInfoGuard<'a>,
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

// SAFETY: the Future retains a shared borrow of Send + Sync Managed inputs,
// Session is Sync, ORT owns callback synchronization, and every native pointer
// remains live until callback completion.
unsafe impl Send for RunFuture<'_> {}

impl Future for RunFuture<'_> {
    type Output = Result<Vec<Output>>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        assert!(!self.finished, "RunFuture polled after completion");
        let Some(result) = self.completion.poll_result(context.waker()) else {
            return Poll::Pending;
        };
        self.finished = true;
        Poll::Ready(match result {
            Ok(values) => self.convert_outputs(values),
            Err(error) => Err(error),
        })
    }
}

impl RunFuture<'_> {
    /// Requests cancellation of the native run.
    ///
    /// Completion remains asynchronous; await the Future to observe the final
    /// ORT status. Cancellation is cooperative and may race with completion.
    pub fn cancel(&self) -> Result<()> {
        self.run_options.terminate()
    }

    fn convert_outputs(&mut self, values: Vec<usize>) -> Result<Vec<Output>> {
        let api = &self.environment.inner.api;
        let guards = values
            .into_iter()
            .map(|value| ValueGuard::new(api, value as *mut sys::OrtValue))
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(guards.len());
        for (name, mut guard) in std::mem::take(&mut self.names).into_iter().zip(guards) {
            let value = guard.ptr;
            guard.disarm();
            outputs.push(Output {
                name,
                tensor: output_to_dlpack(self.environment.clone(), value)?,
            });
        }
        Ok(outputs)
    }
}

impl Drop for RunFuture<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // Best-effort cooperative cancellation shortens Drop when the run has
        // not completed. Drop cannot report a termination-request error.
        let _ = self.run_options.terminate();
        if let Ok(values) = self.completion.wait_result() {
            for value in values {
                // SAFETY: the unconsumed successful callback transferred each
                // output value to this future, which releases it exactly once.
                unsafe {
                    (self.session.environment.inner.api.table().ReleaseValue)(
                        value as *mut sys::OrtValue,
                    )
                };
            }
        }
        self.finished = true;
    }
}

#[derive(Clone, Copy)]
enum IoKind {
    Input,
    Output,
    Initializer,
}

/// A named DLPack output returned by [`Session::run_all`].
pub struct Output {
    name: String,
    tensor: versioned::Dlpack,
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

/// A named runtime value returned by [`Session::run_values`].
pub struct ValueOutput {
    name: String,
    value: Value,
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
    values: Vec<InputValue<'a>>,
}

/// A named map input.
pub struct MapInput<'a> {
    name: CString,
    keys: InputValue<'a>,
    values: InputValue<'a>,
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

/// A heterogeneous model input accepted by [`Session::run_values`].
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
    fn name(&self) -> &CStr {
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
    name: CString,
    tensor: TensorRef<'a>,
}

/// A named caller-owned DLPack tensor used as an inference output buffer.
pub struct OutputBuffer<'a> {
    name: CString,
    tensor: TensorMut<'a>,
}

/// A named compact CPU or CUDA DLPack input for ORT I/O Binding.
pub struct IoBindingInput<'a> {
    name: CString,
    tensor: TensorRef<'a>,
}

/// A named writable compact CPU or CUDA DLPack output for ORT I/O Binding.
pub struct IoBindingOutput<'a> {
    name: CString,
    tensor: TensorMut<'a>,
}

/// A named output whose allocation and dynamic shape are selected by ORT.
#[derive(Debug, Clone)]
pub struct IoBindingDeviceOutput {
    name: CString,
    device: DLDevice,
}

impl IoBindingDeviceOutput {
    /// Requests an ORT-allocated CUDA output on `device_id`.
    pub fn cuda(name: &str, device_id: i32) -> Result<Self> {
        if device_id < 0 {
            return Err(Error::UnsupportedIoBindingDevice {
                device_type: DLDeviceType::CUDA,
                device_id,
            });
        }
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

fn validate_cpu_compact(tensor: &impl TensorView) -> Result<()> {
    let tensor = tensor.as_raw_tensor();
    let device = tensor.device();
    if device.device_type != DLDeviceType::CPU || device.device_id != 0 {
        return Err(Error::UnsupportedDevice {
            device_type: device.device_type,
            device_id: device.device_id,
        });
    }
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

fn validate_iobinding_compact(tensor: &impl TensorView) -> Result<()> {
    let tensor = tensor.as_raw_tensor();
    let device = tensor.device();
    let supported = (device.device_type == DLDeviceType::CPU && device.device_id == 0)
        || (device.device_type == DLDeviceType::CUDA && device.device_id >= 0);
    if !supported {
        return Err(Error::UnsupportedIoBindingDevice {
            device_type: device.device_type,
            device_id: device.device_id,
        });
    }
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

fn create_dlpack_memory_info(api: &sys::Api, device: DLDevice) -> Result<*mut sys::OrtMemoryInfo> {
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
    require_non_null(memory_info, "DLPack memory info")
}

fn create_external_tensor(
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
    require_non_null(value, "I/O Binding tensor value")
}

fn validate_dlpack_flags(flags: DlpackFlags, dtype: DLDataType) -> Result<()> {
    if flags.contains(DlpackFlags::IS_SUBBYTE_TYPE_PADDED) {
        return Err(Error::UnsupportedDlpackFlags { flags, dtype });
    }
    Ok(())
}

fn string_element_count(shape: &[i64]) -> Result<usize> {
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

fn string_input_values<I, S>(shape: &[i64], values: I) -> Result<Vec<CString>>
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
    if values.len() != expected {
        return Err(Error::StringElementCount {
            expected,
            actual: values.len(),
        });
    }
    Ok(values)
}

fn dlpack_to_onnx(dtype: DLDataType) -> Result<sys::ONNXTensorElementDataType> {
    use sys::ONNXTensorElementDataType as O;
    if dtype.lanes != 1 {
        return Err(Error::UnsupportedDlpackType { dtype });
    }
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

fn onnx_to_dlpack(element_type: sys::ONNXTensorElementDataType) -> Result<DLDataType> {
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

fn create_string_tensor(
    api: &sys::Api,
    allocator: *mut sys::OrtAllocator,
    input: &StringInput,
) -> Result<*mut sys::OrtValue> {
    create_string_tensor_parts(api, allocator, &input.shape, &input.values)
}

fn create_string_tensor_parts(
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

fn create_input_value(
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

fn create_sequence_input(
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

fn create_map_input(
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

fn create_container_input(
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
    require_non_null(value, description)
}

fn output_to_value(environment: Environment, value: *mut sys::OrtValue) -> Result<Value> {
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

fn container_output(
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

fn string_output(api: &sys::Api, value: *mut sys::OrtValue) -> Result<StringTensor> {
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

fn output_to_dlpack(
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

struct OutputOwner {
    environment: Environment,
    value: usize,
}

impl Drop for OutputOwner {
    fn drop(&mut self) {
        // SAFETY: this owner received the unique live OrtValue and drops it once.
        unsafe {
            (self.environment.inner.api.table().ReleaseValue)(self.value as *mut sys::OrtValue)
        };
    }
}

struct ValueGuard<'a> {
    api: &'a sys::Api,
    ptr: *mut sys::OrtValue,
}

impl<'a> ValueGuard<'a> {
    fn new(api: &'a sys::Api, ptr: *mut sys::OrtValue) -> Self {
        Self { api, ptr }
    }

    fn disarm(&mut self) {
        self.ptr = ptr::null_mut();
    }
}

impl Drop for ValueGuard<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: a non-null guard pointer is uniquely owned and live.
            unsafe { (self.api.table().ReleaseValue)(self.ptr) };
        }
    }
}

struct MemoryInfoGuard<'a> {
    api: &'a sys::Api,
    memory_info: *mut sys::OrtMemoryInfo,
}

struct IoBindingGuard<'a> {
    api: &'a sys::Api,
    binding: *mut sys::OrtIoBinding,
}

struct AllocatedBufferGuard<'a> {
    api: &'a sys::Api,
    allocator: *mut sys::OrtAllocator,
    allocation: *mut std::ffi::c_void,
}

impl Drop for AllocatedBufferGuard<'_> {
    fn drop(&mut self) {
        if self.allocation.is_null() {
            return;
        }
        // SAFETY: allocation was returned by this exact allocator and is freed once.
        let status = unsafe { (self.api.table().AllocatorFree)(self.allocator, self.allocation) };
        if !status.0.is_null() {
            // SAFETY: Drop cannot report the error, but owns the status.
            unsafe { (self.api.table().ReleaseStatus)(status.0) };
        }
    }
}

impl Drop for IoBindingGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the binding pointer.
        unsafe { (self.api.table().ReleaseIoBinding)(self.binding) };
    }
}

impl Drop for MemoryInfoGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns a live memory-info object.
        unsafe { (self.api.table().ReleaseMemoryInfo)(self.memory_info) };
    }
}

struct ShapeInfoGuard<'a> {
    api: &'a sys::Api,
    info: *mut sys::OrtTensorTypeAndShapeInfo,
}

struct TypeInfoGuard<'a> {
    api: &'a sys::Api,
    type_info: *mut sys::OrtTypeInfo,
}

struct AvailableProvidersGuard<'a> {
    api: &'a sys::Api,
    providers: *mut *mut std::ffi::c_char,
    count: i32,
}

impl Drop for AvailableProvidersGuard<'_> {
    fn drop(&mut self) {
        if !self.providers.is_null() {
            discard_provider_array(self.api, self.providers, self.count);
        }
    }
}

fn discard_provider_array(api: &sys::Api, providers: *mut *mut std::ffi::c_char, count: i32) {
    // SAFETY: providers and count are the unmodified outputs from
    // GetAvailableProviders and this function is called exactly once.
    let status = unsafe { (api.table().ReleaseAvailableProviders)(providers, count) };
    if !status.0.is_null() {
        // ReleaseAvailableProviders is documented never to fail. Still release
        // a defensive unexpected status to avoid leaking it during Drop.
        // SAFETY: a non-null status is owned by the caller.
        unsafe { (api.table().ReleaseStatus)(status.0) };
    }
}

impl Drop for TypeInfoGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns a live type-info object.
        unsafe { (self.api.table().ReleaseTypeInfo)(self.type_info) };
    }
}

impl Drop for ShapeInfoGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns a live shape-info object.
        unsafe { (self.api.table().ReleaseTensorTypeAndShapeInfo)(self.info) };
    }
}

fn cstring(value: &str, kind: &'static str) -> Result<CString> {
    CString::new(value).map_err(|source| Error::InvalidName { kind, source })
}

fn require_non_null<T>(pointer: *mut T, what: &'static str) -> Result<*mut T> {
    if pointer.is_null() {
        Err(Error::NullPointer { what })
    } else {
        Ok(pointer)
    }
}

fn require_const_non_null<T>(pointer: *const T, what: &'static str) -> Result<*const T> {
    if pointer.is_null() {
        Err(Error::NullPointer { what })
    } else {
        Ok(pointer)
    }
}

fn default_allocator(api: &sys::Api) -> Result<*mut sys::OrtAllocator> {
    let mut allocator = ptr::null_mut();
    // SAFETY: allocator is writable output storage; the returned default
    // allocator has process lifetime and is not released by callers.
    check_status(api, unsafe {
        (api.table().GetAllocatorWithDefaultOptions)(&mut allocator)
    })?;
    require_non_null(allocator, "default allocator")
}

fn check_status(api: &sys::Api, status: sys::OrtStatusPtr) -> Result<()> {
    if status.0.is_null() {
        return Ok(());
    }
    // SAFETY: non-null status is owned by the caller until ReleaseStatus.
    let code = unsafe { (api.table().GetErrorCode)(status.0) };
    // SAFETY: the status remains live and owns its diagnostic message.
    let message_ptr = unsafe { (api.table().GetErrorMessage)(status.0) };
    let message = if message_ptr.is_null() {
        "unknown native error".to_owned()
    } else {
        // SAFETY: ORT guarantees a NUL-terminated message for a live status.
        unsafe { CStr::from_ptr(message_ptr) }
            .to_string_lossy()
            .into_owned()
    };
    // SAFETY: all status fields have been copied and this releases it exactly once.
    unsafe { (api.table().ReleaseStatus)(status.0) };
    Err(Error::Runtime { code, message })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn async_and_dlpack_borrows_have_required_thread_traits() {
        assert_send_sync::<Input<'static>>();
        assert_send_sync::<IoBindingInput<'static>>();
        assert_send::<OutputBuffer<'static>>();
        assert_send::<IoBindingOutput<'static>>();
        assert_send::<PendingIoBindingRun<'static>>();
        assert_send::<RunFuture<'static>>();
    }

    #[test]
    fn accepts_null_output_array_for_zero_output_callback() {
        assert!(
            collect_callback_outputs(ptr::null_mut(), 0)
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            collect_callback_outputs(ptr::null_mut(), 1),
            Err(Error::NullPointer {
                what: "async output array"
            })
        ));
    }

    #[test]
    fn validates_string_tensor_inputs() {
        let scalar = StringInput::new("text", Vec::<i64>::new(), ["hello"]).unwrap();
        assert!(scalar.shape().is_empty());
        assert!(matches!(
            StringInput::new("text", vec![2], ["one"]),
            Err(Error::StringElementCount {
                expected: 2,
                actual: 1
            })
        ));
        assert!(matches!(
            StringInput::new("text", vec![-1], std::iter::empty::<&str>()),
            Err(Error::InvalidStringDimension { axis: 0, value: -1 })
        ));
        assert!(matches!(
            StringInput::new("text", vec![1], ["bad\0value"]),
            Err(Error::InvalidStringElement { index: 0, .. })
        ));
    }

    #[test]
    fn maps_common_dlpack_types() {
        let f32_type = DLDataType {
            code: DLDataTypeCode::FLOAT,
            bits: 32,
            lanes: 1,
        };
        assert_eq!(
            dlpack_to_onnx(f32_type).unwrap(),
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        );
        assert_eq!(
            onnx_to_dlpack(dlpack_to_onnx(f32_type).unwrap()).unwrap(),
            f32_type
        );
    }

    #[test]
    fn rejects_vector_lanes() {
        let vector = DLDataType {
            code: DLDataTypeCode::FLOAT,
            bits: 32,
            lanes: 4,
        };
        assert!(matches!(
            dlpack_to_onnx(vector),
            Err(Error::UnsupportedDlpackType { .. })
        ));
    }

    #[test]
    fn round_trips_all_supported_dlpack_types() {
        use sys::ONNXTensorElementDataType as O;

        let cases = [
            (
                DLDataTypeCode::FLOAT,
                16,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            ),
            (
                DLDataTypeCode::FLOAT,
                32,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ),
            (
                DLDataTypeCode::FLOAT,
                64,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            ),
            (
                DLDataTypeCode::BFLOAT,
                16,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
            ),
            (
                DLDataTypeCode::INT,
                8,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            ),
            (
                DLDataTypeCode::INT,
                16,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
            ),
            (
                DLDataTypeCode::INT,
                32,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            ),
            (
                DLDataTypeCode::INT,
                64,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            ),
            (
                DLDataTypeCode::UINT,
                8,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            ),
            (
                DLDataTypeCode::UINT,
                16,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
            ),
            (
                DLDataTypeCode::UINT,
                32,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            ),
            (
                DLDataTypeCode::UINT,
                64,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
            ),
            (
                DLDataTypeCode::BOOL,
                8,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            ),
            (
                DLDataTypeCode::COMPLEX,
                64,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
            ),
            (
                DLDataTypeCode::COMPLEX,
                128,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
            ),
            (
                DLDataTypeCode::FLOAT8_E4M3FN,
                8,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
            ),
            (
                DLDataTypeCode::FLOAT8_E4M3FNUZ,
                8,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
            ),
            (
                DLDataTypeCode::FLOAT8_E5M2,
                8,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
            ),
            (
                DLDataTypeCode::FLOAT8_E5M2FNUZ,
                8,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ,
            ),
            (
                DLDataTypeCode::UINT,
                4,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
            ),
            (
                DLDataTypeCode::INT,
                4,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
            ),
            (
                DLDataTypeCode::FLOAT4_E2M1FN,
                4,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1,
            ),
            (
                DLDataTypeCode::UINT,
                2,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2,
            ),
            (
                DLDataTypeCode::INT,
                2,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2,
            ),
            (
                DLDataTypeCode::FLOAT8_E8M0FNU,
                8,
                O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0,
            ),
        ];

        for (code, bits, onnx) in cases {
            let dlpack = DLDataType {
                code,
                bits,
                lanes: 1,
            };
            assert_eq!(dlpack_to_onnx(dlpack).unwrap(), onnx);
            assert_eq!(onnx_to_dlpack(onnx).unwrap(), dlpack);
        }
    }

    #[test]
    fn rejects_onnx_strings_for_dlpack() {
        assert!(matches!(
            onnx_to_dlpack(sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
            Err(Error::UnsupportedOnnxType { .. })
        ));
    }

    #[test]
    fn rejects_padded_subbyte_dlpack_layout() {
        let mut storage = Box::new(vec![0_u8; 2]);
        let data = storage.as_mut_ptr();
        let mut initialized = Dynamic::compact([2_i64].as_slice())
            .initialize(storage)
            .unwrap();
        initialized
            .set_data(data.cast())
            .set_device(DLDevice {
                device_type: DLDeviceType::CPU,
                device_id: 0,
            })
            .set_dtype(DLDataType {
                code: DLDataTypeCode::INT,
                bits: 4,
                lanes: 1,
            });
        // SAFETY: the test allocation intentionally models one padded byte per
        // logical int4 element and accurately declares that layout in flags.
        unsafe {
            initialized.set_flags_unchecked(DlpackFlags::IS_SUBBYTE_TYPE_PADDED);
        }
        // SAFETY: all descriptor fields and owned storage are initialized.
        let tensor = unsafe { initialized.finish() };
        assert!(matches!(
            Input::from_dlpack("x", &tensor),
            Err(Error::UnsupportedDlpackFlags { .. })
        ));
    }

    #[test]
    fn rejects_read_only_preallocated_output() {
        let mut storage = Box::new(vec![0_f32; 2]);
        let data = storage.as_mut_ptr();
        let mut initialized = Dynamic::compact([2_i64].as_slice())
            .initialize(storage)
            .unwrap();
        initialized
            .set_data(data.cast())
            .set_device(DLDevice {
                device_type: DLDeviceType::CPU,
                device_id: 0,
            })
            .set_dtype(DLDataType::F32)
            .set_flags(DlpackFlags::READ_ONLY)
            .unwrap();
        // SAFETY: all descriptor fields and owned storage are initialized.
        let mut tensor = unsafe { initialized.finish() };
        assert!(matches!(
            OutputBuffer::from_dlpack("y", &mut tensor),
            Err(Error::DlpackTensor { .. })
        ));
    }

    #[test]
    fn builds_typed_cuda_options() {
        let provider = CudaExecutionProvider::new()
            .with_device_id(2)
            .with_memory_limit(1024)
            .with_arena_extend_strategy(CudaArenaExtendStrategy::SameAsRequested)
            .with_cudnn_conv_algo_search(CudnnConvAlgoSearch::Heuristic)
            .with_copy_in_default_stream(true)
            .with_cudnn_max_workspace(false)
            .with_tf32(true)
            .with_cuda_graph(false);
        assert_eq!(
            provider.options.0,
            [
                ("device_id".to_owned(), "2".to_owned()),
                ("gpu_mem_limit".to_owned(), "1024".to_owned()),
                (
                    "arena_extend_strategy".to_owned(),
                    "kSameAsRequested".to_owned()
                ),
                ("cudnn_conv_algo_search".to_owned(), "HEURISTIC".to_owned()),
                ("do_copy_in_default_stream".to_owned(), "1".to_owned()),
                ("cudnn_conv_use_max_workspace".to_owned(), "0".to_owned()),
                ("use_tf32".to_owned(), "1".to_owned()),
                ("enable_cuda_graph".to_owned(), "0".to_owned()),
            ]
        );
    }

    #[test]
    fn provider_option_overwrites_existing_key() {
        let provider = TensorRtExecutionProvider::new()
            .with_fp16(false)
            .option("trt_fp16_enable", "1");
        assert_eq!(
            provider.options.0,
            [("trt_fp16_enable".to_owned(), "1".to_owned())]
        );
    }

    #[test]
    fn rejects_nul_in_provider_options() {
        let provider = DnnlExecutionProvider::new().option("bad\0key", "value");
        assert!(matches!(
            provider.options.encode("oneDNN provider option"),
            Err(Error::InvalidName { .. })
        ));
    }

    #[test]
    fn accepts_compact_cuda_iobinding_descriptor() {
        let mut storage = Box::new(vec![1.0_f32, 2.0]);
        let data = storage.as_mut_ptr();
        let mut initialized = Dynamic::compact([2_i64].as_slice())
            .initialize(storage)
            .unwrap();
        initialized
            .set_data(data.cast())
            .set_device(DLDevice {
                device_type: DLDeviceType::CUDA,
                device_id: 3,
            })
            .set_dtype(DLDataType::F32);
        // SAFETY: this test validates metadata only. The pointer is never
        // dereferenced as CUDA memory or passed to ONNX Runtime.
        let tensor = unsafe { initialized.finish() };
        let input = IoBindingInput::from_dlpack("x", &tensor).unwrap();
        assert_eq!(input.device().device_type, DLDeviceType::CUDA);
        assert_eq!(input.device().device_id, 3);
    }

    #[test]
    fn rejects_negative_cuda_iobinding_device() {
        let mut storage = Box::new(vec![1.0_f32]);
        let data = storage.as_mut_ptr();
        let mut initialized = Dynamic::compact([1_i64].as_slice())
            .initialize(storage)
            .unwrap();
        initialized
            .set_data(data.cast())
            .set_device(DLDevice {
                device_type: DLDeviceType::CUDA,
                device_id: -1,
            })
            .set_dtype(DLDataType::F32);
        // SAFETY: this test validates metadata only and never dereferences data.
        let tensor = unsafe { initialized.finish() };
        assert!(matches!(
            IoBindingInput::from_dlpack("x", &tensor),
            Err(Error::UnsupportedIoBindingDevice { .. })
        ));
    }
}
