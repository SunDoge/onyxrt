use std::{
    ffi::{CStr, OsStr},
    path::PathBuf,
    ptr,
    sync::Arc,
};

use onyxrt_sys as sys;
use snafu::Snafu;

use crate::{
    ffi::{
        self, AllocatedBufferGuard, AvailableProvidersGuard, api_function, check_status, cstring,
        default_allocator, discard_provider_array, require_const_non_null, require_non_null,
        status_call,
    },
    provider::{CpuExecutionProvider, ExecutionMode, GraphOptimizationLevel, LogLevel},
    session::SessionBuilder,
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Ffi {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(display("failed to initialize ONNX Runtime"))]
    LoadRuntime { source: sys::Error },

    #[snafu(display("ONNX Runtime returned invalid provider count {count}"))]
    InvalidProviderCount { count: i32 },
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
    }
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
            sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1 => Self::Float4E2M1,
            sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2 => Self::Uint2,
            sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2 => Self::Int2,
            sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0 => Self::Float8E8M0,
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
    pub(super) name: String,
    pub(super) element_type: ElementType,
    pub(super) dimensions: Vec<Dimension>,
}

/// Tensor type metadata used inside recursive model value descriptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorTypeInfo {
    pub(super) element_type: ElementType,
    pub(super) dimensions: Vec<Dimension>,
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
    pub(super) name: String,
    pub(super) value_type: ValueTypeInfo,
}

/// Read-only metadata embedded in an ONNX model.
pub struct ModelMetadata {
    pub(super) environment: Environment,
    pub(super) metadata: usize,
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
        status_call!(api, ModelMetadataGetVersion(self.as_ptr(), &mut version))?;
        Ok(version)
    }

    /// Looks up a custom metadata value. Missing keys return `None`.
    pub fn custom(&self, key: &str) -> Result<Option<String>> {
        let key = cstring(key, "model metadata key")?;
        let api = &self.environment.inner.api;
        let allocator = default_allocator(api)?;
        let mut value = ptr::null_mut();
        // SAFETY: metadata, allocator, and key are live; value is writable.
        status_call!(
            api,
            ModelMetadataLookupCustomMetadataMap(
                self.as_ptr(),
                allocator,
                key.as_ptr(),
                &mut value,
            )
        )?;
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
                MetadataStringKind::Producer => api_function!(api, ModelMetadataGetProducerName)(
                    self.as_ptr(),
                    allocator,
                    &mut value,
                ),
                MetadataStringKind::GraphName => api_function!(api, ModelMetadataGetGraphName)(
                    self.as_ptr(),
                    allocator,
                    &mut value,
                ),
                MetadataStringKind::Domain => {
                    api_function!(api, ModelMetadataGetDomain)(self.as_ptr(), allocator, &mut value)
                }
                MetadataStringKind::Description => api_function!(api, ModelMetadataGetDescription)(
                    self.as_ptr(),
                    allocator,
                    &mut value,
                ),
                MetadataStringKind::GraphDescription => api_function!(
                    api,
                    ModelMetadataGetGraphDescription
                )(
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
            api_function!(self.environment.inner.api, ReleaseModelMetadata)(
                self.metadata as *mut sys::OrtModelMetadata,
            )
        };
    }
}

#[derive(Clone, Copy)]
pub(super) enum MetadataStringKind {
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
    pub(super) inner: Arc<EnvironmentInner>,
}

pub(super) struct EnvironmentInner {
    pub(super) api: sys::Api,
    pub(super) env: usize,
}

// SAFETY: ORT environments are process-wide configuration objects documented
// for use by sessions on multiple threads. The pointer is only released in Drop.
unsafe impl Send for EnvironmentInner {}
// SAFETY: EnvironmentInner exposes no mutation and ORT permits concurrent use.
unsafe impl Sync for EnvironmentInner {}

impl Drop for EnvironmentInner {
    fn drop(&mut self) {
        // SAFETY: env was created by this exact API table and is released once.
        unsafe { api_function!(self.api, ReleaseEnv)(self.env as *mut sys::OrtEnv) };
    }
}

impl Environment {
    pub(super) fn same_runtime(left: &Self, right: &Self) -> bool {
        Arc::ptr_eq(&left.inner, &right.inner)
    }

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
        status_call!(api, GetAvailableProviders(&mut providers, &mut count))?;
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
            unsafe { api_function!(api, CreateEnv)(self.severity.into(), name.as_ptr(), &mut env) };
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
