use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use onyxrt_sys as sys;
use snafu::{Snafu, ensure};

use crate::{
    ffi::{
        self, AllocatedBufferGuard, TypeInfoGuard, api_function, check_status, cstring,
        default_allocator, require_const_non_null, require_non_null, status_call,
    },
    io_binding::IoBinding,
    model::{
        Dimension, Environment, ModelMetadata, TensorInfo, TensorTypeInfo, ValueInfo, ValueTypeInfo,
    },
    provider::{
        self, CpuExecutionProvider, CudaExecutionProvider, DnnlExecutionProvider, ExecutionMode,
        ExecutionProvider, GraphOptimizationLevel, LogLevel, NamedExecutionProvider,
        NativeModelPath, OpenVinoExecutionProvider, TensorRtExecutionProvider,
    },
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Clone, Copy)]
enum IoKind {
    Input,
    Output,
    Initializer,
}

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Ffi {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(transparent)]
    Provider { source: provider::Error },

    #[snafu(display("external initializer belongs to a different ONNX Runtime environment"))]
    ForeignInitializer,

    #[snafu(display("LoRA adapter belongs to a different ONNX Runtime environment"))]
    ForeignLoraAdapter,

    #[snafu(display("unsupported ONNX value type {value_type:?}; expected a tensor"))]
    UnsupportedValueType { value_type: sys::ONNXType },
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
    }
}

/// Reusable configuration for creating ONNX Runtime inference sessions.
///
/// Model loading borrows this builder, so one configured instance can create
/// multiple independent sessions when an application switches models.
#[derive(Clone)]
pub struct SessionBuilder {
    pub(super) environment: Environment,
    pub(super) intra_threads: Option<i32>,
    pub(super) inter_threads: Option<i32>,
    pub(super) optimization: GraphOptimizationLevel,
    pub(super) execution_mode: ExecutionMode,
    pub(super) cpu_provider: CpuExecutionProvider,
    pub(super) execution_providers: Vec<ExecutionProvider>,
    pub(super) memory_pattern: bool,
    pub(super) profiling_prefix: Option<PathBuf>,
    pub(super) optimized_model_path: Option<PathBuf>,
    pub(super) log_id: Option<String>,
    pub(super) log_level: Option<LogLevel>,
    pub(super) log_verbosity: Option<i32>,
    pub(super) deterministic_compute: Option<bool>,
    pub(super) config_entries: Vec<(String, String)>,
    pub(super) custom_ops_libraries: Vec<PathBuf>,
    pub(super) external_data_files: Vec<(String, Vec<u8>)>,
}

/// A named value supplied while creating a session.
///
/// The value only needs to remain live until session creation returns because
/// ONNX Runtime consumes external initializers while loading the model.
pub struct ExternalInitializer<'value> {
    name: CString,
    value: *const sys::OrtValue,
    environment: &'value Environment,
    _borrow: PhantomData<&'value sys::OrtValue>,
}

impl<'value> ExternalInitializer<'value> {
    /// Creates a borrowed external initializer without copying tensor storage.
    pub fn new<'buffer>(
        name: &str,
        value: &'value crate::value::ValueMut<'buffer>,
    ) -> Result<Self> {
        Ok(Self {
            name: cstring(name, "external initializer")?,
            value: value.as_ptr(),
            environment: value.environment(),
            _borrow: PhantomData,
        })
    }

    /// Returns the initializer name.
    pub fn name(&self) -> &CStr {
        &self.name
    }

    /// Returns the borrowed native value pointer.
    pub fn value_ptr(&self) -> *const sys::OrtValue {
        self.value
    }
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

    /// Supplies one external-data file from memory during model loading.
    ///
    /// The builder owns the bytes until session creation finishes. `name`
    /// must match the relative file name stored in the ONNX model.
    pub fn external_data_file(
        mut self,
        name: impl Into<String>,
        contents: impl Into<Vec<u8>>,
    ) -> Self {
        self.external_data_files
            .push((name.into(), contents.into()));
        self
    }

    /// Configures the built-in CPU execution provider.
    pub fn cpu_provider(mut self, provider: CpuExecutionProvider) -> Self {
        self.cpu_provider = provider;
        self
    }

    /// Appends an execution provider with the next available priority.
    ///
    /// Providers are tried in the order this method is called. The
    /// built-in CPU provider remains available as the final fallback.
    pub fn execution_provider(mut self, provider: impl Into<ExecutionProvider>) -> Self {
        self.execution_providers.push(provider.into());
        self
    }

    /// Builds a session directly from an ONNX model path.
    ///
    /// The native path API preserves the model directory, allowing ONNX
    /// Runtime to resolve relative external-data tensor files correctly.
    pub fn build_from_file(&self, path: impl AsRef<Path>) -> Result<Session> {
        self.build_from_file_with_external_initializers(path, &[])
    }

    /// Builds from a model path while supplying its external initializers.
    pub fn build_from_file_with_external_initializers(
        &self,
        path: impl AsRef<Path>,
        initializers: &[ExternalInitializer<'_>],
    ) -> Result<Session> {
        let path = path.as_ref();
        let native_path = NativeModelPath::new(path)?;
        let options = self.create_options()?;
        self.apply_external_initializers(options.options, initializers)?;
        let api = &self.environment.inner.api;
        let mut session = ptr::null_mut();
        // SAFETY: env, native_path, and options remain live through the call;
        // session points to writable output storage.
        let status = unsafe {
            api_function!(api, CreateSession)(
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
            environment: self.environment.clone(),
            session: session as usize,
        })
    }

    /// Builds a session from an in-memory ONNX model.
    pub fn build_from_bytes(&self, model: &[u8]) -> Result<Session> {
        self.build_from_bytes_with_external_initializers(model, &[])
    }

    /// Builds from model bytes while supplying its external initializers.
    pub fn build_from_bytes_with_external_initializers(
        &self,
        model: &[u8],
        initializers: &[ExternalInitializer<'_>],
    ) -> Result<Session> {
        let options = self.create_options()?;
        self.apply_external_initializers(options.options, initializers)?;
        let api = &self.environment.inner.api;
        let mut session = ptr::null_mut();
        // SAFETY: model and options remain live through the call; env is valid.
        let status = unsafe {
            api_function!(api, CreateSessionFromArray)(
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
            environment: self.environment.clone(),
            session: session as usize,
        })
    }

    fn apply_external_initializers(
        &self,
        options: *mut sys::OrtSessionOptions,
        initializers: &[ExternalInitializer<'_>],
    ) -> Result<()> {
        if initializers.is_empty() {
            return Ok(());
        }
        for initializer in initializers {
            ensure!(
                Arc::ptr_eq(&self.environment.inner, &initializer.environment.inner),
                ForeignInitializerSnafu
            );
        }
        let names = initializers
            .iter()
            .map(|initializer| initializer.name.as_ptr())
            .collect::<Vec<_>>();
        let values = initializers
            .iter()
            .map(ExternalInitializer::value_ptr)
            .collect::<Vec<_>>();
        let api = &self.environment.inner.api;
        // SAFETY: arrays have matching lengths and every value, string, and
        // backing allocation remains live through session creation.
        status_call!(
            api,
            AddExternalInitializers(options, names.as_ptr(), values.as_ptr(), values.len())
        )?;
        Ok(())
    }

    fn create_options(&self) -> Result<SessionOptionsGuard<'_>> {
        let api = &self.environment.inner.api;
        let mut options = ptr::null_mut();
        // SAFETY: output points to writable storage.
        status_call!(api, CreateSessionOptions(&mut options))?;
        let options = require_non_null(options, "session options")?;
        let guard = SessionOptionsGuard { api, options };

        // SAFETY: options is a live, uniquely configured options object.
        status_call!(
            api,
            SetSessionGraphOptimizationLevel(options, self.optimization.into())
        )?;
        // SAFETY: options is live and the execution-mode enum is valid.
        status_call!(
            api,
            SetSessionExecutionMode(options, self.execution_mode.into())
        )?;
        // SAFETY: options is live and uniquely configured before session creation.
        let arena_status = unsafe {
            if self.cpu_provider.arena_enabled() {
                api_function!(api, EnableCpuMemArena)(options)
            } else {
                api_function!(api, DisableCpuMemArena)(options)
            }
        };
        check_status(api, arena_status)?;
        // SAFETY: options is live and uniquely configured before session creation.
        let memory_pattern_status = unsafe {
            if self.memory_pattern {
                api_function!(api, EnableMemPattern)(options)
            } else {
                api_function!(api, DisableMemPattern)(options)
            }
        };
        check_status(api, memory_pattern_status)?;
        if let Some(prefix) = &self.profiling_prefix {
            let prefix = NativeModelPath::new(prefix)?;
            // SAFETY: prefix uses ORT's platform path representation and stays
            // live through the synchronous configuration call.
            status_call!(api, EnableProfiling(options, prefix.as_ptr()))?;
        }
        if let Some(path) = &self.optimized_model_path {
            let path = NativeModelPath::new(path)?;
            // SAFETY: path is platform-native and live for the call.
            status_call!(api, SetOptimizedModelFilePath(options, path.as_ptr()))?;
        }
        if let Some(id) = &self.log_id {
            let id = cstring(id, "session log id")?;
            // SAFETY: id is NUL-terminated and live for the call.
            status_call!(api, SetSessionLogId(options, id.as_ptr()))?;
        }
        if let Some(level) = self.log_level {
            let level: sys::OrtLoggingLevel = level.into();
            // SAFETY: the logging level is a valid ORT enum value.
            status_call!(api, SetSessionLogSeverityLevel(options, level.0 as i32))?;
        }
        if let Some(verbosity) = self.log_verbosity {
            // SAFETY: options is live and uniquely configured.
            status_call!(api, SetSessionLogVerbosityLevel(options, verbosity))?;
        }
        if let Some(enabled) = self.deterministic_compute {
            // SAFETY: options is live and uniquely configured.
            status_call!(api, SetDeterministicCompute(options, enabled))?;
        }
        for (key, value) in &self.config_entries {
            let key = cstring(key, "session configuration key")?;
            let value = cstring(value, "session configuration value")?;
            // SAFETY: both strings are NUL-terminated and live for the call.
            status_call!(
                api,
                AddSessionConfigEntry(options, key.as_ptr(), value.as_ptr())
            )?;
        }
        for path in &self.custom_ops_libraries {
            let path = NativeModelPath::new(path)?;
            // SAFETY: path uses ORT's native representation and remains live
            // through the synchronous registration call.
            status_call!(api, RegisterCustomOpsLibrary_V2(options, path.as_ptr()))?;
        }
        if !self.external_data_files.is_empty() {
            let names = self
                .external_data_files
                .iter()
                .map(|(name, _)| NativeModelPath::new(Path::new(name)))
                .collect::<provider::Result<Vec<_>>>()?;
            let name_pointers = names.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();
            let buffers = self
                .external_data_files
                .iter()
                .map(|(_, contents)| contents.as_ptr().cast_mut().cast())
                .collect::<Vec<*mut std::ffi::c_char>>();
            let lengths = self
                .external_data_files
                .iter()
                .map(|(_, contents)| contents.len())
                .collect::<Vec<_>>();
            // SAFETY: all arrays have matching lengths and the builder retains
            // every name and byte buffer until session creation completes.
            status_call!(
                api,
                AddExternalInitializersFromFilesInMemory(
                    options,
                    name_pointers.as_ptr(),
                    buffers.as_ptr(),
                    lengths.as_ptr(),
                    buffers.len(),
                )
            )?;
        }
        if let Some(threads) = self.intra_threads {
            // SAFETY: options remains valid and uniquely borrowed here.
            status_call!(api, SetIntraOpNumThreads(options, threads))?;
        }
        if let Some(threads) = self.inter_threads {
            // SAFETY: options remains valid and uniquely borrowed here.
            status_call!(api, SetInterOpNumThreads(options, threads))?;
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

pub(super) fn append_cuda_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &CudaExecutionProvider,
) -> Result<()> {
    let encoded = provider.options.encode("CUDA provider option")?;
    let mut options = ptr::null_mut();
    // SAFETY: output points to writable storage owned by the guard below.
    status_call!(api, CreateCUDAProviderOptions(&mut options))?;
    let options = require_non_null(options, "CUDA provider options")?;
    let guard = CudaProviderOptionsGuard { api, options };
    if encoded.len() != 0 {
        // SAFETY: the key/value pointer arrays have matching lengths and all
        // C strings remain live through the synchronous call.
        status_call!(
            api,
            UpdateCUDAProviderOptions(
                options,
                encoded.key_pointers.as_ptr(),
                encoded.value_pointers.as_ptr(),
                encoded.len(),
            )
        )?;
    }
    // SAFETY: both option objects are live for the duration of the call.
    status_call!(
        api,
        SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, options)
    )?;
    drop(guard);
    Ok(())
}

pub(super) fn append_tensorrt_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &TensorRtExecutionProvider,
) -> Result<()> {
    let encoded = provider.options.encode("TensorRT provider option")?;
    let mut options = ptr::null_mut();
    // SAFETY: output points to writable storage owned by the guard below.
    status_call!(api, CreateTensorRTProviderOptions(&mut options))?;
    let options = require_non_null(options, "TensorRT provider options")?;
    let guard = TensorRtProviderOptionsGuard { api, options };
    if encoded.len() != 0 {
        // SAFETY: the key/value pointer arrays have matching lengths and all
        // C strings remain live through the synchronous call.
        status_call!(
            api,
            UpdateTensorRTProviderOptions(
                options,
                encoded.key_pointers.as_ptr(),
                encoded.value_pointers.as_ptr(),
                encoded.len(),
            )
        )?;
    }
    // SAFETY: both option objects are live for the duration of the call.
    status_call!(
        api,
        SessionOptionsAppendExecutionProvider_TensorRT_V2(session_options, options,)
    )?;
    drop(guard);
    Ok(())
}

pub(super) fn append_openvino_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &OpenVinoExecutionProvider,
) -> Result<()> {
    let encoded = provider.options.encode("OpenVINO provider option")?;
    // SAFETY: the key/value pointer arrays have matching lengths and all C
    // strings remain live through the synchronous call.
    status_call!(
        api,
        SessionOptionsAppendExecutionProvider_OpenVINO_V2(
            session_options,
            encoded.key_pointers.as_ptr(),
            encoded.value_pointers.as_ptr(),
            encoded.len(),
        )
    )?;
    Ok(())
}

pub(super) fn append_dnnl_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &DnnlExecutionProvider,
) -> Result<()> {
    let encoded = provider.options.encode("oneDNN provider option")?;
    let mut options = ptr::null_mut();
    // SAFETY: output points to writable storage owned by the guard below.
    status_call!(api, CreateDnnlProviderOptions(&mut options))?;
    let options = require_non_null(options, "oneDNN provider options")?;
    let guard = DnnlProviderOptionsGuard { api, options };
    if encoded.len() != 0 {
        // SAFETY: the key/value pointer arrays have matching lengths and all
        // C strings remain live through the synchronous call.
        status_call!(
            api,
            UpdateDnnlProviderOptions(
                options,
                encoded.key_pointers.as_ptr(),
                encoded.value_pointers.as_ptr(),
                encoded.len(),
            )
        )?;
    }
    // SAFETY: both option objects are live for the duration of the call.
    status_call!(
        api,
        SessionOptionsAppendExecutionProvider_Dnnl(session_options, options)
    )?;
    drop(guard);
    Ok(())
}

pub(super) fn append_named_provider(
    api: &sys::Api,
    session_options: *mut sys::OrtSessionOptions,
    provider: &NamedExecutionProvider,
) -> Result<()> {
    let name = cstring(&provider.name, "execution provider")?;
    let encoded = provider.options.encode("execution provider option")?;
    // SAFETY: provider name, key/value arrays, and their C strings remain live
    // through the synchronous call.
    status_call!(
        api,
        SessionOptionsAppendExecutionProvider(
            session_options,
            name.as_ptr(),
            encoded.key_pointers.as_ptr(),
            encoded.value_pointers.as_ptr(),
            encoded.len(),
        )
    )?;
    Ok(())
}

pub(super) struct SessionOptionsGuard<'a> {
    api: &'a sys::Api,
    options: *mut sys::OrtSessionOptions,
}

impl Drop for SessionOptionsGuard<'_> {
    fn drop(&mut self) {
        if !self.options.is_null() {
            // SAFETY: this guard uniquely owns the options pointer.
            unsafe { api_function!(self.api, ReleaseSessionOptions)(self.options) };
        }
    }
}

pub(super) struct CudaProviderOptionsGuard<'a> {
    api: &'a sys::Api,
    options: *mut sys::OrtCUDAProviderOptionsV2,
}

impl Drop for CudaProviderOptionsGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the options pointer.
        unsafe { api_function!(self.api, ReleaseCUDAProviderOptions)(self.options) };
    }
}

pub(super) struct TensorRtProviderOptionsGuard<'a> {
    api: &'a sys::Api,
    options: *mut sys::OrtTensorRTProviderOptionsV2,
}

pub(super) struct DnnlProviderOptionsGuard<'a> {
    api: &'a sys::Api,
    options: *mut sys::OrtDnnlProviderOptions,
}

impl Drop for DnnlProviderOptionsGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the options pointer.
        unsafe { api_function!(self.api, ReleaseDnnlProviderOptions)(self.options) };
    }
}

impl Drop for TensorRtProviderOptionsGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the options pointer.
        unsafe { api_function!(self.api, ReleaseTensorRTProviderOptions)(self.options) };
    }
}

/// An immutable ONNX Runtime LoRA adapter that can be shared across runs.
#[derive(Clone)]
pub struct LoraAdapter {
    inner: Arc<LoraAdapterInner>,
}

struct LoraAdapterInner {
    environment: Environment,
    adapter: usize,
}

// SAFETY: adapters are immutable after creation and ORT supports using one
// adapter from multiple independently configured runs.
unsafe impl Send for LoraAdapterInner {}
// SAFETY: the wrapper only exposes shared immutable access to the adapter.
unsafe impl Sync for LoraAdapterInner {}

impl LoraAdapter {
    /// Loads an adapter from an ONNX Runtime adapter-format file.
    pub fn from_file(environment: Environment, path: impl AsRef<Path>) -> Result<Self> {
        let path = NativeModelPath::new(path.as_ref())?;
        let api = &environment.inner.api;
        let mut adapter = ptr::null_mut();
        // SAFETY: the native path is valid for the call, null selects CPU
        // storage, and output storage is writable.
        status_call!(
            api,
            CreateLoraAdapter(path.as_ptr(), ptr::null_mut(), &mut adapter)
        )?;
        Self::from_created(environment, adapter)
    }

    /// Copies adapter-format bytes into a new CPU-resident adapter.
    pub fn from_bytes(environment: Environment, bytes: &[u8]) -> Result<Self> {
        let api = &environment.inner.api;
        let mut adapter = ptr::null_mut();
        // SAFETY: bytes remain live for the synchronous copying constructor,
        // null selects CPU storage, and output storage is writable.
        status_call!(
            api,
            CreateLoraAdapterFromArray(
                bytes.as_ptr().cast(),
                bytes.len(),
                ptr::null_mut(),
                &mut adapter,
            )
        )?;
        Self::from_created(environment, adapter)
    }

    fn from_created(environment: Environment, adapter: *mut sys::OrtLoraAdapter) -> Result<Self> {
        let adapter = require_non_null(adapter, "LoRA adapter")?;
        Ok(Self {
            inner: Arc::new(LoraAdapterInner {
                environment,
                adapter: adapter as usize,
            }),
        })
    }

    /// Returns the borrowed native adapter pointer.
    pub fn as_ptr(&self) -> *const sys::OrtLoraAdapter {
        self.inner.adapter as *const sys::OrtLoraAdapter
    }
}

impl Drop for LoraAdapterInner {
    fn drop(&mut self) {
        // SAFETY: the shared inner object uniquely owns the adapter pointer.
        unsafe {
            api_function!(self.environment.inner.api, ReleaseLoraAdapter)(
                self.adapter as *mut sys::OrtLoraAdapter,
            )
        };
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
    lora_adapters: Vec<LoraAdapter>,
}

// SAFETY: ORT run options may be moved between threads. Configuration requires
// &mut self, while the terminate operations are specifically provided by ORT
// for cooperative cancellation of an active run.
unsafe impl Send for RunOptions {}
// SAFETY: shared access only exposes ORT's thread-safe termination operations
// and immutable configuration queries.
unsafe impl Sync for RunOptions {}

impl RunOptions {
    pub(super) fn new(environment: Environment) -> Result<Self> {
        let api = &environment.inner.api;
        let mut run_options = ptr::null_mut();
        // SAFETY: output points to writable storage.
        status_call!(api, CreateRunOptions(&mut run_options))?;
        let run_options = require_non_null(run_options, "run options")?;
        Ok(Self {
            environment,
            run_options: run_options as usize,
            lora_adapters: Vec::new(),
        })
    }

    /// Sets the run tag included in runtime diagnostics and profiling data.
    pub fn set_tag(&mut self, tag: &str) -> Result<&mut Self> {
        let tag = cstring(tag, "run tag")?;
        let api = &self.environment.inner.api;
        // SAFETY: the options are live and tag is NUL-terminated for the call.
        status_call!(api, RunOptionsSetRunTag(self.as_mut_ptr(), tag.as_ptr()))?;
        Ok(self)
    }

    /// Sets the minimum log severity for this run.
    pub fn set_log_level(&mut self, level: LogLevel) -> Result<&mut Self> {
        let level: sys::OrtLoggingLevel = level.into();
        let api = &self.environment.inner.api;
        // SAFETY: the options are live and the logging enum value is valid.
        status_call!(
            api,
            RunOptionsSetRunLogSeverityLevel(self.as_mut_ptr(), level.0 as i32)
        )?;
        Ok(self)
    }

    /// Sets the run-specific verbose logging level.
    pub fn set_log_verbosity(&mut self, verbosity: i32) -> Result<&mut Self> {
        let api = &self.environment.inner.api;
        // SAFETY: the options are live and uniquely configured through &mut self.
        status_call!(
            api,
            RunOptionsSetRunLogVerbosityLevel(self.as_mut_ptr(), verbosity)
        )?;
        Ok(self)
    }

    /// Adds a provider/runtime-specific run configuration entry.
    pub fn add_config_entry(&mut self, key: &str, value: &str) -> Result<&mut Self> {
        let key = cstring(key, "run configuration key")?;
        let value = cstring(value, "run configuration value")?;
        let api = &self.environment.inner.api;
        // SAFETY: options and both NUL-terminated strings are live for the call.
        status_call!(
            api,
            AddRunConfigEntry(self.as_mut_ptr(), key.as_ptr(), value.as_ptr())
        )?;
        Ok(self)
    }

    /// Activates a LoRA adapter for runs using these options.
    ///
    /// This method takes ownership of the adapter handle. Clone the adapter
    /// explicitly when it must also remain available to other run options.
    pub fn add_lora_adapter(&mut self, adapter: LoraAdapter) -> Result<&mut Self> {
        ensure!(
            Arc::ptr_eq(&self.environment.inner, &adapter.inner.environment.inner),
            ForeignLoraAdapterSnafu
        );
        let api = &self.environment.inner.api;
        // SAFETY: both objects are live and this RunOptions retains the adapter.
        status_call!(
            api,
            RunOptionsAddActiveLoraAdapter(self.as_mut_ptr(), adapter.as_ptr())
        )?;
        self.lora_adapters.push(adapter);
        Ok(self)
    }

    /// Requests cooperative termination of a run using these options.
    pub fn terminate(&self) -> Result<()> {
        let api = &self.environment.inner.api;
        // SAFETY: ORT permits termination to be requested while a run is active.
        status_call!(api, RunOptionsSetTerminate(self.as_mut_ptr()))?;
        Ok(())
    }

    /// Clears a previous cooperative termination request.
    pub fn reset_terminate(&self) -> Result<()> {
        let api = &self.environment.inner.api;
        // SAFETY: the options are live and ORT synchronizes its terminate state.
        status_call!(api, RunOptionsUnsetTerminate(self.as_mut_ptr()))?;
        Ok(())
    }

    pub(super) fn belongs_to(&self, session: &Session) -> bool {
        Arc::ptr_eq(&self.environment.inner, &session.environment.inner)
    }

    /// Returns the borrowed native run-options pointer.
    pub fn as_ptr(&self) -> *const sys::OrtRunOptions {
        self.run_options as *const sys::OrtRunOptions
    }

    pub(super) fn as_mut_ptr(&self) -> *mut sys::OrtRunOptions {
        self.run_options as *mut sys::OrtRunOptions
    }
}

impl Drop for RunOptions {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns the live options pointer.
        unsafe { api_function!(self.environment.inner.api, ReleaseRunOptions)(self.as_mut_ptr()) };
    }
}

/// An ONNX Runtime inference session.
pub struct Session {
    pub(super) environment: Environment,
    pub(super) session: usize,
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
            api_function!(self.environment.inner.api, ReleaseSession)(
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
        status_call!(
            api,
            CreateIoBinding(self.session as *mut sys::OrtSession, &mut binding)
        )?;
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
        status_call!(
            api,
            SessionGetModelMetadata(self.session as *const sys::OrtSession, &mut metadata,)
        )?;
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
        status_call!(
            api,
            SessionGetProfilingStartTimeNs(self.session as *const sys::OrtSession, &mut timestamp,)
        )?;
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
        status_call!(
            api,
            SessionEndProfiling(self.session as *mut sys::OrtSession, allocator, &mut path,)
        )?;
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
                IoKind::Input => api_function!(api, SessionGetInputCount)(session, &mut count),
                IoKind::Output => api_function!(api, SessionGetOutputCount)(session, &mut count),
                IoKind::Initializer => {
                    api_function!(api, SessionGetOverridableInitializerCount)(session, &mut count)
                }
            }
        };
        check_status(api, status)?;

        let mut allocator = ptr::null_mut();
        // SAFETY: allocator is writable output storage; the returned default
        // allocator has process lifetime and must not be released by callers.
        status_call!(api, GetAllocatorWithDefaultOptions(&mut allocator))?;
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
                IoKind::Input => api_function!(api, SessionGetInputName)(
                    session,
                    index,
                    allocator,
                    &mut raw_name,
                ),
                IoKind::Output => api_function!(api, SessionGetOutputName)(
                    session,
                    index,
                    allocator,
                    &mut raw_name,
                ),
                IoKind::Initializer => api_function!(api, SessionGetOverridableInitializerName)(
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
        status_call!(api, AllocatorFree(allocator, raw_name.cast()))?;

        let mut type_info = ptr::null_mut();
        // SAFETY: session is live and type_info is writable output storage.
        let status =
            unsafe {
                match kind {
                    IoKind::Input => {
                        api_function!(api, SessionGetInputTypeInfo)(session, index, &mut type_info)
                    }
                    IoKind::Output => {
                        api_function!(api, SessionGetOutputTypeInfo)(session, index, &mut type_info)
                    }
                    IoKind::Initializer => api_function!(
                        api,
                        SessionGetOverridableInitializerTypeInfo
                    )(session, index, &mut type_info),
                }
            };
        check_status(api, status)?;
        let type_info = require_non_null(type_info, "model value type info")?;
        let _type_guard = TypeInfoGuard { api, type_info };
        let value_type = read_value_type(api, type_info)?;

        Ok(ValueInfo { name, value_type })
    }
}

pub(super) fn value_type_kind(value_type: &ValueTypeInfo) -> sys::ONNXType {
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

pub(super) fn read_value_type(
    api: &sys::Api,
    type_info: *mut sys::OrtTypeInfo,
) -> Result<ValueTypeInfo> {
    let mut value_type = sys::ONNXType::ONNX_TYPE_UNKNOWN;
    // SAFETY: type_info is live and value_type is writable.
    status_call!(api, GetOnnxTypeFromTypeInfo(type_info, &mut value_type))?;
    match value_type {
        sys::ONNXType::ONNX_TYPE_TENSOR => {
            read_tensor_type(api, type_info).map(ValueTypeInfo::Tensor)
        }
        sys::ONNXType::ONNX_TYPE_SEQUENCE => {
            let mut sequence_info = ptr::null();
            // SAFETY: type_info is a live sequence description and owns the
            // borrowed sequence-info view.
            status_call!(
                api,
                CastTypeInfoToSequenceTypeInfo(type_info, &mut sequence_info)
            )?;
            let sequence_info = require_const_non_null(sequence_info, "sequence type info")?;
            let mut element = ptr::null_mut();
            // SAFETY: sequence_info is live and element is writable output.
            status_call!(api, GetSequenceElementType(sequence_info, &mut element))?;
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
            status_call!(api, CastTypeInfoToMapTypeInfo(type_info, &mut map_info))?;
            let map_info = require_const_non_null(map_info, "map type info")?;
            let mut key = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            // SAFETY: map_info is live and key is writable.
            status_call!(api, GetMapKeyType(map_info, &mut key))?;
            let mut value = ptr::null_mut();
            // SAFETY: map_info is live and value is writable output.
            status_call!(api, GetMapValueType(map_info, &mut value))?;
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
            status_call!(
                api,
                CastTypeInfoToOptionalTypeInfo(type_info, &mut optional_info)
            )?;
            let optional_info = require_const_non_null(optional_info, "optional type info")?;
            let mut contained = ptr::null_mut();
            // SAFETY: optional_info is live and contained is writable output.
            status_call!(
                api,
                GetOptionalContainedTypeInfo(optional_info, &mut contained)
            )?;
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

pub(super) fn read_tensor_type(
    api: &sys::Api,
    type_info: *mut sys::OrtTypeInfo,
) -> Result<TensorTypeInfo> {
    let mut tensor_info = ptr::null();
    // SAFETY: type_info is a tensor type and owns the returned borrowed view.
    status_call!(api, CastTypeInfoToTensorInfo(type_info, &mut tensor_info))?;
    let tensor_info = require_const_non_null(tensor_info, "tensor type info")?;
    let mut element_type = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    // SAFETY: tensor_info remains live with its owning type_info.
    status_call!(api, GetTensorElementType(tensor_info, &mut element_type))?;
    let mut rank = 0;
    // SAFETY: tensor_info is live and rank is writable.
    status_call!(api, GetDimensionsCount(tensor_info, &mut rank))?;
    let mut dimensions = vec![0_i64; rank];
    // SAFETY: dimensions contains rank writable entries.
    status_call!(
        api,
        GetDimensions(tensor_info, dimensions.as_mut_ptr(), rank)
    )?;
    let mut symbols = vec![ptr::null(); rank];
    // SAFETY: symbols contains rank writable entries and borrowed strings
    // remain valid while type_info is live.
    status_call!(
        api,
        GetSymbolicDimensions(tensor_info, symbols.as_mut_ptr(), rank)
    )?;
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
