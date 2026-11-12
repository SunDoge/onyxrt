use std::{
    ffi::CStr,
    marker::PhantomData,
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use dlpark::versioned;
use onyxrt_sys as sys;
use snafu::{Snafu, ensure};

use crate::{
    ffi::{
        self, AllocatedBufferGuard, IoBindingGuard, MemoryInfoGuard, TypeInfoGuard, ValueGuard,
        api_function, check_status, cstring, default_allocator, require_const_non_null,
        require_non_null, status_call,
    },
    io_binding::{AsyncCompletion, IoBinding, PendingIoBindingRun, RunFuture, async_run_callback},
    model::{
        Dimension, Environment, ModelMetadata, TensorInfo, TensorTypeInfo, ValueInfo, ValueTypeInfo,
    },
    provider::{
        self, CpuExecutionProvider, CudaExecutionProvider, DnnlExecutionProvider, ExecutionMode,
        ExecutionProvider, GraphOptimizationLevel, LogLevel, NamedExecutionProvider,
        NativeModelPath, OpenVinoExecutionProvider, TensorRtExecutionProvider,
    },
    value::{
        self, Input, IoBindingDeviceOutput, IoBindingInput, IoBindingOutput, IoKind, Output,
        OutputBuffer, ValueInput, ValueOutput, create_dlpack_memory_info, create_external_tensor,
        create_map_input, create_sequence_input, create_string_tensor, dlpack_to_onnx,
        output_to_dlpack, output_to_value,
    },
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Ffi {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(transparent)]
    Provider { source: provider::Error },

    #[snafu(transparent)]
    Value { source: value::Error },

    #[snafu(display("run options belong to a different ONNX Runtime environment"))]
    ForeignRunOptions,

    #[snafu(display("invalid DLPack tensor: {source}"))]
    DlpackTensor { source: dlpark::tensor::Error },

    #[snafu(display("ONNX Runtime returned null for {what}"))]
    NullPointer { what: &'static str },

    #[snafu(display("unsupported ONNX value type {value_type:?}; expected a tensor"))]
    UnsupportedValueType { value_type: sys::ONNXType },

    #[snafu(display("ONNX Runtime replaced preallocated output `{name}`"))]
    ReplacedOutput { name: String },

    #[snafu(display("ONNX Runtime I/O Binding returned {actual} outputs; expected {expected}"))]
    BoundOutputCount { expected: usize, actual: usize },
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
    }
}

/// Builder for an ONNX Runtime inference session.
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
            environment: self.environment,
            session: session as usize,
        })
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
    pub(super) fn new(environment: Environment) -> Result<Self> {
        let api = &environment.inner.api;
        let mut run_options = ptr::null_mut();
        // SAFETY: output points to writable storage.
        status_call!(api, CreateRunOptions(&mut run_options))?;
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

    pub(super) fn ensure_session(&self, session: &Session) -> Result<()> {
        ensure!(self.belongs_to(session), ForeignRunOptionsSnafu);
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
            .map(|info| cstring(info.name(), "output").map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        let names = infos.into_iter().map(|info| info.name).collect::<Vec<_>>();

        let mut memory_info = ptr::null_mut();
        // SAFETY: output storage is writable and enum values are valid.
        status_call!(
            api,
            CreateCpuMemoryInfo(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        )?;
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
            status_call!(
                api,
                CreateTensorWithDataAsOrtValue(
                    memory_info,
                    bytes.as_ptr().cast_mut().cast(),
                    bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    dlpack_to_onnx(input.tensor.dtype())?,
                    &mut value,
                )
            )?;
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
        status_call!(
            api,
            CreateCpuMemoryInfo(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        )?;
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
            .map(|name| cstring(name, "output").map_err(Error::from))
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
            api_function!(api, Run)(
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
                unsafe { api_function!(api, ReleaseValue)(value) };
            }
            return Err(error.into());
        }
        if output_values.iter().any(|value| value.is_null()) {
            for value in output_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: every non-null output is owned by this call.
                unsafe { api_function!(api, ReleaseValue)(value) };
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
        status_call!(
            api,
            CreateCpuMemoryInfo(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        )?;
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
            status_call!(
                api,
                CreateTensorWithDataAsOrtValue(
                    memory_info,
                    bytes.as_ptr().cast_mut().cast(),
                    bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    dlpack_to_onnx(input.tensor.dtype())?,
                    &mut value,
                )
            )?;
            input_values.push(ValueGuard::new(
                api,
                require_non_null(value, "input value")?,
            ));
        }

        let output_cstrings = output_names
            .iter()
            .map(|name| cstring(name, "output").map_err(Error::from))
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
            api_function!(api, Run)(
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
                unsafe { api_function!(api, ReleaseValue)(value) };
            }
            return Err(error.into());
        }

        drop(memory_guard);
        if output_values.iter().any(|value| value.is_null()) {
            for value in output_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: every non-null entry is an owned ORT output.
                unsafe { api_function!(api, ReleaseValue)(value) };
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
        status_call!(
            api,
            CreateCpuMemoryInfo(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut memory_info,
            )
        )?;
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
            status_call!(
                api,
                CreateTensorWithDataAsOrtValue(
                    memory_info,
                    bytes.as_ptr().cast_mut().cast(),
                    bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    dlpack_to_onnx(input.tensor.dtype())?,
                    &mut value,
                )
            )?;
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
            status_call!(
                api,
                CreateTensorWithDataAsOrtValue(
                    memory_info,
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                    shape.as_ptr(),
                    shape.len(),
                    dlpack_to_onnx(dtype)?,
                    &mut value,
                )
            )?;
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
            api_function!(api, Run)(
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
        status_call!(
            api,
            CreateIoBinding(self.session as *mut sys::OrtSession, &mut binding)
        )?;
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
            status_call!(
                api,
                BindInput(binding, input.name.as_ptr(), value.cast_const())
            )?;
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
            status_call!(
                api,
                BindOutput(binding, output.name.as_ptr(), value.cast_const())
            )?;
            memory_infos.push(memory_guard);
            values.push(value_guard);
        }

        // SAFETY: binding owns valid bound values and the provider may use this
        // call to synchronize a producer stream before inference.
        status_call!(api, SynchronizeBoundInputs(binding))?;
        // SAFETY: the session, binding, OrtValues, and all external buffers
        // remain live for the synchronous run.
        status_call!(
            api,
            RunWithBinding(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                binding,
            )
        )?;
        // SAFETY: binding remains live and this blocks until provider writes to
        // the bound output buffers have completed.
        status_call!(api, SynchronizeBoundOutputs(binding))?;

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
        status_call!(
            api,
            CreateIoBinding(self.session as *mut sys::OrtSession, &mut binding)
        )?;
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
            status_call!(
                api,
                BindInput(binding, input.name.as_ptr(), value.cast_const())
            )?;
            memory_infos.push(memory_guard);
            input_values.push(value_guard);
        }
        for output in outputs {
            let memory_info = create_dlpack_memory_info(api, output.device)?;
            let memory_guard = MemoryInfoGuard { api, memory_info };
            // SAFETY: binding, name, and memory info are live. ORT allocates an
            // output with the runtime shape on the requested device.
            status_call!(
                api,
                BindOutputToDevice(binding, output.name.as_ptr(), memory_info)
            )?;
            memory_infos.push(memory_guard);
        }

        // SAFETY: provider-specific synchronization makes bound inputs visible.
        status_call!(api, SynchronizeBoundInputs(binding))?;
        // SAFETY: all binding resources remain live through the run.
        status_call!(
            api,
            RunWithBinding(
                self.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                binding,
            )
        )?;
        // SAFETY: wait until device output allocations are ready for consumers.
        status_call!(api, SynchronizeBoundOutputs(binding))?;

        let mut allocator = ptr::null_mut();
        // SAFETY: allocator is writable output storage; the default allocator
        // has process lifetime and is not released by callers.
        status_call!(api, GetAllocatorWithDefaultOptions(&mut allocator))?;
        let allocator = require_non_null(allocator, "default allocator")?;
        let mut raw_values = ptr::null_mut();
        let mut value_count = 0;
        // SAFETY: all output pointers are writable and binding is live.
        status_call!(
            api,
            GetBoundOutputValues(binding, allocator, &mut raw_values, &mut value_count,)
        )?;
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
                unsafe { api_function!(api, ReleaseValue)(value) };
            }
            return Err(Error::BoundOutputCount {
                expected: outputs.len(),
                actual: value_count,
            });
        }
        if raw_values.iter().any(|value| value.is_null()) {
            for value in raw_values.into_iter().filter(|value| !value.is_null()) {
                // SAFETY: each non-null returned output is caller-owned.
                unsafe { api_function!(api, ReleaseValue)(value) };
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
        status_call!(
            api,
            CreateIoBinding(self.session as *mut sys::OrtSession, &mut binding)
        )?;
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
            status_call!(
                api,
                BindInput(binding, input.name.as_ptr(), value.cast_const())
            )?;
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
            status_call!(
                api,
                BindOutput(binding, output.name.as_ptr(), value.cast_const())
            )?;
            memory_infos.push(memory_guard);
            values.push(value_guard);
        }

        // SAFETY: binding is live; provider-specific synchronization makes
        // producer work visible before inference starts.
        status_call!(api, SynchronizeBoundInputs(binding))?;

        options.add_config_entry("disable_synchronize_execution_providers", "1")?;
        // SAFETY: all bindings and external buffers remain live in the guard
        // returned below. This call submits work without the final EP sync.
        status_call!(
            api,
            RunWithBinding(
                self.session as *mut sys::OrtSession,
                options.as_ptr(),
                binding,
            )
        )?;

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
