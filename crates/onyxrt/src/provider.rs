use std::{
    ffi::CString,
    path::{Path, PathBuf},
};

use onyxrt_sys as sys;
use snafu::Snafu;
#[cfg(target_os = "windows")]
use snafu::ensure;

use crate::ffi::{self, cstring};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Ffi {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(display("model path `{}` contains an interior NUL", path.display()))]
    InvalidModelPath { path: PathBuf },
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
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
pub(super) struct ProviderOptions(pub(super) Vec<(String, String)>);

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

    pub(super) fn encode(&self, kind: &'static str) -> Result<EncodedProviderOptions> {
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

pub(super) struct EncodedProviderOptions {
    _keys: Vec<CString>,
    _values: Vec<CString>,
    pub(super) key_pointers: Vec<*const std::ffi::c_char>,
    pub(super) value_pointers: Vec<*const std::ffi::c_char>,
}

impl EncodedProviderOptions {
    pub(super) fn len(&self) -> usize {
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
    pub(super) options: ProviderOptions,
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
    pub(super) options: ProviderOptions,
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
    pub(super) options: ProviderOptions,
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
    pub(super) options: ProviderOptions,
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
/// Providers including `QNN`, `SNPE`, `XNNPACK`, `CoreML`, and `AZURE` can use
/// this entry point. CUDA, TensorRT, and OpenVINO use their dedicated builder
/// methods instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedExecutionProvider {
    pub(super) name: String,
    pub(super) options: ProviderOptions,
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
#[non_exhaustive]
pub enum ExecutionProvider {
    /// NVIDIA CUDA execution provider.
    Cuda(CudaExecutionProvider),
    /// NVIDIA TensorRT execution provider.
    TensorRt(TensorRtExecutionProvider),
    /// Intel OpenVINO execution provider.
    OpenVino(OpenVinoExecutionProvider),
    /// Intel oneDNN execution provider.
    Dnnl(DnnlExecutionProvider),
    /// Provider configured through ONNX Runtime's generic provider API.
    Named(NamedExecutionProvider),
}

impl From<CudaExecutionProvider> for ExecutionProvider {
    fn from(provider: CudaExecutionProvider) -> Self {
        Self::Cuda(provider)
    }
}

impl From<TensorRtExecutionProvider> for ExecutionProvider {
    fn from(provider: TensorRtExecutionProvider) -> Self {
        Self::TensorRt(provider)
    }
}

impl From<OpenVinoExecutionProvider> for ExecutionProvider {
    fn from(provider: OpenVinoExecutionProvider) -> Self {
        Self::OpenVino(provider)
    }
}

impl From<DnnlExecutionProvider> for ExecutionProvider {
    fn from(provider: DnnlExecutionProvider) -> Self {
        Self::Dnnl(provider)
    }
}

impl From<NamedExecutionProvider> for ExecutionProvider {
    fn from(provider: NamedExecutionProvider) -> Self {
        Self::Named(provider)
    }
}

pub(super) const fn bool_option(enabled: bool) -> &'static str {
    if enabled { "1" } else { "0" }
}

#[cfg(not(target_os = "windows"))]
pub(super) struct NativeModelPath(CString);

#[cfg(not(target_os = "windows"))]
impl NativeModelPath {
    pub(super) fn new(path: &Path) -> Result<Self> {
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

    pub(super) fn as_ptr(&self) -> *const sys::os_char {
        self.0.as_ptr()
    }
}

#[cfg(target_os = "windows")]
pub(super) struct NativeModelPath(Vec<u16>);

#[cfg(target_os = "windows")]
impl NativeModelPath {
    pub(super) fn new(path: &Path) -> Result<Self> {
        use std::os::windows::ffi::OsStrExt;

        let mut encoded = path.as_os_str().encode_wide().collect::<Vec<_>>();
        ensure!(
            !encoded.contains(&0),
            InvalidModelPathSnafu {
                path: path.to_owned(),
            }
        );
        encoded.push(0);
        Ok(Self(encoded))
    }

    pub(super) fn as_ptr(&self) -> *const sys::os_char {
        self.0.as_ptr()
    }
}
