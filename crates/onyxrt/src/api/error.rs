use super::*;

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
