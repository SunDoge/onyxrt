//! Safe, dynamically loaded ONNX Runtime bindings.
//!
//! The [`value`] module is the core value and inference API. Optional DLPack
//! interoperability is available through the `dlpack` feature.

#[cfg(feature = "dlpack")]
pub mod dlpack;
pub mod io_binding;
pub mod model;
pub mod provider;
pub mod session;
pub mod value;

mod ffi;
