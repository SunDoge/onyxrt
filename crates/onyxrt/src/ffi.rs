use std::{
    ffi::{CStr, CString, NulError},
    ptr,
};

use onyxrt_sys as sys;
use snafu::Snafu;

pub(super) type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
pub(super) enum Error {
    #[snafu(display("failed to resolve ONNX Runtime API function"))]
    Api { source: sys::Error },

    #[snafu(display("ONNX Runtime error {code:?}: {message}"))]
    Runtime {
        code: sys::OrtErrorCode,
        message: String,
    },

    #[snafu(display("ONNX Runtime returned null for {what}"))]
    NullPointer { what: &'static str },

    #[snafu(display("{kind} name contains an interior NUL byte"))]
    InvalidName {
        kind: &'static str,
        source: NulError,
    },
}

pub(super) struct AllocatedBufferGuard<'a> {
    pub(super) api: &'a sys::Api,
    pub(super) allocator: *mut sys::OrtAllocator,
    pub(super) allocation: *mut std::ffi::c_void,
}

impl Drop for AllocatedBufferGuard<'_> {
    fn drop(&mut self) {
        if self.allocation.is_null() {
            return;
        }
        // SAFETY: allocation was returned by this exact allocator and is freed once.
        let status =
            unsafe { api_function!(self.api, AllocatorFree)(self.allocator, self.allocation) };
        if !status.is_null() {
            // SAFETY: Drop cannot report the error, but owns the status.
            unsafe { api_function!(self.api, ReleaseStatus)(status) };
        }
    }
}

pub(super) struct ShapeInfoGuard<'a> {
    pub(super) api: &'a sys::Api,
    pub(super) info: *mut sys::OrtTensorTypeAndShapeInfo,
}

pub(super) struct TypeInfoGuard<'a> {
    pub(super) api: &'a sys::Api,
    pub(super) type_info: *mut sys::OrtTypeInfo,
}

pub(super) struct AvailableProvidersGuard<'a> {
    pub(super) api: &'a sys::Api,
    pub(super) providers: *mut *mut std::ffi::c_char,
    pub(super) count: i32,
}

impl Drop for AvailableProvidersGuard<'_> {
    fn drop(&mut self) {
        if !self.providers.is_null() {
            discard_provider_array(self.api, self.providers, self.count);
        }
    }
}

pub(super) fn discard_provider_array(
    api: &sys::Api,
    providers: *mut *mut std::ffi::c_char,
    count: i32,
) {
    // SAFETY: providers and count are the unmodified outputs from
    // GetAvailableProviders and this function is called exactly once.
    let status = unsafe { api_function!(api, ReleaseAvailableProviders)(providers, count) };
    if !status.is_null() {
        // ReleaseAvailableProviders is documented never to fail. Still release
        // a defensive unexpected status to avoid leaking it during Drop.
        // SAFETY: a non-null status is owned by the caller.
        unsafe { api_function!(api, ReleaseStatus)(status) };
    }
}

impl Drop for TypeInfoGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns a live type-info object.
        unsafe { api_function!(self.api, ReleaseTypeInfo)(self.type_info) };
    }
}

impl Drop for ShapeInfoGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns a live shape-info object.
        unsafe { api_function!(self.api, ReleaseTensorTypeAndShapeInfo)(self.info) };
    }
}

pub(super) fn cstring(value: &str, kind: &'static str) -> Result<CString> {
    CString::new(value).map_err(|source| Error::InvalidName { kind, source })
}

pub(super) fn require_non_null<T>(pointer: *mut T, what: &'static str) -> Result<*mut T> {
    if pointer.is_null() {
        Err(Error::NullPointer { what })
    } else {
        Ok(pointer)
    }
}

pub(super) fn require_const_non_null<T>(pointer: *const T, what: &'static str) -> Result<*const T> {
    if pointer.is_null() {
        Err(Error::NullPointer { what })
    } else {
        Ok(pointer)
    }
}

pub(super) fn default_allocator(api: &sys::Api) -> Result<*mut sys::OrtAllocator> {
    let mut allocator = ptr::null_mut();
    // SAFETY: allocator is writable output storage; the returned default
    // allocator has process lifetime and is not released by callers.
    status_call!(api, GetAllocatorWithDefaultOptions(&mut allocator))?;
    require_non_null(allocator, "default allocator")
}

pub(super) fn check_status(api: &sys::Api, status: sys::OrtStatusPtr) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    // SAFETY: non-null status is owned by the caller until ReleaseStatus.
    let code = unsafe { api_function!(api, GetErrorCode)(status) };
    // SAFETY: the status remains live and owns its diagnostic message.
    let message_ptr = unsafe { api_function!(api, GetErrorMessage)(status) };
    let message = if message_ptr.is_null() {
        "unknown native error".to_owned()
    } else {
        // SAFETY: ORT guarantees a NUL-terminated message for a live status.
        unsafe { CStr::from_ptr(message_ptr) }
            .to_string_lossy()
            .into_owned()
    };
    // SAFETY: all status fields have been copied and this releases it exactly once.
    unsafe { api_function!(api, ReleaseStatus)(status) };
    Err(Error::Runtime { code, message })
}

macro_rules! api_function {
    ($api:expr, $name:ident) => {{
        ($api)
            .function(($api).table().$name, stringify!($name))
            .expect("negotiated ONNX Runtime API function is missing")
    }};
}

macro_rules! status_call {
    ($api:expr, $name:ident($($argument:expr),* $(,)?)) => {{
        match ($api).function(($api).table().$name, stringify!($name)) {
            Ok(function) => {
                // SAFETY: this crate-private macro is used only after each
                // invocation establishes the C function's preconditions.
                let status = unsafe { function($($argument),*) };
                $crate::ffi::check_status($api, status)
            }
            Err(source) => Err($crate::ffi::Error::Api { source }),
        }
    }};
}

pub(crate) use {api_function, status_call};
