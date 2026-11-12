use std::{
    ffi::{CStr, CString, NulError},
    ptr::{self, NonNull},
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

pub(super) struct AllocatedBuffer<'a> {
    pub(super) api: &'a sys::Api,
    pub(super) allocator: *mut sys::OrtAllocator,
    pub(super) allocation: *mut std::ffi::c_void,
}

impl Drop for AllocatedBuffer<'_> {
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

pub(super) trait OrtReleasable {
    unsafe fn release(api: &sys::Api, pointer: *mut Self);
}

pub(super) struct OrtOwned<'api, T: OrtReleasable> {
    api: &'api sys::Api,
    pointer: NonNull<T>,
}

impl<'api, T: OrtReleasable> OrtOwned<'api, T> {
    pub(super) fn new(api: &'api sys::Api, pointer: *mut T, what: &'static str) -> Result<Self> {
        Ok(Self {
            api,
            pointer: NonNull::new(pointer).ok_or(Error::NullPointer { what })?,
        })
    }

    pub(super) fn as_ptr(&self) -> *mut T {
        self.pointer.as_ptr()
    }
}

impl<T: OrtReleasable> Drop for OrtOwned<'_, T> {
    fn drop(&mut self) {
        // SAFETY: this owner contains the unique non-null pointer returned by
        // ORT and invokes the release operation paired with its native type.
        unsafe { T::release(self.api, self.pointer.as_ptr()) };
    }
}

pub(super) struct AvailableProviders<'a> {
    pub(super) api: &'a sys::Api,
    pub(super) providers: *mut *mut std::ffi::c_char,
    pub(super) count: i32,
}

impl Drop for AvailableProviders<'_> {
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

macro_rules! impl_ort_releasable {
    ($native:ty, $release:ident) => {
        impl OrtReleasable for $native {
            unsafe fn release(api: &sys::Api, pointer: *mut Self) {
                // SAFETY: OrtOwned guarantees that pointer is the unique live
                // object returned by this API and calls the paired release once.
                unsafe { api_function!(api, $release)(pointer) };
            }
        }
    };
}

impl_ort_releasable!(
    sys::OrtTensorTypeAndShapeInfo,
    ReleaseTensorTypeAndShapeInfo
);
impl_ort_releasable!(sys::OrtTypeInfo, ReleaseTypeInfo);
impl_ort_releasable!(sys::OrtSessionOptions, ReleaseSessionOptions);
impl_ort_releasable!(sys::OrtCUDAProviderOptionsV2, ReleaseCUDAProviderOptions);
impl_ort_releasable!(
    sys::OrtTensorRTProviderOptionsV2,
    ReleaseTensorRTProviderOptions
);
impl_ort_releasable!(sys::OrtDnnlProviderOptions, ReleaseDnnlProviderOptions);

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
