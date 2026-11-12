use std::{
    marker::PhantomData,
    ptr,
    sync::{Arc, Condvar, Mutex},
    task::Waker,
};

use onyxrt_sys as sys;
use snafu::{Snafu, ensure};

use crate::{
    ffi::{self, api_function, check_status, status_call},
    model::Environment,
    session::{RunOptions, Session},
    value::{Input as ValueInput, Output as ValueOutput},
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
    Session { source: crate::session::Error },

    #[snafu(display("run options belong to a different ONNX Runtime environment"))]
    ForeignRunOptions,

    #[snafu(display("value belongs to a different ONNX Runtime environment"))]
    ForeignValue,

    #[snafu(display("ONNX Runtime returned null for {what}"))]
    NullPointer { what: &'static str },

    #[snafu(display("ONNX Runtime async callback returned {actual} outputs; expected {expected}"))]
    AsyncOutputCount { expected: usize, actual: usize },

    #[snafu(display("internal Rust callback processing panicked"))]
    AsyncCallbackPanicked,
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
    }
}

/// A reusable ONNX Runtime I/O Binding object associated with one session.
///
/// Call [`Self::bind`] to attach caller-owned `OrtValue` inputs and outputs.
/// The native object can be rebound after the scoped binding is dropped.
pub struct IoBinding<'session> {
    pub(super) session: &'session Session,
    pub(super) binding: *mut sys::OrtIoBinding,
}

// SAFETY: the binding is uniquely owned and may be moved, while operations
// require &mut self. Its session supports concurrent runs.
unsafe impl Send for IoBinding<'_> {}

impl<'session> IoBinding<'session> {
    /// Binds caller-owned values for repeated synchronized runs.
    pub fn bind<'binding>(
        &'binding mut self,
        inputs: &'binding [ValueInput<'binding>],
        outputs: &'binding mut [ValueOutput<'binding>],
    ) -> Result<BoundIoBinding<'binding, 'session>> {
        self.clear();
        let session = self.session;
        let api = &session.environment.inner.api;
        let result = (|| {
            for input in inputs {
                ensure!(
                    Environment::same_runtime(&session.environment, input.environment()),
                    ForeignValueSnafu
                );
                // SAFETY: the scoped result retains the value borrow until clear.
                status_call!(
                    api,
                    BindInput(self.binding, input.name().as_ptr(), input.value_ptr())
                )?;
            }
            for output in outputs.iter() {
                ensure!(
                    Environment::same_runtime(&session.environment, output.environment()),
                    ForeignValueSnafu
                );
                // SAFETY: output storage remains exclusively borrowed until clear.
                status_call!(
                    api,
                    BindOutput(self.binding, output.name().as_ptr(), output.value_ptr())
                )?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.clear();
            return Err(error);
        }
        Ok(BoundIoBinding {
            binding: self,
            _borrows: PhantomData,
        })
    }

    /// Clears all native input and output bindings.
    pub fn clear(&mut self) {
        let api = &self.session.environment.inner.api;
        // SAFETY: the binding is live and uniquely accessed through &mut self.
        unsafe {
            api_function!(api, ClearBoundInputs)(self.binding);
            api_function!(api, ClearBoundOutputs)(self.binding);
        }
    }
}

impl Drop for IoBinding<'_> {
    fn drop(&mut self) {
        self.clear();
        // SAFETY: this object uniquely owns the live native binding.
        unsafe {
            api_function!(self.session.environment.inner.api, ReleaseIoBinding)(self.binding)
        };
    }
}

/// A scoped set of caller-owned values attached to an I/O binding.
pub struct BoundIoBinding<'binding, 'session> {
    binding: &'binding mut IoBinding<'session>,
    _borrows: PhantomData<&'binding mut ()>,
}

impl BoundIoBinding<'_, '_> {
    /// Executes and synchronizes the bindings once.
    pub fn run(&mut self) -> Result<()> {
        self.run_impl(None)
    }

    /// Executes bindings with explicit reusable run options.
    pub fn run_with_options(&mut self, options: &RunOptions) -> Result<()> {
        ensure!(
            options.belongs_to(self.binding.session),
            ForeignRunOptionsSnafu
        );
        self.run_impl(Some(options))
    }

    fn run_impl(&mut self, options: Option<&RunOptions>) -> Result<()> {
        let session = self.binding.session;
        let api = &session.environment.inner.api;
        // SAFETY: every bound input remains borrowed and live.
        status_call!(api, SynchronizeBoundInputs(self.binding.binding))?;
        // SAFETY: session, binding, options, and all values remain live.
        status_call!(
            api,
            RunWithBinding(
                session.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                self.binding.binding,
            )
        )?;
        // SAFETY: synchronize provider writes before exposing output storage.
        status_call!(api, SynchronizeBoundOutputs(self.binding.binding))?;
        Ok(())
    }
}

impl Drop for BoundIoBinding<'_, '_> {
    fn drop(&mut self) {
        self.binding.clear();
    }
}

pub(super) struct AsyncCompletion {
    api: sys::Api,
    expected_outputs: usize,
    inner: Mutex<AsyncCompletionInner>,
    completed: Condvar,
}

pub(super) struct AsyncCompletionInner {
    result: Option<Result<Vec<usize>>>,
    waker: Option<Waker>,
}

impl AsyncCompletion {
    pub(super) fn new(api: sys::Api, expected_outputs: usize) -> Self {
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

    pub(super) fn poll_result(&self, waker: &Waker) -> Option<Result<Vec<usize>>> {
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

    pub(super) fn wait_result(&self) -> Result<Vec<usize>> {
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

pub(super) unsafe extern "system" fn async_run_callback(
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

pub(super) fn async_callback_result(
    completion: &AsyncCompletion,
    outputs: *mut *mut sys::OrtValue,
    num_outputs: usize,
    status: sys::OrtStatusPtr,
) -> Result<Vec<usize>> {
    if !status.is_null() {
        release_callback_outputs(&completion.api, outputs, num_outputs);
        check_status(&completion.api, status)?;
        return Ok(Vec::new());
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

pub(super) fn collect_callback_outputs(
    outputs: *mut *mut sys::OrtValue,
    num_outputs: usize,
) -> Result<Vec<usize>> {
    if num_outputs == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !outputs.is_null(),
        NullPointerSnafu {
            what: "async output array"
        }
    );
    // SAFETY: ORT supplies num_outputs entries on a successful callback.
    let values = unsafe { std::slice::from_raw_parts(outputs, num_outputs) };
    ensure!(
        values.iter().all(|value| !value.is_null()),
        NullPointerSnafu {
            what: "async output value"
        }
    );
    Ok(values.iter().map(|value| *value as usize).collect())
}

pub(super) fn release_callback_outputs(
    api: &sys::Api,
    outputs: *mut *mut sys::OrtValue,
    num_outputs: usize,
) {
    if outputs.is_null() {
        return;
    }
    // SAFETY: callback supplied an array of num_outputs entries.
    let values = unsafe { std::slice::from_raw_parts(outputs, num_outputs) };
    for value in values.iter().copied().filter(|value| !value.is_null()) {
        // SAFETY: callback output ownership belongs to the caller.
        unsafe { api_function!(api, ReleaseValue)(value) };
    }
}
