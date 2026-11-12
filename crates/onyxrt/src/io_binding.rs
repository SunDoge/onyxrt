use std::{
    ffi::CString,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    ptr,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Waker},
};

use onyxrt_sys as sys;
use snafu::{Snafu, ensure};

use crate::{
    ffi::{
        self, IoBindingGuard, MemoryInfoGuard, ValueGuard, api_function, check_status, status_call,
    },
    model::Environment,
    session::{RunOptions, Session},
    value::{
        self, Input, IoBindingInput, IoBindingOutput, Output, create_dlpack_memory_info,
        create_external_tensor, output_to_dlpack,
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
    Value { source: value::Error },

    #[snafu(transparent)]
    Session { source: crate::session::Error },

    #[snafu(display("run options belong to a different ONNX Runtime environment"))]
    ForeignRunOptions,

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
/// Call [`Self::bind`] to create a scoped set of external DLPack bindings. The
/// native object itself can be rebound after the scoped binding is dropped.
pub struct IoBinding<'session> {
    pub(super) session: &'session Session,
    pub(super) binding: *mut sys::OrtIoBinding,
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
                status_call!(
                    api,
                    BindInput(self.binding, input.name.as_ptr(), value.cast_const(),)
                )?;
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
                status_call!(
                    api,
                    BindOutput(self.binding, output.name.as_ptr(), value.cast_const(),)
                )?;
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
        ensure!(
            options.belongs_to(self.binding.session),
            ForeignRunOptionsSnafu
        );
        self.run_impl(Some(options))
    }

    fn run_impl(&mut self, options: Option<&RunOptions>) -> Result<()> {
        let session = self.binding.session;
        let api = &session.environment.inner.api;
        // SAFETY: all external values remain retained by self.
        status_call!(api, SynchronizeBoundInputs(self.binding.binding))?;
        // SAFETY: the session, native binding, options, and external values live
        // through the synchronous call.
        status_call!(
            api,
            RunWithBinding(
                session.session as *mut sys::OrtSession,
                options.map_or(ptr::null(), RunOptions::as_ptr),
                self.binding.binding,
            )
        )?;
        // SAFETY: wait until all provider writes to external outputs complete.
        status_call!(api, SynchronizeBoundOutputs(self.binding.binding))?;
        Ok(())
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
    pub(super) session: &'a Session,
    // Declared before values and memory info so normal field destruction
    // releases binding-held references first.
    pub(super) binding: Option<IoBindingGuard<'a>>,
    pub(super) _values: Vec<ValueGuard<'a>>,
    pub(super) _memory_infos: Vec<MemoryInfoGuard<'a>>,
    pub(super) _borrows: PhantomData<&'a mut ()>,
    pub(super) synchronized: bool,
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
        status_call!(api, SynchronizeBoundOutputs(binding))?;
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
        let status = unsafe { api_function!(api, SynchronizeBoundOutputs)(binding) };
        if !status.is_null() {
            // Drop cannot report synchronization errors, but must release the
            // owned status object.
            // SAFETY: a non-null status is owned by the caller.
            unsafe { api_function!(api, ReleaseStatus)(status) };
        }
        self.synchronized = true;
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

/// Future returned by [`Session::run_all_async`].
#[must_use = "futures do nothing unless awaited or polled"]
pub struct RunFuture<'a> {
    pub(super) session: &'a Session,
    pub(super) environment: Environment,
    pub(super) completion: Arc<AsyncCompletion>,
    pub(super) names: Vec<String>,
    pub(super) _input_values: Vec<ValueGuard<'a>>,
    pub(super) _memory_info: MemoryInfoGuard<'a>,
    pub(super) run_options: RunOptions,
    pub(super) _input_names: Vec<CString>,
    pub(super) input_name_ptrs: Vec<*const std::ffi::c_char>,
    pub(super) input_value_ptrs: Vec<*const sys::OrtValue>,
    pub(super) _output_names: Vec<CString>,
    pub(super) output_name_ptrs: Vec<*const std::ffi::c_char>,
    pub(super) output_slots: Vec<*mut sys::OrtValue>,
    pub(super) _input_borrow: PhantomData<&'a [Input<'a>]>,
    pub(super) finished: bool,
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
        self.run_options.terminate()?;
        Ok(())
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
                    api_function!(self.session.environment.inner.api, ReleaseValue)(
                        value as *mut sys::OrtValue,
                    )
                };
            }
        }
        self.finished = true;
    }
}
