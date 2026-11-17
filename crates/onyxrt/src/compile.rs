//! Ahead-of-time model compilation for execution providers that emit EPContext models.

use std::{ptr, ptr::NonNull, slice};

use onyxrt_sys as sys;
use snafu::Snafu;

use crate::{
    ffi::{self, AllocatedBuffer, api_function, check_status, default_allocator, require_non_null},
    provider::{self, NativeModelPath},
    session::{self, SessionBuilder},
};

macro_rules! compile_status_call {
    ($compiler:expr, $table:expr, $name:ident($($argument:expr),* $(,)?)) => {{
        match compile_function($table.$name, stringify!($name)) {
            Ok(function) => {
                // SAFETY: this module-private macro is used only where each
                // call site documents and establishes the native preconditions.
                let status = unsafe { function($($argument),*) };
                check_status(&$compiler.session.environment.inner.api, status).map_err(Error::from)
            }
            Err(error) => Err(error),
        }
    }};
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Ffi {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(transparent)]
    Session { source: session::Error },

    #[snafu(transparent)]
    Provider { source: provider::Error },

    #[snafu(display("ONNX Runtime compile API function `{name}` is unavailable"))]
    MissingFunction { name: &'static str },

    #[snafu(display(
        "in-memory compilation requires embedded EP context data; use compile_file for external provider data"
    ))]
    EmbeddedContextRequired,
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
    }
}

/// Controls which initializer weights are omitted from a compiled EPContext model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WeightlessMode {
    /// Embed weights normally.
    #[default]
    Disabled,
    /// Omit only initializers that were already external in the source model.
    ExternalInitializers,
    /// Omit all initializers and obtain them from the source model at runtime.
    AllInitializers,
}

impl From<WeightlessMode> for sys::OrtWeightlessSupport {
    fn from(value: WeightlessMode) -> Self {
        match value {
            WeightlessMode::Disabled => Self::OrtWeightlessSupport_NONE,
            WeightlessMode::ExternalInitializers => Self::OrtWeightlessSupport_EXTERNAL_ONLY,
            WeightlessMode::AllInitializers => Self::OrtWeightlessSupport_ALL,
        }
    }
}

/// Reusable model-compilation configuration derived from a [`SessionBuilder`].
///
/// Provider settings are copied from the session builder. Compilation is only
/// supported by providers that implement ONNX Runtime's EPContext compile API.
#[derive(Clone)]
pub struct ModelCompiler {
    session: SessionBuilder,
    weightless_mode: WeightlessMode,
    embed_ep_context: bool,
    fail_if_no_nodes_compiled: bool,
    fail_if_output_exists: bool,
}

impl SessionBuilder {
    /// Creates a reusable compiler with the same execution-provider settings.
    pub fn compiler(&self) -> ModelCompiler {
        ModelCompiler {
            session: self.clone(),
            weightless_mode: WeightlessMode::Disabled,
            embed_ep_context: true,
            fail_if_no_nodes_compiled: false,
            fail_if_output_exists: false,
        }
    }
}

impl ModelCompiler {
    /// Selects whether compiled EPContext nodes omit initializer weights.
    pub fn weightless_mode(mut self, mode: WeightlessMode) -> Self {
        self.weightless_mode = mode;
        self
    }

    /// Selects whether provider context data is embedded in the ONNX output.
    ///
    /// Embedding is enabled by default so byte output is self-contained.
    pub fn embed_ep_context(mut self, embed: bool) -> Self {
        self.embed_ep_context = embed;
        self
    }

    /// Makes compilation fail when the provider compiles no graph nodes.
    pub fn fail_if_no_nodes_compiled(mut self, enabled: bool) -> Self {
        self.fail_if_no_nodes_compiled = enabled;
        self
    }

    /// Makes file output fail instead of replacing an existing file.
    pub fn fail_if_output_exists(mut self, enabled: bool) -> Self {
        self.fail_if_output_exists = enabled;
        self
    }

    /// Compiles a model file to an EPContext ONNX model file.
    pub fn compile_file(
        &self,
        input: impl AsRef<std::path::Path>,
        output: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        let input = NativeModelPath::new(input.as_ref())?;
        let output = NativeModelPath::new(output.as_ref())?;
        let options = self.create_options()?;
        let table = options.table;

        // SAFETY: options and the native path remain live through CompileModel.
        compile_status_call!(
            self,
            table,
            ModelCompilationOptions_SetInputModelPath(options.as_ptr(), input.as_ptr())
        )?;
        // SAFETY: options and the native path remain live through CompileModel.
        compile_status_call!(
            self,
            table,
            ModelCompilationOptions_SetOutputModelPath(options.as_ptr(), output.as_ptr())
        )?;
        self.compile(&options)
    }

    /// Compiles in-memory ONNX bytes and returns the serialized EPContext model.
    ///
    /// Provider context data is embedded in the returned ONNX buffer. In a
    /// weightless mode, the source model or its external weights are still
    /// required when loading the compiled model.
    pub fn compile_bytes(&self, input: &[u8]) -> Result<Vec<u8>> {
        if !self.embed_ep_context {
            return Err(Error::EmbeddedContextRequired);
        }
        let options = self.create_options()?;
        let table = options.table;
        // SAFETY: the input bytes remain live through CompileModel.
        compile_status_call!(
            self,
            table,
            ModelCompilationOptions_SetInputModelFromBuffer(
                options.as_ptr(),
                input.as_ptr().cast(),
                input.len()
            )
        )?;

        let api = &self.session.environment.inner.api;
        let allocator = default_allocator(api)?;
        let mut output = ptr::null_mut();
        let mut output_len = 0;
        // SAFETY: allocator has process lifetime and both output locations
        // remain live through CompileModel.
        compile_status_call!(
            self,
            table,
            ModelCompilationOptions_SetOutputModelBuffer(
                options.as_ptr(),
                allocator,
                &mut output,
                &mut output_len
            )
        )?;
        self.compile(&options)?;

        let output = require_non_null(output.cast::<u8>(), "compiled model buffer")?;
        let allocation = AllocatedBuffer {
            api,
            allocator,
            allocation: output.cast(),
        };
        // SAFETY: ORT allocated output_len initialized bytes at output and the
        // allocation guard remains live while they are copied.
        let bytes = unsafe { slice::from_raw_parts(output, output_len) }.to_vec();
        drop(allocation);
        Ok(bytes)
    }

    fn create_options(&self) -> Result<CompilationOptions<'_>> {
        let api = &self.session.environment.inner.api;
        // SAFETY: the negotiated v32 API table owns a process-lifetime compile table.
        let table = unsafe { api_function!(api, GetCompileApi)() };
        let table = require_non_null(table.cast_mut(), "compile API table")?;
        // SAFETY: the table pointer is non-null and owned by the loaded runtime.
        let table = unsafe { &*table };
        let session_options = self.session.create_options()?;
        let mut options = ptr::null_mut();
        let create = compile_function(
            table.CreateModelCompilationOptionsFromSessionOptions,
            "CreateModelCompilationOptionsFromSessionOptions",
        )?;
        // SAFETY: environment and session options are live; options is writable.
        let status = unsafe {
            create(
                self.session.environment.inner.env as *const sys::OrtEnv,
                session_options.as_ptr(),
                &mut options,
            )
        };
        check_status(api, status)?;
        let options = CompilationOptions {
            table,
            pointer: NonNull::new(require_non_null(options, "model compilation options")?)
                .expect("require_non_null returned a non-null pointer"),
        };

        // SAFETY: options is live and the graph optimization enum is valid.
        compile_status_call!(
            self,
            table,
            ModelCompilationOptions_SetGraphOptimizationLevel(
                options.as_ptr(),
                self.session.optimization.into()
            )
        )?;
        // SAFETY: options is live.
        compile_status_call!(
            self,
            table,
            ModelCompilationOptions_SetEpContextEmbedMode(options.as_ptr(), self.embed_ep_context)
        )?;
        // SAFETY: options is live and only selectable weightless modes are exposed.
        compile_status_call!(
            self,
            table,
            ModelCompilationOptions_SetWeightlessMode(
                options.as_ptr(),
                self.weightless_mode.into()
            )
        )?;

        let mut flags = 0;
        if self.fail_if_no_nodes_compiled {
            flags |= sys::OrtCompileApiFlags::OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED.0;
        }
        if self.fail_if_output_exists {
            flags |= sys::OrtCompileApiFlags::OrtCompileApiFlags_ERROR_IF_OUTPUT_FILE_EXISTS.0;
        }
        if flags != 0 {
            // SAFETY: options is live and flags contains only documented bits.
            compile_status_call!(
                self,
                table,
                ModelCompilationOptions_SetFlags(options.as_ptr(), flags)
            )?;
        }
        Ok(options)
    }

    fn compile(&self, options: &CompilationOptions<'_>) -> Result<()> {
        // SAFETY: environment and fully configured options remain live for the call.
        compile_status_call!(
            self,
            options.table,
            CompileModel(
                self.session.environment.inner.env as *const sys::OrtEnv,
                options.as_ptr()
            )
        )
    }
}

struct CompilationOptions<'api> {
    table: &'api sys::OrtCompileApi,
    pointer: NonNull<sys::OrtModelCompilationOptions>,
}

impl CompilationOptions<'_> {
    fn as_ptr(&self) -> *mut sys::OrtModelCompilationOptions {
        self.pointer.as_ptr()
    }
}

impl Drop for CompilationOptions<'_> {
    fn drop(&mut self) {
        if let Some(release) = self.table.ReleaseModelCompilationOptions {
            // SAFETY: this guard uniquely owns the native options pointer.
            unsafe { release(self.pointer.as_ptr()) };
        }
    }
}

fn compile_function<T: Copy>(function: Option<T>, name: &'static str) -> Result<T> {
    function.ok_or(Error::MissingFunction { name })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_only_selectable_weightless_modes() {
        assert_eq!(
            sys::OrtWeightlessSupport::from(WeightlessMode::Disabled),
            sys::OrtWeightlessSupport::OrtWeightlessSupport_NONE
        );
        assert_eq!(
            sys::OrtWeightlessSupport::from(WeightlessMode::AllInitializers),
            sys::OrtWeightlessSupport::OrtWeightlessSupport_ALL
        );
    }
}
