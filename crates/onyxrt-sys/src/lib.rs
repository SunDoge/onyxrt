//! Low-level ONNX Runtime C ABI with dynamic-loading and static-link modes.

#[cfg(all(feature = "load-dynamic", feature = "link-static"))]
compile_error!("features `load-dynamic` and `link-static` are mutually exclusive");
#[cfg(not(any(feature = "load-dynamic", feature = "link-static")))]
compile_error!("enable exactly one of `load-dynamic` or `link-static`");

use std::{
    ffi::{CStr, OsStr},
    ptr::NonNull,
    sync::Arc,
};

#[cfg(feature = "load-dynamic")]
use std::{
    collections::HashSet,
    env,
    ffi::OsString,
    mem::ManuallyDrop,
    path::{Path, PathBuf},
};

#[cfg(feature = "load-dynamic")]
use libloading::Library;
use snafu::Snafu;

mod bindings;
pub use bindings::*;

/// Errors produced while loading the ONNX Runtime C API.
#[derive(Debug, Snafu)]
pub enum Error {
    /// A dynamic library could not be opened.
    #[cfg(feature = "load-dynamic")]
    #[snafu(display("failed to load ONNX Runtime library `{}`", path.display()))]
    LoadLibrary {
        /// Attempted path or loader name.
        path: PathBuf,
        /// Platform loader error.
        source: libloading::Error,
    },

    /// The library does not export `OrtGetApiBase`.
    #[cfg(feature = "load-dynamic")]
    #[snafu(display("`{}` does not export OrtGetApiBase", path.display()))]
    MissingEntryPoint {
        /// Loaded library path.
        path: PathBuf,
        /// Platform loader error.
        source: libloading::Error,
    },

    /// No candidate dynamic library could be loaded.
    #[cfg(feature = "load-dynamic")]
    #[snafu(display(
        "could not load ONNX Runtime after trying {attempted} candidate(s); last error: {source}"
    ))]
    NoUsableLibrary {
        /// Number of distinct loader names and paths tried.
        attempted: usize,
        /// Error from the final candidate.
        source: Box<Error>,
    },

    /// The entry point unexpectedly returned null.
    #[snafu(display("OrtGetApiBase returned null for {origin}"))]
    NullApiBase {
        /// Description of the linked or loaded runtime.
        origin: String,
    },

    /// An exact runtime path was requested in a statically linked build.
    #[snafu(display(
        "runtime library paths require the `load-dynamic` feature; this build uses `link-static`"
    ))]
    DynamicLoadingUnavailable,

    /// The runtime does not support the API version used to compile this crate.
    #[snafu(display(
        "ONNX Runtime {runtime_version} does not support C API version {api_version}"
    ))]
    UnsupportedApi {
        /// Requested C API version.
        api_version: u32,
        /// Runtime version string.
        runtime_version: String,
    },
}

/// Negotiated ONNX Runtime API table.
#[derive(Clone)]
pub struct Api {
    inner: Arc<ApiInner>,
}

struct ApiInner {
    api: NonNull<OrtApi>,
    version: String,
    origin: String,
    // ONNX Runtime owns process-global registries and worker infrastructure.
    // Unloading it during Rust static destruction is not supported reliably,
    // so retain a dynamically loaded module until exit without calling dlclose.
    #[cfg(feature = "load-dynamic")]
    _library: ManuallyDrop<Library>,
}

// SAFETY: the ONNX Runtime API table is immutable process-global data, and
// the retained library handle may be moved between threads.
unsafe impl Send for ApiInner {}
// SAFETY: the table is immutable and the ONNX Runtime C API permits concurrent
// use of its functions.
unsafe impl Sync for ApiInner {}

impl Api {
    /// Loads the runtime from an exact dynamic library path.
    #[cfg(feature = "load-dynamic")]
    pub fn load(path: impl AsRef<OsStr>) -> Result<Self, Error> {
        let path = PathBuf::from(path.as_ref());
        // SAFETY: the library is retained in ApiInner for the full lifetime of
        // every symbol and API-table pointer derived from it.
        let library = unsafe { Library::new(&path) }.map_err(|source| Error::LoadLibrary {
            path: path.clone(),
            source,
        })?;
        Self::from_library(library, path)
    }

    /// Reports that loading by path is unavailable in static-link mode.
    #[cfg(feature = "link-static")]
    pub fn load(_path: impl AsRef<OsStr>) -> Result<Self, Error> {
        Err(Error::DynamicLoadingUnavailable)
    }

    /// Acquires the configured runtime and negotiates C API v21.
    ///
    /// Dynamic mode checks `ORT_DYLIB_PATH` and platform loader names. Static
    /// mode calls the linked `OrtGetApiBase` symbol directly.
    pub fn load_default() -> Result<Self, Error> {
        #[cfg(feature = "link-static")]
        {
            // SAFETY: link-static guarantees that the ORT entry point is linked
            // into the final binary with the declaration supplied by ort-sys.
            let base = unsafe { OrtGetApiBase() };
            let origin = "the statically linked runtime".to_owned();
            let (api, version) = negotiate(base, &origin)?;
            Ok(Self {
                inner: Arc::new(ApiInner {
                    api,
                    version,
                    origin,
                }),
            })
        }

        #[cfg(feature = "load-dynamic")]
        if let Some(path) = env::var_os("ORT_DYLIB_PATH") {
            return Self::load(path);
        }

        #[cfg(feature = "load-dynamic")]
        {
            let mut candidates = Vec::new();
            for variable in ["ORT_LIB_PATH", "ORT_LIB_LOCATION"] {
                if let Some(value) = env::var_os(variable) {
                    let path = PathBuf::from(value);
                    if path.is_dir() {
                        candidates.extend(candidates_in_directory(&path));
                    } else {
                        return Self::load(path);
                    }
                }
            }
            candidates.extend(default_library_names().into_iter().map(PathBuf::from));
            for directory in dynamic_search_directories() {
                candidates.extend(candidates_in_directory(&directory));
            }

            let mut seen = HashSet::new();
            candidates.retain(|candidate| seen.insert(candidate.clone()));
            let mut last_error = None;
            let attempted = candidates.len();
            for candidate in candidates {
                match Self::load(candidate) {
                    Ok(api) => return Ok(api),
                    Err(error) => last_error = Some(error),
                }
            }
            Err(Error::NoUsableLibrary {
                attempted,
                source: Box::new(last_error.expect("each supported platform has a library name")),
            })
        }
    }

    #[cfg(feature = "load-dynamic")]
    fn from_library(library: Library, path: PathBuf) -> Result<Self, Error> {
        type GetApiBase = unsafe extern "system" fn() -> *const OrtApiBase;
        // SAFETY: the requested symbol has the signature defined by the ORT C API.
        let get_api_base =
            unsafe { library.get::<GetApiBase>(b"OrtGetApiBase\0") }.map_err(|source| {
                Error::MissingEntryPoint {
                    path: path.clone(),
                    source,
                }
            })?;
        // SAFETY: calling the stable ORT entry point has no preconditions.
        let base = unsafe { get_api_base() };
        let origin = format!("dynamic library `{}`", path.display());
        let (api, version) = negotiate(base, &origin)?;

        Ok(Self {
            inner: Arc::new(ApiInner {
                api,
                version,
                origin,
                _library: ManuallyDrop::new(library),
            }),
        })
    }

    /// Returns the loaded runtime's version string.
    pub fn version(&self) -> &str {
        &self.inner.version
    }

    /// Returns a human-readable description of where the runtime came from.
    pub fn origin(&self) -> &str {
        &self.inner.origin
    }

    /// Returns the immutable C API function table.
    pub fn table(&self) -> &OrtApi {
        // SAFETY: the configured runtime and its immutable API table outlive self.
        unsafe { self.inner.api.as_ref() }
    }
}

fn negotiate(base: *const OrtApiBase, origin: &str) -> Result<(NonNull<OrtApi>, String), Error> {
    let base = NonNull::new(base.cast_mut()).ok_or_else(|| Error::NullApiBase {
        origin: origin.to_owned(),
    })?;
    // SAFETY: base is non-null and points to the immutable process-global table.
    let base_ref = unsafe { base.as_ref() };
    // SAFETY: this returns a static NUL-terminated string owned by the runtime.
    let version_ptr = unsafe { (base_ref.GetVersionString)() };
    let version = if version_ptr.is_null() {
        "unknown".to_owned()
    } else {
        // SAFETY: guaranteed NUL-terminated and valid for the runtime lifetime.
        unsafe { CStr::from_ptr(version_ptr) }
            .to_string_lossy()
            .into_owned()
    };
    // SAFETY: requesting the compile-time API version is the prescribed ABI handshake.
    let api = unsafe { (base_ref.GetApi)(ORT_API_VERSION) };
    let api = NonNull::new(api.cast_mut()).ok_or_else(|| Error::UnsupportedApi {
        api_version: ORT_API_VERSION,
        runtime_version: version.clone(),
    })?;
    Ok((api, version))
}

#[cfg(feature = "load-dynamic")]
fn default_library_names() -> Vec<OsString> {
    if cfg!(target_os = "windows") {
        vec![OsString::from("onnxruntime.dll")]
    } else if cfg!(target_os = "macos") {
        vec![OsString::from("libonnxruntime.dylib")]
    } else {
        vec![
            OsString::from("libonnxruntime.so"),
            OsString::from("libonnxruntime.so.1"),
            OsString::from("libonnxruntime.so.1.21"),
            OsString::from("libonnxruntime.so.1.21.0"),
        ]
    }
}

#[cfg(feature = "load-dynamic")]
fn candidates_in_directory(directory: &Path) -> Vec<PathBuf> {
    let mut candidates = default_library_names()
        .into_iter()
        .map(|name| directory.join(name))
        .collect::<Vec<_>>();
    let Ok(entries) = std::fs::read_dir(directory) else {
        return candidates;
    };
    let mut versioned = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                return false;
            };
            if cfg!(target_os = "macos") {
                name.starts_with("libonnxruntime.") && name.ends_with(".dylib")
            } else if cfg!(target_os = "windows") {
                name.eq_ignore_ascii_case("onnxruntime.dll")
            } else {
                name.starts_with("libonnxruntime.so.")
            }
        })
        .collect::<Vec<_>>();
    versioned.sort_by(|left, right| right.file_name().cmp(&left.file_name()));
    candidates.extend(versioned);
    candidates
}

#[cfg(feature = "load-dynamic")]
fn dynamic_search_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    for variable in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"] {
        if let Some(paths) = env::var_os(variable) {
            directories.extend(env::split_paths(&paths));
        }
    }
    for root in [
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/lib"),
        PathBuf::from("/opt/onnxruntime/lib"),
    ] {
        directories.push(root.clone());
        if let Ok(entries) = std::fs::read_dir(&root) {
            directories.extend(
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| {
                        path.is_dir()
                            && path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .is_some_and(|name| name.contains(env::consts::ARCH))
                    }),
            );
        }
    }
    let mut seen = HashSet::new();
    directories.retain(|directory| seen.insert(directory.clone()));
    directories
}
