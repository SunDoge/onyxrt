use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
};

use sha2::{Digest, Sha256};
use usage_rs::{Args, Cli, Subcommands};

const ORT_VERSION: &str = "1.31.1";
const HEADER_URL: &str = "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.1/include/onnxruntime/core/session/onnxruntime_c_api.h";
const HEADER_BASE_URL: &str = "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.1/include/onnxruntime/core/session";
const HEADERS: &[(&str, &str)] = &[
    (
        "onnxruntime_c_api.h",
        "43d584f596ef66820421317df32a79a95fc1d39ee46cb3971a870c08d8ba4baa",
    ),
    (
        "onnxruntime_error_code.h",
        "5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721",
    ),
    (
        "onnxruntime_ep_c_api.h",
        "b64036772115dbc6f06a900487ce22d34e862100db03e2ad62210758e27a19b2",
    ),
];

/// Maintain generated source files for onyxrt.
#[derive(Cli)]
#[usage(bin = "cargo xtask", version)]
struct Application {
    /// Maintenance task to run.
    #[usage(subcommand)]
    command: Command,
}

#[derive(Subcommands)]
enum Command {
    /// Download the pinned ONNX Runtime C API header.
    DownloadHeader(DownloadHeader),
    /// Generate Rust FFI bindings from the vendored header.
    Bindgen(GenerateBindings),
}

#[derive(Args)]
struct DownloadHeader {
    /// Verify the vendored header without changing it.
    #[usage(long)]
    check: bool,
}

#[derive(Args)]
struct GenerateBindings {
    /// Verify the committed bindings without changing them.
    #[usage(long)]
    check: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let application = Application::parse();
    match application.command {
        Command::DownloadHeader(arguments) => download_header(arguments.check),
        Command::Bindgen(arguments) => generate_bindings(arguments.check),
    }
}

fn download_header(check: bool) -> Result<(), Box<dyn Error>> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("onyxrt-xtask/", env!("CARGO_PKG_VERSION")))
        .build()?;
    for &(name, expected_hash) in HEADERS {
        let contents = client
            .get(format!("{HEADER_BASE_URL}/{name}"))
            .send()?
            .error_for_status()?
            .bytes()?;
        let actual_hash = format!("{:x}", Sha256::digest(&contents));
        if actual_hash != expected_hash {
            return Err(format!(
                "downloaded ONNX Runtime v{ORT_VERSION} {name} has SHA-256 {actual_hash}; expected {expected_hash}"
            )
            .into());
        }

        let destination = header_directory().join(name);
        if check {
            check_contents(&destination, &contents, "ONNX Runtime header")?;
        } else if write_if_changed(&destination, &contents)? {
            println!("downloaded {name} to {}", destination.display());
        }
    }
    println!("vendored ONNX Runtime v{ORT_VERSION} header bundle is current");
    Ok(())
}

fn generate_bindings(check: bool) -> Result<(), Box<dyn Error>> {
    let header = header_path();
    if !header.is_file() {
        return Err(format!(
            "{} is missing; run `cargo xtask download-header` first",
            header.display()
        )
        .into());
    }
    for &(name, expected_hash) in HEADERS {
        verify_file_hash(
            &header_directory().join(name),
            expected_hash,
            "vendored ONNX Runtime header",
        )?;
    }

    // ORTCHAR_T is a platform-dependent macro in the upstream header. If
    // bindgen expands it while running on Linux, path-taking APIs are emitted
    // permanently as `*const c_char` and fail to compile on Windows. Turn the
    // two conditional macro definitions into a real typedef so blocklisting
    // preserves ORTCHAR_T in every function signature.
    let header_contents = fs::read_to_string(&header)?
        .replace("#define ORTCHAR_T wchar_t", "typedef wchar_t ORTCHAR_T;")
        .replace("#define ORTCHAR_T char", "typedef char ORTCHAR_T;");
    let bindings = bindgen::Builder::default()
        .header_contents("onnxruntime_c_api.h", &header_contents)
        .clang_arg(format!("-I{}", header_directory().display()))
        .clang_arg("-DORT_API_MANUAL_INIT")
        .allowlist_function("OrtGetApiBase")
        .allowlist_type("(ONNX|Ort|GraphOptimizationLevel|ExecutionMode).*")
        .allowlist_var("(ONNX|ORT)_.*")
        .blocklist_type("ORTCHAR_T")
        .raw_line("#[cfg(target_os = \"windows\")]")
        .raw_line("pub type ORTCHAR_T = u16;")
        .raw_line("#[cfg(not(target_os = \"windows\"))]")
        .raw_line("pub type ORTCHAR_T = ::core::ffi::c_char;")
        .raw_line("pub type os_char = ORTCHAR_T;")
        .newtype_enum("(ONNX|Ort|GraphOptimizationLevel|ExecutionMode).*")
        .derive_debug(true)
        .derive_copy(true)
        .derive_default(false)
        .layout_tests(false)
        .override_abi(bindgen::Abi::System, ".*")
        .formatter(bindgen::Formatter::Prettyplease)
        .generate_comments(true)
        .generate()
        .map_err(|error| format!("failed to generate ONNX Runtime bindings: {error}"))?;

    let bindings = bindings.to_string();
    let generated = format!(
        "// @generated by `cargo xtask bindgen`; do not edit by hand.\n\
         // Source: ONNX Runtime v{ORT_VERSION} ({HEADER_URL})\n\n\
         // Copyright (c) Microsoft Corporation. Licensed under the MIT License.\n\n\
         #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]\n\
         #![allow(\n\
             clippy::doc_lazy_continuation,\n\
             clippy::missing_safety_doc,\n\
             clippy::tabs_in_doc_comments,\n\
             clippy::type_complexity\n\
         )]\n\n\
         {bindings}\n"
    );
    let generated = format_rust_source(&generated)?;

    let destination = bindings_path();
    if check {
        check_contents(&destination, generated.as_bytes(), "generated bindings")?;
        println!("generated ONNX Runtime bindings are current");
    } else if write_if_changed(&destination, generated.as_bytes())? {
        println!("generated bindings at {}", destination.display());
    } else {
        println!("generated ONNX Runtime bindings are already current");
    }
    Ok(())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be located directly under the workspace root")
        .to_owned()
}

fn header_path() -> PathBuf {
    header_directory().join("onnxruntime_c_api.h")
}

fn header_directory() -> PathBuf {
    workspace_root().join("vendor/onnxruntime")
}

fn bindings_path() -> PathBuf {
    workspace_root().join("crates/onyxrt-sys/src/bindings.rs")
}

fn format_rust_source(source: &str) -> Result<String, Box<dyn Error>> {
    let temporary_directory = workspace_root().join("target/xtask");
    fs::create_dir_all(&temporary_directory)?;
    let temporary = temporary_directory.join(format!("bindings-{}.rs", std::process::id()));
    fs::write(&temporary, source)?;
    let status = ProcessCommand::new("rustfmt")
        .args(["--edition", "2024"])
        .arg(&temporary)
        .status()
        .map_err(|error| format!("failed to execute rustfmt: {error}"))?;
    if !status.success() {
        return Err(format!("rustfmt failed while formatting {}", temporary.display()).into());
    }
    let formatted = fs::read_to_string(&temporary)?;
    fs::remove_file(temporary)?;
    Ok(formatted)
}

fn verify_file_hash(path: &Path, expected: &str, description: &str) -> Result<(), Box<dyn Error>> {
    let contents = fs::read(path)?;
    let actual = format!("{:x}", Sha256::digest(contents));
    if actual != expected {
        return Err(format!(
            "{description} at {} has SHA-256 {actual}; expected {expected}; run `cargo xtask download-header`",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn check_contents(path: &Path, expected: &[u8], description: &str) -> Result<(), Box<dyn Error>> {
    let actual = fs::read(path).map_err(|error| {
        format!(
            "cannot read {description} at {}: {error}; regenerate it without `--check`",
            path.display()
        )
    })?;
    if actual != expected {
        return Err(format!(
            "{description} at {} is stale; regenerate it without `--check`",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn write_if_changed(path: &Path, contents: &[u8]) -> Result<bool, Box<dyn Error>> {
    if fs::read(path).is_ok_and(|current| current == contents) {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    Ok(true)
}
