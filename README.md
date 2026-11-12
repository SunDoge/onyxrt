# onyxrt

Safe Rust bindings for the ONNX Runtime C API. The runtime is loaded dynamically,
so applications can use an official ONNX Runtime release or a library shipped in
a Python wheel without compiling ONNX Runtime themselves.

The workspace targets ONNX Runtime C API v21 (ONNX Runtime 1.21 or newer):

- `onyxrt-sys` contains the generated ABI definitions and dynamic loader.
- `onyxrt` provides environments, reusable session configuration, execution
  providers, values, I/O binding, and native asynchronous runs.
- DLPack interoperability through [`dlpark`](https://crates.io/crates/dlpark)
  is optional.

## Runtime loading

```toml
[dependencies]
onyxrt = "0.1"
```

Discovery checks `ORT_DYLIB_PATH` first, then `ORT_LIB_PATH` and
`ORT_LIB_LOCATION`, platform loader names, loader paths, and standard system
library directories. `Environment::runtime_origin()` reports the selected
library. Set `ORT_DYLIB_PATH` when an exact library must be used.

Official ONNX Runtime GitHub release archives are the preferred native
distribution. The `onnxruntime` and `onnxruntime-gpu` PyPI wheels are also
usable: keep the matching files in `onnxruntime/capi` together and point
`ORT_DYLIB_PATH` at `libonnxruntime.so.<version>` or `onnxruntime.dll`. CUDA and
TensorRT still require compatible NVIDIA driver and runtime libraries.

## Sessions and execution providers

`SessionBuilder` is reusable: each model load creates an independent immutable
native session while retaining the same provider and optimization settings.

```rust,no_run
use onyxrt::{
    model::Environment,
    provider::{CudaExecutionProvider, CudnnConvAlgoSearch},
};

# fn example() -> Result<(), Box<dyn std::error::Error>> {
let environment = Environment::builder().build()?;
let sessions = environment
    .session_builder()
    .intra_threads(4)
    .execution_provider(
        CudaExecutionProvider::new()
            .with_device_id(0)
            .with_cudnn_conv_algo_search(CudnnConvAlgoSearch::Heuristic),
    );

let first = sessions.build_from_file("first.onnx")?;
let second = sessions.build_from_file("second.onnx")?;
# let _ = (first, second);
# Ok(())
# }
```

Providers are attempted in the order passed to `execution_provider`; the
built-in CPU provider remains the final fallback. CUDA, TensorRT, oneDNN,
legacy OpenVINO, and generic named providers are supported. Named providers
cover APIs such as QNN, SNPE, XNNPACK, CoreML, and Azure where available in the
loaded runtime.

`build_from_file` uses ONNX Runtime's native path API, preserving relative
resolution for external-data models. Use `build_from_bytes` for in-memory
models. The builder also supports owned external-data files, external
initializers, custom-op libraries, LoRA adapters, profiling, optimized-model
output, memory-pattern control, deterministic compute, and arbitrary session
configuration entries.

## Values and inference

The core API accepts `value::Value`. This keeps tensor-framework policy outside
`Session` and gives all adapters one execution path.

```rust,ignore
use onyxrt::{
    model::Environment,
    value::{Input, MemoryInfo, Value},
};

let environment = Environment::builder().build()?;
let session = environment.session_builder().build_from_file("model.onnx")?;
let memory = MemoryInfo::cpu(environment.clone())?;
let mut data = [1.0_f32, 2.0, 3.0];
let value = Value::tensor(environment, &memory, &mut data, [3])?;
let inputs = [Input::new("input", &value)?];
let outputs = session.run(&inputs, &["output"])?;
let output: &[f32] = outputs[0].as_slice()?;
# let _ = output;
```

`Value::allocate_tensor` creates ORT-owned CPU tensors. A session `Allocator`
and `Value::allocate_tensor_in` cover provider allocations. Unsafe
`Value::from_raw_parts` is available for integrations that already own device
memory. COO, CSR, and block-sparse values are also supported.

Use `Session::run_into` for caller-owned preallocated outputs. For repeated
runs, bind the same values once:

```rust,ignore
let mut binding = session.io_binding()?;
let mut bound = binding.bind(&inputs, &mut outputs)?;
bound.run()?;
bound.run_with_options(&options)?;
```

## Asynchronous inference

`Session::run_async` wraps ONNX Runtime's native callback-based `RunAsync` as a
Rust future without requiring Tokio or another executor dependency:

```rust,ignore
let outputs = session.run_async(&inputs, &["output"])?.await?;
```

ORT schedules the work on its own thread pool. Dropping an unfinished future
requests cooperative cancellation and waits for the native callback before
releasing borrowed inputs. This is different from merely wrapping synchronous
inference in `spawn_blocking`: the submission and completion are native ORT
operations, although both approaches ultimately use background worker threads.

## Optional DLPack adapter

DLPack is not part of the default feature set:

```toml
[dependencies]
onyxrt = { version = "0.1", features = ["dlpack", "dlpack-ndarray"] }
```

`dlpack::Value` validates and adapts a compact CPU or CUDA tensor without a
copy, then exposes the same value consumed by `Session`:

```rust,ignore
use onyxrt::{dlpack, value::Input};

let value = dlpack::Value::from_managed(environment.clone(), &managed)?;
let inputs = [Input::new("input", value.as_value())?];
let outputs = session.run(&inputs, &["output"])?;
```

Framework support is controlled independently with `dlpack-candle`,
`dlpack-cudarc`, `dlpack-half`, `dlpack-image`, `dlpack-ndarray`,
`dlpack-pyo3`, and `dlpack-safetensors`. These forward only the corresponding
`dlpark` features. The `dlpark` `cpu-all` aggregate is intentionally not
exposed because it is a test convenience rather than an onyxrt API choice.

## Provider integration tests

Native integration tests are ignored by default because they require a matching
runtime, provider dependencies, and sometimes hardware:

```text
ONYXRT_TEST_DYLIB=/path/libonnxruntime.so cargo test -p onyxrt --test cpu_identity cpu_value_identity_round_trip -- --ignored
ONYXRT_TEST_DYLIB=/path/libonnxruntime.so cargo test -p onyxrt --features dlpack,dlpack-ndarray --test cpu_identity cpu_dlpack_identity_round_trip -- --ignored
ONYXRT_TEST_DNNL=1 cargo test -p onyxrt --test cpu_identity dnnl_prebuilt_provider_creates_session -- --ignored
ONYXRT_TEST_CUDA=1 ONYXRT_TEST_DYLIB=/path/libonnxruntime.so cargo test -p onyxrt --test cpu_identity cuda_prebuilt_provider_creates_session -- --ignored
ONYXRT_TEST_TENSORRT=1 ONYXRT_TEST_DYLIB=/path/libonnxruntime.so cargo test -p onyxrt --test cpu_identity tensorrt_prebuilt_provider_creates_session -- --ignored
```

## Updating the C bindings

Generated bindings are committed so downstream builds do not need Clang. The
`usage-rs` based workspace task downloads the pinned ONNX Runtime header and
regenerates bindings:

```text
cargo xtask download-header
cargo xtask bindgen
cargo xtask download-header --check
cargo xtask bindgen --check
```

## License

onyxrt is licensed under the MIT License. Generated ONNX Runtime bindings
retain Microsoft's copyright notice and MIT license.
