# onyxrt

Rust bindings for the ONNX Runtime C API, with SNAFU errors and DLPack tensor
interoperability through [`dlpark`](https://crates.io/crates/dlpark).

The workspace targets ONNX Runtime C API v21 (ONNX Runtime 1.21 or newer) and
supports CPU, CUDA, TensorRT, oneDNN, legacy OpenVINO, and named providers:

- `onyxrt-sys`: generated ABI definitions and dynamic-library discovery/loading.
- `onyxrt`: safe environment/session APIs, execution-provider configuration,
  and zero-copy CPU DLPack exchange.

## Runtime loading

ONNX Runtime is loaded dynamically through `libloading`. Discovery checks
`ORT_DYLIB_PATH` first, then `ORT_LIB_PATH`/
`ORT_LIB_LOCATION`, platform loader names, versioned libraries in loader paths,
and standard system library directories. `Environment::runtime_origin()`
reports the selected library. Set `ORT_DYLIB_PATH` to force an exact path.

```toml
onyxrt = "0.1"
```

## Execution providers

The built-in CPU provider is always the fallback. Accelerator providers are
appended in the order their builder methods are called, with the first having
the highest priority:

```rust,no_run
use onyxrt::{
    model::Environment,
    provider::{
        CudaArenaExtendStrategy, CudaExecutionProvider, CudnnConvAlgoSearch,
    },
};

# fn example() -> Result<(), Box<dyn std::error::Error>> {
let env = Environment::builder().build()?;
let session = env
    .session_builder()
    .cuda_provider(
        CudaExecutionProvider::new()
            .with_device_id(0)
            .with_arena_extend_strategy(CudaArenaExtendStrategy::NextPowerOfTwo)
            .with_cudnn_conv_algo_search(CudnnConvAlgoSearch::Heuristic)
            .with_copy_in_default_stream(true),
    )
    .build_from_file("model.onnx")?;
# let _ = session;
# Ok(())
# }
```

`TensorRtExecutionProvider`, `DnnlExecutionProvider`, and
`OpenVinoExecutionProvider` use the corresponding provider-options APIs.
`NamedExecutionProvider` covers providers exposed through ORT's generic API,
including QNN, SNPE, XNNPACK, CoreML, and Azure in C API v21.

For device-resident tensors, `IoBindingInput` and `IoBindingOutput` bind
preallocated CPU or CUDA DLPack storage directly through `OrtIoBinding`:

```rust,no_run
use onyxrt::{
    session::Session,
    value::{IoBindingInput, IoBindingOutput},
};

fn run_on_cuda(
    session: &Session,
    input: &dlpark::versioned::Dlpack,
    output: &mut dlpark::versioned::Dlpack,
) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = [IoBindingInput::from_dlpack("input", input)?];
    let mut outputs = [IoBindingOutput::from_dlpack("output", output)?];
    session.run_with_iobinding(&inputs, &mut outputs)
}
```

The DLPack tensors must be compact and outputs must be preallocated with the
exact model shape and dtype. CUDA pointers remain opaque to Rust; ONNX Runtime
receives them as `Cuda` device allocations and synchronizes bound inputs and
outputs around the run. `Session::run_all` and `Session::run_into` remain the
simpler CPU-only paths.

## Low-level values

`onyxrt::raw` exposes RAII wrappers around `OrtMemoryInfo` and `OrtValue` for
framework integrations that do not use DLPack. Typed CPU slices are checked
against their shape and element type, while `Value::from_raw_parts` is the
unsafe escape hatch for device memory or custom allocators:

```rust,ignore
use onyxrt::{
    model::Environment,
    raw::{Input, MemoryInfo, Value},
};

let environment = Environment::builder().build()?;
let session = environment
    .session_builder()
    .build_from_file("model.onnx")?;
let memory = MemoryInfo::cpu(environment.clone())?;
let mut data = [1.0_f32, 2.0, 3.0];
let value = Value::tensor(environment.clone(), &memory, &mut data, [3])?;
let inputs = [Input::new("input", &value)?];
let outputs = session.run_raw(&inputs, &["output"])?;
let output: &[f32] = outputs[0].as_slice()?;
# let _ = output;
```

`Value::allocate_tensor` creates an ORT-owned CPU tensor. For provider memory,
create a session `Allocator` from a `MemoryInfo` and use
`Value::allocate_tensor_in`; the resulting value is lifetime-bound to the
allocator so the native allocator cannot be released too early. Memory
descriptors expose their name, allocator type, device id, memory type, and
CPU/GPU/FPGA device category.

The raw layer also supports allocator-owned COO, CSR, and block-sparse tensors.
Sparse metadata is safe to query, but borrowing sparse values or indices as a
Rust slice is `unsafe`: ONNX Runtime 1.21 does not expose the sparse allocation's
`OrtMemoryInfo`, so callers must know that the allocator is CPU-accessible.

Existing raw values can be attached to a reusable `IoBinding` with
`IoBinding::bind_raw`. The returned scoped binding retains input and output
borrows, synchronizes provider work, and may be executed repeatedly.

For direct extensions, `Environment::raw_api`, `Session::as_ptr`,
`RunOptions::as_ptr`, `MemoryInfo::as_ptr`, and `Value::{as_ptr, as_mut_ptr,
into_raw}` expose the underlying handles without transferring ownership unless
explicitly documented.

For dynamic output shapes, use `IoBindingDeviceOutput::cpu` or
`IoBindingDeviceOutput::cuda` with `Session::run_with_iobinding_outputs`.
ONNX Runtime allocates each output on the requested device and onyxrt returns
it as an owning DLPack tensor.

## Asynchronous runs

`Session::run_all_async`, `Session::run_values_async`, and
`Session::run_raw_async` wrap ONNX Runtime's native callback-based `RunAsync`
as Rust futures without requiring Tokio or another runtime dependency:

```rust,ignore
let outputs = session.run_all_async(&inputs)?.await?;
let raw_outputs = session.run_raw_async(&raw_inputs, &["output"])?.await?;
```

`run_values_async` supports the same numeric, string, sequence, and map values
as synchronous `run_values`.

ORT schedules this work on its own intra-op thread pool. Dropping an unfinished
future requests cooperative cancellation and then waits for its native callback
so borrowed DLPack storage cannot be freed while ORT is still reading it.
`RunFuture::cancel` requests cancellation explicitly while still allowing the
caller to await and inspect the final ORT status.

For pre-bound CUDA buffers, `submit_with_iobinding` disables the provider's
end-of-run synchronization and returns a guard that retains all tensor borrows.
This is deferred synchronization rather than a native `Future`:

```rust,ignore
let pending = session.submit_with_iobinding(&inputs, &mut outputs)?;
// CPU work that does not touch the bound buffers can run here.
pending.wait()?;
```

Dropping `pending` also synchronizes for memory safety, but `wait()` is
preferred because it reports synchronization errors. CPU fallback nodes,
device copies, CUDA Graph behavior, or provider internals can still make the
submission call block.

Official GitHub release archives are the preferred native distribution. The
`onnxruntime` and `onnxruntime-gpu` PyPI wheels can also be unzipped: their
`onnxruntime/capi` directory contains the core library and matching provider
libraries. Keep all matching files together and point `ORT_DYLIB_PATH` at the
core `libonnxruntime.so.<version>` (or `onnxruntime.dll`). CUDA and TensorRT
still require compatible NVIDIA driver/runtime libraries.

Provider integration tests are opt-in because they require matching hardware
and native dependencies:

```text
ONYXRT_TEST_DNNL=1 cargo test -p onyxrt --test cpu_identity -- --ignored
ONYXRT_TEST_CUDA=1 ONYXRT_TEST_DYLIB=/path/libonnxruntime.so cargo test -p onyxrt --test cpu_identity -- --ignored
ONYXRT_TEST_TENSORRT=1 ONYXRT_TEST_DYLIB=/path/libonnxruntime.so cargo test -p onyxrt --test cpu_identity -- --ignored
```

```rust,no_run
use onyxrt::{
    model::Environment,
    provider::CpuExecutionProvider,
    value::Input,
};

# fn example(input: &dlpark::versioned::Dlpack) -> Result<(), Box<dyn std::error::Error>> {
let env = Environment::builder().name("example").build()?;
let session = env
    .session_builder()
    .cpu_provider(CpuExecutionProvider::new().with_arena(true))
    .build_from_file("model.onnx")?;
let inputs = [Input::from_dlpack("input", input)?];
let outputs = session.run_all(&inputs)?;
let output: &dlpark::versioned::Dlpack = outputs[0].tensor();
# Ok(())
# }
```

`SessionBuilder::build_from_file` uses ONNX Runtime's native path API instead
of reading the model into memory. This preserves relative resolution for ONNX
external-data files. Use `build_from_bytes` for genuinely in-memory models.

Caller-owned writable DLPack tensors can also be bound as preallocated outputs
with `OutputBuffer` and `Session::run_into`. ONNX Runtime writes directly into
the framework allocation without an intermediate output allocation or copy.

Run-specific logging, provider configuration, and cooperative termination use
reusable `RunOptions` created by `Session::run_options`:

```rust,ignore
let mut options = session.run_options()?;
options
    .set_tag("request-42")?
    .set_log_level(onyxrt::provider::LogLevel::Warning)?;
let outputs = session.run_all_with_options(&inputs, &options)?;
```

For repeated preallocated runs, reuse the native binding and its bound values:

```rust,ignore
let mut binding = session.io_binding()?;
let mut bound = binding.bind(&inputs, &mut outputs)?;
bound.run()?;
bound.run_with_options(&options)?;
```

`SessionBuilder` additionally supports memory-pattern control, profiling,
optimized-model output, deterministic compute, session logging, and arbitrary
ORT session configuration entries. Custom-op shared libraries can be added with
`SessionBuilder::custom_ops_library`.

Models using ONNX external data can supply owned in-memory files with
`SessionBuilder::external_data_file`. To replace named external initializers
without copying their tensor buffers, construct `session::ExternalInitializer`
values from `raw::Value` and call
`build_from_{file,bytes}_with_external_initializers`; the borrows only need to
remain valid until session creation returns.

LoRA adapters can be loaded with `LoraAdapter::from_file` or
`LoraAdapter::from_bytes`, then moved into `RunOptions::add_lora_adapter`.
`LoraAdapter` is cloneable for explicit sharing across multiple run options,
and each `RunOptions` retains its adapters through every native run.

## Runtime values

`Session::run_values` accepts heterogeneous `ValueInput` values. Numeric
tensors retain the zero-copy DLPack path, while string tensors use
`StringInput` and return Rust-owned `StringTensor` values:

```rust,ignore
let inputs = [ValueInput::Strings(StringInput::new(
    "text",
    vec![2],
    ["hello", "世界"],
)?)];
let outputs = session.run_values(&inputs, &["output"])?;
```

The output `Value` enum also represents sequence, map, and optional values
recursively. Sparse values use the `raw` API described above; opaque values
remain unsupported by the high-level value enum.

Sequence and map inputs use the same recursive value model. Tensor children
remain zero-copy DLPack borrows:

```rust,ignore
let input = ValueInput::Sequence(SequenceInput::new(
    "items",
    [
        InputValue::from_dlpack(&first)?,
        InputValue::from_dlpack(&second)?,
    ],
)?)];
let outputs = session.run_values(&input, &["length"])?;
```

`MapInput::new` accepts separate `InputValue` key and value components. ORT's
C API v21 does not provide an equivalent constructor for optional inputs, so
optional values are currently output-only.

`Session::input_values`, `output_values`, and
`overridable_initializer_values` expose recursive model type metadata through
`ValueTypeInfo`. Dense tensors include element type and symbolic or fixed
dimensions; sequence, map, and optional descriptions retain their nested ONNX
types. The existing `inputs`, `outputs`, and `overridable_initializers` helpers
remain convenient tensor-only views.

## Updating the C bindings

`onyxrt-sys` commits generated bindings so downstream builds do not require
Clang. The workspace `xtask` uses `usage-rs` for its CLI and pins both the ONNX
Runtime v1.21.0 header URL and its SHA-256 digest:

```text
cargo xtask download-header
cargo xtask bindgen
```

The download task verifies the upstream response before updating the vendored
header. Binding generation requires libclang and rustfmt. CI checks the
committed output without modifying it:

```text
cargo xtask download-header --check
cargo xtask bindgen --check
```

## License

onyxrt is licensed under the MIT License. The generated ONNX Runtime bindings
retain Microsoft's copyright notice and MIT license.
