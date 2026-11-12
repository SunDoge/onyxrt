# dlpark integration notes

These are API gaps found while implementing ONNX Runtime asynchronous and
device-resident DLPack interop. They are candidates for dlpark issues or PRs.

## Cross-thread validated borrows

The original API used the same `TensorRef` concept for raw descriptors and
views borrowed from `Managed<M>`. Raw-pointer fields then made managed views
`!Send + !Sync` by auto-trait inference even though `Managed<M>` documents and
implements `Send + Sync`.

That is conservative for raw descriptors, because their unsafe contract
does not promise a thread-safe owner. It is unnecessarily restrictive for a
view borrowed specifically from `Managed`, whose metadata-immutability and
cross-thread deleter contracts are stronger.

The local path dependency now represents provenance explicitly:

```rust,ignore
pub struct RawTensorRef<'a> { /* !Send + !Sync */ }
pub struct RawTensorMut<'a> { /* !Send + !Sync */ }
pub struct TensorRef<'a> { /* Send + Sync */ }
pub struct TensorMut<'a> { /* Send */ }

impl<M: ManagedTensorBase> Managed<M> {
    pub fn validate(&self) -> Result<TensorRef<'_>, Error>;
    pub fn validate_mut(&mut self) -> Result<TensorMut<'_>, Error>;
}
```

Raw descriptors are validated through `RawTensorRef::from_raw` and
`RawTensorMut::from_raw`. Generic descriptor consumers can use `TensorView`.
This avoids declaring raw descriptors cross-thread safe while onyxrt no longer
needs downstream unsafe impls for its DLPack input and output borrow types.

## Owned, representation-erased tensor

Async FFI consumers often need to take ownership of either legacy or versioned
DLPack while copying only metadata. `AllocationDeleter` handles the eventual
deleter, but consumers must currently assemble data pointer, shape, strides,
dtype, device, byte offset, flags, and deleter themselves.

A representation-erased `OwnedTensor` built from `Managed<M>` would make this
pattern reusable and would provide a natural `Send + Sync` async input type.

## Device stream readiness

A validated tensor describes a device pointer but carries no Rust-level token
expressing producer-stream readiness. Python exchange handles a requested
stream during capsule production, while native Rust consumers must coordinate
streams externally.

A small optional readiness/event abstraction, or explicit documentation and
examples for CUDA stream handoff, would reduce accidental device-wide
synchronization in consumers such as ONNX Runtime I/O Binding.

## Device allocation helpers

`TensorView::as_raw_tensor()`, `offset_bytes_ptr()`, and `num_bytes()` are
sufficient for opaque
device bindings and worked well. A documented example showing that this pair is
the intended device-agnostic FFI path would make the contract easier to find.
