# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/SunDoge/onyxrt/compare/onyxrt-v0.1.1...onyxrt-v0.2.0) - 2026-11-12

### Added

- expand tensor memory support
- [**breaking**] split value ownership and add DLPack export

### Added

- add accessors for runtime tensor element types and shapes
- add zero-copy conversion from owned ORT tensors into DLPack
- add typed `half::f16` and `half::bf16` tensor support behind the `half`
  feature
- add NPU memory classification and CUDA/ROCm device and pinned-host DLPack
  mappings

### Fixed

- allow empty CPU tensors whose runtime data pointer is null to be borrowed as empty slices

### Changed

- split values into owned `Value`, read-only borrowed `ValueRef`, and writable
  borrowed `ValueMut`
- make `MemoryInfo` borrow its environment instead of retaining an `Arc` clone
- consolidate simple native cleanup guards into one typed `OrtOwned` owner
- update the runtime baseline to ONNX Runtime 1.31.1 / C API v31

## [0.1.1](https://github.com/SunDoge/onyxrt/releases/tag/onyxrt-v0.1.1) - 2026-11-12

### Other

- publish the initial crates.io release

## [0.1.0](https://github.com/SunDoge/onyxrt/releases/tag/onyxrt-v0.1.0) - 2026-11-12

### Added

- make values framework-agnostic
- make session configuration reusable
- expand advanced inference APIs
- expose raw container values
- extend low-level runtime API
- add low-level runtime values
- add ONNX Runtime bindings

### Other

- stabilize cross-platform release checks
- align license with ONNX Runtime
- configure git-only releases
- simplify status call wrapper
- generalize status calls
- expose namespaced runtime APIs
- modularize dynamic runtime bindings
- generate ONNX Runtime bindings
