#!/usr/bin/env bash
set -euo pipefail

: "${ORT_LIB_LOCATION:?set ORT_LIB_LOCATION to the directory containing ONNX Runtime static archives}"

if [[ -n "${ORT_PREFER_DYNAMIC_LINK:-}" ]]; then
    echo "ORT_PREFER_DYNAMIC_LINK must be unset for the static-link test" >&2
    exit 2
fi

ONYXRT_TEST_LINKED=1 cargo test \
    -p onyxrt \
    --no-default-features \
    --features link-static \
    --test cpu_identity \
    cpu_dlpack_identity_round_trip \
    -- --ignored
