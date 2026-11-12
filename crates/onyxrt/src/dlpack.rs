//! Optional zero-copy DLPack adapters for ONNX Runtime values.

use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase, TensorRef, TensorView,
    ffi::{DLDataType, DLDataTypeCode, DLDeviceType},
};
use onyxrt_sys as sys;
use snafu::{Snafu, ensure};
use std::ptr::NonNull;

use crate::{
    model::{ElementType, Environment},
    value::{self, MemoryInfo},
};

/// Result type for DLPack conversion operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors produced while adapting DLPack tensors.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    /// The DLPack descriptor is invalid.
    #[snafu(display("invalid DLPack tensor: {source}"))]
    InvalidTensor { source: dlpark::tensor::Error },

    /// The tensor is not compact in memory.
    #[snafu(display("DLPack tensor must use a compact layout"))]
    NonCompact,

    /// The DLPack device cannot be represented by ONNX Runtime.
    #[snafu(display("unsupported DLPack device {device_type:?}:{device_id}"))]
    UnsupportedDevice {
        /// DLPack device category.
        device_type: DLDeviceType,
        /// Device ordinal.
        device_id: i32,
    },

    /// The DLPack element type has no matching ONNX type.
    #[snafu(display(
        "unsupported DLPack dtype (code={}, bits={}, lanes={})",
        dtype.code.0,
        dtype.bits,
        dtype.lanes
    ))]
    UnsupportedType {
        /// Unsupported DLPack element type.
        dtype: DLDataType,
    },

    /// Packed sub-byte storage cannot be described unambiguously.
    #[snafu(display("unsupported DLPack flags {flags:?}"))]
    UnsupportedFlags {
        /// Unsupported DLPack flags.
        flags: DlpackFlags,
    },

    /// ONNX Runtime rejected the external tensor.
    #[snafu(transparent)]
    Value { source: value::Error },
}

/// A read-only `OrtValue` backed by a borrowed DLPack tensor.
///
/// This wrapper retains both the validated tensor borrow and its memory
/// descriptor. It exposes only shared access to the underlying value, so a
/// shared DLPack tensor cannot be turned into mutable Rust storage.
pub struct Value<'tensor> {
    value: value::Value<'tensor>,
    _memory_info: MemoryInfo,
    _tensor: TensorRef<'tensor>,
}

impl<'tensor> Value<'tensor> {
    /// Validates and adapts a compact CPU or CUDA DLPack tensor without copying.
    pub fn from_managed<M>(environment: Environment, tensor: &'tensor Managed<M>) -> Result<Self>
    where
        M: ManagedTensorBase,
    {
        let flags = tensor.flags();
        ensure!(
            !flags.contains(DlpackFlags::IS_SUBBYTE_TYPE_PADDED),
            UnsupportedFlagsSnafu { flags }
        );
        let tensor = tensor
            .validate()
            .map_err(|source| Error::InvalidTensor { source })?;
        ensure!(
            tensor
                .as_raw_tensor()
                .is_compact()
                .map_err(|source| Error::InvalidTensor { source })?,
            NonCompactSnafu
        );
        let device = tensor.device();
        let memory_info = match (device.device_type, device.device_id) {
            (DLDeviceType::CPU, 0) => MemoryInfo::cpu(environment.clone())?,
            (DLDeviceType::CUDA, device_id) if device_id >= 0 => {
                MemoryInfo::cuda(environment.clone(), device_id)?
            }
            (device_type, device_id) => {
                return Err(Error::UnsupportedDevice {
                    device_type,
                    device_id,
                });
            }
        };
        let raw_tensor = tensor.as_raw_tensor();
        // SAFETY: validation established a live compact allocation. The
        // returned wrapper retains the Managed borrow for the OrtValue lifetime.
        let data = unsafe { raw_tensor.offset_bytes_ptr() }
            .map_err(|source| Error::InvalidTensor { source })?;
        let element_type = element_type(raw_tensor.dtype())?;
        // SAFETY: data, shape, type, byte length, and device descriptor all
        // come from the validated DLPack tensor retained by this wrapper.
        let value = unsafe {
            value::Value::from_raw_parts(
                environment,
                &memory_info,
                NonNull::new(data.cast_mut().cast()).unwrap_or_else(NonNull::dangling),
                raw_tensor.num_bytes(),
                raw_tensor.shape(),
                element_type,
            )?
        };
        Ok(Self {
            value,
            _memory_info: memory_info,
            _tensor: tensor,
        })
    }

    /// Borrows the adapted ONNX Runtime value.
    pub fn as_value(&self) -> &value::Value<'tensor> {
        &self.value
    }
}

fn element_type(dtype: DLDataType) -> Result<ElementType> {
    use sys::ONNXTensorElementDataType as O;
    ensure!(dtype.lanes == 1, UnsupportedTypeSnafu { dtype });
    let value = match (dtype.code, dtype.bits) {
        (DLDataTypeCode::FLOAT, 16) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        (DLDataTypeCode::FLOAT, 32) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        (DLDataTypeCode::FLOAT, 64) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        (DLDataTypeCode::BFLOAT, 16) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        (DLDataTypeCode::INT, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        (DLDataTypeCode::INT, 16) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        (DLDataTypeCode::INT, 32) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        (DLDataTypeCode::INT, 64) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        (DLDataTypeCode::UINT, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        (DLDataTypeCode::UINT, 16) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        (DLDataTypeCode::UINT, 32) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        (DLDataTypeCode::UINT, 64) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        (DLDataTypeCode::BOOL, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        (DLDataTypeCode::COMPLEX, 64) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
        (DLDataTypeCode::COMPLEX, 128) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
        (DLDataTypeCode::FLOAT8_E4M3FN, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
        (DLDataTypeCode::FLOAT8_E4M3FNUZ, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
        (DLDataTypeCode::FLOAT8_E5M2, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
        (DLDataTypeCode::FLOAT8_E5M2FNUZ, 8) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ,
        (DLDataTypeCode::UINT, 4) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
        (DLDataTypeCode::INT, 4) => O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
        (DLDataTypeCode::FLOAT4_E2M1FN, 4) => sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1,
        (DLDataTypeCode::UINT, 2) => sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2,
        (DLDataTypeCode::INT, 2) => sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2,
        (DLDataTypeCode::FLOAT8_E8M0FNU, 8) => sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0,
        _ => return Err(Error::UnsupportedType { dtype }),
    };
    Ok(value.into())
}
