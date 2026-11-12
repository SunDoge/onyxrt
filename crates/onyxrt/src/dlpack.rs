//! Optional zero-copy DLPack adapters for ONNX Runtime values.

use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase, TensorView,
    ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType},
    metadata::Dynamic,
    versioned,
};
use onyxrt_sys as sys;
use snafu::{Snafu, ensure};
use std::ptr::NonNull;

use crate::{
    model::{ElementType, Environment},
    value::{self, DeviceType, MemoryInfo, Value, ValueRef},
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

    /// The ONNX Runtime device has no safe DLPack representation.
    #[snafu(display("unsupported ONNX Runtime device {name} ({device_type:?}:{device_id})"))]
    UnsupportedOrtDevice {
        /// Allocator or execution-provider name.
        name: String,
        /// Broad ONNX Runtime device category.
        device_type: DeviceType,
        /// Device ordinal.
        device_id: i32,
    },

    /// The ONNX element type has no DLPack representation.
    #[snafu(display("unsupported ONNX tensor element type {element_type:?}"))]
    UnsupportedElementType {
        /// Unsupported ONNX element type.
        element_type: ElementType,
    },

    /// DLPack metadata allocation or validation failed.
    #[snafu(display("could not create DLPack metadata: {source}"))]
    Metadata { source: dlpark::metadata::Error },

    /// ONNX Runtime rejected the external tensor.
    #[snafu(transparent)]
    Value { source: value::Error },
}

/// Transfers an owned dense CPU or CUDA ORT tensor into a DLPack handle.
///
/// The returned DLPack deleter owns `value`; dropping the handle releases the
/// underlying `OrtValue`. Borrowed [`ValueRef`] and [`crate::value::ValueMut`]
/// values are deliberately not accepted because DLPack may outlive them.
pub fn into_managed(value: Value) -> Result<versioned::Dlpack> {
    let info = value.tensor_info()?;
    let memory = value.memory_info()?;
    let device = match (memory.device_type, memory.device_id) {
        (DeviceType::Cpu, 0) => DLDevice::CPU,
        (DeviceType::Gpu, device_id)
            if device_id >= 0 && memory.name.eq_ignore_ascii_case("cuda") =>
        {
            DLDevice::cuda(device_id)
        }
        (device_type, device_id) => {
            return Err(Error::UnsupportedOrtDevice {
                name: memory.name,
                device_type,
                device_id,
            });
        }
    };
    let dtype = dlpack_type(info.element_type)?;
    let data = value.tensor_data_ptr()?;
    let mut initialized = Dynamic::compact(info.shape)
        .initialize(Box::new(value))
        .map_err(|source| Error::Metadata { source })?;
    initialized
        .set_data(data)
        .set_device(device)
        .set_dtype(dtype)
        .set_byte_offset(0);
    // SAFETY: the descriptor owns the OrtValue containing `data`, metadata is
    // copied into the managed allocation, and the device/dtype were derived
    // from ORT's own tensor descriptors.
    Ok(unsafe { initialized.finish() })
}

/// Validates and adapts a compact CPU or CUDA DLPack tensor without copying.
pub fn from_managed<'tensor, M>(
    environment: Environment,
    tensor: &'tensor Managed<M>,
) -> Result<ValueRef<'tensor>>
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
    Ok(unsafe {
        ValueRef::from_raw_parts(
            environment,
            &memory_info,
            NonNull::new(data.cast_mut().cast()).unwrap_or_else(NonNull::dangling),
            raw_tensor.num_bytes(),
            raw_tensor.shape(),
            element_type,
        )?
    })
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

fn dlpack_type(element_type: ElementType) -> Result<DLDataType> {
    let dtype = match element_type {
        ElementType::Float32 => DLDataType::F32,
        ElementType::Uint8 => DLDataType::U8,
        ElementType::Int8 => DLDataType::I8,
        ElementType::Uint16 => DLDataType::U16,
        ElementType::Int16 => DLDataType::I16,
        ElementType::Int32 => DLDataType::I32,
        ElementType::Int64 => DLDataType::I64,
        ElementType::Bool => DLDataType::BOOL,
        ElementType::Float16 => DLDataType::scalar(DLDataTypeCode::FLOAT, 16),
        ElementType::Float64 => DLDataType::F64,
        ElementType::Uint32 => DLDataType::U32,
        ElementType::Uint64 => DLDataType::U64,
        ElementType::Complex64 => DLDataType::C64,
        ElementType::Complex128 => DLDataType::C128,
        ElementType::BFloat16 => DLDataType::scalar(DLDataTypeCode::BFLOAT, 16),
        ElementType::Float8E4M3Fn => DLDataType::F8E4M3FN,
        ElementType::Float8E4M3Fnuz => DLDataType::F8E4M3FNUZ,
        ElementType::Float8E5M2 => DLDataType::F8E5M2,
        ElementType::Float8E5M2Fnuz => DLDataType::F8E5M2FNUZ,
        ElementType::Uint4 => DLDataType::scalar(DLDataTypeCode::UINT, 4),
        ElementType::Int4 => DLDataType::scalar(DLDataTypeCode::INT, 4),
        ElementType::Float4E2M1 => DLDataType::F4E2M1FN,
        ElementType::Uint2 => DLDataType::scalar(DLDataTypeCode::UINT, 2),
        ElementType::Int2 => DLDataType::scalar(DLDataTypeCode::INT, 2),
        ElementType::Float8E8M0 => DLDataType::F8E8M0FNU,
        ElementType::Undefined | ElementType::String => {
            return Err(Error::UnsupportedElementType { element_type });
        }
    };
    Ok(dtype)
}
