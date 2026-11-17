//! Execution-provider device discovery and memory-capacity queries.

use std::ffi::CStr;

use onyxrt_sys as sys;
use snafu::Snafu;

use crate::{
    ffi::{self, api_function, require_const_non_null, status_call},
    model::Environment,
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Ffi {
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl From<ffi::Error> for Error {
    fn from(source: ffi::Error) -> Self {
        Self::Ffi {
            source: Box::new(source),
        }
    }
}

/// Hardware class exposed by an execution-provider device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HardwareDeviceType {
    Cpu,
    Gpu,
    Npu,
    Unknown(u32),
}

impl From<sys::OrtHardwareDeviceType> for HardwareDeviceType {
    fn from(value: sys::OrtHardwareDeviceType) -> Self {
        match value {
            sys::OrtHardwareDeviceType::OrtHardwareDeviceType_CPU => Self::Cpu,
            sys::OrtHardwareDeviceType::OrtHardwareDeviceType_GPU => Self::Gpu,
            sys::OrtHardwareDeviceType::OrtHardwareDeviceType_NPU => Self::Npu,
            sys::OrtHardwareDeviceType(raw) => Self::Unknown(raw),
        }
    }
}

/// Memory pool associated with an execution-provider device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceMemoryType {
    /// Device-local memory, such as GPU VRAM.
    Default,
    /// Shared or pinned memory accessible from the host.
    HostAccessible,
}

impl From<DeviceMemoryType> for sys::OrtDeviceMemoryType {
    fn from(value: DeviceMemoryType) -> Self {
        match value {
            DeviceMemoryType::Default => Self::OrtDeviceMemoryType_DEFAULT,
            DeviceMemoryType::HostAccessible => Self::OrtDeviceMemoryType_HOST_ACCESSIBLE,
        }
    }
}

/// Current capacity information for one device memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryCapacity {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

/// A device advertised by an ONNX Runtime execution provider.
///
/// The native device descriptor is owned by the environment. Keeping this
/// value alive therefore also keeps its [`Environment`] alive.
#[derive(Clone)]
pub struct EpDevice {
    environment: Environment,
    pointer: usize,
}

// SAFETY: OrtEpDevice descriptors are immutable and owned by the environment.
unsafe impl Send for EpDevice {}
// SAFETY: all descriptor accessors are read-only. Capacity queries delegate
// synchronization to the execution provider as required by the C API.
unsafe impl Sync for EpDevice {}

impl EpDevice {
    pub(crate) fn from_raw(environment: Environment, pointer: *const sys::OrtEpDevice) -> Self {
        Self {
            environment,
            pointer: pointer as usize,
        }
    }

    /// Execution-provider registration name, such as `CUDAExecutionProvider`.
    pub fn provider_name(&self) -> Result<String> {
        self.ep_string(EpString::Name)
    }

    /// Vendor name reported by the execution provider.
    pub fn provider_vendor(&self) -> Result<String> {
        self.ep_string(EpString::Vendor)
    }

    /// Hardware class of the physical or virtual device.
    pub fn hardware_type(&self) -> Result<HardwareDeviceType> {
        let api = &self.environment.inner.api;
        // SAFETY: the EP descriptor is live while its environment is live.
        let hardware = unsafe { api_function!(api, EpDevice_Device)(self.as_ptr()) };
        let hardware = require_const_non_null(hardware, "execution-provider hardware device")?;
        // SAFETY: hardware is an immutable descriptor owned by the environment.
        Ok(unsafe { api_function!(api, HardwareDevice_Type)(hardware) }.into())
    }

    /// Numeric hardware vendor identifier.
    pub fn vendor_id(&self) -> Result<u32> {
        let hardware = self.hardware_device()?;
        let api = &self.environment.inner.api;
        // SAFETY: hardware is an immutable descriptor owned by the environment.
        Ok(unsafe { api_function!(api, HardwareDevice_VendorId)(hardware) })
    }

    /// Numeric hardware device identifier within the vendor namespace.
    pub fn device_id(&self) -> Result<u32> {
        let hardware = self.hardware_device()?;
        let api = &self.environment.inner.api;
        // SAFETY: hardware is an immutable descriptor owned by the environment.
        Ok(unsafe { api_function!(api, HardwareDevice_DeviceId)(hardware) })
    }

    /// Hardware vendor name.
    pub fn hardware_vendor(&self) -> Result<String> {
        let hardware = self.hardware_device()?;
        let api = &self.environment.inner.api;
        // SAFETY: hardware is an immutable descriptor owned by the environment.
        let value = unsafe { api_function!(api, HardwareDevice_Vendor)(hardware) };
        copy_c_string(value, "hardware vendor").map_err(Into::into)
    }

    /// Queries both total and currently available memory.
    ///
    /// Providers may return `ORT_NOT_IMPLEMENTED`. On CUDA, querying available
    /// memory may initialize a device context and should not be polled rapidly.
    pub fn memory_capacity(&self, memory_type: DeviceMemoryType) -> Result<MemoryCapacity> {
        let api = &self.environment.inner.api;
        let mut total_bytes = 0;
        let mut available_bytes = 0;
        status_call!(
            api,
            EpDevice_GetMemoryCapacity(
                self.as_ptr(),
                memory_type.into(),
                &mut total_bytes,
                &mut available_bytes
            )
        )?;
        Ok(MemoryCapacity {
            total_bytes,
            available_bytes,
        })
    }

    /// Queries only total memory, avoiding provider work needed solely to
    /// estimate currently available memory.
    pub fn total_memory_capacity(&self, memory_type: DeviceMemoryType) -> Result<u64> {
        let api = &self.environment.inner.api;
        let mut total_bytes = 0;
        status_call!(
            api,
            EpDevice_GetMemoryCapacity(
                self.as_ptr(),
                memory_type.into(),
                &mut total_bytes,
                std::ptr::null_mut()
            )
        )?;
        Ok(total_bytes)
    }

    fn hardware_device(&self) -> Result<*const sys::OrtHardwareDevice> {
        let api = &self.environment.inner.api;
        // SAFETY: the EP descriptor is live while its environment is live.
        let hardware = unsafe { api_function!(api, EpDevice_Device)(self.as_ptr()) };
        require_const_non_null(hardware, "execution-provider hardware device").map_err(Into::into)
    }

    fn ep_string(&self, field: EpString) -> Result<String> {
        let api = &self.environment.inner.api;
        // SAFETY: the EP descriptor is live while its environment is live.
        let value = unsafe {
            match field {
                EpString::Name => api_function!(api, EpDevice_EpName)(self.as_ptr()),
                EpString::Vendor => api_function!(api, EpDevice_EpVendor)(self.as_ptr()),
            }
        };
        copy_c_string(value, field.description()).map_err(Into::into)
    }

    fn as_ptr(&self) -> *const sys::OrtEpDevice {
        self.pointer as *const sys::OrtEpDevice
    }
}

enum EpString {
    Name,
    Vendor,
}

impl EpString {
    fn description(&self) -> &'static str {
        match self {
            Self::Name => "execution-provider name",
            Self::Vendor => "execution-provider vendor",
        }
    }
}

fn copy_c_string(value: *const std::ffi::c_char, what: &'static str) -> ffi::Result<String> {
    let value = require_const_non_null(value, what)?;
    // SAFETY: ORT documents descriptor strings as NUL-terminated and owned by
    // the descriptor's environment.
    Ok(unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_known_and_unknown_hardware_types() {
        assert_eq!(
            HardwareDeviceType::from(sys::OrtHardwareDeviceType::OrtHardwareDeviceType_NPU),
            HardwareDeviceType::Npu
        );
        assert_eq!(
            HardwareDeviceType::from(sys::OrtHardwareDeviceType(99)),
            HardwareDeviceType::Unknown(99)
        );
    }
}
