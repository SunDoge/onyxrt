use onyxrt::{
    model::Environment,
    provider::{CudaExecutionProvider, DnnlExecutionProvider, TensorRtExecutionProvider},
    value::{AllocatorType, DeviceType, Input, MemoryInfo, MemoryType, Value, ValueMut},
};

#[cfg(feature = "dlpack")]
use dlpark::{TryFromDlpack, allocation::dynamic};
#[cfg(feature = "dlpack")]
use ndarray::{ArrayViewD, arr1};
#[cfg(feature = "dlpack")]
use onyxrt::dlpack;

const IDENTITY_MODEL: &[u8] = &[
    0x08, 0x08, 0x42, 0x02, 0x10, 0x0d, 0x3a, 0x3e, 0x0a, 0x10, 0x0a, 0x01, b'x', 0x12, 0x01, b'y',
    0x22, 0x08, b'I', b'd', b'e', b'n', b't', b'i', b't', b'y', 0x12, 0x08, b'i', b'd', b'e', b'n',
    b't', b'i', b't', b'y', 0x5a, 0x0f, 0x0a, 0x01, b'x', 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12,
    0x04, 0x0a, 0x02, 0x08, 0x03, 0x62, 0x0f, 0x0a, 0x01, b'y', 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01,
    0x12, 0x04, 0x0a, 0x02, 0x08, 0x03,
];

#[test]
#[ignore = "requires an ONNX Runtime native library; see README"]
fn cpu_value_identity_round_trip() {
    let environment = test_environment();
    let sessions = environment.session_builder();
    let first = sessions.build_from_bytes(IDENTITY_MODEL).unwrap();
    let replacement = sessions.build_from_bytes(IDENTITY_MODEL).unwrap();
    assert_eq!(replacement.inputs().unwrap()[0].name(), "x");

    let memory = MemoryInfo::cpu(&environment).unwrap();
    let npu_memory = MemoryInfo::new(
        &environment,
        "OpenVINO_RT_NPU",
        AllocatorType::Device,
        0,
        MemoryType::Default,
    )
    .unwrap();
    assert_eq!(npu_memory.device_type(), DeviceType::Npu);
    assert_eq!(
        MemoryInfo::cuda_pinned(&environment, 0)
            .unwrap()
            .device_type(),
        DeviceType::Gpu
    );
    assert_eq!(
        MemoryInfo::rocm(&environment, 0).unwrap().device_type(),
        DeviceType::Gpu
    );
    assert_eq!(
        MemoryInfo::rocm_pinned(&environment, 0)
            .unwrap()
            .device_type(),
        DeviceType::Gpu
    );
    let mut data = [1.5_f32, -2.0, 7.25];
    let value = ValueMut::tensor(environment.clone(), &memory, &mut data, [3]).unwrap();
    let inputs = [Input::new("x", &value).unwrap()];
    let outputs = first.run(&inputs, &["y"]).unwrap();
    let info = outputs[0].tensor_info().unwrap();
    assert_eq!(info.element_type(), onyxrt::model::ElementType::Float32);
    assert_eq!(info.shape(), &[3]);
    assert_eq!(outputs[0].as_slice::<f32>().unwrap(), &[1.5, -2.0, 7.25]);

    let empty = Value::allocate_tensor::<f32>(environment.clone(), [0, 6]).unwrap();
    assert!(empty.as_slice::<f32>().unwrap().is_empty());

    #[cfg(feature = "half")]
    {
        let expected = [half::f16::from_f32(1.5), half::f16::from_f32(-2.0)];
        let mut data = expected;
        let borrowed = ValueMut::tensor(environment.clone(), &memory, &mut data, [2]).unwrap();
        assert_eq!(borrowed.as_slice::<half::f16>().unwrap(), &expected);

        let mut owned = Value::allocate_tensor::<half::f16>(environment.clone(), [2]).unwrap();
        owned
            .as_slice_mut::<half::f16>()
            .unwrap()
            .copy_from_slice(&expected);
        assert_eq!(owned.as_slice::<half::f16>().unwrap(), &expected);
    }
}

#[cfg(feature = "dlpack")]
#[test]
#[ignore = "requires an ONNX Runtime native library; see README"]
fn cpu_dlpack_identity_round_trip() {
    let environment = test_environment();
    let session = environment
        .session_builder()
        .build_from_bytes(IDENTITY_MODEL)
        .unwrap();

    let initialized: dynamic::Initialized<dlpark::ffi::DLManagedTensorVersioned> =
        Box::new(arr1(&[1.5_f32, -2.0, 7.25])).try_into().unwrap();
    // SAFETY: the ndarray adapter populated a complete owned descriptor.
    let tensor = unsafe { initialized.finish() };
    let value = dlpack::from_managed(environment.clone(), &tensor).unwrap();
    let inputs = [Input::new("x", &value).unwrap()];
    let outputs = session.run(&inputs, &["y"]).unwrap();
    assert_eq!(outputs[0].as_slice::<f32>().unwrap(), &[1.5, -2.0, 7.25]);

    let output = dlpack::into_managed(outputs.into_iter().next().unwrap()).unwrap();
    // SAFETY: the owned ORT tensor remains alive in the DLPack context and the
    // immutable view does not alias mutable access.
    let view = unsafe { ArrayViewD::<f32>::try_from_dlpack(&output, ()) }.unwrap();
    assert_eq!(view.as_slice().unwrap(), &[1.5, -2.0, 7.25]);

    let empty = Value::allocate_tensor::<f32>(environment, [0, 6]).unwrap();
    let empty = dlpack::into_managed(empty).unwrap();
    let empty = empty.validate().unwrap();
    assert_eq!(empty.shape(), &[0, 6]);
    assert_eq!(empty.num_elements(), 0);
}

#[test]
#[ignore = "requires a oneDNN-enabled ONNX Runtime; see README"]
fn dnnl_prebuilt_provider_creates_session() {
    if std::env::var_os("ONYXRT_TEST_DNNL").is_none() {
        return;
    }
    test_environment()
        .session_builder()
        .execution_provider(DnnlExecutionProvider::new())
        .build_from_bytes(IDENTITY_MODEL)
        .unwrap();
}

#[test]
#[ignore = "requires a CUDA-enabled ONNX Runtime and GPU; see README"]
fn cuda_prebuilt_provider_creates_session() {
    if std::env::var_os("ONYXRT_TEST_CUDA").is_none() {
        return;
    }
    test_environment()
        .session_builder()
        .execution_provider(CudaExecutionProvider::new().with_device_id(0))
        .build_from_bytes(IDENTITY_MODEL)
        .unwrap();
}

#[test]
#[ignore = "requires a TensorRT-enabled ONNX Runtime and GPU; see README"]
fn tensorrt_prebuilt_provider_creates_session() {
    if std::env::var_os("ONYXRT_TEST_TENSORRT").is_none() {
        return;
    }
    test_environment()
        .session_builder()
        .execution_provider(TensorRtExecutionProvider::new().with_device_id(0))
        .build_from_bytes(IDENTITY_MODEL)
        .unwrap();
}

fn test_environment() -> Environment {
    let builder = Environment::builder().name("onyxrt-integration-test");
    let builder = match std::env::var_os("ONYXRT_TEST_DYLIB") {
        Some(library) => builder.library_path(library),
        None => builder,
    };
    builder.build().unwrap()
}
