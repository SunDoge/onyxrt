use onyxrt::{
    model::Environment,
    provider::{CudaExecutionProvider, DnnlExecutionProvider, TensorRtExecutionProvider},
    value::{Input, MemoryInfo, Value},
};

#[cfg(feature = "dlpack")]
use dlpark::allocation::dynamic;
#[cfg(feature = "dlpack")]
use ndarray::arr1;
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

    let memory = MemoryInfo::cpu(environment.clone()).unwrap();
    let mut data = [1.5_f32, -2.0, 7.25];
    let value = Value::tensor(environment, &memory, &mut data, [3]).unwrap();
    let inputs = [Input::new("x", &value).unwrap()];
    let outputs = first.run(&inputs, &["y"]).unwrap();
    assert_eq!(outputs[0].as_slice::<f32>().unwrap(), &[1.5, -2.0, 7.25]);
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
    let value = dlpack::Value::from_managed(environment, &tensor).unwrap();
    let inputs = [Input::new("x", value.as_value()).unwrap()];
    let outputs = session.run(&inputs, &["y"]).unwrap();
    assert_eq!(outputs[0].as_slice::<f32>().unwrap(), &[1.5, -2.0, 7.25]);
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
