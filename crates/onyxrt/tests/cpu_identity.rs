use dlpark::{
    allocation::dynamic,
    ffi::{DLDataType, DLDevice, DLDeviceType, DLManagedTensorVersioned},
    metadata::Dynamic,
};
use ndarray::arr1;
use onyxrt::{
    CpuExecutionProvider, CudaExecutionProvider, Dimension, DnnlExecutionProvider, ElementType,
    Environment, Input, InputValue, IoBindingDeviceOutput, IoBindingInput, IoBindingOutput,
    LogLevel, MapInput, OutputBuffer, SequenceInput, StringInput, TensorRtExecutionProvider, Value,
    ValueInput, ValueTypeInfo,
};

// A minimal ONNX ModelProto: float[3] y = Identity(float[3] x), opset 13.
const IDENTITY_MODEL: &[u8] = &[
    0x08, 0x08, // ir_version = 8
    0x42, 0x02, 0x10, 0x0d, // opset_import { version: 13 }
    0x3a, 0x3e, // graph (62 bytes)
    0x0a, 0x10, 0x0a, 0x01, b'x', 0x12, 0x01, b'y', 0x22, 0x08, b'I', b'd', b'e', b'n', b't', b'i',
    b't', b'y', // node
    0x12, 0x08, b'i', b'd', b'e', b'n', b't', b'i', b't', b'y', // graph name
    0x5a, 0x0f, 0x0a, 0x01, b'x', 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08,
    0x03, // input float[3]
    0x62, 0x0f, 0x0a, 0x01, b'y', 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08,
    0x03, // output float[3]
];

// float[1] x, y -> sequence<float[1]> z = SequenceConstruct(x, y), opset 13.
const SEQUENCE_MODEL: &[u8] = &[
    0x08, 0x08, // ir_version = 8
    0x42, 0x02, 0x10, 0x0d, // opset_import { version: 13 }
    0x3a, 0x5f, // graph (95 bytes)
    0x0a, 0x1c, // node (28 bytes)
    0x0a, 0x01, b'x', 0x0a, 0x01, b'y', 0x12, 0x01, b'z', 0x22, 0x11, b'S', b'e', b'q', b'u', b'e',
    b'n', b'c', b'e', b'C', b'o', b'n', b's', b't', b'r', b'u', b'c',
    b't', // SequenceConstruct node
    0x12, 0x08, b's', b'e', b'q', b'u', b'e', b'n', b'c', b'e', // graph name
    0x5a, 0x0f, 0x0a, 0x01, b'x', 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08,
    0x01, // input x: float[1]
    0x5a, 0x0f, 0x0a, 0x01, b'y', 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08,
    0x01, // input y: float[1]
    0x62, 0x13, 0x0a, 0x01, b'z', 0x12, 0x0e, 0x22, 0x0c, 0x0a, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12,
    0x04, 0x0a, 0x02, 0x08, 0x01, // output: sequence<float[1]>
];

// sequence<float[1]> x -> scalar int64 y = SequenceLength(x), opset 13.
const SEQUENCE_INPUT_MODEL: &[u8] = &[
    0x08, 0x08, 0x42, 0x02, 0x10, 0x0d, 0x3a, 0x4a, // graph (74 bytes)
    0x0a, 0x16, 0x0a, 0x01, b'x', 0x12, 0x01, b'y', 0x22, 0x0e, b'S', b'e', b'q', b'u', b'e', b'n',
    b'c', b'e', b'L', b'e', b'n', b'g', b't', b'h', 0x12, 0x0e, b's', b'e', b'q', b'u', b'e', b'n',
    b'c', b'e', b'_', b'i', b'n', b'p', b'u', b't', // graph name
    0x5a, 0x13, 0x0a, 0x01, b'x', 0x12, 0x0e, 0x22, 0x0c, 0x0a, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12,
    0x04, 0x0a, 0x02, 0x08, 0x01, // sequence<float[1]>
    0x62, 0x0b, 0x0a, 0x01, b'y', 0x12, 0x06, 0x0a, 0x04, 0x08, 0x07, 0x12,
    0x00,
    // scalar int64
];

// A node-free graph passing map<int64, float> input x directly to output x.
const MAP_PASSTHROUGH_MODEL: &[u8] = &[
    0x08, 0x08, 0x42, 0x02, 0x10, 0x0d, 0x3a, 0x30, // graph (48 bytes)
    0x12, 0x0c, b'm', b'a', b'p', b'_', b'i', b'd', b'e', b'n', b't', b'i', b't', b'y', 0x5a, 0x0f,
    0x0a, 0x01, b'x', 0x12, 0x0a, 0x2a, 0x08, 0x08, 0x07, 0x12, 0x04, 0x0a, 0x02, 0x08,
    0x01, // input map<int64, float>
    0x62, 0x0f, 0x0a, 0x01, b'x', 0x12, 0x0a, 0x2a, 0x08, 0x08, 0x07, 0x12, 0x04, 0x0a, 0x02, 0x08,
    0x01, // output map<int64, float>
];

#[test]
#[ignore = "requires an ONNX Runtime native library; see README"]
fn cpu_dlpack_identity_round_trip() {
    let library = std::env::var_os("ONYXRT_TEST_DYLIB");
    if library.is_none() && std::env::var_os("ONYXRT_TEST_DEFAULT_DISCOVERY").is_none() {
        eprintln!(
            "skipping runtime integration test: set ONYXRT_TEST_DYLIB or ONYXRT_TEST_DEFAULT_DISCOVERY"
        );
        return;
    }
    let environment_builder = Environment::builder().name("onyxrt-integration-test");
    let environment_builder = match library {
        Some(library) => environment_builder.library_path(library),
        None => environment_builder,
    };
    let environment = environment_builder.build().unwrap();
    assert!(!environment.runtime_version().is_empty());
    assert!(!environment.runtime_origin().is_empty());
    let providers = environment.available_providers().unwrap();
    assert!(
        providers.iter().any(|name| name == "CPUExecutionProvider"),
        "available providers did not include CPUExecutionProvider: {providers:?}"
    );

    let model_path =
        std::env::temp_dir().join(format!("onyxrt-identity-{}.onnx", std::process::id()));
    std::fs::write(&model_path, IDENTITY_MODEL).unwrap();
    let file_session = environment
        .session_builder()
        .build_from_file(&model_path)
        .unwrap();
    assert_eq!(file_session.inputs().unwrap()[0].name(), "x");
    drop(file_session);
    std::fs::remove_file(&model_path).unwrap();

    let session = environment
        .session_builder()
        .cpu_provider(CpuExecutionProvider::new().with_arena(false))
        .build_from_bytes(IDENTITY_MODEL)
        .unwrap();

    let mut string_model = IDENTITY_MODEL.to_vec();
    let element_type_offsets = string_model
        .windows(4)
        .enumerate()
        .filter_map(|(index, bytes)| (bytes == [0x0a, 0x08, 0x08, 0x01]).then_some(index + 3))
        .collect::<Vec<_>>();
    assert_eq!(element_type_offsets.len(), 2);
    for offset in element_type_offsets {
        string_model[offset] = 0x08;
    }
    let string_session = environment
        .session_builder()
        .build_from_bytes(&string_model)
        .unwrap();
    let string_inputs = [ValueInput::Strings(
        StringInput::new("x", vec![3], ["hello", "", "世界"]).unwrap(),
    )];
    let mut string_outputs = string_session.run_values(&string_inputs, &["y"]).unwrap();
    assert_eq!(string_outputs[0].name(), "y");
    match string_outputs.remove(0).into_value() {
        Value::Strings(tensor) => {
            assert_eq!(tensor.shape(), &[3]);
            assert_eq!(tensor.values(), &["hello", "", "世界"]);
        }
        Value::Tensor(_) => panic!("string Identity returned a numeric tensor"),
        _ => panic!("string Identity returned a container value"),
    }

    let sequence_session = environment
        .session_builder()
        .build_from_bytes(SEQUENCE_MODEL)
        .unwrap();
    let sequence_metadata = sequence_session.output_values().unwrap();
    assert_eq!(sequence_metadata[0].name(), "z");
    let ValueTypeInfo::Sequence(element) = sequence_metadata[0].value_type() else {
        panic!("SequenceConstruct output did not report sequence metadata");
    };
    let ValueTypeInfo::Tensor(element) = element.as_ref() else {
        panic!("SequenceConstruct element did not report tensor metadata");
    };
    assert_eq!(element.element_type(), ElementType::Float32);
    assert_eq!(element.dimensions(), &[Dimension::Fixed(1)]);

    let sequence_x: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[3.0_f32])).try_into().unwrap();
    let sequence_y: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[-4.0_f32])).try_into().unwrap();
    // SAFETY: the ndarray adapters populated complete owned descriptors.
    let sequence_x = unsafe { sequence_x.finish() };
    // SAFETY: the ndarray adapters populated complete owned descriptors.
    let sequence_y = unsafe { sequence_y.finish() };
    let sequence_inputs = [
        ValueInput::Tensor(Input::from_dlpack("x", &sequence_x).unwrap()),
        ValueInput::Tensor(Input::from_dlpack("y", &sequence_y).unwrap()),
    ];
    let mut sequence_outputs = sequence_session
        .run_values(&sequence_inputs, &["z"])
        .unwrap();
    let Value::Sequence(elements) = sequence_outputs.remove(0).into_value() else {
        panic!("SequenceConstruct did not return a sequence");
    };
    assert_eq!(elements.len(), 2);
    for (element, expected) in elements.into_iter().zip([3.0_f32, -4.0]) {
        let Value::Tensor(element) = element else {
            panic!("SequenceConstruct returned a non-tensor element");
        };
        let element = element.validate().unwrap();
        // SAFETY: model metadata declares compact CPU f32[1] sequence elements.
        assert_eq!(unsafe { element.cpu_slice::<f32>() }.unwrap(), &[expected]);
    }

    let sequence_input_session = environment
        .session_builder()
        .build_from_bytes(SEQUENCE_INPUT_MODEL)
        .unwrap();
    let sequence_input_values = [ValueInput::Sequence(
        SequenceInput::new(
            "x",
            [
                InputValue::from_dlpack(&sequence_x).unwrap(),
                InputValue::from_dlpack(&sequence_y).unwrap(),
            ],
        )
        .unwrap(),
    )];
    let mut sequence_length = sequence_input_session
        .run_values(&sequence_input_values, &["y"])
        .unwrap();
    let Value::Tensor(sequence_length) = sequence_length.remove(0).into_value() else {
        panic!("SequenceLength did not return a tensor");
    };
    let sequence_length = sequence_length.validate().unwrap();
    assert_eq!(sequence_length.shape(), &[]);
    // SAFETY: the model declares a scalar int64 SequenceLength output.
    assert_eq!(unsafe { sequence_length.cpu_slice::<i64>() }.unwrap(), &[2]);

    let map_session = environment
        .session_builder()
        .build_from_bytes(MAP_PASSTHROUGH_MODEL)
        .unwrap();
    let map_metadata = map_session.input_values().unwrap();
    assert!(matches!(
        map_metadata[0].value_type(),
        ValueTypeInfo::Map {
            key: ElementType::Int64,
            value
        } if matches!(value.as_ref(), ValueTypeInfo::Tensor(t) if t.element_type() == ElementType::Float32)
    ));
    let map_keys: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[7_i64, 11])).try_into().unwrap();
    let map_values: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[1.5_f32, -2.25])).try_into().unwrap();
    // SAFETY: the ndarray adapters populated complete owned descriptors.
    let map_keys = unsafe { map_keys.finish() };
    // SAFETY: the ndarray adapters populated complete owned descriptors.
    let map_values = unsafe { map_values.finish() };
    let map_inputs = [ValueInput::Map(
        MapInput::new(
            "x",
            InputValue::from_dlpack(&map_keys).unwrap(),
            InputValue::from_dlpack(&map_values).unwrap(),
        )
        .unwrap(),
    )];
    let mut map_outputs = map_session.run_values(&map_inputs, &["x"]).unwrap();
    let Value::Map { keys, values } = map_outputs.remove(0).into_value() else {
        panic!("map passthrough did not return a map");
    };
    let Value::Tensor(keys) = *keys else {
        panic!("map keys were not a tensor");
    };
    let Value::Tensor(values) = *values else {
        panic!("map values were not a tensor");
    };
    let keys = keys.validate().unwrap();
    let values = values.validate().unwrap();
    // SAFETY: model metadata declares int64 keys and float32 values.
    assert_eq!(unsafe { keys.cpu_slice::<i64>() }.unwrap(), &[7, 11]);
    // SAFETY: model metadata declares int64 keys and float32 values.
    assert_eq!(unsafe { values.cpu_slice::<f32>() }.unwrap(), &[1.5, -2.25]);

    let inputs = session.inputs().unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].name(), "x");
    assert_eq!(inputs[0].element_type(), ElementType::Float32);
    assert_eq!(inputs[0].dimensions(), &[Dimension::Fixed(3)]);
    let input_values = session.input_values().unwrap();
    assert_eq!(input_values[0].name(), "x");
    let ValueTypeInfo::Tensor(input_type) = input_values[0].value_type() else {
        panic!("Identity input did not report tensor type metadata");
    };
    assert_eq!(input_type.element_type(), ElementType::Float32);
    assert_eq!(input_type.dimensions(), &[Dimension::Fixed(3)]);
    let output_info = session.outputs().unwrap();
    assert_eq!(output_info.len(), 1);
    assert_eq!(output_info[0].name(), "y");
    let output_values = session.output_values().unwrap();
    assert_eq!(output_values[0].name(), "y");
    assert!(matches!(
        output_values[0].value_type(),
        ValueTypeInfo::Tensor(_)
    ));
    assert!(session.overridable_initializers().unwrap().is_empty());
    let metadata = session.metadata().unwrap();
    assert_eq!(metadata.graph_name().unwrap(), "identity");
    let _model_version = metadata.version().unwrap();
    assert_eq!(metadata.custom("missing-key").unwrap(), None);

    let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[1.25_f32, -2.5, 9.0])).try_into().unwrap();
    // SAFETY: the ndarray adapter populated a complete owned descriptor.
    let input = unsafe { initialized.finish() };
    let bindings = [Input::from_dlpack("x", &input).unwrap()];
    let outputs = session.run_all(&bindings).unwrap();
    assert_eq!(outputs[0].name(), "y");

    let output = outputs[0].tensor().validate().unwrap();
    assert_eq!(output.shape(), &[3]);
    // SAFETY: the output descriptor was validated and its dtype is f32.
    let values = unsafe { output.cpu_slice::<f32>() }.unwrap();
    assert_eq!(values, &[1.25, -2.5, 9.0]);

    let value_inputs = [ValueInput::Tensor(Input::from_dlpack("x", &input).unwrap())];
    let mut value_outputs = session.run_values(&value_inputs, &["y"]).unwrap();
    match value_outputs.remove(0).into_value() {
        Value::Tensor(tensor) => {
            let tensor = tensor.validate().unwrap();
            // SAFETY: Identity preserves the validated compact f32 CPU tensor type.
            let values = unsafe { tensor.cpu_slice::<f32>() }.unwrap();
            assert_eq!(values, &[1.25, -2.5, 9.0]);
        }
        Value::Strings(_) => panic!("numeric Identity returned a string tensor"),
        _ => panic!("numeric Identity returned a container value"),
    }

    let mut run_options = session.run_options().unwrap();
    run_options
        .set_tag("cpu-identity")
        .unwrap()
        .set_log_level(LogLevel::Warning)
        .unwrap()
        .set_log_verbosity(0)
        .unwrap();
    run_options.terminate().unwrap();
    run_options.reset_terminate().unwrap();
    let configured_outputs = session
        .run_all_with_options(&bindings, &run_options)
        .unwrap();
    let configured_output = configured_outputs[0].tensor().validate().unwrap();
    // SAFETY: Identity preserves the validated compact f32 CPU tensor type.
    let configured_values = unsafe { configured_output.cpu_slice::<f32>() }.unwrap();
    assert_eq!(configured_values, &[1.25, -2.5, 9.0]);

    let async_outputs = block_on(session.run_all_async(&bindings).unwrap()).unwrap();
    let async_output = async_outputs[0].tensor().validate().unwrap();
    // SAFETY: Identity preserves the validated f32 CPU tensor type.
    let async_values = unsafe { async_output.cpu_slice::<f32>() }.unwrap();
    assert_eq!(async_values, &[1.25, -2.5, 9.0]);

    // Cancelling a borrowed native run waits for its callback before allowing
    // the input allocation to be released.
    drop(session.run_all_async(&bindings).unwrap());

    let bad_bindings = [Input::from_dlpack("missing", &input).unwrap()];
    assert!(block_on(session.run_all_async(&bad_bindings).unwrap()).is_err());

    let output_initialized: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[0.0_f32, 0.0, 0.0])).try_into().unwrap();
    // SAFETY: the ndarray adapter populated writable owned CPU storage.
    let mut caller_output = unsafe { output_initialized.finish() };
    {
        let mut output_buffers = [OutputBuffer::from_dlpack("y", &mut caller_output).unwrap()];
        session.run_into(&bindings, &mut output_buffers).unwrap();
    }
    let caller_output_view = caller_output.validate().unwrap();
    // SAFETY: the preallocated output is validated f32 CPU storage.
    let caller_values = unsafe { caller_output_view.cpu_slice::<f32>() }.unwrap();
    assert_eq!(caller_values, &[1.25, -2.5, 9.0]);

    let bound_output_initialized: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[0.0_f32, 0.0, 0.0])).try_into().unwrap();
    // SAFETY: the ndarray adapter populated writable owned CPU storage.
    let mut bound_output = unsafe { bound_output_initialized.finish() };
    {
        let io_inputs = [IoBindingInput::from_dlpack("x", &input).unwrap()];
        let mut io_outputs = [IoBindingOutput::from_dlpack("y", &mut bound_output).unwrap()];
        let mut reusable = session.io_binding().unwrap();
        let mut bound = reusable.bind(&io_inputs, &mut io_outputs).unwrap();
        bound.run().unwrap();
        bound.run_with_options(&run_options).unwrap();
    }
    let bound_output_view = bound_output.validate().unwrap();
    // SAFETY: the bound output is validated f32 CPU storage.
    let bound_values = unsafe { bound_output_view.cpu_slice::<f32>() }.unwrap();
    assert_eq!(bound_values, &[1.25, -2.5, 9.0]);

    let pending_output_initialized: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[0.0_f32, 0.0, 0.0])).try_into().unwrap();
    // SAFETY: the ndarray adapter populated writable owned CPU storage.
    let mut pending_output = unsafe { pending_output_initialized.finish() };
    {
        let io_inputs = [IoBindingInput::from_dlpack("x", &input).unwrap()];
        let mut io_outputs = [IoBindingOutput::from_dlpack("y", &mut pending_output).unwrap()];
        session
            .submit_with_iobinding(&io_inputs, &mut io_outputs)
            .unwrap()
            .wait()
            .unwrap();
    }
    let pending_output_view = pending_output.validate().unwrap();
    // SAFETY: wait completed all writes to validated f32 CPU storage.
    let pending_values = unsafe { pending_output_view.cpu_slice::<f32>() }.unwrap();
    assert_eq!(pending_values, &[1.25, -2.5, 9.0]);

    let io_inputs = [IoBindingInput::from_dlpack("x", &input).unwrap()];
    let requested_outputs = [IoBindingDeviceOutput::cpu("y").unwrap()];
    let dynamic_outputs = session
        .run_with_iobinding_outputs(&io_inputs, &requested_outputs)
        .unwrap();
    let dynamic_output = dynamic_outputs[0].tensor().validate().unwrap();
    assert_eq!(dynamic_output.device().device_type, DLDeviceType::CPU);
    assert_eq!(dynamic_output.shape(), &[3]);
    // SAFETY: ORT allocated a validated compact f32 CPU output.
    let dynamic_values = unsafe { dynamic_output.cpu_slice::<f32>() }.unwrap();
    assert_eq!(dynamic_values, &[1.25, -2.5, 9.0]);

    // Verify that imported DLPack views apply byte_offset rather than passing
    // the allocation base to ONNX Runtime.
    let mut storage = Box::new(vec![999.0_f32, 4.0, 5.0, 6.0]);
    let base = storage.as_mut_ptr();
    let mut initialized = Dynamic::compact([3_i64].as_slice())
        .initialize(storage)
        .unwrap();
    initialized
        .set_data(base.cast())
        .set_device(DLDevice {
            device_type: DLDeviceType::CPU,
            device_id: 0,
        })
        .set_dtype(DLDataType::F32)
        .set_byte_offset(std::mem::size_of::<f32>() as u64);
    // SAFETY: storage owns four f32 values and the byte offset exposes the final three.
    let offset_input = unsafe { initialized.finish() };
    let offset_bindings = [Input::from_dlpack("x", &offset_input).unwrap()];
    let offset_outputs = session.run_all(&offset_bindings).unwrap();
    let offset_output = offset_outputs[0].tensor().validate().unwrap();
    // SAFETY: Identity preserves the validated f32 tensor type.
    let offset_values = unsafe { offset_output.cpu_slice::<f32>() }.unwrap();
    assert_eq!(offset_values, &[4.0, 5.0, 6.0]);
}

#[test]
#[ignore = "requires a oneDNN-enabled ONNX Runtime; see README"]
fn dnnl_prebuilt_provider_creates_session() {
    if std::env::var_os("ONYXRT_TEST_DNNL").is_none() {
        eprintln!("skipping oneDNN integration test: ONYXRT_TEST_DNNL is not set");
        return;
    }
    let environment = provider_test_environment();
    let providers = environment.available_providers().unwrap();
    assert!(
        providers.iter().any(|name| name == "DnnlExecutionProvider"),
        "available providers did not include DnnlExecutionProvider: {providers:?}"
    );
    environment
        .session_builder()
        .dnnl_provider(DnnlExecutionProvider::new())
        .build_from_bytes(IDENTITY_MODEL)
        .unwrap();
}

#[test]
#[ignore = "requires a CUDA-enabled ONNX Runtime and GPU; see README"]
fn cuda_prebuilt_provider_creates_session() {
    if std::env::var_os("ONYXRT_TEST_CUDA").is_none() {
        eprintln!("skipping CUDA integration test: ONYXRT_TEST_CUDA is not set");
        return;
    }
    let environment = provider_test_environment();
    let providers = environment.available_providers().unwrap();
    assert!(
        providers.iter().any(|name| name == "CUDAExecutionProvider"),
        "available providers did not include CUDAExecutionProvider: {providers:?}"
    );
    let session = environment
        .session_builder()
        .cuda_provider(CudaExecutionProvider::new().with_device_id(0))
        .build_from_bytes(IDENTITY_MODEL)
        .unwrap();
    let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(arr1(&[2.0_f32, 3.0, 4.0])).try_into().unwrap();
    // SAFETY: the ndarray adapter populated a complete owned descriptor.
    let input = unsafe { initialized.finish() };
    let io_inputs = [IoBindingInput::from_dlpack("x", &input).unwrap()];
    let requested_outputs = [IoBindingDeviceOutput::cuda("y", 0).unwrap()];
    let outputs = session
        .run_with_iobinding_outputs(&io_inputs, &requested_outputs)
        .unwrap();
    assert_eq!(
        outputs[0].tensor().validate().unwrap().device().device_type,
        DLDeviceType::CUDA
    );
}

#[test]
#[ignore = "requires a TensorRT-enabled ONNX Runtime and GPU; see README"]
fn tensorrt_prebuilt_provider_creates_session() {
    if std::env::var_os("ONYXRT_TEST_TENSORRT").is_none() {
        eprintln!("skipping TensorRT integration test: ONYXRT_TEST_TENSORRT is not set");
        return;
    }
    let environment = provider_test_environment();
    let providers = environment.available_providers().unwrap();
    assert!(
        providers
            .iter()
            .any(|name| name == "TensorrtExecutionProvider"),
        "available providers did not include TensorrtExecutionProvider: {providers:?}"
    );
    environment
        .session_builder()
        .tensorrt_provider(TensorRtExecutionProvider::new().with_device_id(0))
        .build_from_bytes(IDENTITY_MODEL)
        .unwrap();
}

fn provider_test_environment() -> Environment {
    let builder = Environment::builder().name("onyxrt-provider-test");
    let builder = match std::env::var_os("ONYXRT_TEST_DYLIB") {
        Some(library) => builder.library_path(library),
        None => builder,
    };
    builder.build().unwrap()
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(output) => return output,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}
