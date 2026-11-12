use super::*;

fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn async_and_dlpack_borrows_have_required_thread_traits() {
    assert_send_sync::<Input<'static>>();
    assert_send_sync::<IoBindingInput<'static>>();
    assert_send::<OutputBuffer<'static>>();
    assert_send::<IoBindingOutput<'static>>();
    assert_send::<PendingIoBindingRun<'static>>();
    assert_send::<RunFuture<'static>>();
}

#[test]
fn accepts_null_output_array_for_zero_output_callback() {
    assert!(
        collect_callback_outputs(ptr::null_mut(), 0)
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        collect_callback_outputs(ptr::null_mut(), 1),
        Err(Error::NullPointer {
            what: "async output array"
        })
    ));
}

#[test]
fn validates_string_tensor_inputs() {
    let scalar = StringInput::new("text", Vec::<i64>::new(), ["hello"]).unwrap();
    assert!(scalar.shape().is_empty());
    assert!(matches!(
        StringInput::new("text", vec![2], ["one"]),
        Err(Error::StringElementCount {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        StringInput::new("text", vec![-1], std::iter::empty::<&str>()),
        Err(Error::InvalidStringDimension { axis: 0, value: -1 })
    ));
    assert!(matches!(
        StringInput::new("text", vec![1], ["bad\0value"]),
        Err(Error::InvalidStringElement { index: 0, .. })
    ));
}

#[test]
fn maps_common_dlpack_types() {
    let f32_type = DLDataType {
        code: DLDataTypeCode::FLOAT,
        bits: 32,
        lanes: 1,
    };
    assert_eq!(
        dlpack_to_onnx(f32_type).unwrap(),
        sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    );
    assert_eq!(
        onnx_to_dlpack(dlpack_to_onnx(f32_type).unwrap()).unwrap(),
        f32_type
    );
}

#[test]
fn rejects_vector_lanes() {
    let vector = DLDataType {
        code: DLDataTypeCode::FLOAT,
        bits: 32,
        lanes: 4,
    };
    assert!(matches!(
        dlpack_to_onnx(vector),
        Err(Error::UnsupportedDlpackType { .. })
    ));
}

#[test]
fn round_trips_all_supported_dlpack_types() {
    use sys::ONNXTensorElementDataType as O;

    let cases = [
        (
            DLDataTypeCode::FLOAT,
            16,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        ),
        (
            DLDataTypeCode::FLOAT,
            32,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ),
        (
            DLDataTypeCode::FLOAT,
            64,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        ),
        (
            DLDataTypeCode::BFLOAT,
            16,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        ),
        (
            DLDataTypeCode::INT,
            8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ),
        (
            DLDataTypeCode::INT,
            16,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        ),
        (
            DLDataTypeCode::INT,
            32,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        ),
        (
            DLDataTypeCode::INT,
            64,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        ),
        (
            DLDataTypeCode::UINT,
            8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        ),
        (
            DLDataTypeCode::UINT,
            16,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        ),
        (
            DLDataTypeCode::UINT,
            32,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        ),
        (
            DLDataTypeCode::UINT,
            64,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        ),
        (
            DLDataTypeCode::BOOL,
            8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        ),
        (
            DLDataTypeCode::COMPLEX,
            64,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
        ),
        (
            DLDataTypeCode::COMPLEX,
            128,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
        ),
        (
            DLDataTypeCode::FLOAT8_E4M3FN,
            8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
        ),
        (
            DLDataTypeCode::FLOAT8_E4M3FNUZ,
            8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
        ),
        (
            DLDataTypeCode::FLOAT8_E5M2,
            8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
        ),
        (
            DLDataTypeCode::FLOAT8_E5M2FNUZ,
            8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ,
        ),
        (
            DLDataTypeCode::UINT,
            4,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
        ),
        (
            DLDataTypeCode::INT,
            4,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
        ),
        (
            DLDataTypeCode::FLOAT4_E2M1FN,
            4,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1,
        ),
        (
            DLDataTypeCode::UINT,
            2,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2,
        ),
        (
            DLDataTypeCode::INT,
            2,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2,
        ),
        (
            DLDataTypeCode::FLOAT8_E8M0FNU,
            8,
            O::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0,
        ),
    ];

    for (code, bits, onnx) in cases {
        let dlpack = DLDataType {
            code,
            bits,
            lanes: 1,
        };
        assert_eq!(dlpack_to_onnx(dlpack).unwrap(), onnx);
        assert_eq!(onnx_to_dlpack(onnx).unwrap(), dlpack);
    }
}

#[test]
fn rejects_onnx_strings_for_dlpack() {
    assert!(matches!(
        onnx_to_dlpack(sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
        Err(Error::UnsupportedOnnxType { .. })
    ));
}

#[test]
fn rejects_padded_subbyte_dlpack_layout() {
    let mut storage = Box::new(vec![0_u8; 2]);
    let data = storage.as_mut_ptr();
    let mut initialized = Dynamic::compact([2_i64].as_slice())
        .initialize(storage)
        .unwrap();
    initialized
        .set_data(data.cast())
        .set_device(DLDevice {
            device_type: DLDeviceType::CPU,
            device_id: 0,
        })
        .set_dtype(DLDataType {
            code: DLDataTypeCode::INT,
            bits: 4,
            lanes: 1,
        });
    // SAFETY: the test allocation intentionally models one padded byte per
    // logical int4 element and accurately declares that layout in flags.
    unsafe {
        initialized.set_flags_unchecked(DlpackFlags::IS_SUBBYTE_TYPE_PADDED);
    }
    // SAFETY: all descriptor fields and owned storage are initialized.
    let tensor = unsafe { initialized.finish() };
    assert!(matches!(
        Input::from_dlpack("x", &tensor),
        Err(Error::UnsupportedDlpackFlags { .. })
    ));
}

#[test]
fn rejects_read_only_preallocated_output() {
    let mut storage = Box::new(vec![0_f32; 2]);
    let data = storage.as_mut_ptr();
    let mut initialized = Dynamic::compact([2_i64].as_slice())
        .initialize(storage)
        .unwrap();
    initialized
        .set_data(data.cast())
        .set_device(DLDevice {
            device_type: DLDeviceType::CPU,
            device_id: 0,
        })
        .set_dtype(DLDataType::F32)
        .set_flags(DlpackFlags::READ_ONLY)
        .unwrap();
    // SAFETY: all descriptor fields and owned storage are initialized.
    let mut tensor = unsafe { initialized.finish() };
    assert!(matches!(
        OutputBuffer::from_dlpack("y", &mut tensor),
        Err(Error::DlpackTensor { .. })
    ));
}

#[test]
fn builds_typed_cuda_options() {
    let provider = CudaExecutionProvider::new()
        .with_device_id(2)
        .with_memory_limit(1024)
        .with_arena_extend_strategy(CudaArenaExtendStrategy::SameAsRequested)
        .with_cudnn_conv_algo_search(CudnnConvAlgoSearch::Heuristic)
        .with_copy_in_default_stream(true)
        .with_cudnn_max_workspace(false)
        .with_tf32(true)
        .with_cuda_graph(false);
    assert_eq!(
        provider.options.0,
        [
            ("device_id".to_owned(), "2".to_owned()),
            ("gpu_mem_limit".to_owned(), "1024".to_owned()),
            (
                "arena_extend_strategy".to_owned(),
                "kSameAsRequested".to_owned()
            ),
            ("cudnn_conv_algo_search".to_owned(), "HEURISTIC".to_owned()),
            ("do_copy_in_default_stream".to_owned(), "1".to_owned()),
            ("cudnn_conv_use_max_workspace".to_owned(), "0".to_owned()),
            ("use_tf32".to_owned(), "1".to_owned()),
            ("enable_cuda_graph".to_owned(), "0".to_owned()),
        ]
    );
}

#[test]
fn provider_option_overwrites_existing_key() {
    let provider = TensorRtExecutionProvider::new()
        .with_fp16(false)
        .option("trt_fp16_enable", "1");
    assert_eq!(
        provider.options.0,
        [("trt_fp16_enable".to_owned(), "1".to_owned())]
    );
}

#[test]
fn rejects_nul_in_provider_options() {
    let provider = DnnlExecutionProvider::new().option("bad\0key", "value");
    assert!(matches!(
        provider.options.encode("oneDNN provider option"),
        Err(Error::InvalidName { .. })
    ));
}

#[test]
fn accepts_compact_cuda_iobinding_descriptor() {
    let mut storage = Box::new(vec![1.0_f32, 2.0]);
    let data = storage.as_mut_ptr();
    let mut initialized = Dynamic::compact([2_i64].as_slice())
        .initialize(storage)
        .unwrap();
    initialized
        .set_data(data.cast())
        .set_device(DLDevice {
            device_type: DLDeviceType::CUDA,
            device_id: 3,
        })
        .set_dtype(DLDataType::F32);
    // SAFETY: this test validates metadata only. The pointer is never
    // dereferenced as CUDA memory or passed to ONNX Runtime.
    let tensor = unsafe { initialized.finish() };
    let input = IoBindingInput::from_dlpack("x", &tensor).unwrap();
    assert_eq!(input.device().device_type, DLDeviceType::CUDA);
    assert_eq!(input.device().device_id, 3);
}

#[test]
fn rejects_negative_cuda_iobinding_device() {
    let mut storage = Box::new(vec![1.0_f32]);
    let data = storage.as_mut_ptr();
    let mut initialized = Dynamic::compact([1_i64].as_slice())
        .initialize(storage)
        .unwrap();
    initialized
        .set_data(data.cast())
        .set_device(DLDevice {
            device_type: DLDeviceType::CUDA,
            device_id: -1,
        })
        .set_dtype(DLDataType::F32);
    // SAFETY: this test validates metadata only and never dereferences data.
    let tensor = unsafe { initialized.finish() };
    assert!(matches!(
        IoBindingInput::from_dlpack("x", &tensor),
        Err(Error::UnsupportedIoBindingDevice { .. })
    ));
}
