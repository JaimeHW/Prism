use super::*;
use prism_core::python_unicode::encode_python_code_point;
use prism_runtime::types::string::value_as_string_ref;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value.as_object_ptr().expect("expected builtin");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_module_exposes_bootstrap_surface() {
    let module = CodecsModule::new();
    assert!(module.get_attr("lookup").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("utf_8_encode")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("raw_unicode_escape_encode")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
}

#[test]
fn test_lookup_known_and_unknown_codec() {
    let lookup = builtin_from_value(CodecsModule::new().get_attr("lookup").unwrap());
    let mut vm = VirtualMachine::new();
    assert!(
        lookup
            .call_with_vm(&mut vm, &[Value::string(intern("utf-8"))])
            .is_ok()
    );
    assert!(
        lookup
            .call_with_vm(&mut vm, &[Value::string(intern("raw-unicode-escape"))])
            .is_ok()
    );
    let err = lookup
        .call_with_vm(&mut vm, &[Value::string(intern("nope-codec"))])
        .expect_err("unknown codec should fail");
    assert!(matches!(err, BuiltinError::KeyError(_)));
}

#[test]
fn test_register_error_roundtrip() {
    static HANDLER: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
        BuiltinFunctionObject::new(Arc::from("tests.handler"), |_args| Ok(Value::none()))
    });

    let module = CodecsModule::new();
    let mut vm = VirtualMachine::new();
    builtin_from_value(module.get_attr("register_error").unwrap())
        .call_with_vm(
            &mut vm,
            &[
                Value::string(intern("codex-handler")),
                builtin_value(&HANDLER),
            ],
        )
        .expect("register_error should succeed");

    let value = builtin_from_value(module.get_attr("lookup_error").unwrap())
        .call_with_vm(&mut vm, &[Value::string(intern("codex-handler"))])
        .expect("lookup_error should succeed");
    assert_eq!(
        value.as_object_ptr().expect("handler should be object"),
        &*HANDLER as *const BuiltinFunctionObject as *const ()
    );
}

#[test]
fn test_registered_error_handlers_are_vm_local() {
    static HANDLER: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
        BuiltinFunctionObject::new(Arc::from("tests.local_handler"), |_args| Ok(Value::none()))
    });

    let module = CodecsModule::new();
    let register_error = builtin_from_value(module.get_attr("register_error").unwrap());
    let lookup_error = builtin_from_value(module.get_attr("lookup_error").unwrap());
    let mut first = VirtualMachine::new();
    let mut second = VirtualMachine::new();

    register_error
        .call_with_vm(
            &mut first,
            &[
                Value::string(intern("vm-local-handler")),
                builtin_value(&HANDLER),
            ],
        )
        .expect("register_error should succeed");

    assert!(
        lookup_error
            .call_with_vm(&mut first, &[Value::string(intern("vm-local-handler"))])
            .is_ok()
    );
    assert!(matches!(
        lookup_error
            .call_with_vm(&mut second, &[Value::string(intern("vm-local-handler"))])
            .expect_err("second VM should not see first VM's handler"),
        BuiltinError::KeyError(_)
    ));
}

#[test]
fn test_lookup_error_exposes_surrogate_handlers() {
    let lookup_error = builtin_from_value(CodecsModule::new().get_attr("lookup_error").unwrap());
    let mut vm = VirtualMachine::new();

    for handler_name in ["surrogateescape", "surrogatepass", "backslashreplace"] {
        let value = lookup_error
            .call_with_vm(&mut vm, &[Value::string(intern(handler_name))])
            .expect("built-in surrogate handler should resolve");
        assert!(
            value.as_object_ptr().is_some(),
            "lookup_error should return callable object for {handler_name}"
        );
    }
}

#[test]
fn test_utf8_encode_returns_tuple() {
    let value = builtin_from_value(CodecsModule::new().get_attr("utf_8_encode").unwrap())
        .call(&[Value::string(intern("hello"))])
        .expect("utf_8_encode should succeed");
    let ptr = value.as_object_ptr().expect("tuple should be object");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let bytes_ptr = tuple
        .get(0)
        .unwrap()
        .as_object_ptr()
        .expect("bytes should be object");
    let bytes = unsafe { &*(bytes_ptr as *const BytesObject) };
    assert_eq!(bytes.as_bytes(), b"hello");
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(5));
}

#[test]
fn test_utf8_encode_supports_surrogatepass() {
    let surrogate = encode_python_code_point(0xDC80).expect("surrogate carrier should encode");
    let text = format!("A{surrogate}");

    let value = builtin_from_value(CodecsModule::new().get_attr("utf_8_encode").unwrap())
        .call(&[
            leak_object_value(StringObject::from_string(text)),
            Value::string(intern("surrogatepass")),
        ])
        .expect("utf_8_encode should accept surrogatepass");

    let ptr = value.as_object_ptr().expect("tuple should be object");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let bytes_ptr = tuple
        .get(0)
        .unwrap()
        .as_object_ptr()
        .expect("bytes should be object");
    let bytes = unsafe { &*(bytes_ptr as *const BytesObject) };
    assert_eq!(bytes.as_bytes(), &[0x41, 0xED, 0xB2, 0x80]);
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(2));
}

#[test]
fn test_raw_unicode_escape_encode_decode_returns_codec_tuples() {
    let encoded = builtin_from_value(
        CodecsModule::new()
            .get_attr("raw_unicode_escape_encode")
            .unwrap(),
    )
    .call(&[Value::string(intern("\u{0100}"))])
    .expect("raw_unicode_escape_encode should succeed");
    let encoded_ptr = encoded.as_object_ptr().expect("tuple should be object");
    let encoded_tuple = unsafe { &*(encoded_ptr as *const TupleObject) };
    let bytes_ptr = encoded_tuple
        .get(0)
        .unwrap()
        .as_object_ptr()
        .expect("bytes should be object");
    let bytes = unsafe { &*(bytes_ptr as *const BytesObject) };
    assert_eq!(bytes.as_bytes(), b"\\u0100");
    assert_eq!(encoded_tuple.get(1).unwrap().as_int(), Some(1));

    let decoded = builtin_from_value(
        CodecsModule::new()
            .get_attr("raw_unicode_escape_decode")
            .unwrap(),
    )
    .call(&[leak_object_value(BytesObject::from_vec_with_type(
        b"\\u0100".to_vec(),
        TypeId::BYTES,
    ))])
    .expect("raw_unicode_escape_decode should succeed");
    let decoded_ptr = decoded.as_object_ptr().expect("tuple should be object");
    let decoded_tuple = unsafe { &*(decoded_ptr as *const TupleObject) };
    let text_ptr = decoded_tuple
        .get(0)
        .unwrap()
        .as_object_ptr()
        .expect("decoded text should be object");
    let text = unsafe { &*(text_ptr as *const StringObject) };
    assert_eq!(text.as_str(), "\u{0100}");
    assert_eq!(decoded_tuple.get(1).unwrap().as_int(), Some(6));
}

#[test]
fn test_utf8_decode_supports_surrogateescape() {
    let value = builtin_from_value(CodecsModule::new().get_attr("utf_8_decode").unwrap())
        .call(&[
            leak_object_value(BytesObject::from_vec_with_type(
                vec![0x41, 0xFF, 0x42],
                TypeId::BYTES,
            )),
            Value::string(intern("surrogateescape")),
        ])
        .expect("utf_8_decode should accept surrogateescape");

    let ptr = value.as_object_ptr().expect("tuple should be object");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let decoded =
        value_as_string_ref(tuple.get(0).unwrap()).expect("decoded text should be string");
    let escaped = encode_python_code_point(0xDCFF).expect("surrogate carrier should encode");
    assert_eq!(decoded.as_str(), format!("A{escaped}B"));
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(3));
}

#[test]
fn test_utf8_decode_supports_backslashreplace() {
    let value = builtin_from_value(CodecsModule::new().get_attr("utf_8_decode").unwrap())
        .call(&[
            leak_object_value(BytesObject::from_vec_with_type(
                vec![0x41, 0xFF, 0x42],
                TypeId::BYTES,
            )),
            Value::string(intern("backslashreplace")),
        ])
        .expect("utf_8_decode should accept backslashreplace");

    let ptr = value.as_object_ptr().expect("tuple should be object");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(3));

    let decoded =
        value_as_string_ref(tuple.get(0).unwrap()).expect("decoded text should be string");
    assert_eq!(decoded.as_str(), r"A\xffB");
}
