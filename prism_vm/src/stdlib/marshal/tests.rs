use super::*;
use prism_core::intern::intern;

fn bytes_to_vec(value: Value) -> Vec<u8> {
    let ptr = value.as_object_ptr().expect("bytes should be heap-backed");
    let bytes = unsafe { &*(ptr as *const BytesObject) };
    bytes.to_vec()
}

fn code_view(value: Value) -> &'static CodeObjectView {
    let ptr = value
        .as_object_ptr()
        .expect("code object should be heap-backed");
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::CODE);
    unsafe { &*(ptr as *const CodeObjectView) }
}

#[test]
fn test_marshal_module_exposes_version_and_callables() {
    let module = MarshalModule::new();

    assert_eq!(
        module.get_attr("version").unwrap().as_int(),
        Some(MARSHAL_VERSION)
    );
    assert!(module.get_attr("dumps").is_ok());
    assert!(module.get_attr("loads").is_ok());
    assert_eq!(
        module.dir(),
        vec![Arc::from("dumps"), Arc::from("loads"), Arc::from("version")]
    );
}

#[test]
fn test_marshal_dumps_and_loads_bool_values() {
    let true_bytes =
        marshal_dumps(&[Value::bool(true)]).expect("marshal.dumps(True) should succeed");
    let false_bytes =
        marshal_dumps(&[Value::bool(false)]).expect("marshal.dumps(False) should succeed");

    assert_eq!(bytes_to_vec(true_bytes), vec![TYPE_TRUE]);
    assert_eq!(bytes_to_vec(false_bytes), vec![TYPE_FALSE]);
    assert_eq!(marshal_loads(&[true_bytes]).unwrap().as_bool(), Some(true));
    assert_eq!(
        marshal_loads(&[false_bytes]).unwrap().as_bool(),
        Some(false)
    );
}

#[test]
fn test_marshal_round_trips_small_and_large_ints() {
    let small = marshal_dumps(&[Value::int(123_456).unwrap()]).expect("small ints should marshal");
    assert_eq!(marshal_loads(&[small]).unwrap().as_int(), Some(123_456));

    let big = bigint_to_value(BigInt::from(1_u8) << 80_u32);
    let round_tripped = marshal_loads(&[marshal_dumps(&[big]).expect("big ints should marshal")])
        .expect("big ints should unmarshal");
    assert_eq!(value_to_bigint(round_tripped), value_to_bigint(big));
}

#[test]
fn test_marshal_round_trips_string_and_bytes() {
    let text = Value::string(intern("prism"));
    let text_round_trip = marshal_loads(&[marshal_dumps(&[text]).expect("strings should marshal")])
        .expect("strings should unmarshal");
    assert_eq!(
        value_as_string_ref(text_round_trip)
            .expect("string round-trip should return a string")
            .as_str(),
        "prism"
    );

    let bytes = bytes_value(b"abc");
    let bytes_round_trip = marshal_loads(&[marshal_dumps(&[bytes]).expect("bytes should marshal")])
        .expect("bytes should unmarshal");
    assert_eq!(bytes_to_vec(bytes_round_trip), b"abc");
}

#[test]
fn test_marshal_round_trips_float_and_tuple_values() {
    let tuple = Value::object_ptr(Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::float(1.25),
        Value::string(intern("ok")),
    ]))) as *const ());

    let round_tripped = marshal_loads(&[marshal_dumps(&[tuple]).expect("tuple should marshal")])
        .expect("tuple should unmarshal");
    let tuple_ptr = round_tripped
        .as_object_ptr()
        .expect("tuple should be heap-backed");
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.get(0).and_then(|value| value.as_float()), Some(1.25));
    assert_eq!(
        tuple
            .get(1)
            .and_then(|value| value_as_string_ref(value))
            .map(|value| value.as_str().to_string()),
        Some("ok".to_string())
    );
}

#[test]
fn test_marshal_round_trips_prism_code_objects() {
    let nested = Arc::new(CodeObject {
        name: Arc::from("child"),
        qualname: Arc::from("parent.child"),
        filename: Arc::from("pkg.py"),
        first_lineno: 7,
        instructions: Box::new([Instruction::op(Opcode::ReturnNone)]),
        constants: Box::new([Constant::Value(Value::none())]),
        locals: Box::new([Arc::from("x")]),
        names: Box::new([Arc::from("global_name")]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        arg_count: 1,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        register_count: 2,
        flags: CodeFlags::NESTED,
        line_table: Box::new([LineTableEntry {
            start_pc: 0,
            end_pc: 1,
            line: 7,
        }]),
        exception_table: Box::new([]),
        nested_code_objects: Box::new([]),
    });
    let nested_const = Value::object_ptr(Arc::into_raw(Arc::clone(&nested)) as *const ());
    let code = Arc::new(CodeObject {
        name: Arc::from("<module>"),
        qualname: Arc::from("<module>"),
        filename: Arc::from("pkg.py"),
        first_lineno: 1,
        instructions: Box::new([
            Instruction::op_di(Opcode::MakeFunction, prism_code::Register(0), 1),
            Instruction::op(Opcode::ReturnNone),
        ]),
        constants: Box::new([
            Constant::Value(Value::string(intern("module-constant"))),
            Constant::Value(nested_const),
        ]),
        locals: Box::new([]),
        names: Box::new([Arc::from("__name__")]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        register_count: 1,
        flags: CodeFlags::MODULE,
        line_table: Box::new([LineTableEntry {
            start_pc: 0,
            end_pc: 2,
            line: 1,
        }]),
        exception_table: Box::new([ExceptionEntry {
            start_pc: 0,
            end_pc: 1,
            handler_pc: 1,
            finally_pc: u32::MAX,
            depth: 0,
            exception_type_idx: u16::MAX,
        }]),
        nested_code_objects: Box::new([nested]),
    });
    let code_value =
        Value::object_ptr(Box::into_raw(Box::new(CodeObjectView::new(code))) as *const ());

    let round_tripped =
        marshal_loads(&[marshal_dumps(&[code_value]).expect("code should marshal")])
            .expect("code should unmarshal");
    let loaded = code_view(round_tripped).code();

    assert_eq!(loaded.name.as_ref(), "<module>");
    assert_eq!(loaded.filename.as_ref(), "pkg.py");
    assert_eq!(loaded.instructions.len(), 2);
    assert_eq!(loaded.constants.len(), 2);
    assert_eq!(loaded.nested_code_objects.len(), 1);
    assert_eq!(loaded.nested_code_objects[0].name.as_ref(), "child");
    assert_eq!(loaded.exception_table[0].handler_pc, 1);

    let nested_ptr = match loaded.constants[1] {
        Constant::Value(value) => value.as_object_ptr().expect("nested code ref"),
        Constant::BigInt(_) => panic!("nested code constant should be a value"),
    };
    assert_eq!(
        nested_ptr,
        Arc::as_ptr(&loaded.nested_code_objects[0]) as *const ()
    );
}

#[test]
fn test_marshal_round_trips_keyword_name_constants_in_code_objects() {
    let kwnames = Box::into_raw(Box::new(KwNamesTuple::new(vec![
        Arc::from("encoding"),
        Arc::from("errors"),
    ])));
    let code = Arc::new(CodeObject {
        name: Arc::from("call_site"),
        qualname: Arc::from("call_site"),
        filename: Arc::from("pkg.py"),
        first_lineno: 1,
        instructions: Box::new([
            Instruction::new(Opcode::CallKw, 0, 1, 0),
            Instruction::new(Opcode::CallKwEx, 2, 0, 0),
        ]),
        constants: vec![Constant::Value(Value::object_ptr(kwnames as *const ()))]
            .into_boxed_slice(),
        locals: Box::new([]),
        names: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        register_count: 3,
        flags: CodeFlags::NONE,
        line_table: Box::new([]),
        exception_table: Box::new([]),
        nested_code_objects: Box::new([]),
    });
    let code_value =
        Value::object_ptr(Box::into_raw(Box::new(CodeObjectView::new(code))) as *const ());

    let round_tripped =
        marshal_loads(&[marshal_dumps(&[code_value]).expect("code should marshal")])
            .expect("code should unmarshal");
    let loaded = code_view(round_tripped).code();
    let kwnames_ptr = match loaded.constants[0] {
        Constant::Value(value) => value.as_object_ptr().expect("keyword names ref"),
        Constant::BigInt(_) => panic!("keyword names constant should be a value"),
    };
    let kwnames = unsafe { &*(kwnames_ptr as *const KwNamesTuple) };

    assert_eq!(kwnames.len(), 2);
    assert_eq!(kwnames.get(0).map(AsRef::as_ref), Some("encoding"));
    assert_eq!(kwnames.get(1).map(AsRef::as_ref), Some("errors"));
}

#[test]
fn test_marshal_loads_ignores_trailing_bytes() {
    let result =
        decode_marshaled_value(&[TYPE_TRUE, 0x99, 0x98]).expect("trailing bytes should be ignored");
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_marshal_rejects_unsupported_values() {
    let err = marshal_dumps(&[bytes_value(b"abc"), Value::none(), Value::none()])
        .expect_err("extra marshal.dumps args should fail");
    assert!(
        err.to_string()
            .contains("takes from 1 to 2 positional arguments")
    );

    let object = crate::builtins::builtin_object(&[]).expect("object() should succeed");
    let err = marshal_dumps(&[object]).expect_err("unsupported values should fail");
    assert!(err.to_string().contains("unmarshallable object"));
}
