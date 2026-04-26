use super::*;

fn new_array_instance() -> Value {
    let instance = crate::builtins::allocate_heap_instance_for_class(&ARRAY_CLASS);
    Value::object_ptr(Box::into_raw(Box::new(instance)) as *const ())
}

fn bytes_value(bytes: &[u8]) -> Value {
    leak_object_value(BytesObject::from_slice(bytes))
}

fn slice_value(slice: SliceObject) -> Value {
    leak_object_value(slice)
}

#[test]
fn test_array_module_exposes_native_heap_type() {
    let module = ArrayModule::new();

    assert_eq!(module.name(), "array");
    assert!(module.get_attr("array").is_ok());
    assert!(module.get_attr("typecodes").is_ok());
    assert!(module.get_attr("_array_reconstructor").is_ok());
}

#[test]
fn test_array_init_sets_typecode_itemsize_and_empty_storage() {
    let value = new_array_instance();
    array_init(&[value, Value::string(intern("i"))]).expect("array init should succeed");

    let object = array_object_ref(value).expect("array instance");
    assert_eq!(
        value_as_string_ref(object.get_property("typecode").unwrap())
            .unwrap()
            .as_str(),
        "i"
    );
    assert_eq!(object.get_property("itemsize").unwrap().as_int(), Some(4));
    assert!(array_bytes(value).unwrap().is_empty());
}

#[test]
fn test_value_as_array_bytes_exports_raw_buffer() {
    let value = new_array_instance();
    array_init(&[
        value,
        Value::string(intern("B")),
        bytes_value(&[0x01, 0x02, 0xef]),
    ])
    .expect("array init should consume bytes");

    assert_eq!(
        value_as_array_bytes(value)
            .expect("array buffer export should succeed")
            .expect("array values should be recognized"),
        vec![0x01, 0x02, 0xef]
    );
    assert!(
        value_as_array_bytes(Value::none())
            .expect("non-array lookup should not fail")
            .is_none()
    );
}

#[test]
fn test_unsigned_byte_array_from_bytes_roundtrips_to_tobytes() {
    let value = new_array_instance();
    array_init(&[
        value,
        Value::string(intern("B")),
        bytes_value(b"socket payload"),
    ])
    .expect("byte array init should succeed");

    let out = array_tobytes(&[value]).expect("tobytes should succeed");
    assert_eq!(bytes_arg(out, "bytes").unwrap(), b"socket payload");
}

#[test]
fn test_int_array_from_list_uses_native_itemsize_and_iteration_values() {
    let value = new_array_instance();
    let list = leak_object_value(ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
    ]));
    array_init(&[value, Value::string(intern("i")), list]).expect("int array init should succeed");

    assert_eq!(array_values(value).unwrap().len(), 2);
    assert_eq!(
        array_getitem(&[value, Value::int(0).unwrap()])
            .unwrap()
            .as_int(),
        Some(10)
    );

    let iter = array_iter(&[value]).expect("array iter should return iterator");
    let mut iter = crate::builtins::get_iterator_mut(&iter).expect("iterator object");
    assert_eq!(iter.next().unwrap().as_int(), Some(10));
    assert_eq!(iter.next().unwrap().as_int(), Some(20));
    assert!(iter.next().is_none());
}

#[test]
fn test_array_getitem_slice_returns_same_typecode_array() {
    let value = new_array_instance();
    let list = leak_object_value(ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
        Value::int(40).unwrap(),
        Value::int(50).unwrap(),
    ]));
    array_init(&[value, Value::string(intern("i")), list]).expect("int array init should succeed");

    let forward = array_getitem(&[value, slice_value(SliceObject::full(1, 5, 2))])
        .expect("array slicing should succeed");
    assert_eq!(array_spec(forward).unwrap().typecode, 'i');
    assert_eq!(
        array_values(forward)
            .unwrap()
            .iter()
            .map(|value| value.as_int().unwrap())
            .collect::<Vec<_>>(),
        vec![20, 40]
    );

    let reversed = array_getitem(&[value, slice_value(SliceObject::new(None, None, Some(-1)))])
        .expect("reverse array slicing should succeed");
    assert_eq!(
        array_values(reversed)
            .unwrap()
            .iter()
            .map(|value| value.as_int().unwrap())
            .collect::<Vec<_>>(),
        vec![50, 40, 30, 20, 10]
    );
    assert_eq!(array_values(value).unwrap().len(), 5);
}

#[test]
fn test_frombytes_appends_complete_items_and_rejects_partial_items() {
    let value = new_array_instance();
    array_init(&[value, Value::string(intern("i"))]).expect("array init should succeed");

    array_frombytes(&[value, bytes_value(&1_i32.to_ne_bytes())])
        .expect("frombytes should append complete int item");
    assert_eq!(array_values(value).unwrap()[0].as_int(), Some(1));

    let err = array_frombytes(&[value, bytes_value(&[1, 2, 3])])
        .expect_err("frombytes should reject partial items");
    assert!(matches!(err, BuiltinError::ValueError(_)));
}
