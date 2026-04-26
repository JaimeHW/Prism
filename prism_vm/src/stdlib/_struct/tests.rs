use super::*;
use prism_core::intern::interned_by_ptr;

fn bytes_to_vec(value: Value) -> Vec<u8> {
    let ptr = value.as_object_ptr().expect("bytes-like object");
    unsafe { &*(ptr as *const BytesObject) }.to_vec()
}

fn bytearray_value(data: &[u8]) -> Value {
    leak_object_value(BytesObject::bytearray_from_slice(data))
}

fn tuple_items(value: Value) -> Vec<Value> {
    let ptr = value.as_object_ptr().expect("tuple object");
    unsafe { &*(ptr as *const TupleObject) }.as_slice().to_vec()
}

#[test]
fn test_calcsize_supports_pickle_formats() {
    for (format, expected) in [
        ("<B", 1_i64),
        ("<H", 2),
        ("<I", 4),
        ("<Q", 8),
        ("<i", 4),
        (">d", 8),
        ("P", std::mem::size_of::<usize>() as i64),
        ("n", std::mem::size_of::<isize>() as i64),
        ("N", std::mem::size_of::<usize>() as i64),
    ] {
        let result = calcsize_builtin(&[Value::string(intern(format))]).expect("calcsize");
        assert_eq!(result.as_int(), Some(expected));
    }
}

#[test]
fn test_calcsize_accepts_cpython_bytes_formats_and_native_alignment() {
    let zip_header =
        calcsize_builtin(&[bytes_value(b"<4s4H2LH")]).expect("zipfile byte format should parse");
    assert_eq!(zip_header.as_int(), Some(22));

    let aligned = calcsize_builtin(&[Value::string(intern("nP0n"))])
        .expect("native ssize, pointer, and zero-repeat alignment should parse");
    let expected = align_up(
        align_up(std::mem::size_of::<isize>(), std::mem::align_of::<usize>())
            + std::mem::size_of::<usize>(),
        std::mem::align_of::<isize>(),
    );
    assert_eq!(aligned.as_int(), Some(expected as i64));

    let err = calcsize_builtin(&[Value::string(intern("=n"))])
        .expect_err("native ssize_t code is only valid in native mode");
    assert!(err.to_string().contains("bad char in struct format"));

    let nul_err = calcsize_builtin(&[bytes_value(b"\0")]).expect_err("embedded null should fail");
    assert!(nul_err.to_string().contains("embedded null character"));
}

#[test]
fn test_pointer_format_is_native_only() {
    let value = Value::int(0x1234).unwrap();
    let packed = pack_builtin(&[Value::string(intern("P")), value]).expect("pack pointer");
    assert_eq!(bytes_to_vec(packed).len(), std::mem::size_of::<usize>());

    let unpacked = unpack_builtin(&[Value::string(intern("P")), packed]).expect("unpack");
    assert_eq!(tuple_items(unpacked), vec![value]);

    let err = calcsize_builtin(&[Value::string(intern("<P"))])
        .expect_err("standard pointer format should be rejected");
    assert!(err.to_string().contains("bad char in struct format"));
}

#[test]
fn test_pack_and_unpack_cover_pickle_formats() {
    let cases = [
        ("<B", Value::int(0x7f).unwrap(), vec![0x7f]),
        ("<H", Value::int(0x1234).unwrap(), vec![0x34, 0x12]),
        (
            "<I",
            Value::int(0x1234_5678).unwrap(),
            vec![0x78, 0x56, 0x34, 0x12],
        ),
        (
            "<Q",
            prism_runtime::types::int::bigint_to_value(BigInt::from(0x0102_0304_0506_0708_u64)),
            vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
        ),
        ("<i", Value::int(-2).unwrap(), vec![0xfe, 0xff, 0xff, 0xff]),
    ];

    for (format, input, expected) in cases {
        let packed = pack_builtin(&[Value::string(intern(format)), input]).expect("pack");
        assert_eq!(bytes_to_vec(packed), expected);

        let unpacked = unpack_builtin(&[Value::string(intern(format)), packed]).expect("unpack");
        let items = tuple_items(unpacked);
        assert_eq!(items.len(), 1);
        assert_eq!(
            prism_runtime::types::int::value_to_bigint(items[0]),
            prism_runtime::types::int::value_to_bigint(input)
        );
    }

    let packed = pack_builtin(&[Value::string(intern(">d")), Value::float(1.5)]).expect("pack");
    let unpacked = unpack_builtin(&[Value::string(intern(">d")), packed]).expect("unpack");
    let items = tuple_items(unpacked);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].as_float(), Some(1.5));
}

#[test]
fn test_pack_and_unpack_cover_cpython_bytes_formats() {
    let packed = pack_builtin(&[
        Value::string(intern("<4s4H2LH")),
        bytes_value(b"PK\x05\x06"),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
        Value::int(5).unwrap(),
        Value::int(6).unwrap(),
        Value::int(7).unwrap(),
    ])
    .expect("pack zip header");
    assert_eq!(bytes_to_vec(packed).len(), 22);

    assert_eq!(
        bytes_to_vec(pack_builtin(&[Value::string(intern("4s")), bytes_value(b"xy")]).unwrap()),
        b"xy\0\0"
    );
    assert_eq!(
        bytes_to_vec(pack_builtin(&[Value::string(intern("3s")), bytes_value(b"wxyz")]).unwrap()),
        b"wxy"
    );
    assert_eq!(
        bytes_to_vec(
            pack_builtin(&[Value::string(intern("3p")), bytearray_value(b"abcd")]).unwrap()
        ),
        b"\x02ab"
    );
    assert_eq!(
        bytes_to_vec(pack_builtin(&[Value::string(intern("c")), bytes_value(b"z")]).unwrap()),
        b"z"
    );

    let unpacked = unpack_builtin(&[Value::string(intern("2s3pc")), bytes_value(b"ab\x02cdZ")])
        .expect("unpack string formats");
    let items = tuple_items(unpacked);
    assert_eq!(bytes_to_vec(items[0]), b"ab");
    assert_eq!(bytes_to_vec(items[1]), b"cd");
    assert_eq!(bytes_to_vec(items[2]), b"Z");
}

#[test]
fn test_pack_into_unpack_from_and_iter_unpack_work_with_offsets() {
    let target = leak_object_value(BytesObject::repeat_with_type(0, 10, TypeId::BYTEARRAY));
    pack_into_builtin(&[
        Value::string(intern("<I")),
        target,
        Value::int(2).unwrap(),
        Value::int(0x1122_3344).unwrap(),
    ])
    .expect("pack_into");

    assert_eq!(
        bytes_to_vec(target),
        vec![0, 0, 0x44, 0x33, 0x22, 0x11, 0, 0, 0, 0]
    );

    let unpacked =
        unpack_from_builtin(&[Value::string(intern("<I")), target, Value::int(2).unwrap()])
            .expect("unpack_from");
    assert_eq!(
        tuple_items(unpacked),
        vec![Value::int(0x1122_3344).unwrap()]
    );

    let iter = iter_unpack_builtin(&[
        Value::string(intern("<H")),
        bytes_value(&[1, 0, 2, 0, 3, 0, 4, 0]),
    ])
    .expect("iter_unpack");
    let first = crate::builtins::builtin_next(&[iter]).expect("first tuple");
    let second = crate::builtins::builtin_next(&[iter]).expect("second tuple");
    assert_eq!(tuple_items(first), vec![Value::int(1).unwrap()]);
    assert_eq!(tuple_items(second), vec![Value::int(2).unwrap()]);
}

#[test]
fn test_struct_constructor_binds_format_and_methods() {
    let receiver = struct_constructor(&[bytes_value(b"<I")]).expect("Struct()");
    let ptr = receiver.as_object_ptr().expect("struct helper object");
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    let format = shaped.get_property("format").expect("format");
    assert_eq!(
        interned_by_ptr(format.as_string_object_ptr().unwrap() as *const u8)
            .expect("interned format")
            .as_str(),
        "<I"
    );
    assert_eq!(shaped.get_property("size").unwrap().as_int(), Some(4));

    let pack_value = shaped.get_property("pack").expect("pack method");
    let pack_ptr = pack_value.as_object_ptr().expect("pack builtin object");
    let pack_builtin = unsafe { &*(pack_ptr as *const BuiltinFunctionObject) };
    let packed = pack_builtin
        .call(&[Value::int(7).unwrap()])
        .expect("bound pack");
    assert_eq!(bytes_to_vec(packed), vec![7, 0, 0, 0]);
}
