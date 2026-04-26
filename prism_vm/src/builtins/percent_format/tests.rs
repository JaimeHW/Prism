use super::{percent_format_bytes, percent_format_string};
use crate::builtins::BuiltinError;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::{BytesObject, value_as_bytes_ref};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;

fn render(format: &str, arguments: Value) -> String {
    let value = percent_format_string(format, arguments).expect("format should succeed");
    value_as_string_ref(value)
        .expect("result should be a string")
        .as_str()
        .to_string()
}

fn render_bytes(format: &[u8], arguments: Value) -> Vec<u8> {
    let template = BytesObject::from_slice(format);
    let value = percent_format_bytes(&template, arguments).expect("format should succeed");
    value_as_bytes_ref(value)
        .expect("result should be bytes")
        .as_bytes()
        .to_vec()
}

fn boxed_value<T>(object: T) -> (Value, *mut T) {
    let ptr = Box::into_raw(Box::new(object));
    (Value::object_ptr(ptr as *const ()), ptr)
}

unsafe fn drop_boxed<T>(ptr: *mut T) {
    drop(unsafe { Box::from_raw(ptr) });
}

#[test]
fn test_percent_format_single_string_argument() {
    assert_eq!(
        render("hello %s", Value::string(intern("world"))),
        "hello world"
    );
}

#[test]
fn test_percent_format_tuple_arguments() {
    let tuple = TupleObject::from_slice(&[Value::string(intern("value")), Value::int(7).unwrap()]);
    let (value, ptr) = boxed_value(tuple);
    assert_eq!(render("%s = %d", value), "value = 7");
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_percent_format_mapping_argument() {
    let mut dict = DictObject::new();
    dict.set(
        Value::string(intern("prog")),
        Value::string(intern("prism")),
    );
    let (value, ptr) = boxed_value(dict);
    assert_eq!(render("%(prog)s", value), "prism");
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_percent_format_repr_and_percent_escape() {
    assert_eq!(render("%r %%", Value::string(intern("x"))), "'x' %");
}

#[test]
fn test_percent_format_string_precision_and_width() {
    assert_eq!(render("%-6.3s", Value::string(intern("python"))), "pyt   ");
}

#[test]
fn test_percent_format_integer_flags() {
    assert_eq!(render("%#06x", Value::int(31).unwrap()), "0x001f");
}

#[test]
fn test_percent_format_char_from_integer() {
    assert_eq!(render("%c", Value::int(65).unwrap()), "A");
}

#[test]
fn test_percent_format_rejects_missing_arguments() {
    let err = percent_format_string("%s %s", Value::string(intern("only")))
        .expect_err("format should fail");
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("not enough arguments"));
}

#[test]
fn test_percent_format_rejects_extra_tuple_arguments() {
    let tuple = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let (value, ptr) = boxed_value(tuple);
    let err = percent_format_string("%d", value).expect_err("format should fail");
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("not all arguments converted"));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_percent_format_bytes_inserts_raw_byte_arguments() {
    let (argument, argument_ptr) = boxed_value(BytesObject::from_slice(b"GLIBC_2.9"));
    assert_eq!(
        render_bytes(b"[xxx%sxxx]", argument),
        b"[xxxGLIBC_2.9xxx]".to_vec()
    );
    unsafe { drop_boxed(argument_ptr) };
}

#[test]
fn test_percent_format_bytes_honors_tuple_width_precision_and_numeric_specs() {
    let (argument, argument_ptr) = boxed_value(BytesObject::from_slice(b"abcdef"));
    let tuple = TupleObject::from_slice(&[argument, Value::int(31).unwrap()]);
    let (tuple_value, tuple_ptr) = boxed_value(tuple);
    assert_eq!(
        render_bytes(b"%-5.3s:%#06x", tuple_value),
        b"abc  :0x001f".to_vec()
    );
    unsafe {
        drop_boxed(tuple_ptr);
        drop_boxed(argument_ptr);
    }
}

#[test]
fn test_percent_format_bytearray_preserves_receiver_type() {
    let template = BytesObject::bytearray_from_slice(b"%c:%b");
    let (argument, argument_ptr) = boxed_value(BytesObject::from_slice(b"z"));
    let tuple = TupleObject::from_slice(&[Value::int(65).unwrap(), argument]);
    let (tuple_value, tuple_ptr) = boxed_value(tuple);
    let result =
        percent_format_bytes(&template, tuple_value).expect("bytearray format should succeed");
    assert_eq!(value_as_bytes_ref(result).unwrap().as_bytes(), b"A:z");
    assert_eq!(
        value_as_bytes_ref(result).unwrap().header.type_id,
        TypeId::BYTEARRAY
    );

    unsafe {
        drop_boxed(tuple_ptr);
        drop_boxed(argument_ptr);
    }
}
