use super::*;
use crate::builtins::itertools::{builtin_iter, builtin_next, builtin_range};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::intern::intern;
use prism_core::intern::interned_by_ptr;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::type_builtins::builtin_class_mro;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::string::StringObject;
use std::sync::Arc;

fn value_to_string(value: Value) -> String {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .expect("tagged string should provide pointer");
        return interned_by_ptr(ptr as *const u8)
            .expect("interned pointer should resolve")
            .as_str()
            .to_string();
    }

    let ptr = value
        .as_object_ptr()
        .expect("string value should be object-backed");
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::STR);
    let s = unsafe { &*(ptr as *const StringObject) };
    s.as_str().to_string()
}

fn value_to_bytes(value: Value) -> Vec<u8> {
    let ptr = value
        .as_object_ptr()
        .expect("bytes result should be heap allocated");
    let bytes = unsafe { &*(ptr as *const BytesObject) };
    bytes.to_vec()
}

fn boxed_value<T>(obj: T) -> (Value, *mut T) {
    let ptr = Box::into_raw(Box::new(obj));
    (Value::object_ptr(ptr as *const ()), ptr)
}

unsafe fn drop_boxed<T>(ptr: *mut T) {
    drop(unsafe { Box::from_raw(ptr) });
}

fn class_value(class: PyClassObject) -> (Value, *const PyClassObject) {
    let class = Arc::new(class);
    let ptr = Arc::into_raw(class);
    (Value::object_ptr(ptr as *const ()), ptr)
}

unsafe fn drop_class(ptr: *const PyClassObject) {
    drop(unsafe { Arc::from_raw(ptr) });
}

fn namespace_builtin(
    namespace: &mut DictObject,
    name: &str,
    func: fn(&[Value]) -> Result<Value, BuiltinError>,
) -> *mut crate::builtins::BuiltinFunctionObject {
    let builtin = Box::new(crate::builtins::BuiltinFunctionObject::new(
        Arc::from(name),
        func,
    ));
    let ptr = Box::into_raw(builtin);
    namespace.set(
        Value::string(intern(name)),
        Value::object_ptr(ptr as *const ()),
    );
    ptr
}

fn heap_class(
    name: &str,
) -> (
    Value,
    *const PyClassObject,
    *mut TupleObject,
    *mut DictObject,
) {
    let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let class_value = builtin_type(&[Value::string(intern(name)), bases_value, namespace_value])
        .expect("type() should build a heap class");
    let class_ptr = class_value
        .as_object_ptr()
        .expect("heap class should be object-backed") as *const PyClassObject;
    (
        class_value,
        class_ptr,
        bases_ptr,
        namespace_ptr as *mut DictObject,
    )
}

fn heap_class_with_metaclass(
    metaclass: Value,
    name: &str,
) -> (
    Value,
    *const PyClassObject,
    *mut TupleObject,
    *mut DictObject,
) {
    let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let class_value = builtin_type_new(&[
        metaclass,
        Value::string(intern(name)),
        bases_value,
        namespace_value,
    ])
    .expect("type.__new__ should build a heap class");
    let class_ptr = class_value
        .as_object_ptr()
        .expect("heap class should be object-backed") as *const PyClassObject;
    (
        class_value,
        class_ptr,
        bases_ptr,
        namespace_ptr as *mut DictObject,
    )
}

fn heap_metaclass_with_hook(
    name: &str,
    hook_name: &str,
    hook: fn(&[Value]) -> Result<Value, BuiltinError>,
) -> (
    Value,
    *const PyClassObject,
    *mut TupleObject,
    *mut DictObject,
    *mut crate::builtins::BuiltinFunctionObject,
) {
    let type_type = crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE);
    let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[type_type]));
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let namespace = unsafe { &mut *namespace_ptr };
    let hook_ptr = namespace_builtin(namespace, hook_name, hook);
    let metaclass = builtin_type(&[Value::string(intern(name)), bases_value, namespace_value])
        .expect("type() should build a heap metaclass");
    let metaclass_ptr = metaclass
        .as_object_ptr()
        .expect("heap metaclass should be object-backed")
        as *const PyClassObject;
    (
        metaclass,
        metaclass_ptr,
        bases_ptr,
        namespace_ptr as *mut DictObject,
        hook_ptr,
    )
}

#[test]
fn test_object_new_allocates_native_dict_backing_for_heap_dict_subclass() {
    let class = PyClassObject::new(
        intern("DictSubclass"),
        &[ClassId(TypeId::DICT.raw())],
        |id| {
            (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                builtin_class_mro(TypeId::from_raw(id.0))
                    .into_iter()
                    .collect()
            })
        },
    )
    .expect("dict subclass should build");
    let (class_value, class_ptr) = class_value(class);

    let result = builtin_object_new(&[class_value]).expect("object.__new__ should succeed");
    let result_ptr = result
        .as_object_ptr()
        .expect("object.__new__ should return a heap instance");
    let shaped = unsafe { &*(result_ptr as *const ShapedObject) };

    assert!(shaped.has_dict_backing());

    unsafe {
        drop_boxed(result_ptr as *mut ShapedObject);
        drop_class(class_ptr);
    }
}

#[test]
fn test_object_new_allocates_native_list_backing_for_heap_list_subclass() {
    let class = PyClassObject::new(
        intern("ListSubclass"),
        &[ClassId(TypeId::LIST.raw())],
        |id| {
            (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                builtin_class_mro(TypeId::from_raw(id.0))
                    .into_iter()
                    .collect()
            })
        },
    )
    .expect("list subclass should build");
    let (class_value, class_ptr) = class_value(class);

    let result = builtin_object_new(&[class_value]).expect("object.__new__ should succeed");
    let result_ptr = result
        .as_object_ptr()
        .expect("object.__new__ should return a heap instance");
    let shaped = unsafe { &*(result_ptr as *const ShapedObject) };

    assert!(shaped.has_list_backing());

    unsafe {
        drop_boxed(result_ptr as *mut ShapedObject);
        drop_class(class_ptr);
    }
}

#[test]
fn test_int_from_int() {
    let result = builtin_int(&[Value::int(42).unwrap()]).unwrap();
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_int_from_float() {
    let result = builtin_int(&[Value::float(3.9)]).unwrap();
    assert_eq!(result.as_int(), Some(3));
}

#[test]
fn test_int_from_ascii_string_default_base() {
    let result = builtin_int(&[Value::string(intern("42"))]).unwrap();
    assert_eq!(result.as_int(), Some(42));

    let signed = builtin_int(&[Value::string(intern("  -17 "))]).unwrap();
    assert_eq!(signed.as_int(), Some(-17));
}

#[test]
fn test_int_from_bytes_and_bytearray_default_base() {
    let (bytes_value, bytes_ptr) = boxed_value(BytesObject::from_slice(b"01"));
    let bytes_result = builtin_int(&[bytes_value]).unwrap();
    assert_eq!(bytes_result.as_int(), Some(1));

    let (bytearray_value, bytearray_ptr) = boxed_value(BytesObject::bytearray_from_slice(b"255"));
    let bytearray_result = builtin_int(&[bytearray_value]).unwrap();
    assert_eq!(bytearray_result.as_int(), Some(255));

    unsafe {
        drop_boxed(bytes_ptr);
        drop_boxed(bytearray_ptr);
    }
}

#[test]
fn test_int_with_explicit_base_parses_prefixed_text() {
    let value = builtin_int(&[Value::string(intern("0x_FF")), Value::int(16).unwrap()]).unwrap();
    assert_eq!(value.as_int(), Some(255));

    let (bytes_value, bytes_ptr) = boxed_value(BytesObject::from_slice(b"0b_1010"));
    let binary = builtin_int(&[bytes_value, Value::int(0).unwrap()]).unwrap();
    assert_eq!(binary.as_int(), Some(10));

    unsafe {
        drop_boxed(bytes_ptr);
    }
}

#[test]
fn test_int_to_bytes_defaults_and_zero_length() {
    let default_bytes = builtin_int_to_bytes(&[Value::int(0).unwrap()], &[])
        .expect("0.to_bytes() should use CPython defaults");
    assert_eq!(value_to_bytes(default_bytes), vec![0]);

    let empty_bytes = builtin_int_to_bytes(&[Value::int(0).unwrap(), Value::int(0).unwrap()], &[])
        .expect("0.to_bytes(0) should succeed");
    assert_eq!(value_to_bytes(empty_bytes), Vec::<u8>::new());
}

#[test]
fn test_int_to_bytes_supports_signed_little_endian_and_bool_receivers() {
    let little_endian = builtin_int_to_bytes(
        &[
            Value::int(0x1234).unwrap(),
            Value::int(2).unwrap(),
            Value::string(intern("little")),
        ],
        &[],
    )
    .expect("little-endian encoding should succeed");
    assert_eq!(value_to_bytes(little_endian), vec![0x34, 0x12]);

    let signed_negative = builtin_int_to_bytes(
        &[
            Value::int(-1).unwrap(),
            Value::int(1).unwrap(),
            Value::string(intern("big")),
        ],
        &[("signed", Value::bool(true))],
    )
    .expect("signed encoding should support negative values");
    assert_eq!(value_to_bytes(signed_negative), vec![0xFF]);

    let boolean = builtin_int_to_bytes(&[Value::bool(true)], &[])
        .expect("bool should inherit int.to_bytes()");
    assert_eq!(value_to_bytes(boolean), vec![1]);
}

#[test]
fn test_int_to_bytes_reports_cpython_compatible_argument_errors() {
    let byteorder_err = builtin_int_to_bytes(
        &[
            Value::int(1).unwrap(),
            Value::int(1).unwrap(),
            Value::int(5).unwrap(),
        ],
        &[],
    )
    .expect_err("non-string byteorder should fail");
    assert!(
        byteorder_err
            .to_string()
            .contains("to_bytes() argument 'byteorder' must be str, not int")
    );

    let length_err =
        builtin_int_to_bytes(&[Value::int(1).unwrap(), Value::string(intern("x"))], &[])
            .expect_err("non-integer length should fail");
    assert!(
        length_err
            .to_string()
            .contains("'str' object cannot be interpreted as an integer")
    );

    let unsigned_negative = builtin_int_to_bytes(
        &[
            Value::int(-1).unwrap(),
            Value::int(1).unwrap(),
            Value::string(intern("big")),
        ],
        &[],
    )
    .expect_err("negative unsigned encoding should fail");
    assert!(
        unsigned_negative
            .to_string()
            .contains("can't convert negative int to unsigned")
    );

    let signed_overflow = builtin_int_to_bytes(
        &[
            Value::int(128).unwrap(),
            Value::int(1).unwrap(),
            Value::string(intern("big")),
        ],
        &[("signed", Value::bool(true))],
    )
    .expect_err("signed one-byte overflow should fail");
    assert!(
        signed_overflow
            .to_string()
            .contains("int too big to convert")
    );
}

#[test]
fn test_int_rejects_non_string_with_explicit_base() {
    let err = builtin_int(&[Value::int(12).unwrap(), Value::int(10).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("explicit base"));
}

#[test]
fn test_float_from_int() {
    let result = builtin_float(&[Value::int(42).unwrap()]).unwrap();
    assert_eq!(result.as_float(), Some(42.0));
}

#[test]
fn test_bool_truthy() {
    let result = builtin_bool(&[Value::int(1).unwrap()]).unwrap();
    assert!(result.is_truthy());

    let result = builtin_bool(&[Value::int(0).unwrap()]).unwrap();
    assert!(!result.is_truthy());
}

#[test]
fn test_str_empty() {
    let result = builtin_str(&[]).unwrap();
    assert_eq!(value_to_string(result), "");
}

#[test]
fn test_str_renders_exception_display_text() {
    let exc = crate::builtins::get_exception_type("ValueError")
        .expect("ValueError should exist")
        .construct(&[Value::string(intern("boom"))]);
    let rendered = builtin_str(&[exc]).expect("str() should render exceptions");

    assert_eq!(value_to_string(rendered), "boom");
}

#[test]
fn test_str_identity_for_tagged_string() {
    let value = Value::string(intern("alpha"));
    let result = builtin_str(&[value]).unwrap();
    assert_eq!(result, value);
}

#[test]
fn test_str_from_numeric_uses_text_form() {
    let int_result = builtin_str(&[Value::int(42).unwrap()]).unwrap();
    assert_eq!(value_to_string(int_result), "42");

    let float_result = builtin_str(&[Value::float(3.5)]).unwrap();
    assert_eq!(value_to_string(float_result), "3.5");
}

#[test]
fn test_str_decodes_bytes_with_encoding_and_errors() {
    let encoded = to_object_value(BytesObject::from_slice("café".as_bytes()));
    let decoded = builtin_str(&[
        encoded,
        Value::string(intern("utf-8")),
        Value::string(intern("strict")),
    ])
    .expect("str(bytes, encoding, errors) should decode bytes");
    assert_eq!(value_to_string(decoded), "café");

    let bytearray = to_object_value(BytesObject::bytearray_from_slice(&[0x41, 0xFF, 0x42]));
    let escaped = builtin_str(&[
        bytearray,
        Value::string(intern("ascii")),
        Value::string(intern("backslashreplace")),
    ])
    .expect("str(bytearray, encoding, errors) should decode bytearray");
    assert_eq!(value_to_string(escaped), r"A\xffB");
}

#[test]
fn test_str_decode_form_validates_argument_types_and_source_kind() {
    let bad_encoding = builtin_str(&[Value::int(1).unwrap(), Value::int(2).unwrap()])
        .expect_err("encoding must be a string");
    assert!(
        bad_encoding
            .to_string()
            .contains("str() argument 'encoding' must be str, not int")
    );

    let bad_errors = builtin_str(&[
        to_object_value(BytesObject::from_slice(b"x")),
        Value::string(intern("utf-8")),
        Value::int(2).unwrap(),
    ])
    .expect_err("errors must be a string");
    assert!(
        bad_errors
            .to_string()
            .contains("str() argument 'errors' must be str, not int")
    );

    let decoding_str = builtin_str(&[Value::string(intern("x")), Value::string(intern("utf-8"))])
        .expect_err("decoding a string should fail");
    assert!(
        decoding_str
            .to_string()
            .contains("decoding str is not supported")
    );

    let non_bytes = builtin_str(&[Value::int(1).unwrap(), Value::string(intern("utf-8"))])
        .expect_err("non-bytes decode source should fail");
    assert!(
        non_bytes
            .to_string()
            .contains("decoding to str: need a bytes-like object, int found")
    );
}

#[test]
fn test_str_constructor_accepts_keyword_decode_arguments() {
    let result = call_builtin_type_kw(
        TypeId::STR,
        &[],
        &[
            ("object", to_object_value(BytesObject::from_slice(b"foo"))),
            ("errors", Value::string(intern("strict"))),
        ],
    )
    .expect("str(object=bytes, errors=...) should default to utf-8");
    assert_eq!(value_to_string(result), "foo");

    let empty = call_builtin_type_kw(
        TypeId::STR,
        &[],
        &[
            ("encoding", Value::string(intern("utf-8"))),
            ("errors", Value::string(intern("ignore"))),
        ],
    )
    .expect("str() should still default object to empty string");
    assert_eq!(value_to_string(empty), "");
}

#[test]
fn test_str_constructor_keyword_validation_matches_cpython() {
    let duplicate = call_builtin_type_kw(
        TypeId::STR,
        &[to_object_value(BytesObject::from_slice(b"foo"))],
        &[("object", to_object_value(BytesObject::from_slice(b"bar")))],
    )
    .expect_err("duplicate object keyword should fail");
    assert!(
        duplicate
            .to_string()
            .contains("str() got multiple values for argument 'object'")
    );

    let unexpected =
        call_builtin_type_kw(TypeId::STR, &[], &[("bogus", Value::string(intern("x")))])
            .expect_err("unexpected keyword should fail");
    assert!(
        unexpected
            .to_string()
            .contains("str() got an unexpected keyword argument 'bogus'")
    );
}

#[test]
fn test_property_constructor_accepts_doc_keyword() {
    let result = call_builtin_type_kw(
        TypeId::PROPERTY,
        &[],
        &[("doc", Value::string(intern("docs")))],
    )
    .expect("property(doc=...) should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("property() should return descriptor object");
    let descriptor = unsafe { &*(ptr as *const PropertyDescriptor) };
    assert_eq!(descriptor.doc(), Some(Value::string(intern("docs"))));
    unsafe { drop_boxed(ptr as *mut PropertyDescriptor) };
}

#[test]
fn test_property_constructor_rejects_duplicate_doc_argument() {
    let err = call_builtin_type_kw(
        TypeId::PROPERTY,
        &[
            Value::none(),
            Value::none(),
            Value::none(),
            Value::string(intern("positional")),
        ],
        &[("doc", Value::string(intern("keyword")))],
    )
    .expect_err("duplicate property doc should fail");
    assert!(
        err.to_string()
            .contains("property() got multiple values for argument 'doc'")
    );
}

#[test]
fn test_dict_constructor_accepts_keyword_arguments() {
    let result = call_builtin_type_kw(
        TypeId::DICT,
        &[],
        &[
            ("alpha", Value::int(1).unwrap()),
            ("beta", Value::int(2).unwrap()),
        ],
    )
    .expect("dict(alpha=1, beta=2) should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("dict() should return a dict object");
    let dict = unsafe { &*(ptr as *const DictObject) };

    assert_eq!(
        dict.get(Value::string(intern("alpha")))
            .and_then(|value| value.as_int()),
        Some(1)
    );
    assert_eq!(
        dict.get(Value::string(intern("beta")))
            .and_then(|value| value.as_int()),
        Some(2)
    );
}

#[test]
fn test_dict_constructor_keywords_override_positional_mapping_entries() {
    let mut source = DictObject::new();
    source.set(Value::string(intern("alpha")), Value::int(1).unwrap());
    source.set(Value::string(intern("gamma")), Value::int(3).unwrap());
    let source_ptr = Box::into_raw(Box::new(source));

    let result = call_builtin_type_kw(
        TypeId::DICT,
        &[Value::object_ptr(source_ptr as *const ())],
        &[("alpha", Value::int(9).unwrap())],
    )
    .expect("dict(mapping, alpha=9) should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("dict() should return a dict object");
    let dict = unsafe { &*(ptr as *const DictObject) };

    assert_eq!(
        dict.get(Value::string(intern("alpha")))
            .and_then(|value| value.as_int()),
        Some(9)
    );
    assert_eq!(
        dict.get(Value::string(intern("gamma")))
            .and_then(|value| value.as_int()),
        Some(3)
    );

    unsafe {
        drop(Box::from_raw(source_ptr));
    }
}

#[test]
fn test_slice_stop_only_constructor() {
    let result = builtin_slice(&[Value::int(5).unwrap()]).expect("slice(stop) should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("slice() should return object");
    let slice = unsafe { &*(ptr as *const SliceObject) };

    assert_eq!(slice.start(), None);
    assert_eq!(slice.stop(), Some(5));
    assert_eq!(slice.step(), None);

    unsafe { drop_boxed(ptr as *mut SliceObject) };
}

#[test]
fn test_slice_full_constructor() {
    let result = builtin_slice(&[
        Value::int(1).unwrap(),
        Value::int(9).unwrap(),
        Value::int(2).unwrap(),
    ])
    .expect("slice(start, stop, step) should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("slice() should return object");
    let slice = unsafe { &*(ptr as *const SliceObject) };

    assert_eq!(slice.start(), Some(1));
    assert_eq!(slice.stop(), Some(9));
    assert_eq!(slice.step(), Some(2));

    unsafe { drop_boxed(ptr as *mut SliceObject) };
}

#[test]
fn test_slice_constructor_accepts_none_components() {
    let result = builtin_slice(&[Value::none(), Value::int(4).unwrap(), Value::none()])
        .expect("slice(None, 4, None) should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("slice() should return object");
    let slice = unsafe { &*(ptr as *const SliceObject) };

    assert_eq!(slice.start(), None);
    assert_eq!(slice.stop(), Some(4));
    assert_eq!(slice.step(), None);

    unsafe { drop_boxed(ptr as *mut SliceObject) };
}

#[test]
fn test_slice_constructor_rejects_zero_step() {
    let err = builtin_slice(&[
        Value::int(1).unwrap(),
        Value::int(5).unwrap(),
        Value::int(0).unwrap(),
    ])
    .expect_err("slice(..., step=0) should fail");
    assert!(err.to_string().contains("slice step cannot be zero"));
}

#[test]
fn test_slice_constructor_rejects_non_integer_components() {
    let err = builtin_slice(&[Value::float(3.5)]).expect_err("slice(float) should fail");
    assert!(
        err.to_string()
            .contains("slice indices must be integers or None")
    );
}

#[test]
fn test_list_empty() {
    let result = builtin_list(&[]).unwrap();
    let ptr = result.as_object_ptr().expect("list() should return object");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 0);
    unsafe { drop_boxed(ptr as *mut ListObject) };
}

#[test]
fn test_list_from_range() {
    let range = builtin_range(&[Value::int(0).unwrap(), Value::int(4).unwrap()]).unwrap();
    let list_value = builtin_list(&[range]).unwrap();
    let ptr = list_value.as_object_ptr().unwrap();
    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 4);
    assert_eq!(list.get(0).unwrap().as_int(), Some(0));
    assert_eq!(list.get(3).unwrap().as_int(), Some(3));
    unsafe { drop_boxed(ptr as *mut ListObject) };
}

#[test]
fn test_list_consumes_iterator_state() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let (list_value, list_ptr) = boxed_value(list);
    let iterator = builtin_iter(&[list_value]).unwrap();

    let built = builtin_list(&[iterator]).unwrap();
    let built_ptr = built.as_object_ptr().unwrap();
    let built_list = unsafe { &*(built_ptr as *const ListObject) };
    assert_eq!(built_list.len(), 3);

    let next_result = builtin_next(&[iterator]);
    assert!(next_result.is_err());

    unsafe { drop_boxed(built_ptr as *mut ListObject) };
    unsafe { drop_boxed(list_ptr) };
    unsafe { drop_boxed(iterator.as_object_ptr().unwrap() as *mut IteratorObject) };
}

#[test]
fn test_tuple_empty_and_from_iterable() {
    let empty = builtin_tuple(&[]).unwrap();
    let empty_ptr = empty.as_object_ptr().unwrap();
    let empty_tuple = unsafe { &*(empty_ptr as *const TupleObject) };
    assert_eq!(empty_tuple.len(), 0);
    unsafe { drop_boxed(empty_ptr as *mut TupleObject) };

    let source = ListObject::from_slice(&[Value::int(9).unwrap(), Value::int(8).unwrap()]);
    let (source_value, source_ptr) = boxed_value(source);
    let tuple_value = builtin_tuple(&[source_value]).unwrap();
    let tuple_ptr = tuple_value.as_object_ptr().unwrap();
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.get(0).unwrap().as_int(), Some(9));
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(8));
    unsafe { drop_boxed(tuple_ptr as *mut TupleObject) };
    unsafe { drop_boxed(source_ptr) };
}

#[test]
fn test_tuple_new_exposed_on_type_object_builds_tuple_instances() {
    let tuple_type = builtin_type_object_for_type_id(TypeId::TUPLE);
    let tuple_new = builtin_getattr(&[tuple_type, Value::string(intern("__new__"))])
        .expect("tuple.__new__ should resolve");
    let tuple_new_ptr = tuple_new
        .as_object_ptr()
        .expect("tuple.__new__ should be a builtin function object");
    let builtin = unsafe { &*(tuple_new_ptr as *const crate::builtins::BuiltinFunctionObject) };

    let (source_value, source_ptr) = boxed_value(ListObject::from_slice(&[
        Value::int(4).unwrap(),
        Value::int(5).unwrap(),
    ]));

    let result = builtin
        .call(&[tuple_type, source_value])
        .expect("tuple.__new__(tuple, iterable) should succeed");
    let result_ptr = result
        .as_object_ptr()
        .expect("tuple.__new__ should return a tuple object");
    let tuple = unsafe { &*(result_ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.get(0).unwrap().as_int(), Some(4));
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(5));

    unsafe { drop_boxed(result_ptr as *mut TupleObject) };
    unsafe { drop_boxed(source_ptr) };
}

#[test]
fn test_tuple_new_returns_empty_tuple_without_iterable() {
    let tuple_type = builtin_type_object_for_type_id(TypeId::TUPLE);

    let result = builtin_tuple_new(&[tuple_type]).expect("tuple.__new__(tuple) should succeed");
    let result_ptr = result
        .as_object_ptr()
        .expect("tuple.__new__(tuple) should return a tuple");
    let tuple = unsafe { &*(result_ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 0);

    unsafe { drop_boxed(result_ptr as *mut TupleObject) };
}

#[test]
fn test_tuple_new_builds_tuple_backed_subclass_instances() {
    let (tuple_instance, tuple_instance_ptr) = boxed_value(TupleObject::empty());
    let tuple_type = builtin_type(&[tuple_instance]).unwrap();
    let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[tuple_type]));
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());

    let subclass = builtin_type(&[
        Value::string(intern("TupleChild")),
        bases_value,
        namespace_value,
    ])
    .expect("tuple subclass should be constructible");
    let (source_value, source_ptr) = boxed_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]));

    let result = builtin_tuple_new(&[subclass, source_value])
        .expect("tuple.__new__ should allocate tuple-backed subclass instances");
    let result_ptr = result
        .as_object_ptr()
        .expect("tuple subclass instance should be heap allocated");
    assert_eq!(crate::ops::objects::extract_type_id(result_ptr), {
        let class = unsafe { &*(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
        class.class_type_id()
    });
    let shaped = unsafe { &*(result_ptr as *const ShapedObject) };
    let tuple = shaped
        .tuple_backing()
        .expect("tuple subclass should retain native tuple storage");
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.as_slice()[0].as_int(), Some(1));
    assert_eq!(tuple.as_slice()[1].as_int(), Some(2));

    let empty_result = builtin_object_new(&[subclass])
        .expect("object.__new__ should allocate tuple-backed subclass instances");
    let empty_ptr = empty_result
        .as_object_ptr()
        .expect("empty tuple subclass should be heap allocated");
    let empty_shaped = unsafe { &*(empty_ptr as *const ShapedObject) };
    assert_eq!(
        empty_shaped
            .tuple_backing()
            .expect("tuple subclass should have empty tuple backing")
            .len(),
        0
    );

    unsafe { drop_boxed(tuple_instance_ptr) };
    unsafe { drop_boxed(source_ptr) };
    unsafe { drop_boxed(result_ptr as *mut ShapedObject) };
    unsafe { drop_boxed(empty_ptr as *mut ShapedObject) };
    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
}

#[test]
fn test_module_type_constructor_builds_module_objects() {
    let doc = Value::int(7).unwrap();
    let result = call_builtin_type(
        TypeId::MODULE,
        &[Value::string(intern("dynamic_module")), doc],
    )
    .expect("module(name, doc) should construct a module object");
    let ptr = result
        .as_object_ptr()
        .expect("module constructor should return a heap object");
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::MODULE);

    let module = unsafe { &*(ptr as *const ModuleObject) };
    assert_eq!(module.name(), "dynamic_module");
    assert_eq!(
        module.get_attr("__name__").map(value_to_string).as_deref(),
        Some("dynamic_module")
    );
    assert_eq!(module.get_attr("__doc__"), Some(doc));
    assert!(module.get_attr("__loader__").unwrap().is_none());
    assert!(module.get_attr("__package__").unwrap().is_none());
    assert!(module.get_attr("__spec__").unwrap().is_none());
}

#[test]
fn test_module_new_exposed_on_type_object_builds_module_objects() {
    let module_type = builtin_type_object_for_type_id(TypeId::MODULE);
    let module_new = builtin_getattr(&[module_type, Value::string(intern("__new__"))])
        .expect("module.__new__ should resolve");
    let module_new_ptr = module_new
        .as_object_ptr()
        .expect("module.__new__ should be a builtin function object");
    let builtin = unsafe { &*(module_new_ptr as *const crate::builtins::BuiltinFunctionObject) };

    let result = builtin
        .call(&[module_type, Value::string(intern("via_new"))])
        .expect("module.__new__(module, name) should construct a module");
    let ptr = result
        .as_object_ptr()
        .expect("module.__new__ should return a heap object");
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::MODULE);

    let module = unsafe { &*(ptr as *const ModuleObject) };
    assert_eq!(module.name(), "via_new");
    assert!(module.get_attr("__doc__").unwrap().is_none());
}

#[test]
fn test_core_builtin_new_wrappers_dispatch_to_runtime_constructors() {
    let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
    assert_eq!(
        builtin_int_new(&[int_type, Value::int(7).unwrap()])
            .unwrap()
            .as_int(),
        Some(7)
    );

    let float_type = builtin_type(&[Value::float(0.0)]).unwrap();
    assert_eq!(
        builtin_float_new(&[float_type, Value::int(2).unwrap()])
            .unwrap()
            .as_float(),
        Some(2.0)
    );

    let str_type = builtin_type(&[Value::string(intern("seed"))]).unwrap();
    let str_value = builtin_str_new(&[str_type, Value::string(intern("seed"))]).unwrap();
    let str_ptr = str_value
        .as_string_object_ptr()
        .expect("str.__new__ should return an interned string");
    assert_eq!(
        interned_by_ptr(str_ptr as *const u8).unwrap().as_str(),
        "seed"
    );

    let bool_type = builtin_type(&[Value::bool(false)]).unwrap();
    assert_eq!(
        builtin_bool_new(&[bool_type, Value::int(1).unwrap()])
            .unwrap()
            .as_bool(),
        Some(true)
    );

    let list_type = builtin_type(&[builtin_list(&[]).unwrap()]).unwrap();
    let list_value = builtin_list_new(&[list_type]).unwrap();
    let list_ptr = list_value
        .as_object_ptr()
        .expect("list.__new__ should return a list object");
    assert_eq!(crate::ops::objects::extract_type_id(list_ptr), TypeId::LIST);

    let dict_type = builtin_type(&[builtin_dict(&[]).unwrap()]).unwrap();
    let dict_value = builtin_dict_new(&[dict_type]).unwrap();
    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("dict.__new__ should return a dict object");
    assert_eq!(crate::ops::objects::extract_type_id(dict_ptr), TypeId::DICT);

    let set_type = builtin_type(&[builtin_set(&[]).unwrap()]).unwrap();
    let set_value = builtin_set_new(&[set_type]).unwrap();
    let set_ptr = set_value
        .as_object_ptr()
        .expect("set.__new__ should return a set object");
    assert_eq!(crate::ops::objects::extract_type_id(set_ptr), TypeId::SET);

    let frozenset_type = builtin_type(&[builtin_frozenset(&[]).unwrap()]).unwrap();
    let frozenset_value = builtin_frozenset_new(&[frozenset_type]).unwrap();
    let frozenset_ptr = frozenset_value
        .as_object_ptr()
        .expect("frozenset.__new__ should return a frozenset object");
    assert_eq!(
        crate::ops::objects::extract_type_id(frozenset_ptr),
        TypeId::FROZENSET
    );
}

#[test]
fn test_builtin_float_getformat_reports_native_layout() {
    let float_type = builtin_type_object_for_type_id(TypeId::FLOAT);
    let expected = Value::string(intern(native_float_format_description()));

    let double_format = builtin_float_getformat(&[float_type, Value::string(intern("double"))])
        .expect("float.__getformat__('double') should succeed");
    assert_eq!(double_format, expected);

    let float_format = builtin_float_getformat(&[float_type, Value::string(intern("float"))])
        .expect("float.__getformat__('float') should succeed");
    assert_eq!(float_format, expected);
}

#[test]
fn test_builtin_float_getformat_rejects_invalid_arguments() {
    let float_type = builtin_type_object_for_type_id(TypeId::FLOAT);

    let invalid_kind =
        builtin_float_getformat(&[float_type, Value::string(intern("bogus"))]).unwrap_err();
    assert!(matches!(invalid_kind, BuiltinError::ValueError(_)));
    assert!(invalid_kind.to_string().contains("'double' or 'float'"));

    let invalid_type = builtin_float_getformat(&[float_type, Value::int(1).unwrap()]).unwrap_err();
    assert!(matches!(invalid_type, BuiltinError::TypeError(_)));

    let missing_arg = builtin_float_getformat(&[float_type]).unwrap_err();
    assert!(matches!(missing_arg, BuiltinError::TypeError(_)));
    assert!(missing_arg.to_string().contains("takes exactly 1 argument"));
}

#[test]
fn test_str_new_builds_heap_subclass_instances_with_native_string_storage() {
    let str_type = builtin_type_object_for_type_id(TypeId::STR);
    let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[str_type]));
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let subclass = builtin_type(&[
        Value::string(intern("StrChild")),
        bases_value,
        namespace_value,
    ])
    .expect("str subclass should be constructible");

    let value =
        builtin_str_new(&[subclass, Value::string(intern("seed"))]).expect("str subclass new");
    let ptr = value
        .as_object_ptr()
        .expect("str subclass instance should be heap allocated");
    let class = unsafe { &*(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    assert_eq!(
        crate::ops::objects::extract_type_id(ptr),
        class.class_type_id()
    );

    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    assert_eq!(
        shaped
            .string_backing()
            .expect("string backing should exist")
            .as_str(),
        "seed"
    );

    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_boxed(ptr as *mut ShapedObject) };
    unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
}

#[test]
fn test_bytes_new_builds_heap_subclass_instances_with_native_byte_storage() {
    let bytes_type = builtin_type_object_for_type_id(TypeId::BYTES);
    let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[bytes_type]));
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let subclass = builtin_type(&[
        Value::string(intern("BytesChild")),
        bases_value,
        namespace_value,
    ])
    .expect("bytes subclass should be constructible");

    let (source_value, source_ptr) = boxed_value(BytesObject::from_slice(b"seed"));
    let value = builtin_bytes_new(&[subclass, source_value]).expect("bytes subclass new");
    let ptr = value
        .as_object_ptr()
        .expect("bytes subclass instance should be heap allocated");
    let class = unsafe { &*(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    assert_eq!(
        crate::ops::objects::extract_type_id(ptr),
        class.class_type_id()
    );

    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    assert_eq!(
        shaped
            .bytes_backing()
            .expect("bytes backing should exist")
            .as_bytes(),
        b"seed"
    );

    unsafe { drop_boxed(source_ptr) };
    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_boxed(ptr as *mut ShapedObject) };
    unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
}

#[test]
fn test_int_new_builds_heap_subclass_instances_with_native_integer_storage() {
    let int_type = builtin_type_object_for_type_id(TypeId::INT);
    let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[int_type]));
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let subclass = builtin_type(&[
        Value::string(intern("IntChild")),
        bases_value,
        namespace_value,
    ])
    .expect("int subclass should be constructible");

    let value =
        builtin_int_new(&[subclass, Value::string(intern("123"))]).expect("int subclass new");
    let ptr = value
        .as_object_ptr()
        .expect("int subclass instance should be heap allocated");
    let class = unsafe { &*(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    assert_eq!(
        crate::ops::objects::extract_type_id(ptr),
        class.class_type_id()
    );

    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    assert_eq!(
        shaped.int_backing().expect("integer backing should exist"),
        &BigInt::from(123_i64)
    );
    assert_eq!(value_to_bigint(value), Some(BigInt::from(123_i64)));
    assert_eq!(builtin_int(&[value]).unwrap().as_int(), Some(123));

    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_boxed(ptr as *mut ShapedObject) };
    unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
}

#[test]
fn test_object_new_allocates_native_integer_storage_for_int_subclasses() {
    let int_type = builtin_type_object_for_type_id(TypeId::INT);
    let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[int_type]));
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let subclass = builtin_type(&[
        Value::string(intern("ObjectNewIntChild")),
        bases_value,
        namespace_value,
    ])
    .expect("int subclass should be constructible");

    let value = builtin_object_new(&[subclass]).expect("object.__new__ should allocate");
    let ptr = value
        .as_object_ptr()
        .expect("int subclass instance should be heap allocated");
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    assert_eq!(
        shaped.int_backing().expect("integer backing should exist"),
        &BigInt::zero()
    );

    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_boxed(ptr as *mut ShapedObject) };
    unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
}

#[test]
fn test_object_new_allocates_native_byte_storage_for_bytes_subclasses() {
    let bytes_type = builtin_type_object_for_type_id(TypeId::BYTES);
    let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[bytes_type]));
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let subclass = builtin_type(&[
        Value::string(intern("ObjectNewBytesChild")),
        bases_value,
        namespace_value,
    ])
    .expect("bytes subclass should be constructible");

    let value = builtin_object_new(&[subclass]).expect("object.__new__ should allocate");
    let ptr = value
        .as_object_ptr()
        .expect("bytes subclass instance should be heap allocated");
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    assert!(shaped.has_bytes_backing());
    assert!(shaped.bytes_backing().unwrap().is_empty());

    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_boxed(ptr as *mut ShapedObject) };
    unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
}

#[test]
fn test_builtin_new_wrappers_validate_receiver_types() {
    let err = builtin_int_new(&[Value::int(1).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("int.__new__(X): X must be a type"));

    let bool_type = builtin_type_object_for_type_id(TypeId::BOOL);
    let err = builtin_int_new(&[bool_type, Value::int(0).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    match err {
        BuiltinError::TypeError(message) => {
            assert_eq!(message, "int.__new__(bool) is not safe, use bool.__new__()");
        }
        _ => panic!("expected TypeError for int.__new__(bool, 0)"),
    }

    let err = builtin_list_new(&[]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(
        err.to_string()
            .contains("list.__new__() takes at least 1 argument")
    );
}

#[test]
fn test_object_init_accepts_single_receiver() {
    let instance = builtin_object(&[]).expect("object() should succeed");
    let result = builtin_object_init(&[instance]).expect("object.__init__ should succeed");
    assert!(result.is_none());
}

#[test]
fn test_object_init_rejects_extra_arguments() {
    let instance = builtin_object(&[]).expect("object() should succeed");
    let err = builtin_object_init(&[instance, Value::int(1).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("object.__init__()"));
}

#[test]
fn test_object_init_subclass_accepts_single_class_receiver() {
    let object_type = builtin_type_object_for_type_id(TypeId::OBJECT);
    let result = builtin_object_init_subclass(&[object_type], &[])
        .expect("object.__init_subclass__ should succeed");
    assert!(result.is_none());
}

#[test]
fn test_object_init_subclass_rejects_keyword_arguments() {
    let object_type = builtin_type_object_for_type_id(TypeId::OBJECT);
    let err = builtin_object_init_subclass(&[object_type], &[("token", Value::int(1).unwrap())])
        .unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(
        err.to_string()
            .contains("object.__init_subclass__() takes no keyword arguments")
    );
}

#[test]
fn test_type_init_accepts_metaclass_construction_signature() {
    let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
    let type_value = builtin_type(&[
        Value::string(intern("MetaInitTarget")),
        bases_value,
        namespace_value,
    ])
    .expect("type() should construct a heap type");

    let result = builtin_type_init(&[
        type_value,
        Value::string(intern("MetaInitTarget")),
        bases_value,
        namespace_value,
    ])
    .expect("type.__init__ should accept the class creation signature");

    assert!(result.is_none());

    unsafe { drop_class(type_value.as_object_ptr().unwrap() as *const PyClassObject) };
    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
}

#[test]
fn test_mappingproxy_type_wraps_dict_arguments() {
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("token")), Value::int(7).unwrap());
    let dict_value = to_object_value(dict);

    let proxy = builtin_mappingproxy(&[dict_value]).expect("mappingproxy(dict) should succeed");
    let proxy_ptr = proxy
        .as_object_ptr()
        .expect("mappingproxy should allocate a proxy object");
    assert_eq!(
        crate::ops::objects::extract_type_id(proxy_ptr),
        TypeId::MAPPING_PROXY
    );

    let proxy = unsafe { &*(proxy_ptr as *const MappingProxyObject) };
    assert_eq!(
        crate::builtins::builtin_mapping_proxy_get_item_static(
            proxy,
            Value::string(intern("token"))
        )
        .expect("proxy lookup should succeed"),
        Some(Value::int(7).unwrap())
    );
}

#[test]
fn test_mappingproxy_type_rejects_non_mappings() {
    let err =
        builtin_mappingproxy(&[Value::int(3).unwrap()]).expect_err("mappingproxy(int) should fail");
    match err {
        BuiltinError::TypeError(message) => {
            assert!(message.contains("must be a mapping"));
        }
        other => panic!("expected TypeError, got {other:?}"),
    }
}

#[test]
fn test_type_init_rejects_invalid_argument_count() {
    let type_value = builtin_type_object_for_type_id(TypeId::TYPE);
    let err = builtin_type_init(&[type_value, Value::string(intern("oops"))]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("type.__init__()"));
}

#[test]
fn test_set_empty_and_deduplicated() {
    let empty = builtin_set(&[]).unwrap();
    let empty_ptr = empty.as_object_ptr().unwrap();
    let empty_set = unsafe { &*(empty_ptr as *const SetObject) };
    assert_eq!(empty_set.len(), 0);
    unsafe { drop_boxed(empty_ptr as *mut SetObject) };

    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]);
    let (list_value, list_ptr) = boxed_value(list);
    let set_value = builtin_set(&[list_value]).unwrap();
    let set_ptr = set_value.as_object_ptr().unwrap();
    let set = unsafe { &*(set_ptr as *const SetObject) };
    assert_eq!(set.len(), 2);
    assert!(set.contains(Value::int(1).unwrap()));
    assert!(set.contains(Value::int(2).unwrap()));
    unsafe { drop_boxed(set_ptr as *mut SetObject) };
    unsafe { drop_boxed(list_ptr) };
}

#[test]
fn test_frozenset_empty_and_deduplicated() {
    let empty = builtin_frozenset(&[]).unwrap();
    let empty_ptr = empty.as_object_ptr().unwrap();
    assert_eq!(
        crate::ops::objects::extract_type_id(empty_ptr),
        TypeId::FROZENSET
    );
    let empty_set = unsafe { &*(empty_ptr as *const SetObject) };
    assert_eq!(empty_set.len(), 0);
    unsafe { drop_boxed(empty_ptr as *mut SetObject) };

    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]);
    let (list_value, list_ptr) = boxed_value(list);
    let frozen = builtin_frozenset(&[list_value]).unwrap();
    let frozen_ptr = frozen.as_object_ptr().unwrap();
    assert_eq!(
        crate::ops::objects::extract_type_id(frozen_ptr),
        TypeId::FROZENSET
    );
    let frozen_set = unsafe { &*(frozen_ptr as *const SetObject) };
    assert_eq!(frozen_set.len(), 2);
    assert!(frozen_set.contains(Value::int(1).unwrap()));
    assert!(frozen_set.contains(Value::int(2).unwrap()));

    unsafe { drop_boxed(frozen_ptr as *mut SetObject) };
    unsafe { drop_boxed(list_ptr) };
}

#[test]
fn test_frozenset_identity_for_existing_frozenset() {
    let source = builtin_frozenset(&[]).unwrap();
    let source_ptr = source.as_object_ptr().unwrap();

    let again = builtin_frozenset(&[source]).unwrap();
    assert_eq!(again, source);

    unsafe { drop_boxed(source_ptr as *mut SetObject) };
}

#[test]
fn test_set_and_frozenset_reject_unhashable_elements() {
    let inner = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let (inner_value, inner_ptr) = boxed_value(inner);
    let outer = ListObject::from_slice(&[inner_value]);
    let (outer_value, outer_ptr) = boxed_value(outer);

    let set_err = builtin_set(&[outer_value]).expect_err("set should reject unhashable items");
    assert!(set_err.to_string().contains("unhashable type: 'list'"));

    let frozen_err =
        builtin_frozenset(&[outer_value]).expect_err("frozenset should reject unhashable items");
    assert!(frozen_err.to_string().contains("unhashable type: 'list'"));

    unsafe {
        drop_boxed(outer_ptr);
        drop_boxed(inner_ptr);
    }
}

#[test]
fn test_dict_empty_and_copy() {
    let empty = builtin_dict(&[]).unwrap();
    let empty_ptr = empty.as_object_ptr().unwrap();
    let empty_dict = unsafe { &*(empty_ptr as *const DictObject) };
    assert_eq!(empty_dict.len(), 0);
    unsafe { drop_boxed(empty_ptr as *mut DictObject) };

    let mut source = DictObject::new();
    source.set(Value::int(1).unwrap(), Value::int(10).unwrap());
    let (source_value, source_ptr) = boxed_value(source);
    let copied = builtin_dict(&[source_value]).unwrap();
    let copied_ptr = copied.as_object_ptr().unwrap();
    let copied_dict = unsafe { &*(copied_ptr as *const DictObject) };
    assert_eq!(copied_dict.len(), 1);
    assert_eq!(
        copied_dict.get(Value::int(1).unwrap()).unwrap().as_int(),
        Some(10)
    );
    unsafe { drop_boxed(copied_ptr as *mut DictObject) };
    unsafe { drop_boxed(source_ptr) };
}

#[test]
fn test_dict_from_pair_sequence() {
    let pair1 = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(11).unwrap()]);
    let pair2 = TupleObject::from_slice(&[Value::int(2).unwrap(), Value::int(22).unwrap()]);
    let (pair1_value, pair1_ptr) = boxed_value(pair1);
    let (pair2_value, pair2_ptr) = boxed_value(pair2);

    let pairs = ListObject::from_slice(&[pair1_value, pair2_value]);
    let (pairs_value, pairs_ptr) = boxed_value(pairs);

    let dict_value = builtin_dict(&[pairs_value]).unwrap();
    let dict_ptr = dict_value.as_object_ptr().unwrap();
    let dict = unsafe { &*(dict_ptr as *const DictObject) };

    assert_eq!(dict.len(), 2);
    assert_eq!(dict.get(Value::int(1).unwrap()).unwrap().as_int(), Some(11));
    assert_eq!(dict.get(Value::int(2).unwrap()).unwrap().as_int(), Some(22));

    unsafe { drop_boxed(dict_ptr as *mut DictObject) };
    unsafe { drop_boxed(pairs_ptr) };
    unsafe { drop_boxed(pair1_ptr) };
    unsafe { drop_boxed(pair2_ptr) };
}

#[test]
fn test_dict_invalid_pair_length_errors() {
    let bad_pair = TupleObject::from_slice(&[Value::int(1).unwrap()]);
    let (bad_pair_value, bad_pair_ptr) = boxed_value(bad_pair);
    let pairs = ListObject::from_slice(&[bad_pair_value]);
    let (pairs_value, pairs_ptr) = boxed_value(pairs);

    let err = builtin_dict(&[pairs_value]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("has length 1; 2 is required"));

    unsafe { drop_boxed(pairs_ptr) };
    unsafe { drop_boxed(bad_pair_ptr) };
}

#[test]
fn test_dict_fromkeys_builds_mapping_with_default_none() {
    let dict_type = builtin_type_object_for_type_id(TypeId::DICT);
    let (keys_value, keys_ptr) = boxed_value(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]));

    let result = builtin_dict_fromkeys(&[dict_type, keys_value]).unwrap();
    let result_ptr = result.as_object_ptr().unwrap();
    let dict = unsafe { &*(result_ptr as *const DictObject) };

    assert_eq!(dict.len(), 2);
    assert!(dict.get(Value::int(1).unwrap()).unwrap().is_none());
    assert!(dict.get(Value::int(2).unwrap()).unwrap().is_none());

    unsafe { drop_boxed(result_ptr as *mut DictObject) };
    unsafe { drop_boxed(keys_ptr) };
}

#[test]
fn test_dict_fromkeys_reuses_requested_value_for_each_key() {
    let dict_type = builtin_type_object_for_type_id(TypeId::DICT);
    let (keys_value, keys_ptr) = boxed_value(ListObject::from_slice(&[
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ]));

    let result = builtin_dict_fromkeys(&[dict_type, keys_value, Value::int(99).unwrap()]).unwrap();
    let result_ptr = result.as_object_ptr().unwrap();
    let dict = unsafe { &*(result_ptr as *const DictObject) };

    assert_eq!(dict.get(Value::int(3).unwrap()).unwrap().as_int(), Some(99));
    assert_eq!(dict.get(Value::int(4).unwrap()).unwrap().as_int(), Some(99));

    unsafe { drop_boxed(result_ptr as *mut DictObject) };
    unsafe { drop_boxed(keys_ptr) };
}

#[test]
fn test_dict_fromkeys_rejects_non_dict_receivers() {
    let err =
        builtin_dict_fromkeys(&[builtin_type_object_for_type_id(TypeId::LIST), Value::none()])
            .unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("built-in dict type"));
}

#[test]
fn test_str_maketrans_exposed_on_type_object_builds_mapping_from_dict() {
    let str_type = builtin_type_object_for_type_id(TypeId::STR);
    let method = builtin_getattr(&[str_type, Value::string(intern("maketrans"))])
        .expect("str.maketrans should resolve");
    let method_ptr = method
        .as_object_ptr()
        .expect("str.maketrans should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    let mut mapping = DictObject::new();
    mapping.set(Value::string(intern("a")), Value::string(intern("x")));
    mapping.set(
        Value::int('b' as i64).unwrap(),
        Value::int('y' as i64).unwrap(),
    );
    mapping.set(Value::string(intern("c")), Value::none());
    let mapping_ptr = Box::into_raw(Box::new(mapping));

    let result = builtin
        .call(&[Value::object_ptr(mapping_ptr as *const ())])
        .expect("str.maketrans(dict) should succeed");
    let result_ptr = result.as_object_ptr().expect("result should be a dict");
    let dict = unsafe { &*(result_ptr as *const DictObject) };

    assert_eq!(
        value_to_string(dict.get(Value::int('a' as i64).unwrap()).unwrap()),
        "x"
    );
    assert_eq!(
        dict.get(Value::int('b' as i64).unwrap()).unwrap().as_int(),
        Some('y' as i64)
    );
    assert!(dict.get(Value::int('c' as i64).unwrap()).unwrap().is_none());

    unsafe { drop_boxed(result_ptr as *mut DictObject) };
    unsafe { drop_boxed(mapping_ptr) };
}

#[test]
fn test_str_maketrans_string_forms_build_expected_translation_table() {
    let str_type = builtin_type_object_for_type_id(TypeId::STR);
    let method = builtin_getattr(&[str_type, Value::string(intern("maketrans"))])
        .expect("str.maketrans should resolve");
    let method_ptr = method
        .as_object_ptr()
        .expect("str.maketrans should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    let result = builtin
        .call(&[
            Value::string(intern("ab")),
            Value::string(intern("xy")),
            Value::string(intern("z")),
        ])
        .expect("str.maketrans(string, string, delete) should succeed");
    let result_ptr = result.as_object_ptr().expect("result should be a dict");
    let dict = unsafe { &*(result_ptr as *const DictObject) };

    assert_eq!(
        dict.get(Value::int('a' as i64).unwrap()).unwrap().as_int(),
        Some('x' as i64)
    );
    assert_eq!(
        dict.get(Value::int('b' as i64).unwrap()).unwrap().as_int(),
        Some('y' as i64)
    );
    assert!(dict.get(Value::int('z' as i64).unwrap()).unwrap().is_none());

    unsafe { drop_boxed(result_ptr as *mut DictObject) };
}

#[test]
fn test_bytes_maketrans_builds_256_byte_translation_table() {
    let bytes_type = builtin_type_object_for_type_id(TypeId::BYTES);
    let method = builtin_getattr(&[bytes_type, Value::string(intern("maketrans"))])
        .expect("bytes.maketrans should resolve");
    let method_ptr = method
        .as_object_ptr()
        .expect("bytes.maketrans should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    let (from_value, from_ptr) = boxed_value(BytesObject::from_slice(b"ab"));
    let (to_value, to_ptr) = boxed_value(BytesObject::bytearray_from_slice(b"xy"));
    let result = builtin
        .call(&[from_value, to_value])
        .expect("bytes.maketrans should accept bytes-like arguments");
    let table = value_to_bytes(result);

    assert_eq!(table.len(), 256);
    assert_eq!(table[b'a' as usize], b'x');
    assert_eq!(table[b'b' as usize], b'y');
    assert_eq!(table[b'z' as usize], b'z');

    unsafe { drop_boxed(result.as_object_ptr().unwrap() as *mut BytesObject) };

    let (memoryview_source, memoryview_source_ptr) = boxed_value(BytesObject::from_slice(b"c"));
    let memoryview_value =
        builtin_memoryview(&[memoryview_source]).expect("memoryview(bytes) should work");
    let (memoryview_target, memoryview_target_ptr) = boxed_value(BytesObject::from_slice(b"z"));
    let memoryview_result = builtin
        .call(&[memoryview_value, memoryview_target])
        .expect("bytes.maketrans should accept memoryview arguments");
    let memoryview_table = value_to_bytes(memoryview_result);
    assert_eq!(memoryview_table[b'c' as usize], b'z');

    unsafe {
        drop_boxed(memoryview_result.as_object_ptr().unwrap() as *mut BytesObject);
        drop_boxed(memoryview_value.as_object_ptr().unwrap() as *mut MemoryViewObject);
        drop_boxed(memoryview_source_ptr);
        drop_boxed(memoryview_target_ptr);
    }
    unsafe { drop_boxed(from_ptr) };
    unsafe { drop_boxed(to_ptr) };
}

#[test]
fn test_bytes_maketrans_rejects_mismatched_lengths() {
    let (from_value, from_ptr) = boxed_value(BytesObject::from_slice(b"a"));
    let (to_value, to_ptr) = boxed_value(BytesObject::from_slice(b"xy"));
    let err = builtin_bytes_maketrans(&[from_value, to_value])
        .expect_err("mismatched translation tables should fail");
    assert!(matches!(err, BuiltinError::ValueError(_)));

    unsafe { drop_boxed(from_ptr) };
    unsafe { drop_boxed(to_ptr) };
}

#[test]
fn test_object_constructor() {
    let value = builtin_object(&[]).unwrap();
    let ptr = value
        .as_object_ptr()
        .expect("object() should return object");
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::OBJECT);
    unsafe { drop_boxed(ptr as *mut ShapedObject) };
}

#[test]
fn test_object_constructor_arity_error() {
    let err = builtin_object(&[Value::none()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_type_builtin_returns_type_object_and_is_cached() {
    let t1 = builtin_type(&[Value::int(1).unwrap()]).unwrap();
    let t2 = builtin_type(&[Value::int(2).unwrap()]).unwrap();
    assert_eq!(t1, t2);

    let ptr = t1.as_object_ptr().expect("type() should return object");
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::TYPE);
}

#[test]
fn test_type_builtin_returns_exception_class_for_exception_instances() {
    let exc = crate::builtins::get_exception_type("ValueError")
        .expect("ValueError should exist")
        .construct(&[Value::string(intern("boom"))]);
    let exc_type = builtin_type(&[exc]).expect("type() should accept exception instances");
    let type_name = builtin_getattr(&[exc_type, Value::string(intern("__name__"))])
        .expect("__name__ should be readable");

    assert_eq!(type_name, Value::string(intern("ValueError")));
}

#[test]
fn test_type_builtin_three_arg_form_builds_class_and_copies_namespace() {
    let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());

    let mut namespace = DictObject::new();
    namespace.set(Value::string(intern("answer")), Value::int(42).unwrap());
    let (namespace_value, namespace_ptr) = boxed_value(namespace);

    let class_value = builtin_type(&[
        Value::string(intern("Dynamic")),
        bases_value,
        namespace_value,
    ])
    .unwrap();
    let class_ptr = class_value
        .as_object_ptr()
        .expect("type() should return class object");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };

    assert_eq!(class.name().as_str(), "Dynamic");
    assert_eq!(
        class.get_attr(&intern("answer")).unwrap().as_int(),
        Some(42)
    );
    assert_eq!(class.mro(), &[class.class_id(), ClassId::OBJECT]);

    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_class(class_ptr as *const PyClassObject) };
}

#[test]
fn test_type_builtin_three_arg_form_accepts_dict_subclass_namespace() {
    let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());

    let mut namespace =
        ShapedObject::new_dict_backed(TypeId::from_raw(900), shape_registry().empty_shape());
    namespace
        .dict_backing_mut()
        .expect("dict subclass namespace should expose dict backing")
        .set(Value::string(intern("answer")), Value::int(42).unwrap());
    let (namespace_value, namespace_ptr) = boxed_value(namespace);

    let class_value = builtin_type(&[
        Value::string(intern("DynamicSubclassNamespace")),
        bases_value,
        namespace_value,
    ])
    .expect("type() should accept dict subclass namespaces");
    let class_ptr = class_value
        .as_object_ptr()
        .expect("type() should return class object");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };

    assert_eq!(class.name().as_str(), "DynamicSubclassNamespace");
    assert_eq!(
        class.get_attr(&intern("answer")).unwrap().as_int(),
        Some(42)
    );

    unsafe { drop_boxed(namespace_ptr as *mut ShapedObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_class(class_ptr as *const PyClassObject) };
}

#[test]
fn test_type_new_builtin_supports_explicit_metaclass_argument() {
    let metaclass = crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE);
    let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());

    let mut namespace = DictObject::new();
    namespace.set(
        Value::string(intern("_member_names_")),
        Value::int(7).unwrap(),
    );
    namespace.set(Value::string(intern("answer")), Value::int(42).unwrap());
    let (namespace_value, namespace_ptr) = boxed_value(namespace);

    let class_value = builtin_type_new(&[
        metaclass,
        Value::string(intern("DynamicMeta")),
        bases_value,
        namespace_value,
    ])
    .expect("type.__new__ should build class");
    let class_ptr = class_value
        .as_object_ptr()
        .expect("type.__new__ should return class object");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };

    assert_eq!(class.name().as_str(), "DynamicMeta");
    assert_eq!(
        class.get_attr(&intern("answer")).unwrap().as_int(),
        Some(42)
    );
    assert!(class.get_attr(&intern("_member_names_")).is_some());

    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_class(class_ptr as *const PyClassObject) };
}

#[test]
fn test_type_builtin_three_arg_form_supports_builtin_base_types() {
    let (tuple_instance, tuple_instance_ptr) = boxed_value(TupleObject::empty());
    let tuple_type = builtin_type(&[tuple_instance]).unwrap();

    let bases = TupleObject::from_slice(&[tuple_type]);
    let (bases_value, bases_ptr) = boxed_value(bases);
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());

    let class_value = builtin_type(&[
        Value::string(intern("TupleChild")),
        bases_value,
        namespace_value,
    ])
    .unwrap();
    let class_ptr = class_value
        .as_object_ptr()
        .expect("type() should return class object");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };

    assert_eq!(class.bases(), &[ClassId(TypeId::TUPLE.raw())]);
    assert_eq!(
        class.mro(),
        &[
            class.class_id(),
            ClassId(TypeId::TUPLE.raw()),
            ClassId::OBJECT
        ]
    );

    unsafe { drop_boxed(tuple_instance_ptr) };
    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_class(class_ptr as *const PyClassObject) };
}

#[test]
fn test_type_builtin_returns_heap_class_for_heap_instances() {
    let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
    let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());

    let class_value = builtin_type(&[
        Value::string(intern("HeapInstanceType")),
        bases_value,
        namespace_value,
    ])
    .expect("type() should build a heap class");
    let class_ptr = class_value
        .as_object_ptr()
        .expect("type() should return a heap class");

    let instance = builtin_object_new(&[class_value]).expect("object.__new__ should allocate");
    let instance_ptr = instance
        .as_object_ptr()
        .expect("object.__new__ should return a heap instance");

    assert_eq!(builtin_type(&[instance]).unwrap(), class_value);

    unsafe { drop_boxed(instance_ptr as *mut ShapedObject) };
    unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
    unsafe { drop_boxed(bases_ptr) };
    unsafe { drop_class(class_ptr as *const PyClassObject) };
}

#[test]
fn test_type_builtin_three_arg_form_validates_inputs() {
    let err = builtin_type(&[Value::int(1).unwrap(), Value::none(), Value::none()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("argument 1"));

    let err =
        builtin_type(&[Value::string(intern("C")), Value::none(), Value::none()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("argument 2"));

    let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
    let err = builtin_type(&[Value::string(intern("C")), bases_value, Value::none()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("argument 3"));

    unsafe { drop_boxed(bases_ptr) };
}

#[test]
fn test_isinstance_true_and_false_cases() {
    let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
    let float_type = builtin_type(&[Value::float(0.0)]).unwrap();
    let object_value = builtin_object(&[]).unwrap();
    let object_ptr = object_value.as_object_ptr().unwrap();
    let object_type = builtin_type(&[object_value]).unwrap();
    unsafe { drop_boxed(object_ptr as *mut ShapedObject) };

    assert!(
        builtin_isinstance(&[Value::int(5).unwrap(), int_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(
        !builtin_isinstance(&[Value::int(5).unwrap(), float_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(
        builtin_isinstance(&[Value::int(5).unwrap(), object_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );
}

#[test]
fn test_isinstance_bool_is_subclass_of_int() {
    let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
    assert!(
        builtin_isinstance(&[Value::bool(true), int_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );
}

#[test]
fn test_isinstance_tuple_of_types() {
    let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
    let str_type = builtin_type(&[Value::string(intern("s"))]).unwrap();
    let tuple = TupleObject::from_slice(&[str_type, int_type]);
    let (tuple_value, tuple_ptr) = boxed_value(tuple);

    let result = builtin_isinstance(&[Value::int(9).unwrap(), tuple_value]).unwrap();
    assert!(result.as_bool().unwrap());
    unsafe { drop_boxed(tuple_ptr) };
}

#[test]
fn test_isinstance_invalid_type_spec_error() {
    let err = builtin_isinstance(&[Value::int(1).unwrap(), Value::int(2).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("arg 2 must be a type"));
}

#[test]
fn test_isinstance_uses_heap_metaclass_for_class_objects() {
    let type_type = crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE);

    let (metaclass_bases, metaclass_bases_ptr) = boxed_value(TupleObject::from_slice(&[type_type]));
    let (metaclass_namespace, metaclass_namespace_ptr) = boxed_value(DictObject::new());
    let metaclass = builtin_type(&[
        Value::string(intern("MetaEnumType")),
        metaclass_bases,
        metaclass_namespace,
    ])
    .expect("type() should build a heap metaclass");
    let metaclass_ptr = metaclass
        .as_object_ptr()
        .expect("heap metaclass should be object-backed");

    let (class_bases, class_bases_ptr) = boxed_value(TupleObject::empty());
    let (class_namespace, class_namespace_ptr) = boxed_value(DictObject::new());
    let enum_like = builtin_type_new(&[
        metaclass,
        Value::string(intern("EnumLike")),
        class_bases,
        class_namespace,
    ])
    .expect("type.__new__ should build a class with the custom metaclass");
    let enum_like_ptr = enum_like
        .as_object_ptr()
        .expect("heap class should be object-backed");

    assert_eq!(builtin_type(&[enum_like]).unwrap(), metaclass);
    assert!(
        builtin_isinstance(&[enum_like, metaclass])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(
        builtin_isinstance(&[enum_like, type_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );

    unsafe {
        drop_boxed(metaclass_bases_ptr);
        drop_boxed(metaclass_namespace_ptr as *mut DictObject);
        drop_boxed(class_bases_ptr);
        drop_boxed(class_namespace_ptr as *mut DictObject);
        drop_class(enum_like_ptr as *const PyClassObject);
        drop_class(metaclass_ptr as *const PyClassObject);
    }
}

#[test]
fn test_issubclass_semantics() {
    let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
    let bool_type = builtin_type(&[Value::bool(true)]).unwrap();
    let object_value = builtin_object(&[]).unwrap();
    let object_ptr = object_value.as_object_ptr().unwrap();
    let object_type = builtin_type(&[object_value]).unwrap();
    unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
    let float_type = builtin_type(&[Value::float(0.0)]).unwrap();

    assert!(
        builtin_issubclass(&[int_type, object_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(
        builtin_issubclass(&[bool_type, int_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(
        !builtin_issubclass(&[float_type, int_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );
}

#[test]
fn test_issubclass_tuple_of_targets() {
    let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
    let str_type = builtin_type(&[Value::string(intern("x"))]).unwrap();
    let object_value = builtin_object(&[]).unwrap();
    let object_ptr = object_value.as_object_ptr().unwrap();
    let object_type = builtin_type(&[object_value]).unwrap();
    unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
    let targets = TupleObject::from_slice(&[str_type, object_type]);
    let (targets_value, targets_ptr) = boxed_value(targets);

    let result = builtin_issubclass(&[int_type, targets_value]).unwrap();
    assert!(result.as_bool().unwrap());
    unsafe { drop_boxed(targets_ptr) };
}

#[test]
fn test_issubclass_arg_validation() {
    let object_value = builtin_object(&[]).unwrap();
    let object_ptr = object_value.as_object_ptr().unwrap();
    let object_type = builtin_type(&[object_value]).unwrap();
    unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
    let err1 = builtin_issubclass(&[Value::int(1).unwrap(), object_type]).unwrap_err();
    assert!(matches!(err1, BuiltinError::TypeError(_)));
    assert!(err1.to_string().contains("arg 1 must be a class"));

    let int_type = builtin_type(&[Value::int(1).unwrap()]).unwrap();
    let err2 = builtin_issubclass(&[int_type, Value::int(2).unwrap()]).unwrap_err();
    assert!(matches!(err2, BuiltinError::TypeError(_)));
    assert!(err2.to_string().contains("arg 2 must be a class"));
}

#[test]
fn test_issubclass_accepts_exception_type_values_without_heap_class_casts() {
    let value_error =
        crate::builtins::exception_type_value_for_id(ExceptionTypeId::ValueError.as_u8() as u16)
            .expect("ValueError type should exist");
    let exception =
        crate::builtins::exception_type_value_for_id(ExceptionTypeId::Exception.as_u8() as u16)
            .expect("Exception type should exist");
    let int_type = builtin_type_object_for_type_id(TypeId::INT);

    assert!(
        builtin_issubclass(&[value_error, exception])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(
        !builtin_issubclass(&[value_error, int_type])
            .unwrap()
            .as_bool()
            .unwrap()
    );

    let mut vm = VirtualMachine::new();
    assert!(
        builtin_issubclass_vm(&mut vm, &[value_error, exception])
            .unwrap()
            .as_bool()
            .unwrap()
    );
}

#[test]
fn test_issubclass_vm_honors_metaclass_subclasscheck() {
    fn subclasscheck(args: &[Value]) -> Result<Value, BuiltinError> {
        assert_eq!(args.len(), 1);
        Ok(Value::bool(true))
    }

    let (metaclass, metaclass_ptr, metaclass_bases_ptr, metaclass_namespace_ptr, hook_ptr) =
        heap_metaclass_with_hook("MetaSubclassCheck", "__subclasscheck__", subclasscheck);
    let (target, target_ptr, target_bases_ptr, target_namespace_ptr) =
        heap_class_with_metaclass(metaclass, "VirtualBase");
    let (candidate, candidate_ptr, candidate_bases_ptr, candidate_namespace_ptr) =
        heap_class("VirtualCandidate");

    let mut vm = VirtualMachine::new();
    assert!(
        builtin_issubclass_vm(&mut vm, &[candidate, target])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(
        !builtin_issubclass(&[candidate, target])
            .unwrap()
            .as_bool()
            .unwrap()
    );

    unsafe {
        drop_boxed(hook_ptr);
        drop_boxed(target_namespace_ptr);
        drop_boxed(target_bases_ptr);
        drop_boxed(candidate_namespace_ptr);
        drop_boxed(candidate_bases_ptr);
        drop_boxed(metaclass_namespace_ptr);
        drop_boxed(metaclass_bases_ptr);
        drop_class(candidate_ptr);
        drop_class(target_ptr);
        drop_class(metaclass_ptr);
    }
}

#[test]
fn test_isinstance_vm_honors_metaclass_instancecheck() {
    fn instancecheck(args: &[Value]) -> Result<Value, BuiltinError> {
        assert_eq!(args.len(), 1);
        Ok(Value::bool(true))
    }

    let (metaclass, metaclass_ptr, metaclass_bases_ptr, metaclass_namespace_ptr, hook_ptr) =
        heap_metaclass_with_hook("MetaInstanceCheck", "__instancecheck__", instancecheck);
    let (target, target_ptr, target_bases_ptr, target_namespace_ptr) =
        heap_class_with_metaclass(metaclass, "VirtualTarget");
    let (candidate, candidate_ptr, candidate_bases_ptr, candidate_namespace_ptr) =
        heap_class("VirtualCandidate");
    let instance =
        builtin_object_new(&[candidate]).expect("object.__new__ should build a heap instance");
    let instance_ptr = instance
        .as_object_ptr()
        .expect("object.__new__ should return a heap instance");

    let mut vm = VirtualMachine::new();
    assert!(
        builtin_isinstance_vm(&mut vm, &[instance, target])
            .unwrap()
            .as_bool()
            .unwrap()
    );

    unsafe {
        drop_boxed(instance_ptr as *mut ShapedObject);
        drop_boxed(hook_ptr);
        drop_boxed(target_namespace_ptr);
        drop_boxed(target_bases_ptr);
        drop_boxed(candidate_namespace_ptr);
        drop_boxed(candidate_bases_ptr);
        drop_boxed(metaclass_namespace_ptr);
        drop_boxed(metaclass_bases_ptr);
        drop_class(candidate_ptr);
        drop_class(target_ptr);
        drop_class(metaclass_ptr);
    }
}

#[test]
fn test_isinstance_vm_preserves_exact_type_match_before_instancecheck_hook() {
    fn instancecheck(args: &[Value]) -> Result<Value, BuiltinError> {
        assert_eq!(args.len(), 1);
        Ok(Value::bool(false))
    }

    let (metaclass, metaclass_ptr, metaclass_bases_ptr, metaclass_namespace_ptr, hook_ptr) =
        heap_metaclass_with_hook("MetaExactInstance", "__instancecheck__", instancecheck);
    let (target, target_ptr, target_bases_ptr, target_namespace_ptr) =
        heap_class_with_metaclass(metaclass, "ExactTarget");
    let instance =
        builtin_object_new(&[target]).expect("object.__new__ should build a heap instance");
    let instance_ptr = instance
        .as_object_ptr()
        .expect("object.__new__ should return a heap instance");

    let mut vm = VirtualMachine::new();
    assert!(
        builtin_isinstance_vm(&mut vm, &[instance, target])
            .unwrap()
            .as_bool()
            .unwrap()
    );

    unsafe {
        drop_boxed(instance_ptr as *mut ShapedObject);
        drop_boxed(hook_ptr);
        drop_boxed(target_namespace_ptr);
        drop_boxed(target_bases_ptr);
        drop_boxed(metaclass_namespace_ptr);
        drop_boxed(metaclass_bases_ptr);
        drop_class(target_ptr);
        drop_class(metaclass_ptr);
    }
}

#[test]
fn test_attribute_builtins_roundtrip_with_tagged_name() {
    let object = builtin_object(&[]).unwrap();
    let object_ptr = object.as_object_ptr().unwrap();
    let name = Value::string(intern("field"));

    builtin_setattr(&[object, name, Value::int(42).unwrap()]).unwrap();
    assert_eq!(builtin_getattr(&[object, name]).unwrap().as_int(), Some(42));
    assert!(builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());

    // Distinguish explicit None assignment from deletion.
    builtin_setattr(&[object, name, Value::none()]).unwrap();
    assert!(builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());
    assert!(builtin_getattr(&[object, name]).unwrap().is_none());

    builtin_delattr(&[object, name]).unwrap();
    assert!(!builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());

    let err = builtin_getattr(&[object, name]).unwrap_err();
    assert!(matches!(err, BuiltinError::AttributeError(_)));

    let fallback = Value::int(7).unwrap();
    assert_eq!(
        builtin_getattr(&[object, name, fallback]).unwrap(),
        fallback
    );

    unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
}

#[test]
fn test_attribute_builtins_accept_heap_string_name() {
    let object = builtin_object(&[]).unwrap();
    let object_ptr = object.as_object_ptr().unwrap();
    let (name, name_ptr) = boxed_value(StringObject::new("heap_name"));

    builtin_setattr(&[object, name, Value::int(11).unwrap()]).unwrap();
    assert_eq!(builtin_getattr(&[object, name]).unwrap().as_int(), Some(11));
    assert!(builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());
    builtin_delattr(&[object, name]).unwrap();
    assert!(!builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());

    unsafe { drop_boxed(name_ptr) };
    unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
}

#[test]
fn test_getattr_vm_conversion_preserves_python_attribute_error_semantics() {
    let err = crate::error::RuntimeError::exception(
        ExceptionTypeId::AttributeError.as_u8() as u16,
        "_mock_methods",
    );
    assert!(matches!(
        runtime_error_to_builtin_error(err),
        BuiltinError::AttributeError(message) if message == "_mock_methods"
    ));
}

#[test]
fn test_attribute_builtins_roundtrip_for_class_objects() {
    let (class_value, class_ptr) = class_value(PyClassObject::new_simple(intern("Example")));
    let name = Value::string(intern("field"));

    builtin_setattr(&[class_value, name, Value::int(42).unwrap()]).unwrap();
    assert_eq!(
        builtin_getattr(&[class_value, name]).unwrap().as_int(),
        Some(42)
    );
    assert!(
        builtin_hasattr(&[class_value, name])
            .unwrap()
            .as_bool()
            .unwrap()
    );

    builtin_delattr(&[class_value, name]).unwrap();
    assert!(
        !builtin_hasattr(&[class_value, name])
            .unwrap()
            .as_bool()
            .unwrap()
    );

    unsafe { drop_class(class_ptr) };
}

#[test]
fn test_builtin_getattr_reads_builtin_function_metadata() {
    fn metadata_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(Value::int(i64::try_from(args.len()).unwrap_or(i64::MAX))
            .expect("metadata builtin result should fit"))
    }

    let builtin = Box::new(crate::builtins::BuiltinFunctionObject::new(
        Arc::from("sys.gettrace"),
        metadata_builtin,
    ));
    let builtin_ptr = Box::into_raw(builtin);
    let builtin_value = Value::object_ptr(builtin_ptr as *const ());

    assert_eq!(
        builtin_getattr(&[builtin_value, Value::string(intern("__name__"))]).unwrap(),
        Value::string(intern("gettrace"))
    );
    assert_eq!(
        builtin_getattr(&[builtin_value, Value::string(intern("__qualname__"))]).unwrap(),
        Value::string(intern("sys.gettrace"))
    );
    assert!(
        builtin_getattr(&[builtin_value, Value::string(intern("__doc__"))])
            .unwrap()
            .is_none()
    );
    assert!(
        builtin_getattr(&[builtin_value, Value::string(intern("__self__"))])
            .unwrap()
            .is_none()
    );

    unsafe { drop_boxed(builtin_ptr) };
}

#[test]
fn test_method_type_constructor_creates_bound_method() {
    fn callable_probe(_args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(Value::none())
    }

    let (callable, callable_ptr) = boxed_value(crate::builtins::BuiltinFunctionObject::new(
        Arc::from("test.callable_probe"),
        callable_probe,
    ));
    let instance = Value::int(7).unwrap();

    let result =
        call_builtin_type(TypeId::METHOD, &[callable, instance]).expect("method() should bind");
    let result_ptr = result
        .as_object_ptr()
        .expect("method() should return a bound method object");
    assert_eq!(
        crate::ops::objects::extract_type_id(result_ptr),
        TypeId::METHOD
    );

    let bound = unsafe { &*(result_ptr as *const BoundMethod) };
    assert_eq!(bound.function(), callable);
    assert_eq!(bound.instance(), instance);

    unsafe {
        drop_boxed(result_ptr as *mut BoundMethod);
        drop_boxed(callable_ptr);
    }
}

#[test]
fn test_method_type_constructor_rejects_none_instance() {
    fn callable_probe(_args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(Value::none())
    }

    let (callable, callable_ptr) = boxed_value(crate::builtins::BuiltinFunctionObject::new(
        Arc::from("test.callable_probe"),
        callable_probe,
    ));

    let err = call_builtin_type(TypeId::METHOD, &[callable, Value::none()])
        .expect_err("method() should reject None instance");
    assert!(
        matches!(err, BuiltinError::TypeError(ref message) if message == "instance must not be None")
    );

    unsafe { drop_boxed(callable_ptr) };
}

#[test]
fn test_method_type_constructor_rejects_non_callable_receiver() {
    let err = call_builtin_type(
        TypeId::METHOD,
        &[Value::int(42).unwrap(), Value::int(7).unwrap()],
    )
    .expect_err("method() should reject non-callable receiver");
    assert!(
        matches!(err, BuiltinError::TypeError(ref message) if message == "first argument must be callable")
    );
}
