use super::*;
use crate::builtins::BuiltinFunctionObject;
use crate::builtins::itertools::{builtin_iter, builtin_next};
use crate::import::ModuleObject;
use prism_core::intern::intern;
use prism_core::python_unicode::encode_python_code_point;
use prism_core::value::SMALL_INT_MAX;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::{ClassMethodDescriptor, StaticMethodDescriptor};
use prism_runtime::object::shape::Shape;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::types::function::FunctionObject;
use std::sync::Arc;

fn boxed_value<T>(obj: T) -> (Value, *mut T) {
    let ptr = Box::into_raw(Box::new(obj));
    (Value::object_ptr(ptr as *const ()), ptr)
}

unsafe fn drop_boxed<T>(ptr: *mut T) {
    drop(unsafe { Box::from_raw(ptr) });
}

fn tagged_string_value_to_rust_string(value: Value) -> String {
    assert!(value.is_string(), "expected tagged string, got {value:?}");
    let ptr = value
        .as_string_object_ptr()
        .expect("tagged string pointer missing") as *const u8;
    prism_core::intern::interned_by_ptr(ptr)
        .expect("tagged string pointer not interned")
        .as_str()
        .to_string()
}

#[test]
fn test_len_tagged_string() {
    let value = Value::string(intern("hello"));
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(5));
}

#[test]
fn test_len_string_counts_unicode_scalar_values() {
    let tagged = Value::string(intern("tmpæ"));
    assert_eq!(builtin_len(&[tagged]).unwrap().as_int(), Some(4));

    let (heap, ptr) = boxed_value(StringObject::new("hé 🦀"));
    assert_eq!(builtin_len(&[heap]).unwrap().as_int(), Some(4));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_list_object() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(3));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_tuple_object() {
    let tuple = TupleObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
        Value::int(40).unwrap(),
    ]);
    let (value, ptr) = boxed_value(tuple);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(4));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_tuple_backed_object() {
    let object = ShapedObject::new_tuple_backed(
        TypeId::OBJECT,
        Shape::empty(),
        TupleObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]),
    );
    let (value, ptr) = boxed_value(object);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(3));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_heap_list_subclass_uses_native_backing() {
    let mut object = ShapedObject::new_list_backed(TypeId::from_raw(512), Shape::empty());
    object
        .list_backing_mut()
        .expect("list backing should exist")
        .extend([Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let (value, ptr) = boxed_value(object);

    let result = builtin_len(&[value]).unwrap();

    assert_eq!(result.as_int(), Some(2));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_dict_object() {
    let mut dict = DictObject::new();
    dict.set(Value::int(1).unwrap(), Value::int(11).unwrap());
    dict.set(Value::int(2).unwrap(), Value::int(22).unwrap());
    let (value, ptr) = boxed_value(dict);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(2));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_mappingproxy_object() {
    let class = Arc::new(PyClassObject::new_simple(intern("SizedProxy")));
    class.set_attr(intern("token"), Value::int(7).unwrap());
    class.set_attr(intern("label"), Value::string(intern("ready")));

    let proxy = MappingProxyObject::for_user_class(Arc::as_ptr(&class));
    let (value, ptr) = boxed_value(proxy);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(2));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_set_object() {
    let set = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let (value, ptr) = boxed_value(set);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(3));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_frozenset_object() {
    let mut set = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]);
    set.header.type_id = TypeId::FROZENSET;
    let (value, ptr) = boxed_value(set);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(2));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_bytes_object() {
    let bytes = BytesObject::from_slice(b"hello");
    let (value, ptr) = boxed_value(bytes);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(5));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_bytearray_object() {
    let bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3, 4]);
    let (value, ptr) = boxed_value(bytearray);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(4));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_range_object() {
    let range = RangeObject::new(0, 10, 2);
    let (value, ptr) = boxed_value(range);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(5));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_string_object() {
    let string = StringObject::new("runtime");
    let (value, ptr) = boxed_value(string);
    let result = builtin_len(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(7));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_arity_error() {
    let result = builtin_len(&[]);
    assert!(matches!(result, Err(BuiltinError::TypeError(_))));
}

#[test]
fn test_len_non_sized_type_error() {
    let result = builtin_len(&[Value::int(42).unwrap()]);
    assert!(matches!(result, Err(BuiltinError::TypeError(_))));
}

#[test]
fn test_abs_int() {
    let result = builtin_abs(&[Value::int(-42).unwrap()]).unwrap();
    assert_eq!(result.as_int(), Some(42));

    let result = builtin_abs(&[Value::int(42).unwrap()]).unwrap();
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_abs_bool_returns_int_result() {
    let result = builtin_abs(&[Value::bool(true)]).unwrap();
    assert_eq!(result.as_int(), Some(1));
    assert!(!result.is_bool());
}

#[test]
fn test_abs_float() {
    let result = builtin_abs(&[Value::float(-3.14)]).unwrap();
    assert_eq!(result.as_float(), Some(3.14));
}

#[test]
fn test_abs_error() {
    let result = builtin_abs(&[Value::none()]);
    assert!(result.is_err());
}

#[test]
fn test_min() {
    let result = builtin_min(&[
        Value::int(5).unwrap(),
        Value::int(3).unwrap(),
        Value::int(8).unwrap(),
    ])
    .unwrap();
    assert_eq!(result.as_int(), Some(3));
}

#[test]
fn test_max() {
    let result = builtin_max(&[
        Value::int(5).unwrap(),
        Value::int(3).unwrap(),
        Value::int(8).unwrap(),
    ])
    .unwrap();
    assert_eq!(result.as_int(), Some(8));
}

#[test]
fn test_max_iterable_uses_elements_instead_of_returning_the_iterable() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_max(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(3));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_max_iterable_supports_string_ordering() {
    let values = SetObject::from_slice(&[
        Value::string(intern("a")),
        Value::string(intern("c")),
        Value::string(intern("b")),
    ]);
    let (value, ptr) = boxed_value(values);
    let result = builtin_max(&[value]).unwrap();
    assert_eq!(value_as_string_ref(result).unwrap().as_str(), "c");
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_max_vm_supports_default_for_empty_iterable() {
    let mut vm = VirtualMachine::new();
    let list = ListObject::from_slice(&[]);
    let (value, ptr) = boxed_value(list);
    let fallback = Value::int(99).unwrap();
    let result = builtin_max_vm_kw(&mut vm, &[value], &[("default", fallback)]).unwrap();
    assert_eq!(result.as_int(), Some(99));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_max_vm_rejects_default_with_multiple_positional_arguments() {
    let mut vm = VirtualMachine::new();
    let error = builtin_max_vm_kw(
        &mut vm,
        &[Value::int(1).unwrap(), Value::int(2).unwrap()],
        &[("default", Value::int(3).unwrap())],
    )
    .unwrap_err();
    assert!(matches!(error, BuiltinError::TypeError(_)));
}

#[test]
fn test_max_vm_honors_key_keyword() {
    let mut vm = VirtualMachine::new();
    let result = builtin_max_vm_kw(
        &mut vm,
        &[Value::int(0).unwrap(), Value::int(7).unwrap()],
        &[(
            "key",
            crate::builtins::builtin_type_object_for_type_id(TypeId::BOOL),
        )],
    )
    .unwrap();
    assert_eq!(result.as_int(), Some(7));
}

#[test]
fn test_sum_int_list() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_sum(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(10));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_float_list() {
    let list = ListObject::from_slice(&[Value::float(1.5), Value::float(2.0), Value::float(3.5)]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_sum(&[value]).unwrap();
    assert_eq!(result.as_float(), Some(7.0));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_mixed_numeric_promotes_to_float() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::float(2.5)]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_sum(&[value]).unwrap();
    assert_eq!(result.as_float(), Some(3.5));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_bool_items() {
    let list = ListObject::from_slice(&[Value::bool(true), Value::bool(false), Value::bool(true)]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_sum(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(2));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_with_int_start() {
    let list = ListObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_sum(&[value, Value::int(10).unwrap()]).unwrap();
    assert_eq!(result.as_int(), Some(15));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_with_float_start() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_sum(&[value, Value::float(0.5)]).unwrap();
    assert_eq!(result.as_float(), Some(3.5));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_range() {
    let range = RangeObject::new(1, 5, 1);
    let (value, ptr) = boxed_value(range);
    let result = builtin_sum(&[value]).unwrap();
    assert_eq!(result.as_int(), Some(10));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_iterator_consumes_iterator_state() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let (list_value, list_ptr) = boxed_value(list);
    let iter = builtin_iter(&[list_value]).unwrap();

    let result = builtin_sum(&[iter]).unwrap();
    assert_eq!(result.as_int(), Some(6));

    // Iterator should be exhausted after sum consumes it.
    let next_result = builtin_next(&[iter]);
    assert!(next_result.is_err());
    unsafe { drop_boxed(list_ptr) };
}

#[test]
fn test_sum_non_iterable_error() {
    let err = builtin_sum(&[Value::int(42).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_sum_non_numeric_start_error() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let (value, ptr) = boxed_value(list);
    let err = builtin_sum(&[value, Value::none()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_string_start_error() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let (value, ptr) = boxed_value(list);
    let err = builtin_sum(&[value, Value::string(intern("x"))]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    let msg = err.to_string();
    assert!(msg.contains("can't sum strings"));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_string_item_error() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::string(intern("x"))]);
    let (value, ptr) = boxed_value(list);
    let err = builtin_sum(&[value]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    let msg = err.to_string();
    assert!(msg.contains("can't sum strings"));
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_overflow_small_int_domain() {
    let list =
        ListObject::from_slice(&[Value::int(SMALL_INT_MAX).unwrap(), Value::int(1).unwrap()]);
    let (value, ptr) = boxed_value(list);
    let result = builtin_sum(&[value]).unwrap();
    assert_eq!(
        value_to_bigint(result),
        Some(BigInt::from(SMALL_INT_MAX) + BigInt::from(1_i64))
    );
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_sum_bigint_and_float_reports_exact_overflow_boundary() {
    let huge = bigint_to_value(BigInt::from(1_u8) << 10000_u32);
    let list = ListObject::from_slice(&[huge, Value::float(1.0)]);
    let (value, ptr) = boxed_value(list);
    let err = builtin_sum(&[value]).unwrap_err();
    assert!(matches!(err, BuiltinError::OverflowError(_)));
    assert!(
        err.to_string()
            .contains("int too large to convert to float"),
        "unexpected error: {err}"
    );
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_pow_int() {
    let result = builtin_pow(&[Value::int(2).unwrap(), Value::int(10).unwrap()]).unwrap();
    assert_eq!(result.as_int(), Some(1024));
}

#[test]
fn test_pow_mod() {
    let result = builtin_pow(&[
        Value::int(2).unwrap(),
        Value::int(10).unwrap(),
        Value::int(100).unwrap(),
    ])
    .unwrap();
    assert_eq!(result.as_int(), Some(24)); // 1024 % 100 = 24
}

#[test]
fn test_min_prefers_exact_bigint_ordering_over_float_rounding() {
    let huge = bigint_to_value(BigInt::from(1_u8) << 80_u32);
    let result = builtin_min(&[huge, Value::float(2f64.powi(80))]).unwrap();
    assert_eq!(value_to_bigint(result), Some(BigInt::from(1_u8) << 80_u32));
}

#[test]
fn test_max_prefers_exact_bigint_ordering_over_float_rounding() {
    let huge_plus_one = bigint_to_value((BigInt::from(1_u8) << 80_u32) + BigInt::from(1_u8));
    let result = builtin_max(&[huge_plus_one, Value::float(2f64.powi(80))]).unwrap();
    assert_eq!(
        value_to_bigint(result),
        Some((BigInt::from(1_u8) << 80_u32) + BigInt::from(1_u8))
    );
}

#[test]
fn test_min_handles_negative_bigint_float_boundary_exactly() {
    let huge_negative = bigint_to_value(-((BigInt::from(1_u8) << 80_u32) + BigInt::from(1_u8)));
    let result = builtin_min(&[huge_negative, Value::float(-2f64.powi(80))]).unwrap();
    assert_eq!(
        value_to_bigint(result),
        Some(-((BigInt::from(1_u8) << 80_u32) + BigInt::from(1_u8)))
    );
}

#[test]
fn test_pow_promotes_large_integer_results_to_bigint() {
    let result = builtin_pow(&[Value::int(2).unwrap(), Value::int(100).unwrap()]).unwrap();
    assert_eq!(value_to_bigint(result), Some(BigInt::from(1_u8) << 100_u32));
}

#[test]
fn test_pow_negative_modular_exponent_uses_inverse() {
    let result = builtin_pow(&[
        Value::int(2).unwrap(),
        Value::int(-1).unwrap(),
        Value::int(5).unwrap(),
    ])
    .unwrap();
    assert_eq!(result.as_int(), Some(3));
}

#[test]
fn test_pow_negative_modulus_matches_python_sign_rules() {
    let result = builtin_pow(&[
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(-5).unwrap(),
    ])
    .unwrap();
    assert_eq!(result.as_int(), Some(-2));

    let inverse = builtin_pow(&[
        Value::int(2).unwrap(),
        Value::int(-1).unwrap(),
        Value::int(-5).unwrap(),
    ])
    .unwrap();
    assert_eq!(inverse.as_int(), Some(-2));
}

#[test]
fn test_pow_non_invertible_negative_modular_exponent_errors() {
    let err = builtin_pow(&[
        Value::int(2).unwrap(),
        Value::int(-1).unwrap(),
        Value::int(4).unwrap(),
    ])
    .unwrap_err();
    assert!(matches!(err, BuiltinError::ValueError(_)));
    assert!(
        err.to_string()
            .contains("base is not invertible for the given modulus"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pow_zero_to_negative_power_raises_zero_division() {
    let err = builtin_pow(&[Value::int(0).unwrap(), Value::int(-1).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::Raised(_)));
    assert!(
        err.to_string()
            .contains("0.0 cannot be raised to a negative power"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_round_int() {
    let result = builtin_round(&[Value::int(42).unwrap()]).unwrap();
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_round_float() {
    let result = builtin_round(&[Value::float(3.7)]).unwrap();
    assert_eq!(result.as_int(), Some(4));
}

#[test]
fn test_divmod_int_returns_tuple() {
    let result = builtin_divmod(&[Value::int(17).unwrap(), Value::int(5).unwrap()]).unwrap();
    let ptr = result.as_object_ptr().expect("divmod should return tuple");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.get(0).unwrap().as_int(), Some(3));
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(2));
}

#[test]
fn test_divmod_int_negative_divisor() {
    let result = builtin_divmod(&[Value::int(7).unwrap(), Value::int(-3).unwrap()]).unwrap();
    let ptr = result.as_object_ptr().expect("divmod should return tuple");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.get(0).unwrap().as_int(), Some(-3));
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(-2));
}

#[test]
fn test_divmod_float_returns_tuple() {
    let result = builtin_divmod(&[Value::float(7.5), Value::float(2.0)]).unwrap();
    let ptr = result.as_object_ptr().expect("divmod should return tuple");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.get(0).unwrap().as_float(), Some(3.0));
    assert_eq!(tuple.get(1).unwrap().as_float(), Some(1.5));
}

#[test]
fn test_divmod_mixed_numeric() {
    let result = builtin_divmod(&[Value::int(7).unwrap(), Value::float(2.0)]).unwrap();
    let ptr = result.as_object_ptr().expect("divmod should return tuple");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.get(0).unwrap().as_float(), Some(3.0));
    assert_eq!(tuple.get(1).unwrap().as_float(), Some(1.0));
}

#[test]
fn test_divmod_zero_errors() {
    let int_err = builtin_divmod(&[Value::int(1).unwrap(), Value::int(0).unwrap()]);
    assert!(matches!(int_err, Err(BuiltinError::ValueError(_))));

    let float_err = builtin_divmod(&[Value::float(1.0), Value::float(0.0)]);
    assert!(matches!(float_err, Err(BuiltinError::ValueError(_))));
}

#[test]
fn test_repr_primitives() {
    assert_eq!(
        tagged_string_value_to_rust_string(builtin_repr(&[Value::none()]).unwrap()),
        "None"
    );
    assert_eq!(
        tagged_string_value_to_rust_string(builtin_repr(&[Value::bool(true)]).unwrap()),
        "True"
    );
    assert_eq!(
        tagged_string_value_to_rust_string(builtin_repr(&[Value::int(42).unwrap()]).unwrap()),
        "42"
    );
    assert_eq!(
        tagged_string_value_to_rust_string(builtin_repr(&[Value::float(1.5)]).unwrap()),
        "1.5"
    );
}

#[test]
fn test_repr_tagged_string_escaping() {
    let value = Value::string(intern("a'b\n"));
    let repr = tagged_string_value_to_rust_string(builtin_repr(&[value]).unwrap());
    assert_eq!(repr, "'a\\'b\\n'");
}

#[test]
fn test_repr_runtime_string() {
    let (value, ptr) = boxed_value(StringObject::new("runtime"));
    let repr = tagged_string_value_to_rust_string(builtin_repr(&[value]).unwrap());
    assert_eq!(repr, "'runtime'");
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_repr_and_ascii_escape_internal_surrogate_carriers_as_python_surrogates() {
    let surrogate =
        encode_python_code_point(0xDC80).expect("surrogate should map into carrier range");
    let text = format!("A{surrogate}");
    let (value, ptr) = boxed_value(StringObject::from_string(text));

    let repr = tagged_string_value_to_rust_string(builtin_repr(&[value]).unwrap());
    assert_eq!(repr, "'A\\udc80'");

    let ascii = tagged_string_value_to_rust_string(builtin_ascii(&[value]).unwrap());
    assert_eq!(ascii, "'A\\udc80'");

    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_repr_containers() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let (list_value, list_ptr) = boxed_value(list);
    let list_repr = tagged_string_value_to_rust_string(builtin_repr(&[list_value]).unwrap());
    assert_eq!(list_repr, "[1, 2]");
    unsafe { drop_boxed(list_ptr) };

    let tuple = TupleObject::from_slice(&[Value::int(1).unwrap()]);
    let (tuple_value, tuple_ptr) = boxed_value(tuple);
    let tuple_repr = tagged_string_value_to_rust_string(builtin_repr(&[tuple_value]).unwrap());
    assert_eq!(tuple_repr, "(1,)");
    unsafe { drop_boxed(tuple_ptr) };

    let mut dict = DictObject::new();
    dict.set(Value::int(1).unwrap(), Value::int(2).unwrap());
    let (dict_value, dict_ptr) = boxed_value(dict);
    let dict_repr = tagged_string_value_to_rust_string(builtin_repr(&[dict_value]).unwrap());
    assert_eq!(dict_repr, "{1: 2}");
    unsafe { drop_boxed(dict_ptr) };

    let mut set = SetObject::new();
    set.add(Value::int(3).unwrap());
    let (set_value, set_ptr) = boxed_value(set);
    let set_repr = tagged_string_value_to_rust_string(builtin_repr(&[set_value]).unwrap());
    assert_eq!(set_repr, "{3}");
    unsafe { drop_boxed(set_ptr) };
}

#[test]
fn test_repr_range_object() {
    let (range_value, range_ptr) = boxed_value(RangeObject::new(1, 6, 2));
    let repr = tagged_string_value_to_rust_string(builtin_repr(&[range_value]).unwrap());
    assert_eq!(repr, "range(1, 6, 2)");
    unsafe { drop_boxed(range_ptr) };
}

#[test]
fn test_repr_complex_object() {
    let (complex_value, complex_ptr) = boxed_value(ComplexObject::new(1.0, 0.0));
    let repr = tagged_string_value_to_rust_string(builtin_repr(&[complex_value]).unwrap());
    assert_eq!(repr, "(1+0j)");
    unsafe { drop_boxed(complex_ptr) };
}

#[test]
fn test_repr_bytes_and_bytearray_objects() {
    let (bytes_value, bytes_ptr) = boxed_value(BytesObject::from_slice(b"a'\n\\"));
    let bytes_repr = tagged_string_value_to_rust_string(builtin_repr(&[bytes_value]).unwrap());
    assert_eq!(bytes_repr, "b'a\\'\\n\\\\'");
    unsafe { drop_boxed(bytes_ptr) };

    let (bytearray_value, bytearray_ptr) =
        boxed_value(BytesObject::bytearray_from_slice(&[0, 65, 255]));
    let bytearray_repr =
        tagged_string_value_to_rust_string(builtin_repr(&[bytearray_value]).unwrap());
    assert_eq!(bytearray_repr, "bytearray(b'\\x00A\\xff')");
    unsafe { drop_boxed(bytearray_ptr) };
}

#[test]
fn test_repr_exception_uses_exception_type_and_args() {
    let exc = crate::builtins::get_exception_type("ValueError")
        .expect("ValueError should exist")
        .construct(&[Value::string(intern("boom"))]);
    let repr = tagged_string_value_to_rust_string(builtin_repr(&[exc]).unwrap());

    assert_eq!(repr, "ValueError('boom')");
}

#[test]
fn test_repr_staticmethod_and_classmethod_wrappers() {
    let function = FunctionObject::new(
        Arc::new(prism_code::CodeObject::new("demo", "<test>")),
        Arc::from("demo"),
        None,
        None,
    );
    let (function_value, function_ptr) = boxed_value(function);
    let inner_repr = repr_value(function_value, 0).expect("function repr should succeed");

    let (staticmethod_value, staticmethod_ptr) =
        boxed_value(StaticMethodDescriptor::new(function_value));
    let staticmethod_repr =
        tagged_string_value_to_rust_string(builtin_repr(&[staticmethod_value]).unwrap());
    assert_eq!(staticmethod_repr, format!("<staticmethod({inner_repr})>"));

    let (classmethod_value, classmethod_ptr) =
        boxed_value(ClassMethodDescriptor::new(function_value));
    let classmethod_repr =
        tagged_string_value_to_rust_string(builtin_repr(&[classmethod_value]).unwrap());
    assert_eq!(classmethod_repr, format!("<classmethod({inner_repr})>"));

    unsafe {
        drop_boxed(classmethod_ptr);
        drop_boxed(staticmethod_ptr);
        drop_boxed(function_ptr);
    }
}

#[test]
fn test_repr_builtin_function_uses_cpython_display_name() {
    fn sample_builtin(_args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(Value::none())
    }

    let (function_value, function_ptr) = boxed_value(BuiltinFunctionObject::new(
        Arc::from("time.sleep"),
        sample_builtin,
    ));
    let repr = tagged_string_value_to_rust_string(builtin_repr(&[function_value]).unwrap());

    assert_eq!(repr, "<built-in function sleep>");
    unsafe { drop_boxed(function_ptr) };
}

#[test]
fn test_ascii_non_ascii_escaping() {
    let tagged = Value::string(intern("hé"));
    let tagged_ascii = tagged_string_value_to_rust_string(builtin_ascii(&[tagged]).unwrap());
    assert_eq!(tagged_ascii, "'h\\xe9'");

    let (runtime, ptr) = boxed_value(StringObject::new("漢"));
    let runtime_ascii = tagged_string_value_to_rust_string(builtin_ascii(&[runtime]).unwrap());
    assert_eq!(runtime_ascii, "'\\u6f22'");
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_repr_ascii_arity_errors() {
    let repr_err = builtin_repr(&[]);
    assert!(matches!(repr_err, Err(BuiltinError::TypeError(_))));

    let ascii_err = builtin_ascii(&[]);
    assert!(matches!(ascii_err, Err(BuiltinError::TypeError(_))));
}

#[test]
fn test_id_distinguishes_interned_string_values() {
    let platform = Value::string(intern("platform"));
    let uname_result = Value::string(intern("uname_result"));

    let platform_id = builtin_id(&[platform]).unwrap().as_int().unwrap();
    let same_platform_id = builtin_id(&[Value::string(intern("platform"))])
        .unwrap()
        .as_int()
        .unwrap();
    let uname_result_id = builtin_id(&[uname_result]).unwrap().as_int().unwrap();

    assert_ne!(platform_id, 0);
    assert_eq!(platform_id, same_platform_id);
    assert_ne!(platform_id, uname_result_id);
}

#[test]
fn test_hash_int() {
    let result = builtin_hash(&[Value::int(42).unwrap()]).unwrap();
    assert!(result.as_int().is_some());
}

#[test]
fn test_hash_int_float_equivalence() {
    let int_hash = builtin_hash(&[Value::int(42).unwrap()]).unwrap();
    let float_hash = builtin_hash(&[Value::float(42.0)]).unwrap();
    assert_eq!(int_hash.as_int(), float_hash.as_int());
}

#[test]
fn test_hash_tagged_string_by_content() {
    let a = builtin_hash(&[Value::string(intern("alpha"))]).unwrap();
    let b = builtin_hash(&[Value::string(intern("alpha"))]).unwrap();
    assert_eq!(a.as_int(), b.as_int());
}

#[test]
fn test_hash_runtime_string_matches_tagged_string() {
    let tagged = builtin_hash(&[Value::string(intern("runtime"))]).unwrap();
    let (runtime_value, ptr) = boxed_value(StringObject::new("runtime"));
    let runtime = builtin_hash(&[runtime_value]).unwrap();
    assert_eq!(tagged.as_int(), runtime.as_int());
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_hash_module_object_uses_identity_hash() {
    let module = Arc::new(ModuleObject::new("abc"));
    let value = Value::object_ptr(Arc::as_ptr(&module) as *const ());
    let hash = builtin_hash(&[value]).expect("module objects should be hashable");
    assert!(hash.as_int().is_some());
}

#[test]
fn test_hash_tuple_by_contents() {
    let tuple1 = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let (tuple1_value, tuple1_ptr) = boxed_value(tuple1);
    let hash1 = builtin_hash(&[tuple1_value]).unwrap();

    let tuple2 = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let (tuple2_value, tuple2_ptr) = boxed_value(tuple2);
    let hash2 = builtin_hash(&[tuple2_value]).unwrap();

    assert_eq!(hash1.as_int(), hash2.as_int());
    unsafe { drop_boxed(tuple1_ptr) };
    unsafe { drop_boxed(tuple2_ptr) };
}

#[test]
fn test_hash_tuple_unhashable_member_error() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let (list_value, list_ptr) = boxed_value(list);
    let tuple = TupleObject::from_slice(&[list_value]);
    let (tuple_value, tuple_ptr) = boxed_value(tuple);

    let err = builtin_hash(&[tuple_value]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("unhashable type"));

    unsafe { drop_boxed(tuple_ptr) };
    unsafe { drop_boxed(list_ptr) };
}

#[test]
fn test_hash_unhashable_containers_error() {
    let (list_value, list_ptr) = boxed_value(ListObject::new());
    let list_err = builtin_hash(&[list_value]).unwrap_err();
    assert!(list_err.to_string().contains("unhashable type: 'list'"));
    unsafe { drop_boxed(list_ptr) };

    let (dict_value, dict_ptr) = boxed_value(DictObject::new());
    let dict_err = builtin_hash(&[dict_value]).unwrap_err();
    assert!(dict_err.to_string().contains("unhashable type: 'dict'"));
    unsafe { drop_boxed(dict_ptr) };

    let (set_value, set_ptr) = boxed_value(SetObject::new());
    let set_err = builtin_hash(&[set_value]).unwrap_err();
    assert!(set_err.to_string().contains("unhashable type: 'set'"));
    unsafe { drop_boxed(set_ptr) };
}

#[test]
fn test_callable() {
    let result = builtin_callable(&[Value::int(42).unwrap()]).unwrap();
    assert!(!result.is_truthy());

    let result = builtin_callable(&[Value::none()]).unwrap();
    assert!(!result.is_truthy());
}

fn dummy_builtin(_args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::none())
}

#[test]
fn test_callable_builtin_function_true() {
    let builtin = BuiltinFunctionObject::new("dummy".into(), dummy_builtin);
    let (value, ptr) = boxed_value(builtin);
    let result = builtin_callable(&[value]).unwrap();
    assert!(result.is_truthy());
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_callable_type_object_true() {
    #[repr(C)]
    struct DummyTypeObject {
        header: ObjectHeader,
    }

    let dummy = DummyTypeObject {
        header: ObjectHeader::new(TypeId::TYPE),
    };
    let (value, ptr) = boxed_value(dummy);
    let result = builtin_callable(&[value]).unwrap();
    assert!(result.is_truthy());
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_callable_non_callable_object_false() {
    let (value, ptr) = boxed_value(ListObject::new());
    let result = builtin_callable(&[value]).unwrap();
    assert!(!result.is_truthy());
    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_callable_string_false() {
    let result = builtin_callable(&[Value::string(intern("name"))]).unwrap();
    assert!(!result.is_truthy());
}
