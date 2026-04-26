use super::*;
use crate::intern::intern;
use std::sync::Arc;

#[test]
fn test_value_size() {
    assert_eq!(std::mem::size_of::<Value>(), 8);
}

#[test]
fn test_none_creation() {
    let v = Value::none();
    assert!(v.is_none());
    assert!(v.is_tagged());
    assert!(!v.is_float());
    assert!(!v.is_bool());
    assert!(!v.is_int());
}

#[test]
fn test_bool_true() {
    let v = Value::bool(true);
    assert!(v.is_bool());
    assert_eq!(v.as_bool(), Some(true));
    assert!(v.is_truthy());
}

#[test]
fn test_bool_false() {
    let v = Value::bool(false);
    assert!(v.is_bool());
    assert_eq!(v.as_bool(), Some(false));
    assert!(!v.is_truthy());
}

#[test]
fn test_string_does_not_bump_interned_arc_refcount() {
    let interned = intern("stable-pointer");
    let arc = interned.get_arc();
    let baseline = Arc::strong_count(&arc);

    for _ in 0..256 {
        let _ = Value::string(interned.clone());
    }

    assert_eq!(Arc::strong_count(&arc), baseline);
}

#[test]
fn test_int_zero() {
    let v = Value::int(0).unwrap();
    assert!(v.is_int());
    assert_eq!(v.as_int(), Some(0));
    assert!(!v.is_truthy());
}

#[test]
fn test_int_positive() {
    let v = Value::int(42).unwrap();
    assert!(v.is_int());
    assert_eq!(v.as_int(), Some(42));
    assert!(v.is_truthy());
}

#[test]
fn test_int_negative() {
    let v = Value::int(-42).unwrap();
    assert!(v.is_int());
    assert_eq!(v.as_int(), Some(-42));
    assert!(v.is_truthy());
}

#[test]
fn test_int_max_small() {
    let v = Value::int(SMALL_INT_MAX).unwrap();
    assert!(v.is_int());
    assert_eq!(v.as_int(), Some(SMALL_INT_MAX));
}

#[test]
fn test_int_min_small() {
    let v = Value::int(SMALL_INT_MIN).unwrap();
    assert!(v.is_int());
    assert_eq!(v.as_int(), Some(SMALL_INT_MIN));
}

#[test]
fn test_int_too_large() {
    let v = Value::int(SMALL_INT_MAX + 1);
    assert!(v.is_none());
}

#[test]
fn test_int_too_small() {
    let v = Value::int(SMALL_INT_MIN - 1);
    assert!(v.is_none());
}

#[test]
fn test_float_positive() {
    let v = Value::float(3.14);
    assert!(v.is_float());
    assert_eq!(v.as_float(), Some(3.14));
    assert!(v.is_truthy());
}

#[test]
fn test_float_negative() {
    let v = Value::float(-2.71);
    assert!(v.is_float());
    assert_eq!(v.as_float(), Some(-2.71));
}

#[test]
fn test_float_zero() {
    let v = Value::float(0.0);
    assert!(v.is_float());
    assert_eq!(v.as_float(), Some(0.0));
    assert!(!v.is_truthy());
}

#[test]
fn test_float_negative_zero() {
    let v = Value::float(-0.0);
    assert!(v.is_float());
    let f = v.as_float().unwrap();
    assert!(f == 0.0);
}

#[test]
fn test_float_infinity() {
    let v = Value::float(f64::INFINITY);
    assert!(v.is_float());
    assert_eq!(v.as_float(), Some(f64::INFINITY));
}

#[test]
fn test_float_neg_infinity() {
    let v = Value::float(f64::NEG_INFINITY);
    assert!(v.is_float());
    assert_eq!(v.as_float(), Some(f64::NEG_INFINITY));
}

#[test]
fn test_float_nan() {
    let v = Value::float(f64::NAN);
    assert!(v.is_float());
    assert!(v.as_float().unwrap().is_nan());
}

#[test]
fn test_float_coerce_from_int() {
    let v = Value::int(42).unwrap();
    assert_eq!(v.as_float_coerce(), Some(42.0));
}

#[test]
fn test_float_coerce_from_float() {
    let v = Value::float(3.14);
    assert_eq!(v.as_float_coerce(), Some(3.14));
}

#[test]
fn test_type_name_none() {
    assert_eq!(Value::none().type_name(), "NoneType");
}

#[test]
fn test_type_name_bool() {
    assert_eq!(Value::bool(true).type_name(), "bool");
}

#[test]
fn test_type_name_int() {
    assert_eq!(Value::int(42).unwrap().type_name(), "int");
}

#[test]
fn test_type_name_float() {
    assert_eq!(Value::float(3.14).type_name(), "float");
}

#[test]
fn test_equality_none() {
    assert_eq!(Value::none(), Value::none());
}

#[test]
fn test_equality_bool() {
    assert_eq!(Value::bool(true), Value::bool(true));
    assert_eq!(Value::bool(false), Value::bool(false));
    assert_ne!(Value::bool(true), Value::bool(false));
}

#[test]
fn test_equality_int() {
    assert_eq!(Value::int(42).unwrap(), Value::int(42).unwrap());
    assert_ne!(Value::int(42).unwrap(), Value::int(43).unwrap());
}

#[test]
fn test_equality_float() {
    assert_eq!(Value::float(3.14), Value::float(3.14));
    assert_ne!(Value::float(3.14), Value::float(3.15));
}

#[test]
fn test_equality_nan() {
    // NaN != NaN in IEEE 754
    assert_ne!(Value::float(f64::NAN), Value::float(f64::NAN));
}

#[test]
fn test_equality_int_float_coercion() {
    // Python: 1 == 1.0
    assert_eq!(Value::int(1).unwrap(), Value::float(1.0));
    assert_eq!(Value::float(1.0), Value::int(1).unwrap());
    assert_eq!(Value::int(0).unwrap(), Value::float(0.0));
}

#[test]
fn test_equality_int_float_mismatch() {
    assert_ne!(Value::int(1).unwrap(), Value::float(1.5));
}

#[test]
fn test_hash_int_float_equivalence() {
    use std::collections::hash_map::DefaultHasher;

    let int_val = Value::int(42).unwrap();
    let float_val = Value::float(42.0);

    let mut hasher1 = DefaultHasher::new();
    int_val.hash(&mut hasher1);
    let hash1 = hasher1.finish();

    let mut hasher2 = DefaultHasher::new();
    float_val.hash(&mut hasher2);
    let hash2 = hasher2.finish();

    assert_eq!(hash1, hash2);
}

#[test]
fn test_debug_none() {
    let v = Value::none();
    assert!(format!("{:?}", v).contains("None"));
}

#[test]
fn test_debug_bool() {
    let v = Value::bool(true);
    assert!(format!("{:?}", v).contains("True"));
}

#[test]
fn test_debug_int() {
    let v = Value::int(42).unwrap();
    assert!(format!("{:?}", v).contains("42"));
}

#[test]
fn test_debug_float() {
    let v = Value::float(3.14);
    let debug = format!("{:?}", v);
    assert!(debug.contains("3.14"));
}

#[test]
fn test_display_none() {
    assert_eq!(format!("{}", Value::none()), "None");
}

#[test]
fn test_display_bool_true() {
    assert_eq!(format!("{}", Value::bool(true)), "True");
}

#[test]
fn test_display_bool_false() {
    assert_eq!(format!("{}", Value::bool(false)), "False");
}

#[test]
fn test_display_int() {
    assert_eq!(format!("{}", Value::int(42).unwrap()), "42");
    assert_eq!(format!("{}", Value::int(-42).unwrap()), "-42");
}

#[test]
fn test_display_float() {
    let v = Value::float(3.14);
    assert_eq!(format!("{}", v), "3.14");
}

#[test]
fn test_display_float_integer_value() {
    let v = Value::float(42.0);
    assert_eq!(format!("{}", v), "42.0");
}

#[test]
fn test_from_bool() {
    let v: Value = true.into();
    assert!(v.is_bool());
    assert_eq!(v.as_bool(), Some(true));
}

#[test]
fn test_from_f64() {
    let v: Value = 3.14_f64.into();
    assert!(v.is_float());
}

#[test]
fn test_from_f32() {
    let v: Value = 3.14_f32.into();
    assert!(v.is_float());
}

#[test]
fn test_try_from_i64_success() {
    let v: Result<Value, _> = 42_i64.try_into();
    assert!(v.is_ok());
    assert_eq!(v.unwrap().as_int(), Some(42));
}

#[test]
fn test_try_from_i64_overflow() {
    let v: Result<Value, _> = i64::MAX.try_into();
    assert!(v.is_err());
}

#[test]
fn test_from_i32() {
    let v: Value = 42_i32.into();
    assert!(v.is_int());
    assert_eq!(v.as_int(), Some(42));
}

#[test]
fn test_from_i16() {
    let v: Value = 42_i16.into();
    assert!(v.is_int());
}

#[test]
fn test_from_i8() {
    let v: Value = 42_i8.into();
    assert!(v.is_int());
}

#[test]
fn test_from_u32() {
    let v: Value = 42_u32.into();
    assert!(v.is_int());
}

#[test]
fn test_from_u16() {
    let v: Value = 42_u16.into();
    assert!(v.is_int());
}

#[test]
fn test_from_u8() {
    let v: Value = 42_u8.into();
    assert!(v.is_int());
}

#[test]
fn test_bits_roundtrip() {
    let values = [
        Value::none(),
        Value::bool(true),
        Value::bool(false),
        Value::int(0).unwrap(),
        Value::int(42).unwrap(),
        Value::int(-42).unwrap(),
        Value::float(0.0),
        Value::float(3.14),
        Value::float(-2.71),
    ];

    for v in values {
        let bits = v.to_bits();
        let reconstructed = Value::from_bits(bits);
        assert_eq!(v.to_bits(), reconstructed.to_bits());
    }
}

#[test]
fn test_all_small_int_range() {
    // Test boundaries
    for i in [
        SMALL_INT_MIN,
        SMALL_INT_MIN + 1,
        -1,
        0,
        1,
        SMALL_INT_MAX - 1,
        SMALL_INT_MAX,
    ] {
        let v = Value::int(i).expect("Should fit");
        assert_eq!(v.as_int(), Some(i), "Failed for {}", i);
    }
}

#[test]
fn test_negative_int_sign_extension() {
    let values = [-1_i64, -2, -100, -1000, -1_000_000, SMALL_INT_MIN];
    for i in values {
        let v = Value::int(i).unwrap();
        assert_eq!(v.as_int(), Some(i), "Sign extension failed for {}", i);
    }
}

#[test]
fn test_object_pointer() {
    let data = Box::new(42_u64);
    let ptr = Box::into_raw(data) as *const ();

    let v = Value::object_ptr(ptr);
    assert!(v.is_object());
    assert_eq!(v.as_object_ptr(), Some(ptr));

    // Cleanup
    unsafe {
        drop(Box::from_raw(ptr as *mut u64));
    }
}

#[test]
#[should_panic(expected = "Pointer too large for NaN-boxing")]
fn test_object_pointer_rejects_non_canonical_payload() {
    let ptr = ((PAYLOAD_MASK + 1) as usize) as *const ();
    let _ = Value::object_ptr(ptr);
}

#[test]
fn test_truthiness_comprehensive() {
    // Falsy values
    assert!(!Value::none().is_truthy());
    assert!(!Value::bool(false).is_truthy());
    assert!(!Value::int(0).unwrap().is_truthy());
    assert!(!Value::float(0.0).is_truthy());

    // Truthy values
    assert!(Value::bool(true).is_truthy());
    assert!(Value::int(1).unwrap().is_truthy());
    assert!(Value::int(-1).unwrap().is_truthy());
    assert!(Value::float(0.1).is_truthy());
    assert!(Value::float(-0.1).is_truthy());
    assert!(Value::float(f64::INFINITY).is_truthy());
}

#[test]
fn test_as_bool_on_non_bool() {
    assert_eq!(Value::none().as_bool(), None);
    assert_eq!(Value::int(42).unwrap().as_bool(), None);
    assert_eq!(Value::float(3.14).as_bool(), None);
}

#[test]
fn test_as_int_on_non_int() {
    assert_eq!(Value::none().as_int(), None);
    assert_eq!(Value::bool(true).as_int(), None);
    assert_eq!(Value::float(3.14).as_int(), None);
}

#[test]
fn test_as_float_on_non_float() {
    assert_eq!(Value::none().as_float(), None);
    assert_eq!(Value::bool(true).as_float(), None);
    assert_eq!(Value::int(42).unwrap().as_float(), None);
}

#[test]
fn test_value_in_hashmap() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    map.insert(Value::int(1).unwrap(), "one");
    map.insert(Value::int(2).unwrap(), "two");

    assert_eq!(map.get(&Value::int(1).unwrap()), Some(&"one"));
    assert_eq!(map.get(&Value::float(1.0)), Some(&"one")); // Coercion!
}

#[test]
fn test_value_in_hashset() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(Value::int(42).unwrap());

    assert!(set.contains(&Value::int(42).unwrap()));
    assert!(set.contains(&Value::float(42.0))); // Same hash
}

#[test]
fn test_special_floats() {
    let values = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::EPSILON,
    ];

    for f in values {
        let v = Value::float(f);
        assert!(v.is_float());
        assert_eq!(v.as_float(), Some(f));
    }
}

#[test]
fn test_subnormal_floats() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    assert!(subnormal.is_subnormal());

    let v = Value::float(subnormal);
    assert!(v.is_float());
    assert_eq!(v.as_float(), Some(subnormal));
}
