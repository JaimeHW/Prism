use super::*;
use crate::profiler::CodeId;
use prism_core::intern::{intern, interned_by_ptr};

fn value_to_rust_string(value: Value) -> String {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .expect("tagged string should expose a string payload") as *const u8;
        return interned_by_ptr(ptr)
            .expect("tagged string should resolve through the interner")
            .as_str()
            .to_string();
    }

    let ptr = value
        .as_object_ptr()
        .expect("string result should be object-backed");
    let string = unsafe { &*(ptr as *const StringObject) };
    string.as_str().to_string()
}

#[test]
fn test_speculation_from_operand_pair() {
    assert_eq!(
        Speculation::from_operand_pair(OperandPair(0x11)),
        Speculation::IntInt
    );
    assert_eq!(
        Speculation::from_operand_pair(OperandPair(0x22)),
        Speculation::FloatFloat
    );
    assert_eq!(
        Speculation::from_operand_pair(OperandPair(0x12)),
        Speculation::IntFloat
    );
    assert_eq!(
        Speculation::from_operand_pair(OperandPair(0x21)),
        Speculation::FloatInt
    );
    assert_eq!(
        Speculation::from_operand_pair(OperandPair(0x99)),
        Speculation::None
    );
}

#[test]
fn test_speculation_cache_insert_get() {
    let mut cache = SpeculationCache::new();
    let site = ICSiteId::new(CodeId::new(123), 42);

    assert!(cache.get(site).is_none());

    cache.insert(site, Speculation::IntInt);
    assert_eq!(cache.get(site), Some(Speculation::IntInt));

    cache.insert(site, Speculation::FloatFloat);
    assert_eq!(cache.get(site), Some(Speculation::FloatFloat));
}

#[test]
fn test_speculation_cache_invalidate() {
    let mut cache = SpeculationCache::new();
    let site = ICSiteId::new(CodeId::new(456), 100);

    cache.insert(site, Speculation::IntInt);
    assert_eq!(cache.get(site), Some(Speculation::IntInt));

    cache.invalidate(site);
    assert!(cache.get(site).is_none());
}

#[test]
fn test_spec_add_int_success() {
    let a = Value::int(10).unwrap();
    let b = Value::int(20).unwrap();

    let (result, value) = spec_add_int(a, b);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(30));
}

#[test]
fn test_spec_add_int_deopt_on_float() {
    let a = Value::int(10).unwrap();
    let b = Value::float(20.5);

    let (result, _) = spec_add_int(a, b);
    assert_eq!(result, SpecResult::Deopt);
}

#[test]
fn test_spec_add_float_success() {
    let a = Value::float(10.5);
    let b = Value::float(20.5);

    let (result, value) = spec_add_float(a, b);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_float(), Some(31.0));
}

#[test]
fn test_spec_add_float_with_int_promotion() {
    let a = Value::int(10).unwrap();
    let b = Value::float(20.5);

    let (result, value) = spec_add_float(a, b);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_float(), Some(30.5));
}

#[test]
fn test_spec_div_zero_overflow() {
    let a = Value::float(10.0);
    let b = Value::float(0.0);

    let (result, _) = spec_div_float(a, b);
    assert_eq!(result, SpecResult::Overflow);
}

#[test]
fn test_spec_floor_div_int() {
    let a = Value::int(7).unwrap();
    let b = Value::int(3).unwrap();

    let (result, value) = spec_floor_div_int(a, b);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(2));
}

#[test]
fn test_spec_mod_int() {
    let a = Value::int(7).unwrap();
    let b = Value::int(3).unwrap();

    let (result, value) = spec_mod_int(a, b);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(1));
}

#[test]
fn test_spec_floor_div_int_uses_python_negative_divisor_rules() {
    let (result, value) = spec_floor_div_int(Value::int(10).unwrap(), Value::int(-3).unwrap());
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(-4));

    let (result, value) = spec_floor_div_int(Value::int(-10).unwrap(), Value::int(-3).unwrap());
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(3));
}

#[test]
fn test_spec_mod_int_uses_python_negative_divisor_rules() {
    let (result, value) = spec_mod_int(Value::int(10).unwrap(), Value::int(-3).unwrap());
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(-2));

    let (result, value) = spec_mod_int(Value::int(-10).unwrap(), Value::int(-3).unwrap());
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(-1));
}

#[test]
fn test_spec_pow_int() {
    let a = Value::int(2).unwrap();
    let b = Value::int(10).unwrap();

    let (result, value) = spec_pow_int(a, b);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(1024));
}

// =========================================================================
// String Speculation Tests
// =========================================================================

#[test]
fn test_spec_str_concat_deopt_on_non_strings() {
    // Test that non-string operands cause deopt
    let vm = VirtualMachine::new();
    let a = Value::int(10).unwrap();
    let b = Value::int(20).unwrap();

    let (result, _) = spec_str_concat(&vm, a, b).unwrap();
    assert_eq!(result, SpecResult::Deopt);
}

#[test]
fn test_spec_str_concat_deopt_on_mixed_types() {
    // String + int should deopt (not supported for concat)
    let vm = VirtualMachine::new();
    let str_obj = StringObject::new("hello");
    let boxed = Box::new(str_obj);
    let ptr = Box::into_raw(boxed) as *const ();
    let a = Value::object_ptr(ptr);
    let b = Value::int(5).unwrap();

    let (result, _) = spec_str_concat(&vm, a, b).unwrap();
    assert_eq!(result, SpecResult::Deopt);

    // Cleanup
    unsafe {
        drop(Box::from_raw(ptr as *mut StringObject));
    }
}

#[test]
fn test_spec_str_concat_accepts_tagged_strings() {
    let vm = VirtualMachine::new();
    let (result, value) = spec_str_concat(
        &vm,
        Value::string(intern("hello")),
        Value::string(intern(" world")),
    )
    .unwrap();
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value_to_rust_string(value), "hello world");
}

#[test]
fn test_spec_str_repeat_deopt_on_non_string_int() {
    // float * int should deopt
    let vm = VirtualMachine::new();
    let a = Value::float(3.14);
    let b = Value::int(5).unwrap();

    let (result, _) = spec_str_repeat(&vm, a, b).unwrap();
    assert_eq!(result, SpecResult::Deopt);
}

#[test]
fn test_spec_str_repeat_negative_count() {
    let vm = VirtualMachine::new();
    let (result, value) =
        spec_str_repeat(&vm, Value::string(intern("hello")), Value::int(-1).unwrap()).unwrap();
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value_to_rust_string(value), "");
}

#[test]
fn test_spec_str_repeat_int_str_order() {
    // int * str should also work (Python supports both orderings)
    let vm = VirtualMachine::new();
    let str_obj = StringObject::new("ab");
    let boxed = Box::new(str_obj);
    let ptr = Box::into_raw(boxed) as *const ();
    let a = Value::int(3).unwrap();
    let b = Value::object_ptr(ptr);

    let (result, value) = spec_str_repeat(&vm, a, b).unwrap();
    assert_eq!(result, SpecResult::Success);

    // Verify the result is a valid object pointer
    assert!(value.is_object());

    // Cleanup the original test string object.
    unsafe {
        drop(Box::from_raw(ptr as *mut StringObject));
    }
    assert_eq!(value_to_rust_string(value), "ababab");
}

#[test]
fn test_spec_str_repeat_accepts_tagged_strings() {
    let vm = VirtualMachine::new();
    let (result, value) =
        spec_str_repeat(&vm, Value::string(intern("ab")), Value::int(3).unwrap()).unwrap();
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value_to_rust_string(value), "ababab");
}

#[test]
fn test_spec_str_len_success() {
    let str_obj = StringObject::new("hello world");
    let boxed = Box::new(str_obj);
    let ptr = Box::into_raw(boxed) as *const ();
    let a = Value::object_ptr(ptr);

    let (result, value) = spec_str_len(a);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(11));

    // Cleanup
    unsafe {
        drop(Box::from_raw(ptr as *mut StringObject));
    }
}

#[test]
fn test_spec_str_len_deopt_on_non_string() {
    let a = Value::int(42).unwrap();

    let (result, _) = spec_str_len(a);
    assert_eq!(result, SpecResult::Deopt);
}

#[test]
fn test_spec_str_len_empty_string() {
    let str_obj = StringObject::new("");
    let boxed = Box::new(str_obj);
    let ptr = Box::into_raw(boxed) as *const ();
    let a = Value::object_ptr(ptr);

    let (result, value) = spec_str_len(a);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(0));

    // Cleanup
    unsafe {
        drop(Box::from_raw(ptr as *mut StringObject));
    }
}

#[test]
fn test_spec_str_len_accepts_tagged_strings() {
    let (result, value) = spec_str_len(Value::string(intern("hello")));
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(5));
}

#[test]
fn test_spec_str_len_counts_unicode_scalar_values() {
    let str_obj = StringObject::new("hé 🦀");
    let boxed = Box::new(str_obj);
    let ptr = Box::into_raw(boxed) as *const ();
    let a = Value::object_ptr(ptr);

    let (result, value) = spec_str_len(a);
    assert_eq!(result, SpecResult::Success);
    assert_eq!(value.as_int(), Some(4));

    unsafe {
        drop(Box::from_raw(ptr as *mut StringObject));
    }
}
