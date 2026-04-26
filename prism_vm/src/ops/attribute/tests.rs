use super::*;

// =========================================================================
// AttrLookupResult Tests
// =========================================================================

#[test]
fn test_lookup_result_not_found() {
    let result = AttrLookupResult::NotFound;
    assert!(!result.is_found());
    assert!(!result.needs_descriptor_get());
    assert!(result.simple_value().is_none());
}

#[test]
fn test_lookup_result_instance_attr() {
    let value = Value::int(42).unwrap();
    let result = AttrLookupResult::InstanceAttr(value);
    assert!(result.is_found());
    assert!(!result.needs_descriptor_get());
    assert_eq!(result.simple_value(), Some(value));
}

#[test]
fn test_lookup_result_class_attr() {
    let value = Value::bool(true);
    let result = AttrLookupResult::ClassAttr {
        value,
        owner_type: TypeId::OBJECT,
    };
    assert!(result.is_found());
    assert!(!result.needs_descriptor_get());
    assert_eq!(result.simple_value(), Some(value));
}

#[test]
fn test_lookup_result_data_descriptor() {
    let result = AttrLookupResult::DataDescriptor {
        descriptor: Value::none(),
        owner_type: TypeId::OBJECT,
    };
    assert!(result.is_found());
    assert!(result.needs_descriptor_get());
    assert!(result.simple_value().is_none());
}

#[test]
fn test_lookup_result_non_data_descriptor() {
    let result = AttrLookupResult::NonDataDescriptor {
        descriptor: Value::none(),
        owner_type: TypeId::FUNCTION,
    };
    assert!(result.is_found());
    assert!(result.needs_descriptor_get());
    assert!(result.simple_value().is_none());
}

// =========================================================================
// AttrIC Tests
// =========================================================================

#[test]
fn test_ic_new_is_invalid() {
    let ic = AttrIC::new();
    assert!(!ic.matches(1));
    assert!(!ic.matches(0));
}

#[test]
fn test_ic_update_and_match() {
    let mut ic = AttrIC::new();
    ic.update(12345, 3);

    assert!(ic.matches(12345));
    assert!(!ic.matches(12346));
    assert_eq!(ic.slot_index, 3);
}

#[test]
fn test_ic_invalidate() {
    let mut ic = AttrIC::new();
    ic.update(12345, 3);
    assert!(ic.matches(12345));

    ic.invalidate();
    assert!(!ic.matches(12345));
}

// =========================================================================
// Descriptor Detection Tests
// =========================================================================

#[test]
fn test_is_data_descriptor_primitives() {
    // Primitives are never descriptors
    assert!(!is_data_descriptor(Value::none()));
    assert!(!is_data_descriptor(Value::int(42).unwrap()));
    assert!(!is_data_descriptor(Value::bool(true)));
    assert!(!is_data_descriptor(Value::float(3.14)));
}

#[test]
fn test_is_non_data_descriptor_primitives() {
    // Primitives are never descriptors
    assert!(!is_non_data_descriptor(Value::none()));
    assert!(!is_non_data_descriptor(Value::int(42).unwrap()));
    assert!(!is_non_data_descriptor(Value::bool(true)));
    assert!(!is_non_data_descriptor(Value::float(3.14)));
}

// =========================================================================
// User-Defined Type Detection Tests
// =========================================================================

#[test]
fn test_builtin_types_not_user_defined() {
    assert!(!is_user_defined_type(TypeId::OBJECT));
    assert!(!is_user_defined_type(TypeId::INT));
    assert!(!is_user_defined_type(TypeId::STR));
    assert!(!is_user_defined_type(TypeId::LIST));
    assert!(!is_user_defined_type(TypeId::DICT));
}

#[test]
fn test_user_defined_types() {
    // First user type starts at 256
    let user_type = TypeId::from_raw(256);
    assert!(is_user_defined_type(user_type));

    let higher_user_type = TypeId::from_raw(1000);
    assert!(is_user_defined_type(higher_user_type));
}

// =========================================================================
// Descriptor Type Detection Tests
// =========================================================================

#[test]
fn test_function_is_non_data_descriptor() {
    assert!(is_non_data_descriptor_type(TypeId::FUNCTION));
    assert!(is_non_data_descriptor_type(TypeId::METHOD));
}

#[test]
fn test_builtin_types_not_descriptors() {
    assert!(!is_data_descriptor_type(TypeId::INT));
    assert!(!is_data_descriptor_type(TypeId::STR));
    assert!(!is_non_data_descriptor_type(TypeId::INT));
    assert!(!is_non_data_descriptor_type(TypeId::STR));
}
