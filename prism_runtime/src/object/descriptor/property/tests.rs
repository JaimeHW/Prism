use super::*;

#[test]
fn test_property_getter_only() {
    let getter = Value::int_unchecked(100); // Placeholder
    let prop = PropertyDescriptor::new_getter(getter);

    assert!(prop.has_getter());
    assert!(!prop.has_setter());
    assert!(!prop.has_deleter());
    assert_eq!(prop.kind(), DescriptorKind::Property);
    assert_eq!(prop.header.type_id, TypeId::PROPERTY);
    assert!(prop.is_data_descriptor());
}

#[test]
fn test_property_getter_setter() {
    let getter = Value::int_unchecked(100);
    let setter = Value::int_unchecked(200);
    let prop = PropertyDescriptor::new_getter_setter(getter, setter);

    assert!(prop.has_getter());
    assert!(prop.has_setter());
    assert!(!prop.has_deleter());
}

#[test]
fn test_property_full() {
    let getter = Value::int_unchecked(100);
    let setter = Value::int_unchecked(200);
    let deleter = Value::int_unchecked(300);
    let doc = Value::int_unchecked(400);

    let prop = PropertyDescriptor::new_full(Some(getter), Some(setter), Some(deleter), Some(doc));

    assert!(prop.has_getter());
    assert!(prop.has_setter());
    assert!(prop.has_deleter());
    assert!(prop.property_flags().contains(PropertyFlags::HAS_DOC));
}

#[test]
fn test_property_flags() {
    let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
    let flags = prop.flags();

    assert!(flags.contains(DescriptorFlags::HAS_GET));
    assert!(flags.contains(DescriptorFlags::DATA_DESCRIPTOR));
    assert!(!flags.contains(DescriptorFlags::HAS_SET));
}

#[test]
fn test_property_with_setter() {
    let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
    assert!(!prop.has_setter());

    let prop2 = prop.with_setter(Value::int_unchecked(2));
    assert!(prop2.has_getter());
    assert!(prop2.has_setter());
}

#[test]
fn test_property_with_deleter() {
    let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
    assert!(!prop.has_deleter());

    let prop2 = prop.with_deleter(Value::int_unchecked(3));
    assert!(prop2.has_getter());
    assert!(prop2.has_deleter());
}

#[test]
fn test_property_get_class_access() {
    let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
    // When accessed through class (obj=None), should return the property itself
    let result = prop.get(None, Value::none());
    assert!(result.is_ok());
    assert!(result.unwrap().as_object_ptr().is_some());
}

#[test]
fn test_property_set_readonly_error() {
    let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
    let result = prop.set(Value::int_unchecked(0), Value::int_unchecked(42));
    assert!(result.is_err());
}

#[test]
fn test_property_delete_unsupported_error() {
    let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
    let result = prop.delete(Value::int_unchecked(0));
    assert!(result.is_err());
}

#[test]
fn test_property_no_getter_error() {
    let prop = PropertyDescriptor::new_full(None, Some(Value::int_unchecked(1)), None, None);
    let result = prop.get(Some(Value::int_unchecked(0)), Value::none());
    assert!(result.is_err());
}
