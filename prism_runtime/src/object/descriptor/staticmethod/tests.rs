use super::*;

#[test]
fn test_staticmethod_creation() {
    let func = Value::int_unchecked(100);
    let sm = StaticMethodDescriptor::new(func);

    assert_eq!(sm.function(), func);
    assert_eq!(sm.header.type_id, TypeId::STATICMETHOD);
}

#[test]
fn test_staticmethod_kind() {
    let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
    assert_eq!(sm.kind(), DescriptorKind::StaticMethod);
}

#[test]
fn test_staticmethod_flags() {
    let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
    let flags = sm.flags();

    assert!(flags.contains(DescriptorFlags::HAS_GET));
    assert!(flags.contains(DescriptorFlags::STATICMETHOD));
    assert!(!flags.contains(DescriptorFlags::HAS_SET));
    assert!(!flags.contains(DescriptorFlags::METHOD));
    assert!(!flags.contains(DescriptorFlags::CLASSMETHOD));
}

#[test]
fn test_staticmethod_is_non_data() {
    let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
    assert!(!sm.is_data_descriptor());
}

#[test]
fn test_staticmethod_get_through_class() {
    let func = Value::int_unchecked(100);
    let class = Value::int_unchecked(200);
    let sm = StaticMethodDescriptor::new(func);

    // Access through class (obj=None)
    let result = sm.get(None, class);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), func);
}

#[test]
fn test_staticmethod_get_through_instance() {
    let func = Value::int_unchecked(100);
    let instance = Value::int_unchecked(42);
    let class = Value::int_unchecked(200);
    let sm = StaticMethodDescriptor::new(func);

    // Access through instance (obj=instance)
    let result = sm.get(Some(instance), class);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), func);
}

#[test]
fn test_staticmethod_always_returns_same_function() {
    let func = Value::int_unchecked(100);
    let sm = StaticMethodDescriptor::new(func);

    // Access multiple times - should always return same function
    assert_eq!(sm.get(None, Value::none()).unwrap(), func);
    assert_eq!(
        sm.get(Some(Value::int_unchecked(1)), Value::none())
            .unwrap(),
        func
    );
    assert_eq!(
        sm.get(Some(Value::int_unchecked(2)), Value::none())
            .unwrap(),
        func
    );
}

#[test]
fn test_staticmethod_set_error() {
    let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
    let result = sm.set(Value::int_unchecked(0), Value::int_unchecked(42));
    assert!(result.is_err());
}

#[test]
fn test_staticmethod_delete_error() {
    let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
    let result = sm.delete(Value::int_unchecked(0));
    assert!(result.is_err());
}

#[test]
fn test_staticmethod_size() {
    assert_eq!(
        std::mem::size_of::<StaticMethodDescriptor>(),
        std::mem::size_of::<ObjectHeader>() + std::mem::size_of::<Value>()
    );
}
