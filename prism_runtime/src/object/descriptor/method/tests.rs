use super::*;
use prism_core::intern::intern;

#[test]
fn test_method_descriptor_creation() {
    let func = Value::int_unchecked(100); // Placeholder function
    let desc = MethodDescriptor::new(func);

    assert_eq!(desc.function(), func);
    assert!(desc.name().is_none());
}

#[test]
fn test_method_descriptor_named() {
    let func = Value::int_unchecked(100);
    let name = intern("my_method");
    let desc = MethodDescriptor::new_named(func, name.clone());

    assert_eq!(desc.name(), Some(&name));
}

#[test]
fn test_method_descriptor_kind() {
    let desc = MethodDescriptor::new(Value::int_unchecked(1));
    assert_eq!(desc.kind(), DescriptorKind::Method);
}

#[test]
fn test_method_descriptor_flags() {
    let desc = MethodDescriptor::new(Value::int_unchecked(1));
    let flags = desc.flags();

    assert!(flags.contains(DescriptorFlags::HAS_GET));
    assert!(flags.contains(DescriptorFlags::METHOD));
    assert!(!flags.contains(DescriptorFlags::HAS_SET));
    assert!(!flags.contains(DescriptorFlags::DATA_DESCRIPTOR));
}

#[test]
fn test_method_descriptor_is_non_data() {
    let desc = MethodDescriptor::new(Value::int_unchecked(1));
    assert!(!desc.is_data_descriptor());
}

#[test]
fn test_method_descriptor_class_access() {
    let func = Value::int_unchecked(100);
    let desc = MethodDescriptor::new(func);

    // Access through class (obj=None) returns the function
    let result = desc.get(None, Value::none());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), func);
}

#[test]
fn test_method_descriptor_bind() {
    let func = Value::int_unchecked(100);
    let desc = MethodDescriptor::new(func);
    let instance = Value::int_unchecked(42);

    let bound = desc.bind(instance);
    assert_eq!(bound.function(), func);
    assert_eq!(bound.instance(), instance);
}

#[test]
fn test_bound_method_creation() {
    let func = Value::int_unchecked(100);
    let instance = Value::int_unchecked(42);
    let bound = BoundMethod::new(func, instance);

    assert_eq!(bound.function(), func);
    assert_eq!(bound.instance(), instance);
    assert_eq!(bound.header.type_id, TypeId::METHOD);
}

#[test]
fn test_bound_method_equality() {
    let func = Value::int_unchecked(100);
    let instance = Value::int_unchecked(42);

    let bound1 = BoundMethod::new(func, instance);
    let bound2 = BoundMethod::new(func, instance);
    let bound3 = BoundMethod::new(func, Value::int_unchecked(43));

    assert_eq!(bound1, bound2);
    assert_ne!(bound1, bound3);
}

#[test]
fn test_bound_method_hash() {
    let func = Value::int_unchecked(100);
    let instance = Value::int_unchecked(42);

    let bound1 = BoundMethod::new(func, instance);
    let bound2 = BoundMethod::new(func, instance);

    assert_eq!(bound1.hash(), bound2.hash());
}

#[test]
fn test_bound_method_call() {
    let func = Value::int_unchecked(100);
    let instance = Value::int_unchecked(42);
    let bound = BoundMethod::new(func, instance);

    let result = bound.call(&[]);
    assert!(result.is_ok());
}

#[test]
fn test_unbound_method_creation() {
    let func = Value::int_unchecked(100);
    let class = Value::int_unchecked(200);
    let unbound = UnboundMethod::new(func, class);

    assert_eq!(unbound.function(), func);
    assert_eq!(unbound.class(), class);
}

#[test]
fn test_unbound_method_bind() {
    let func = Value::int_unchecked(100);
    let class = Value::int_unchecked(200);
    let instance = Value::int_unchecked(42);

    let unbound = UnboundMethod::new(func, class);
    let bound = unbound.bind(instance);

    assert!(bound.is_ok());
    let bound = bound.unwrap();
    assert_eq!(bound.function(), func);
    assert_eq!(bound.instance(), instance);
}

#[test]
fn test_bound_method_size() {
    // BoundMethod should be small: header (16) + function (8) + instance (8) = 32
    assert_eq!(std::mem::size_of::<BoundMethod>(), 32);
}

#[test]
fn test_method_descriptor_set_error() {
    let desc = MethodDescriptor::new(Value::int_unchecked(1));
    let result = desc.set(Value::int_unchecked(0), Value::int_unchecked(42));
    assert!(result.is_err());
}

#[test]
fn test_method_descriptor_delete_error() {
    let desc = MethodDescriptor::new(Value::int_unchecked(1));
    let result = desc.delete(Value::int_unchecked(0));
    assert!(result.is_err());
}
