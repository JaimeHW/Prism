use super::*;

#[test]
fn test_classmethod_creation() {
    let func = Value::int_unchecked(100);
    let cm = ClassMethodDescriptor::new(func);

    assert_eq!(cm.function(), func);
}

#[test]
fn test_classmethod_kind() {
    let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
    assert_eq!(cm.kind(), DescriptorKind::ClassMethod);
}

#[test]
fn test_classmethod_flags() {
    let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
    let flags = cm.flags();

    assert!(flags.contains(DescriptorFlags::HAS_GET));
    assert!(flags.contains(DescriptorFlags::CLASSMETHOD));
    assert!(!flags.contains(DescriptorFlags::HAS_SET));
    assert!(!flags.contains(DescriptorFlags::METHOD));
}

#[test]
fn test_classmethod_is_non_data() {
    let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
    assert!(!cm.is_data_descriptor());
}

#[test]
fn test_classmethod_bind() {
    let func = Value::int_unchecked(100);
    let class = Value::int_unchecked(200);
    let cm = ClassMethodDescriptor::new(func);

    let bound = cm.bind(class);
    assert_eq!(bound.function(), func);
    assert_eq!(bound.instance(), class); // Note: "instance" is actually the class
}

#[test]
fn test_classmethod_header_type() {
    let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
    assert_eq!(cm.header.type_id, TypeId::CLASSMETHOD);
}

#[test]
fn test_classmethod_get_through_class() {
    let func = Value::int_unchecked(100);
    let class = Value::int_unchecked(200);
    let cm = ClassMethodDescriptor::new(func);

    // Access through class (obj=None, objtype=class)
    let result = cm.get(None, class);
    assert!(result.is_ok());
    let ptr = result.unwrap().as_object_ptr().unwrap();
    let bound = unsafe { &*(ptr as *const BoundMethod) };
    assert_eq!(bound.function(), func);
    assert_eq!(bound.instance(), class);
}

#[test]
fn test_classmethod_get_through_instance() {
    let func = Value::int_unchecked(100);
    let instance = Value::int_unchecked(42);
    let class = Value::int_unchecked(200);
    let cm = ClassMethodDescriptor::new(func);

    // Access through instance (obj=instance, objtype=class)
    let result = cm.get(Some(instance), class);
    assert!(result.is_ok());
    let ptr = result.unwrap().as_object_ptr().unwrap();
    let bound = unsafe { &*(ptr as *const BoundMethod) };
    assert_eq!(bound.function(), func);
    assert_eq!(bound.instance(), class);
}

#[test]
fn test_classmethod_bind_value_uses_bound_heap() {
    let heap = prism_gc::heap::GcHeap::with_defaults();
    let _binding = crate::allocation_context::RuntimeHeapBinding::register(&heap);
    let cm = ClassMethodDescriptor::new(Value::int_unchecked(100));

    let value = cm.bind_value(Value::int_unchecked(200));
    let ptr = value
        .as_object_ptr()
        .expect("bound classmethod should allocate a method object");

    assert!(heap.contains(ptr));
}

#[test]
fn test_classmethod_set_error() {
    let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
    let result = cm.set(Value::int_unchecked(0), Value::int_unchecked(42));
    assert!(result.is_err());
}

#[test]
fn test_classmethod_delete_error() {
    let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
    let result = cm.delete(Value::int_unchecked(0));
    assert!(result.is_err());
}
