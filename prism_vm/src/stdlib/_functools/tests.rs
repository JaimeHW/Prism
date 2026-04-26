use super::*;

#[test]
fn test_module_exposes_partial_type() {
    let module = FunctoolsNativeModule::new();
    assert_eq!(module.name(), "_functools");
    assert!(module.get_attr("partial").is_ok());
    assert_eq!(module.dir(), vec![Arc::from("partial")]);
}

#[test]
fn test_partial_type_is_registered_native_heap_type() {
    let ptr = partial_class_value()
        .as_object_ptr()
        .expect("partial type should be an object");
    assert_eq!(extract_type_id(ptr), TypeId::TYPE);
    assert!(PARTIAL_CLASS.is_native_heaptype());
    assert!(global_class_bitmap(PARTIAL_CLASS.class_id()).is_some());
}
