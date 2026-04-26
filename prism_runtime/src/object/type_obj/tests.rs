use super::*;
use prism_core::intern::intern;

#[test]
fn test_type_id_builtin() {
    assert!(TypeId::INT.is_builtin());
    assert!(TypeId::LIST.is_builtin());
    assert!(!TypeId(256).is_builtin());
}

#[test]
fn test_type_object_retains_runtime_type_id() {
    let ty = TypeObject::new(
        TypeId::from_raw(TypeId::FIRST_USER_TYPE + 7),
        intern("HeapCarrier"),
        None,
        0,
        TypeFlags::HEAPTYPE,
    );

    assert_eq!(ty.header.type_id, TypeId::TYPE);
    assert_eq!(ty.type_id(), TypeId::from_raw(TypeId::FIRST_USER_TYPE + 7));
}

#[test]
fn test_type_id_names() {
    assert_eq!(TypeId::INT.name(), "int");
    assert_eq!(TypeId::LIST.name(), "list");
    assert_eq!(TypeId::FUNCTION.name(), "function");
    assert_eq!(TypeId::CELL_VIEW.name(), "cell");
    assert_eq!(TypeId::GENERIC_ALIAS.name(), "generic_alias");
    assert_eq!(TypeId::UNION.name(), "union");
    assert_eq!(TypeId::MAPPING_PROXY.name(), "mappingproxy");
    assert_eq!(TypeId::WRAPPER_DESCRIPTOR.name(), "wrapper_descriptor");
    assert_eq!(TypeId::METHOD_WRAPPER.name(), "method-wrapper");
    assert_eq!(TypeId::METHOD_DESCRIPTOR.name(), "method_descriptor");
    assert_eq!(
        TypeId::CLASSMETHOD_DESCRIPTOR.name(),
        "classmethod_descriptor"
    );
    assert_eq!(TypeId::GETSET_DESCRIPTOR.name(), "getset_descriptor");
    assert_eq!(TypeId::MEMBER_DESCRIPTOR.name(), "member_descriptor");
    assert_eq!(TypeId::TRACEBACK.name(), "traceback");
    assert_eq!(TypeId::FRAME.name(), "frame");
    assert_eq!(TypeId::ELLIPSIS.name(), "ellipsis");
    assert_eq!(TypeId::NOT_IMPLEMENTED.name(), "NotImplementedType");
    assert_eq!(TypeId::DICT_KEYS.name(), "dict_keys");
    assert_eq!(TypeId::DICT_VALUES.name(), "dict_values");
    assert_eq!(TypeId::DICT_ITEMS.name(), "dict_items");
    assert_eq!(TypeId::MEMORYVIEW.name(), "memoryview");
    assert_eq!(TypeId::DEQUE.name(), "deque");
    assert_eq!(TypeId::COMPLEX.name(), "complex");
}

#[test]
fn test_type_flags() {
    let flags = TypeFlags::SEQUENCE | TypeFlags::ITERABLE;
    assert!(flags.contains(TypeFlags::SEQUENCE));
    assert!(flags.contains(TypeFlags::ITERABLE));
    assert!(!flags.contains(TypeFlags::MAPPING));
}
