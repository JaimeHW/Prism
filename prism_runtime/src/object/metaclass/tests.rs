use super::*;
use prism_core::Value;

// =========================================================================
// MetaclassError Tests
// =========================================================================

#[test]
fn test_metaclass_error_display_conflict() {
    let err = MetaclassError::Conflict {
        meta1: TypeId::from_raw(100),
        meta2: TypeId::from_raw(200),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("metaclass conflict"));
    assert!(msg.contains("100"));
    assert!(msg.contains("200"));
}

#[test]
fn test_metaclass_error_display_resolution_failed() {
    let err = MetaclassError::ResolutionFailed {
        message: "test error".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("resolution failed"));
    assert!(msg.contains("test error"));
}

#[test]
fn test_metaclass_error_display_new_failed() {
    let err = MetaclassError::NewFailed {
        message: "__new__ error".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("__new__ failed"));
}

#[test]
fn test_metaclass_error_display_init_failed() {
    let err = MetaclassError::InitFailed {
        message: "__init__ error".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("__init__ failed"));
}

#[test]
fn test_metaclass_error_display_call_failed() {
    let err = MetaclassError::CallFailed {
        message: "__call__ error".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("__call__ failed"));
}

#[test]
fn test_metaclass_error_display_invalid() {
    let err = MetaclassError::InvalidMetaclass {
        type_id: TypeId::from_raw(42),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("invalid metaclass"));
    assert!(msg.contains("42"));
}

// =========================================================================
// TypeMetaclass Tests
// =========================================================================

#[test]
fn test_type_metaclass_new() {
    let meta = TypeMetaclass::new();
    assert_eq!(meta.type_id(), TYPE_METACLASS_ID);
    assert_eq!(meta.classes_created(), 0);
}

#[test]
fn test_type_metaclass_name() {
    let meta = TypeMetaclass::new();
    assert_eq!(meta.name().as_str(), "type");
}

#[test]
fn test_type_metaclass_is_subclass_of_self() {
    let meta = TypeMetaclass::new();
    assert!(meta.is_subclass_of(TYPE_METACLASS_ID));
}

#[test]
fn test_type_metaclass_not_subclass_of_other() {
    let meta = TypeMetaclass::new();
    assert!(!meta.is_subclass_of(TypeId::from_raw(999)));
}

#[test]
fn test_type_metaclass_compute_flags_empty() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    let flags = meta.compute_flags(&namespace);
    assert!(flags.is_empty());
}

#[test]
fn test_type_metaclass_compute_flags_with_init() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    namespace.set(intern("__init__"), Value::int_unchecked(0));
    let flags = meta.compute_flags(&namespace);
    assert!(flags.contains(ClassFlags::HAS_INIT));
}

#[test]
fn test_type_metaclass_compute_flags_with_new() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    namespace.set(intern("__new__"), Value::int_unchecked(0));
    let flags = meta.compute_flags(&namespace);
    assert!(flags.contains(ClassFlags::HAS_NEW));
}

#[test]
fn test_type_metaclass_compute_flags_with_slots() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    namespace.set(intern("__slots__"), Value::int_unchecked(0));
    let flags = meta.compute_flags(&namespace);
    assert!(flags.contains(ClassFlags::HAS_SLOTS));
}

#[test]
fn test_type_metaclass_compute_flags_with_multiple() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    namespace.set(intern("__init__"), Value::int_unchecked(0));
    namespace.set(intern("__eq__"), Value::int_unchecked(0));
    namespace.set(intern("__hash__"), Value::int_unchecked(0));
    let flags = meta.compute_flags(&namespace);
    assert!(flags.contains(ClassFlags::HAS_INIT));
    assert!(flags.contains(ClassFlags::HAS_EQ));
    assert!(flags.contains(ClassFlags::HASHABLE));
}

#[test]
fn test_type_metaclass_instantiation_hint_default() {
    let meta = TypeMetaclass::new();
    assert_eq!(meta.instantiation_hint(), InstantiationHint::Generic);
}

// =========================================================================
// MetaclassResolver Tests
// =========================================================================

#[test]
fn test_resolver_no_bases_no_explicit() {
    let result = MetaclassResolver::resolve(None, &[], |_| TYPE_METACLASS_ID);
    assert_eq!(result.unwrap(), TYPE_METACLASS_ID);
}

#[test]
fn test_resolver_explicit_type() {
    let result = MetaclassResolver::resolve(Some(TYPE_METACLASS_ID), &[], |_| TYPE_METACLASS_ID);
    assert_eq!(result.unwrap(), TYPE_METACLASS_ID);
}

#[test]
fn test_resolver_bases_with_type_metaclass() {
    let bases = [ClassId(100), ClassId(200)];
    let result = MetaclassResolver::resolve(None, &bases, |_| TYPE_METACLASS_ID);
    assert_eq!(result.unwrap(), TYPE_METACLASS_ID);
}

#[test]
fn test_resolver_base_with_custom_metaclass() {
    let custom_meta = TypeId::from_raw(500);
    let bases = [ClassId(100)];
    let result = MetaclassResolver::resolve(None, &bases, |_| custom_meta);
    assert_eq!(result.unwrap(), custom_meta);
}

#[test]
fn test_resolver_conflict() {
    let custom_meta1 = TypeId::from_raw(500);
    let custom_meta2 = TypeId::from_raw(600);
    let bases = [ClassId(100), ClassId(200)];

    // Different metaclasses with explicit should conflict
    let result = MetaclassResolver::resolve(Some(custom_meta1), &bases, |id| {
        if id == ClassId(100) {
            custom_meta2
        } else {
            TYPE_METACLASS_ID
        }
    });

    assert!(matches!(result, Err(MetaclassError::Conflict { .. })));
}

// =========================================================================
// MetaclassCache Tests
// =========================================================================

#[test]
fn test_metaclass_cache_new() {
    let cache = MetaclassCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_metaclass_cache_with_type() {
    let type_meta = Arc::new(TypeMetaclass::new());
    let cache = MetaclassCache::with_type_metaclass(type_meta);
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);
    assert!(cache.contains(TYPE_METACLASS_ID));
}

#[test]
fn test_metaclass_cache_get_existing() {
    let type_meta = Arc::new(TypeMetaclass::new());
    let cache = MetaclassCache::with_type_metaclass(type_meta);

    let result = cache.get(TYPE_METACLASS_ID);
    assert!(result.is_some());
    assert_eq!(result.unwrap().type_id(), TYPE_METACLASS_ID);
}

#[test]
fn test_metaclass_cache_get_missing() {
    let cache = MetaclassCache::new();
    let result = cache.get(TYPE_METACLASS_ID);
    assert!(result.is_none());
}

#[test]
fn test_metaclass_cache_register() {
    let cache = MetaclassCache::new();
    let type_meta = Arc::new(TypeMetaclass::new());
    cache.register(type_meta as Arc<dyn Metaclass>);

    assert!(cache.contains(TYPE_METACLASS_ID));
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_metaclass_cache_stats() {
    let type_meta = Arc::new(TypeMetaclass::new());
    let cache = MetaclassCache::with_type_metaclass(type_meta);

    // Hit
    let _ = cache.get(TYPE_METACLASS_ID);
    // Miss
    let _ = cache.get(TypeId::from_raw(999));

    let (hits, misses) = cache.stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);
}

#[test]
fn test_metaclass_cache_clear() {
    let type_meta = Arc::new(TypeMetaclass::new());
    let cache = MetaclassCache::with_type_metaclass(type_meta);
    assert!(!cache.is_empty());

    cache.clear();
    assert!(cache.is_empty());
}

// =========================================================================
// ClassFactory Tests
// =========================================================================

#[test]
fn test_class_factory_new() {
    let factory = ClassFactory::new();
    assert_eq!(factory.classes_created(), 0);
}

#[test]
fn test_class_factory_type_metaclass() {
    let factory = ClassFactory::new();
    let meta = factory.type_metaclass();
    assert_eq!(meta.type_id(), TYPE_METACLASS_ID);
}

#[test]
fn test_class_factory_get_metaclass() {
    let factory = ClassFactory::new();
    let result = factory.get_metaclass(TYPE_METACLASS_ID);
    assert!(result.is_some());
}

#[test]
fn test_class_factory_get_unknown_metaclass() {
    let factory = ClassFactory::new();
    let result = factory.get_metaclass(TypeId::from_raw(999));
    assert!(result.is_none());
}

#[test]
fn test_class_factory_cache_stats() {
    let factory = ClassFactory::new();
    let _ = factory.get_metaclass(TYPE_METACLASS_ID);
    let _ = factory.get_metaclass(TypeId::from_raw(999));

    let (hits, misses) = factory.cache_stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);
}

// =========================================================================
// Instantiation Hint Tests
// =========================================================================

#[test]
fn test_compute_instantiation_hint_inline() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    let flags = ClassFlags::empty();

    let hint = meta.compute_instantiation_hint(flags, &namespace);
    assert_eq!(hint, InstantiationHint::InlineSlots);
}

#[test]
fn test_compute_instantiation_hint_with_init_only() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    let flags = ClassFlags::HAS_INIT;

    let hint = meta.compute_instantiation_hint(flags, &namespace);
    assert_eq!(hint, InstantiationHint::DefaultInit);
}

#[test]
fn test_compute_instantiation_hint_with_new() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    let flags = ClassFlags::HAS_NEW;

    let hint = meta.compute_instantiation_hint(flags, &namespace);
    assert_eq!(hint, InstantiationHint::Generic);
}

#[test]
fn test_compute_instantiation_hint_with_slots() {
    let meta = TypeMetaclass::new();
    let namespace = ClassDict::new();
    namespace.set(intern("__slots__"), Value::int_unchecked(0));
    let flags = ClassFlags::HAS_SLOTS;

    let hint = meta.compute_instantiation_hint(flags, &namespace);
    assert_eq!(hint, InstantiationHint::FixedSlots);
}
