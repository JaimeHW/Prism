use super::*;
use crate::object::type_obj::TypeFlags;
use prism_core::intern::intern;
use std::sync::Arc;
use std::thread;

fn leaked_type(type_id: TypeId, name: &'static str) -> &'static TypeObject {
    Box::leak(Box::new(TypeObject::new(
        type_id,
        intern(name),
        None,
        0,
        TypeFlags::empty(),
    )))
}

#[test]
fn test_registry_creation() {
    let registry = TypeRegistry::new();
    assert!(registry.is_empty());
}

#[test]
fn test_allocate_type_id() {
    let registry = TypeRegistry::new();
    let id1 = registry.allocate_type_id();
    let id2 = registry.allocate_type_id();
    assert_eq!(id1.raw(), 256);
    assert_eq!(id2.raw(), 257);
    assert!(!id1.is_builtin());
}

#[test]
fn test_register_and_get_uses_dense_index() {
    let registry = TypeRegistry::new();
    let type_id = TypeId::from_raw(TypeId::FIRST_USER_TYPE + 1);
    let type_obj = leaked_type(type_id, "DenseLookup");

    registry.register(type_id, type_obj);

    let loaded = registry
        .get(type_id)
        .expect("registered type should be returned");
    assert!(std::ptr::eq(loaded, type_obj));
    assert!(registry.contains(type_id));
    assert_eq!(registry.len(), 1);
}

#[test]
fn test_register_uses_overflow_table_for_large_type_ids() {
    let registry = TypeRegistry::new();
    let type_id = TypeId::from_raw(TypeId::FIRST_USER_TYPE + FAST_TABLE_CAPACITY as u32 + 64);
    let type_obj = leaked_type(type_id, "OverflowLookup");

    registry.register(type_id, type_obj);

    assert!(std::ptr::eq(registry.get(type_id).unwrap(), type_obj));
}

#[test]
fn test_register_same_type_is_idempotent() {
    let registry = TypeRegistry::new();
    let type_id = TypeId::from_raw(TypeId::FIRST_USER_TYPE + 8);
    let type_obj = leaked_type(type_id, "Idempotent");

    registry.register(type_id, type_obj);
    registry.register(type_id, type_obj);

    assert_eq!(registry.len(), 1);
}

#[test]
fn test_concurrent_reads_are_lock_free_and_stable() {
    let registry = Arc::new(TypeRegistry::new());
    let type_id = TypeId::from_raw(TypeId::FIRST_USER_TYPE + 16);
    let type_obj = leaked_type(type_id, "ConcurrentLookup");
    registry.register(type_id, type_obj);

    let mut threads = Vec::new();
    for _ in 0..8 {
        let registry = Arc::clone(&registry);
        threads.push(thread::spawn(move || {
            for _ in 0..10_000 {
                let loaded = registry.get(type_id).expect("type should stay published");
                assert!(std::ptr::eq(loaded, type_obj));
            }
        }));
    }

    for handle in threads {
        handle.join().expect("reader thread should complete");
    }
}
