use super::*;

#[test]
fn test_handler_entry_new() {
    let handler = 0x12345678 as *const ();
    let entry = HandlerEntry::new(handler);

    assert_eq!(entry.get(), handler);
    assert_eq!(entry.patch_count(), 0);
}

#[test]
fn test_handler_entry_set() {
    let entry = HandlerEntry::new(ptr::null());
    let new_handler = 0x12345678 as *const ();

    entry.set(new_handler);

    assert_eq!(entry.get(), new_handler);
    assert_eq!(entry.patch_count(), 1);
}

#[test]
fn test_handler_entry_compare_exchange() {
    let initial = 0x1000 as *const ();
    let entry = HandlerEntry::new(initial);

    let new = 0x2000 as *const ();
    let result = entry.compare_exchange(initial, new);

    assert!(result.is_ok());
    assert_eq!(entry.get(), new);

    // Try with wrong expected
    let wrong = 0x3000 as *const ();
    let result = entry.compare_exchange(initial, wrong);

    assert!(result.is_err());
    assert_eq!(entry.get(), new);
}

#[test]
fn test_dynamic_dispatch_table_new() {
    let default = 0x1000 as *const ();
    let table = DynamicDispatchTable::new(default);

    assert!(!table.is_active());

    for i in 0..MAX_OPCODES {
        assert_eq!(table.get(i as u8), default);
    }
}

#[test]
fn test_dynamic_dispatch_table_patch() {
    let default = 0x1000 as *const ();
    let table = DynamicDispatchTable::new(default);

    let new_handler = 0x2000 as *const ();
    let old = table.patch(0x10, new_handler);

    assert_eq!(old, default);
    assert_eq!(table.get(0x10), new_handler);

    let stats = table.stats();
    assert_eq!(stats.total_patches, 1);
    assert_eq!(stats.patched_entries, 1);
}

#[test]
fn test_dynamic_dispatch_table_patch_if() {
    let default = 0x1000 as *const ();
    let table = DynamicDispatchTable::new(default);

    let new = 0x2000 as *const ();
    let result = table.patch_if(0x20, default, new);

    assert!(result.is_ok());
    assert_eq!(table.get(0x20), new);

    // Try with wrong expected
    let wrong = 0x3000 as *const ();
    let result = table.patch_if(0x20, default, wrong);

    assert!(result.is_err());
}

#[test]
fn test_dynamic_dispatch_table_activate() {
    let table = DynamicDispatchTable::new(ptr::null());

    assert!(!table.is_active());
    table.activate();
    assert!(table.is_active());
    table.deactivate();
    assert!(!table.is_active());
}

#[test]
fn test_jit_entry_registry() {
    let mut registry = JitEntryRegistry::new();

    assert!(registry.is_empty());

    registry.register(1, 0x1000 as *const u8, 0x1100 as *const u8);
    registry.register(2, 0x2000 as *const u8, 0x2100 as *const u8);

    assert_eq!(registry.len(), 2);

    let entry = registry.get(1);
    assert!(entry.is_some());
    assert_eq!(entry.unwrap().code_id, 1);
}

#[test]
fn test_jit_entry_registry_invalidate() {
    let mut registry = JitEntryRegistry::new();

    registry.register(1, 0x1000 as *const u8, 0x1100 as *const u8);

    assert!(registry.get(1).is_some());

    registry.invalidate(1);

    assert!(registry.get(1).is_none()); // Inactive entries not returned
}
