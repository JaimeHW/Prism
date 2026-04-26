use super::*;

#[test]
fn test_patchpoint_registry_creation() {
    let registry = PatchpointRegistry::new();
    let stats = registry.stats();
    assert_eq!(stats.total_patchpoints, 0);
}

#[test]
fn test_patchpoint_registration() {
    let registry = PatchpointRegistry::new();

    let code_id = 1;
    let bc_offset = 10;
    let nop_addr = 0x1000 as *mut u8;

    let id = registry.register(code_id, bc_offset, nop_addr);
    assert!(id > 0);

    let pp = registry.lookup(code_id, bc_offset);
    assert!(pp.is_some());
    assert_eq!(pp.unwrap().bc_offset, bc_offset);
}

#[test]
fn test_patchpoint_lookup_by_id() {
    let registry = PatchpointRegistry::new();

    let id = registry.register(1, 20, 0x2000 as *mut u8);
    let pp = registry.lookup_by_id(id);

    assert!(pp.is_some());
    assert_eq!(pp.unwrap().id, id);
}

#[test]
fn test_invalidate_code() {
    let registry = PatchpointRegistry::new();

    registry.register(1, 10, 0x1000 as *mut u8);
    registry.register(1, 20, 0x1100 as *mut u8);
    registry.register(2, 10, 0x2000 as *mut u8);

    let removed = registry.invalidate_code(1);
    assert_eq!(removed, 2);

    assert!(registry.lookup(1, 10).is_none());
    assert!(registry.lookup(1, 20).is_none());
    assert!(registry.lookup(2, 10).is_some());
}

#[test]
fn test_osr_entry_stub_builder() {
    let mut builder = OsrEntryStubBuilder::new();
    let target = 0x12345678 as *const u8;

    let stub = builder.build(target, 64, 8);

    assert!(!stub.code.is_empty());
    assert!(stub.size() > 0);
}

#[test]
fn test_osr_patch_error_display() {
    let err = OsrPatchError::PatchpointNotFound;
    assert_eq!(format!("{}", err), "Patchpoint not found");

    let err = OsrPatchError::InvalidNopSequence;
    assert_eq!(format!("{}", err), "Invalid NOP sequence at patchpoint");
}
