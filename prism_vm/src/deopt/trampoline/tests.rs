use super::*;

#[test]
fn test_trampoline_creation() {
    let handler = 0x12345678 as *const u8;
    let trampoline = DeoptTrampoline::new(handler);

    assert_eq!(trampoline.allocated_count(), 0);
    assert!(!trampoline.is_full());
    assert_eq!(trampoline.handler_address(), handler);
}

#[test]
fn test_trampoline_allocate() {
    let trampoline = DeoptTrampoline::new(std::ptr::null());

    let result = trampoline.allocate(100, 1, super::super::DeoptReason::TypeGuard);
    assert!(result.is_some());

    let (deopt_id, addr) = result.unwrap();
    assert_eq!(deopt_id, 0);
    assert!(!addr.is_null());

    assert_eq!(trampoline.allocated_count(), 1);
}

#[test]
fn test_trampoline_get_address() {
    let trampoline = DeoptTrampoline::new(std::ptr::null());

    let addr = trampoline.get_address(0);
    assert!(addr.is_some());

    let addr = trampoline.get_address(MAX_TRAMPOLINE_ENTRIES as u32);
    assert!(addr.is_none());
}

#[test]
fn test_trampoline_reset() {
    let trampoline = DeoptTrampoline::new(std::ptr::null());

    trampoline.allocate(100, 1, super::super::DeoptReason::TypeGuard);
    trampoline.allocate(200, 2, super::super::DeoptReason::Overflow);

    assert_eq!(trampoline.allocated_count(), 2);

    trampoline.reset();
    assert_eq!(trampoline.allocated_count(), 0);
}

#[test]
fn test_constants() {
    assert_eq!(TRAMPOLINE_ENTRY_SIZE, 32);
    assert_eq!(MAX_TRAMPOLINE_ENTRIES, 128);
    assert!(MAX_TRAMPOLINE_ENTRIES * TRAMPOLINE_ENTRY_SIZE <= TRAMPOLINE_PAGE_SIZE);
}
