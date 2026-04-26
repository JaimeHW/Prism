use super::*;

#[test]
fn test_trap_context_creation() {
    let ctx = TrapContext::new(0x1000, 0x2000, 0x3000, 0x4000);
    assert_eq!(ctx.fault_addr, 0x1000);
    assert_eq!(ctx.return_addr, 0x2000);
    assert_eq!(ctx.rsp, 0x3000);
    assert_eq!(ctx.rbp, 0x4000);
    assert!(ctx.thread_id != 0);
}

#[test]
fn test_is_safepoint_address_no_page() {
    // With no page set, should return false
    let old = SAFEPOINT_PAGE_ADDR.swap(0, Ordering::SeqCst);
    assert!(!is_safepoint_address(0x12345));
    SAFEPOINT_PAGE_ADDR.store(old, Ordering::SeqCst);
}

#[test]
fn test_is_safepoint_address_with_page() {
    let old = SAFEPOINT_PAGE_ADDR.swap(0x10000, Ordering::SeqCst);

    assert!(is_safepoint_address(0x10000));
    assert!(is_safepoint_address(0x10FFF));
    assert!(!is_safepoint_address(0x11000)); // Past page
    assert!(!is_safepoint_address(0x00000)); // Before page

    SAFEPOINT_PAGE_ADDR.store(old, Ordering::SeqCst);
}

#[test]
fn test_handler_error_display() {
    assert_eq!(
        HandlerError::AlreadyInstalled.to_string(),
        "handler already installed"
    );
    assert_eq!(
        HandlerError::NotInstalled.to_string(),
        "handler not installed"
    );
}
