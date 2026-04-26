use super::*;

#[test]
fn test_safepoint_page_creation() {
    let page = SafepointPage::new().expect("Failed to create safepoint page");
    assert_eq!(page.state(), SafepointState::Enabled);
    assert!(!page.is_armed());
    assert!(page.poll_address() != 0);
}

#[test]
fn test_safepoint_arm_disarm() {
    let page = SafepointPage::new().expect("Failed to create safepoint page");

    // Arm the page
    page.arm().expect("Failed to arm");
    assert!(page.is_armed());
    assert_eq!(page.state(), SafepointState::Armed);

    // Disarm the page
    page.disarm().expect("Failed to disarm");
    assert!(!page.is_armed());
    assert_eq!(page.state(), SafepointState::Enabled);
}

#[test]
fn test_safepoint_double_arm_fails() {
    let page = SafepointPage::new().expect("Failed to create safepoint page");

    page.arm().expect("First arm should succeed");

    // Second arm should fail
    let result = page.arm();
    assert!(result.is_err());

    page.disarm().expect("Disarm should succeed");
}

#[test]
fn test_safepoint_contains_address() {
    let page = SafepointPage::new().expect("Failed to create safepoint page");
    let addr = page.poll_address();

    assert!(page.contains_address(addr));
    assert!(page.contains_address(addr + 100));
    assert!(!page.contains_address(addr + SAFEPOINT_PAGE_SIZE));
    assert!(!page.contains_address(0));
}

#[test]
fn test_safepoint_thread_counting() {
    let page = SafepointPage::new().expect("Failed to create safepoint page");

    assert_eq!(page.stopped_count(), 0);

    let count1 = page.thread_stopped();
    assert_eq!(count1, 1);
    assert_eq!(page.stopped_count(), 1);

    let count2 = page.thread_stopped();
    assert_eq!(count2, 2);
    assert_eq!(page.stopped_count(), 2);
}

#[test]
fn test_safepoint_state_from_u32() {
    assert_eq!(SafepointState::from_u32(0), Some(SafepointState::Enabled));
    assert_eq!(SafepointState::from_u32(1), Some(SafepointState::Armed));
    assert_eq!(SafepointState::from_u32(2), Some(SafepointState::Triggered));
    assert_eq!(SafepointState::from_u32(99), None);
}

#[test]
fn test_safepoint_cache_line_aligned() {
    // Verify SafepointPage is cache-line aligned (64 bytes)
    assert_eq!(std::mem::align_of::<SafepointPage>(), 64);
}
