use super::*;

#[test]
fn test_mutator_state() {
    assert!(MutatorState::AtSafepoint.is_safe_for_gc());
    assert!(MutatorState::Blocked.is_safe_for_gc());
    assert!(MutatorState::Native.is_safe_for_gc());
    assert!(!MutatorState::Running.is_safe_for_gc());
}

#[test]
fn test_mutator_thread() {
    let thread = MutatorThread::new(42);
    assert_eq!(thread.id, 42);
    assert_eq!(thread.get_state(), MutatorState::Running);

    thread.set_state(MutatorState::AtSafepoint);
    assert_eq!(thread.get_state(), MutatorState::AtSafepoint);
}

#[test]
fn test_coordinator_creation() {
    let page = Arc::new(SafepointPage::new().unwrap());
    let coord = SafepointCoordinator::new(page);
    assert_eq!(coord.thread_count(), 0);
    assert!(!coord.is_stopped());
}

#[test]
fn test_coordinator_register_unregister() {
    let page = Arc::new(SafepointPage::new().unwrap());
    let coord = SafepointCoordinator::new(page);

    let thread = coord.register_thread(100);
    assert_eq!(coord.thread_count(), 1);
    assert_eq!(thread.id, 100);

    coord.unregister_thread(100);
    assert_eq!(coord.thread_count(), 0);
}

#[test]
fn test_stop_the_world_no_threads() {
    let page = Arc::new(SafepointPage::new().unwrap());
    let coord = SafepointCoordinator::new(page);

    // With no threads, stop_the_world should succeed immediately
    let guard = coord.stop_the_world().unwrap();
    assert!(coord.is_stopped());
    drop(guard);
    assert!(!coord.is_stopped());
}

#[test]
fn test_try_stop_timeout() {
    let page = Arc::new(SafepointPage::new().unwrap());
    let coord = SafepointCoordinator::new(page);

    // Register a thread but don't stop it - should timeout
    let _thread = coord.register_thread(1);

    let result = coord.try_stop_the_world(Duration::from_millis(10)).unwrap();
    assert!(result.is_none()); // Should timeout
    assert!(!coord.is_stopped()); // Page should be disarmed
}

#[test]
fn test_safepoint_guard_drop() {
    let page = Arc::new(SafepointPage::new().unwrap());
    let coord = SafepointCoordinator::new(page);

    {
        let _guard = coord.stop_the_world().unwrap();
        assert!(coord.is_stopped());
    }
    // Guard dropped, should resume
    assert!(!coord.is_stopped());
}
