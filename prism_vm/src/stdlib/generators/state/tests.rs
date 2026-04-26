use super::*;

// ════════════════════════════════════════════════════════════════════════
// GeneratorState Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_state_size() {
    assert_eq!(std::mem::size_of::<GeneratorState>(), 1);
}

#[test]
fn test_state_values() {
    assert_eq!(GeneratorState::Created as u8, 0);
    assert_eq!(GeneratorState::Running as u8, 1);
    assert_eq!(GeneratorState::Suspended as u8, 2);
    assert_eq!(GeneratorState::Exhausted as u8, 3);
}

#[test]
fn test_state_from_bits() {
    assert_eq!(GeneratorState::from_bits(0), GeneratorState::Created);
    assert_eq!(GeneratorState::from_bits(1), GeneratorState::Running);
    assert_eq!(GeneratorState::from_bits(2), GeneratorState::Suspended);
    assert_eq!(GeneratorState::from_bits(3), GeneratorState::Exhausted);
}

#[test]
fn test_state_from_bits_masks() {
    // Higher bits should be masked off
    assert_eq!(GeneratorState::from_bits(0b100), GeneratorState::Created);
    assert_eq!(GeneratorState::from_bits(0b101), GeneratorState::Running);
    assert_eq!(
        GeneratorState::from_bits(0xFFFF_FF02),
        GeneratorState::Suspended
    );
}

#[test]
fn test_state_is_resumable() {
    assert!(GeneratorState::Created.is_resumable());
    assert!(!GeneratorState::Running.is_resumable());
    assert!(GeneratorState::Suspended.is_resumable());
    assert!(!GeneratorState::Exhausted.is_resumable());
}

#[test]
fn test_state_is_finished() {
    assert!(!GeneratorState::Created.is_finished());
    assert!(!GeneratorState::Running.is_finished());
    assert!(!GeneratorState::Suspended.is_finished());
    assert!(GeneratorState::Exhausted.is_finished());
}

#[test]
fn test_state_can_yield() {
    assert!(!GeneratorState::Created.can_yield());
    assert!(GeneratorState::Running.can_yield());
    assert!(!GeneratorState::Suspended.can_yield());
    assert!(!GeneratorState::Exhausted.can_yield());
}

#[test]
fn test_state_names() {
    assert_eq!(GeneratorState::Created.name(), "GEN_CREATED");
    assert_eq!(GeneratorState::Running.name(), "GEN_RUNNING");
    assert_eq!(GeneratorState::Suspended.name(), "GEN_SUSPENDED");
    assert_eq!(GeneratorState::Exhausted.name(), "GEN_CLOSED");
}

#[test]
fn test_state_display() {
    assert_eq!(format!("{}", GeneratorState::Created), "GEN_CREATED");
    assert_eq!(format!("{}", GeneratorState::Exhausted), "GEN_CLOSED");
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorHeader Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_header_size() {
    assert_eq!(std::mem::size_of::<GeneratorHeader>(), 4);
}

#[test]
fn test_header_new() {
    let header = GeneratorHeader::new();
    assert_eq!(header.state(), GeneratorState::Created);
    assert_eq!(header.resume_index(), 0);
}

#[test]
fn test_header_with_state_and_index() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Suspended, 42);
    assert_eq!(header.state(), GeneratorState::Suspended);
    assert_eq!(header.resume_index(), 42);
}

#[test]
fn test_header_state_and_index() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Running, 100);
    let (state, index) = header.state_and_index();
    assert_eq!(state, GeneratorState::Running);
    assert_eq!(index, 100);
}

#[test]
fn test_header_set_state() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Created, 50);
    header.set_state(GeneratorState::Running);
    assert_eq!(header.state(), GeneratorState::Running);
    assert_eq!(header.resume_index(), 50); // Index preserved
}

#[test]
fn test_header_set_state_and_index() {
    let header = GeneratorHeader::new();
    header.set_state_and_index(GeneratorState::Suspended, 999);
    assert_eq!(header.state(), GeneratorState::Suspended);
    assert_eq!(header.resume_index(), 999);
}

#[test]
fn test_header_try_start_created() {
    let header = GeneratorHeader::new();
    let prev = header.try_start();
    assert_eq!(prev, Some(GeneratorState::Created));
    assert_eq!(header.state(), GeneratorState::Running);
}

#[test]
fn test_header_try_start_suspended() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Suspended, 5);
    let prev = header.try_start();
    assert_eq!(prev, Some(GeneratorState::Suspended));
    assert_eq!(header.state(), GeneratorState::Running);
    assert_eq!(header.resume_index(), 5); // Preserved
}

#[test]
fn test_header_try_start_running_fails() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Running, 0);
    let prev = header.try_start();
    assert_eq!(prev, None);
    assert_eq!(header.state(), GeneratorState::Running);
}

#[test]
fn test_header_try_start_exhausted_fails() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Exhausted, 0);
    let prev = header.try_start();
    assert_eq!(prev, None);
    assert_eq!(header.state(), GeneratorState::Exhausted);
}

#[test]
fn test_header_suspend() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Running, 0);
    header.suspend(10);
    assert_eq!(header.state(), GeneratorState::Suspended);
    assert_eq!(header.resume_index(), 10);
}

#[test]
fn test_header_exhaust() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Running, 25);
    header.exhaust();
    assert_eq!(header.state(), GeneratorState::Exhausted);
    assert_eq!(header.resume_index(), 25); // Preserved for debugging
}

#[test]
fn test_header_is_running() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Running, 0);
    assert!(header.is_running());

    let header2 = GeneratorHeader::new();
    assert!(!header2.is_running());
}

#[test]
fn test_header_is_resumable() {
    assert!(GeneratorHeader::new().is_resumable());
    assert!(GeneratorHeader::with_state_and_index(GeneratorState::Suspended, 0).is_resumable());
    assert!(!GeneratorHeader::with_state_and_index(GeneratorState::Running, 0).is_resumable());
    assert!(!GeneratorHeader::with_state_and_index(GeneratorState::Exhausted, 0).is_resumable());
}

#[test]
fn test_header_is_exhausted() {
    assert!(!GeneratorHeader::new().is_exhausted());
    assert!(GeneratorHeader::with_state_and_index(GeneratorState::Exhausted, 0).is_exhausted());
}

#[test]
fn test_header_max_resume_index() {
    let max_index = GeneratorHeader::MAX_RESUME_INDEX;
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Suspended, max_index);
    assert_eq!(header.resume_index(), max_index);
    assert_eq!(header.state(), GeneratorState::Suspended);
}

#[test]
fn test_header_clone() {
    let header1 = GeneratorHeader::with_state_and_index(GeneratorState::Suspended, 42);
    let header2 = header1.clone();
    assert_eq!(header2.state(), GeneratorState::Suspended);
    assert_eq!(header2.resume_index(), 42);
}

#[test]
fn test_header_debug() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Running, 100);
    let debug = format!("{:?}", header);
    assert!(debug.contains("Running"));
    assert!(debug.contains("100"));
}

#[test]
fn test_header_raw() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Suspended, 1);
    let raw = header.raw();
    // State = 2 (bits 0-1), resume = 1 (bits 2+)
    // Expected: (1 << 2) | 2 = 4 | 2 = 6
    assert_eq!(raw, 6);
}

// ════════════════════════════════════════════════════════════════════════
// State Machine Transition Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_lifecycle() {
    let header = GeneratorHeader::new();

    // Created -> Running (first next())
    assert!(header.try_start().is_some());
    assert_eq!(header.state(), GeneratorState::Running);

    // Running -> Suspended (yield)
    header.suspend(1);
    assert_eq!(header.state(), GeneratorState::Suspended);
    assert_eq!(header.resume_index(), 1);

    // Suspended -> Running (next())
    assert!(header.try_start().is_some());
    assert_eq!(header.state(), GeneratorState::Running);

    // Running -> Suspended (another yield)
    header.suspend(2);
    assert_eq!(header.state(), GeneratorState::Suspended);
    assert_eq!(header.resume_index(), 2);

    // Suspended -> Running -> Exhausted (return)
    assert!(header.try_start().is_some());
    header.exhaust();
    assert_eq!(header.state(), GeneratorState::Exhausted);

    // Exhausted cannot restart
    assert!(header.try_start().is_none());
}

#[test]
fn test_reentry_detection() {
    let header = GeneratorHeader::new();

    // Start running
    header.try_start();
    assert!(header.is_running());

    // Cannot start again while running
    assert!(header.try_start().is_none());
    assert!(header.is_running());
}

#[test]
fn test_close_from_created() {
    let header = GeneratorHeader::new();
    header.exhaust();
    assert!(header.is_exhausted());
    assert!(header.try_start().is_none());
}

#[test]
fn test_close_from_suspended() {
    let header = GeneratorHeader::with_state_and_index(GeneratorState::Suspended, 5);
    header.set_state(GeneratorState::Exhausted);
    assert!(header.is_exhausted());
    assert_eq!(header.resume_index(), 5); // Preserved
}
