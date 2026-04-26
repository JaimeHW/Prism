use super::*;

#[test]
fn test_osr_decision_should_compile() {
    assert!(!OsrDecision::Cold.should_compile());
    assert!(!OsrDecision::Warming.should_compile());
    assert!(OsrDecision::Hot.should_compile());
    assert!(OsrDecision::VeryHot.should_compile());
    assert!(!OsrDecision::Pending.should_compile());
    assert!(!OsrDecision::Ready.should_compile());
}

#[test]
fn test_osr_decision_can_enter() {
    assert!(!OsrDecision::Cold.can_enter());
    assert!(!OsrDecision::Hot.can_enter());
    assert!(OsrDecision::Ready.can_enter());
}

#[test]
fn test_loop_info_cold_to_warming() {
    let mut info = LoopInfo::new(100, 120);
    assert_eq!(info.decision, OsrDecision::Cold);

    let decision = info.record_iteration();
    assert_eq!(decision, OsrDecision::Warming);
    assert_eq!(info.trip_count, 1);
}

#[test]
fn test_loop_info_warming_to_hot() {
    let mut info = LoopInfo::new(100, 120);

    // Warm up to threshold
    for _ in 0..OSR_LOOP_THRESHOLD {
        info.record_iteration();
    }

    assert_eq!(info.decision, OsrDecision::Hot);
}

#[test]
fn test_loop_info_hot_to_very_hot() {
    let mut info = LoopInfo::new(100, 120);

    // Warm up past very hot threshold
    for _ in 0..=OSR_HOT_LOOP_THRESHOLD {
        info.record_iteration();
    }

    assert_eq!(info.decision, OsrDecision::VeryHot);
}

#[test]
fn test_osr_trigger_pending_queue() {
    let mut trigger = OsrTrigger::new();
    let code_id = CodeId::new(12345);
    let mut loop_info = LoopInfo::new(100, 120);

    // Warm up to hot
    for _ in 0..OSR_LOOP_THRESHOLD {
        let decision = trigger.record_back_edge(code_id, 100, 120, &mut loop_info);
        if decision == OsrDecision::Pending {
            break;
        }
    }

    // Should be pending
    assert!(trigger.pending_count() > 0 || loop_info.decision == OsrDecision::Pending);
}

#[test]
fn test_osr_trigger_mark_ready() {
    let mut trigger = OsrTrigger::new();
    let code_id = CodeId::new(12345);

    // Add to pending
    let loop_info = LoopInfo::new(100, 120);
    trigger.add_pending(code_id, loop_info);
    assert_eq!(trigger.pending_count(), 1);

    // Mark ready
    trigger.mark_ready(code_id, 100);
    assert_eq!(trigger.pending_count(), 0);
    assert_eq!(trigger.ready_count(), 1);
    assert!(trigger.is_ready(code_id, 100));
}

#[test]
fn test_osr_trigger_pop_pending_prioritizes_very_hot() {
    let mut trigger = OsrTrigger::new();
    let code1 = CodeId::new(1);
    let code2 = CodeId::new(2);

    // Add hot loop first
    let hot_loop = LoopInfo {
        header_offset: 100,
        back_edge_offset: 120,
        trip_count: OSR_LOOP_THRESHOLD,
        decision: OsrDecision::Pending,
    };
    trigger.add_pending(code1, hot_loop);

    // Add very hot loop second
    let very_hot_loop = LoopInfo {
        header_offset: 200,
        back_edge_offset: 220,
        trip_count: OSR_HOT_LOOP_THRESHOLD + 1,
        decision: OsrDecision::VeryHot,
    };
    trigger.add_pending(code2, very_hot_loop);

    // Very hot should be popped first
    let (popped_code, popped_loop) = trigger.pop_pending().unwrap();
    assert_eq!(popped_code, code2);
    assert_eq!(popped_loop.trip_count, OSR_HOT_LOOP_THRESHOLD + 1);
}
