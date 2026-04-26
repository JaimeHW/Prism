use super::*;

// ════════════════════════════════════════════════════════════════════════
// UnwindAction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_unwind_action_jump_to_handler() {
    let action = UnwindAction::JumpToHandler {
        handler_pc: 100,
        stack_depth: 5,
    };

    assert!(action.continues());
    assert!(!action.propagates());
    assert_eq!(action.handler_pc(), Some(100));
    assert_eq!(action.stack_depth(), Some(5));
}

#[test]
fn test_unwind_action_execute_finally() {
    let action = UnwindAction::ExecuteFinally {
        finally_pc: 200,
        stack_depth: 3,
        reraise: true,
    };

    assert!(!action.continues());
    assert!(!action.propagates());
    assert_eq!(action.handler_pc(), Some(200));
    assert_eq!(action.stack_depth(), Some(3));
}

#[test]
fn test_unwind_action_propagate() {
    let action = UnwindAction::PropagateToFrame { target_frame_id: 1 };

    assert!(!action.continues());
    assert!(action.propagates());
    assert_eq!(action.handler_pc(), None);
    assert_eq!(action.stack_depth(), None);
}

#[test]
fn test_unwind_action_unhandled() {
    let action = UnwindAction::Unhandled;

    assert!(!action.continues());
    assert!(action.propagates());
}

#[test]
fn test_unwind_action_continue() {
    let action = UnwindAction::Continue;

    assert!(action.continues());
    assert!(!action.propagates());
}

// ════════════════════════════════════════════════════════════════════════
// UnwindResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_unwind_result_handler_found() {
    let result = UnwindResult::handler_found(100, 5, 3);

    assert!(result.found_handler());
    assert_eq!(result.frames_unwound, 0);
    assert_eq!(result.handlers_examined, 3);
    assert!(result.finally_queue.is_empty());
}

#[test]
fn test_unwind_result_unhandled() {
    let result = UnwindResult::unhandled(5, 10);

    assert!(!result.found_handler());
    assert_eq!(result.frames_unwound, 5);
    assert_eq!(result.handlers_examined, 10);
}

#[test]
fn test_unwind_result_propagate() {
    let result = UnwindResult::propagate(42, 3);

    assert!(!result.found_handler());
    assert!(result.action.propagates());
    if let UnwindAction::PropagateToFrame { target_frame_id } = result.action {
        assert_eq!(target_frame_id, 42);
    } else {
        panic!("Expected PropagateToFrame");
    }
}

// ════════════════════════════════════════════════════════════════════════
// FinallyEntry Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_finally_entry_new() {
    let entry = FinallyEntry::new(100, 5, 1, true);

    assert_eq!(entry.finally_pc, 100);
    assert_eq!(entry.stack_depth, 5);
    assert_eq!(entry.frame_id, 1);
    assert!(entry.reraise);
}

#[test]
fn test_finally_entry_size() {
    // Should be compact
    assert!(std::mem::size_of::<FinallyEntry>() <= 16);
}

// ════════════════════════════════════════════════════════════════════════
// UnwindInfo Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_unwind_info_new() {
    let info = UnwindInfo::new(50, 1, 10, 42);

    assert_eq!(info.pc, 50);
    assert_eq!(info.frame_id, 1);
    assert_eq!(info.stack_depth, 10);
    assert_eq!(info.exception_type_id, 42);
}

// ════════════════════════════════════════════════════════════════════════
// UnwinderStats Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_unwinder_stats_new() {
    let stats = UnwinderStats::new();

    assert_eq!(stats.unwind_count, 0);
    assert_eq!(stats.handlers_found, 0);
}

#[test]
fn test_unwinder_stats_avg_frames() {
    let mut stats = UnwinderStats::new();

    assert_eq!(stats.avg_frames_unwound(), 0.0);

    stats.unwind_count = 10;
    stats.frames_unwound = 25;
    assert!((stats.avg_frames_unwound() - 2.5).abs() < 0.001);
}

#[test]
fn test_unwinder_stats_handler_rate() {
    let mut stats = UnwinderStats::new();

    assert_eq!(stats.handler_found_rate(), 0.0);

    stats.unwind_count = 100;
    stats.handlers_found = 75;
    assert!((stats.handler_found_rate() - 75.0).abs() < 0.001);
}

#[test]
fn test_unwinder_stats_reset() {
    let mut stats = UnwinderStats::new();
    stats.unwind_count = 100;
    stats.handlers_found = 50;

    stats.reset();

    assert_eq!(stats.unwind_count, 0);
    assert_eq!(stats.handlers_found, 0);
}

// ════════════════════════════════════════════════════════════════════════
// Unwinder Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_unwinder_new() {
    let unwinder = Unwinder::new();

    assert_eq!(unwinder.stats().unwind_count, 0);
}

#[test]
fn test_unwinder_search_empty_stack() {
    let mut unwinder = Unwinder::new();
    let stack = HandlerStack::new();
    let info = UnwindInfo::new(50, 0, 10, 0);

    let result = unwinder.search_handler(&stack, &info);

    assert_eq!(result, HandlerSearchResult::NotFound);
    assert_eq!(unwinder.stats().unwind_count, 1);
    assert_eq!(unwinder.stats().unhandled, 1);
}

#[test]
fn test_unwinder_search_finds_handler() {
    let mut unwinder = Unwinder::new();
    let mut stack = HandlerStack::new();
    stack.push(HandlerFrame::new(0, 5, 0));

    let info = UnwindInfo::new(50, 0, 10, 0);
    let result = unwinder.search_handler(&stack, &info);

    if let HandlerSearchResult::Found { handler, .. } = result {
        assert_eq!(handler.handler_idx, 0);
        assert_eq!(handler.stack_depth, 5);
    } else {
        panic!("Expected handler found");
    }
}

#[test]
fn test_unwinder_search_finds_parent_frame_handler() {
    let mut unwinder = Unwinder::new();
    let mut stack = HandlerStack::new();
    // Handler in parent frame
    stack.push(HandlerFrame::new(0, 5, 0));

    // Searching in child frame
    let info = UnwindInfo::new(50, 1, 10, 0);
    let result = unwinder.search_handler(&stack, &info);

    if let HandlerSearchResult::Found { handler, .. } = result {
        assert_eq!(handler.frame_id, 0); // Found parent's handler
    } else {
        panic!("Expected handler found");
    }
}

#[test]
fn test_unwinder_unwind_finds_handler() {
    let mut unwinder = Unwinder::new();
    let mut stack = HandlerStack::new();
    stack.push(HandlerFrame::new(5, 10, 0));

    let info = UnwindInfo::new(50, 0, 15, 0);
    let result = unwinder.unwind(&stack, &info);

    assert!(result.found_handler());
    assert_eq!(unwinder.stats().handlers_found, 1);
}

#[test]
fn test_unwinder_unwind_no_handler() {
    let mut unwinder = Unwinder::new();
    let stack = HandlerStack::new();

    let info = UnwindInfo::new(50, 0, 10, 0);
    let result = unwinder.unwind(&stack, &info);

    assert!(!result.found_handler());
    assert_eq!(result.action, UnwindAction::Unhandled);
}

#[test]
fn test_unwinder_record_unwind() {
    let mut unwinder = Unwinder::new();

    unwinder.record_unwind(5);
    assert_eq!(unwinder.stats().frames_unwound, 5);
    assert_eq!(unwinder.stats().max_frames_unwound, 5);

    unwinder.record_unwind(3);
    assert_eq!(unwinder.stats().frames_unwound, 8);
    assert_eq!(unwinder.stats().max_frames_unwound, 5);

    unwinder.record_unwind(10);
    assert_eq!(unwinder.stats().max_frames_unwound, 10);
}

#[test]
fn test_unwinder_reset_stats() {
    let mut unwinder = Unwinder::new();
    unwinder.record_unwind(5);

    unwinder.reset_stats();

    assert_eq!(unwinder.stats().frames_unwound, 0);
    assert_eq!(unwinder.stats().max_frames_unwound, 0);
}

// ════════════════════════════════════════════════════════════════════════
// Integration Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_nested_handlers_unwind() {
    let mut unwinder = Unwinder::new();
    let mut stack = HandlerStack::new();

    // Outermost handler
    stack.push(HandlerFrame::new(0, 2, 0));
    // Middle handler
    stack.push(HandlerFrame::new(1, 5, 0));
    // Innermost handler
    stack.push(HandlerFrame::new(2, 8, 0));

    let info = UnwindInfo::new(50, 0, 10, 0);
    let result = unwinder.unwind(&stack, &info);

    // Should find innermost handler first
    if let UnwindAction::JumpToHandler {
        handler_pc,
        stack_depth,
    } = result.action
    {
        assert_eq!(handler_pc, 2); // handler_idx 2
        assert_eq!(stack_depth, 8);
    } else {
        panic!("Expected JumpToHandler");
    }
}

#[test]
fn test_cross_frame_unwind() {
    let mut unwinder = Unwinder::new();
    let mut stack = HandlerStack::new();

    // Handler in frame 0
    stack.push(HandlerFrame::new(0, 5, 0));
    // Handler in frame 1
    stack.push(HandlerFrame::new(1, 3, 1));

    // Exception in frame 2 (no handler)
    let info = UnwindInfo::new(50, 2, 10, 0);
    let result = unwinder.unwind(&stack, &info);

    // Should find handler in frame 1
    assert!(result.found_handler());
}
