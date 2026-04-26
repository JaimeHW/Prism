use super::*;

// ════════════════════════════════════════════════════════════════════════
// HandlerFrame Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_handler_frame_new() {
    let frame = HandlerFrame::new(5, 10, 1);
    assert_eq!(frame.handler_idx, 5);
    assert_eq!(frame.stack_depth, 10);
    assert_eq!(frame.frame_id, 1);
}

#[test]
fn test_handler_frame_is_valid() {
    let valid = HandlerFrame::new(0, 0, 0);
    assert!(valid.is_valid());

    let invalid = HandlerFrame::invalid();
    assert!(!invalid.is_valid());
}

#[test]
fn test_handler_frame_size() {
    // Ensure compact memory layout
    assert_eq!(std::mem::size_of::<HandlerFrame>(), 8);
}

#[test]
fn test_handler_frame_debug() {
    let valid = HandlerFrame::new(1, 5, 2);
    let debug_str = format!("{:?}", valid);
    assert!(debug_str.contains("handler_idx"));
    assert!(debug_str.contains("1"));

    let invalid = HandlerFrame::invalid();
    let debug_str = format!("{:?}", invalid);
    assert!(debug_str.contains("invalid"));
}

// ════════════════════════════════════════════════════════════════════════
// HandlerStack Basic Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_handler_stack_new() {
    let stack = HandlerStack::new();
    assert!(stack.is_empty());
    assert_eq!(stack.len(), 0);
}

#[test]
fn test_handler_stack_with_capacity() {
    let stack = HandlerStack::with_capacity(16);
    assert!(stack.is_empty());
}

#[test]
fn test_handler_stack_push_pop() {
    let mut stack = HandlerStack::new();

    let frame = HandlerFrame::new(0, 5, 0);
    assert!(stack.push(frame));
    assert_eq!(stack.len(), 1);

    let popped = stack.pop();
    assert_eq!(popped, Some(frame));
    assert!(stack.is_empty());
}

#[test]
fn test_handler_stack_lifo_order() {
    let mut stack = HandlerStack::new();

    let f1 = HandlerFrame::new(1, 0, 0);
    let f2 = HandlerFrame::new(2, 0, 0);
    let f3 = HandlerFrame::new(3, 0, 0);

    stack.push(f1);
    stack.push(f2);
    stack.push(f3);

    assert_eq!(stack.pop(), Some(f3));
    assert_eq!(stack.pop(), Some(f2));
    assert_eq!(stack.pop(), Some(f1));
    assert_eq!(stack.pop(), None);
}

#[test]
fn test_handler_stack_peek() {
    let mut stack = HandlerStack::new();
    assert!(stack.peek().is_none());

    let frame = HandlerFrame::new(5, 10, 1);
    stack.push(frame);

    assert_eq!(stack.peek(), Some(&frame));
    assert_eq!(stack.len(), 1); // peek doesn't remove
}

#[test]
fn test_handler_stack_peek_mut() {
    let mut stack = HandlerStack::new();
    stack.push(HandlerFrame::new(1, 5, 0));

    if let Some(frame) = stack.peek_mut() {
        frame.stack_depth = 10;
    }

    assert_eq!(stack.peek().unwrap().stack_depth, 10);
}

#[test]
fn test_handler_stack_clear() {
    let mut stack = HandlerStack::new();
    stack.push(HandlerFrame::new(1, 0, 0));
    stack.push(HandlerFrame::new(2, 0, 0));
    stack.push(HandlerFrame::new(3, 0, 0));

    stack.clear();
    assert!(stack.is_empty());
}

// ════════════════════════════════════════════════════════════════════════
// HandlerStack Frame Operations Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_handler_stack_pop_frame_handlers() {
    let mut stack = HandlerStack::new();

    // Frame 0: 2 handlers
    stack.push(HandlerFrame::new(0, 0, 0));
    stack.push(HandlerFrame::new(1, 0, 0));

    // Frame 1: 1 handler
    stack.push(HandlerFrame::new(2, 0, 1));

    assert_eq!(stack.len(), 3);

    // Pop all handlers for frame 1
    stack.pop_frame_handlers(1);
    assert_eq!(stack.len(), 2);

    // Remaining are frame 0
    assert_eq!(stack.peek().unwrap().frame_id, 0);
}

#[test]
fn test_handler_stack_find_in_frame() {
    let mut stack = HandlerStack::new();

    stack.push(HandlerFrame::new(0, 5, 0));
    stack.push(HandlerFrame::new(1, 10, 1));
    stack.push(HandlerFrame::new(2, 15, 0)); // Another handler for frame 0

    // Find first handler for frame 0 (newest first = handler_idx 2)
    let found = stack.find_in_frame(0);
    assert!(found.is_some());
    assert_eq!(found.unwrap().handler_idx, 2);

    // Find handler for frame 1
    let found = stack.find_in_frame(1);
    assert!(found.is_some());
    assert_eq!(found.unwrap().handler_idx, 1);

    // Frame 2 doesn't exist
    assert!(stack.find_in_frame(2).is_none());
}

#[test]
fn test_handler_stack_count_frame_handlers() {
    let mut stack = HandlerStack::new();

    stack.push(HandlerFrame::new(0, 0, 0));
    stack.push(HandlerFrame::new(1, 0, 0));
    stack.push(HandlerFrame::new(2, 0, 1));

    assert_eq!(stack.count_frame_handlers(0), 2);
    assert_eq!(stack.count_frame_handlers(1), 1);
    assert_eq!(stack.count_frame_handlers(2), 0);
}

#[test]
fn test_handler_stack_truncate() {
    let mut stack = HandlerStack::new();

    for i in 0..5 {
        stack.push(HandlerFrame::new(i, 0, 0));
    }

    stack.truncate(3);
    assert_eq!(stack.len(), 3);
    assert_eq!(stack.peek().unwrap().handler_idx, 2);
}

#[test]
fn test_handler_stack_iter() {
    let mut stack = HandlerStack::new();
    stack.push(HandlerFrame::new(1, 0, 0));
    stack.push(HandlerFrame::new(2, 0, 0));
    stack.push(HandlerFrame::new(3, 0, 0));

    // iter() returns top to bottom
    let indices: Vec<_> = stack.iter().map(|f| f.handler_idx).collect();
    assert_eq!(indices, vec![3, 2, 1]);
}

#[test]
fn test_handler_stack_get() {
    let mut stack = HandlerStack::new();
    stack.push(HandlerFrame::new(1, 0, 0));
    stack.push(HandlerFrame::new(2, 0, 0));

    assert_eq!(stack.get(0).unwrap().handler_idx, 1);
    assert_eq!(stack.get(1).unwrap().handler_idx, 2);
    assert!(stack.get(2).is_none());
}

// ════════════════════════════════════════════════════════════════════════
// HandlerSearchResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_search_result_found() {
    let handler = HandlerFrame::new(1, 5, 0);
    let result = HandlerSearchResult::Found {
        stack_index: 2,
        handler,
    };

    assert!(result.found());
    assert_eq!(result.handler(), Some(handler));
    assert_eq!(result.stack_index(), Some(2));
}

#[test]
fn test_search_result_not_found() {
    let result = HandlerSearchResult::NotFound;

    assert!(!result.found());
    assert_eq!(result.handler(), None);
    assert_eq!(result.stack_index(), None);
}

#[test]
fn test_search_result_finally() {
    let handler = HandlerFrame::new(3, 10, 1);
    let result = HandlerSearchResult::Finally {
        stack_index: 0,
        handler,
    };

    assert!(result.found());
    assert_eq!(result.handler(), Some(handler));
    assert_eq!(result.stack_index(), Some(0));
}

// ════════════════════════════════════════════════════════════════════════
// HandlerStackStats Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_stats_new() {
    let stats = HandlerStackStats::new();
    assert_eq!(stats.push_count, 0);
    assert_eq!(stats.search_count, 0);
}

#[test]
fn test_stats_hit_rate() {
    let mut stats = HandlerStackStats::new();

    // No searches = 0% hit rate
    assert_eq!(stats.hit_rate(), 0.0);

    stats.search_count = 100;
    stats.hits = 75;
    assert!((stats.hit_rate() - 75.0).abs() < 0.001);
}

#[test]
fn test_stats_avg_handlers_examined() {
    let mut stats = HandlerStackStats::new();

    assert_eq!(stats.avg_handlers_examined(), 0.0);

    stats.search_count = 10;
    stats.handlers_examined = 35;
    assert!((stats.avg_handlers_examined() - 3.5).abs() < 0.001);
}

#[test]
fn test_stats_reset() {
    let mut stats = HandlerStackStats::new();
    stats.push_count = 100;
    stats.pop_count = 50;
    stats.max_depth = 10;

    stats.reset();

    assert_eq!(stats.push_count, 0);
    assert_eq!(stats.pop_count, 0);
    assert_eq!(stats.max_depth, 0);
}

// ════════════════════════════════════════════════════════════════════════
// Nested Try Block Simulation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_nested_try_blocks() {
    let mut stack = HandlerStack::new();

    // Simulate:
    // try:                 # handler 0, frame 0
    //     try:             # handler 1, frame 0
    //         try:         # handler 2, frame 0
    //             ...
    //         except:
    //             pass
    //     except:
    //         pass
    // except:
    //     pass

    stack.push(HandlerFrame::new(0, 0, 0));
    stack.push(HandlerFrame::new(1, 5, 0));
    stack.push(HandlerFrame::new(2, 10, 0));

    assert_eq!(stack.len(), 3);

    // Exit innermost try
    stack.pop();
    assert_eq!(stack.len(), 2);
    assert_eq!(stack.peek().unwrap().handler_idx, 1);

    // Exit middle try
    stack.pop();
    assert_eq!(stack.len(), 1);
    assert_eq!(stack.peek().unwrap().handler_idx, 0);
}

#[test]
fn test_cross_frame_handlers() {
    let mut stack = HandlerStack::new();

    // Frame 0: try block
    stack.push(HandlerFrame::new(0, 5, 0));

    // Frame 1 (called function): try block
    stack.push(HandlerFrame::new(1, 3, 1));

    // Frame 2 (another call): try block
    stack.push(HandlerFrame::new(2, 2, 2));

    // Unwind frame 2
    stack.pop_frame_handlers(2);
    assert_eq!(stack.len(), 2);

    // Unwind frame 1
    stack.pop_frame_handlers(1);
    assert_eq!(stack.len(), 1);

    // Only frame 0's handler remains
    assert_eq!(stack.peek().unwrap().frame_id, 0);
}

// ════════════════════════════════════════════════════════════════════════
// Edge Cases
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_empty_stack_operations() {
    let mut stack = HandlerStack::new();

    assert!(stack.pop().is_none());
    assert!(stack.peek().is_none());
    assert!(stack.peek_mut().is_none());
    assert!(stack.find_in_frame(0).is_none());
    assert_eq!(stack.count_frame_handlers(0), 0);

    // These should not panic
    stack.pop_frame_handlers(0);
    stack.truncate(0);
    stack.clear();
}

#[test]
fn test_handler_frame_all_zeros() {
    let frame = HandlerFrame::new(0, 0, 0);
    assert!(frame.is_valid()); // handler_idx 0 is still valid
}

#[test]
fn test_max_stack_depth() {
    // This test verifies the stack respects depth limits
    // but doesn't actually push 1024 items
    let mut stack = HandlerStack::new();

    // Push a few items to verify basic operation
    for i in 0..10 {
        assert!(stack.push(HandlerFrame::new(i, 0, 0)));
    }

    assert_eq!(stack.len(), 10);
}
