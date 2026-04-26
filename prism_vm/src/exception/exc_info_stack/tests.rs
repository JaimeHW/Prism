use super::*;

// =========================================================================
// EntryFlags Tests
// =========================================================================

#[test]
fn test_entry_flags_empty() {
    let flags = EntryFlags::EMPTY;
    assert_eq!(flags.as_raw(), 0);
    assert!(!flags.is_explicit());
    assert!(!flags.has_cause());
    assert!(!flags.is_context_suppressed());
}

#[test]
fn test_entry_flags_set_and_check() {
    let mut flags = EntryFlags::EMPTY;

    flags.set(EntryFlags::EXPLICIT);
    assert!(flags.is_explicit());
    assert!(!flags.has_cause());

    flags.set(EntryFlags::HAS_CAUSE);
    assert!(flags.is_explicit());
    assert!(flags.has_cause());
}

#[test]
fn test_entry_flags_clear() {
    let mut flags = EntryFlags::from_raw(0xFF);

    flags.clear(EntryFlags::EXPLICIT);
    assert!(!flags.is_explicit());
    assert!(flags.has_cause()); // Other flags still set
}

#[test]
fn test_entry_flags_from_raw() {
    let flags = EntryFlags::from_raw(EntryFlags::EXPLICIT | EntryFlags::HAS_CAUSE);
    assert!(flags.is_explicit());
    assert!(flags.has_cause());
    assert!(!flags.is_context_suppressed());
}

#[test]
fn test_entry_flags_debug() {
    let mut flags = EntryFlags::EMPTY;
    flags.set(EntryFlags::EXPLICIT);
    flags.set(EntryFlags::HAS_CAUSE);

    let debug = format!("{:?}", flags);
    assert!(debug.contains("EXPLICIT"));
    assert!(debug.contains("HAS_CAUSE"));
}

// =========================================================================
// ExcInfoEntry Tests
// =========================================================================

#[test]
fn test_exc_info_entry_new() {
    let entry = ExcInfoEntry::new(24, None); // 24 = TypeError
    assert_eq!(entry.type_id(), 24);
    assert!(entry.value().is_none());
    assert_eq!(entry.traceback_id(), 0);
    assert_eq!(entry.frame_id(), 0);
    assert_eq!(entry.pc(), 0);
}

#[test]
fn test_exc_info_entry_with_value() {
    let value = Value::int(42).unwrap();
    let entry = ExcInfoEntry::new(5, Some(value.clone())); // 5 = StopIteration

    assert_eq!(entry.type_id(), 5);
    assert!(entry.value().is_some());
    assert_eq!(entry.value_cloned().unwrap().as_int(), Some(42));
}

#[test]
fn test_exc_info_entry_with_context() {
    let value = Value::none();
    let entry = ExcInfoEntry::with_context(4, Some(value), 100, 1, 50);

    assert_eq!(entry.type_id(), 4);
    assert_eq!(entry.traceback_id(), 100);
    assert_eq!(entry.frame_id(), 1);
    assert_eq!(entry.pc(), 50);
}

#[test]
fn test_exc_info_entry_empty() {
    let entry = ExcInfoEntry::empty();
    assert_eq!(entry.type_id(), 0);
    assert!(entry.value().is_none());
    assert!(!entry.is_active());
}

#[test]
fn test_exc_info_entry_is_active() {
    let empty = ExcInfoEntry::empty();
    assert!(!empty.is_active());

    let with_type = ExcInfoEntry::new(24, None);
    assert!(with_type.is_active());

    let with_value = ExcInfoEntry::new(0, Some(Value::none()));
    assert!(with_value.is_active());
}

#[test]
fn test_exc_info_entry_set_value() {
    let mut entry = ExcInfoEntry::new(24, None);
    assert!(entry.value().is_none());

    entry.set_value(Some(Value::int(100).unwrap()));
    assert!(entry.value().is_some());
}

#[test]
fn test_exc_info_entry_set_traceback_id() {
    let mut entry = ExcInfoEntry::new(24, None);
    assert_eq!(entry.traceback_id(), 0);

    entry.set_traceback_id(500);
    assert_eq!(entry.traceback_id(), 500);
}

#[test]
fn test_exc_info_entry_set_from_raise_from() {
    let mut entry = ExcInfoEntry::new(24, None);
    entry.set_from_raise_from();

    assert!(entry.flags().has(EntryFlags::FROM_RAISE_FROM));
    assert!(entry.flags().has_cause());
    assert!(entry.flags().is_context_suppressed());
}

#[test]
fn test_exc_info_entry_flags_mut() {
    let mut entry = ExcInfoEntry::new(24, None);
    entry.flags_mut().set(EntryFlags::HANDLING);

    assert!(entry.flags().has(EntryFlags::HANDLING));
}

#[test]
fn test_exc_info_entry_debug() {
    let entry = ExcInfoEntry::new(24, Some(Value::none()));
    let debug = format!("{:?}", entry);

    assert!(debug.contains("ExcInfoEntry"));
    assert!(debug.contains("type_id: 24"));
    assert!(debug.contains("has_value: true"));
}

// =========================================================================
// ExcInfoStack Basic Tests
// =========================================================================

#[test]
fn test_exc_info_stack_new() {
    let stack = ExcInfoStack::new();
    assert!(stack.is_empty());
    assert_eq!(stack.len(), 0);
}

#[test]
fn test_exc_info_stack_with_capacity() {
    let stack = ExcInfoStack::with_capacity(16);
    assert!(stack.is_empty());
}

#[test]
fn test_exc_info_stack_push_pop() {
    let mut stack = ExcInfoStack::new();

    let entry = ExcInfoEntry::new(24, None);
    assert!(stack.push(entry));
    assert_eq!(stack.len(), 1);

    let popped = stack.pop();
    assert!(popped.is_some());
    assert_eq!(popped.unwrap().type_id(), 24);
    assert!(stack.is_empty());
}

#[test]
fn test_exc_info_stack_peek() {
    let mut stack = ExcInfoStack::new();
    assert!(stack.peek().is_none());

    stack.push(ExcInfoEntry::new(24, None));
    stack.push(ExcInfoEntry::new(5, None));

    let top = stack.peek();
    assert!(top.is_some());
    assert_eq!(top.unwrap().type_id(), 5);
    assert_eq!(stack.len(), 2); // Peek doesn't remove
}

#[test]
fn test_exc_info_stack_peek_mut() {
    let mut stack = ExcInfoStack::new();
    stack.push(ExcInfoEntry::new(24, None));

    if let Some(entry) = stack.peek_mut() {
        entry.set_traceback_id(999);
    }

    assert_eq!(stack.peek().unwrap().traceback_id(), 999);
}

#[test]
fn test_exc_info_stack_lifo_order() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));
    stack.push(ExcInfoEntry::new(3, None));

    assert_eq!(stack.pop().unwrap().type_id(), 3);
    assert_eq!(stack.pop().unwrap().type_id(), 2);
    assert_eq!(stack.pop().unwrap().type_id(), 1);
    assert!(stack.pop().is_none());
}

#[test]
fn test_exc_info_stack_get() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));
    stack.push(ExcInfoEntry::new(3, None));

    assert_eq!(stack.get(0).unwrap().type_id(), 1); // Bottom
    assert_eq!(stack.get(2).unwrap().type_id(), 3); // Top
    assert!(stack.get(10).is_none());
}

#[test]
fn test_exc_info_stack_clear() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));
    assert_eq!(stack.len(), 2);

    stack.clear();
    assert!(stack.is_empty());
}

#[test]
fn test_exc_info_stack_truncate() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));
    stack.push(ExcInfoEntry::new(3, None));
    stack.push(ExcInfoEntry::new(4, None));

    stack.truncate(2);
    assert_eq!(stack.len(), 2);
    assert_eq!(stack.peek().unwrap().type_id(), 2);
}

// =========================================================================
// ExcInfoStack Iterator Tests
// =========================================================================

#[test]
fn test_exc_info_stack_iter_top_to_bottom() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));
    stack.push(ExcInfoEntry::new(3, None));

    let type_ids: Vec<u16> = stack.iter().map(|e| e.type_id()).collect();
    assert_eq!(type_ids, vec![3, 2, 1]); // Top to bottom
}

#[test]
fn test_exc_info_stack_iter_bottom_to_top() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));
    stack.push(ExcInfoEntry::new(3, None));

    let type_ids: Vec<u16> = stack.iter_bottom_up().map(|e| e.type_id()).collect();
    assert_eq!(type_ids, vec![1, 2, 3]); // Bottom to top
}

// =========================================================================
// ExcInfoStack Overflow Tests
// =========================================================================

#[test]
fn test_exc_info_stack_max_depth() {
    assert_eq!(ExcInfoStack::max_depth(), 255);
}

#[test]
fn test_exc_info_stack_overflow_protection() {
    let mut stack = ExcInfoStack::new();

    // Fill to max
    for i in 0..MAX_DEPTH {
        assert!(stack.push(ExcInfoEntry::new(i as u16, None)));
    }

    assert_eq!(stack.len(), MAX_DEPTH);

    // Next push should fail
    assert!(!stack.push(ExcInfoEntry::new(999, None)));
    assert_eq!(stack.len(), MAX_DEPTH);
    assert_eq!(stack.stats().overflows, 1);
}

// =========================================================================
// ExcInfoStack Statistics Tests
// =========================================================================

#[test]
fn test_exc_info_stack_stats_pushes() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));
    stack.push(ExcInfoEntry::new(3, None));

    assert_eq!(stack.stats().pushes, 3);
}

#[test]
fn test_exc_info_stack_stats_pops() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));

    stack.pop();
    stack.pop();
    stack.pop(); // Extra pop on empty stack

    assert_eq!(stack.stats().pops, 2); // Only successful pops counted
}

#[test]
fn test_exc_info_stack_stats_peak_depth() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.push(ExcInfoEntry::new(2, None));
    stack.push(ExcInfoEntry::new(3, None));
    stack.pop();
    stack.pop();
    stack.push(ExcInfoEntry::new(4, None));

    assert_eq!(stack.len(), 2);
    assert_eq!(stack.stats().peak_depth, 3); // Peak was 3
}

#[test]
fn test_exc_info_stack_reset_stats() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None));
    stack.pop();

    assert_eq!(stack.stats().pushes, 1);
    assert_eq!(stack.stats().pops, 1);

    stack.reset_stats();

    assert_eq!(stack.stats().pushes, 0);
    assert_eq!(stack.stats().pops, 0);
    assert_eq!(stack.stats().peak_depth, 0);
}

// =========================================================================
// ExcInfoStack Current Exc Info Tests
// =========================================================================

#[test]
fn test_exc_info_stack_current_exc_info_empty() {
    let stack = ExcInfoStack::new();
    let (type_id, value, tb) = stack.current_exc_info();

    assert!(type_id.is_none());
    assert!(value.is_none());
    assert!(tb.is_none());
}

#[test]
fn test_exc_info_stack_current_exc_info_with_entry() {
    let mut stack = ExcInfoStack::new();

    let mut entry = ExcInfoEntry::new(24, Some(Value::int(42).unwrap()));
    entry.set_traceback_id(100);
    stack.push(entry);

    let (type_id, value, tb) = stack.current_exc_info();

    assert_eq!(type_id, Some(24));
    assert!(value.is_some());
    assert_eq!(tb, Some(100));
}

#[test]
fn test_exc_info_stack_current_exc_info_no_traceback() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(24, Some(Value::none())));

    let (type_id, value, tb) = stack.current_exc_info();

    assert_eq!(type_id, Some(24));
    assert!(value.is_some());
    assert!(tb.is_none()); // traceback_id is 0
}

// =========================================================================
// ExcInfoStack Find Active Tests
// =========================================================================

#[test]
fn test_exc_info_stack_find_active_empty() {
    let stack = ExcInfoStack::new();
    assert!(stack.find_active().is_none());
}

#[test]
fn test_exc_info_stack_find_active_all_empty_entries() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::empty());
    stack.push(ExcInfoEntry::empty());

    assert!(stack.find_active().is_none());
}

#[test]
fn test_exc_info_stack_find_active_returns_top() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None)); // Active
    stack.push(ExcInfoEntry::empty()); // Empty
    stack.push(ExcInfoEntry::new(3, None)); // Active - top

    let active = stack.find_active();
    assert!(active.is_some());
    assert_eq!(active.unwrap().type_id(), 3);
}

#[test]
fn test_exc_info_stack_find_active_skips_empty() {
    let mut stack = ExcInfoStack::new();

    stack.push(ExcInfoEntry::new(1, None)); // Active
    stack.push(ExcInfoEntry::empty()); // Empty - on top

    let active = stack.find_active();
    assert!(active.is_some());
    assert_eq!(active.unwrap().type_id(), 1);
}

// =========================================================================
// ExcInfoStack Default Tests
// =========================================================================

#[test]
fn test_exc_info_stack_debug() {
    let mut stack = ExcInfoStack::new();
    stack.push(ExcInfoEntry::new(24, None));

    let debug = format!("{:?}", stack);
    assert!(debug.contains("ExcInfoStack"));
    assert!(debug.contains("depth: 1"));
}

#[test]
fn test_exc_info_stack_stats_debug() {
    let stats = ExcInfoStackStats {
        pushes: 10,
        pops: 5,
        peak_depth: 3,
        overflows: 0,
    };

    let debug = format!("{:?}", stats);
    assert!(debug.contains("pushes: 10"));
    assert!(debug.contains("pops: 5"));
    assert!(debug.contains("peak_depth: 3"));
}

// =========================================================================
// Memory Layout Tests
// =========================================================================

#[test]
fn test_entry_flags_size() {
    assert_eq!(std::mem::size_of::<EntryFlags>(), 1);
}

#[test]
fn test_exc_info_stack_inline_capacity() {
    assert_eq!(INLINE_CAPACITY, 4);
}

// =========================================================================
// Integration Scenario Tests
// =========================================================================

#[test]
fn test_nested_try_except_scenario() {
    let mut stack = ExcInfoStack::new();

    // Outer try block enters except handler
    stack.push(ExcInfoEntry::new(24, Some(Value::int(1).unwrap())));

    // Inner try block enters except handler
    stack.push(ExcInfoEntry::new(5, Some(Value::int(2).unwrap())));

    // Inner handler exits - restore outer exception
    let inner = stack.pop().unwrap();
    assert_eq!(inner.type_id(), 5);

    // Outer exception is now current
    let current = stack.peek().unwrap();
    assert_eq!(current.type_id(), 24);

    // Outer handler exits
    stack.pop();
    assert!(stack.is_empty());
}

#[test]
fn test_raise_from_scenario() {
    let mut stack = ExcInfoStack::new();

    // Original exception
    let mut original = ExcInfoEntry::new(24, Some(Value::int(1).unwrap()));
    stack.push(original);

    // raise NewException from original_exception
    let mut chained = ExcInfoEntry::new(5, Some(Value::int(2).unwrap()));
    chained.set_from_raise_from();
    stack.push(chained);

    let top = stack.peek().unwrap();
    assert!(top.flags().has(EntryFlags::FROM_RAISE_FROM));
    assert!(top.flags().has_cause());
    assert!(top.flags().is_context_suppressed());
}

#[test]
fn test_finally_block_scenario() {
    let mut stack = ExcInfoStack::new();

    // Exception raised
    stack.push(ExcInfoEntry::new(24, Some(Value::int(1).unwrap())));

    // Entering finally block - save exception state
    let saved_depth = stack.len();
    assert_eq!(saved_depth, 1);

    // Finally block runs...
    // (exception state preserved on stack)

    // Exiting finally - exception still there for potential reraise
    assert!(stack.peek().is_some());
    assert_eq!(stack.peek().unwrap().type_id(), 24);
}
