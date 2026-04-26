use super::*;

// ════════════════════════════════════════════════════════════════════════
// HandlerFlags Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flags_empty() {
    let flags = HandlerFlags::EMPTY;
    assert!(!flags.is_finally());
    assert!(!flags.is_named());
    assert!(!flags.is_with_exit());
}

#[test]
fn test_flags_finally() {
    let flags = HandlerFlags::from_raw(HandlerFlags::FINALLY);
    assert!(flags.is_finally());
    assert!(!flags.is_named());
}

#[test]
fn test_flags_named() {
    let flags = HandlerFlags::from_raw(HandlerFlags::NAMED);
    assert!(flags.is_named());
    assert!(!flags.is_finally());
}

#[test]
fn test_flags_with_exit() {
    let flags = HandlerFlags::from_raw(HandlerFlags::WITH_EXIT);
    assert!(flags.is_with_exit());
}

#[test]
fn test_flags_combined() {
    let flags = HandlerFlags::from_raw(HandlerFlags::FINALLY | HandlerFlags::NAMED);
    assert!(flags.is_finally());
    assert!(flags.is_named());
    assert!(!flags.is_with_exit());
}

#[test]
fn test_flags_debug() {
    let flags = HandlerFlags::from_raw(HandlerFlags::FINALLY);
    let debug = format!("{:?}", flags);
    assert!(debug.contains("FINALLY"));
}

// ════════════════════════════════════════════════════════════════════════
// HandlerEntry Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_entry_for_type() {
    let entry = HandlerEntry::for_type(10, 50, 100, ExceptionTypeId::TypeError, 2);
    assert_eq!(entry.pc_start, 10);
    assert_eq!(entry.pc_end, 50);
    assert_eq!(entry.handler_pc, 100);
    assert_eq!(entry.stack_depth, 2);
}

#[test]
fn test_entry_catch_all() {
    let entry = HandlerEntry::catch_all(0, 100, 200, 0);
    assert_eq!(entry.type_filter, CATCH_ALL);
    assert!(!entry.is_finally());
}

#[test]
fn test_entry_catch_exception() {
    let entry = HandlerEntry::catch_exception(0, 100, 200, 0);
    assert_eq!(entry.type_filter, CATCH_EXCEPTION);
}

#[test]
fn test_entry_finally() {
    let entry = HandlerEntry::finally(0, 100, 200, 0);
    assert!(entry.is_finally());
    assert_eq!(entry.type_filter, CATCH_ALL);
}

#[test]
fn test_entry_contains() {
    let entry = HandlerEntry::catch_all(10, 50, 100, 0);
    assert!(!entry.contains(9));
    assert!(entry.contains(10));
    assert!(entry.contains(25));
    assert!(entry.contains(49));
    assert!(!entry.contains(50));
    assert!(!entry.contains(100));
}

#[test]
fn test_entry_matches_type() {
    let entry = HandlerEntry::for_type(0, 100, 200, ExceptionTypeId::OSError, 0);

    // Exact match
    assert!(entry.matches(ExceptionTypeId::OSError));

    // Subclass match
    assert!(entry.matches(ExceptionTypeId::FileNotFoundError));
    assert!(entry.matches(ExceptionTypeId::PermissionError));

    // Non-match
    assert!(!entry.matches(ExceptionTypeId::TypeError));
    assert!(!entry.matches(ExceptionTypeId::ValueError));
}

#[test]
fn test_entry_matches_catch_all() {
    let entry = HandlerEntry::catch_all(0, 100, 200, 0);

    // Catches everything
    assert!(entry.matches(ExceptionTypeId::TypeError));
    assert!(entry.matches(ExceptionTypeId::SystemExit));
    assert!(entry.matches(ExceptionTypeId::KeyboardInterrupt));
    assert!(entry.matches(ExceptionTypeId::StopIteration));
}

#[test]
fn test_entry_matches_catch_exception() {
    let entry = HandlerEntry::catch_exception(0, 100, 200, 0);

    // Catches Exception subclasses
    assert!(entry.matches(ExceptionTypeId::TypeError));
    assert!(entry.matches(ExceptionTypeId::ValueError));

    // Does NOT catch non-Exception types
    assert!(!entry.matches(ExceptionTypeId::SystemExit));
    assert!(!entry.matches(ExceptionTypeId::KeyboardInterrupt));
    assert!(!entry.matches(ExceptionTypeId::GeneratorExit));
}

#[test]
fn test_entry_range() {
    let entry = HandlerEntry::catch_all(10, 50, 100, 0);
    assert_eq!(entry.range(), (10, 50));
}

// ════════════════════════════════════════════════════════════════════════
// HandlerTable Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_table_empty() {
    let table = HandlerTable::empty();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}

#[test]
fn test_table_from_entries() {
    let entries = vec![
        HandlerEntry::catch_all(10, 50, 100, 0),
        HandlerEntry::catch_all(0, 100, 200, 0),
    ];

    let table = HandlerTable::from_entries(entries);
    assert_eq!(table.len(), 2);

    // Should be sorted by pc_start
    assert_eq!(table.entries()[0].pc_start, 0);
    assert_eq!(table.entries()[1].pc_start, 10);
}

#[test]
fn test_table_find_handler_simple() {
    let table = HandlerTable::from_entries(vec![HandlerEntry::for_type(
        10,
        50,
        100,
        ExceptionTypeId::TypeError,
        0,
    )]);

    // PC 25 is in range, TypeError matches
    let handler = table.find_handler(25, ExceptionTypeId::TypeError);
    assert!(handler.is_some());
    assert_eq!(handler.unwrap().handler_pc, 100);

    // PC 5 is out of range
    assert!(table.find_handler(5, ExceptionTypeId::TypeError).is_none());

    // PC 25, but ValueError doesn't match
    assert!(
        table
            .find_handler(25, ExceptionTypeId::ValueError)
            .is_none()
    );
}

#[test]
fn test_table_find_handler_nested() {
    let table = HandlerTable::from_entries(vec![
        HandlerEntry::for_type(0, 100, 200, ExceptionTypeId::Exception, 0),
        HandlerEntry::for_type(20, 60, 150, ExceptionTypeId::TypeError, 0),
    ]);

    // PC 30 with TypeError should match inner handler first
    let handler = table.find_handler(30, ExceptionTypeId::TypeError);
    assert!(handler.is_some());

    // PC 70 with TypeError should match outer handler
    let handler = table.find_handler(70, ExceptionTypeId::TypeError);
    assert!(handler.is_some());
    assert_eq!(handler.unwrap().handler_pc, 200);
}

#[test]
fn test_table_find_all_handlers() {
    let table = HandlerTable::from_entries(vec![
        HandlerEntry::catch_all(0, 100, 200, 0),
        HandlerEntry::catch_all(20, 60, 150, 0),
    ]);

    let handlers: Vec<_> = table.find_all_handlers(30).collect();
    assert_eq!(handlers.len(), 2);
}

#[test]
fn test_table_find_finally() {
    let table = HandlerTable::from_entries(vec![
        HandlerEntry::for_type(0, 100, 200, ExceptionTypeId::TypeError, 0),
        HandlerEntry::finally(0, 100, 300, 0),
    ]);

    let finally = table.find_finally(50);
    assert!(finally.is_some());
    assert!(finally.unwrap().is_finally());
    assert_eq!(finally.unwrap().handler_pc, 300);
}

#[test]
fn test_table_find_finally_none() {
    let table = HandlerTable::from_entries(vec![HandlerEntry::for_type(
        0,
        100,
        200,
        ExceptionTypeId::TypeError,
        0,
    )]);

    assert!(table.find_finally(50).is_none());
}

// ════════════════════════════════════════════════════════════════════════
// HandlerTableBuilder Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_builder_new() {
    let builder = HandlerTableBuilder::new();
    let table = builder.build();
    assert!(table.is_empty());
}

#[test]
fn test_builder_with_capacity() {
    let builder = HandlerTableBuilder::with_capacity(10);
    let table = builder.build();
    assert!(table.is_empty());
}

#[test]
fn test_builder_add() {
    let mut builder = HandlerTableBuilder::new();
    builder.add(HandlerEntry::catch_all(0, 100, 200, 0));
    let table = builder.build();
    assert_eq!(table.len(), 1);
}

#[test]
fn test_builder_add_type_handler() {
    let mut builder = HandlerTableBuilder::new();
    builder.add_type_handler(0, 100, 200, ExceptionTypeId::ValueError, 0);
    let table = builder.build();

    assert!(
        table
            .find_handler(50, ExceptionTypeId::ValueError)
            .is_some()
    );
}

#[test]
fn test_builder_add_catch_all() {
    let mut builder = HandlerTableBuilder::new();
    builder.add_catch_all(0, 100, 200, 0);
    let table = builder.build();

    assert!(table.find_handler(50, ExceptionTypeId::TypeError).is_some());
}

#[test]
fn test_builder_add_finally() {
    let mut builder = HandlerTableBuilder::new();
    builder.add_finally(0, 100, 200, 0);
    let table = builder.build();

    assert!(table.find_finally(50).is_some());
}

#[test]
fn test_builder_chained() {
    let mut builder = HandlerTableBuilder::new();
    builder.add(HandlerEntry::for_type(
        0,
        50,
        100,
        ExceptionTypeId::TypeError,
        0,
    ));
    builder.add(HandlerEntry::for_type(
        0,
        50,
        150,
        ExceptionTypeId::ValueError,
        0,
    ));
    let table = builder.build();

    assert_eq!(table.len(), 2);
}

// ════════════════════════════════════════════════════════════════════════
// Memory Layout Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_handler_entry_size() {
    // HandlerEntry should be compact
    // 4 + 4 + 4 + 2 + 1 + 1 = 16 bytes
    assert_eq!(std::mem::size_of::<HandlerEntry>(), 16);
}

#[test]
fn test_handler_flags_size() {
    assert_eq!(std::mem::size_of::<HandlerFlags>(), 1);
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_handler_at_boundary() {
    let entry = HandlerEntry::catch_all(10, 20, 100, 0);

    // Exactly at start
    assert!(entry.contains(10));

    // Exactly before end
    assert!(entry.contains(19));

    // At end (exclusive)
    assert!(!entry.contains(20));
}

#[test]
fn test_find_handler_empty_table() {
    let table = HandlerTable::empty();
    assert!(table.find_handler(50, ExceptionTypeId::TypeError).is_none());
}

#[test]
fn test_find_handler_before_all() {
    let table = HandlerTable::from_entries(vec![HandlerEntry::catch_all(100, 200, 300, 0)]);

    assert!(table.find_handler(50, ExceptionTypeId::TypeError).is_none());
}

#[test]
fn test_find_handler_after_all() {
    let table = HandlerTable::from_entries(vec![HandlerEntry::catch_all(100, 200, 300, 0)]);

    assert!(
        table
            .find_handler(250, ExceptionTypeId::TypeError)
            .is_none()
    );
}
