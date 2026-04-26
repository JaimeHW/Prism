use super::*;

// ════════════════════════════════════════════════════════════════════════
// PendingExceptionEntry Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pending_entry_new() {
    let entry = PendingExceptionEntry::new(100, 0, 5);

    assert_eq!(entry.start_pc, 100);
    assert_eq!(entry.end_pc, 0);
    assert_eq!(entry.handler_pc, 0);
    assert_eq!(entry.finally_pc, NO_FINALLY);
    assert_eq!(entry.depth, 0);
    assert_eq!(entry.stack_depth, 5);
}

#[test]
fn test_pending_entry_finalize() {
    let mut entry = PendingExceptionEntry::new(10, 1, 2);
    entry.end_pc = 50;
    entry.handler_pc = 100;
    entry.finally_pc = 150;

    let finalized = entry.finalize(42);

    assert_eq!(finalized.start_pc, 10);
    assert_eq!(finalized.end_pc, 50);
    assert_eq!(finalized.handler_pc, 100);
    assert_eq!(finalized.finally_pc, 150);
    assert_eq!(finalized.depth, 1);
    assert_eq!(finalized.exception_type_idx, 42);
}

#[test]
fn test_pending_entry_finalize_catch_all() {
    let entry = PendingExceptionEntry::new(0, 0, 0);
    let finalized = entry.finalize(CATCH_ALL);

    assert_eq!(finalized.exception_type_idx, CATCH_ALL);
}

// ════════════════════════════════════════════════════════════════════════
// ExceptionContext Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_new() {
    let ctx = ExceptionContext::new();

    assert_eq!(ctx.current_depth(), 0);
    assert!(!ctx.in_try_block());
    assert!(ctx.entries().is_empty());
}

#[test]
fn test_context_begin_try() {
    let mut ctx = ExceptionContext::new();

    let handle = ctx.begin_try(10, 5);

    assert_eq!(handle, 0);
    assert_eq!(ctx.current_depth(), 1);
    assert!(ctx.in_try_block());
}

#[test]
fn test_context_nested_try() {
    let mut ctx = ExceptionContext::new();

    let h1 = ctx.begin_try(10, 0);
    assert_eq!(ctx.current_depth(), 1);

    let h2 = ctx.begin_try(20, 1);
    assert_eq!(ctx.current_depth(), 2);
    assert_eq!(h1, 0);
    assert_eq!(h2, 1);

    ctx.end_try();
    assert_eq!(ctx.current_depth(), 1);

    ctx.end_try();
    assert_eq!(ctx.current_depth(), 0);
    assert!(!ctx.in_try_block());
}

#[test]
fn test_context_set_pcs() {
    let mut ctx = ExceptionContext::new();
    let handle = ctx.begin_try(10, 0);

    ctx.end_try_body(handle, 50);
    ctx.set_handler_pc(handle, 100);
    ctx.set_finally_pc(handle, 150);
    ctx.finalize_handler(handle, 42);

    let entries = ctx.into_entries();
    assert_eq!(entries.len(), 1);

    let entry = &entries[0];
    assert_eq!(entry.start_pc, 10);
    assert_eq!(entry.end_pc, 50);
    assert_eq!(entry.handler_pc, 100);
    assert_eq!(entry.finally_pc, 150);
    assert_eq!(entry.exception_type_idx, 42);
}

#[test]
fn test_context_multiple_handlers() {
    let mut ctx = ExceptionContext::new();
    let handle = ctx.begin_try(10, 0);

    ctx.end_try_body(handle, 50);
    ctx.set_handler_pc(handle, 100);

    // Two handlers with different types
    ctx.finalize_handler(handle, 1); // Handler for type 1
    ctx.finalize_handler(handle, 2); // Handler for type 2

    let entries = ctx.into_entries();
    assert_eq!(entries.len(), 2);

    assert_eq!(entries[0].exception_type_idx, 1);
    assert_eq!(entries[1].exception_type_idx, 2);
}

#[test]
fn test_context_into_entries_sorted() {
    let mut ctx = ExceptionContext::new();

    // Add entries in reverse order
    let h1 = ctx.begin_try(100, 0);
    ctx.end_try_body(h1, 150);
    ctx.set_handler_pc(h1, 200);
    ctx.finalize_handler(h1, 1);
    ctx.end_try();

    let h2 = ctx.begin_try(10, 0);
    ctx.end_try_body(h2, 50);
    ctx.set_handler_pc(h2, 60);
    ctx.finalize_handler(h2, 2);
    ctx.end_try();

    let entries = ctx.into_entries();

    // Should be sorted by start_pc
    assert_eq!(entries[0].start_pc, 10);
    assert_eq!(entries[1].start_pc, 100);
}

#[test]
fn test_context_end_try_at_zero_depth() {
    let mut ctx = ExceptionContext::new();

    // Should not panic or underflow
    ctx.end_try();
    assert_eq!(ctx.current_depth(), 0);
}

// ════════════════════════════════════════════════════════════════════════
// CompiledHandler Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_compiled_handler_size() {
    // CompiledHandler should be reasonably sized
    let size = std::mem::size_of::<CompiledHandler>();
    assert!(size <= 24, "CompiledHandler is {} bytes", size);
}

// ════════════════════════════════════════════════════════════════════════
// Constant Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_catch_all_constant() {
    assert_eq!(CATCH_ALL, 0xFFFF);
    assert_eq!(CATCH_ALL, u16::MAX);
}

#[test]
fn test_no_finally_constant() {
    assert_eq!(NO_FINALLY, u32::MAX);
}

// ════════════════════════════════════════════════════════════════════════
// Integration Tests (require full compiler setup)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_entry_size() {
    // ExceptionEntry should be compact for cache efficiency
    let size = std::mem::size_of::<ExceptionEntry>();
    assert_eq!(size, 20, "ExceptionEntry should be 20 bytes, was {}", size);
}
