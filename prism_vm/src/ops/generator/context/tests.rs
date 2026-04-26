use super::*;
use prism_core::Value;

// Helper to create a dangling generator pointer for tests
fn dangling_generator() -> NonNull<GeneratorObject> {
    NonNull::dangling()
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorExecutionState Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_state_can_continue() {
    assert!(!GeneratorExecutionState::Idle.can_continue());
    assert!(GeneratorExecutionState::Running.can_continue());
    assert!(!GeneratorExecutionState::Suspended.can_continue());
    assert!(!GeneratorExecutionState::Closing.can_continue());
    assert!(GeneratorExecutionState::Throwing.can_continue());
    assert!(!GeneratorExecutionState::Completed.can_continue());
}

#[test]
fn test_state_is_active() {
    assert!(!GeneratorExecutionState::Idle.is_active());
    assert!(GeneratorExecutionState::Running.is_active());
    assert!(GeneratorExecutionState::Suspended.is_active());
    assert!(GeneratorExecutionState::Closing.is_active());
    assert!(GeneratorExecutionState::Throwing.is_active());
    assert!(!GeneratorExecutionState::Completed.is_active());
}

#[test]
fn test_state_can_resume() {
    assert!(!GeneratorExecutionState::Idle.can_resume());
    assert!(!GeneratorExecutionState::Running.can_resume());
    assert!(GeneratorExecutionState::Suspended.can_resume());
    assert!(!GeneratorExecutionState::Closing.can_resume());
    assert!(!GeneratorExecutionState::Throwing.can_resume());
    assert!(!GeneratorExecutionState::Completed.can_resume());
}

#[test]
fn test_state_can_close() {
    assert!(!GeneratorExecutionState::Idle.can_close());
    assert!(GeneratorExecutionState::Running.can_close());
    assert!(GeneratorExecutionState::Suspended.can_close());
    assert!(!GeneratorExecutionState::Closing.can_close());
    assert!(!GeneratorExecutionState::Throwing.can_close());
    assert!(!GeneratorExecutionState::Completed.can_close());
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorContext Construction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_new() {
    let ctx = GeneratorContext::new();
    assert!(!ctx.is_active());
    assert!(ctx.current_generator().is_none());
    assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
    assert_eq!(ctx.nesting_depth(), 0);
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorContext Enter/Exit Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_enter() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);

    assert!(ctx.is_active());
    assert!(ctx.current_generator().is_some());
    assert_eq!(ctx.state(), GeneratorExecutionState::Running);
}

#[test]
fn test_context_exit() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.exit();

    assert!(!ctx.is_active());
    assert!(ctx.current_generator().is_none());
    assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
}

#[test]
fn test_context_enter_exit_preserves_stats() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.exit();

    let stats = ctx.stats();
    assert_eq!(stats.activations, 1);
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorContext Suspend/Resume Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_suspend() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.suspend();

    assert_eq!(ctx.state(), GeneratorExecutionState::Suspended);

    let stats = ctx.stats();
    assert_eq!(stats.total_yields, 1);
}

#[test]
fn test_context_resume_without_value() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.suspend();
    ctx.resume(None);

    assert_eq!(ctx.state(), GeneratorExecutionState::Running);
    assert!(ctx.take_send_value().is_none());

    let stats = ctx.stats();
    assert_eq!(stats.total_resumes, 1);
}

#[test]
fn test_context_resume_with_value() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.suspend();
    ctx.resume(Some(Value::int(42).unwrap()));

    assert_eq!(ctx.state(), GeneratorExecutionState::Running);
    assert!(ctx.has_send_value());

    let sent = ctx.take_send_value();
    assert_eq!(sent.unwrap().as_int(), Some(42));

    // Value should be taken
    assert!(!ctx.has_send_value());
}

#[test]
fn test_context_resume_clears_on_exit() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.suspend();
    ctx.resume(Some(Value::int(42).unwrap()));
    ctx.exit();

    assert!(!ctx.has_send_value());
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorContext Close Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_begin_close() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.begin_close();

    assert_eq!(ctx.state(), GeneratorExecutionState::Closing);

    let stats = ctx.stats();
    assert_eq!(stats.total_closes, 1);
}

#[test]
fn test_context_complete() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.complete();

    assert_eq!(ctx.state(), GeneratorExecutionState::Completed);
    assert!(ctx.is_active()); // Still has generator ref
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorContext Throw Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_throw() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.throw(Value::int(999).unwrap());

    assert_eq!(ctx.state(), GeneratorExecutionState::Throwing);
    assert!(ctx.has_thrown_exception());

    let stats = ctx.stats();
    assert_eq!(stats.total_throws, 1);
}

#[test]
fn test_context_take_thrown_exception() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.throw(Value::int(999).unwrap());

    let exc = ctx.take_thrown_exception();
    assert_eq!(exc.unwrap().as_int(), Some(999));
    assert!(!ctx.has_thrown_exception());
}

#[test]
fn test_context_throw_clears_on_exit() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.throw(Value::int(999).unwrap());
    ctx.exit();

    assert!(!ctx.has_thrown_exception());
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorContext Nesting Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_nesting() {
    let mut ctx = GeneratorContext::new();
    let gen1 = dangling_generator();
    let gen2 = dangling_generator();

    ctx.enter(gen1);
    assert_eq!(ctx.nesting_depth(), 0);

    ctx.enter(gen2);
    assert_eq!(ctx.nesting_depth(), 1);

    ctx.exit();
    assert_eq!(ctx.nesting_depth(), 0);
    assert!(ctx.is_active()); // gen1 still active

    ctx.exit();
    assert!(!ctx.is_active());
}

#[test]
fn test_context_deep_nesting() {
    let mut ctx = GeneratorContext::new();

    for i in 0..10 {
        ctx.enter(dangling_generator());
        assert_eq!(ctx.nesting_depth(), i.min(9)); // First enter doesn't increment
    }

    let stats = ctx.stats();
    assert_eq!(stats.max_nesting_depth, 9);
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorContext Reset Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_reset() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.suspend();
    ctx.resume(Some(Value::int(42).unwrap()));
    ctx.throw(Value::int(999).unwrap());

    ctx.reset();

    assert!(!ctx.is_active());
    assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
    assert_eq!(ctx.nesting_depth(), 0);
    assert!(!ctx.has_send_value());
    assert!(!ctx.has_thrown_exception());
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorContext Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_multiple_yield_resume_cycles() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);

    for i in 0..5 {
        ctx.suspend();
        assert_eq!(ctx.state(), GeneratorExecutionState::Suspended);

        ctx.resume(Some(Value::int(i).unwrap()));
        assert_eq!(ctx.state(), GeneratorExecutionState::Running);

        let val = ctx.take_send_value();
        assert_eq!(val.unwrap().as_int(), Some(i));
    }

    let stats = ctx.stats();
    assert_eq!(stats.total_yields, 5);
    assert_eq!(stats.total_resumes, 5);
}

#[test]
fn test_context_peek_does_not_consume() {
    let mut ctx = GeneratorContext::new();
    let generator = dangling_generator();

    ctx.enter(generator);
    ctx.suspend();
    ctx.resume(Some(Value::int(42).unwrap()));

    // Peek should not consume
    assert_eq!(ctx.peek_send_value().unwrap().as_int(), Some(42));
    assert_eq!(ctx.peek_send_value().unwrap().as_int(), Some(42));

    // Take consumes
    assert_eq!(ctx.take_send_value().unwrap().as_int(), Some(42));
    assert!(ctx.peek_send_value().is_none());
}

#[test]
fn test_context_stats_accumulate() {
    let mut ctx = GeneratorContext::new();

    for _ in 0..3 {
        ctx.enter(dangling_generator());
        ctx.suspend();
        ctx.resume(None);
        ctx.begin_close();
        ctx.exit();
    }

    let stats = ctx.stats();
    assert_eq!(stats.activations, 3);
    assert_eq!(stats.total_yields, 3);
    assert_eq!(stats.total_resumes, 3);
    assert_eq!(stats.total_closes, 3);
}

// ════════════════════════════════════════════════════════════════════════
// Size and Performance Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_size() {
    let size = std::mem::size_of::<GeneratorContext>();
    // Should be reasonably compact
    assert!(size <= 128, "GeneratorContext too large: {} bytes", size);
}

#[test]
fn test_state_size() {
    let size = std::mem::size_of::<GeneratorExecutionState>();
    assert_eq!(size, 1, "State should be 1 byte enum");
}

#[test]
fn test_stats_size() {
    let size = std::mem::size_of::<GeneratorContextStats>();
    assert!(size <= 48, "Stats too large: {} bytes", size);
}
