use super::*;

// ════════════════════════════════════════════════════════════════════════
// ExceptionState Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_state_from_u8() {
    assert_eq!(ExceptionState::from_u8(0), Some(ExceptionState::Normal));
    assert_eq!(
        ExceptionState::from_u8(1),
        Some(ExceptionState::Propagating)
    );
    assert_eq!(ExceptionState::from_u8(2), Some(ExceptionState::Handling));
    assert_eq!(ExceptionState::from_u8(3), Some(ExceptionState::Finally));
    assert_eq!(ExceptionState::from_u8(4), Some(ExceptionState::Unhandled));
    assert_eq!(ExceptionState::from_u8(5), None);
}

#[test]
fn test_exception_state_as_u8() {
    assert_eq!(ExceptionState::Normal.as_u8(), 0);
    assert_eq!(ExceptionState::Propagating.as_u8(), 1);
    assert_eq!(ExceptionState::Handling.as_u8(), 2);
    assert_eq!(ExceptionState::Finally.as_u8(), 3);
    assert_eq!(ExceptionState::Unhandled.as_u8(), 4);
}

#[test]
fn test_exception_state_predicates() {
    assert!(ExceptionState::Normal.is_normal());
    assert!(!ExceptionState::Normal.has_exception());

    assert!(!ExceptionState::Propagating.is_normal());
    assert!(ExceptionState::Propagating.has_exception());
    assert!(ExceptionState::Propagating.is_propagating());

    assert!(ExceptionState::Handling.is_in_handler());
    assert!(ExceptionState::Finally.is_in_handler());

    assert!(ExceptionState::Unhandled.is_unhandled());
}

#[test]
fn test_exception_state_name() {
    assert_eq!(ExceptionState::Normal.name(), "Normal");
    assert_eq!(ExceptionState::Propagating.name(), "Propagating");
    assert_eq!(ExceptionState::Handling.name(), "Handling");
    assert_eq!(ExceptionState::Finally.name(), "Finally");
    assert_eq!(ExceptionState::Unhandled.name(), "Unhandled");
}

#[test]
fn test_exception_state_size() {
    assert_eq!(std::mem::size_of::<ExceptionState>(), 1);
}

// ════════════════════════════════════════════════════════════════════════
// ExceptionContextFlags Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flags_empty() {
    let flags = ExceptionContextFlags::EMPTY;
    assert_eq!(flags.as_raw(), 0);
    assert!(!flags.is_explicit_raise());
    assert!(!flags.should_reraise());
    assert!(!flags.is_handled());
}

#[test]
fn test_flags_set_clear() {
    let mut flags = ExceptionContextFlags::EMPTY;

    flags.set(ExceptionContextFlags::EXPLICIT_RAISE);
    assert!(flags.is_explicit_raise());

    flags.set(ExceptionContextFlags::RERAISE_AFTER_FINALLY);
    assert!(flags.should_reraise());

    flags.clear(ExceptionContextFlags::EXPLICIT_RAISE);
    assert!(!flags.is_explicit_raise());
    assert!(flags.should_reraise());
}

// ════════════════════════════════════════════════════════════════════════
// ExceptionContext Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_context_new() {
    let ctx = ExceptionContext::new();
    assert!(ctx.is_normal());
    assert!(!ctx.has_exception());
    assert_eq!(ctx.depth(), 0);
    assert_eq!(ctx.handler_pc(), 0);
    assert_eq!(ctx.resume_pc(), 0);
}

#[test]
fn test_context_size() {
    assert_eq!(std::mem::size_of::<ExceptionContext>(), 16);
}

#[test]
fn test_context_begin_propagation() {
    let mut ctx = ExceptionContext::new();
    ctx.begin_propagation(42);

    assert_eq!(ctx.state(), ExceptionState::Propagating);
    assert_eq!(ctx.frame_id(), 42);
    assert_eq!(ctx.depth(), 1);
}

#[test]
fn test_context_begin_handling() {
    let mut ctx = ExceptionContext::new();
    ctx.begin_propagation(0);
    ctx.begin_handling(100, 200);

    assert_eq!(ctx.state(), ExceptionState::Handling);
    assert_eq!(ctx.handler_pc(), 100);
    assert_eq!(ctx.resume_pc(), 200);
}

#[test]
fn test_context_begin_finally() {
    let mut ctx = ExceptionContext::new();
    ctx.begin_propagation(0);
    ctx.begin_finally(150, 250);

    assert_eq!(ctx.state(), ExceptionState::Finally);
    assert_eq!(ctx.handler_pc(), 150);
    assert_eq!(ctx.resume_pc(), 250);
}

#[test]
fn test_context_clear() {
    let mut ctx = ExceptionContext::new();
    ctx.begin_propagation(0);
    ctx.begin_handling(100, 200);
    ctx.clear();

    assert!(ctx.is_normal());
    assert!(ctx.flags().is_handled());
    assert_eq!(ctx.handler_pc(), 0);
}

#[test]
fn test_context_reset() {
    let mut ctx = ExceptionContext::new();
    ctx.begin_propagation(5);
    ctx.begin_handling(100, 200);
    ctx.set_explicit_raise();
    ctx.reset();

    assert!(ctx.is_normal());
    assert_eq!(ctx.depth(), 0);
    assert!(!ctx.flags().is_explicit_raise());
}

#[test]
fn test_context_reraise_flag() {
    let mut ctx = ExceptionContext::new();
    ctx.set_reraise();
    assert!(ctx.flags().should_reraise());

    ctx.clear_reraise();
    assert!(!ctx.flags().should_reraise());
}

// ════════════════════════════════════════════════════════════════════════
// TransitionResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_transition_result_ok() {
    let result = TransitionResult::Ok;
    assert!(result.is_ok());
    assert!(!result.is_err());
}

#[test]
fn test_transition_result_invalid() {
    let result = TransitionResult::InvalidTransition {
        from: ExceptionState::Normal,
        to: ExceptionState::Handling,
    };
    assert!(!result.is_ok());
    assert!(result.is_err());
}

// ════════════════════════════════════════════════════════════════════════
// ExceptionStats Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_stats_new() {
    let stats = ExceptionStats::new();
    assert_eq!(stats.raised, 0);
    assert_eq!(stats.caught, 0);
}

#[test]
fn test_stats_catch_rate() {
    let mut stats = ExceptionStats::new();
    assert_eq!(stats.catch_rate(), 0.0);

    stats.raised = 100;
    stats.caught = 75;
    assert!((stats.catch_rate() - 75.0).abs() < 0.001);
}

#[test]
fn test_stats_reset() {
    let mut stats = ExceptionStats::new();
    stats.raised = 100;
    stats.caught = 50;
    stats.reset();

    assert_eq!(stats.raised, 0);
    assert_eq!(stats.caught, 0);
}

// ════════════════════════════════════════════════════════════════════════
// ExceptionStateMachine Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_state_machine_new() {
    let sm = ExceptionStateMachine::new();
    assert!(sm.is_normal());
    assert_eq!(sm.stats().raised, 0);
}

#[test]
fn test_state_machine_raise() {
    let mut sm = ExceptionStateMachine::new();
    let result = sm.raise(0);

    assert!(result.is_ok());
    assert_eq!(sm.state(), ExceptionState::Propagating);
    assert_eq!(sm.stats().raised, 1);
}

#[test]
fn test_state_machine_raise_from_handler() {
    let mut sm = ExceptionStateMachine::new();
    sm.raise(0);
    sm.enter_handler(100, 200);

    // Can raise a new exception from handler
    let result = sm.raise(0);
    assert!(result.is_ok());
    assert_eq!(sm.state(), ExceptionState::Propagating);
}

#[test]
fn test_state_machine_enter_handler() {
    let mut sm = ExceptionStateMachine::new();
    sm.raise(0);

    let result = sm.enter_handler(100, 200);
    assert!(result.is_ok());
    assert_eq!(sm.state(), ExceptionState::Handling);
    assert_eq!(sm.stats().caught, 1);
}

#[test]
fn test_state_machine_enter_handler_invalid() {
    let mut sm = ExceptionStateMachine::new();

    // Can't enter handler from Normal
    let result = sm.enter_handler(100, 200);
    assert!(result.is_err());
}

#[test]
fn test_state_machine_enter_finally() {
    let mut sm = ExceptionStateMachine::new();
    sm.raise(0);

    let result = sm.enter_finally(150, 250);
    assert!(result.is_ok());
    assert_eq!(sm.state(), ExceptionState::Finally);
    assert_eq!(sm.stats().finally_ran, 1);
}

#[test]
fn test_state_machine_exit_handler() {
    let mut sm = ExceptionStateMachine::new();
    sm.raise(0);
    sm.enter_handler(100, 200);

    let result = sm.exit_handler();
    assert!(result.is_ok());
    assert!(sm.is_normal());
}

#[test]
fn test_state_machine_exit_finally_no_reraise() {
    let mut sm = ExceptionStateMachine::new();
    sm.raise(0);
    sm.enter_finally(100, 200);

    let result = sm.exit_finally();
    assert!(result.is_ok());
    assert!(sm.is_normal());
}

#[test]
fn test_state_machine_exit_finally_with_reraise() {
    let mut sm = ExceptionStateMachine::new();
    sm.raise(0);
    sm.enter_finally(100, 200);
    sm.context_mut().set_reraise();

    let result = sm.exit_finally();
    assert!(result.is_ok());
    assert_eq!(sm.state(), ExceptionState::Propagating);
    assert_eq!(sm.stats().reraised, 1);
}

#[test]
fn test_state_machine_mark_unhandled() {
    let mut sm = ExceptionStateMachine::new();
    sm.raise(0);

    let result = sm.mark_unhandled();
    assert!(result.is_ok());
    assert_eq!(sm.state(), ExceptionState::Unhandled);
    assert_eq!(sm.stats().propagated, 1);
}

#[test]
fn test_state_machine_reset() {
    let mut sm = ExceptionStateMachine::new();
    sm.raise(0);
    sm.enter_handler(100, 200);
    sm.reset();

    assert!(sm.is_normal());
    assert_eq!(sm.context().depth(), 0);
}

// ════════════════════════════════════════════════════════════════════════
// Full Exception Flow Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_flow_caught() {
    let mut sm = ExceptionStateMachine::new();

    // Normal → Raise → Propagating
    assert!(sm.raise(0).is_ok());
    assert_eq!(sm.state(), ExceptionState::Propagating);

    // Propagating → Handler found → Handling
    assert!(sm.enter_handler(100, 200).is_ok());
    assert_eq!(sm.state(), ExceptionState::Handling);

    // Handling → Exit → Normal
    assert!(sm.exit_handler().is_ok());
    assert!(sm.is_normal());

    assert_eq!(sm.stats().raised, 1);
    assert_eq!(sm.stats().caught, 1);
    assert_eq!(sm.stats().propagated, 0);
}

#[test]
fn test_full_flow_unhandled() {
    let mut sm = ExceptionStateMachine::new();

    // Normal → Raise → Propagating
    assert!(sm.raise(0).is_ok());

    // No handler → Unhandled
    assert!(sm.mark_unhandled().is_ok());
    assert_eq!(sm.state(), ExceptionState::Unhandled);

    assert_eq!(sm.stats().raised, 1);
    assert_eq!(sm.stats().propagated, 1);
}

#[test]
fn test_full_flow_finally_and_reraise() {
    let mut sm = ExceptionStateMachine::new();

    // Raise
    assert!(sm.raise(0).is_ok());

    // Enter finally (no handler, but finally must run)
    assert!(sm.enter_finally(100, 200).is_ok());
    sm.context_mut().set_reraise();

    // Exit finally → Reraise
    assert!(sm.exit_finally().is_ok());
    assert_eq!(sm.state(), ExceptionState::Propagating);
    assert_eq!(sm.stats().reraised, 1);
}

#[test]
fn test_nested_exception() {
    let mut sm = ExceptionStateMachine::new();

    // First exception
    sm.raise(0);
    sm.enter_handler(100, 200);

    // Raise another exception in handler
    sm.raise(0);
    assert_eq!(sm.stats().raised, 2);
    assert_eq!(sm.context().depth(), 2);

    // Handle the second exception
    sm.enter_handler(300, 400);
    sm.exit_handler();

    assert!(sm.is_normal());
}
