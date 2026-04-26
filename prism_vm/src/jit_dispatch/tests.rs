use super::*;

#[test]
fn test_dispatch_result_was_executed() {
    assert!(DispatchResult::Executed(Value::none()).was_executed());
    assert!(
        DispatchResult::Deopt {
            bc_offset: 0,
            reason: DeoptReason::TypeGuard
        }
        .was_executed()
    );
    assert!(DispatchResult::Exception(RuntimeError::internal("test")).was_executed());
    assert!(!DispatchResult::NotCompiled.was_executed());
}

#[test]
fn test_dispatch_result_needs_interpreter() {
    assert!(!DispatchResult::Executed(Value::none()).needs_interpreter());
    assert!(
        DispatchResult::Deopt {
            bc_offset: 0,
            reason: DeoptReason::TypeGuard
        }
        .needs_interpreter()
    );
    assert!(!DispatchResult::Exception(RuntimeError::internal("test")).needs_interpreter());
    assert!(DispatchResult::NotCompiled.needs_interpreter());
}

#[test]
fn test_dispatch_stats_rates() {
    let stats = DispatchStats {
        attempts: 100,
        hits: 80,
        misses: 20,
        deopts: 8,
        exceptions: 0,
        tail_calls: 0,
    };
    assert!((stats.hit_rate() - 0.8).abs() < 0.001);
    assert!((stats.deopt_rate() - 0.1).abs() < 0.001);
}
