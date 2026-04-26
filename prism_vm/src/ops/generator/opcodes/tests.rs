use super::*;

// ════════════════════════════════════════════════════════════════════════
// GeneratorControlFlow Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_control_flow_continue() {
    let cf = GeneratorControlFlow::Continue;
    assert!(!cf.is_yield());
    assert!(!cf.is_yield_from());
    assert!(!cf.is_done());
}

#[test]
fn test_control_flow_yield() {
    let cf = GeneratorControlFlow::Yield {
        value: Value::int(42).unwrap(),
        resume_index: 5,
    };

    assert!(cf.is_yield());
    assert!(!cf.is_yield_from());
    assert!(!cf.is_done());
    assert_eq!(cf.yield_value().unwrap().as_int(), Some(42));
}

#[test]
fn test_control_flow_yield_from() {
    let cf = GeneratorControlFlow::YieldFrom {
        sub_generator: Value::none(),
        resume_index: 3,
    };

    assert!(!cf.is_yield());
    assert!(cf.is_yield_from());
    assert!(!cf.is_done());
}

#[test]
fn test_control_flow_stop_iteration() {
    let cf = GeneratorControlFlow::StopIteration {
        value: Value::none(),
    };

    assert!(!cf.is_yield());
    assert!(cf.is_done());
}

#[test]
fn test_control_flow_throw() {
    let cf = GeneratorControlFlow::Throw {
        exception: Value::none(),
    };

    assert!(!cf.is_yield());
    assert!(!cf.is_done());
}

#[test]
fn test_control_flow_closed() {
    let cf = GeneratorControlFlow::Closed;

    assert!(!cf.is_yield());
    assert!(cf.is_done());
}

// ════════════════════════════════════════════════════════════════════════
// YieldState Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_yield_state_new() {
    let state = YieldState::new(Value::int(42).unwrap(), 5, 0b1010, 3);

    assert_eq!(state.yield_value.as_int(), Some(42));
    assert_eq!(state.resume_index, 5);
    assert_eq!(state.liveness, 0b1010);
    assert_eq!(state.result_reg, 3);
}

// ════════════════════════════════════════════════════════════════════════
// yield_value Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_yield_value_handler() {
    let cf = yield_value(Value::int(100).unwrap(), 7, 2);

    match cf {
        GeneratorControlFlow::Yield {
            value,
            resume_index,
        } => {
            assert_eq!(value.as_int(), Some(100));
            assert_eq!(resume_index, 7);
        }
        _ => panic!("Expected Yield"),
    }
}

#[test]
fn test_yield_value_none() {
    let cf = yield_value(Value::none(), 0, 0);

    assert!(cf.is_yield());
    assert!(cf.yield_value().unwrap().is_none());
}

// ════════════════════════════════════════════════════════════════════════
// yield_from Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_yield_from_handler() {
    let cf = yield_from(Value::int(999).unwrap(), 3);

    match cf {
        GeneratorControlFlow::YieldFrom {
            sub_generator,
            resume_index,
        } => {
            assert_eq!(sub_generator.as_int(), Some(999));
            assert_eq!(resume_index, 3);
        }
        _ => panic!("Expected YieldFrom"),
    }
}

// ════════════════════════════════════════════════════════════════════════
// gen_start Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gen_start_none() {
    let cf = gen_start(None);
    assert!(matches!(cf, GeneratorControlFlow::Continue));
}

#[test]
fn test_gen_start_with_none_value() {
    let cf = gen_start(Some(Value::none()));
    assert!(matches!(cf, GeneratorControlFlow::Continue));
}

#[test]
fn test_gen_start_with_non_none() {
    let cf = gen_start(Some(Value::int(42).unwrap()));
    assert!(matches!(cf, GeneratorControlFlow::Throw { .. }));
}

// ════════════════════════════════════════════════════════════════════════
// throw_into Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_throw_into_handler() {
    let cf = throw_into(Value::int(999).unwrap());

    match cf {
        GeneratorControlFlow::Throw { exception } => {
            assert_eq!(exception.as_int(), Some(999));
        }
        _ => panic!("Expected Throw"),
    }
}

// ════════════════════════════════════════════════════════════════════════
// close_generator Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_close_generator_handler() {
    let cf = close_generator();
    assert!(matches!(cf, GeneratorControlFlow::Closed));
    assert!(cf.is_done());
}

// ════════════════════════════════════════════════════════════════════════
// get_yield_value Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_get_yield_value_from_yield() {
    let cf = GeneratorControlFlow::Yield {
        value: Value::int(42).unwrap(),
        resume_index: 0,
    };

    let value = get_yield_value(&cf);
    assert_eq!(value.unwrap().as_int(), Some(42));
}

#[test]
fn test_get_yield_value_from_stop_iteration() {
    let cf = GeneratorControlFlow::StopIteration {
        value: Value::int(99).unwrap(),
    };

    let value = get_yield_value(&cf);
    assert_eq!(value.unwrap().as_int(), Some(99));
}

#[test]
fn test_get_yield_value_from_continue() {
    let cf = GeneratorControlFlow::Continue;
    assert!(get_yield_value(&cf).is_none());
}

#[test]
fn test_get_yield_value_from_throw() {
    let cf = GeneratorControlFlow::Throw {
        exception: Value::none(),
    };
    assert!(get_yield_value(&cf).is_none());
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorState Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_generator_state_can_resume() {
    assert!(GeneratorState::Created.can_resume());
    assert!(!GeneratorState::Running.can_resume());
    assert!(GeneratorState::Suspended.can_resume());
    assert!(!GeneratorState::Closed.can_resume());
}

#[test]
fn test_generator_state_is_finished() {
    assert!(!GeneratorState::Created.is_finished());
    assert!(!GeneratorState::Running.is_finished());
    assert!(!GeneratorState::Suspended.is_finished());
    assert!(GeneratorState::Closed.is_finished());
}

// ════════════════════════════════════════════════════════════════════════
// is_generator Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_generator_primitives() {
    // Primitives are not generators
    assert!(!is_generator(&Value::none()));
    assert!(!is_generator(&Value::int(42).unwrap()));
    assert!(!is_generator(&Value::bool(true)));
}

// ════════════════════════════════════════════════════════════════════════
// Integration Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_yield_sequence() {
    // Simulate a generator that yields 1, 2, 3
    let yields = [
        yield_value(Value::int(1).unwrap(), 0, 0),
        yield_value(Value::int(2).unwrap(), 1, 0),
        yield_value(Value::int(3).unwrap(), 2, 0),
    ];

    for (i, cf) in yields.iter().enumerate() {
        assert!(cf.is_yield());
        assert_eq!(cf.yield_value().unwrap().as_int(), Some((i + 1) as i64));
    }
}

#[test]
fn test_yield_and_stop() {
    let cf1 = yield_value(Value::int(42).unwrap(), 0, 0);
    let cf2 = GeneratorControlFlow::StopIteration {
        value: Value::none(),
    };

    assert!(!cf1.is_done());
    assert!(cf2.is_done());
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_yield_max_resume_index() {
    let cf = yield_value(Value::none(), u32::MAX, 0);

    match cf {
        GeneratorControlFlow::Yield { resume_index, .. } => {
            assert_eq!(resume_index, u32::MAX);
        }
        _ => panic!("Expected Yield"),
    }
}

#[test]
fn test_yield_from_chain() {
    // Simulate nested yield from
    let cf1 = yield_from(Value::int(1).unwrap(), 0);
    let cf2 = yield_from(Value::int(2).unwrap(), 1);

    assert!(cf1.is_yield_from());
    assert!(cf2.is_yield_from());
}
