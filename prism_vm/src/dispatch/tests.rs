use super::*;

// =========================================================================
// ControlFlow Tests
// =========================================================================

#[test]
fn test_control_flow_continue() {
    let cf = ControlFlow::Continue;
    assert!(matches!(cf, ControlFlow::Continue));
}

#[test]
fn test_control_flow_jump() {
    let cf = ControlFlow::Jump(10);
    if let ControlFlow::Jump(offset) = cf {
        assert_eq!(offset, 10);
    } else {
        panic!("Expected Jump");
    }

    // Negative jump
    let cf = ControlFlow::Jump(-5);
    if let ControlFlow::Jump(offset) = cf {
        assert_eq!(offset, -5);
    } else {
        panic!("Expected Jump");
    }
}

#[test]
fn test_control_flow_return() {
    let cf = ControlFlow::Return(Value::int(42).unwrap());
    if let ControlFlow::Return(v) = cf {
        assert_eq!(v.as_int(), Some(42));
    } else {
        panic!("Expected Return");
    }
}

#[test]
fn test_control_flow_exception() {
    let cf = ControlFlow::Exception {
        type_id: 5,
        handler_pc: 100,
    };
    if let ControlFlow::Exception {
        type_id,
        handler_pc,
    } = cf
    {
        assert_eq!(type_id, 5);
        assert_eq!(handler_pc, 100);
    } else {
        panic!("Expected Exception");
    }
}

#[test]
fn test_control_flow_exception_unknown_handler() {
    let cf = ControlFlow::Exception {
        type_id: 1,
        handler_pc: 0, // Unknown handler
    };
    if let ControlFlow::Exception { handler_pc, .. } = cf {
        assert_eq!(handler_pc, 0);
    } else {
        panic!("Expected Exception");
    }
}

#[test]
fn test_control_flow_reraise() {
    let cf = ControlFlow::Reraise;
    assert!(matches!(cf, ControlFlow::Reraise));
}

#[test]
fn test_control_flow_enter_handler() {
    let cf = ControlFlow::EnterHandler {
        handler_pc: 50,
        stack_depth: 3,
    };
    if let ControlFlow::EnterHandler {
        handler_pc,
        stack_depth,
    } = cf
    {
        assert_eq!(handler_pc, 50);
        assert_eq!(stack_depth, 3);
    } else {
        panic!("Expected EnterHandler");
    }
}

#[test]
fn test_control_flow_enter_finally() {
    let cf = ControlFlow::EnterFinally {
        finally_pc: 75,
        stack_depth: 2,
        reraise: true,
    };
    if let ControlFlow::EnterFinally {
        finally_pc,
        stack_depth,
        reraise,
    } = cf
    {
        assert_eq!(finally_pc, 75);
        assert_eq!(stack_depth, 2);
        assert!(reraise);
    } else {
        panic!("Expected EnterFinally");
    }
}

#[test]
fn test_control_flow_enter_finally_no_reraise() {
    let cf = ControlFlow::EnterFinally {
        finally_pc: 80,
        stack_depth: 1,
        reraise: false,
    };
    if let ControlFlow::EnterFinally { reraise, .. } = cf {
        assert!(!reraise);
    } else {
        panic!("Expected EnterFinally");
    }
}

#[test]
fn test_control_flow_exit_handler() {
    let cf = ControlFlow::ExitHandler;
    assert!(matches!(cf, ControlFlow::ExitHandler));
}

#[test]
fn test_control_flow_yield() {
    let cf = ControlFlow::Yield {
        value: Value::int(100).unwrap(),
        resume_point: 25,
    };
    if let ControlFlow::Yield {
        value,
        resume_point,
    } = cf
    {
        assert_eq!(value.as_int(), Some(100));
        assert_eq!(resume_point, 25);
    } else {
        panic!("Expected Yield");
    }
}

#[test]
fn test_control_flow_resume() {
    let cf = ControlFlow::Resume {
        send_value: Value::none(),
    };
    if let ControlFlow::Resume { send_value } = cf {
        assert!(send_value.is_none());
    } else {
        panic!("Expected Resume");
    }
}

#[test]
fn test_control_flow_resume_with_value() {
    let cf = ControlFlow::Resume {
        send_value: Value::int(42).unwrap(),
    };
    if let ControlFlow::Resume { send_value } = cf {
        assert_eq!(send_value.as_int(), Some(42));
    } else {
        panic!("Expected Resume");
    }
}

#[test]
fn test_control_flow_error() {
    let err = RuntimeError::internal("test error");
    let cf = ControlFlow::Error(err);
    assert!(matches!(cf, ControlFlow::Error(_)));
}

#[test]
fn test_control_flow_size() {
    // Ensure ControlFlow remains reasonably sized
    // Note: The Error variant contains RuntimeError which includes a Vec<TracebackEntry>
    // making it the largest variant. On 64-bit systems the size is ~88 bytes.
    let size = std::mem::size_of::<ControlFlow>();
    assert!(
        size <= 104,
        "ControlFlow size is {} bytes, expected <= 104",
        size
    );
}
