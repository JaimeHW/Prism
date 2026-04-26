use super::*;

// =========================================================================
// DisplayHook Tests
// =========================================================================

#[test]
fn test_display_hook_new() {
    let hook = DisplayHook::new();
    assert!(!hook.is_custom());
}

#[test]
fn test_display_hook_set() {
    let mut hook = DisplayHook::new();
    hook.set(Arc::new(|_| Some("custom".to_string())));
    assert!(hook.is_custom());
}

#[test]
fn test_display_hook_reset() {
    let mut hook = DisplayHook::new();
    hook.set(Arc::new(|_| Some("custom".to_string())));
    hook.reset();
    assert!(!hook.is_custom());
}

#[test]
fn test_display_hook_custom() {
    let mut hook = DisplayHook::new();
    hook.set(Arc::new(|_| Some("CUSTOM".to_string())));
    let result = hook.display(Value::int(42).unwrap());
    assert_eq!(result, "CUSTOM");
}

#[test]
fn test_display_hook_none_passthrough() {
    let mut hook = DisplayHook::new();
    hook.set(Arc::new(|_| None));
    // Should fall back to default
    let result = hook.display(Value::int(42).unwrap());
    assert_eq!(result, "42");
}

#[test]
fn test_default_display_int() {
    let result = default_display(Value::int(123).unwrap());
    assert_eq!(result, "123");
}

#[test]
fn test_default_display_float() {
    let result = default_display(Value::float(3.14));
    assert!(result.starts_with("3.14"));
}

#[test]
fn test_default_display_bool_true() {
    let result = default_display(Value::bool(true));
    assert_eq!(result, "True");
}

#[test]
fn test_default_display_bool_false() {
    let result = default_display(Value::bool(false));
    assert_eq!(result, "False");
}

#[test]
fn test_default_display_none() {
    let result = default_display(Value::none());
    assert_eq!(result, "");
}

// =========================================================================
// ExceptHook Tests
// =========================================================================

#[test]
fn test_except_hook_new() {
    let hook = ExceptHook::new();
    assert!(!hook.is_custom());
}

#[test]
fn test_except_hook_set() {
    let mut hook = ExceptHook::new();
    hook.set(Arc::new(|_, _, _| "custom".to_string()));
    assert!(hook.is_custom());
}

#[test]
fn test_except_hook_reset() {
    let mut hook = ExceptHook::new();
    hook.set(Arc::new(|_, _, _| "custom".to_string()));
    hook.reset();
    assert!(!hook.is_custom());
}

#[test]
fn test_except_hook_format_default() {
    let hook = ExceptHook::new();
    let result = hook.format("ValueError", "invalid value", "");
    assert!(result.contains("ValueError"));
    assert!(result.contains("invalid value"));
}

#[test]
fn test_except_hook_format_with_traceback() {
    let hook = ExceptHook::new();
    let result = hook.format("TypeError", "wrong type", "  File test.py, line 1\n");
    assert!(result.contains("Traceback"));
    assert!(result.contains("test.py"));
}

#[test]
fn test_except_hook_format_custom() {
    let mut hook = ExceptHook::new();
    hook.set(Arc::new(|t, v, _| format!("ERROR: {} - {}", t, v)));
    let result = hook.format("KeyError", "missing", "");
    assert_eq!(result, "ERROR: KeyError - missing");
}

// =========================================================================
// TraceEvent Tests
// =========================================================================

#[test]
fn test_trace_event_as_str() {
    assert_eq!(TraceEvent::Call.as_str(), "call");
    assert_eq!(TraceEvent::Line.as_str(), "line");
    assert_eq!(TraceEvent::Return.as_str(), "return");
    assert_eq!(TraceEvent::Exception.as_str(), "exception");
    assert_eq!(TraceEvent::CCall.as_str(), "c_call");
    assert_eq!(TraceEvent::CReturn.as_str(), "c_return");
    assert_eq!(TraceEvent::CException.as_str(), "c_exception");
    assert_eq!(TraceEvent::Opcode.as_str(), "opcode");
}

// =========================================================================
// TraceHook Tests
// =========================================================================

#[test]
fn test_trace_hook_new() {
    let hook = TraceHook::new();
    assert!(!hook.is_tracing());
}

#[test]
fn test_trace_hook_set() {
    let mut hook = TraceHook::new();
    hook.set(Arc::new(|_, _, _| {}));
    assert!(hook.is_tracing());
}

#[test]
fn test_trace_hook_clear() {
    let mut hook = TraceHook::new();
    hook.set(Arc::new(|_, _, _| {}));
    hook.clear();
    assert!(!hook.is_tracing());
}

#[test]
fn test_trace_hook_opcodes() {
    let mut hook = TraceHook::new();

    // Without trace function, opcode tracing is off
    hook.enable_opcodes();
    assert!(!hook.traces_opcodes());

    // With trace function
    hook.set(Arc::new(|_, _, _| {}));
    assert!(hook.traces_opcodes());

    hook.disable_opcodes();
    assert!(!hook.traces_opcodes());
}

#[test]
fn test_trace_hook_trace_with_fn() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let counter = Arc::new(AtomicU32::new(0));
    let counter_clone = counter.clone();

    let mut hook = TraceHook::new();
    hook.set(Arc::new(move |_, _, _| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    }));

    hook.trace(TraceEvent::Call, "test.py", 1);
    hook.trace(TraceEvent::Line, "test.py", 2);

    assert_eq!(counter.load(Ordering::SeqCst), 2);
}
