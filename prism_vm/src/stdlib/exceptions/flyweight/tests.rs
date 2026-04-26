use super::*;

// ════════════════════════════════════════════════════════════════════════
// Flyweight Access Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_stop_iteration() {
    let exc = FlyweightPool::stop_iteration();
    assert_eq!(exc.type_id(), ExceptionTypeId::StopIteration);
    assert!(exc.is_flyweight());
}

#[test]
fn test_stop_async_iteration() {
    let exc = FlyweightPool::stop_async_iteration();
    assert_eq!(exc.type_id(), ExceptionTypeId::StopAsyncIteration);
    assert!(exc.is_flyweight());
}

#[test]
fn test_generator_exit() {
    let exc = FlyweightPool::generator_exit();
    assert_eq!(exc.type_id(), ExceptionTypeId::GeneratorExit);
    assert!(exc.is_flyweight());
}

#[test]
fn test_keyboard_interrupt() {
    let exc = FlyweightPool::keyboard_interrupt();
    assert_eq!(exc.type_id(), ExceptionTypeId::KeyboardInterrupt);
    assert!(exc.is_flyweight());
}

#[test]
fn test_memory_error() {
    let exc = FlyweightPool::memory_error();
    assert_eq!(exc.type_id(), ExceptionTypeId::MemoryError);
    assert!(exc.is_flyweight());
}

#[test]
fn test_recursion_error() {
    let exc = FlyweightPool::recursion_error();
    assert_eq!(exc.type_id(), ExceptionTypeId::RecursionError);
    assert!(exc.is_flyweight());
}

// ════════════════════════════════════════════════════════════════════════
// Flyweight Get Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_get_stop_iteration() {
    let exc = FlyweightPool::get(ExceptionTypeId::StopIteration);
    assert!(exc.is_some());
    assert_eq!(exc.unwrap().type_id(), ExceptionTypeId::StopIteration);
}

#[test]
fn test_get_generator_exit() {
    let exc = FlyweightPool::get(ExceptionTypeId::GeneratorExit);
    assert!(exc.is_some());
}

#[test]
fn test_get_none_for_regular_exception() {
    assert!(FlyweightPool::get(ExceptionTypeId::TypeError).is_none());
    assert!(FlyweightPool::get(ExceptionTypeId::ValueError).is_none());
    assert!(FlyweightPool::get(ExceptionTypeId::KeyError).is_none());
}

// ════════════════════════════════════════════════════════════════════════
// Has Flyweight Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_has_flyweight_true() {
    assert!(FlyweightPool::has_flyweight(ExceptionTypeId::StopIteration));
    assert!(FlyweightPool::has_flyweight(
        ExceptionTypeId::StopAsyncIteration
    ));
    assert!(FlyweightPool::has_flyweight(ExceptionTypeId::GeneratorExit));
    assert!(FlyweightPool::has_flyweight(
        ExceptionTypeId::KeyboardInterrupt
    ));
    assert!(FlyweightPool::has_flyweight(ExceptionTypeId::MemoryError));
    assert!(FlyweightPool::has_flyweight(
        ExceptionTypeId::RecursionError
    ));
}

#[test]
fn test_has_flyweight_false() {
    assert!(!FlyweightPool::has_flyweight(ExceptionTypeId::TypeError));
    assert!(!FlyweightPool::has_flyweight(ExceptionTypeId::ValueError));
    assert!(!FlyweightPool::has_flyweight(ExceptionTypeId::OSError));
    assert!(!FlyweightPool::has_flyweight(ExceptionTypeId::RuntimeError));
}

// ════════════════════════════════════════════════════════════════════════
// Convenience Function Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_raise_stop_iteration() {
    let exc = raise_stop_iteration();
    assert_eq!(exc.type_id(), ExceptionTypeId::StopIteration);
}

#[test]
fn test_raise_generator_exit() {
    let exc = raise_generator_exit();
    assert_eq!(exc.type_id(), ExceptionTypeId::GeneratorExit);
}

#[test]
fn test_stop_iteration_with_value() {
    let exc = stop_iteration_with_value(prism_core::Value::none());
    assert_eq!(exc.type_id(), ExceptionTypeId::StopIteration);
    assert!(!exc.is_flyweight()); // Has args, so not flyweight
    assert!(exc.args().is_some());
}

// ════════════════════════════════════════════════════════════════════════
// Static Reference Stability Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_same_instance() {
    // Multiple calls should return the same static reference
    let a = FlyweightPool::stop_iteration();
    let b = FlyweightPool::stop_iteration();
    assert!(std::ptr::eq(a, b));
}

#[test]
fn test_flyweight_get_same_instance() {
    let a = FlyweightPool::stop_iteration();
    let b = FlyweightPool::get(ExceptionTypeId::StopIteration).unwrap();
    assert!(std::ptr::eq(a, b));
}

// ════════════════════════════════════════════════════════════════════════
// Flyweight Properties Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_is_normalized() {
    let exc = FlyweightPool::stop_iteration();
    assert!(exc.is_normalized());
}

#[test]
fn test_flyweight_no_args() {
    let exc = FlyweightPool::stop_iteration();
    assert!(exc.args().is_none());
}

#[test]
fn test_flyweight_no_traceback() {
    let exc = FlyweightPool::generator_exit();
    assert!(!exc.has_traceback());
}

#[test]
fn test_flyweight_no_cause() {
    let exc = FlyweightPool::stop_iteration();
    assert!(exc.cause().is_none());
}

#[test]
fn test_flyweight_no_context() {
    let exc = FlyweightPool::stop_iteration();
    assert!(exc.context().is_none());
}

// ════════════════════════════════════════════════════════════════════════
// Control Flow Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_control_flow_exceptions() {
    // All control-flow exceptions should have flyweights
    assert!(FlyweightPool::has_flyweight(ExceptionTypeId::StopIteration));
    assert!(FlyweightPool::has_flyweight(
        ExceptionTypeId::StopAsyncIteration
    ));
    assert!(FlyweightPool::has_flyweight(ExceptionTypeId::GeneratorExit));
}

// ════════════════════════════════════════════════════════════════════════
// Thread Safety Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_thread_safe_access() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let exc = FlyweightPool::stop_iteration();
                exc.type_id()
            })
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().unwrap(), ExceptionTypeId::StopIteration);
    }
}
