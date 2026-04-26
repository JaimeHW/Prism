use super::*;
use prism_code::CodeObject;
use std::sync::Arc;

fn test_code() -> Arc<CodeObject> {
    Arc::new(CodeObject::new("test_generator", "<test>"))
}

// ════════════════════════════════════════════════════════════════════════
// Construction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_generator_size() {
    let size = std::mem::size_of::<GeneratorObject>();
    // Includes ObjectHeader (16 bytes) + generator state/storage fields.
    assert!(size <= 192, "GeneratorObject too large: {}", size);
}

#[test]
fn test_generator_layout_header_first() {
    assert_eq!(std::mem::offset_of!(GeneratorObject, header), 0);
}

#[test]
fn test_generator_new() {
    let code = test_code();
    let generator = GeneratorObject::new(code);

    assert_eq!(generator.state(), GeneratorState::Created);
    assert_eq!(generator.resume_index(), 0);
    assert_eq!(generator.ip(), 0);
    assert!(generator.is_resumable());
    assert!(!generator.is_exhausted());
    assert!(!generator.is_running());
}

#[test]
fn test_generator_with_flags() {
    let code = test_code();
    let flags = GeneratorFlags::IS_COROUTINE | GeneratorFlags::INLINE_STORAGE;
    let generator = GeneratorObject::with_flags(code, flags);

    assert!(generator.is_coroutine());
    assert!(!generator.is_async());
    assert_eq!(generator.flags(), flags);
}

#[test]
fn test_generator_from_code_maps_coroutine_flags() {
    let mut code = CodeObject::new("test_generator", "<test>");
    code.flags = CodeFlags::COROUTINE;
    let generator = GeneratorObject::from_code(Arc::new(code));
    assert!(generator.is_coroutine());
    assert!(!generator.is_async());
}

#[test]
fn test_generator_from_code_maps_async_generator_flags() {
    let mut code = CodeObject::new("test_async_generator", "<test>");
    code.flags = CodeFlags::ASYNC_GENERATOR;
    let generator = GeneratorObject::from_code(Arc::new(code));
    assert!(generator.is_async());
}

#[test]
fn test_generator_from_value_roundtrip() {
    let generator = GeneratorObject::new(test_code());
    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    let value = Value::object_ptr(ptr);
    let recovered = GeneratorObject::from_value(value).expect("generator should roundtrip");
    assert_eq!(recovered.header.type_id, TypeId::GENERATOR);
}

#[test]
fn test_seed_locals_captures_values() {
    let mut generator = GeneratorObject::new(test_code());
    let mut regs = [Value::none(); 256];
    regs[0] = Value::int(7).unwrap();
    regs[2] = Value::int(11).unwrap();
    generator.seed_locals(&regs, LivenessMap::from_bits(0b101));

    let mut restored = [Value::none(); 256];
    generator.restore(&mut restored);
    assert_eq!(restored[0].as_int(), Some(7));
    assert_eq!(restored[2].as_int(), Some(11));
}

// ════════════════════════════════════════════════════════════════════════
// Lifecycle Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_generator_closure_round_trips_through_object_state() {
    let mut generator = GeneratorObject::new(test_code());
    let closure = Arc::new(crate::frame::ClosureEnv::with_unbound_cells(1));

    generator.set_closure(Arc::clone(&closure));

    assert!(generator.flags().contains(GeneratorFlags::HAS_CLOSURE));
    assert!(generator.closure().is_some());
    assert!(Arc::ptr_eq(
        generator.closure().expect("closure should be present"),
        &closure
    ));
}

#[test]
fn test_generator_start() {
    let code = test_code();
    let generator = GeneratorObject::new(code);

    let prev = generator.try_start();
    assert_eq!(prev, Some(GeneratorState::Created));
    assert!(generator.is_running());
    assert!(!generator.is_resumable());
}

#[test]
fn test_generator_start_twice_fails() {
    let code = test_code();
    let generator = GeneratorObject::new(code);

    generator.try_start();
    let second = generator.try_start();
    assert_eq!(second, None);
}

#[test]
fn test_generator_suspend() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    generator.try_start();

    let mut registers = [Value::none(); 256];
    registers[0] = Value::int(42).unwrap();
    registers[2] = Value::int(99).unwrap();

    let liveness = LivenessMap::from_bits(0b101);
    generator.suspend(100, 1, &registers, liveness);

    assert_eq!(generator.state(), GeneratorState::Suspended);
    assert_eq!(generator.resume_index(), 1);
    assert_eq!(generator.ip(), 100);
}

#[test]
fn test_generator_suspend_restore() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    generator.try_start();

    let mut registers = [Value::none(); 256];
    registers[0] = Value::int(10).unwrap();
    registers[2] = Value::int(20).unwrap();
    registers[5] = Value::int(30).unwrap();

    let liveness = LivenessMap::from_bits(0b100101);
    generator.suspend(50, 2, &registers, liveness);

    // Restore to fresh registers
    let mut new_regs = [Value::none(); 256];
    generator.restore(&mut new_regs);

    assert_eq!(new_regs[0].as_int().unwrap(), 10);
    assert_eq!(new_regs[2].as_int().unwrap(), 20);
    assert_eq!(new_regs[5].as_int().unwrap(), 30);
}

#[test]
fn test_generator_exhaust() {
    let code = test_code();
    let generator = GeneratorObject::new(code);

    generator.try_start();
    generator.exhaust();

    assert!(generator.is_exhausted());
    assert!(!generator.is_resumable());
    assert_eq!(generator.try_start(), None);
}

#[test]
fn test_generator_full_lifecycle() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    // Created -> Running
    assert_eq!(generator.try_start(), Some(GeneratorState::Created));

    // Running -> Suspended (yield 1)
    let mut regs = [Value::none(); 256];
    regs[0] = Value::int(100).unwrap();
    generator.suspend(10, 0, &regs, LivenessMap::from_bits(0b1));

    // Suspended -> Running
    assert_eq!(generator.try_start(), Some(GeneratorState::Suspended));

    // Running -> Suspended (yield 2)
    regs[0] = Value::int(200).unwrap();
    generator.suspend(20, 1, &regs, LivenessMap::from_bits(0b1));

    // Suspended -> Running -> Exhausted
    assert_eq!(generator.try_start(), Some(GeneratorState::Suspended));
    generator.exhaust();

    // Cannot restart
    assert_eq!(generator.try_start(), None);
}

// ════════════════════════════════════════════════════════════════════════
// Send/Throw Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_send_value() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    assert!(generator.peek_receive_value().is_none());

    generator.set_send_value(Value::int(42).unwrap());
    assert!(generator.peek_receive_value().is_some());

    let value = generator.take_receive_value();
    assert_eq!(value.unwrap().as_int().unwrap(), 42);
    assert!(generator.take_receive_value().is_none());
}

#[test]
fn test_send_none_value() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    generator.set_send_value(Value::none());
    let value = generator.take_receive_value();
    assert!(value.unwrap().is_none());
}

// ════════════════════════════════════════════════════════════════════════
// Flag Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_started_flag() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    assert!(!generator.is_started());
    generator.mark_started();
    assert!(generator.is_started());
}

#[test]
fn test_coroutine_flag() {
    let code = test_code();
    let generator = GeneratorObject::with_flags(code, GeneratorFlags::IS_COROUTINE);

    assert!(generator.is_coroutine());
    assert!(!generator.is_async());
}

#[test]
fn test_async_flag() {
    let code = test_code();
    let generator = GeneratorObject::with_flags(code, GeneratorFlags::IS_ASYNC);

    assert!(generator.is_async());
    assert!(!generator.is_coroutine());
}

#[test]
fn test_multiple_flags() {
    let code = test_code();
    let flags = GeneratorFlags::IS_COROUTINE | GeneratorFlags::HAS_JIT | GeneratorFlags::STARTED;
    let generator = GeneratorObject::with_flags(code, flags);

    assert!(generator.is_coroutine());
    assert!(generator.is_started());
    assert!(generator.flags().contains(GeneratorFlags::HAS_JIT));
}

// ════════════════════════════════════════════════════════════════════════
// Clone Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_generator_clone() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    generator.try_start();
    let mut regs = [Value::none(); 256];
    regs[0] = Value::int(42).unwrap();
    generator.suspend(100, 1, &regs, LivenessMap::from_bits(0b1));
    generator.set_send_value(Value::int(99).unwrap());

    let cloned = generator.clone();

    assert_eq!(cloned.state(), generator.state());
    assert_eq!(cloned.resume_index(), generator.resume_index());
    assert_eq!(cloned.ip(), generator.ip());
}

// ════════════════════════════════════════════════════════════════════════
// Debug Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_generator_debug() {
    let code = test_code();
    let generator = GeneratorObject::new(code);

    let debug = format!("{:?}", generator);
    assert!(debug.contains("GeneratorObject"));
    assert!(debug.contains("Created"));
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorFlags Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flags_union() {
    let flags1 = GeneratorFlags::HAS_JIT;
    let flags2 = GeneratorFlags::IS_COROUTINE;
    let combined = flags1 | flags2;

    assert!(combined.contains(GeneratorFlags::HAS_JIT));
    assert!(combined.contains(GeneratorFlags::IS_COROUTINE));
    assert!(!combined.contains(GeneratorFlags::IS_ASYNC));
}

#[test]
fn test_flags_bitor_assign() {
    let mut flags = GeneratorFlags::EMPTY;
    flags |= GeneratorFlags::STARTED;
    assert!(flags.contains(GeneratorFlags::STARTED));
}
