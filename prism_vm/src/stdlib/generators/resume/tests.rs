use super::*;
use prism_code::CodeObject;
use std::sync::Arc;

fn test_code() -> Arc<CodeObject> {
    Arc::new(CodeObject::new("test_generator", "<test>"))
}

// ════════════════════════════════════════════════════════════════════════
// ResumePoint Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_resume_point_new() {
    let point = ResumePoint::new(100, LivenessMap::from_bits(0b101));
    assert_eq!(point.ip, 100);
    assert_eq!(point.liveness.bits(), 0b101);
    assert_eq!(point.stack_depth, 0);
    assert!(!point.is_yield_from);
}

#[test]
fn test_resume_point_with_depth() {
    let point = ResumePoint::with_depth(200, LivenessMap::from_bits(0b11), 5);
    assert_eq!(point.ip, 200);
    assert_eq!(point.stack_depth, 5);
    assert!(!point.is_yield_from);
}

#[test]
fn test_resume_point_yield_from() {
    let point = ResumePoint::yield_from(300, LivenessMap::from_bits(0b1));
    assert_eq!(point.ip, 300);
    assert!(point.is_yield_from);
}

// ════════════════════════════════════════════════════════════════════════
// ResumeTable Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_resume_table_add() {
    let mut table = ResumeTable::new();
    assert!(table.is_empty());

    let idx0 = table.add(ResumePoint::new(10, LivenessMap::from_bits(0b1)));
    let idx1 = table.add(ResumePoint::new(20, LivenessMap::from_bits(0b11)));

    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(table.len(), 2);
}

#[test]
fn test_resume_table_get() {
    let mut table = ResumeTable::new();
    table.add(ResumePoint::new(100, LivenessMap::from_bits(0b101)));
    table.add(ResumePoint::new(200, LivenessMap::from_bits(0b111)));

    assert_eq!(table.get(0).unwrap().ip, 100);
    assert_eq!(table.get(1).unwrap().ip, 200);
    assert!(table.get(2).is_none());
}

#[test]
fn test_resume_table_get_unchecked() {
    let mut table = ResumeTable::new();
    table.add(ResumePoint::new(50, LivenessMap::from_bits(0)));

    unsafe {
        let point = table.get_unchecked(0);
        assert_eq!(point.ip, 50);
    }
}

#[test]
fn test_resume_table_from_points() {
    let points = vec![
        ResumePoint::new(10, LivenessMap::from_bits(0b1)),
        ResumePoint::new(20, LivenessMap::from_bits(0b10)),
        ResumePoint::new(30, LivenessMap::from_bits(0b11)),
    ];

    let table = ResumeTable::from_points(&points);
    assert_eq!(table.len(), 3);
    assert_eq!(table.get(0).unwrap().ip, 10);
    assert_eq!(table.get(2).unwrap().ip, 30);
}

// ════════════════════════════════════════════════════════════════════════
// ResumeAction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_resume_action_execute() {
    let action = ResumeAction::Execute {
        ip: 100,
        liveness: LivenessMap::from_bits(0b101),
    };
    assert!(action.should_execute());
    assert_eq!(action.ip(), Some(100));
}

#[test]
fn test_resume_action_exhausted() {
    let action = ResumeAction::Exhausted;
    assert!(!action.should_execute());
    assert!(action.ip().is_none());
}

#[test]
fn test_resume_action_already_running() {
    let action = ResumeAction::AlreadyRunning;
    assert!(!action.should_execute());
}

#[test]
fn test_resume_action_invalid_index() {
    let action = ResumeAction::InvalidIndex(99);
    assert!(!action.should_execute());
}

// ════════════════════════════════════════════════════════════════════════
// Prepare Resume Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_prepare_resume_created() {
    let code = test_code();
    let generator = super::super::object::GeneratorObject::new(code);
    let table = ResumeTable::new();

    let action = prepare_resume(&generator, &table);
    assert!(action.should_execute());
    assert_eq!(action.ip(), Some(0));
}

#[test]
fn test_prepare_resume_suspended() {
    let code = test_code();
    let mut generator = super::super::object::GeneratorObject::new(code);

    // Start the generator
    generator.try_start();

    // Suspend at yield point 0
    let liveness = LivenessMap::from_bits(0b1);
    let regs = [Value::none(); 256];
    generator.suspend(50, 0, &regs, liveness);

    // Create table with the resume point
    let mut table = ResumeTable::new();
    table.add(ResumePoint::new(50, liveness));

    let action = prepare_resume(&generator, &table);
    assert!(action.should_execute());
    assert_eq!(action.ip(), Some(50));
}

#[test]
fn test_prepare_resume_running() {
    let code = test_code();
    let generator = super::super::object::GeneratorObject::new(code);
    generator.try_start();

    let table = ResumeTable::new();
    let action = prepare_resume(&generator, &table);
    assert_eq!(action, ResumeAction::AlreadyRunning);
}

#[test]
fn test_prepare_resume_exhausted() {
    let code = test_code();
    let generator = super::super::object::GeneratorObject::new(code);
    generator.try_start();
    generator.exhaust();

    let table = ResumeTable::new();
    let action = prepare_resume(&generator, &table);
    assert_eq!(action, ResumeAction::Exhausted);
}

#[test]
fn test_prepare_resume_invalid_index() {
    let code = test_code();
    let mut generator = super::super::object::GeneratorObject::new(code);

    // Start and suspend at index 5
    generator.try_start();
    let regs = [Value::none(); 256];
    generator.suspend(100, 5, &regs, LivenessMap::from_bits(0));

    // But table only has 2 points
    let mut table = ResumeTable::new();
    table.add(ResumePoint::new(10, LivenessMap::empty()));
    table.add(ResumePoint::new(20, LivenessMap::empty()));

    let action = prepare_resume(&generator, &table);
    assert_eq!(action, ResumeAction::InvalidIndex(5));
}

// ════════════════════════════════════════════════════════════════════════
// ResumeTableBuilder Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_builder_add_yield() {
    let mut builder = ResumeTableBuilder::new();
    assert!(builder.is_empty());

    let idx0 = builder.add_yield(10, LivenessMap::from_bits(0b1));
    let idx1 = builder.add_yield(20, LivenessMap::from_bits(0b11));

    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(builder.len(), 2);
}

#[test]
fn test_builder_add_yield_with_depth() {
    let mut builder = ResumeTableBuilder::new();
    let idx = builder.add_yield_with_depth(100, LivenessMap::from_bits(0b101), 3);

    assert_eq!(idx, 0);

    let table = builder.build();
    assert_eq!(table.get(0).unwrap().stack_depth, 3);
}

#[test]
fn test_builder_add_yield_from() {
    let mut builder = ResumeTableBuilder::new();
    let idx = builder.add_yield_from(50, LivenessMap::from_bits(0b1));

    assert_eq!(idx, 0);

    let table = builder.build();
    assert!(table.get(0).unwrap().is_yield_from);
}

#[test]
fn test_builder_build() {
    let mut builder = ResumeTableBuilder::with_capacity(10);
    builder.add_yield(10, LivenessMap::from_bits(0b1));
    builder.add_yield_from(20, LivenessMap::from_bits(0b10));

    let table = builder.build();
    assert_eq!(table.len(), 2);
    assert!(!table.get(0).unwrap().is_yield_from);
    assert!(table.get(1).unwrap().is_yield_from);
}

// ════════════════════════════════════════════════════════════════════════
// Helper Function Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_can_resume() {
    let code = test_code();
    let generator = super::super::object::GeneratorObject::new(code);

    assert!(can_resume(&generator));

    generator.try_start();
    assert!(!can_resume(&generator)); // Running

    generator.exhaust();
    assert!(!can_resume(&generator)); // Exhausted
}

#[test]
fn test_suspend_at_yield() {
    let code = test_code();
    let mut generator = super::super::object::GeneratorObject::new(code);
    generator.try_start();

    let mut regs = [Value::none(); 256];
    regs[0] = Value::int(42).unwrap();
    regs[1] = Value::int(99).unwrap();

    let liveness = LivenessMap::from_bits(0b11);
    suspend_at_yield(&mut generator, 100, 1, &regs, liveness);

    assert_eq!(generator.state(), GeneratorState::Suspended);
    assert_eq!(generator.ip(), 100);
    assert_eq!(generator.resume_index(), 1);
}

#[test]
fn test_exhaust_generator() {
    let code = test_code();
    let generator = super::super::object::GeneratorObject::new(code);
    generator.try_start();

    exhaust_generator(&generator);
    assert!(generator.is_exhausted());
}

#[test]
fn test_restore_generator_state() {
    let code = test_code();
    let mut generator = super::super::object::GeneratorObject::new(code);
    generator.try_start();

    // Suspend with some values
    let mut regs = [Value::none(); 256];
    regs[0] = Value::int(42).unwrap();
    regs[2] = Value::int(100).unwrap();

    let liveness = LivenessMap::from_bits(0b101);
    suspend_at_yield(&mut generator, 50, 0, &regs, liveness);

    // Restore to new registers
    let mut new_regs = [Value::none(); 256];
    restore_generator_state(&generator, &mut new_regs);

    assert_eq!(new_regs[0].as_int().unwrap(), 42);
    assert_eq!(new_regs[2].as_int().unwrap(), 100);
}
