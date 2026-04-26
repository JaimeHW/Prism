use super::*;

#[test]
fn test_deopt_reason() {
    assert_eq!(DeoptReason::TypeGuardFailed.as_str(), "TypeGuardFailed");
    assert_eq!(DeoptReason::IntegerOverflow as u8, 1);
}

#[test]
fn test_deopt_info() {
    let info = DeoptInfo::new(100, 200, DeoptReason::TypeGuardFailed, 0);
    assert_eq!(info.bc_offset, 100);
    assert_eq!(info.native_offset, 200);
    assert_eq!(info.reason, DeoptReason::TypeGuardFailed);
}

#[test]
fn test_deopt_stub_generator() {
    let mut generator = DeoptStubGenerator::new();
    // Use the assembler to create labels
    let mut asm = crate::backend::x64::Assembler::new();
    let label1 = asm.create_label();
    let label2 = asm.create_label();

    generator.register_deopt(label1, 42, DeoptReason::TypeGuardFailed);
    generator.register_deopt(label2, 84, DeoptReason::IntegerOverflow);

    assert_eq!(generator.deopt_count(), 2);

    let frame = FrameLayout::minimal(2);
    let infos = generator.emit_stubs(&mut asm, &frame);
    assert_eq!(infos.len(), 2);
    assert_eq!(infos[0].bc_offset, 42);
    assert_eq!(infos[1].bc_offset, 84);
}

#[test]
fn test_encode_deopt_exit_layout() {
    let encoded = encode_deopt_exit(0x123456, DeoptReason::InlineCacheMiss);
    assert_eq!((encoded & 0xFF) as u8, ExitReason::Deoptimize as u8);
    assert_eq!(
        ((encoded >> 8) & 0xFF) as u8,
        DeoptReason::InlineCacheMiss as u8
    );
    assert_eq!(((encoded >> 16) & 0x00FF_FFFF) as u32, 0x123456);
}

#[test]
fn test_deopt_counter() {
    let mut counter = DeoptCounter::new();

    // Should allow recompilation initially
    assert!(counter.record(DeoptReason::TypeGuardFailed));
    assert_eq!(counter.total, 1);

    // After many deopts, should give up
    for _ in 0..15 {
        counter.record(DeoptReason::TypeGuardFailed);
    }

    assert!(counter.should_abandon_jit());
}

#[test]
fn test_deopt_state() {
    let state = DeoptState::new(100, 8);
    assert_eq!(state.bc_offset, 100);
    assert_eq!(state.register_values.len(), 8);
    assert!(!state.in_exception_handler);
}
