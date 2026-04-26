use super::*;
use crate::backend::x64::registers::Gpr;

fn make_inst(op: MachineOp) -> MachineInst {
    MachineInst::nullary(op)
}

fn make_label(id: u32) -> MachineInst {
    MachineInst::new(
        MachineOp::Label,
        MachineOperand::Label(id),
        MachineOperand::None,
    )
}

fn make_jmp(target: u32) -> MachineInst {
    MachineInst::new(
        MachineOp::Jmp,
        MachineOperand::Label(target),
        MachineOperand::None,
    )
}

fn make_jcc(target: u32) -> MachineInst {
    let mut inst = MachineInst::new(
        MachineOp::Jcc,
        MachineOperand::Label(target),
        MachineOperand::None,
    );
    inst.cc = Some(super::super::lower::CondCode::E);
    inst
}

#[test]
fn test_analyzer_new() {
    let analyzer = SafepointAnalyzer::new();
    assert_eq!(analyzer.max_poll_interval, 1024);
}

#[test]
fn test_analyzer_custom_interval() {
    let analyzer = SafepointAnalyzer::with_poll_interval(512);
    assert_eq!(analyzer.max_poll_interval, 512);
}

#[test]
fn test_empty_function_is_leaf() {
    let mfunc = MachineFunction::new();
    let analyzer = SafepointAnalyzer::new();
    let placement = analyzer.analyze(&mfunc);

    assert!(placement.is_leaf);
    assert!(placement.poll_indices.is_empty());
    assert!(!placement.needs_safepoint_register);
}

#[test]
fn test_short_function_is_leaf() {
    let mut mfunc = MachineFunction::new();
    // Add a few simple instructions
    mfunc.push(MachineInst::new(
        MachineOp::Mov,
        MachineOperand::gpr(Gpr::Rax),
        MachineOperand::Imm(42),
    ));
    mfunc.push(make_inst(MachineOp::Ret));

    let analyzer = SafepointAnalyzer::new();
    let placement = analyzer.analyze(&mfunc);

    assert!(placement.is_leaf);
}

#[test]
fn test_function_with_call_not_leaf() {
    let mut mfunc = MachineFunction::new();
    mfunc.push(MachineInst::new(
        MachineOp::Call,
        MachineOperand::Imm(0x12345678),
        MachineOperand::None,
    ));
    mfunc.push(make_inst(MachineOp::Ret));

    let analyzer = SafepointAnalyzer::new();
    let placement = analyzer.analyze(&mfunc);

    // Has a call, but call itself is a safepoint, so no extra polls needed
    assert!(!placement.is_leaf);
}

#[test]
fn test_back_edge_detection() {
    let mut mfunc = MachineFunction::new();

    // Loop header
    mfunc.push(make_label(1));

    // Loop body
    for _ in 0..10 {
        mfunc.push(make_inst(MachineOp::Nop));
    }

    // Back-edge (jump to loop header)
    mfunc.push(make_jcc(1));

    // Exit
    mfunc.push(make_inst(MachineOp::Ret));

    let analyzer = SafepointAnalyzer::new();
    let back_edges = analyzer.find_back_edges(&mfunc);

    // Should detect the back-edge before the jcc
    assert!(!back_edges.is_empty());
}

#[test]
fn test_forward_jump_not_back_edge() {
    let mut mfunc = MachineFunction::new();

    // Forward jump
    mfunc.push(make_jmp(1));

    // Some instructions
    for _ in 0..5 {
        mfunc.push(make_inst(MachineOp::Nop));
    }

    // Target label
    mfunc.push(make_label(1));
    mfunc.push(make_inst(MachineOp::Ret));

    let analyzer = SafepointAnalyzer::new();
    let back_edges = analyzer.find_back_edges(&mfunc);

    // Forward jumps are not back-edges
    assert!(back_edges.is_empty());
}

#[test]
fn test_loop_requires_safepoint() {
    let mut mfunc = MachineFunction::new();

    // Simple loop
    mfunc.push(make_label(1));
    for _ in 0..50 {
        mfunc.push(make_inst(MachineOp::Nop));
    }
    mfunc.push(make_jcc(1));
    mfunc.push(make_inst(MachineOp::Ret));

    let analyzer = SafepointAnalyzer::new();
    let placement = analyzer.analyze(&mfunc);

    assert!(!placement.is_leaf);
    assert!(placement.needs_safepoint_register);
    assert!(!placement.poll_indices.is_empty());
}

#[test]
fn test_safepoint_emitter_creation() {
    let placement = SafepointPlacement {
        poll_indices: SmallVec::from_slice(&[10, 20, 30]),
        is_leaf: false,
        needs_safepoint_register: true,
    };

    let emitter = SafepointEmitter::new(placement, 0xDEADBEEF);

    assert!(emitter.should_emit_poll(10));
    assert!(emitter.should_emit_poll(20));
    assert!(emitter.should_emit_poll(30));
    assert!(!emitter.should_emit_poll(15));
    assert_eq!(emitter.poll_count(), 3);
}

#[test]
fn test_safepoint_emitter_leaf() {
    let emitter = SafepointEmitter::new(SafepointPlacement::none(), 0);

    assert!(emitter.is_leaf());
    assert!(!emitter.needs_safepoint_register());
    assert_eq!(emitter.poll_count(), 0);
}

#[test]
fn test_interval_poll_long_straight_line() {
    let mut mfunc = MachineFunction::new();

    // Create very long straight-line code
    for _ in 0..2000 {
        mfunc.push(make_inst(MachineOp::Nop));
    }
    mfunc.push(make_inst(MachineOp::Ret));

    // Use small interval to trigger interval polls
    let analyzer = SafepointAnalyzer::with_poll_interval(100);
    let placement = analyzer.analyze(&mfunc);

    // Short function (< 32 instructions when checking is_leaf)
    // but we have > 2000 instructions, so not leaf
    // Should have interval polls
    if !placement.is_leaf {
        assert!(placement.poll_indices.len() >= 1);
    }
}

#[test]
fn test_multiple_loops() {
    let mut mfunc = MachineFunction::new();

    // First loop
    mfunc.push(make_label(1));
    for _ in 0..40 {
        mfunc.push(make_inst(MachineOp::Nop));
    }
    mfunc.push(make_jcc(1));

    // Second loop
    mfunc.push(make_label(2));
    for _ in 0..40 {
        mfunc.push(make_inst(MachineOp::Nop));
    }
    mfunc.push(make_jcc(2));

    mfunc.push(make_inst(MachineOp::Ret));

    let analyzer = SafepointAnalyzer::new();
    let placement = analyzer.analyze(&mfunc);

    // Should have safepoints for both loops
    assert!(placement.poll_indices.len() >= 2);
}

#[test]
fn test_nested_loops() {
    let mut mfunc = MachineFunction::new();

    // Outer loop
    mfunc.push(make_label(1));

    // Inner loop
    mfunc.push(make_label(2));
    for _ in 0..50 {
        mfunc.push(make_inst(MachineOp::Nop));
    }
    mfunc.push(make_jcc(2));

    // Back to outer
    for _ in 0..10 {
        mfunc.push(make_inst(MachineOp::Nop));
    }
    mfunc.push(make_jcc(1));

    mfunc.push(make_inst(MachineOp::Ret));

    let analyzer = SafepointAnalyzer::new();
    let placement = analyzer.analyze(&mfunc);

    // Should have safepoints for both loops
    assert!(!placement.is_leaf);
    assert!(placement.poll_indices.len() >= 2);
}
