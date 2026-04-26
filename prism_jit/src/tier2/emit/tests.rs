use super::*;
use crate::ir::builder::{
    ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
};
use crate::regalloc::{AllocationMap, VReg};
use crate::tier2::lower::InstructionSelector;

#[test]
fn test_code_emitter_empty() {
    let mfunc = MachineFunction::new();
    let result = CodeEmitter::emit(&mfunc);
    assert!(result.is_ok());
}

#[test]
fn test_code_emitter_with_mov() {
    let mut mfunc = MachineFunction::new();
    mfunc.push(MachineInst::new(
        MachineOp::Mov,
        MachineOperand::gpr(Gpr::Rax),
        MachineOperand::Imm(42),
    ));
    mfunc.push(MachineInst::nullary(MachineOp::Ret));

    let result = CodeEmitter::emit(&mfunc);
    assert!(result.is_ok());

    let code = result.unwrap();
    assert!(!code.code.as_slice().is_empty());
}

#[test]
fn test_code_emitter_with_arithmetic() {
    let mut mfunc = MachineFunction::new();

    // MOV RAX, 10
    mfunc.push(MachineInst::new(
        MachineOp::Mov,
        MachineOperand::gpr(Gpr::Rax),
        MachineOperand::Imm(10),
    ));

    // MOV RBX, 5
    mfunc.push(MachineInst::new(
        MachineOp::Mov,
        MachineOperand::gpr(Gpr::Rbx),
        MachineOperand::Imm(5),
    ));

    // ADD RAX, RBX
    mfunc.push(MachineInst::binary(
        MachineOp::Add,
        MachineOperand::gpr(Gpr::Rax),
        MachineOperand::gpr(Gpr::Rax),
        MachineOperand::gpr(Gpr::Rbx),
    ));

    // RET
    mfunc.push(MachineInst::nullary(MachineOp::Ret));

    let result = CodeEmitter::emit(&mfunc);
    assert!(result.is_ok());
}

#[test]
fn test_stack_map_entry() {
    let entry = StackMapEntry {
        code_offset: 0x10,
        bc_offset: Some(12),
        gc_slots: vec![-8, -16],
        gc_regs: vec![Gpr::Rax, Gpr::Rbx],
    };
    assert_eq!(entry.bc_offset, Some(12));
    assert_eq!(entry.gc_slots.len(), 2);
    assert_eq!(entry.gc_regs.len(), 2);
}

#[test]
fn test_call_safepoint_uses_machine_gc_roots() {
    let mut mfunc = MachineFunction::new();
    mfunc.gc_roots.stack_slots = vec![-24, -8];
    mfunc.gc_roots.regs = vec![Gpr::Rbx, Gpr::R12];
    mfunc.push(MachineInst::new(
        MachineOp::Call,
        MachineOperand::Imm(0x1234),
        MachineOperand::None,
    ));
    mfunc.push(MachineInst::nullary(MachineOp::Ret));

    let code = CodeEmitter::emit(&mfunc).expect("emission should succeed");
    assert_eq!(code.stack_maps.len(), 1);
    assert_eq!(code.stack_maps[0].bc_offset, None);
    assert_eq!(code.stack_maps[0].gc_slots, vec![-24, -8]);
    assert_eq!(code.stack_maps[0].gc_regs, vec![Gpr::Rbx, Gpr::R12]);
}

#[test]
fn test_call_safepoint_records_origin_bc_offset() {
    use crate::ir::node::NodeId;

    let mut mfunc = MachineFunction::new();
    mfunc.node_bc_offsets = vec![None, Some(37)];
    mfunc.push(
        MachineInst::new(
            MachineOp::Call,
            MachineOperand::Imm(0x1234),
            MachineOperand::None,
        )
        .with_origin(NodeId::new(1)),
    );
    mfunc.push(MachineInst::nullary(MachineOp::Ret));

    let code = CodeEmitter::emit(&mfunc).expect("emission should succeed");
    assert_eq!(code.stack_maps.len(), 1);
    assert_eq!(code.stack_maps[0].bc_offset, Some(37));
}

#[test]
fn test_poll_safepoint_uses_machine_gc_roots() {
    let mut mfunc = MachineFunction::new();
    let loop_label = mfunc.new_label();
    mfunc.gc_roots.stack_slots = vec![-32];
    mfunc.gc_roots.regs = vec![Gpr::R13];
    mfunc.add_label(loop_label);
    mfunc.push(MachineInst::nullary(MachineOp::Nop));
    mfunc.push(MachineInst::new(
        MachineOp::Jmp,
        MachineOperand::Label(loop_label),
        MachineOperand::None,
    ));

    let code = CodeEmitter::emit_with_safepoint(&mfunc, Some(0x1000))
        .expect("safepoint-enabled emission should succeed");
    assert!(
        !code.stack_maps.is_empty(),
        "loop back-edge should produce at least one safepoint poll"
    );
    assert!(code.stack_maps.iter().all(|entry| {
        entry.gc_slots == vec![-32] && entry.gc_regs == vec![Gpr::R13] && entry.bc_offset.is_none()
    }));
}

#[test]
fn test_code_emitter_rejects_unresolved_vreg_operand() {
    let mut mfunc = MachineFunction::new();
    mfunc.push(MachineInst::new(
        MachineOp::Mov,
        MachineOperand::gpr(Gpr::Rax),
        MachineOperand::VReg(VReg::new(0)),
    ));
    mfunc.push(MachineInst::nullary(MachineOp::Ret));

    let err =
        CodeEmitter::emit(&mfunc).expect_err("unresolved virtual registers must fail emission");
    assert!(err.contains("unresolved virtual register"));
}

#[test]
fn test_code_emitter_rejects_unsupported_machine_op() {
    let mut mfunc = MachineFunction::new();
    mfunc.push(MachineInst::nullary(MachineOp::Vaddpd256));
    mfunc.push(MachineInst::nullary(MachineOp::Ret));

    let err = CodeEmitter::emit(&mfunc).expect_err("unsupported machine op must fail emission");
    assert!(err.contains("not supported"));
}

#[test]
fn test_full_pipeline() {
    // Build IR
    let mut builder = GraphBuilder::new(4, 0);
    let const_1 = builder.const_int(1);
    let const_2 = builder.const_int(2);
    let sum = builder.int_add(const_1, const_2);
    let _ret = builder.return_value(sum);
    let graph = builder.finish();

    // Instruction selection
    let alloc_map = AllocationMap::new();
    let mfunc = InstructionSelector::select(&graph, &alloc_map)
        .expect("instruction selection should succeed");

    // Code emission (will have virtual registers)
    // This is a smoke test - full regalloc integration would resolve VRegs
    assert!(!mfunc.insts.is_empty());
}

#[test]
fn test_cond_code_conversion() {
    assert_eq!(CondCode::E.to_condition(), Condition::Equal);
    assert_eq!(CondCode::L.to_condition(), Condition::Less);
    assert_eq!(CondCode::G.to_condition(), Condition::Greater);
}
