use super::*;
use crate::ir::builder::{
    ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
};
use crate::ir::node::InputList;

#[test]
fn test_condition_code_inverse() {
    assert_eq!(CondCode::E.inverse(), CondCode::Ne);
    assert_eq!(CondCode::L.inverse(), CondCode::Ge);
    assert_eq!(CondCode::G.inverse(), CondCode::Le);
}

#[test]
fn test_machine_inst_creation() {
    let inst = MachineInst::new(
        MachineOp::Mov,
        MachineOperand::gpr(Gpr::Rax),
        MachineOperand::Imm(42),
    );
    assert_eq!(inst.op, MachineOp::Mov);
    assert!(inst.dst.is_reg());
    assert!(inst.src1.is_imm());
}

#[test]
fn test_machine_function_labels() {
    let mut mfunc = MachineFunction::new();
    let l1 = mfunc.new_label();
    let l2 = mfunc.new_label();
    assert_ne!(l1, l2);
}

#[test]
fn test_instruction_selection_simple() {
    let mut builder = GraphBuilder::new(4, 0);
    let const_1 = builder.const_int(1);
    let const_2 = builder.const_int(2);
    let sum = builder.int_add(const_1, const_2);
    let _ret = builder.return_value(sum);
    let graph = builder.finish();

    let alloc_map = AllocationMap::new();
    let mfunc = InstructionSelector::select(&graph, &alloc_map)
        .expect("instruction selection should succeed");

    // Should have generated some instructions
    assert!(!mfunc.insts.is_empty());
}

#[test]
fn test_instruction_selection_materializes_parameters_from_frame_base() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).expect("parameter 0 should exist");
    let p1 = builder.parameter(1).expect("parameter 1 should exist");
    let _ret = builder.return_value(p1);
    let graph = builder.finish();

    let alloc_map = AllocationMap::new();
    let mfunc = InstructionSelector::select(&graph, &alloc_map)
        .expect("instruction selection should materialize parameters");

    let expected_p0_src = MachineOperand::Mem(MemOperand::base_disp(FRAME_BASE_SCRATCH_GPR, 0));
    let expected_p1_src = MachineOperand::Mem(MemOperand::base_disp(
        FRAME_BASE_SCRATCH_GPR,
        VM_REGISTER_SLOT_SIZE,
    ));

    assert!(mfunc.insts.iter().any(|inst| inst.origin == Some(p0)
        && inst.op == MachineOp::Mov
        && inst.src1 == expected_p0_src));
    assert!(mfunc.insts.iter().any(|inst| inst.origin == Some(p1)
        && inst.op == MachineOp::Mov
        && inst.src1 == expected_p1_src));
}

#[test]
fn test_instruction_selection_materializes_spilled_parameters_before_direct_regs() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).expect("parameter 0 should exist");
    let p1 = builder.parameter(1).expect("parameter 1 should exist");
    let _ret = builder.return_value(p1);
    let graph = builder.finish();

    let mut alloc_map = AllocationMap::new();
    let spill = alloc_map.alloc_spill_slot();
    alloc_map.set(VReg::new(0), Allocation::Spill(spill));
    alloc_map.set(VReg::new(1), Allocation::Register(PReg::Gpr(Gpr::Rcx)));

    let mfunc = InstructionSelector::select(&graph, &alloc_map)
        .expect("instruction selection should materialize spilled parameters");

    let first_p0_inst = mfunc
        .insts
        .iter()
        .position(|inst| inst.origin == Some(p0))
        .expect("parameter 0 materialization should emit instructions");
    let first_p1_inst = mfunc
        .insts
        .iter()
        .position(|inst| inst.origin == Some(p1))
        .expect("parameter 1 materialization should emit instructions");
    assert!(
        first_p0_inst < first_p1_inst,
        "spilled parameters must be materialized before direct-register parameters"
    );

    assert!(mfunc.insts.iter().any(|inst| {
        inst.origin == Some(p0)
            && inst.op == MachineOp::Mov
            && inst.dst == MachineOperand::gpr(PARAM_STACK_TEMP_GPR)
            && inst.src1 == MachineOperand::Mem(MemOperand::base_disp(FRAME_BASE_SCRATCH_GPR, 0))
    }));
    assert!(mfunc.insts.iter().any(|inst| {
        inst.origin == Some(p0)
            && inst.op == MachineOp::Mov
            && inst.dst == MachineOperand::StackSlot(spill.offset())
            && inst.src1 == MachineOperand::gpr(PARAM_STACK_TEMP_GPR)
    }));
}

#[test]
fn test_instruction_selection_rejects_unsupported_int_pow() {
    let mut graph = Graph::new();
    let c1 = graph.add_node(Operator::ConstInt(2), InputList::Single(graph.start));
    let c2 = graph.add_node(Operator::ConstInt(3), InputList::Single(graph.start));
    let pow = graph.add_node(
        Operator::IntOp(ArithOp::Pow),
        InputList::from_slice(&[c1, c2]),
    );
    let _ret = graph.add_node(
        Operator::Control(ControlOp::Return),
        InputList::from_slice(&[graph.start, pow]),
    );

    let alloc_map = AllocationMap::new();
    let err = InstructionSelector::select(&graph, &alloc_map)
        .expect_err("unsupported Pow lowering should return error");
    assert!(err.contains("does not support"));
}

#[test]
fn test_instruction_selection_lowers_if_with_projection_targets() {
    let mut graph = Graph::new();
    let cond = graph.add_node(Operator::ConstBool(true), InputList::Single(graph.start));
    let if_node = graph.add_node(
        Operator::Control(ControlOp::If),
        InputList::from_slice(&[graph.start, cond]),
    );
    let true_proj = graph.add_node(Operator::Projection(0), InputList::Single(if_node));
    let false_proj = graph.add_node(Operator::Projection(1), InputList::Single(if_node));
    let true_val = graph.add_node(Operator::ConstInt(1), InputList::Empty);
    let false_val = graph.add_node(Operator::ConstInt(0), InputList::Empty);
    let true_ret = graph.add_node(
        Operator::Control(ControlOp::Return),
        InputList::from_slice(&[true_proj, true_val]),
    );
    let false_ret = graph.add_node(
        Operator::Control(ControlOp::Return),
        InputList::from_slice(&[false_proj, false_val]),
    );

    let alloc_map = AllocationMap::new();
    let mfunc = InstructionSelector::select(&graph, &alloc_map)
        .expect("If control with projections should lower to machine branches");

    assert!(mfunc.insts.iter().any(|inst| {
        inst.op == MachineOp::Test && inst.origin == Some(if_node) && inst.src1 == inst.src2
    }));
    assert!(mfunc.insts.iter().any(|inst| {
        inst.op == MachineOp::Jcc && inst.origin == Some(if_node) && inst.cc == Some(CondCode::Ne)
    }));
    assert!(mfunc.insts.iter().any(|inst| {
        inst.op == MachineOp::Jmp
            && inst.origin == Some(if_node)
            && matches!(inst.dst, MachineOperand::Label(_))
    }));
    assert!(
        mfunc
            .insts
            .iter()
            .any(|inst| inst.op == MachineOp::Label && inst.origin.is_none()),
        "branch targets should be materialized as labels",
    );
    assert!(
        mfunc
            .insts
            .iter()
            .any(|inst| inst.op == MachineOp::Ret && inst.origin == Some(true_ret))
    );
    assert!(
        mfunc
            .insts
            .iter()
            .any(|inst| inst.op == MachineOp::Ret && inst.origin == Some(false_ret))
    );
}

#[test]
fn test_instruction_selection_materializes_labels_for_projection_region_targets() {
    let mut graph = Graph::new();
    let cond = graph.add_node(Operator::ConstBool(true), InputList::Single(graph.start));
    let if_node = graph.add_node(
        Operator::Control(ControlOp::If),
        InputList::from_slice(&[graph.start, cond]),
    );
    let true_proj = graph.add_node(Operator::Projection(0), InputList::Single(if_node));
    let false_proj = graph.add_node(Operator::Projection(1), InputList::Single(if_node));
    let merge = graph.add_node(
        Operator::Control(ControlOp::Region),
        InputList::from_slice(&[true_proj, false_proj]),
    );
    let ret_val = graph.add_node(Operator::ConstInt(1), InputList::Empty);
    let _ret = graph.add_node(
        Operator::Control(ControlOp::Return),
        InputList::from_slice(&[merge, ret_val]),
    );

    let alloc_map = AllocationMap::new();
    let mfunc = InstructionSelector::select(&graph, &alloc_map)
        .expect("region targets fed by projections should still be scheduled and labeled");

    let mut referenced = std::collections::BTreeSet::new();
    let mut defined = std::collections::BTreeSet::new();
    for inst in &mfunc.insts {
        match inst.op {
            MachineOp::Jmp | MachineOp::Jcc => {
                if let MachineOperand::Label(id) = inst.dst {
                    referenced.insert(id);
                }
            }
            MachineOp::Label => {
                if let MachineOperand::Label(id) = inst.dst {
                    defined.insert(id);
                }
            }
            _ => {}
        }
    }

    assert!(
        !referenced.is_empty(),
        "test must generate at least one branch target label"
    );
    for label in referenced {
        assert!(
            defined.contains(&label),
            "branch references label {label}, but no LABEL pseudo-instruction defines it"
        );
    }
}

#[test]
fn test_instruction_selection_rejects_projection_not_from_if() {
    let mut graph = Graph::new();
    let _bad_proj = graph.add_node(Operator::Projection(0), InputList::Single(graph.start));

    let alloc_map = AllocationMap::new();
    let err = InstructionSelector::select(&graph, &alloc_map)
        .expect_err("projection not sourced from If must fail lowering");
    assert!(err.contains("projections from If"));
}

#[test]
fn test_instruction_selection_rejects_if_without_projection_edges() {
    let mut graph = Graph::new();
    let cond = graph.add_node(Operator::ConstBool(true), InputList::Single(graph.start));
    let _if_node = graph.add_node(
        Operator::Control(ControlOp::If),
        InputList::from_slice(&[graph.start, cond]),
    );

    let alloc_map = AllocationMap::new();
    let err = InstructionSelector::select(&graph, &alloc_map)
        .expect_err("If lowering must fail without any live projection targets");
    assert!(err.contains("projection-based control targets"));
}

#[test]
fn test_instruction_selection_collects_machine_gc_root_metadata() {
    let mut builder = GraphBuilder::new(3, 1);
    let p0 = builder.parameter(0).expect("parameter should exist");
    let c1 = builder.const_int(1);
    let sum = builder.int_add(p0, c1);
    let _ret = builder.return_value(sum);
    let graph = builder.finish();

    let mut alloc_map = AllocationMap::new();
    let spill = alloc_map.alloc_spill_slot();
    alloc_map.set(VReg::new(0), Allocation::Register(PReg::Gpr(Gpr::Rbx)));
    alloc_map.set(VReg::new(1), Allocation::Spill(spill));
    alloc_map.set(VReg::new(2), Allocation::Register(PReg::Gpr(Gpr::Rbx)));

    let mfunc = InstructionSelector::select(&graph, &alloc_map)
        .expect("instruction selection should collect machine GC roots");

    assert_eq!(mfunc.gc_roots.regs, vec![Gpr::Rbx]);
    assert_eq!(mfunc.gc_roots.stack_slots, vec![spill.offset()]);
}

#[test]
fn test_instruction_selection_materializes_spilled_divisor_for_floor_div() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).expect("parameter 0 should exist");
    let p1 = builder.parameter(1).expect("parameter 1 should exist");
    let q = builder.int_div(p0, p1);
    let _ret = builder.return_value(q);
    let graph = builder.finish();

    let mut alloc_map = AllocationMap::new();
    let spill = alloc_map.alloc_spill_slot();
    alloc_map.set(VReg::new(0), Allocation::Register(PReg::Gpr(Gpr::Rax)));
    alloc_map.set(VReg::new(1), Allocation::Spill(spill));

    let mfunc = InstructionSelector::select(&graph, &alloc_map)
        .expect("floor-div lowering should support spilled divisors");

    let floor_div_node = q;
    assert!(mfunc.insts.iter().any(|inst| {
        inst.origin == Some(floor_div_node)
            && inst.op == MachineOp::Mov
            && inst.dst == MachineOperand::gpr(FRAME_BASE_SCRATCH_GPR)
            && inst.src1 == MachineOperand::StackSlot(spill.offset())
    }));
    assert!(mfunc.insts.iter().any(|inst| {
        inst.origin == Some(floor_div_node)
            && inst.op == MachineOp::Idiv
            && inst.src1 == MachineOperand::gpr(FRAME_BASE_SCRATCH_GPR)
    }));
}

#[test]
fn test_instruction_selection_emits_float_neg_with_zero_minus_operand() {
    let mut builder = GraphBuilder::new(2, 0);
    let val = builder.const_float(3.5);
    let neg = builder.float_neg(val);
    let _ret = builder.return_value(neg);
    let graph = builder.finish();

    let alloc_map = AllocationMap::new();
    let mfunc =
        InstructionSelector::select(&graph, &alloc_map).expect("float neg lowering should succeed");

    assert!(
        mfunc
            .insts
            .iter()
            .any(|inst| inst.origin == Some(neg) && inst.op == MachineOp::Xorpd)
    );
    assert!(
        mfunc
            .insts
            .iter()
            .any(|inst| inst.origin == Some(neg) && inst.op == MachineOp::Subsd)
    );
}

#[test]
fn test_cond_code_from_cmp_op() {
    assert_eq!(CondCode::from_cmp_op(CmpOp::Eq, true), CondCode::E);
    assert_eq!(CondCode::from_cmp_op(CmpOp::Lt, true), CondCode::L);
    assert_eq!(CondCode::from_cmp_op(CmpOp::Lt, false), CondCode::B);
}
