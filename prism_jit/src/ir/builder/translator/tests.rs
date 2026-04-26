use super::*;
use crate::ir::operators::{ArithOp, BitwiseOp, CallKind, CmpOp, Operator};
use prism_code::Register;

#[test]
fn test_translate_simple_return_is_ok() {
    let mut code = CodeObject::new("simple", "<test>");
    code.register_count = 1;
    code.instructions =
        vec![Instruction::op_d(Opcode::ReturnNone, Register::new(0))].into_boxed_slice();

    let builder = GraphBuilder::new(1, 0);
    let translator = BytecodeTranslator::new(builder, &code);
    let graph = translator.translate();
    assert!(graph.is_ok());
}

#[test]
fn test_translate_rejects_unsupported_opcode() {
    let mut code = CodeObject::new("unsupported", "<test>");
    code.register_count = 1;
    code.instructions =
        vec![Instruction::op_d(Opcode::BuildClass, Register::new(0))].into_boxed_slice();

    let builder = GraphBuilder::new(1, 0);
    let translator = BytecodeTranslator::new(builder, &code);
    let err = translator.translate().unwrap_err();
    assert!(err.contains("unsupported opcode"));
}

#[test]
fn test_translate_rejects_invalid_constant_index() {
    let mut code = CodeObject::new("bad_const", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 99),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(1, 0);
    let translator = BytecodeTranslator::new(builder, &code);
    let err = translator.translate().unwrap_err();
    assert!(err.contains("invalid constant index"));
}

#[test]
fn test_translate_rejects_jump_target_underflow() {
    let mut code = CodeObject::new("bad_jump_underflow", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        Instruction::op_di(Opcode::Jump, Register::new(0), (-2_i16) as u16),
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(1, 0);
    let translator = BytecodeTranslator::new(builder, &code);
    let err = translator.translate().unwrap_err();
    assert!(err.contains("invalid jump target"));
}

#[test]
fn test_translate_rejects_jump_target_overflow() {
    let mut code = CodeObject::new("bad_jump_overflow", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        Instruction::op_di(Opcode::Jump, Register::new(0), 10),
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(1, 0);
    let translator = BytecodeTranslator::new(builder, &code);
    let err = translator.translate().unwrap_err();
    assert!(err.contains("invalid jump target"));
}

#[test]
fn test_translate_accepts_jump_to_end() {
    let mut code = CodeObject::new("jump_to_end", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        // offset 0 -> next is 1, rel 1 => target 2 (end sentinel)
        Instruction::op_di(Opcode::Jump, Register::new(0), 1),
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(1, 0);
    let translator = BytecodeTranslator::new(builder, &code);
    assert!(translator.translate().is_ok());
}

#[test]
fn test_translate_rejects_invalid_opcode_byte() {
    let mut code = CodeObject::new("invalid_opcode", "<test>");
    code.register_count = 1;
    code.instructions = vec![Instruction::from_raw(0xFF00_0000)].into_boxed_slice();

    let builder = GraphBuilder::new(1, 0);
    let translator = BytecodeTranslator::new(builder, &code);
    let err = translator.translate().unwrap_err();
    assert!(err.contains("invalid opcode byte"));
}

#[test]
fn test_translate_rejects_uninitialized_register_read() {
    let mut code = CodeObject::new("uninitialized_read", "<test>");
    code.register_count = 3;
    code.instructions = vec![
        Instruction::op_dss(
            Opcode::Add,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(3, 0);
    let translator = BytecodeTranslator::new(builder, &code);
    let err = translator.translate().unwrap_err();
    assert!(err.contains("uninitialized register"));
    assert!(err.contains("r0") || err.contains("r1"));
}

#[test]
fn test_translate_reads_argument_registers_from_parameters() {
    let mut code = CodeObject::new("arg_add", "<test>");
    code.register_count = 2;
    code.arg_count = 2;
    code.instructions = vec![
        Instruction::op_dss(
            Opcode::Add,
            Register::new(0),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(2, 2);
    let translator = BytecodeTranslator::new(builder, &code);
    let graph = translator
        .translate()
        .expect("argument registers should be seeded");

    let mut found_arg_add = false;
    for (_, node) in graph.iter() {
        if let Operator::GenericOp(ArithOp::Add) = node.op {
            let lhs = graph.node(node.inputs.get(0).unwrap()).op;
            let rhs = graph.node(node.inputs.get(1).unwrap()).op;
            assert!(matches!(lhs, Operator::Parameter(0)));
            assert!(matches!(rhs, Operator::Parameter(1)));
            found_arg_add = true;
            break;
        }
    }

    assert!(
        found_arg_add,
        "expected translated add node using parameter inputs"
    );
}

#[test]
fn test_translate_call_method_uses_method_self_and_explicit_args() {
    let mut code = CodeObject::new("call_method", "<test>");
    code.register_count = 4;
    code.arg_count = 4; // r0..r3 seeded as parameters
    code.instructions = vec![
        // r0 = r1(r2, r3)
        Instruction::new(Opcode::CallMethod, 0, 1, 1),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(4, 4);
    let translator = BytecodeTranslator::new(builder, &code);
    let graph = translator
        .translate()
        .expect("call method translation should succeed");

    let mut call_verified = false;
    for (_, node) in graph.iter() {
        if let Operator::Call(CallKind::Direct) = node.op {
            assert_eq!(node.inputs.len(), 3);
            let func_op = graph.node(node.inputs.get(0).unwrap()).op;
            let self_op = graph.node(node.inputs.get(1).unwrap()).op;
            let arg0_op = graph.node(node.inputs.get(2).unwrap()).op;
            assert!(matches!(func_op, Operator::Parameter(1)));
            assert!(matches!(self_op, Operator::Parameter(2)));
            assert!(matches!(arg0_op, Operator::Parameter(3)));
            call_verified = true;
            break;
        }
    }

    assert!(call_verified, "expected translated CallMethod call node");
}

#[test]
fn test_translate_call_method_rejects_uninitialized_self_slot() {
    let mut code = CodeObject::new("call_method_uninit_self", "<test>");
    code.register_count = 4;
    code.arg_count = 2; // r2 (self slot) is uninitialized
    code.instructions = vec![
        Instruction::new(Opcode::CallMethod, 0, 1, 0),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(4, 2);
    let translator = BytecodeTranslator::new(builder, &code);
    let err = translator.translate().unwrap_err();
    assert!(err.contains("uninitialized register"));
    assert!(err.contains("r2"));
}

#[test]
fn test_translate_jump_if_none_and_not_none_use_int_cmp_against_none() {
    let mut code = CodeObject::new("jump_if_none", "<test>");
    code.register_count = 1;
    code.arg_count = 1;
    code.instructions = vec![
        Instruction::op_di(Opcode::JumpIfNone, Register::new(0), 1),
        Instruction::op_di(Opcode::JumpIfNotNone, Register::new(0), 1),
        Instruction::op_d(Opcode::Return, Register::new(0)),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(1, 1);
    let translator = BytecodeTranslator::new(builder, &code);
    let graph = translator.translate().expect("translation should succeed");

    assert!(
        graph
            .iter()
            .any(|(_, node)| matches!(node.op, Operator::IntCmp(CmpOp::Eq))),
        "JumpIfNone should lower through IntCmp(Eq) against ConstNone"
    );
    assert!(
        graph
            .iter()
            .any(|(_, node)| matches!(node.op, Operator::IntCmp(CmpOp::Ne))),
        "JumpIfNotNone should lower through IntCmp(Ne) against ConstNone"
    );
    assert!(
        graph
            .iter()
            .any(|(_, node)| matches!(node.op, Operator::ConstNone)),
        "None checks should materialize a ConstNone node"
    );
}

#[test]
fn test_translate_load_store_local_updates_register_aliases() {
    let mut code = CodeObject::new("local_alias", "<test>");
    code.register_count = 3;
    code.arg_count = 1;
    code.instructions = vec![
        Instruction::op_ds(Opcode::Move, Register::new(1), Register::new(0)),
        // Store r1 into local/register slot 2
        Instruction::op_di(Opcode::StoreLocal, Register::new(1), 2),
        // Load slot 2 back into r0
        Instruction::op_di(Opcode::LoadLocal, Register::new(0), 2),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(3, 1);
    let translator = BytecodeTranslator::new(builder, &code);
    let graph = translator.translate().expect("translation should succeed");

    let mut saw_parameter_return = false;
    for (_, node) in graph.iter() {
        if let Operator::Control(crate::ir::operators::ControlOp::Return) = node.op {
            let value = node.inputs.get(1).expect("return must carry value input");
            if matches!(graph.node(value).op, Operator::Parameter(0)) {
                saw_parameter_return = true;
                break;
            }
        }
    }
    assert!(
        saw_parameter_return,
        "local load/store aliasing should preserve the underlying SSA value"
    );
}

#[test]
fn test_translate_typed_arithmetic_and_bitwise_opcodes() {
    let mut code = CodeObject::new("typed_ops", "<test>");
    code.register_count = 12;
    code.arg_count = 2;
    code.instructions = vec![
        Instruction::op_dss(
            Opcode::FloorDivInt,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_dss(
            Opcode::ModInt,
            Register::new(3),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_ds(Opcode::NegInt, Register::new(4), Register::new(0)),
        Instruction::op_ds(Opcode::PosInt, Register::new(5), Register::new(0)),
        Instruction::op_dss(
            Opcode::DivFloat,
            Register::new(6),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_ds(Opcode::NegFloat, Register::new(7), Register::new(0)),
        Instruction::op_dss(
            Opcode::BitwiseAnd,
            Register::new(8),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_dss(
            Opcode::BitwiseOr,
            Register::new(9),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_dss(
            Opcode::BitwiseXor,
            Register::new(10),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_ds(Opcode::BitwiseNot, Register::new(11), Register::new(0)),
        Instruction::op_dss(
            Opcode::Shl,
            Register::new(2),
            Register::new(2),
            Register::new(1),
        ),
        Instruction::op_dss(
            Opcode::Shr,
            Register::new(3),
            Register::new(3),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let builder = GraphBuilder::new(12, 2);
    let translator = BytecodeTranslator::new(builder, &code);
    let graph = translator
        .translate()
        .expect("typed opcode translation should succeed");

    let has_op = |pred: fn(Operator) -> bool| graph.iter().any(|(_, node)| pred(node.op));
    assert!(has_op(|op| op == Operator::IntOp(ArithOp::FloorDiv)));
    assert!(has_op(|op| op == Operator::IntOp(ArithOp::Mod)));
    assert!(has_op(|op| op == Operator::IntOp(ArithOp::Neg)));
    assert!(has_op(|op| op == Operator::FloatOp(ArithOp::TrueDiv)));
    assert!(has_op(|op| op == Operator::FloatOp(ArithOp::Neg)));
    assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::And)));
    assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Or)));
    assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Xor)));
    assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Not)));
    assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Shl)));
    assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Shr)));
}
