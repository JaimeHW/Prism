use super::*;
use prism_code::{Constant, Instruction, Register};
use prism_core::Value;

fn boxed_constants(constants: Vec<Value>) -> Box<[Constant]> {
    constants
        .into_iter()
        .map(Constant::Value)
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

#[test]
fn test_lower_code_to_templates_rejects_invalid_jump_target() {
    let mut code = CodeObject::new("invalid_jump", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        Instruction::op_di(Opcode::Jump, Register::new(0), 32_767),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let err = lower_code_to_templates(&code).unwrap_err();
    assert!(err.contains("invalid jump target"));
}

#[test]
fn test_lower_code_to_templates_rejects_unlowered_opcode() {
    let mut code = CodeObject::new("unsupported", "<test>");
    code.register_count = 2;
    code.instructions = vec![
        Instruction::op_dss(
            Opcode::CallMethod,
            Register::new(0),
            Register::new(1),
            Register::new(0),
        ),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let err = lower_code_to_templates(&code).unwrap_err();
    assert!(err.contains("unsupported opcode"));
    assert!(err.contains("CallMethod"));
}

#[test]
fn test_lower_code_to_templates_rejects_invalid_constant_index() {
    let mut code = CodeObject::new("bad_const_idx", "<test>");
    code.register_count = 1;
    code.constants = boxed_constants(vec![Value::int(1).unwrap()]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 3),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let err = lower_code_to_templates(&code).unwrap_err();
    assert!(err.contains("invalid constant index"));
}

#[test]
fn test_lower_code_to_templates_accepts_simple_function() {
    let mut code = CodeObject::new("simple", "<test>");
    code.register_count = 1;
    code.constants = boxed_constants(vec![Value::int(7).unwrap()]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(!templates.is_empty());
}

#[test]
fn test_lower_code_to_templates_maps_return_none() {
    let mut code = CodeObject::new("ret_none", "<test>");
    code.register_count = 0;
    code.instructions = vec![Instruction::op(Opcode::ReturnNone)].into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert_eq!(templates.len(), 1);
    assert!(matches!(
        templates[0],
        TemplateInstruction::ReturnNone { bc_offset: 0 }
    ));
}

#[test]
fn test_lower_code_to_templates_maps_primitive_load_consts() {
    let mut code = CodeObject::new("consts", "<test>");
    code.register_count = 4;
    code.constants = boxed_constants(vec![
        Value::none(),
        Value::bool(true),
        Value::int(7).unwrap(),
        Value::float(3.5),
    ]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_di(Opcode::LoadConst, Register::new(2), 2),
        Instruction::op_di(Opcode::LoadConst, Register::new(3), 3),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[0],
        TemplateInstruction::LoadNone {
            bc_offset: 0,
            dst: 0
        }
    ));
    assert!(matches!(
        templates[1],
        TemplateInstruction::LoadBool {
            bc_offset: 4,
            dst: 1,
            value: true
        }
    ));
    assert!(matches!(
        templates[2],
        TemplateInstruction::LoadInt {
            bc_offset: 8,
            dst: 2,
            value: 7
        }
    ));
    assert!(matches!(
        templates[3],
        TemplateInstruction::LoadFloat {
            bc_offset: 12,
            dst: 3,
            value
        } if (value - 3.5).abs() < f64::EPSILON
    ));
}

#[test]
fn test_lower_code_to_templates_computes_relative_jump_targets() {
    let mut code = CodeObject::new("jump_relative", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        // offset 0 -> next_pc 4 + (1 * 4) => 8
        Instruction::op_di(Opcode::Jump, Register::new(0), 1),
        Instruction::op_d(Opcode::LoadNone, Register::new(0)),
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[0],
        TemplateInstruction::Jump {
            bc_offset: 0,
            target: 8
        }
    ));
}

#[test]
fn test_lower_code_to_templates_rejects_jump_underflow() {
    let mut code = CodeObject::new("jump_underflow", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        // offset 0 -> next_pc 4 + (-2 * 4) => -4 (invalid)
        Instruction::op_di(Opcode::Jump, Register::new(0), (-2_i16) as u16),
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();

    let err = lower_code_to_templates(&code).unwrap_err();
    assert!(err.contains("jump target overflow"));
}

#[test]
fn test_lower_code_to_templates_allows_jump_to_end_sentinel() {
    let mut code = CodeObject::new("jump_end", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        // offset 0 -> next_pc 4 + (1 * 4) => 8 == max_offset
        Instruction::op_di(Opcode::Jump, Register::new(0), 1),
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[0],
        TemplateInstruction::Jump {
            bc_offset: 0,
            target: 8
        }
    ));
}

#[test]
fn test_lower_generic_add_int_specializes_to_int_add() {
    let mut code = CodeObject::new("add_int", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::int(10).unwrap(), Value::int(32).unwrap()]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::Add,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[2],
        TemplateInstruction::IntAdd {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1
        }
    ));
}

#[test]
fn test_lower_generic_add_float_specializes_to_float_add() {
    let mut code = CodeObject::new("add_float", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::float(1.5), Value::float(2.5)]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::Add,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[2],
        TemplateInstruction::FloatAdd {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1
        }
    ));
}

#[test]
fn test_lower_generic_add_rejects_mixed_numeric_types() {
    let mut code = CodeObject::new("add_mixed", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::int(1).unwrap(), Value::float(2.0)]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::Add,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let err = lower_code_to_templates(&code).unwrap_err();
    assert!(err.contains("monomorphic numeric operands"));
    assert!(err.contains("Add"));
}

#[test]
fn test_lower_generic_floor_div_int_specializes_to_int_div() {
    let mut code = CodeObject::new("floordiv_int", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::int(9).unwrap(), Value::int(2).unwrap()]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::FloorDiv,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[2],
        TemplateInstruction::IntDiv {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1
        }
    ));
}

#[test]
fn test_lower_generic_floor_div_float_specializes_to_float_floor_div() {
    let mut code = CodeObject::new("floordiv_float", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::float(9.0), Value::float(2.0)]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::FloorDiv,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[2],
        TemplateInstruction::FloatFloorDiv {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1
        }
    ));
}

#[test]
fn test_lower_generic_mod_float_specializes_to_float_mod() {
    let mut code = CodeObject::new("mod_float", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::float(9.5), Value::float(2.0)]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::Mod,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[2],
        TemplateInstruction::FloatMod {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1
        }
    ));
}

#[test]
fn test_lower_true_div_int_rejected() {
    let mut code = CodeObject::new("truediv_int", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::int(7).unwrap(), Value::int(2).unwrap()]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::TrueDiv,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let err = lower_code_to_templates(&code).unwrap_err();
    assert!(err.contains("does not support int TrueDiv"));
}

#[test]
fn test_lower_true_div_float_specializes_to_float_div() {
    let mut code = CodeObject::new("truediv_float", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::float(7.0), Value::float(2.0)]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::TrueDiv,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[2],
        TemplateInstruction::FloatDiv {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1
        }
    ));
}

#[test]
fn test_lower_generic_gt_float_specializes_to_float_gt() {
    let mut code = CodeObject::new("gt_float", "<test>");
    code.register_count = 3;
    code.constants = boxed_constants(vec![Value::float(3.0), Value::float(2.0)]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::Gt,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[2],
        TemplateInstruction::FloatGt {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1
        }
    ));
}

#[test]
fn test_lower_move_propagates_known_type() {
    let mut code = CodeObject::new("move_type_prop", "<test>");
    code.register_count = 4;
    code.constants = boxed_constants(vec![Value::int(5).unwrap()]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_ds(Opcode::Move, Register::new(1), Register::new(0)),
        Instruction::op_dss(
            Opcode::Add,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[2],
        TemplateInstruction::IntAdd {
            bc_offset: 8,
            dst: 2,
            lhs: 1,
            rhs: 0
        }
    ));
}

#[test]
fn test_lower_load_true_false_opcodes() {
    let mut code = CodeObject::new("bool_load_ops", "<test>");
    code.register_count = 2;
    code.instructions = vec![
        Instruction::op_d(Opcode::LoadTrue, Register::new(0)),
        Instruction::op_d(Opcode::LoadFalse, Register::new(1)),
        Instruction::op_d(Opcode::Return, Register::new(1)),
    ]
    .into_boxed_slice();

    let templates = lower_code_to_templates(&code).unwrap();
    assert!(matches!(
        templates[0],
        TemplateInstruction::LoadBool {
            bc_offset: 0,
            dst: 0,
            value: true
        }
    ));
    assert!(matches!(
        templates[1],
        TemplateInstruction::LoadBool {
            bc_offset: 4,
            dst: 1,
            value: false
        }
    ));
}
