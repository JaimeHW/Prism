use super::*;
use crate::instruction::{Opcode, Register};

#[test]
fn test_code_flags() {
    let flags = CodeFlags::GENERATOR | CodeFlags::NESTED;
    assert!(flags.contains(CodeFlags::GENERATOR));
    assert!(flags.contains(CodeFlags::NESTED));
    assert!(!flags.contains(CodeFlags::COROUTINE));
}

#[test]
fn test_code_flags_from_bits_accepts_known_flags() {
    let bits = (CodeFlags::GENERATOR | CodeFlags::MODULE).bits();
    let flags = CodeFlags::from_bits(bits).expect("known flag set should decode");
    assert!(flags.contains(CodeFlags::GENERATOR));
    assert!(flags.contains(CodeFlags::MODULE));
}

#[test]
fn test_code_flags_from_bits_rejects_unknown_flags() {
    assert!(CodeFlags::from_bits(CodeFlags::ALL_BITS | (1 << 31)).is_none());
}

#[test]
fn test_code_object_new() {
    let code = CodeObject::new("test_func", "test.py");
    assert_eq!(&*code.name, "test_func");
    assert_eq!(&*code.filename, "test.py");
    assert_eq!(code.instructions.len(), 0);
}

#[test]
fn test_validate_accepts_well_formed_extended_name_operand() {
    let mut code = CodeObject::new("attr", "test.py");
    code.names = (0..=0x0123)
        .map(|index| Arc::<str>::from(format!("name_{index}")))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    code.instructions = vec![
        Instruction::new(Opcode::GetAttr, 0, 1, u8::MAX),
        Instruction::op_di(Opcode::AttrName, Register::new(0), 0x0123),
    ]
    .into_boxed_slice();

    assert!(code.validate().is_ok());
}

#[test]
fn test_validate_rejects_invalid_opcode() {
    let mut code = CodeObject::new("bad", "test.py");
    code.instructions = vec![Instruction::from_raw(0xFF00_0000)].into_boxed_slice();

    assert!(matches!(
        code.validate(),
        Err(CodeValidationError {
            kind: CodeValidationErrorKind::InvalidOpcode { opcode: 0xFF },
            ..
        })
    ));
}

#[test]
fn test_validate_rejects_constant_index_out_of_bounds() {
    let mut code = CodeObject::new("bad_const", "test.py");
    code.instructions =
        vec![Instruction::op_di(Opcode::LoadConst, Register::new(0), 7)].into_boxed_slice();

    assert!(matches!(
        code.validate(),
        Err(CodeValidationError {
            kind: CodeValidationErrorKind::PoolIndexOutOfBounds {
                pool: "constant",
                index: 7,
                len: 0,
            },
            ..
        })
    ));
}

#[test]
fn test_validate_rejects_missing_attr_name_extension() {
    let mut code = CodeObject::new("bad_attr", "test.py");
    code.instructions = vec![Instruction::new(Opcode::GetAttr, 0, 1, u8::MAX)].into_boxed_slice();

    assert!(matches!(
        code.validate(),
        Err(CodeValidationError {
            kind: CodeValidationErrorKind::MissingExtension {
                opcode: Opcode::GetAttr,
                extension: Opcode::AttrName,
            },
            ..
        })
    ));
}

#[test]
fn test_validate_rejects_jump_out_of_bounds() {
    let mut code = CodeObject::new("bad_jump", "test.py");
    code.instructions =
        vec![Instruction::op_di(Opcode::Jump, Register::new(0), 10u16)].into_boxed_slice();

    assert!(matches!(
        code.validate(),
        Err(CodeValidationError {
            kind: CodeValidationErrorKind::JumpTargetOutOfBounds { .. },
            ..
        })
    ));
}

#[test]
fn test_line_table_lookup() {
    let mut code = CodeObject::new("test", "test.py");
    code.line_table = vec![
        LineTableEntry {
            start_pc: 0,
            end_pc: 5,
            line: 10,
        },
        LineTableEntry {
            start_pc: 5,
            end_pc: 10,
            line: 15,
        },
    ]
    .into_boxed_slice();

    assert_eq!(code.line_for_pc(0), Some(10));
    assert_eq!(code.line_for_pc(4), Some(10));
    assert_eq!(code.line_for_pc(5), Some(15));
    assert_eq!(code.line_for_pc(9), Some(15));
    assert_eq!(code.line_for_pc(10), None);
}

#[test]
fn test_code_positions_follow_instruction_line_ranges() {
    let mut code = CodeObject::new("test", "test.py");
    code.instructions = vec![
        Instruction::op(Opcode::Nop),
        Instruction::op(Opcode::Nop),
        Instruction::op(Opcode::Nop),
        Instruction::op(Opcode::Nop),
    ]
    .into_boxed_slice();
    code.line_table = vec![
        LineTableEntry {
            start_pc: 0,
            end_pc: 1,
            line: 10,
        },
        LineTableEntry {
            start_pc: 1,
            end_pc: 4,
            line: 14,
        },
    ]
    .into_boxed_slice();

    let positions: Vec<_> = code.positions().collect();
    assert_eq!(
        positions,
        vec![
            (Some(10), Some(10), None, None),
            (Some(14), Some(14), None, None),
            (Some(14), Some(14), None, None),
            (Some(14), Some(14), None, None),
        ]
    );
}

#[test]
fn test_code_position_defaults_to_unknown_when_line_is_missing() {
    let mut code = CodeObject::new("test", "test.py");
    code.instructions = vec![Instruction::op(Opcode::Nop)].into_boxed_slice();

    assert_eq!(code.position_for_pc(0), (None, None, None, None));
}
