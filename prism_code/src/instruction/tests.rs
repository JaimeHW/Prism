use super::*;

#[test]
fn test_instruction_encoding() {
    let inst = Instruction::new(Opcode::Add, 5, 10, 15);
    assert_eq!(inst.opcode(), Opcode::Add as u8);
    assert_eq!(inst.dst(), Register(5));
    assert_eq!(inst.src1(), Register(10));
    assert_eq!(inst.src2(), Register(15));
}

#[test]
fn test_instruction_imm16() {
    let inst = Instruction::op_di(Opcode::LoadConst, Register(3), 0x1234);
    assert_eq!(inst.opcode(), Opcode::LoadConst as u8);
    assert_eq!(inst.dst(), Register(3));
    assert_eq!(inst.imm16(), 0x1234);
}

#[test]
fn test_instruction_size() {
    assert_eq!(std::mem::size_of::<Instruction>(), 4);
}

#[test]
fn test_opcode_from_u8() {
    assert_eq!(Opcode::from_u8(0x00), Some(Opcode::Nop));
    assert_eq!(Opcode::from_u8(0x38), Some(Opcode::Add));
    assert_eq!(Opcode::from_u8(0xFF), None);
}

#[test]
fn test_instruction_display() {
    let add = Instruction::op_dss(Opcode::Add, Register(0), Register(1), Register(2));
    assert!(add.to_string().contains("Add"));

    let load = Instruction::op_di(Opcode::LoadConst, Register(5), 42);
    assert!(load.to_string().contains("LoadConst"));
}

#[test]
fn test_register_display() {
    assert_eq!(Register(0).to_string(), "r0");
    assert_eq!(Register(255).to_string(), "r255");
}
