use super::*;

#[test]
fn test_basic_assembly() {
    let mut asm = Arm64Assembler::new();
    asm.nop();
    asm.ret();
    let code = asm.finalize().unwrap();
    assert_eq!(code.len(), 8); // 2 instructions × 4 bytes
}

#[test]
fn test_labels() {
    let mut asm = Arm64Assembler::new();
    let label = asm.create_label();
    asm.b(label);
    asm.nop();
    asm.bind_label(label);
    asm.ret();
    let code = asm.finalize().unwrap();
    assert_eq!(code.len(), 12);
}

#[test]
fn test_mov_imm64_zero() {
    let mut asm = Arm64Assembler::new();
    asm.mov_imm64(Gpr::X0, 0);
    let code = asm.finalize().unwrap();
    assert_eq!(code.len(), 4);
}

#[test]
fn test_mov_imm64_small() {
    let mut asm = Arm64Assembler::new();
    asm.mov_imm64(Gpr::X0, 0x1234);
    let code = asm.finalize().unwrap();
    assert_eq!(code.len(), 4); // Just MOVZ
}

#[test]
fn test_mov_imm64_large() {
    let mut asm = Arm64Assembler::new();
    asm.mov_imm64(Gpr::X0, 0x1234_5678_9ABC_DEF0);
    let code = asm.finalize().unwrap();
    // Should be MOVZ + MOVK × 3
    assert_eq!(code.len(), 16);
}

#[test]
fn test_conditional_branch() {
    let mut asm = Arm64Assembler::new();
    let skip = asm.create_label();
    asm.cmp_imm(Gpr::X0, 0);
    asm.bcond(Condition::Eq, skip);
    asm.add_imm(Gpr::X0, Gpr::X0, 1);
    asm.bind_label(skip);
    asm.ret();
    let code = asm.finalize().unwrap();
    assert!(code.len() > 0);
}
