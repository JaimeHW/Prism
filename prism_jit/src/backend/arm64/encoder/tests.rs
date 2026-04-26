use super::*;

#[test]
fn test_condition_encoding() {
    assert_eq!(Condition::Eq.encoding(), 0);
    assert_eq!(Condition::Ne.encoding(), 1);
    assert_eq!(Condition::Ge.encoding(), 10);
    assert_eq!(Condition::Lt.encoding(), 11);
}

#[test]
fn test_condition_invert() {
    assert_eq!(Condition::Eq.invert(), Condition::Ne);
    assert_eq!(Condition::Ne.invert(), Condition::Eq);
    assert_eq!(Condition::Ge.invert(), Condition::Lt);
}

#[test]
fn test_encoded_inst() {
    let inst = EncodedInst::new(0x12345678);
    assert_eq!(inst.bits(), 0x12345678);
    assert_eq!(inst.to_le_bytes(), [0x78, 0x56, 0x34, 0x12]);
}

#[test]
fn test_nop() {
    assert_eq!(encode_nop().bits(), 0xD503201F);
}

#[test]
fn test_ret() {
    assert_eq!(encode_ret(None).bits(), 0xD65F03C0);
}

#[test]
fn test_b_encoding() {
    let b = encode_b(0);
    assert_eq!(b.bits() >> 26, 0b000101);
}

#[test]
fn test_bl_encoding() {
    let bl = encode_bl(0);
    assert_eq!(bl.bits() >> 26, 0b100101);
}

#[test]
fn test_add_imm() {
    let add = encode_add_imm(Gpr::X0, Gpr::X1, 42, false);
    assert_eq!(add.bits() & 0x1F, 0); // dst = X0
    assert_eq!((add.bits() >> 5) & 0x1F, 1); // src = X1
}

#[test]
fn test_movz() {
    let movz = encode_movz(Gpr::X0, 0x1234, 0);
    assert_eq!(movz.bits() & 0x1F, 0);
    assert_eq!((movz.bits() >> 5) & 0xFFFF, 0x1234);
}

#[test]
fn test_cmp_reg() {
    let cmp = encode_cmp_reg(Gpr::X1, Gpr::X2);
    // Destination should be XZR (31)
    assert_eq!(cmp.bits() & 0x1F, 31);
}
