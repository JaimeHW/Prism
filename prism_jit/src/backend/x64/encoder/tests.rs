use super::*;

#[test]
fn test_rex_encoding() {
    assert_eq!(
        Rex {
            w: true,
            r: false,
            x: false,
            b: false
        }
        .encode(),
        0x48
    );
    assert_eq!(
        Rex {
            w: true,
            r: true,
            x: false,
            b: true
        }
        .encode(),
        0x4D
    );
    assert_eq!(
        Rex {
            w: false,
            r: false,
            x: false,
            b: true
        }
        .encode(),
        0x41
    );
}

#[test]
fn test_modrm_encoding() {
    assert_eq!(modrm(Mod::Direct, 3, 0), 0xD8);
    assert_eq!(modrm(Mod::Indirect, 0, 5), 0x05);
    assert_eq!(modrm(Mod::IndirectDisp8, 1, 2), 0x4A);
}

#[test]
fn test_mov_rr() {
    let enc = encode_mov_rr(Gpr::Rax, Gpr::Rbx);
    assert_eq!(enc.as_slice(), &[0x48, 0x89, 0xD8]);

    let enc = encode_mov_rr(Gpr::R8, Gpr::R9);
    assert_eq!(enc.as_slice(), &[0x4D, 0x89, 0xC8]);
}

#[test]
fn test_mov_ri64() {
    let enc = encode_mov_ri64(Gpr::Rax, 0x123456789ABCDEF0u64 as i64);
    assert_eq!(enc.len(), 10);
    assert_eq!(enc.as_slice()[0], 0x48); // REX.W
    assert_eq!(enc.as_slice()[1], 0xB8); // MOV RAX, imm64
}

#[test]
fn test_add_rr() {
    let enc = encode_add_rr(Gpr::Rax, Gpr::Rcx);
    assert_eq!(enc.as_slice(), &[0x48, 0x01, 0xC8]);
}

#[test]
fn test_add_ri8() {
    let enc = encode_add_ri8(Gpr::Rax, 5);
    assert_eq!(enc.as_slice(), &[0x48, 0x83, 0xC0, 0x05]);
}

#[test]
fn test_sub_rr() {
    let enc = encode_sub_rr(Gpr::Rcx, Gpr::Rdx);
    assert_eq!(enc.as_slice(), &[0x48, 0x29, 0xD1]);
}

#[test]
fn test_imul_rr() {
    let enc = encode_imul_rr(Gpr::Rax, Gpr::Rcx);
    assert_eq!(enc.as_slice(), &[0x48, 0x0F, 0xAF, 0xC1]);
}

#[test]
fn test_cmp_rr() {
    let enc = encode_cmp_rr(Gpr::Rax, Gpr::Rbx);
    assert_eq!(enc.as_slice(), &[0x48, 0x39, 0xD8]);
}

#[test]
fn test_jmp_rel8() {
    let enc = encode_jmp_rel8(5);
    assert_eq!(enc.as_slice(), &[0xEB, 0x05]);
}

#[test]
fn test_jcc_rel32() {
    let enc = encode_jcc_rel32(Condition::Equal, 0x1000);
    assert_eq!(enc.len(), 6);
    assert_eq!(enc.as_slice()[0..2], [0x0F, 0x84]);
}

#[test]
fn test_call_ret() {
    let enc = encode_call_rel32(0);
    assert_eq!(enc.as_slice(), &[0xE8, 0x00, 0x00, 0x00, 0x00]);

    let enc = encode_ret();
    assert_eq!(enc.as_slice(), &[0xC3]);
}

#[test]
fn test_push_pop() {
    let enc = encode_push(Gpr::Rbx);
    assert_eq!(enc.as_slice(), &[0x53]);

    let enc = encode_push(Gpr::R12);
    assert_eq!(enc.as_slice(), &[0x41, 0x54]);

    let enc = encode_pop(Gpr::Rax);
    assert_eq!(enc.as_slice(), &[0x58]);
}

#[test]
fn test_memory_simple() {
    let mem = MemOperand::base(Gpr::Rax);
    let enc = encode_mov_rm(Gpr::Rcx, &mem);
    assert_eq!(enc.as_slice(), &[0x48, 0x8B, 0x08]);
}

#[test]
fn test_memory_disp8() {
    let mem = MemOperand::base_disp(Gpr::Rbp, -8);
    let enc = encode_mov_rm(Gpr::Rax, &mem);
    assert_eq!(enc.as_slice(), &[0x48, 0x8B, 0x45, 0xF8]);
}

#[test]
fn test_memory_sib() {
    let mem = MemOperand::base_index(Gpr::Rax, Gpr::Rcx, Scale::X8);
    let enc = encode_mov_rm(Gpr::Rdx, &mem);
    assert_eq!(enc.as_slice(), &[0x48, 0x8B, 0x14, 0xC8]);
}

#[test]
fn test_sse_movsd() {
    let enc = encode_movsd_rr(Xmm::Xmm0, Xmm::Xmm1);
    assert_eq!(enc.as_slice(), &[0xF2, 0x0F, 0x10, 0xC1]);
}

#[test]
fn test_sse_addsd() {
    let enc = encode_addsd(Xmm::Xmm0, Xmm::Xmm1);
    assert_eq!(enc.as_slice(), &[0xF2, 0x0F, 0x58, 0xC1]);
}

#[test]
fn test_cvt_instructions() {
    let enc = encode_cvtsi2sd(Xmm::Xmm0, Gpr::Rax);
    assert!(enc.len() > 0);
    assert_eq!(enc.as_slice()[0], 0xF2);

    let enc = encode_cvttsd2si(Gpr::Rax, Xmm::Xmm0);
    assert!(enc.len() > 0);
}

#[test]
fn test_lea() {
    let mem = MemOperand::base_disp(Gpr::Rbp, 16);
    let enc = encode_lea(Gpr::Rax, &mem);
    assert_eq!(enc.as_slice(), &[0x48, 0x8D, 0x45, 0x10]);
}

#[test]
fn test_condition_invert() {
    assert_eq!(Condition::Equal.invert(), Condition::NotEqual);
    assert_eq!(Condition::Less.invert(), Condition::GreaterEqual);
    assert_eq!(Condition::Above.invert(), Condition::BelowEqual);
}

#[test]
fn test_nop_int3_ud2() {
    assert_eq!(encode_nop().as_slice(), &[0x90]);
    assert_eq!(encode_int3().as_slice(), &[0xCC]);
    assert_eq!(encode_ud2().as_slice(), &[0x0F, 0x0B]);
}
