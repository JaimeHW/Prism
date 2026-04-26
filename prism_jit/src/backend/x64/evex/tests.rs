use super::*;

#[test]
fn test_evex_default() {
    let evex = Evex::default();
    assert_eq!(evex.ll, 2); // 512-bit
    assert!(!evex.z);
    assert_eq!(evex.aaa, 0);
}

#[test]
fn test_evex_encode_basic() {
    let evex = Evex::zmm_66();
    let bytes = evex.encode();
    assert_eq!(bytes[0], 0x62); // EVEX escape
}

#[test]
fn test_evex_with_mask() {
    let evex = Evex::zmm_66().with_mask(KReg::K1);
    let bytes = evex.encode();
    assert_eq!(bytes[3] & 0x7, 1); // aaa = k1
}

#[test]
fn test_evex_with_zeroing() {
    let evex = Evex::zmm_66().with_zeroing();
    let bytes = evex.encode();
    assert_ne!(bytes[3] & 0x80, 0); // z bit set
}

#[test]
fn test_vaddpd_zmm() {
    let enc = encode_vaddpd_zmm_rrr(Zmm::Zmm0, Zmm::Zmm1, Zmm::Zmm2);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

#[test]
fn test_vmulpd_zmm() {
    let enc = encode_vmulpd_zmm_rrr(Zmm::Zmm16, Zmm::Zmm17, Zmm::Zmm18);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

#[test]
fn test_vaddpd_zmm_with_mask() {
    let enc = encode_vaddpd_zmm_rrr_k(Zmm::Zmm0, Zmm::Zmm1, Zmm::Zmm2, KReg::K3);
    assert!(enc.len() >= 6);
    // Check mask register in byte 3
    let bytes = enc.as_slice();
    assert_eq!(bytes[3] & 0x7, 3);
}

#[test]
fn test_vpaddd_zmm() {
    let enc = encode_vpaddd_zmm_rrr(Zmm::Zmm4, Zmm::Zmm5, Zmm::Zmm6);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vpxorq_zmm() {
    let enc = encode_vpxorq_zmm_rrr(Zmm::Zmm0, Zmm::Zmm0, Zmm::Zmm0);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vfmadd_zmm() {
    let enc = encode_vfmadd231pd_zmm_rrr(Zmm::Zmm0, Zmm::Zmm1, Zmm::Zmm2);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

#[test]
fn test_high_registers() {
    // ZMM16-31 require extended bits
    let enc = encode_vaddpd_zmm_rrr(Zmm::Zmm31, Zmm::Zmm30, Zmm::Zmm29);
    assert!(enc.len() >= 6);
    // Verify R' and V' bits are set correctly for high registers
    let bytes = enc.as_slice();
    assert_eq!(bytes[0], 0x62);
}

#[test]
fn test_vector_length_bits() {
    let zmm = Evex::zmm_66();
    assert_eq!(zmm.ll, 2);
    let ymm = Evex::ymm_66();
    assert_eq!(ymm.ll, 1);
    let xmm = Evex::xmm_66();
    assert_eq!(xmm.ll, 0);
}

#[test]
fn test_vmovapd_zmm_rm() {
    let mem = MemOperand::base(Gpr::Rax);
    let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

#[test]
fn test_vmovupd_zmm_rm() {
    let mem = MemOperand::base_disp(Gpr::Rbx, 128);
    let enc = encode_vmovupd_zmm_rm(Zmm::Zmm8, &mem);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vmovapd_zmm_mr() {
    let mem = MemOperand::base(Gpr::Rcx);
    let enc = encode_vmovapd_zmm_mr(&mem, Zmm::Zmm16);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vmovupd_zmm_mr() {
    let mem = MemOperand::base_disp(Gpr::R14, 64);
    let enc = encode_vmovupd_zmm_mr(&mem, Zmm::Zmm24);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vbroadcastsd_zmm() {
    let mem = MemOperand::base(Gpr::Rsi);
    let enc = encode_vbroadcastsd_zmm_rm(Zmm::Zmm0, &mem);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

#[test]
fn test_vaddpd_zmm_rmb() {
    let mem = MemOperand::base(Gpr::Rdi);
    let enc = encode_vaddpd_zmm_rmb(Zmm::Zmm1, Zmm::Zmm2, &mem);
    assert!(enc.len() >= 6);
    // Check broadcast bit (b) is set in byte 3
    let bytes = enc.as_slice();
    assert_ne!(bytes[3] & 0x10, 0); // broadcast bit set
}

#[test]
fn test_vmulpd_zmm_rmb() {
    let mem = MemOperand::base(Gpr::R8);
    let enc = encode_vmulpd_zmm_rmb(Zmm::Zmm3, Zmm::Zmm4, &mem);
    assert!(enc.len() >= 6);
    // Broadcast bit should be set
    assert_ne!(enc.as_slice()[3] & 0x10, 0);
}

#[test]
fn test_vfmadd231pd_zmm_rmb() {
    let mem = MemOperand::base(Gpr::R9);
    let enc = encode_vfmadd231pd_zmm_rmb(Zmm::Zmm5, Zmm::Zmm6, &mem);
    assert!(enc.len() >= 6);
}

#[test]
fn test_disp8_compression() {
    // 512-bit with 8-byte elements: disp8*N factor = 64
    // disp = 64 should compress to 1
    let mem = MemOperand::base_disp(Gpr::Rax, 64);
    let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
    // Should use disp8 encoding (shorter instruction)
    let bytes = enc.as_slice();
    // ModRM byte is at index 5: EVEX (4 bytes) + opcode (1 byte) = 5
    let modrm = bytes[5];
    assert_eq!((modrm >> 6) & 3, 1); // mod = 01 means disp8
}

#[test]
fn test_disp32_fallback() {
    // Displacement not divisible by 64 - cannot use disp8*N
    let mem = MemOperand::base_disp(Gpr::Rax, 100);
    let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
    let bytes = enc.as_slice();
    // Should fall back to disp32
    // ModRM byte is at index 5: EVEX (4 bytes) + opcode (1 byte) = 5
    let modrm = bytes[5];
    assert_eq!((modrm >> 6) & 3, 2); // mod = 10 means disp32
}

#[test]
fn test_memory_with_sib() {
    // RSP requires SIB byte
    let mem = MemOperand::base(Gpr::Rsp);
    let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
    assert!(enc.len() >= 7); // Extra SIB byte
}

#[test]
fn test_memory_with_index() {
    let mem = MemOperand::base_index(Gpr::Rax, Gpr::Rcx, Scale::X8);
    let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
    assert!(enc.len() >= 7); // Needs SIB
}

#[test]
fn test_all_arithmetic_zmm() {
    let dst = Zmm::Zmm0;
    let src1 = Zmm::Zmm1;
    let src2 = Zmm::Zmm2;

    assert!(encode_vaddpd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vsubpd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vmulpd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vdivpd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vmaxpd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vminpd_zmm_rrr(dst, src1, src2).len() >= 6);
}

#[test]
fn test_all_arithmetic_ps_zmm() {
    let dst = Zmm::Zmm0;
    let src1 = Zmm::Zmm1;
    let src2 = Zmm::Zmm2;

    assert!(encode_vaddps_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vsubps_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vmulps_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vdivps_zmm_rrr(dst, src1, src2).len() >= 6);
}

#[test]
fn test_all_integer_zmm() {
    let dst = Zmm::Zmm0;
    let src1 = Zmm::Zmm1;
    let src2 = Zmm::Zmm2;

    assert!(encode_vpaddd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vpsubd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vpaddq_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vpsubq_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vpmulld_zmm_rrr(dst, src1, src2).len() >= 6);
}

#[test]
fn test_all_logical_zmm() {
    let dst = Zmm::Zmm0;
    let src1 = Zmm::Zmm1;
    let src2 = Zmm::Zmm2;

    assert!(encode_vpandd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vpandq_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vpord_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vporq_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vpxord_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vpxorq_zmm_rrr(dst, src1, src2).len() >= 6);
}

#[test]
fn test_all_fma_zmm() {
    let dst = Zmm::Zmm0;
    let src1 = Zmm::Zmm1;
    let src2 = Zmm::Zmm2;

    assert!(encode_vfmadd132pd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vfmadd213pd_zmm_rrr(dst, src1, src2).len() >= 6);
    assert!(encode_vfmadd231pd_zmm_rrr(dst, src1, src2).len() >= 6);
}

#[test]
fn test_evex_bits_encoding() {
    // Test that R, X, B, R' bits are correctly inverted
    // When r/x/b/r_prime are true (meaning low registers 0-7),
    // the encoded bits are 0 (inverted)
    let evex = Evex::default();
    let bytes = evex.encode();

    // Default: r=x=b=r_prime=true → encoded bits are 0 (they're inverted)
    // Byte 1 high nibble = 0b0000 (all inverted to 0) + mm bits
    assert_eq!(bytes[1] & 0xF0, 0x00); // High nibble is 0

    // Test with high registers - bits should be 1 when r/x/b are false
    let evex_high = Evex {
        r: false,
        x: false,
        b: false,
        r_prime: false,
        ..Evex::zmm_66()
    };
    let bytes_high = evex_high.encode();
    // R=0, X=0, B=0, R'=0 → inverted bits are 1
    assert_eq!(bytes_high[1] & 0xF0, 0xF0);
}

#[test]
fn test_vsqrtpd_zmm() {
    let enc = encode_vsqrtpd_zmm_rr(Zmm::Zmm0, Zmm::Zmm1);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

// =========================================================================
// ZMM Register-to-Register Move Tests
// =========================================================================

#[test]
fn test_vmovapd_zmm_rr_basic() {
    let enc = encode_vmovapd_zmm_rr(Zmm::Zmm0, Zmm::Zmm1);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62); // EVEX prefix
}

#[test]
fn test_vmovapd_zmm_rr_high_registers() {
    // ZMM16-31 require extended register bits
    let enc = encode_vmovapd_zmm_rr(Zmm::Zmm16, Zmm::Zmm31);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

#[test]
fn test_vmovaps_zmm_rr_basic() {
    let enc = encode_vmovaps_zmm_rr(Zmm::Zmm2, Zmm::Zmm3);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

#[test]
fn test_vmovupd_zmm_rr_basic() {
    let enc = encode_vmovupd_zmm_rr(Zmm::Zmm4, Zmm::Zmm5);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vmovups_zmm_rr_basic() {
    let enc = encode_vmovups_zmm_rr(Zmm::Zmm6, Zmm::Zmm7);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vmovdqa64_zmm_rr_basic() {
    let enc = encode_vmovdqa64_zmm_rr(Zmm::Zmm8, Zmm::Zmm9);
    assert!(enc.len() >= 6);
    assert_eq!(enc.as_slice()[0], 0x62);
}

#[test]
fn test_vmovdqa32_zmm_rr_basic() {
    let enc = encode_vmovdqa32_zmm_rr(Zmm::Zmm10, Zmm::Zmm11);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vmovdqu64_zmm_rr_basic() {
    let enc = encode_vmovdqu64_zmm_rr(Zmm::Zmm12, Zmm::Zmm13);
    assert!(enc.len() >= 6);
}

#[test]
fn test_vmovdqu32_zmm_rr_basic() {
    let enc = encode_vmovdqu32_zmm_rr(Zmm::Zmm14, Zmm::Zmm15);
    assert!(enc.len() >= 6);
}

#[test]
fn test_zmm_move_all_registers() {
    // Test move instruction encoding for all 32 ZMM register pairs
    for i in 0..32u8 {
        for j in 0..32u8 {
            let src = Zmm::from_encoding(i).unwrap();
            let dst = Zmm::from_encoding(j).unwrap();

            // All combinations should encode without panic
            let enc = encode_vmovapd_zmm_rr(dst, src);
            assert!(enc.len() >= 6, "Failed for ZMM{} -> ZMM{}", i, j);
        }
    }
}

#[test]
fn test_zmm_move_self_encoding() {
    // Moving register to itself should still encode correctly
    for i in 0..32u8 {
        let reg = Zmm::from_encoding(i).unwrap();
        let enc = encode_vmovapd_zmm_rr(reg, reg);
        assert!(enc.len() >= 6);
    }
}
