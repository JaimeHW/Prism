use super::*;

#[test]
fn test_ymm_encoding() {
    assert_eq!(Ymm::Ymm0.encoding(), 0);
    assert_eq!(Ymm::Ymm8.encoding(), 8);
    assert!(!Ymm::Ymm7.high_bit());
    assert!(Ymm::Ymm8.high_bit());
}

#[test]
fn test_vex_2byte() {
    let vex = Vex {
        l: true,
        pp: 1,
        mm: 1,
        w: false,
        r: true,
        x: true,
        b: true,
        vvvv: 0,
    };
    assert!(vex.can_use_2byte());
    let bytes = vex.encode_2byte();
    assert_eq!(bytes[0], 0xC5);
}

#[test]
fn test_vex_3byte() {
    let vex = Vex {
        l: true,
        pp: 1,
        mm: 2,
        w: false,
        r: true,
        x: true,
        b: true,
        vvvv: 0,
    };
    assert!(!vex.can_use_2byte());
    let bytes = vex.encode_3byte();
    assert_eq!(bytes[0], 0xC4);
}

#[test]
fn test_vaddpd() {
    let enc = encode_vaddpd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    assert!(enc.len() > 0);
    // Check for VEX prefix
    let slice = enc.as_slice();
    assert!(slice[0] == 0xC5 || slice[0] == 0xC4);
}

#[test]
fn test_vmovapd_rm() {
    use super::super::registers::Gpr;
    let mem = MemOperand::base(Gpr::Rax);
    let enc = encode_vmovapd_rm(Ymm::Ymm0, &mem);
    assert!(enc.len() > 0);
}

#[test]
fn test_ymm_xmm_conversion() {
    let ymm = Ymm::Ymm5;
    let xmm = ymm.to_xmm();
    assert_eq!(xmm, Xmm::Xmm5);
    assert_eq!(Ymm::from_xmm(xmm), ymm);
}

#[test]
fn test_zmm_encoding() {
    assert_eq!(Zmm::Zmm0.encoding(), 0);
    assert_eq!(Zmm::Zmm16.encoding(), 16);
    assert!(Zmm::Zmm16.ext_bit());
    assert!(!Zmm::Zmm15.ext_bit());
}

#[test]
fn test_vsubpd_encoding() {
    let enc = encode_vsubpd_rrr(Ymm::Ymm2, Ymm::Ymm3, Ymm::Ymm4);
    assert!(enc.len() > 0);
    let slice = enc.as_slice();
    // VEX prefix start
    assert!(slice[0] == 0xC5 || slice[0] == 0xC4);
}

#[test]
fn test_vmulpd_encoding() {
    let enc = encode_vmulpd_rrr(Ymm::Ymm5, Ymm::Ymm6, Ymm::Ymm7);
    assert!(enc.len() >= 4); // VEX + opcode + modrm
}

#[test]
fn test_vdivpd_encoding() {
    let enc = encode_vdivpd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    assert!(enc.len() >= 4);
}

#[test]
fn test_packed_single_operations() {
    let add = encode_vaddps_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    let sub = encode_vsubps_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    let mul = encode_vmulps_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    let div = encode_vdivps_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);

    // All should have reasonable length
    assert!(add.len() >= 4);
    assert!(sub.len() >= 4);
    assert!(mul.len() >= 4);
    assert!(div.len() >= 4);
}

#[test]
fn test_integer_operations() {
    let paddd = encode_vpaddd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    let psubd = encode_vpsubd_rrr(Ymm::Ymm3, Ymm::Ymm4, Ymm::Ymm5);
    let paddq = encode_vpaddq_rrr(Ymm::Ymm6, Ymm::Ymm7, Ymm::Ymm0);
    let psubq = encode_vpsubq_rrr(Ymm::Ymm1, Ymm::Ymm2, Ymm::Ymm3);

    assert!(paddd.len() >= 4);
    assert!(psubd.len() >= 4);
    assert!(paddq.len() >= 4);
    assert!(psubq.len() >= 4);
}

#[test]
fn test_logical_operations() {
    let andpd = encode_vandpd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    let orpd = encode_vorpd_rrr(Ymm::Ymm3, Ymm::Ymm4, Ymm::Ymm5);
    let xorpd = encode_vxorpd_rrr(Ymm::Ymm6, Ymm::Ymm7, Ymm::Ymm0);

    assert!(andpd.len() >= 4);
    assert!(orpd.len() >= 4);
    assert!(xorpd.len() >= 4);
}

#[test]
fn test_fma_operations() {
    let fma132 = encode_vfmadd132pd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    let fma213 = encode_vfmadd213pd_rrr(Ymm::Ymm3, Ymm::Ymm4, Ymm::Ymm5);
    let fma231 = encode_vfmadd231pd_rrr(Ymm::Ymm6, Ymm::Ymm7, Ymm::Ymm0);

    // FMA uses 3-byte VEX with W bit
    assert!(fma132.len() >= 5);
    assert!(fma213.len() >= 5);
    assert!(fma231.len() >= 5);

    // All should use C4 prefix (3-byte VEX for 0F38 map)
    assert_eq!(fma132.as_slice()[0], 0xC4);
    assert_eq!(fma213.as_slice()[0], 0xC4);
    assert_eq!(fma231.as_slice()[0], 0xC4);
}

#[test]
fn test_comparison_operations() {
    // VCMPPD with EQ predicate (0)
    let cmp_eq = encode_vcmppd_rrri(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2, 0);
    // VCMPPD with LT predicate (1)
    let cmp_lt = encode_vcmppd_rrri(Ymm::Ymm3, Ymm::Ymm4, Ymm::Ymm5, 1);

    assert!(cmp_eq.len() >= 5); // VEX + opcode + modrm + imm8
    assert!(cmp_lt.len() >= 5);
}

#[test]
fn test_conversion_operations() {
    let pd2ps = encode_vcvtpd2ps_rr(Xmm::Xmm0, Ymm::Ymm1);
    let ps2pd = encode_vcvtps2pd_rr(Ymm::Ymm2, Xmm::Xmm3);
    let dq2pd = encode_vcvtdq2pd_rr(Ymm::Ymm4, Xmm::Xmm5);
    let pd2dq = encode_vcvttpd2dq_rr(Xmm::Xmm6, Ymm::Ymm7);

    assert!(pd2ps.len() >= 5);
    assert!(ps2pd.len() >= 5);
    assert!(dq2pd.len() >= 5);
    assert!(pd2dq.len() >= 5);
}

#[test]
fn test_xmm_128bit_operations() {
    let add_xmm = encode_vaddpd_xmm_rrr(Xmm::Xmm0, Xmm::Xmm1, Xmm::Xmm2);
    let mul_xmm = encode_vmulpd_xmm_rrr(Xmm::Xmm3, Xmm::Xmm4, Xmm::Xmm5);

    // 128-bit operations can use 2-byte VEX
    assert!(add_xmm.len() >= 4);
    assert!(mul_xmm.len() >= 4);

    // L bit should be 0 for 128-bit
    let slice = add_xmm.as_slice();
    if slice[0] == 0xC5 {
        // 2-byte VEX: byte 1 has L bit at position 2
        let l_bit = (slice[1] >> 2) & 1;
        assert_eq!(l_bit, 0); // 128-bit
    }
}

#[test]
fn test_shuffle_operations() {
    let vpermilpd = encode_vpermilpd_rri(Ymm::Ymm0, Ymm::Ymm1, 0b1010);
    let vperm2f128 = encode_vperm2f128_rrri(Ymm::Ymm2, Ymm::Ymm3, Ymm::Ymm4, 0x31);

    assert!(vpermilpd.len() >= 5); // VEX + opcode + modrm + imm8
    assert!(vperm2f128.len() >= 6); // VEX + opcode + modrm + imm8
}

#[test]
fn test_memory_operations() {
    use super::super::registers::Gpr;

    let mem_rax = MemOperand::base(Gpr::Rax);
    let mem_rbx_disp = MemOperand::base_disp(Gpr::Rbx, 64);

    let movapd_rm = encode_vmovapd_rm(Ymm::Ymm0, &mem_rax);
    let movupd_rm = encode_vmovupd_rm(Ymm::Ymm1, &mem_rbx_disp);
    let movapd_mr = encode_vmovapd_mr(&mem_rax, Ymm::Ymm2);
    let movupd_mr = encode_vmovupd_mr(&mem_rbx_disp, Ymm::Ymm3);
    let broadcast = encode_vbroadcastsd_rm(Ymm::Ymm4, &mem_rax);

    assert!(movapd_rm.len() > 0);
    assert!(movupd_rm.len() > 0);
    assert!(movapd_mr.len() > 0);
    assert!(movupd_mr.len() > 0);
    assert!(broadcast.len() > 0);
}

#[test]
fn test_all_ymm_registers() {
    // Ensure high registers (Ymm8-Ymm15) work correctly
    let enc_low = encode_vaddpd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
    let enc_high = encode_vaddpd_rrr(Ymm::Ymm8, Ymm::Ymm9, Ymm::Ymm10);

    assert!(enc_low.len() > 0);
    assert!(enc_high.len() > 0);

    // High register encoding should use 3-byte VEX (C4 prefix)
    assert_eq!(enc_high.as_slice()[0], 0xC4);
}

#[test]
fn test_vex_prefix_fields() {
    // Test all VEX field combinations
    let vex_basic = Vex {
        l: false,
        pp: 0,
        mm: 1,
        w: false,
        r: true,
        x: true,
        b: true,
        vvvv: 0,
    };
    assert!(vex_basic.can_use_2byte());

    let vex_256bit = Vex {
        l: true,
        pp: 1,
        mm: 1,
        w: false,
        r: true,
        x: true,
        b: true,
        vvvv: 5,
    };
    assert!(vex_256bit.can_use_2byte());

    let vex_0f38 = Vex {
        l: true,
        pp: 1,
        mm: 2,
        w: false,
        r: true,
        x: true,
        b: true,
        vvvv: 0,
    };
    assert!(!vex_0f38.can_use_2byte()); // mm != 1 forces 3-byte

    let vex_with_w = Vex {
        l: true,
        pp: 1,
        mm: 1,
        w: true,
        r: true,
        x: true,
        b: true,
        vvvv: 0,
    };
    assert!(!vex_with_w.can_use_2byte()); // w=1 forces 3-byte
}

#[test]
fn test_kreg_encoding() {
    assert_eq!(KReg::K0.encoding(), 0);
    assert_eq!(KReg::K7.encoding(), 7);
}

// =========================================================================
// YmmSet Tests
// =========================================================================

#[test]
fn test_ymmset_empty_and_all() {
    assert!(YmmSet::EMPTY.is_empty());
    assert_eq!(YmmSet::EMPTY.count(), 0);
    assert!(!YmmSet::ALL.is_empty());
    assert_eq!(YmmSet::ALL.count(), 16);
}

#[test]
fn test_ymmset_singleton() {
    let set = YmmSet::singleton(Ymm::Ymm5);
    assert!(set.contains(Ymm::Ymm5));
    assert!(!set.contains(Ymm::Ymm0));
    assert_eq!(set.count(), 1);
}

#[test]
fn test_ymmset_insert_remove() {
    let mut set = YmmSet::EMPTY;
    set = set.insert(Ymm::Ymm0);
    set = set.insert(Ymm::Ymm15);

    assert!(set.contains(Ymm::Ymm0));
    assert!(set.contains(Ymm::Ymm15));
    assert!(!set.contains(Ymm::Ymm7));
    assert_eq!(set.count(), 2);

    set = set.remove(Ymm::Ymm0);
    assert!(!set.contains(Ymm::Ymm0));
    assert!(set.contains(Ymm::Ymm15));
    assert_eq!(set.count(), 1);
}

#[test]
fn test_ymmset_union_intersection() {
    let a = YmmSet::singleton(Ymm::Ymm0)
        .insert(Ymm::Ymm1)
        .insert(Ymm::Ymm2);
    let b = YmmSet::singleton(Ymm::Ymm1)
        .insert(Ymm::Ymm2)
        .insert(Ymm::Ymm3);

    let union = a.union(b);
    assert_eq!(union.count(), 4);
    assert!(union.contains(Ymm::Ymm0));
    assert!(union.contains(Ymm::Ymm3));

    let intersection = a.intersection(b);
    assert_eq!(intersection.count(), 2);
    assert!(intersection.contains(Ymm::Ymm1));
    assert!(intersection.contains(Ymm::Ymm2));
    assert!(!intersection.contains(Ymm::Ymm0));
}

#[test]
fn test_ymmset_difference() {
    let a = YmmSet::singleton(Ymm::Ymm0)
        .insert(Ymm::Ymm1)
        .insert(Ymm::Ymm2);
    let b = YmmSet::singleton(Ymm::Ymm1);

    let diff = a.difference(b);
    assert_eq!(diff.count(), 2);
    assert!(diff.contains(Ymm::Ymm0));
    assert!(diff.contains(Ymm::Ymm2));
    assert!(!diff.contains(Ymm::Ymm1));
}

#[test]
fn test_ymmset_first() {
    assert_eq!(YmmSet::EMPTY.first(), None);

    let set = YmmSet::singleton(Ymm::Ymm5).insert(Ymm::Ymm10);
    assert_eq!(set.first(), Some(Ymm::Ymm5));

    let high_only = YmmSet::singleton(Ymm::Ymm15);
    assert_eq!(high_only.first(), Some(Ymm::Ymm15));
}

#[test]
fn test_ymmset_iter() {
    let set = YmmSet::singleton(Ymm::Ymm3)
        .insert(Ymm::Ymm7)
        .insert(Ymm::Ymm11);

    let regs: Vec<Ymm> = set.iter().collect();
    assert_eq!(regs.len(), 3);
    assert_eq!(regs[0], Ymm::Ymm3);
    assert_eq!(regs[1], Ymm::Ymm7);
    assert_eq!(regs[2], Ymm::Ymm11);
}

#[test]
fn test_ymmset_from_bits() {
    let set = YmmSet::from_bits(0b1010_0101);
    assert!(set.contains(Ymm::Ymm0));
    assert!(set.contains(Ymm::Ymm2));
    assert!(set.contains(Ymm::Ymm5));
    assert!(set.contains(Ymm::Ymm7));
    assert_eq!(set.count(), 4);
}

#[test]
fn test_ymmset_all_registers() {
    for reg in YmmSet::ALL.iter() {
        assert!(YmmSet::ALL.contains(reg));
    }
    assert_eq!(YmmSet::ALL.iter().count(), 16);
}

// =========================================================================
// ZmmSet Tests
// =========================================================================

#[test]
fn test_zmmset_empty_and_all() {
    assert!(ZmmSet::EMPTY.is_empty());
    assert_eq!(ZmmSet::EMPTY.count(), 0);
    assert!(!ZmmSet::ALL.is_empty());
    assert_eq!(ZmmSet::ALL.count(), 32);
}

#[test]
fn test_zmmset_all16_and_upper16() {
    assert_eq!(ZmmSet::ALL16.count(), 16);
    assert_eq!(ZmmSet::UPPER16.count(), 16);

    // Check they don't overlap
    let intersection = ZmmSet::ALL16.intersection(ZmmSet::UPPER16);
    assert!(intersection.is_empty());

    // Check they combine to ALL
    let union = ZmmSet::ALL16.union(ZmmSet::UPPER16);
    assert_eq!(union, ZmmSet::ALL);
}

#[test]
fn test_zmmset_singleton() {
    let set = ZmmSet::singleton(Zmm::Zmm20);
    assert!(set.contains(Zmm::Zmm20));
    assert!(!set.contains(Zmm::Zmm0));
    assert_eq!(set.count(), 1);
}

#[test]
fn test_zmmset_insert_remove() {
    let mut set = ZmmSet::EMPTY;
    set = set.insert(Zmm::Zmm0);
    set = set.insert(Zmm::Zmm31);
    set = set.insert(Zmm::Zmm16);

    assert!(set.contains(Zmm::Zmm0));
    assert!(set.contains(Zmm::Zmm31));
    assert!(set.contains(Zmm::Zmm16));
    assert_eq!(set.count(), 3);

    set = set.remove(Zmm::Zmm16);
    assert!(!set.contains(Zmm::Zmm16));
    assert_eq!(set.count(), 2);
}

#[test]
fn test_zmmset_union_intersection() {
    let a = ZmmSet::singleton(Zmm::Zmm0)
        .insert(Zmm::Zmm16)
        .insert(Zmm::Zmm20);
    let b = ZmmSet::singleton(Zmm::Zmm16)
        .insert(Zmm::Zmm20)
        .insert(Zmm::Zmm30);

    let union = a.union(b);
    assert_eq!(union.count(), 4);

    let intersection = a.intersection(b);
    assert_eq!(intersection.count(), 2);
    assert!(intersection.contains(Zmm::Zmm16));
    assert!(intersection.contains(Zmm::Zmm20));
}

#[test]
fn test_zmmset_difference() {
    let a = ZmmSet::ALL16;
    let b = ZmmSet::singleton(Zmm::Zmm0).insert(Zmm::Zmm1);

    let diff = a.difference(b);
    assert_eq!(diff.count(), 14);
    assert!(!diff.contains(Zmm::Zmm0));
    assert!(!diff.contains(Zmm::Zmm1));
}

#[test]
fn test_zmmset_first() {
    assert_eq!(ZmmSet::EMPTY.first(), None);

    let set = ZmmSet::singleton(Zmm::Zmm20).insert(Zmm::Zmm25);
    assert_eq!(set.first(), Some(Zmm::Zmm20));

    let low = ZmmSet::singleton(Zmm::Zmm0);
    assert_eq!(low.first(), Some(Zmm::Zmm0));
}

#[test]
fn test_zmmset_iter() {
    let set = ZmmSet::singleton(Zmm::Zmm5)
        .insert(Zmm::Zmm15)
        .insert(Zmm::Zmm25);

    let regs: Vec<Zmm> = set.iter().collect();
    assert_eq!(regs.len(), 3);
    assert_eq!(regs[0], Zmm::Zmm5);
    assert_eq!(regs[1], Zmm::Zmm15);
    assert_eq!(regs[2], Zmm::Zmm25);
}

#[test]
fn test_zmmset_from_bits() {
    let set = ZmmSet::from_bits(0xFFFF_0000); // Upper 16
    assert_eq!(set, ZmmSet::UPPER16);
    assert!(set.contains(Zmm::Zmm16));
    assert!(!set.contains(Zmm::Zmm0));
}

#[test]
fn test_zmmset_all_registers() {
    for reg in ZmmSet::ALL.iter() {
        assert!(ZmmSet::ALL.contains(reg));
    }
    assert_eq!(ZmmSet::ALL.iter().count(), 32);
}

// =========================================================================
// Ymm/Zmm Conversion Tests
// =========================================================================

#[test]
fn test_ymm_from_encoding() {
    for i in 0_u8..16 {
        let ymm = Ymm::from_encoding(i);
        assert!(ymm.is_some());
        assert_eq!(ymm.unwrap().encoding(), i);
    }
    assert_eq!(Ymm::from_encoding(16), None);
    assert_eq!(Ymm::from_encoding(255), None);
}

#[test]
fn test_zmm_from_encoding() {
    for i in 0_u8..32 {
        let zmm = Zmm::from_encoding(i);
        assert!(zmm.is_some());
        assert_eq!(zmm.unwrap().encoding(), i);
    }
    assert_eq!(Zmm::from_encoding(32), None);
    assert_eq!(Zmm::from_encoding(255), None);
}

#[test]
fn test_ymm_xmm_conversion_roundtrip() {
    use crate::backend::x64::Xmm;
    for i in 0_u8..16 {
        if let Some(ymm) = Ymm::from_encoding(i) {
            let xmm = ymm.to_xmm();
            let back = Ymm::from_xmm(xmm);
            assert_eq!(back, ymm);
        }
    }
}

#[test]
fn test_zmm_ymm_conversion_roundtrip() {
    for i in 0_u8..16 {
        if let Some(zmm) = Zmm::from_encoding(i) {
            if let Some(ymm) = zmm.to_ymm() {
                let back = Zmm::from_ymm(ymm);
                assert_eq!(back, zmm);
            }
        }
    }
}

#[test]
fn test_zmm_upper_no_ymm() {
    // ZMM16-31 don't have YMM equivalents
    for i in 16_u8..32 {
        if let Some(zmm) = Zmm::from_encoding(i) {
            assert_eq!(zmm.to_ymm(), None);
        }
    }
}

#[test]
fn test_ymm_all_constant() {
    assert_eq!(Ymm::ALL.len(), 16);
    for (i, ymm) in Ymm::ALL.iter().enumerate() {
        assert_eq!(ymm.encoding(), i as u8);
    }
}

#[test]
fn test_zmm_all_constants() {
    assert_eq!(Zmm::ALL.len(), 32);
    assert_eq!(Zmm::ALL16.len(), 16);
    for (i, zmm) in Zmm::ALL.iter().enumerate() {
        assert_eq!(zmm.encoding(), i as u8);
    }
}

// =========================================================================
// Set Identity/Property Tests
// =========================================================================

#[test]
fn test_ymmset_identity_properties() {
    let set = YmmSet::singleton(Ymm::Ymm3).insert(Ymm::Ymm7);

    // Union with empty is identity
    assert_eq!(set.union(YmmSet::EMPTY), set);

    // Intersection with ALL is identity
    assert_eq!(set.intersection(YmmSet::ALL), set);

    // Difference with empty is identity
    assert_eq!(set.difference(YmmSet::EMPTY), set);

    // Difference with self is empty
    assert_eq!(set.difference(set), YmmSet::EMPTY);
}

#[test]
fn test_zmmset_identity_properties() {
    let set = ZmmSet::singleton(Zmm::Zmm10).insert(Zmm::Zmm25);

    // Union with empty is identity
    assert_eq!(set.union(ZmmSet::EMPTY), set);

    // Intersection with ALL is identity
    assert_eq!(set.intersection(ZmmSet::ALL), set);

    // Difference with empty is identity
    assert_eq!(set.difference(ZmmSet::EMPTY), set);

    // Difference with self is empty
    assert_eq!(set.difference(set), ZmmSet::EMPTY);
}

#[test]
fn test_ymmset_debug_format() {
    let set = YmmSet::singleton(Ymm::Ymm0).insert(Ymm::Ymm15);
    let debug_str = format!("{:?}", set);
    assert!(debug_str.contains("YmmSet"));
    assert!(debug_str.contains("ymm0"));
    assert!(debug_str.contains("ymm15"));
}

#[test]
fn test_zmmset_debug_format() {
    let set = ZmmSet::singleton(Zmm::Zmm0).insert(Zmm::Zmm31);
    let debug_str = format!("{:?}", set);
    assert!(debug_str.contains("ZmmSet"));
    assert!(debug_str.contains("zmm0"));
    assert!(debug_str.contains("zmm31"));
}

// =========================================================================
// YMM Register-to-Register Move Tests
// =========================================================================

#[test]
fn test_vmovapd_ymm_rr_basic() {
    let enc = super::encode_vmovapd_ymm_rr(Ymm::Ymm0, Ymm::Ymm1);
    // VEX.256 + 66 prefix + opcode 0x28 + ModR/M
    assert!(enc.len() >= 3);
    // Verify it starts with VEX prefix
    assert!(enc.as_slice()[0] == 0xC5 || enc.as_slice()[0] == 0xC4);
}

#[test]
fn test_vmovapd_ymm_rr_extended_registers() {
    // YMM8 and YMM15 require REX bits via VEX
    let enc = super::encode_vmovapd_ymm_rr(Ymm::Ymm8, Ymm::Ymm15);
    assert!(enc.len() >= 3);
    // With extended registers, 3-byte VEX is needed
    assert_eq!(enc.as_slice()[0], 0xC4);
}

#[test]
fn test_vmovaps_ymm_rr_basic() {
    let enc = super::encode_vmovaps_ymm_rr(Ymm::Ymm2, Ymm::Ymm3);
    assert!(enc.len() >= 3);
    // Should use VEX without 66 prefix (pp=0)
    assert!(enc.as_slice()[0] == 0xC5 || enc.as_slice()[0] == 0xC4);
}

#[test]
fn test_vmovupd_ymm_rr_basic() {
    let enc = super::encode_vmovupd_ymm_rr(Ymm::Ymm4, Ymm::Ymm5);
    assert!(enc.len() >= 3);
    // Uses opcode 0x10
}

#[test]
fn test_vmovups_ymm_rr_basic() {
    let enc = super::encode_vmovups_ymm_rr(Ymm::Ymm6, Ymm::Ymm7);
    assert!(enc.len() >= 3);
}

#[test]
fn test_vmovdqa_ymm_rr_basic() {
    let enc = super::encode_vmovdqa_ymm_rr(Ymm::Ymm0, Ymm::Ymm1);
    assert!(enc.len() >= 3);
    // Uses opcode 0x6F
}

#[test]
fn test_vmovdqu_ymm_rr_basic() {
    let enc = super::encode_vmovdqu_ymm_rr(Ymm::Ymm2, Ymm::Ymm3);
    assert!(enc.len() >= 3);
    // Uses F3 prefix (pp=2)
}

// =========================================================================
// XMM Register-to-Register Move Tests (via VEX encoding)
// =========================================================================

#[test]
fn test_vmovapd_xmm_rr_basic() {
    // Uses L=0 for 128-bit
    let enc = super::encode_vmovapd_xmm_rr(Ymm::Ymm0, Ymm::Ymm1);
    assert!(enc.len() >= 3);
}

#[test]
fn test_vmovaps_xmm_rr_basic() {
    let enc = super::encode_vmovaps_xmm_rr(Ymm::Ymm4, Ymm::Ymm5);
    assert!(enc.len() >= 3);
}

#[test]
fn test_vmovdqa_xmm_rr_basic() {
    let enc = super::encode_vmovdqa_xmm_rr(Ymm::Ymm6, Ymm::Ymm7);
    assert!(enc.len() >= 3);
}

#[test]
fn test_ymm_move_all_registers() {
    // Test move instruction encoding for all YMM register pairs
    for i in 0..16u8 {
        for j in 0..16u8 {
            let src = Ymm::from_encoding(i).unwrap();
            let dst = Ymm::from_encoding(j).unwrap();

            // All combinations should encode without panic
            let enc = super::encode_vmovapd_ymm_rr(dst, src);
            assert!(enc.len() >= 3, "Failed for YMM{} -> YMM{}", i, j);
        }
    }
}

#[test]
fn test_ymm_move_self_encoding() {
    // Moving register to itself should still encode correctly
    for i in 0..16u8 {
        let reg = Ymm::from_encoding(i).unwrap();
        let enc = super::encode_vmovapd_ymm_rr(reg, reg);
        assert!(enc.len() >= 3);
    }
}
