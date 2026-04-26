use super::*;

// =========================================================================
// Kreg Tests
// =========================================================================

#[test]
fn test_kreg_encoding() {
    assert_eq!(Kreg::K0.encoding(), 0);
    assert_eq!(Kreg::K1.encoding(), 1);
    assert_eq!(Kreg::K7.encoding(), 7);
}

#[test]
fn test_kreg_from_encoding() {
    assert_eq!(Kreg::from_encoding(0), Some(Kreg::K0));
    assert_eq!(Kreg::from_encoding(7), Some(Kreg::K7));
    assert_eq!(Kreg::from_encoding(8), None);
}

#[test]
fn test_kreg_is_reserved() {
    assert!(Kreg::K0.is_reserved());
    assert!(!Kreg::K1.is_reserved());
    assert!(!Kreg::K7.is_reserved());
}

#[test]
fn test_kreg_display() {
    assert_eq!(format!("{}", Kreg::K0), "k0");
    assert_eq!(format!("{}", Kreg::K5), "k5");
}

// =========================================================================
// KregSet Tests
// =========================================================================

#[test]
fn test_kreg_set_empty() {
    let set = KregSet::EMPTY;
    assert!(set.is_empty());
    assert_eq!(set.count(), 0);
    assert!(!set.contains(Kreg::K0));
}

#[test]
fn test_kreg_set_all() {
    let set = KregSet::ALL;
    assert!(!set.is_empty());
    assert_eq!(set.count(), 8);
    for i in 0..8 {
        assert!(set.contains(Kreg::from_encoding(i).unwrap()));
    }
}

#[test]
fn test_kreg_set_allocatable() {
    let set = KregSet::ALLOCATABLE;
    assert_eq!(set.count(), 7);
    assert!(!set.contains(Kreg::K0));
    for i in 1..8 {
        assert!(set.contains(Kreg::from_encoding(i).unwrap()));
    }
}

#[test]
fn test_kreg_set_singleton() {
    let set = KregSet::singleton(Kreg::K3);
    assert_eq!(set.count(), 1);
    assert!(set.contains(Kreg::K3));
    assert!(!set.contains(Kreg::K2));
}

#[test]
fn test_kreg_set_insert_remove() {
    let set = KregSet::EMPTY.insert(Kreg::K1).insert(Kreg::K5);
    assert_eq!(set.count(), 2);
    assert!(set.contains(Kreg::K1));
    assert!(set.contains(Kreg::K5));

    let removed = set.remove(Kreg::K1);
    assert_eq!(removed.count(), 1);
    assert!(!removed.contains(Kreg::K1));
    assert!(removed.contains(Kreg::K5));
}

#[test]
fn test_kreg_set_union_intersection() {
    let a = KregSet::EMPTY.insert(Kreg::K1).insert(Kreg::K2);
    let b = KregSet::EMPTY.insert(Kreg::K2).insert(Kreg::K3);

    let union = a.union(b);
    assert_eq!(union.count(), 3);
    assert!(union.contains(Kreg::K1));
    assert!(union.contains(Kreg::K2));
    assert!(union.contains(Kreg::K3));

    let intersection = a.intersection(b);
    assert_eq!(intersection.count(), 1);
    assert!(intersection.contains(Kreg::K2));
}

#[test]
fn test_kreg_set_iter() {
    let set = KregSet::EMPTY
        .insert(Kreg::K1)
        .insert(Kreg::K3)
        .insert(Kreg::K7);
    let regs: Vec<_> = set.iter().collect();
    assert_eq!(regs, vec![Kreg::K1, Kreg::K3, Kreg::K7]);
}

#[test]
fn test_kreg_set_first() {
    let set = KregSet::EMPTY.insert(Kreg::K3).insert(Kreg::K1);
    assert_eq!(set.first(), Some(Kreg::K1));
    assert_eq!(KregSet::EMPTY.first(), None);
}

// =========================================================================
// VectorCallingConvention Tests
// =========================================================================

#[test]
fn test_vcc_host() {
    let vcc = VectorCallingConvention::host();
    // Should have valid argument registers
    assert!(!vcc.int_arg_regs().is_empty());
    assert!(!vcc.xmm_arg_regs().is_empty());
    assert!(!vcc.ymm_arg_regs().is_empty());
    assert!(!vcc.zmm_arg_regs().is_empty());
}

#[test]
fn test_vcc_windows() {
    let vcc = VectorCallingConvention::WINDOWS;

    // Integer args: RCX, RDX, R8, R9
    assert_eq!(vcc.int_arg_regs().len(), 4);
    assert_eq!(vcc.int_arg_regs()[0], Gpr::Rcx);

    // Vector args: 4 registers
    assert_eq!(vcc.max_vector_args(), 4);
    assert_eq!(vcc.ymm_arg_regs()[0], Ymm::Ymm0);
    assert_eq!(vcc.zmm_arg_regs()[0], Zmm::Zmm0);

    // Shadow space
    assert_eq!(vcc.shadow_space(), 32);
    assert_eq!(vcc.red_zone(), 0);
}

#[test]
fn test_vcc_sysv() {
    let vcc = VectorCallingConvention::SYSV;

    // Integer args: RDI, RSI, RDX, RCX, R8, R9
    assert_eq!(vcc.int_arg_regs().len(), 6);
    assert_eq!(vcc.int_arg_regs()[0], Gpr::Rdi);

    // Vector args: 8 registers
    assert_eq!(vcc.max_vector_args(), 8);
    assert_eq!(vcc.ymm_arg_regs().len(), 8);
    assert_eq!(vcc.zmm_arg_regs().len(), 8);

    // No shadow space, 128-byte red zone
    assert_eq!(vcc.shadow_space(), 0);
    assert_eq!(vcc.red_zone(), 128);
}

#[test]
fn test_vcc_return_regs() {
    let vcc = VectorCallingConvention::host();

    assert_eq!(vcc.int_return_reg(), Gpr::Rax);
    assert_eq!(vcc.xmm_return_reg(), Xmm::Xmm0);
    assert_eq!(vcc.ymm_return_reg(), Ymm::Ymm0);
    assert_eq!(vcc.zmm_return_reg(), Zmm::Zmm0);
}

#[test]
fn test_vcc_windows_return_regs() {
    let vcc = VectorCallingConvention::WINDOWS;
    // Windows returns in single register
    assert_eq!(vcc.xmm_return_regs().len(), 1);
    assert_eq!(vcc.ymm_return_regs().len(), 1);
}

#[test]
fn test_vcc_sysv_return_regs() {
    let vcc = VectorCallingConvention::SYSV;
    // SysV can return in up to 2 registers
    assert_eq!(vcc.xmm_return_regs().len(), 2);
    assert_eq!(vcc.ymm_return_regs().len(), 2);
}

#[test]
fn test_vcc_volatile_ymm_all() {
    // YMM upper bits are always volatile
    let windows = VectorCallingConvention::WINDOWS;
    let sysv = VectorCallingConvention::SYSV;

    assert_eq!(windows.volatile_ymms(), YmmSet::ALL);
    assert_eq!(sysv.volatile_ymms(), YmmSet::ALL);
}

#[test]
fn test_vcc_volatile_zmm_all() {
    // ZMM registers are always volatile
    let windows = VectorCallingConvention::WINDOWS;
    let sysv = VectorCallingConvention::SYSV;

    assert_eq!(windows.volatile_zmms(), ZmmSet::ALL);
    assert_eq!(sysv.volatile_zmms(), ZmmSet::ALL);
}

#[test]
fn test_vcc_no_callee_saved_ymm() {
    let windows = VectorCallingConvention::WINDOWS;
    let sysv = VectorCallingConvention::SYSV;

    assert_eq!(windows.callee_saved_ymms(), YmmSet::EMPTY);
    assert_eq!(sysv.callee_saved_ymms(), YmmSet::EMPTY);
}

#[test]
fn test_vcc_volatile_kregs() {
    let vcc = VectorCallingConvention::host();
    let kregs = vcc.volatile_kregs();

    // k1-k7 are volatile, k0 is not counted
    assert_eq!(kregs.count(), 7);
    assert!(!kregs.contains(Kreg::K0));
    for i in 1..8 {
        assert!(kregs.contains(Kreg::from_encoding(i).unwrap()));
    }
}

#[test]
fn test_vcc_stack_alignment() {
    let vcc = VectorCallingConvention::host();

    assert_eq!(vcc.stack_alignment(), 16);
    assert_eq!(vcc.ymm_stack_alignment(), 32);
    assert_eq!(vcc.zmm_stack_alignment(), 64);
}

#[test]
fn test_vcc_windows_needs_ymm_upper_save() {
    let windows = VectorCallingConvention::WINDOWS;
    let sysv = VectorCallingConvention::SYSV;

    assert!(windows.needs_ymm_upper_save());
    assert!(!sysv.needs_ymm_upper_save());
}

// =========================================================================
// CallClobbers Tests
// =========================================================================

#[test]
fn test_call_clobbers_windows() {
    let vcc = VectorCallingConvention::WINDOWS;
    let clobbers = vcc.call_clobbers();

    // Windows volatile GPRs: RAX, RCX, RDX, R8-R11
    assert!(clobbers.clobbers_gpr(Gpr::Rax));
    assert!(clobbers.clobbers_gpr(Gpr::Rcx));
    assert!(clobbers.clobbers_gpr(Gpr::R11));
    assert!(!clobbers.clobbers_gpr(Gpr::Rbx)); // callee-saved
    assert!(!clobbers.clobbers_gpr(Gpr::R12)); // callee-saved

    // Windows volatile XMMs: XMM0-XMM5
    assert!(clobbers.clobbers_xmm(Xmm::Xmm0));
    assert!(clobbers.clobbers_xmm(Xmm::Xmm5));
    assert!(!clobbers.clobbers_xmm(Xmm::Xmm6)); // callee-saved on Windows

    // All YMMs clobbered (upper bits)
    assert!(clobbers.clobbers_ymm(Ymm::Ymm0));
    assert!(clobbers.clobbers_ymm(Ymm::Ymm15));

    // All ZMMs clobbered
    assert!(clobbers.clobbers_zmm(Zmm::Zmm0));
    assert!(clobbers.clobbers_zmm(Zmm::Zmm31));
}

#[test]
fn test_call_clobbers_sysv() {
    let vcc = VectorCallingConvention::SYSV;
    let clobbers = vcc.call_clobbers();

    // SysV volatile GPRs: RAX, RCX, RDX, RSI, RDI, R8-R11
    assert!(clobbers.clobbers_gpr(Gpr::Rax));
    assert!(clobbers.clobbers_gpr(Gpr::Rdi));
    assert!(clobbers.clobbers_gpr(Gpr::Rsi));
    assert!(!clobbers.clobbers_gpr(Gpr::Rbx)); // callee-saved
    assert!(!clobbers.clobbers_gpr(Gpr::R12)); // callee-saved

    // SysV: ALL XMMs are volatile
    for i in 0..16 {
        assert!(clobbers.clobbers_xmm(Xmm::from_encoding(i).unwrap()));
    }
}

#[test]
fn test_call_clobbers_empty() {
    let clobbers = CallClobbers::empty();
    assert_eq!(clobbers.total_count(), 0);
    assert!(!clobbers.clobbers_gpr(Gpr::Rax));
    assert!(!clobbers.clobbers_xmm(Xmm::Xmm0));
}

#[test]
fn test_call_clobbers_preg() {
    let vcc = VectorCallingConvention::host();
    let clobbers = vcc.call_clobbers();

    // Test PReg queries
    assert!(clobbers.clobbers_preg(PReg::Gpr(Gpr::Rax)));
    assert!(clobbers.clobbers_preg(PReg::Xmm(Xmm::Xmm0)));
    assert!(clobbers.clobbers_preg(PReg::Ymm(Ymm::Ymm0)));
    assert!(clobbers.clobbers_preg(PReg::Zmm(Zmm::Zmm0)));
}

#[test]
fn test_call_clobbers_as_pregs() {
    let vcc = VectorCallingConvention::host();
    let pregs = vcc.call_clobbers_as_pregs();

    // Should have a reasonable number of clobbers
    assert!(!pregs.is_empty());
    assert!(pregs.len() > 10); // At least some GPRs + XMMs + YMMs + ZMMs
}

// =========================================================================
// ArgLocationCalc Tests
// =========================================================================

#[test]
fn test_arg_calc_windows_ints() {
    let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::WINDOWS);

    // First 4 ints go in registers
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rcx));
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rdx));
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R8));
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R9));

    // 5th goes on stack
    if let ArgClass::Stack(offset) = calc.next_int() {
        assert!(offset >= 0);
    } else {
        panic!("Expected stack argument");
    }
}

#[test]
fn test_arg_calc_sysv_ints() {
    let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

    // First 6 ints go in registers
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rdi));
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rsi));
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rdx));
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rcx));
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R8));
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R9));

    // 7th goes on stack
    assert!(matches!(calc.next_int(), ArgClass::Stack(_)));
}

#[test]
fn test_arg_calc_vectors() {
    let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

    // 256-bit vector in YMM
    assert_eq!(calc.next_v256(), ArgClass::Ymm(Ymm::Ymm0));
    assert_eq!(calc.next_v256(), ArgClass::Ymm(Ymm::Ymm1));

    // 512-bit vector in ZMM (continues from where we left off)
    assert_eq!(calc.next_v512(), ArgClass::Zmm(Zmm::Zmm2));
}

#[test]
fn test_arg_calc_mixed_windows() {
    let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::WINDOWS);

    // Windows: int and float args share slots
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rcx));
    assert_eq!(calc.next_f64(), ArgClass::Xmm(Xmm::Xmm1)); // Uses slot 2
    assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R8)); // Uses slot 3
}

#[test]
fn test_arg_calc_stack_alignment() {
    let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

    // Use up all vector regs
    for _ in 0..8 {
        calc.next_f64();
    }

    // Add a small stack arg first
    let _small = calc.next_int();

    // Reset and test YMM alignment
    calc.reset();
    for _ in 0..8 {
        calc.next_v256();
    }

    // Next YMM should be on stack with alignment
    if let ArgClass::Stack(offset) = calc.next_v256() {
        assert!(offset % 32 == 0, "YMM stack arg not 32-byte aligned");
    }
}

#[test]
fn test_arg_calc_zmm_alignment() {
    let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

    // Use up all vector regs
    for _ in 0..8 {
        calc.next_v512();
    }

    // ZMM on stack should be 64-byte aligned
    if let ArgClass::Stack(offset) = calc.next_v512() {
        assert!(offset % 64 == 0, "ZMM stack arg not 64-byte aligned");
    }
}

#[test]
fn test_arg_calc_reset() {
    let mut calc = ArgLocationCalc::new();

    calc.next_int();
    calc.next_f64();
    calc.reset();

    // After reset, should start from beginning
    assert_eq!(calc.gpr_idx, 0);
    assert_eq!(calc.vec_idx, 0);
    assert_eq!(calc.stack_offset, 0);
}

#[test]
fn test_arg_calc_stack_size_aligned() {
    let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

    // Force some stack arguments
    for _ in 0..8 {
        calc.next_int();
    }
    calc.next_int(); // 7th goes on stack (8 bytes)

    // Stack size should be 16-byte aligned
    let size = calc.stack_size();
    assert!(size % 16 == 0, "Stack size {} not 16-byte aligned", size);
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_vcc_volatile_vector_count() {
    let vcc = VectorCallingConvention::host();
    let count = vcc.volatile_vector_count();

    // Should include XMMs + YMMs + kregs
    assert!(count > 20);
}

#[test]
fn test_call_clobbers_iter() {
    let vcc = VectorCallingConvention::host();
    let clobbers = vcc.call_clobbers();

    // Ensure iterators work
    let gpr_count: usize = clobbers.iter_gprs().count();
    let xmm_count: usize = clobbers.iter_xmms().count();
    let ymm_count: usize = clobbers.iter_ymms().count();
    let zmm_count: usize = clobbers.iter_zmms().count();

    assert!(gpr_count > 0);
    assert!(xmm_count > 0);
    assert!(ymm_count > 0);
    assert!(zmm_count > 0);
}

#[test]
fn test_kreg_set_debug_format() {
    let set = KregSet::EMPTY.insert(Kreg::K1).insert(Kreg::K3);
    let debug_str = format!("{:?}", set);
    assert!(debug_str.contains("k1"));
    assert!(debug_str.contains("k3"));
}
