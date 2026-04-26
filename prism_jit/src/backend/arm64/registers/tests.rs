use super::*;

#[test]
fn test_gpr_encoding() {
    assert_eq!(Gpr::X0.encoding(), 0);
    assert_eq!(Gpr::X1.encoding(), 1);
    assert_eq!(Gpr::X15.encoding(), 15);
    assert_eq!(Gpr::X30.encoding(), 30);
}

#[test]
fn test_gpr_all_count() {
    assert_eq!(Gpr::ALL.len(), 31);
}

#[test]
fn test_gpr_categories() {
    assert!(Gpr::X0.is_arg());
    assert!(Gpr::X7.is_arg());
    assert!(!Gpr::X8.is_arg());

    assert!(Gpr::X9.is_temp());
    assert!(Gpr::X15.is_temp());
    assert!(!Gpr::X8.is_temp());
    assert!(!Gpr::X16.is_temp());

    assert!(Gpr::X19.is_callee_saved());
    assert!(Gpr::X28.is_callee_saved());
    assert!(!Gpr::X29.is_callee_saved()); // FP, handled specially

    assert!(Gpr::X18.is_reserved());
    assert!(!Gpr::X17.is_reserved());

    assert!(Gpr::X29.is_fp());
    assert!(Gpr::X30.is_lr());
}

#[test]
fn test_gpr_from_encoding() {
    for i in 0..31u8 {
        let reg = Gpr::from_encoding(i).unwrap();
        assert_eq!(reg.encoding(), i);
    }
    assert!(Gpr::from_encoding(31).is_none());
    assert!(Gpr::from_encoding(32).is_none());
}

#[test]
fn test_gpr_set_singleton() {
    let set = GprSet::singleton(Gpr::X5);
    assert!(set.contains(Gpr::X5));
    assert!(!set.contains(Gpr::X4));
    assert!(!set.contains(Gpr::X6));
    assert_eq!(set.count(), 1);
}

#[test]
fn test_gpr_set_operations() {
    let set1 = GprSet::singleton(Gpr::X0).insert(Gpr::X1);
    let set2 = GprSet::singleton(Gpr::X1).insert(Gpr::X2);

    let union = set1.union(set2);
    assert!(union.contains(Gpr::X0));
    assert!(union.contains(Gpr::X1));
    assert!(union.contains(Gpr::X2));
    assert_eq!(union.count(), 3);

    let intersection = set1.intersection(set2);
    assert!(!intersection.contains(Gpr::X0));
    assert!(intersection.contains(Gpr::X1));
    assert!(!intersection.contains(Gpr::X2));
    assert_eq!(intersection.count(), 1);

    let diff = set1.difference(set2);
    assert!(diff.contains(Gpr::X0));
    assert!(!diff.contains(Gpr::X1));
    assert!(!diff.contains(Gpr::X2));
    assert_eq!(diff.count(), 1);
}

#[test]
fn test_gpr_set_all() {
    assert_eq!(GprSet::ALL.count(), 31);
    for i in 0..31 {
        let reg = Gpr::from_encoding(i).unwrap();
        assert!(GprSet::ALL.contains(reg));
    }
}

#[test]
fn test_gpr_set_iter() {
    let set = GprSet::singleton(Gpr::X3).insert(Gpr::X7).insert(Gpr::X15);

    let regs: Vec<_> = set.iter().collect();
    assert_eq!(regs, vec![Gpr::X3, Gpr::X7, Gpr::X15]);
}

#[test]
fn test_gpr_set_first() {
    let empty = GprSet::EMPTY;
    assert!(empty.first().is_none());

    let set = GprSet::singleton(Gpr::X5).insert(Gpr::X10);
    assert_eq!(set.first(), Some(Gpr::X5));
}

#[test]
fn test_calling_convention_arg_regs() {
    for i in 0..8 {
        let reg = CallingConvention::arg_reg(i).unwrap();
        assert_eq!(reg.encoding() as usize, i);
    }
    assert!(CallingConvention::arg_reg(8).is_none());
}

#[test]
fn test_calling_convention_callee_saved() {
    let callee_saved = CallingConvention::callee_saved_gprs();

    // X19-X28 should be callee-saved
    for i in 19..29 {
        let reg = Gpr::from_encoding(i).unwrap();
        assert!(callee_saved.contains(reg), "X{} should be callee-saved", i);
    }

    // X29 (FP) should also be callee-saved
    assert!(callee_saved.contains(Gpr::X29));

    // X0-X18 and X30 should NOT be callee-saved
    for i in 0..19 {
        let reg = Gpr::from_encoding(i).unwrap();
        assert!(
            !callee_saved.contains(reg),
            "X{} should NOT be callee-saved",
            i
        );
    }
    assert!(!callee_saved.contains(Gpr::X30));
}

#[test]
fn test_allocatable_regs() {
    let alloc = AllocatableRegs::new();

    // Should have 26 allocatable GPRs (31 - 5: X16, X17, X18, X29, X30)
    assert_eq!(alloc.gpr_count(), 26);

    // Scratch registers should not be in allocatable set
    assert!(!alloc.gprs.contains(Gpr::X16));
    assert!(!alloc.gprs.contains(Gpr::X17));

    // Reserved register should not be allocatable
    assert!(!alloc.gprs.contains(Gpr::X18));

    // FP and LR should not be allocatable
    assert!(!alloc.gprs.contains(Gpr::X29));
    assert!(!alloc.gprs.contains(Gpr::X30));

    // Argument registers should be allocatable
    for i in 0..8 {
        let reg = Gpr::from_encoding(i).unwrap();
        assert!(alloc.gprs.contains(reg));
    }
}

#[test]
fn test_mem_operand_display() {
    let base_only = MemOperand::base(Gpr::X0);
    assert_eq!(format!("{}", base_only), "[x0]");

    let base_offset = MemOperand::base_offset(Gpr::X1, 16);
    assert_eq!(format!("{}", base_offset), "[x1, #16]");

    let pre_index = MemOperand::pre_index(Gpr::X2, -16);
    assert_eq!(format!("{}", pre_index), "[x2, #-16]!");

    let post_index = MemOperand::post_index(Gpr::X3, 8);
    assert_eq!(format!("{}", post_index), "[x3], #8");
}

#[test]
fn test_extend_encoding() {
    assert_eq!(Extend::Uxtb.encoding(), 0b000);
    assert_eq!(Extend::Uxth.encoding(), 0b001);
    assert_eq!(Extend::Uxtw.encoding(), 0b010);
    assert_eq!(Extend::Uxtx.encoding(), 0b011);
    assert_eq!(Extend::Sxtb.encoding(), 0b100);
    assert_eq!(Extend::Sxth.encoding(), 0b101);
    assert_eq!(Extend::Sxtw.encoding(), 0b110);
    assert_eq!(Extend::Sxtx.encoding(), 0b111);
}

#[test]
fn test_stack_alignment() {
    assert_eq!(CallingConvention::stack_alignment(), 16);
}

#[test]
fn test_special_registers() {
    assert_eq!(SP_ENCODING, 31);
    assert_eq!(ZR_ENCODING, 31);
}

#[test]
fn test_gpr_names() {
    assert_eq!(Gpr::X0.name_64(), "x0");
    assert_eq!(Gpr::X0.name_32(), "w0");
    assert_eq!(Gpr::X29.name_64(), "fp");
    assert_eq!(Gpr::X30.name_64(), "lr");
    assert_eq!(Gpr::X15.name_64(), "x15");
    assert_eq!(Gpr::X15.name_32(), "w15");
}
