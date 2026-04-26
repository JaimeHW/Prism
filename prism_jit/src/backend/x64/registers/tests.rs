use super::*;

#[test]
fn test_gpr_encoding() {
    assert_eq!(Gpr::Rax.encoding(), 0);
    assert_eq!(Gpr::Rcx.encoding(), 1);
    assert_eq!(Gpr::R8.encoding(), 8);
    assert_eq!(Gpr::R15.encoding(), 15);
}

#[test]
fn test_gpr_low_high_bits() {
    assert_eq!(Gpr::Rax.low_bits(), 0);
    assert!(!Gpr::Rax.high_bit());

    assert_eq!(Gpr::R8.low_bits(), 0);
    assert!(Gpr::R8.high_bit());

    assert_eq!(Gpr::R15.low_bits(), 7);
    assert!(Gpr::R15.high_bit());
}

#[test]
fn test_gpr_sib_requirements() {
    assert!(Gpr::Rsp.needs_sib_as_base());
    assert!(Gpr::R12.needs_sib_as_base());
    assert!(!Gpr::Rax.needs_sib_as_base());
}

#[test]
fn test_gpr_displacement_requirements() {
    assert!(Gpr::Rbp.needs_displacement());
    assert!(Gpr::R13.needs_displacement());
    assert!(!Gpr::Rax.needs_displacement());
}

#[test]
fn test_gpr_set_operations() {
    let set = GprSet::EMPTY
        .insert(Gpr::Rax)
        .insert(Gpr::Rcx)
        .insert(Gpr::R8);

    assert!(set.contains(Gpr::Rax));
    assert!(set.contains(Gpr::Rcx));
    assert!(set.contains(Gpr::R8));
    assert!(!set.contains(Gpr::Rdx));
    assert_eq!(set.count(), 3);

    let removed = set.remove(Gpr::Rcx);
    assert!(!removed.contains(Gpr::Rcx));
    assert_eq!(removed.count(), 2);
}

#[test]
fn test_gpr_set_first() {
    let set = GprSet::EMPTY.insert(Gpr::Rdx).insert(Gpr::R8);
    assert_eq!(set.first(), Some(Gpr::Rdx));

    assert_eq!(GprSet::EMPTY.first(), None);
}

#[test]
fn test_gpr_set_iter() {
    let set = GprSet::EMPTY
        .insert(Gpr::Rax)
        .insert(Gpr::Rdx)
        .insert(Gpr::R15);

    let regs: Vec<_> = set.iter().collect();
    assert_eq!(regs, vec![Gpr::Rax, Gpr::Rdx, Gpr::R15]);
}

#[test]
fn test_calling_convention_host() {
    let cc = CallingConvention::host();

    // Verify we have a valid calling convention
    assert!(!cc.int_arg_regs().is_empty());
    assert!(!cc.float_arg_regs().is_empty());
    assert_eq!(cc.int_return_reg(), Gpr::Rax);
    assert_eq!(cc.float_return_reg(), Xmm::Xmm0);
}

#[test]
fn test_windows_calling_convention() {
    let cc = CallingConvention::WindowsX64;

    assert_eq!(cc.int_arg_regs(), &[Gpr::Rcx, Gpr::Rdx, Gpr::R8, Gpr::R9]);
    assert_eq!(cc.shadow_space(), 32);
    assert_eq!(cc.red_zone(), 0);
}

#[test]
fn test_sysv_calling_convention() {
    let cc = CallingConvention::SystemV;

    assert_eq!(
        cc.int_arg_regs(),
        &[Gpr::Rdi, Gpr::Rsi, Gpr::Rdx, Gpr::Rcx, Gpr::R8, Gpr::R9]
    );
    assert_eq!(cc.shadow_space(), 0);
    assert_eq!(cc.red_zone(), 128);
}

#[test]
fn test_allocatable_regs() {
    let alloc = AllocatableRegs::for_host();

    // RSP should never be allocatable
    assert!(!alloc.gprs.contains(Gpr::Rsp));
    // Scratch register should not be allocatable
    assert!(!alloc.gprs.contains(alloc.scratch_gpr));

    // Should have at least 10 allocatable GPRs
    assert!(alloc.gpr_count() >= 10);
}

#[test]
fn test_scale() {
    assert_eq!(Scale::X1.value(), 1);
    assert_eq!(Scale::X2.value(), 2);
    assert_eq!(Scale::X4.value(), 4);
    assert_eq!(Scale::X8.value(), 8);

    assert_eq!(Scale::from_value(4), Some(Scale::X4));
    assert_eq!(Scale::from_value(3), None);
}

#[test]
fn test_mem_operand() {
    let mem = MemOperand::base_disp(Gpr::Rbp, -16);
    assert_eq!(mem.base, Some(Gpr::Rbp));
    assert_eq!(mem.disp, -16);
    assert!(mem.disp_fits_i8());

    let mem_large = MemOperand::base_disp(Gpr::Rax, 1000);
    assert!(!mem_large.disp_fits_i8());

    let mem_sib = MemOperand::base_index(Gpr::Rax, Gpr::Rcx, Scale::X8);
    assert!(mem_sib.needs_sib());
}

#[test]
fn test_xmm_registers() {
    assert_eq!(Xmm::Xmm0.encoding(), 0);
    assert_eq!(Xmm::Xmm8.encoding(), 8);
    assert!(Xmm::Xmm8.high_bit());
    assert!(!Xmm::Xmm7.high_bit());
}

#[test]
fn test_xmm_set() {
    let set = XmmSet::EMPTY.insert(Xmm::Xmm0).insert(Xmm::Xmm8);

    assert!(set.contains(Xmm::Xmm0));
    assert!(set.contains(Xmm::Xmm8));
    assert!(!set.contains(Xmm::Xmm1));
    assert_eq!(set.count(), 2);
}
