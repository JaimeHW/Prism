use super::*;

#[test]
fn test_frame_layout_basic() {
    let layout = FrameLayout::minimal(8);

    assert_eq!(layout.num_registers, 8);
    assert_eq!(layout.num_spills, 0);

    // Check that we can get register slots
    let slot0 = layout.register_slot(0);
    let slot7 = layout.register_slot(7);

    assert!(slot0.disp < 0); // Below RBP
    assert!(slot7.disp > slot0.disp); // Higher index = more positive offset from base
}

#[test]
fn test_frame_layout_with_spills() {
    let saved = GprSet::EMPTY.insert(Gpr::Rbx).insert(Gpr::R12);
    let layout = FrameLayout::new(16, 4, saved);

    assert_eq!(layout.num_registers, 16);
    assert_eq!(layout.num_spills, 4);
    assert!(layout.saved_regs.contains(Gpr::Rbx));
    assert!(layout.saved_regs.contains(Gpr::R12));

    // Frame should be 16-byte aligned
    assert_eq!(layout.frame_size() % 16, 0);
}

#[test]
fn test_register_slot_ordering() {
    let layout = FrameLayout::minimal(4);

    let slot0 = layout.register_slot(0);
    let slot1 = layout.register_slot(1);
    let slot2 = layout.register_slot(2);
    let slot3 = layout.register_slot(3);

    // Registers should be at consecutive 8-byte offsets (growing from base)
    assert_eq!(slot1.disp - slot0.disp, 8);
    assert_eq!(slot2.disp - slot1.disp, 8);
    assert_eq!(slot3.disp - slot2.disp, 8);
}

#[test]
fn test_spill_slots() {
    let layout = FrameLayout::new(4, 2, GprSet::EMPTY);

    let spill0 = layout.spill_slot(0);
    let spill1 = layout.spill_slot(1);

    // Spills should be below registers (more negative offset)
    assert!(spill0.disp < layout.register_slot(3).disp);
    // Consecutive (growing from base)
    assert_eq!(spill1.disp - spill0.disp, 8);
}

#[test]
fn test_register_assignment() {
    let assign = RegisterAssignment::host();

    // Basic sanity checks
    assert_eq!(assign.accumulator, Gpr::Rax);
    assert_ne!(assign.scratch1, assign.scratch2);
    assert_ne!(assign.context, assign.accumulator);
}

#[test]
fn test_jit_calling_convention() {
    let cc = JitCallingConvention::host();

    assert_eq!(cc.return_reg, Gpr::Rax);
    assert_ne!(cc.arg0, cc.arg1);
}
