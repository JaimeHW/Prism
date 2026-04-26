use super::*;

#[test]
fn test_spill_slot_offset_legacy() {
    assert_eq!(SpillSlot::new(0).offset(), -8);
    assert_eq!(SpillSlot::new(1).offset(), -16);
    assert_eq!(SpillSlot::new(2).offset(), -24);
}

#[test]
fn test_spill_slot_width() {
    let gpr_slot = SpillSlot::new_with_offset(-8, SpillWidth::W8);
    let xmm_slot = SpillSlot::new_with_offset(-16, SpillWidth::W16);
    let ymm_slot = SpillSlot::new_with_offset(-32, SpillWidth::W32);
    let zmm_slot = SpillSlot::new_with_offset(-64, SpillWidth::W64);

    assert_eq!(gpr_slot.size(), 8);
    assert_eq!(xmm_slot.size(), 16);
    assert_eq!(ymm_slot.size(), 32);
    assert_eq!(zmm_slot.size(), 64);

    assert!(!gpr_slot.is_vector());
    assert!(xmm_slot.is_vector());
    assert!(ymm_slot.is_vector());
    assert!(zmm_slot.is_vector());

    assert!(!gpr_slot.is_wide_vector());
    assert!(!xmm_slot.is_wide_vector());
    assert!(ymm_slot.is_wide_vector());
    assert!(zmm_slot.is_wide_vector());
}

#[test]
fn test_spill_slot_allocator_basic() {
    let mut alloc = SpillSlotAllocator::new();

    let s1 = alloc.alloc_gpr();
    let s2 = alloc.alloc_gpr();
    let s3 = alloc.alloc_gpr();

    assert_eq!(s1.offset(), -8);
    assert_eq!(s2.offset(), -16);
    assert_eq!(s3.offset(), -24);
    assert_eq!(alloc.total_size(), 24);
    assert_eq!(alloc.num_gpr_slots(), 3);
}

#[test]
fn test_spill_slot_allocator_ymm_alignment() {
    let mut alloc = SpillSlotAllocator::new();

    // Allocate a GPR first (offset becomes -8)
    let _gpr = alloc.alloc_gpr();

    // Now allocate YMM - must be 32-byte aligned
    let ymm = alloc.alloc_ymm();

    // YMM slot should be at offset that's a multiple of 32
    assert_eq!(ymm.offset() % 32, 0);
    assert!(ymm.offset() < -8); // Below the GPR slot
    assert_eq!(ymm.size(), 32);
}

#[test]
fn test_spill_slot_allocator_zmm_alignment() {
    let mut alloc = SpillSlotAllocator::new();

    // Allocate a GPR first
    let _gpr = alloc.alloc_gpr();

    // Now allocate ZMM - must be 64-byte aligned
    let zmm = alloc.alloc_zmm();

    // ZMM slot should be at offset that's a multiple of 64
    assert_eq!(zmm.offset() % 64, 0);
    assert!(zmm.offset() < -8);
    assert_eq!(zmm.size(), 64);
}

#[test]
fn test_spill_slot_allocator_mixed_widths() {
    let mut alloc = SpillSlotAllocator::new();

    let gpr1 = alloc.alloc_gpr(); // -8
    let xmm1 = alloc.alloc_xmm(); // -16 aligned, so -16 - 16 = -32 -> aligned to -32
    let gpr2 = alloc.alloc_gpr(); // -32 - 8 = -40
    let ymm1 = alloc.alloc_ymm(); // Must be 32-aligned
    let zmm1 = alloc.alloc_zmm(); // Must be 64-aligned

    // Verify all slots are within bounds
    assert!(gpr1.offset() < 0);
    assert!(xmm1.offset() < gpr1.offset());
    assert!(gpr2.offset() < xmm1.offset());
    assert!(ymm1.offset() < gpr2.offset());
    assert!(zmm1.offset() < ymm1.offset());

    // Verify alignments
    assert_eq!(xmm1.offset() % 16, 0);
    assert_eq!(ymm1.offset() % 32, 0);
    assert_eq!(zmm1.offset() % 64, 0);

    assert_eq!(alloc.num_gpr_slots(), 2);
    assert_eq!(alloc.num_xmm_slots(), 1);
    assert_eq!(alloc.num_ymm_slots(), 1);
    assert_eq!(alloc.num_zmm_slots(), 1);
}

#[test]
fn test_spill_slot_allocator_for_class() {
    let mut alloc = SpillSlotAllocator::new();

    let s1 = alloc.alloc_for_class(super::super::RegClass::Int);
    let s2 = alloc.alloc_for_class(super::super::RegClass::Float);
    let s3 = alloc.alloc_for_class(super::super::RegClass::Vec256);
    let s4 = alloc.alloc_for_class(super::super::RegClass::Vec512);

    assert_eq!(s1.width(), SpillWidth::W8);
    assert_eq!(s2.width(), SpillWidth::W16);
    assert_eq!(s3.width(), SpillWidth::W32);
    assert_eq!(s4.width(), SpillWidth::W64);
}

#[test]
fn test_spill_source_width() {
    assert_eq!(SpillSource::Gpr(Gpr::Rax).width(), SpillWidth::W8);
    assert_eq!(SpillSource::Xmm(Xmm::Xmm0).width(), SpillWidth::W16);
    assert_eq!(SpillSource::Ymm(Ymm::Ymm0).width(), SpillWidth::W32);
    assert_eq!(SpillSource::Zmm(Zmm::Zmm0).width(), SpillWidth::W64);
}

#[test]
fn test_reload_dest_width() {
    assert_eq!(ReloadDest::Gpr(Gpr::Rax).width(), SpillWidth::W8);
    assert_eq!(ReloadDest::Xmm(Xmm::Xmm0).width(), SpillWidth::W16);
    assert_eq!(ReloadDest::Ymm(Ymm::Ymm0).width(), SpillWidth::W32);
    assert_eq!(ReloadDest::Zmm(Zmm::Zmm0).width(), SpillWidth::W64);
}

#[test]
fn test_spill_code() {
    let mut code = SpillCode::new();
    let mut alloc = SpillSlotAllocator::new();

    let gpr_slot = alloc.alloc_gpr();
    let ymm_slot = alloc.alloc_ymm();

    code.add_spill(
        SpillLocation::After(5),
        gpr_slot,
        SpillSource::Gpr(Gpr::Rax),
    );
    code.add_reload(
        SpillLocation::Before(10),
        gpr_slot,
        ReloadDest::Gpr(Gpr::Rcx),
    );
    code.add_spill(
        SpillLocation::After(15),
        ymm_slot,
        SpillSource::Ymm(Ymm::Ymm0),
    );

    assert_eq!(code.num_spills(), 2);
    assert_eq!(code.num_reloads(), 1);
    assert!(code.total_spill_size() >= 40); // At least 8 + 32 bytes
}

#[test]
fn test_spill_weight_basic() {
    use super::super::RegClass;
    use super::super::VReg;
    use super::super::interval::{LiveInterval, LiveRange, ProgPoint, UsePosition};

    let mut interval = LiveInterval::new(VReg::new(0), RegClass::Int);
    interval.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(10)));
    interval.add_use(UsePosition::def(ProgPoint::after(0)));
    interval.add_use(UsePosition::use_pos(ProgPoint::before(5)));
    interval.add_use(UsePosition::use_pos(ProgPoint::before(9)));

    let calc = SpillWeightCalculator::new();
    let loop_depths = vec![0u8; 20];
    let weight = calc.calculate(&interval, &loop_depths);

    assert!(weight > 0.0);
}

#[test]
fn test_spill_weight_loop_matters() {
    use super::super::RegClass;
    use super::super::VReg;
    use super::super::interval::{LiveInterval, LiveRange, ProgPoint, UsePosition};

    // Interval A: used outside loops
    let mut interval_a = LiveInterval::new(VReg::new(0), RegClass::Int);
    interval_a.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(10)));
    interval_a.add_use(UsePosition::use_pos(ProgPoint::before(5)));

    // Interval B: used inside a loop
    let mut interval_b = LiveInterval::new(VReg::new(1), RegClass::Int);
    interval_b.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(10)));
    interval_b.add_use(UsePosition::use_pos(ProgPoint::before(5)));

    let calc = SpillWeightCalculator::new();

    let depths_no_loop = vec![0u8; 10];
    let mut depths_with_loop = vec![0u8; 10];
    depths_with_loop[5] = 1; // Position 5 is in a loop

    let weight_a = calc.calculate(&interval_a, &depths_no_loop);
    let weight_b = calc.calculate(&interval_b, &depths_with_loop);

    // B should have higher weight (less desirable to spill)
    assert!(weight_b > weight_a);
}

#[test]
fn test_spill_width_from_reg_class() {
    use super::super::RegClass;

    assert_eq!(SpillWidth::from_reg_class(RegClass::Int), SpillWidth::W8);
    assert_eq!(SpillWidth::from_reg_class(RegClass::Any), SpillWidth::W8);
    assert_eq!(SpillWidth::from_reg_class(RegClass::Float), SpillWidth::W16);
    assert_eq!(
        SpillWidth::from_reg_class(RegClass::Vec256),
        SpillWidth::W32
    );
    assert_eq!(
        SpillWidth::from_reg_class(RegClass::Vec512),
        SpillWidth::W64
    );
}

#[test]
fn test_spill_slot_allocator_reset() {
    let mut alloc = SpillSlotAllocator::new();

    alloc.alloc_gpr();
    alloc.alloc_ymm();
    alloc.alloc_zmm();

    assert!(alloc.total_size() > 0);
    assert!(alloc.total_slots() > 0);

    alloc.reset();

    assert_eq!(alloc.total_size(), 0);
    assert_eq!(alloc.total_slots(), 0);
}
