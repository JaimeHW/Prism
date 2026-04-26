use super::*;

#[test]
fn test_vreg_creation() {
    let v1 = VReg::new(0);
    let v2 = VReg::new(1);
    assert_ne!(v1, v2);
    assert_eq!(v1.index(), 0);
    assert_eq!(v2.index(), 1);
}

#[test]
fn test_preg_display() {
    let gpr = PReg::Gpr(Gpr::Rax);
    let xmm = PReg::Xmm(Xmm::Xmm0);
    assert_eq!(format!("{}", gpr), "rax");
    assert_eq!(format!("{}", xmm), "xmm0");
}

#[test]
fn test_allocation_map() {
    let mut map = AllocationMap::new();
    let v1 = VReg::new(0);
    let v2 = VReg::new(1);

    map.set(v1, Allocation::Register(PReg::Gpr(Gpr::Rax)));
    let slot = map.alloc_spill_slot();
    map.set(v2, Allocation::Spill(slot));

    assert!(map.get(v1).is_register());
    assert!(map.get(v2).is_spill());
    assert_eq!(map.spill_slot_count(), 1);
}

#[test]
fn test_default_config() {
    let config = AllocatorConfig::default();
    // Should have 14 GPRs (16 - RSP - R11)
    assert_eq!(config.available_gprs.count(), 14);
    // Should have 15 XMMs (16 - XMM15)
    assert_eq!(config.available_xmms.count(), 15);
}
