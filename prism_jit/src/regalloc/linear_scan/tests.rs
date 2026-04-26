use super::*;
use crate::regalloc::interval::{LiveRange, ProgPoint};

fn make_interval(vreg: u32, start: u32, end: u32, reg_class: RegClass) -> LiveInterval {
    let mut interval = LiveInterval::new(VReg::new(vreg), reg_class);
    interval.add_range(LiveRange::new(
        ProgPoint::before(start),
        ProgPoint::before(end),
    ));
    interval
}

fn make_gpr_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
    make_interval(vreg, start, end, RegClass::Int)
}

fn make_xmm_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
    make_interval(vreg, start, end, RegClass::Float)
}

fn make_ymm_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
    make_interval(vreg, start, end, RegClass::Vec256)
}

fn make_zmm_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
    make_interval(vreg, start, end, RegClass::Vec512)
}

// =========================================================================
// Basic GPR Tests
// =========================================================================

#[test]
fn test_simple_allocation() {
    let intervals = vec![
        make_gpr_interval(0, 0, 10),
        make_gpr_interval(1, 5, 15),
        make_gpr_interval(2, 20, 30),
    ];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    assert!(map.get(VReg::new(0)).is_register());
    assert!(map.get(VReg::new(1)).is_register());
    assert!(map.get(VReg::new(2)).is_register());
    assert_eq!(stats.num_spilled, 0);
}

#[test]
fn test_different_registers_for_overlapping() {
    let intervals = vec![make_gpr_interval(0, 0, 20), make_gpr_interval(1, 10, 30)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, _stats) = allocator.allocate(intervals);

    let r0 = map.get(VReg::new(0)).reg();
    let r1 = map.get(VReg::new(1)).reg();

    assert!(r0.is_some());
    assert!(r1.is_some());
    assert_ne!(r0, r1);
}

#[test]
fn test_register_reuse() {
    let intervals = vec![make_gpr_interval(0, 0, 10), make_gpr_interval(1, 20, 30)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, _stats) = allocator.allocate(intervals);

    let r0 = map.get(VReg::new(0)).reg();
    let r1 = map.get(VReg::new(1)).reg();

    assert_eq!(r0, r1);
}

#[test]
fn test_spill_when_needed() {
    let mut intervals = Vec::new();
    for i in 0..16 {
        intervals.push(make_gpr_interval(i, 0, 100));
    }

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    assert!(stats.num_spilled >= 2);
    assert!(map.spill_slot_count() >= 2);
}

// =========================================================================
// YMM Vector Tests
// =========================================================================

#[test]
fn test_ymm_simple_allocation() {
    let intervals = vec![make_ymm_interval(0, 0, 10)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    let alloc = map.get(VReg::new(0));
    assert!(alloc.is_register());
    assert!(alloc.reg().unwrap().as_ymm().is_some());
    assert_eq!(stats.num_spilled, 0);
    assert_eq!(stats.num_allocated, 1);
}

#[test]
fn test_ymm_multiple_non_overlapping() {
    let intervals = vec![make_ymm_interval(0, 0, 10), make_ymm_interval(1, 20, 30)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    // Both should be allocated, potentially to the same register
    assert!(map.get(VReg::new(0)).is_register());
    assert!(map.get(VReg::new(1)).is_register());
    assert_eq!(stats.num_spilled, 0);
}

#[test]
fn test_ymm_overlapping_different_registers() {
    let intervals = vec![make_ymm_interval(0, 0, 20), make_ymm_interval(1, 10, 30)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, _stats) = allocator.allocate(intervals);

    let r0 = map.get(VReg::new(0)).reg();
    let r1 = map.get(VReg::new(1)).reg();

    assert!(r0.is_some());
    assert!(r1.is_some());
    assert_ne!(r0, r1);
}

// =========================================================================
// ZMM Vector Tests
// =========================================================================

#[test]
fn test_zmm_simple_allocation() {
    let intervals = vec![make_zmm_interval(0, 0, 10)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    let alloc = map.get(VReg::new(0));
    assert!(alloc.is_register());
    assert!(alloc.reg().unwrap().as_zmm().is_some());
    assert_eq!(stats.num_spilled, 0);
}

#[test]
fn test_zmm_prefers_upper_16() {
    // Allocate a single ZMM - should prefer ZMM16-31 (no aliases)
    let intervals = vec![make_zmm_interval(0, 0, 10)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, _stats) = allocator.allocate(intervals);

    let zmm = map.get(VReg::new(0)).reg().unwrap().as_zmm().unwrap();
    assert!(zmm.encoding() >= 16, "Should prefer ZMM16-31");
}

#[test]
fn test_zmm_multiple_overlapping() {
    let mut intervals = Vec::new();
    for i in 0..5 {
        intervals.push(make_zmm_interval(i, 0, 100));
    }

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    // All should be allocated (we have 31 ZMMs available)
    for i in 0..5 {
        assert!(map.get(VReg::new(i)).is_register());
    }
    assert_eq!(stats.num_spilled, 0);
}

// =========================================================================
// Aliasing Tests
// =========================================================================

#[test]
fn test_xmm_ymm_aliasing_conflict() {
    // XMM0 and YMM0 cannot be used simultaneously
    // Allocate XMM first, then YMM - they must get different physical registers
    let intervals = vec![make_xmm_interval(0, 0, 50), make_ymm_interval(1, 0, 50)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    let xmm = map.get(VReg::new(0)).reg();
    let ymm = map.get(VReg::new(1)).reg();

    assert!(xmm.is_some());
    assert!(ymm.is_some());

    // They should NOT have the same encoding
    let xmm_enc = xmm.unwrap().as_xmm().unwrap().encoding();
    let ymm_enc = ymm.unwrap().as_ymm().unwrap().encoding();
    assert_ne!(xmm_enc, ymm_enc, "XMM and YMM should not alias");

    assert_eq!(stats.num_spilled, 0);
}

#[test]
fn test_ymm_zmm_aliasing_conflict() {
    // YMM0-15 and ZMM0-15 cannot be used simultaneously
    let intervals = vec![make_ymm_interval(0, 0, 50), make_zmm_interval(1, 0, 50)];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    let ymm = map.get(VReg::new(0)).reg();
    let zmm = map.get(VReg::new(1)).reg();

    assert!(ymm.is_some());
    assert!(zmm.is_some());

    let ymm_enc = ymm.unwrap().as_ymm().unwrap().encoding();
    let zmm_enc = zmm.unwrap().as_zmm().unwrap().encoding();

    // Either ZMM got a non-aliasing register (16-31) or they're different
    if zmm_enc < 16 {
        assert_ne!(ymm_enc, zmm_enc, "YMM and ZMM0-15 should not alias");
    }

    assert_eq!(stats.num_spilled, 0);
}

#[test]
fn test_zmm16_31_no_aliasing() {
    // ZMM16-31 have no XMM/YMM aliases, so we can use them freely with XMM
    let intervals = vec![
        make_xmm_interval(0, 0, 50),
        make_xmm_interval(1, 0, 50),
        make_xmm_interval(2, 0, 50),
        make_zmm_interval(3, 0, 50), // Should get ZMM16+
    ];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    // All should be allocated
    for i in 0..4 {
        assert!(map.get(VReg::new(i)).is_register());
    }

    // ZMM should be in the upper range
    let zmm = map.get(VReg::new(3)).reg().unwrap().as_zmm().unwrap();
    assert!(zmm.encoding() >= 16);

    assert_eq!(stats.num_spilled, 0);
}

#[test]
fn test_aliasing_with_expiration() {
    // YMM expires, then XMM with same encoding should be usable
    let intervals = vec![
        make_ymm_interval(0, 0, 10),  // YMM, expires at 10
        make_xmm_interval(1, 20, 30), // XMM, starts after YMM expires
    ];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    // Both should be allocated
    assert!(map.get(VReg::new(0)).is_register());
    assert!(map.get(VReg::new(1)).is_register());

    // They CAN have the same encoding since they don't overlap
    // (but don't require it)
    assert_eq!(stats.num_spilled, 0);
}

// =========================================================================
// Mixed Width Tests
// =========================================================================

#[test]
fn test_mixed_width_allocation() {
    let intervals = vec![
        make_gpr_interval(0, 0, 100),
        make_xmm_interval(1, 0, 100),
        make_ymm_interval(2, 0, 100),
        make_zmm_interval(3, 0, 100),
    ];

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    // All should be allocated to correct register types
    assert!(map.get(VReg::new(0)).reg().unwrap().as_gpr().is_some());
    assert!(map.get(VReg::new(1)).reg().unwrap().as_xmm().is_some());
    assert!(map.get(VReg::new(2)).reg().unwrap().as_ymm().is_some());
    assert!(map.get(VReg::new(3)).reg().unwrap().as_zmm().is_some());

    assert_eq!(stats.num_spilled, 0);
}

// =========================================================================
// Spill Tests for Vectors
// =========================================================================

#[test]
fn test_ymm_spill_when_exhausted() {
    // Create 17 overlapping YMM intervals (only 15 available)
    let mut intervals = Vec::new();
    for i in 0..17 {
        intervals.push(make_ymm_interval(i, 0, 100));
    }

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    // Should have at least 2 spills (17 - 15)
    assert!(stats.num_spilled >= 2);
    assert!(map.spill_slot_count() >= 2);
}

#[test]
fn test_zmm_spill_when_exhausted() {
    // Create 33 overlapping ZMM intervals (only 31 available)
    let mut intervals = Vec::new();
    for i in 0..33 {
        intervals.push(make_zmm_interval(i, 0, 100));
    }

    let config = AllocatorConfig::default();
    let allocator = LinearScanAllocator::new(config);
    let (map, stats) = allocator.allocate(intervals);

    // Should have at least 2 spills (33 - 31)
    assert!(stats.num_spilled >= 2);
    assert!(map.spill_slot_count() >= 2);
}

// =========================================================================
// Aliasing Tracker Unit Tests
// =========================================================================

#[test]
fn test_aliasing_tracker_xmm() {
    let mut tracker = AliasingTracker::new();

    assert!(tracker.is_xmm_available(0));
    tracker.use_xmm(0);
    assert!(!tracker.is_xmm_available(0));

    tracker.release_xmm(0);
    assert!(tracker.is_xmm_available(0));
}

#[test]
fn test_aliasing_tracker_ymm_blocks_xmm() {
    let mut tracker = AliasingTracker::new();

    assert!(tracker.is_xmm_available(5));
    assert!(tracker.is_ymm_available(5));

    tracker.use_ymm(5);

    // XMM5 should be blocked
    assert!(!tracker.is_xmm_available(5));
    assert!(!tracker.is_ymm_available(5));

    tracker.release_ymm(5);

    // Both should be available again
    assert!(tracker.is_xmm_available(5));
    assert!(tracker.is_ymm_available(5));
}

#[test]
fn test_aliasing_tracker_zmm_blocks_both() {
    let mut tracker = AliasingTracker::new();

    assert!(tracker.is_xmm_available(3));
    assert!(tracker.is_ymm_available(3));
    assert!(tracker.is_zmm_available(3));

    tracker.use_zmm(3);

    // All should be blocked
    assert!(!tracker.is_xmm_available(3));
    assert!(!tracker.is_ymm_available(3));
    assert!(!tracker.is_zmm_available(3));

    tracker.release_zmm(3);

    // All should be available again
    assert!(tracker.is_xmm_available(3));
    assert!(tracker.is_ymm_available(3));
    assert!(tracker.is_zmm_available(3));
}

#[test]
fn test_aliasing_tracker_zmm16_no_alias() {
    let mut tracker = AliasingTracker::new();

    // ZMM16 has no XMM/YMM alias
    assert!(tracker.is_zmm_available(16));

    // Using ZMM16 should NOT block any XMM/YMM
    tracker.use_zmm(16);

    assert!(!tracker.is_zmm_available(16));
    // XMM0-15 should still be available
    for i in 0..16 {
        assert!(tracker.is_xmm_available(i));
        assert!(tracker.is_ymm_available(i));
    }
}

#[test]
fn test_aliasing_tracker_layered_blocking() {
    let mut tracker = AliasingTracker::new();

    // Use both YMM and ZMM for same slot
    tracker.use_ymm(0);
    tracker.use_zmm(0);

    // XMM0 is blocked by both
    assert!(!tracker.is_xmm_available(0));

    // Release YMM - XMM should still be blocked by ZMM
    tracker.release_ymm(0);
    assert!(!tracker.is_xmm_available(0));

    // Release ZMM - now XMM should be available
    tracker.release_zmm(0);
    assert!(tracker.is_xmm_available(0));
}
