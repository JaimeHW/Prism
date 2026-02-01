//! Linear Scan Register Allocator
//!
//! A fast register allocator using the linear scan algorithm.
//! Provides O(n log n) allocation suitable for JIT compilation.
//!
//! # Algorithm Overview
//!
//! 1. Sort intervals by start position
//! 2. Maintain active set of currently live intervals
//! 3. At each interval start, expire old intervals and allocate
//! 4. If no register available, spill the interval with furthest use
//!
//! # References
//!
//! - Poletto & Sarkar, "Linear Scan Register Allocation" (1999)
//! - Wimmer & Franz, "Linear Scan Register Allocation on SSA Form" (2010)

use super::interval::{LiveInterval, ProgPoint};
use super::{Allocation, AllocationMap, AllocatorConfig, AllocatorStats, PReg, RegClass, VReg};
use crate::backend::x64::registers::{GprSet, Xmm, XmmSet};
use std::collections::BinaryHeap;

// =============================================================================
// Active Interval
// =============================================================================

/// An interval in the active set, ordered by end position.
#[derive(Debug, Clone)]
struct ActiveInterval {
    vreg: VReg,
    end: ProgPoint,
    reg: PReg,
}

impl PartialEq for ActiveInterval {
    fn eq(&self, other: &Self) -> bool {
        self.end == other.end
    }
}

impl Eq for ActiveInterval {}

impl PartialOrd for ActiveInterval {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActiveInterval {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap (earliest end first)
        other.end.cmp(&self.end)
    }
}

// =============================================================================
// Linear Scan Allocator
// =============================================================================

/// The linear scan register allocator.
pub struct LinearScanAllocator {
    /// Configuration.
    #[allow(dead_code)]
    config: AllocatorConfig,
    /// Allocation map being built.
    allocations: AllocationMap,
    /// Active intervals (sorted by end position).
    active_gprs: BinaryHeap<ActiveInterval>,
    active_xmms: BinaryHeap<ActiveInterval>,
    /// Available GPRs.
    free_gprs: GprSet,
    /// Available XMMs.
    free_xmms: XmmSet,
    /// Statistics.
    stats: AllocatorStats,
}

impl LinearScanAllocator {
    /// Create a new allocator with the given configuration.
    pub fn new(config: AllocatorConfig) -> Self {
        let free_gprs = config.available_gprs;
        let free_xmms = config.available_xmms;

        LinearScanAllocator {
            config,
            allocations: AllocationMap::new(),
            active_gprs: BinaryHeap::new(),
            active_xmms: BinaryHeap::new(),
            free_gprs,
            free_xmms,
            stats: AllocatorStats::default(),
        }
    }

    /// Allocate registers for a set of intervals.
    pub fn allocate(mut self, mut intervals: Vec<LiveInterval>) -> (AllocationMap, AllocatorStats) {
        self.stats.num_vregs = intervals.len();

        // Sort intervals by start position
        intervals.sort_by_key(|i| i.start());

        // Process each interval
        for interval in &intervals {
            if interval.is_empty() {
                continue;
            }

            let start = interval.start();

            // Expire old intervals
            self.expire_old_intervals(start, interval.reg_class);

            // Try to allocate
            if !self.try_allocate(interval) {
                // No register available - spill
                self.allocate_with_spill(interval);
            }
        }

        (self.allocations, self.stats)
    }

    /// Expire intervals that end before the given position.
    fn expire_old_intervals(&mut self, pos: ProgPoint, reg_class: RegClass) {
        match reg_class {
            RegClass::Int | RegClass::Any => {
                while let Some(active) = self.active_gprs.peek() {
                    if active.end > pos {
                        break;
                    }
                    let expired = self.active_gprs.pop().unwrap();
                    if let Some(gpr) = expired.reg.as_gpr() {
                        self.free_gprs = self.free_gprs.insert(gpr);
                    }
                }
            }
            RegClass::Float => {
                while let Some(active) = self.active_xmms.peek() {
                    if active.end > pos {
                        break;
                    }
                    let expired = self.active_xmms.pop().unwrap();
                    if let Some(xmm) = expired.reg.as_xmm() {
                        self.free_xmms = self.free_xmms.insert(xmm);
                    }
                }
            }
        }
    }

    /// Try to allocate a register for an interval.
    fn try_allocate(&mut self, interval: &LiveInterval) -> bool {
        match interval.reg_class {
            RegClass::Int => self.try_allocate_gpr(interval),
            RegClass::Float => self.try_allocate_xmm(interval),
            RegClass::Any => {
                // Try GPR first, then XMM
                self.try_allocate_gpr(interval) || self.try_allocate_xmm(interval)
            }
        }
    }

    /// Try to allocate a GPR.
    fn try_allocate_gpr(&mut self, interval: &LiveInterval) -> bool {
        if let Some(gpr) = self.free_gprs.first() {
            self.free_gprs = self.free_gprs.remove(gpr);

            let preg = PReg::Gpr(gpr);
            self.allocations
                .set(interval.vreg, Allocation::Register(preg));

            self.active_gprs.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: preg,
            });

            self.stats.num_allocated += 1;
            true
        } else {
            false
        }
    }

    /// Try to allocate an XMM register.
    fn try_allocate_xmm(&mut self, interval: &LiveInterval) -> bool {
        if let Some(xmm) = self.first_xmm() {
            self.free_xmms = self.free_xmms.remove(xmm);

            let preg = PReg::Xmm(xmm);
            self.allocations
                .set(interval.vreg, Allocation::Register(preg));

            self.active_xmms.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: preg,
            });

            self.stats.num_allocated += 1;
            true
        } else {
            false
        }
    }

    /// Get the first available XMM register.
    fn first_xmm(&self) -> Option<Xmm> {
        for i in 0..16 {
            if let Some(xmm) = Xmm::from_encoding(i) {
                if self.free_xmms.contains(xmm) {
                    return Some(xmm);
                }
            }
        }
        None
    }

    /// Allocate with spilling.
    fn allocate_with_spill(&mut self, interval: &LiveInterval) {
        match interval.reg_class {
            RegClass::Int => self.allocate_with_spill_gpr(interval),
            RegClass::Float => self.allocate_with_spill_xmm(interval),
            RegClass::Any => self.allocate_with_spill_gpr(interval),
        }
    }

    /// Allocate a GPR with potential spilling.
    fn allocate_with_spill_gpr(&mut self, interval: &LiveInterval) {
        // Find the interval with the furthest next use
        let mut spill_candidate: Option<(VReg, ProgPoint, PReg)> = None;

        // Check active intervals for best spill candidate
        for active in self.active_gprs.iter() {
            if active.end > interval.end() {
                match &spill_candidate {
                    None => spill_candidate = Some((active.vreg, active.end, active.reg)),
                    Some((_, furthest, _)) if active.end > *furthest => {
                        spill_candidate = Some((active.vreg, active.end, active.reg));
                    }
                    _ => {}
                }
            }
        }

        if let Some((spill_vreg, _, spill_reg)) = spill_candidate {
            // Spill the candidate
            let slot = self.allocations.alloc_spill_slot();
            self.allocations.set(spill_vreg, Allocation::Spill(slot));

            // Give its register to the current interval
            self.allocations
                .set(interval.vreg, Allocation::Register(spill_reg));

            // Update active set: remove spilled, add new
            self.active_gprs.retain(|a| a.vreg != spill_vreg);
            self.active_gprs.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: spill_reg,
            });

            self.stats.num_spilled += 1;
            self.stats.num_allocated += 1;
        } else {
            // Spill the current interval
            let slot = self.allocations.alloc_spill_slot();
            self.allocations.set(interval.vreg, Allocation::Spill(slot));
            self.stats.num_spilled += 1;
        }
    }

    /// Allocate an XMM with potential spilling.
    fn allocate_with_spill_xmm(&mut self, interval: &LiveInterval) {
        let mut spill_candidate: Option<(VReg, ProgPoint, PReg)> = None;

        for active in self.active_xmms.iter() {
            if active.end > interval.end() {
                match &spill_candidate {
                    None => spill_candidate = Some((active.vreg, active.end, active.reg)),
                    Some((_, furthest, _)) if active.end > *furthest => {
                        spill_candidate = Some((active.vreg, active.end, active.reg));
                    }
                    _ => {}
                }
            }
        }

        if let Some((spill_vreg, _, spill_reg)) = spill_candidate {
            let slot = self.allocations.alloc_spill_slot();
            self.allocations.set(spill_vreg, Allocation::Spill(slot));
            self.allocations
                .set(interval.vreg, Allocation::Register(spill_reg));

            self.active_xmms.retain(|a| a.vreg != spill_vreg);
            self.active_xmms.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: spill_reg,
            });

            self.stats.num_spilled += 1;
            self.stats.num_allocated += 1;
        } else {
            let slot = self.allocations.alloc_spill_slot();
            self.allocations.set(interval.vreg, Allocation::Spill(slot));
            self.stats.num_spilled += 1;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regalloc::interval::{LiveRange, ProgPoint};

    fn make_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        let mut interval = LiveInterval::new(VReg::new(vreg), RegClass::Int);
        interval.add_range(LiveRange::new(
            ProgPoint::before(start),
            ProgPoint::before(end),
        ));
        interval
    }

    #[test]
    fn test_simple_allocation() {
        let intervals = vec![
            make_interval(0, 0, 10),
            make_interval(1, 5, 15),
            make_interval(2, 20, 30),
        ];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // All should be allocated to registers (we have 14 GPRs)
        assert!(map.get(VReg::new(0)).is_register());
        assert!(map.get(VReg::new(1)).is_register());
        assert!(map.get(VReg::new(2)).is_register());
        assert_eq!(stats.num_spilled, 0);
    }

    #[test]
    fn test_different_registers_for_overlapping() {
        // These two overlap, so need different registers
        let intervals = vec![make_interval(0, 0, 20), make_interval(1, 10, 30)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, _stats) = allocator.allocate(intervals);

        let r0 = map.get(VReg::new(0)).reg();
        let r1 = map.get(VReg::new(1)).reg();

        assert!(r0.is_some());
        assert!(r1.is_some());
        assert_ne!(r0, r1); // Must be different
    }

    #[test]
    fn test_register_reuse() {
        // v0: [0, 10), v1: [20, 30) - can reuse register
        let intervals = vec![make_interval(0, 0, 10), make_interval(1, 20, 30)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, _stats) = allocator.allocate(intervals);

        let r0 = map.get(VReg::new(0)).reg();
        let r1 = map.get(VReg::new(1)).reg();

        // Should reuse the same register
        assert_eq!(r0, r1);
    }

    #[test]
    fn test_spill_when_needed() {
        // Create more overlapping intervals than registers
        // With 14 GPRs, creating 16 overlapping intervals should cause spills
        let mut intervals = Vec::new();
        for i in 0..16 {
            intervals.push(make_interval(i, 0, 100));
        }

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // Should have some spills (16 - 14 = 2 minimum)
        assert!(stats.num_spilled >= 2);
        assert!(map.spill_slot_count() >= 2);
    }
}
