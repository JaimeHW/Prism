//! Spill Code Generation
//!
//! Handles spilling virtual registers to the stack when
//! there are not enough physical registers.
//!
//! # Design
//!
//! - Stack slots are allocated in 8-byte units (64-bit)
//! - Spill slots are relative to the frame pointer
//! - Reload/spill instructions are inserted at use/def points

use std::fmt;

// =============================================================================
// Spill Slot
// =============================================================================

/// A stack slot for storing spilled values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpillSlot(u32);

impl SpillSlot {
    /// Create a new spill slot.
    #[inline]
    pub const fn new(index: u32) -> Self {
        SpillSlot(index)
    }

    /// Get the slot index.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// Get the stack offset from the frame pointer.
    /// Spill slots grow downward from RBP.
    #[inline]
    pub const fn offset(self) -> i32 {
        // Stack layout:
        // [RBP]      <- saved frame pointer
        // [RBP - 8]  <- spill slot 0
        // [RBP - 16] <- spill slot 1
        // ...
        -((self.0 as i32 + 1) * 8)
    }

    /// Get the size of this spill slot in bytes.
    #[inline]
    pub const fn size() -> u32 {
        8 // All slots are 8 bytes (64-bit)
    }
}

impl fmt::Display for SpillSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[rbp{}]", self.offset())
    }
}

// =============================================================================
// Spill Location
// =============================================================================

/// Location where a spill/reload should occur.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillLocation {
    /// Before an instruction.
    Before(u32),
    /// After an instruction.
    After(u32),
}

// =============================================================================
// Spill Code
// =============================================================================

/// Spill code to be inserted.
#[derive(Debug, Clone)]
pub struct SpillCode {
    /// Spill operations (store to stack).
    spills: Vec<SpillOp>,
    /// Reload operations (load from stack).
    reloads: Vec<ReloadOp>,
    /// Total stack space needed for spill slots.
    total_spill_size: u32,
}

/// A spill operation (store value to stack).
#[derive(Debug, Clone)]
pub struct SpillOp {
    /// Where to insert the spill.
    pub location: SpillLocation,
    /// The spill slot to store to.
    pub slot: SpillSlot,
    /// Data source (register or virtual register).
    pub source: SpillSource,
}

/// A reload operation (load value from stack).
#[derive(Debug, Clone)]
pub struct ReloadOp {
    /// Where to insert the reload.
    pub location: SpillLocation,
    /// The spill slot to load from.
    pub slot: SpillSlot,
    /// Destination register.
    pub dest: ReloadDest,
}

/// Source of data for a spill.
#[derive(Debug, Clone, Copy)]
pub enum SpillSource {
    /// From a GPR.
    Gpr(crate::backend::x64::registers::Gpr),
    /// From an XMM register.
    Xmm(crate::backend::x64::registers::Xmm),
}

/// Destination for a reload.
#[derive(Debug, Clone, Copy)]
pub enum ReloadDest {
    /// To a GPR.
    Gpr(crate::backend::x64::registers::Gpr),
    /// To an XMM register.
    Xmm(crate::backend::x64::registers::Xmm),
}

impl SpillCode {
    /// Create a new empty spill code container.
    pub fn new() -> Self {
        SpillCode {
            spills: Vec::new(),
            reloads: Vec::new(),
            total_spill_size: 0,
        }
    }

    /// Add a spill operation.
    pub fn add_spill(&mut self, location: SpillLocation, slot: SpillSlot, source: SpillSource) {
        // Track maximum slot for size calculation
        let slot_end = (slot.index() + 1) * 8;
        if slot_end > self.total_spill_size {
            self.total_spill_size = slot_end;
        }

        self.spills.push(SpillOp {
            location,
            slot,
            source,
        });
    }

    /// Add a reload operation.
    pub fn add_reload(&mut self, location: SpillLocation, slot: SpillSlot, dest: ReloadDest) {
        let slot_end = (slot.index() + 1) * 8;
        if slot_end > self.total_spill_size {
            self.total_spill_size = slot_end;
        }

        self.reloads.push(ReloadOp {
            location,
            slot,
            dest,
        });
    }

    /// Get all spill operations.
    pub fn spills(&self) -> &[SpillOp] {
        &self.spills
    }

    /// Get all reload operations.
    pub fn reloads(&self) -> &[ReloadOp] {
        &self.reloads
    }

    /// Get the total stack space needed for spill slots.
    pub fn total_spill_size(&self) -> u32 {
        self.total_spill_size
    }

    /// Check if there are any spill/reload operations.
    pub fn is_empty(&self) -> bool {
        self.spills.is_empty() && self.reloads.is_empty()
    }

    /// Get the number of spill operations.
    pub fn num_spills(&self) -> usize {
        self.spills.len()
    }

    /// Get the number of reload operations.
    pub fn num_reloads(&self) -> usize {
        self.reloads.len()
    }
}

impl Default for SpillCode {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Spill Weight Calculator
// =============================================================================

/// Calculates spill weights for intervals.
pub struct SpillWeightCalculator {
    /// Weight for definitions.
    def_weight: f32,
    /// Weight for uses.
    use_weight: f32,
    /// Multiplier for loop nesting.
    loop_weight: f32,
}

impl SpillWeightCalculator {
    /// Create a new calculator with default weights.
    pub fn new() -> Self {
        SpillWeightCalculator {
            def_weight: 1.0,
            use_weight: 1.0,
            loop_weight: 10.0,
        }
    }

    /// Create with custom loop weight.
    pub fn with_loop_weight(loop_weight: f32) -> Self {
        SpillWeightCalculator {
            def_weight: 1.0,
            use_weight: 1.0,
            loop_weight,
        }
    }

    /// Calculate spill weight for an interval.
    /// Higher weight = less desirable to spill.
    pub fn calculate(&self, interval: &super::interval::LiveInterval, loop_depths: &[u8]) -> f32 {
        let mut weight = 0.0f32;

        for use_pos in interval.uses() {
            let inst_idx = use_pos.pos.inst_index() as usize;
            let loop_depth = loop_depths.get(inst_idx).copied().unwrap_or(0);
            let loop_factor = self.loop_weight.powi(loop_depth as i32);

            match use_pos.kind {
                super::interval::UseKind::Def | super::interval::UseKind::PhiOutput => {
                    weight += self.def_weight * loop_factor;
                }
                super::interval::UseKind::Use
                | super::interval::UseKind::DefUse
                | super::interval::UseKind::PhiInput => {
                    weight += self.use_weight * loop_factor;
                }
            }
        }

        // Normalize by interval length to favor spilling long-lived values
        let length = interval.total_len().max(1) as f32;
        weight / length
    }
}

impl Default for SpillWeightCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spill_slot_offset() {
        assert_eq!(SpillSlot::new(0).offset(), -8);
        assert_eq!(SpillSlot::new(1).offset(), -16);
        assert_eq!(SpillSlot::new(2).offset(), -24);
    }

    #[test]
    fn test_spill_code() {
        use crate::backend::x64::registers::Gpr;

        let mut code = SpillCode::new();

        let slot0 = SpillSlot::new(0);
        let slot1 = SpillSlot::new(1);

        code.add_spill(SpillLocation::After(5), slot0, SpillSource::Gpr(Gpr::Rax));
        code.add_reload(SpillLocation::Before(10), slot0, ReloadDest::Gpr(Gpr::Rcx));
        code.add_spill(SpillLocation::After(15), slot1, SpillSource::Gpr(Gpr::Rdx));

        assert_eq!(code.num_spills(), 2);
        assert_eq!(code.num_reloads(), 1);
        assert_eq!(code.total_spill_size(), 16); // 2 slots * 8 bytes
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
}
