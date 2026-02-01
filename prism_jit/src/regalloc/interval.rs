//! Live Interval Representation
//!
//! This module provides the core data structures for representing
//! live ranges and intervals for register allocation.
//!
//! # Key Concepts
//!
//! - **Program Point**: A position in the instruction stream
//! - **Live Range**: A single contiguous span where a value is live
//! - **Live Interval**: A collection of live ranges for a virtual register
//!
//! # Performance Notes
//!
//! - Ranges are stored sorted for O(log n) intersection checks
//! - Program points use u32 for cache efficiency
//! - Intervals support efficient splitting for live range splitting

use super::{RegClass, VReg};
use std::cmp::Ordering;

// =============================================================================
// Program Point
// =============================================================================

/// A program point representing a position in the instruction sequence.
///
/// Even positions represent instruction boundaries (before the instruction).
/// Odd positions represent within an instruction (useful for tracking def/use).
///
/// This 2x encoding allows distinguishing between "before instruction N"
/// and "after instruction N" without extra space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgPoint(u32);

impl ProgPoint {
    /// The invalid/uninitialized program point.
    pub const INVALID: ProgPoint = ProgPoint(u32::MAX);

    /// Create a program point before an instruction.
    #[inline]
    pub const fn before(inst_idx: u32) -> Self {
        ProgPoint(inst_idx * 2)
    }

    /// Create a program point after an instruction.
    #[inline]
    pub const fn after(inst_idx: u32) -> Self {
        ProgPoint(inst_idx * 2 + 1)
    }

    /// Get the instruction index.
    #[inline]
    pub const fn inst_index(self) -> u32 {
        self.0 / 2
    }

    /// Check if this is a "before" position.
    #[inline]
    pub const fn is_before(self) -> bool {
        self.0 % 2 == 0
    }

    /// Check if this is an "after" position.
    #[inline]
    pub const fn is_after(self) -> bool {
        self.0 % 2 == 1
    }

    /// Get the raw value.
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Create from raw value.
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        ProgPoint(raw)
    }

    /// Get the next program point.
    #[inline]
    pub const fn next(self) -> Self {
        ProgPoint(self.0 + 1)
    }

    /// Get the previous program point.
    #[inline]
    pub const fn prev(self) -> Self {
        ProgPoint(self.0.saturating_sub(1))
    }

    /// Check if this point is valid.
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

impl std::fmt::Display for ProgPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_before() {
            write!(f, "{}b", self.inst_index())
        } else {
            write!(f, "{}a", self.inst_index())
        }
    }
}

// =============================================================================
// Live Range
// =============================================================================

/// A single contiguous span where a value is live.
///
/// The range is half-open: [start, end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    /// Start of the range (inclusive).
    pub start: ProgPoint,
    /// End of the range (exclusive).
    pub end: ProgPoint,
}

impl LiveRange {
    /// Create a new live range.
    #[inline]
    pub const fn new(start: ProgPoint, end: ProgPoint) -> Self {
        LiveRange { start, end }
    }

    /// Check if this range contains a program point.
    #[inline]
    pub const fn contains(&self, point: ProgPoint) -> bool {
        point.raw() >= self.start.raw() && point.raw() < self.end.raw()
    }

    /// Check if this range overlaps with another.
    #[inline]
    pub const fn overlaps(&self, other: &LiveRange) -> bool {
        self.start.raw() < other.end.raw() && other.start.raw() < self.end.raw()
    }

    /// Check if this range is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.start.raw() >= self.end.raw()
    }

    /// Get the length of this range in program points.
    #[inline]
    pub const fn len(&self) -> u32 {
        if self.end.raw() > self.start.raw() {
            self.end.raw() - self.start.raw()
        } else {
            0
        }
    }

    /// Extend this range to include a point.
    pub fn extend_to(&mut self, point: ProgPoint) {
        if point.raw() < self.start.raw() {
            self.start = point;
        }
        if point.raw() >= self.end.raw() {
            self.end = ProgPoint::from_raw(point.raw() + 1);
        }
    }

    /// Merge with another range (must be adjacent or overlapping).
    pub fn merge(&mut self, other: &LiveRange) {
        if other.start.raw() < self.start.raw() {
            self.start = other.start;
        }
        if other.end.raw() > self.end.raw() {
            self.end = other.end;
        }
    }
}

impl PartialOrd for LiveRange {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LiveRange {
    fn cmp(&self, other: &Self) -> Ordering {
        self.start.cmp(&other.start).then(self.end.cmp(&other.end))
    }
}

impl std::fmt::Display for LiveRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

// =============================================================================
// Use Position
// =============================================================================

/// The kind of use at a program point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UseKind {
    /// Definition (output).
    Def,
    /// Use (input).
    Use,
    /// Both def and use (e.g., x = x + 1 in-place).
    DefUse,
    /// Phi input.
    PhiInput,
    /// Phi output.
    PhiOutput,
}

/// A use or def position for a virtual register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsePosition {
    /// The program point.
    pub pos: ProgPoint,
    /// The kind of use.
    pub kind: UseKind,
    /// Whether a register is required (vs. can be spilled).
    pub requires_reg: bool,
}

impl UsePosition {
    /// Create a new use position.
    #[inline]
    pub const fn new(pos: ProgPoint, kind: UseKind, requires_reg: bool) -> Self {
        UsePosition {
            pos,
            kind,
            requires_reg,
        }
    }

    /// Create a def position that requires a register.
    #[inline]
    pub const fn def(pos: ProgPoint) -> Self {
        UsePosition::new(pos, UseKind::Def, true)
    }

    /// Create a use position that requires a register.
    #[inline]
    pub const fn use_pos(pos: ProgPoint) -> Self {
        UsePosition::new(pos, UseKind::Use, true)
    }
}

impl PartialOrd for UsePosition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.pos.cmp(&other.pos))
    }
}

impl Ord for UsePosition {
    fn cmp(&self, other: &Self) -> Ordering {
        self.pos.cmp(&other.pos)
    }
}

// =============================================================================
// Live Interval
// =============================================================================

/// A live interval for a virtual register.
///
/// Contains all the live ranges where the value is live,
/// plus metadata for register allocation decisions.
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// The virtual register this interval is for.
    pub vreg: VReg,
    /// Register class constraint.
    pub reg_class: RegClass,
    /// All live ranges, sorted by start position.
    ranges: Vec<LiveRange>,
    /// All use positions, sorted.
    uses: Vec<UsePosition>,
    /// Spill weight (higher = less desirable to spill).
    pub spill_weight: f32,
    /// Whether this interval can be split.
    pub can_split: bool,
    /// Whether this interval is a split child.
    pub is_split_child: bool,
    /// Parent interval if this is a split.
    pub parent: Option<VReg>,
}

impl LiveInterval {
    /// Create a new empty live interval.
    pub fn new(vreg: VReg, reg_class: RegClass) -> Self {
        LiveInterval {
            vreg,
            reg_class,
            ranges: Vec::new(),
            uses: Vec::new(),
            spill_weight: 1.0,
            can_split: true,
            is_split_child: false,
            parent: None,
        }
    }

    /// Add a live range, merging if adjacent.
    pub fn add_range(&mut self, range: LiveRange) {
        if range.is_empty() {
            return;
        }

        // Find insertion point
        let pos = self.ranges.binary_search(&range).unwrap_or_else(|p| p);

        // Check if we can merge with previous
        if pos > 0 && self.ranges[pos - 1].end.raw() >= range.start.raw() {
            self.ranges[pos - 1].merge(&range);
            // Also check if we need to merge with next
            while pos < self.ranges.len()
                && self.ranges[pos - 1].end.raw() >= self.ranges[pos].start.raw()
            {
                let next = self.ranges.remove(pos);
                self.ranges[pos - 1].merge(&next);
            }
        }
        // Check if we can merge with next
        else if pos < self.ranges.len() && range.end.raw() >= self.ranges[pos].start.raw() {
            self.ranges[pos].merge(&range);
        }
        // Insert as new range
        else {
            self.ranges.insert(pos, range);
        }
    }

    /// Add a use position.
    pub fn add_use(&mut self, use_pos: UsePosition) {
        let pos = self.uses.binary_search(&use_pos).unwrap_or_else(|p| p);
        self.uses.insert(pos, use_pos);
    }

    /// Get the first (earliest) live range.
    pub fn first_range(&self) -> Option<&LiveRange> {
        self.ranges.first()
    }

    /// Get the last (latest) live range.
    pub fn last_range(&self) -> Option<&LiveRange> {
        self.ranges.last()
    }

    /// Get the start position of the interval.
    pub fn start(&self) -> ProgPoint {
        self.ranges
            .first()
            .map(|r| r.start)
            .unwrap_or(ProgPoint::INVALID)
    }

    /// Get the end position of the interval.
    pub fn end(&self) -> ProgPoint {
        self.ranges
            .last()
            .map(|r| r.end)
            .unwrap_or(ProgPoint::INVALID)
    }

    /// Check if the interval contains a point.
    pub fn contains(&self, point: ProgPoint) -> bool {
        // Binary search for the range containing this point
        self.ranges
            .binary_search_by(|r| {
                if r.end.raw() <= point.raw() {
                    Ordering::Less
                } else if r.start.raw() > point.raw() {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_ok()
    }

    /// Check if this interval overlaps with another.
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        let mut i = 0;
        let mut j = 0;

        while i < self.ranges.len() && j < other.ranges.len() {
            if self.ranges[i].overlaps(&other.ranges[j]) {
                return true;
            }
            if self.ranges[i].end <= other.ranges[j].end {
                i += 1;
            } else {
                j += 1;
            }
        }

        false
    }

    /// Get the next use position at or after a point.
    pub fn next_use_after(&self, point: ProgPoint) -> Option<&UsePosition> {
        let pos = self
            .uses
            .binary_search_by(|u| u.pos.cmp(&point))
            .unwrap_or_else(|p| p);
        self.uses.get(pos)
    }

    /// Check if there's a use requiring a register at or after a point.
    pub fn has_register_use_after(&self, point: ProgPoint) -> bool {
        self.uses
            .iter()
            .filter(|u| u.pos >= point)
            .any(|u| u.requires_reg)
    }

    /// Get all use positions.
    pub fn uses(&self) -> &[UsePosition] {
        &self.uses
    }

    /// Get all live ranges.
    pub fn ranges(&self) -> &[LiveRange] {
        &self.ranges
    }

    /// Check if the interval is empty.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Calculate the total length of all ranges.
    pub fn total_len(&self) -> u32 {
        self.ranges.iter().map(|r| r.len()).sum()
    }

    /// Split this interval at a point, returning the second half.
    ///
    /// The split point must be in the middle of the interval.
    /// Returns None if the split is not possible.
    pub fn split_at(&mut self, point: ProgPoint, new_vreg: VReg) -> Option<LiveInterval> {
        if !self.can_split {
            return None;
        }

        // Find the range containing the split point
        let split_idx = self.ranges.iter().position(|r| r.contains(point));

        let mut second_half = LiveInterval::new(new_vreg, self.reg_class);
        second_half.spill_weight = self.spill_weight;
        second_half.is_split_child = true;
        second_half.parent = Some(self.vreg);

        if let Some(idx) = split_idx {
            // Split the range containing the point
            let range = &mut self.ranges[idx];
            let old_end = range.end;
            range.end = point;

            // Add the second half of the split range
            second_half.add_range(LiveRange::new(point, old_end));

            // Move all subsequent ranges to the second half
            for range in self.ranges.drain((idx + 1)..) {
                second_half.add_range(range);
            }

            // Split use positions
            let use_split = self
                .uses
                .iter()
                .position(|u| u.pos >= point)
                .unwrap_or(self.uses.len());
            for use_pos in self.uses.drain(use_split..) {
                second_half.add_use(use_pos);
            }

            Some(second_half)
        } else {
            // Point is in a hole - just move ranges after the point
            let range_idx = self.ranges.iter().position(|r| r.start >= point);

            if let Some(idx) = range_idx {
                for range in self.ranges.drain(idx..) {
                    second_half.add_range(range);
                }

                let use_split = self
                    .uses
                    .iter()
                    .position(|u| u.pos >= point)
                    .unwrap_or(self.uses.len());
                for use_pos in self.uses.drain(use_split..) {
                    second_half.add_use(use_pos);
                }

                Some(second_half)
            } else {
                None
            }
        }
    }
}

impl std::fmt::Display for LiveInterval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: ", self.vreg)?;
        let mut first = true;
        for range in &self.ranges {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "{}", range)?;
            first = false;
        }
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prog_point() {
        let p1 = ProgPoint::before(5);
        let p2 = ProgPoint::after(5);

        assert!(p1 < p2);
        assert!(p1.is_before());
        assert!(p2.is_after());
        assert_eq!(p1.inst_index(), 5);
        assert_eq!(p2.inst_index(), 5);
    }

    #[test]
    fn test_live_range_overlap() {
        let r1 = LiveRange::new(ProgPoint::before(0), ProgPoint::before(10));
        let r2 = LiveRange::new(ProgPoint::before(5), ProgPoint::before(15));
        let r3 = LiveRange::new(ProgPoint::before(10), ProgPoint::before(20));

        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3)); // [0, 10) and [10, 20) don't overlap
    }

    #[test]
    fn test_live_interval_add_range() {
        let mut interval = LiveInterval::new(VReg::new(0), RegClass::Int);

        interval.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(5)));
        interval.add_range(LiveRange::new(ProgPoint::before(10), ProgPoint::before(15)));
        interval.add_range(LiveRange::new(ProgPoint::before(5), ProgPoint::before(10))); // Fills gap

        // Should merge into one range [0, 15)
        assert_eq!(interval.ranges().len(), 1);
        assert_eq!(interval.start(), ProgPoint::before(0));
        assert_eq!(interval.end(), ProgPoint::before(15));
    }

    #[test]
    fn test_live_interval_contains() {
        let mut interval = LiveInterval::new(VReg::new(0), RegClass::Int);
        interval.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(10)));
        interval.add_range(LiveRange::new(ProgPoint::before(20), ProgPoint::before(30)));

        assert!(interval.contains(ProgPoint::before(5)));
        assert!(interval.contains(ProgPoint::before(25)));
        assert!(!interval.contains(ProgPoint::before(15))); // In the hole
    }

    #[test]
    fn test_live_interval_split() {
        let mut interval = LiveInterval::new(VReg::new(0), RegClass::Int);
        interval.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(20)));
        interval.add_use(UsePosition::def(ProgPoint::before(0)));
        interval.add_use(UsePosition::use_pos(ProgPoint::before(5)));
        interval.add_use(UsePosition::use_pos(ProgPoint::before(15)));

        let second = interval.split_at(ProgPoint::before(10), VReg::new(1));

        assert!(second.is_some());
        let second = second.unwrap();

        assert_eq!(interval.end(), ProgPoint::before(10));
        assert_eq!(second.start(), ProgPoint::before(10));
        assert_eq!(second.end(), ProgPoint::before(20));

        // Check uses were split correctly
        // Uses at positions < 10 stay in first half, uses at positions >= 10 go to second
        assert_eq!(interval.uses().len(), 2); // def at 0 and use at 5
        assert_eq!(second.uses().len(), 1); // use at 15
    }
}
