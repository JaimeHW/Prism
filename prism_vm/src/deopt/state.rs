//! Deoptimization State Capture.
//!
//! Provides lazy state capture for deoptimization. Instead of copying
//! all registers, we only record the delta (modified registers since
//! entering JIT code).

use prism_core::Value;
use smallvec::SmallVec;

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of delta entries before spilling to heap.
/// Most deopts modify only a few registers.
pub const MAX_DELTA_ENTRIES: usize = 8;

/// Maximum register slots in a frame.
pub const MAX_REGISTER_SLOTS: usize = 256;

// =============================================================================
// Deopt Reason
// =============================================================================

/// Reason for deoptimization.
///
/// These reasons guide future optimization decisions. A site that
/// repeatedly deopts for the same reason may be patched to bail out
/// unconditionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeoptReason {
    /// Type guard failed - value had unexpected type.
    TypeGuard = 0,
    /// Integer overflow in arithmetic operation.
    Overflow = 1,
    /// Array/string bounds check failed.
    BoundsCheck = 2,
    /// Inline cache miss - unexpected property layout.
    CacheMiss = 3,
    /// Unknown or uncompiled opcode encountered.
    UnknownOp = 4,
    /// Division by zero.
    DivByZero = 5,
    /// Stack overflow detected.
    StackOverflow = 6,
    /// OSR exit - leaving JIT code mid-loop.
    OsrExit = 7,
    /// Uncommon trap - rarely-taken path.
    UncommonTrap = 8,
    /// Polymorphic call site - too many types.
    PolymorphicSite = 9,
    /// Memory allocation failure.
    AllocationFailure = 10,
    /// Explicit deopt request (debugging).
    Explicit = 11,
}

impl DeoptReason {
    /// Convert from raw u8 value.
    #[inline]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::TypeGuard),
            1 => Some(Self::Overflow),
            2 => Some(Self::BoundsCheck),
            3 => Some(Self::CacheMiss),
            4 => Some(Self::UnknownOp),
            5 => Some(Self::DivByZero),
            6 => Some(Self::StackOverflow),
            7 => Some(Self::OsrExit),
            8 => Some(Self::UncommonTrap),
            9 => Some(Self::PolymorphicSite),
            10 => Some(Self::AllocationFailure),
            11 => Some(Self::Explicit),
            _ => None,
        }
    }

    /// Whether this reason indicates the guard should be patched out.
    #[inline]
    pub const fn should_patch_guard(&self) -> bool {
        matches!(
            self,
            Self::PolymorphicSite | Self::UncommonTrap | Self::UnknownOp
        )
    }

    /// Whether this reason can benefit from recompilation.
    #[inline]
    pub const fn triggers_recompile(&self) -> bool {
        matches!(
            self,
            Self::TypeGuard | Self::CacheMiss | Self::PolymorphicSite
        )
    }
}

impl std::fmt::Display for DeoptReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::TypeGuard => "type guard",
            Self::Overflow => "overflow",
            Self::BoundsCheck => "bounds check",
            Self::CacheMiss => "cache miss",
            Self::UnknownOp => "unknown op",
            Self::DivByZero => "division by zero",
            Self::StackOverflow => "stack overflow",
            Self::OsrExit => "OSR exit",
            Self::UncommonTrap => "uncommon trap",
            Self::PolymorphicSite => "polymorphic site",
            Self::AllocationFailure => "allocation failure",
            Self::Explicit => "explicit",
        };
        write!(f, "{}", name)
    }
}

// =============================================================================
// Delta Entry
// =============================================================================

/// A single delta entry: (register slot, value).
#[derive(Debug, Clone)]
pub struct DeltaEntry {
    /// Register slot index.
    pub slot: u8,
    /// Value at this slot.
    pub value: Value,
}

impl DeltaEntry {
    /// Create a new delta entry.
    #[inline]
    pub const fn new(slot: u8, value: Value) -> Self {
        Self { slot, value }
    }
}

// =============================================================================
// Deopt Delta
// =============================================================================

/// Lazy delta of modified registers.
///
/// Uses SmallVec to avoid heap allocation for small deltas (most common case).
#[derive(Debug, Clone, Default)]
pub struct DeoptDelta {
    /// Modified register entries.
    entries: SmallVec<[DeltaEntry; MAX_DELTA_ENTRIES]>,
}

impl DeoptDelta {
    /// Create an empty delta.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: SmallVec::new(),
        }
    }

    /// Create with preallocated capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: SmallVec::with_capacity(cap),
        }
    }

    /// Record a modified register.
    #[inline]
    pub fn record(&mut self, slot: u8, value: Value) {
        // Check if we already have this slot
        for entry in &mut self.entries {
            if entry.slot == slot {
                entry.value = value;
                return;
            }
        }
        self.entries.push(DeltaEntry::new(slot, value));
    }

    /// Get the value for a slot, if modified.
    #[inline]
    pub fn get(&self, slot: u8) -> Option<Value> {
        self.entries
            .iter()
            .find(|e| e.slot == slot)
            .map(|e| e.value)
    }

    /// Iterate over all entries.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &DeltaEntry> {
        self.entries.iter()
    }

    /// Number of modified registers.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no registers were modified.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// =============================================================================
// Deopt State
// =============================================================================

/// Complete deoptimization state for frame reconstruction.
#[derive(Debug, Clone)]
pub struct DeoptState {
    /// Bytecode offset to resume at.
    pub bc_offset: u32,
    /// Reason for deoptimization.
    pub reason: DeoptReason,
    /// Deopt ID (index into trampoline table).
    pub deopt_id: u32,
    /// Code object ID.
    pub code_id: u64,
    /// Modified register delta.
    pub delta: DeoptDelta,
    /// Timestamp of deoptimization (for stats).
    pub timestamp: u64,
}

impl DeoptState {
    /// Create a new deopt state.
    #[inline]
    pub fn new(bc_offset: u32, reason: DeoptReason, deopt_id: u32, code_id: u64) -> Self {
        Self {
            bc_offset,
            reason,
            deopt_id,
            code_id,
            delta: DeoptDelta::new(),
            timestamp: Self::current_timestamp(),
        }
    }

    /// Create with a prebuilt delta.
    #[inline]
    pub fn with_delta(
        bc_offset: u32,
        reason: DeoptReason,
        deopt_id: u32,
        code_id: u64,
        delta: DeoptDelta,
    ) -> Self {
        Self {
            bc_offset,
            reason,
            deopt_id,
            code_id,
            delta,
            timestamp: Self::current_timestamp(),
        }
    }

    /// Record a modified register.
    #[inline]
    pub fn record_modified(&mut self, slot: u8, value: Value) {
        self.delta.record(slot, value);
    }

    /// Get current timestamp.
    #[inline]
    fn current_timestamp() -> u64 {
        // Use TSC or monotonic clock
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: rdtsc is always available on x86-64
            unsafe { std::arch::x86_64::_rdtsc() }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::time::Instant::now().elapsed().as_nanos() as u64
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deopt_reason_from_u8() {
        assert_eq!(DeoptReason::from_u8(0), Some(DeoptReason::TypeGuard));
        assert_eq!(DeoptReason::from_u8(5), Some(DeoptReason::DivByZero));
        assert_eq!(DeoptReason::from_u8(255), None);
    }

    #[test]
    fn test_deopt_reason_display() {
        assert_eq!(format!("{}", DeoptReason::TypeGuard), "type guard");
        assert_eq!(format!("{}", DeoptReason::Overflow), "overflow");
    }

    #[test]
    fn test_deopt_reason_should_patch_guard() {
        assert!(DeoptReason::PolymorphicSite.should_patch_guard());
        assert!(DeoptReason::UncommonTrap.should_patch_guard());
        assert!(!DeoptReason::TypeGuard.should_patch_guard());
    }

    #[test]
    fn test_delta_entry() {
        let entry = DeltaEntry::new(5, Value::from(42));
        assert_eq!(entry.slot, 5);
    }

    #[test]
    fn test_deopt_delta_record() {
        let mut delta = DeoptDelta::new();
        assert!(delta.is_empty());

        delta.record(0, Value::from(10));
        delta.record(5, Value::from(20));

        assert_eq!(delta.len(), 2);
        assert!(delta.get(0).is_some());
        assert!(delta.get(5).is_some());
        assert!(delta.get(3).is_none());
    }

    #[test]
    fn test_deopt_delta_update_existing() {
        let mut delta = DeoptDelta::new();

        delta.record(0, Value::from(10));
        delta.record(0, Value::from(20));

        assert_eq!(delta.len(), 1);
        // Value should be updated
    }

    #[test]
    fn test_deopt_state_creation() {
        let state = DeoptState::new(100, DeoptReason::TypeGuard, 1, 12345);

        assert_eq!(state.bc_offset, 100);
        assert_eq!(state.reason, DeoptReason::TypeGuard);
        assert_eq!(state.deopt_id, 1);
        assert_eq!(state.code_id, 12345);
        assert!(state.delta.is_empty());
    }

    #[test]
    fn test_deopt_state_record_modified() {
        let mut state = DeoptState::new(100, DeoptReason::TypeGuard, 1, 12345);

        state.record_modified(0, Value::from(42));
        state.record_modified(5, Value::from(100));

        assert_eq!(state.delta.len(), 2);
    }
}
