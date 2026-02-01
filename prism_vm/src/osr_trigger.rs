//! On-Stack Replacement (OSR) trigger and loop profiling.
//!
//! This module handles detection of hot loops and triggers OSR compilation.
//! It provides fine-grained loop analysis for optimal OSR entry point selection.

use crate::profiler::CodeId;
use smallvec::SmallVec;

// =============================================================================
// Constants
// =============================================================================

/// Threshold for loop iterations before triggering OSR.
/// Tuned for balance between warm-up time and optimization opportunity.
pub const OSR_LOOP_THRESHOLD: u64 = 5_000;

/// Threshold for very hot loops that warrant aggressive optimization.
pub const OSR_HOT_LOOP_THRESHOLD: u64 = 50_000;

/// Maximum OSR entries tracked per function to limit memory.
pub const MAX_OSR_ENTRIES_PER_FUNCTION: usize = 16;

// =============================================================================
// OSR Decision
// =============================================================================

/// Decision from the OSR system about loop compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OsrDecision {
    /// Loop is cold, no action needed.
    Cold = 0,
    /// Loop is warming up, continue monitoring.
    Warming = 1,
    /// Loop is hot, should trigger OSR compilation.
    Hot = 2,
    /// Loop is very hot, should trigger aggressive optimization.
    VeryHot = 3,
    /// OSR compilation already requested or in progress.
    Pending = 4,
    /// OSR code already available, ready to enter.
    Ready = 5,
}

impl OsrDecision {
    /// Check if OSR compilation should be triggered.
    #[inline(always)]
    pub const fn should_compile(self) -> bool {
        matches!(self, Self::Hot | Self::VeryHot)
    }

    /// Check if OSR entry is possible.
    #[inline(always)]
    pub const fn can_enter(self) -> bool {
        matches!(self, Self::Ready)
    }
}

// =============================================================================
// Loop Info
// =============================================================================

/// Information about a detected loop for OSR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopInfo {
    /// Loop header bytecode offset (target of back-edge).
    pub header_offset: u32,
    /// Back-edge source bytecode offset.
    pub back_edge_offset: u32,
    /// Estimated loop trip count from profiling.
    pub trip_count: u64,
    /// Current OSR decision for this loop.
    pub decision: OsrDecision,
}

impl LoopInfo {
    /// Create new loop info.
    #[inline]
    pub const fn new(header_offset: u32, back_edge_offset: u32) -> Self {
        Self {
            header_offset,
            back_edge_offset,
            trip_count: 0,
            decision: OsrDecision::Cold,
        }
    }

    /// Record a loop iteration and return updated decision.
    #[inline]
    pub fn record_iteration(&mut self) -> OsrDecision {
        self.trip_count = self.trip_count.saturating_add(1);

        // Update decision based on trip count
        self.decision = match self.trip_count {
            0..OSR_LOOP_THRESHOLD => OsrDecision::Warming,
            OSR_LOOP_THRESHOLD..OSR_HOT_LOOP_THRESHOLD => {
                if self.decision == OsrDecision::Cold || self.decision == OsrDecision::Warming {
                    OsrDecision::Hot
                } else {
                    self.decision
                }
            }
            _ => {
                if self.decision == OsrDecision::Hot {
                    OsrDecision::VeryHot
                } else if self.decision == OsrDecision::Warming {
                    OsrDecision::Hot
                } else {
                    self.decision
                }
            }
        };

        self.decision
    }

    /// Mark as pending compilation.
    #[inline]
    pub fn mark_pending(&mut self) {
        if self.decision == OsrDecision::Hot || self.decision == OsrDecision::VeryHot {
            self.decision = OsrDecision::Pending;
        }
    }

    /// Mark as having OSR code ready.
    #[inline]
    pub fn mark_ready(&mut self) {
        self.decision = OsrDecision::Ready;
    }
}

// =============================================================================
// OSR Trigger
// =============================================================================

/// Tracks hot loops and manages OSR triggering decisions.
///
/// Uses a two-level structure:
/// - Level 1: Per-function loop tracking (SmallVec for cache locality)
/// - Level 2: Global pending OSR queue for JIT compilation
#[derive(Debug, Default)]
pub struct OsrTrigger {
    /// Loops pending OSR compilation (code_id, loop_info).
    pending_compilation: SmallVec<[(CodeId, LoopInfo); 8]>,
    /// Loops with OSR code ready (code_id, header_offset).
    ready_entries: SmallVec<[(CodeId, u32); 8]>,
}

impl OsrTrigger {
    /// Create a new OSR trigger.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a loop back-edge and return the OSR decision.
    ///
    /// This is called from the interpreter on every backward branch.
    /// Must be extremely fast for the common case (cold/warming loops).
    #[inline]
    pub fn record_back_edge(
        &mut self,
        code_id: CodeId,
        header_offset: u32,
        _back_edge_offset: u32,
        loop_info: &mut LoopInfo,
    ) -> OsrDecision {
        // Fast path: already ready
        if loop_info.decision == OsrDecision::Ready {
            return OsrDecision::Ready;
        }

        // Fast path: pending compilation
        if loop_info.decision == OsrDecision::Pending {
            // Check if compilation finished
            if self.is_ready(code_id, header_offset) {
                loop_info.mark_ready();
                return OsrDecision::Ready;
            }
            return OsrDecision::Pending;
        }

        // Record iteration and get decision
        let decision = loop_info.record_iteration();

        // If newly hot, add to pending queue
        if decision == OsrDecision::Hot || decision == OsrDecision::VeryHot {
            if !self.is_pending(code_id, header_offset) {
                self.add_pending(code_id, *loop_info);
                loop_info.mark_pending();
            }
        }

        decision
    }

    /// Check if a loop has OSR code ready.
    #[inline]
    pub fn is_ready(&self, code_id: CodeId, header_offset: u32) -> bool {
        self.ready_entries
            .iter()
            .any(|(c, h)| *c == code_id && *h == header_offset)
    }

    /// Check if a loop is pending compilation.
    #[inline]
    fn is_pending(&self, code_id: CodeId, header_offset: u32) -> bool {
        self.pending_compilation
            .iter()
            .any(|(c, l)| *c == code_id && l.header_offset == header_offset)
    }

    /// Add a loop to the pending compilation queue.
    fn add_pending(&mut self, code_id: CodeId, loop_info: LoopInfo) {
        // Limit queue size to prevent memory bloat
        if self.pending_compilation.len() >= 64 {
            // Remove oldest entry (FIFO)
            self.pending_compilation.remove(0);
        }
        self.pending_compilation.push((code_id, loop_info));
    }

    /// Get the next loop pending OSR compilation.
    #[inline]
    pub fn pop_pending(&mut self) -> Option<(CodeId, LoopInfo)> {
        // Prioritize very hot loops
        if let Some(idx) = self.pending_compilation.iter().position(|(_, l)| {
            l.decision == OsrDecision::VeryHot || l.trip_count >= OSR_HOT_LOOP_THRESHOLD
        }) {
            return Some(self.pending_compilation.remove(idx));
        }

        // Otherwise take the oldest hot loop
        if !self.pending_compilation.is_empty() {
            return Some(self.pending_compilation.remove(0));
        }

        None
    }

    /// Mark a loop as having OSR code ready.
    pub fn mark_ready(&mut self, code_id: CodeId, header_offset: u32) {
        // Remove from pending
        self.pending_compilation
            .retain(|(c, l)| !(*c == code_id && l.header_offset == header_offset));

        // Add to ready entries (avoid duplicates)
        if !self.is_ready(code_id, header_offset) {
            if self.ready_entries.len() >= 64 {
                self.ready_entries.remove(0);
            }
            self.ready_entries.push((code_id, header_offset));
        }
    }

    /// Get count of pending compilations.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_compilation.len()
    }

    /// Get count of ready OSR entries.
    #[inline]
    pub fn ready_count(&self) -> usize {
        self.ready_entries.len()
    }

    /// Clear all OSR state.
    pub fn clear(&mut self) {
        self.pending_compilation.clear();
        self.ready_entries.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osr_decision_should_compile() {
        assert!(!OsrDecision::Cold.should_compile());
        assert!(!OsrDecision::Warming.should_compile());
        assert!(OsrDecision::Hot.should_compile());
        assert!(OsrDecision::VeryHot.should_compile());
        assert!(!OsrDecision::Pending.should_compile());
        assert!(!OsrDecision::Ready.should_compile());
    }

    #[test]
    fn test_osr_decision_can_enter() {
        assert!(!OsrDecision::Cold.can_enter());
        assert!(!OsrDecision::Hot.can_enter());
        assert!(OsrDecision::Ready.can_enter());
    }

    #[test]
    fn test_loop_info_cold_to_warming() {
        let mut info = LoopInfo::new(100, 120);
        assert_eq!(info.decision, OsrDecision::Cold);

        let decision = info.record_iteration();
        assert_eq!(decision, OsrDecision::Warming);
        assert_eq!(info.trip_count, 1);
    }

    #[test]
    fn test_loop_info_warming_to_hot() {
        let mut info = LoopInfo::new(100, 120);

        // Warm up to threshold
        for _ in 0..OSR_LOOP_THRESHOLD {
            info.record_iteration();
        }

        assert_eq!(info.decision, OsrDecision::Hot);
    }

    #[test]
    fn test_loop_info_hot_to_very_hot() {
        let mut info = LoopInfo::new(100, 120);

        // Warm up past very hot threshold
        for _ in 0..=OSR_HOT_LOOP_THRESHOLD {
            info.record_iteration();
        }

        assert_eq!(info.decision, OsrDecision::VeryHot);
    }

    #[test]
    fn test_osr_trigger_pending_queue() {
        let mut trigger = OsrTrigger::new();
        let code_id = CodeId::new(12345);
        let mut loop_info = LoopInfo::new(100, 120);

        // Warm up to hot
        for _ in 0..OSR_LOOP_THRESHOLD {
            let decision = trigger.record_back_edge(code_id, 100, 120, &mut loop_info);
            if decision == OsrDecision::Pending {
                break;
            }
        }

        // Should be pending
        assert!(trigger.pending_count() > 0 || loop_info.decision == OsrDecision::Pending);
    }

    #[test]
    fn test_osr_trigger_mark_ready() {
        let mut trigger = OsrTrigger::new();
        let code_id = CodeId::new(12345);

        // Add to pending
        let loop_info = LoopInfo::new(100, 120);
        trigger.add_pending(code_id, loop_info);
        assert_eq!(trigger.pending_count(), 1);

        // Mark ready
        trigger.mark_ready(code_id, 100);
        assert_eq!(trigger.pending_count(), 0);
        assert_eq!(trigger.ready_count(), 1);
        assert!(trigger.is_ready(code_id, 100));
    }

    #[test]
    fn test_osr_trigger_pop_pending_prioritizes_very_hot() {
        let mut trigger = OsrTrigger::new();
        let code1 = CodeId::new(1);
        let code2 = CodeId::new(2);

        // Add hot loop first
        let hot_loop = LoopInfo {
            header_offset: 100,
            back_edge_offset: 120,
            trip_count: OSR_LOOP_THRESHOLD,
            decision: OsrDecision::Pending,
        };
        trigger.add_pending(code1, hot_loop);

        // Add very hot loop second
        let very_hot_loop = LoopInfo {
            header_offset: 200,
            back_edge_offset: 220,
            trip_count: OSR_HOT_LOOP_THRESHOLD + 1,
            decision: OsrDecision::VeryHot,
        };
        trigger.add_pending(code2, very_hot_loop);

        // Very hot should be popped first
        let (popped_code, popped_loop) = trigger.pop_pending().unwrap();
        assert_eq!(popped_code, code2);
        assert_eq!(popped_loop.trip_count, OSR_HOT_LOOP_THRESHOLD + 1);
    }
}
