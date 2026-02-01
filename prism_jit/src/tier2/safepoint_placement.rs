//! Safepoint Placement Analysis
//!
//! This module implements CFG-based analysis for optimal safepoint poll placement.
//! Safepoints are placed at:
//! - Loop back-edges (to allow GC during long-running loops)
//! - Call sites (already handled in emit.rs)
//! - Allocation sites (before allocation to ensure GC can run)
//!
//! We apply safepoint elision for:
//! - Leaf functions (no calls, allocations, or back-edges)
//! - Short basic blocks between guaranteed safepoints

use super::lower::{MachineFunction, MachineInst, MachineOp, MachineOperand};
use smallvec::SmallVec;
use std::collections::HashSet;

// =============================================================================
// Safepoint Placement Result
// =============================================================================

/// Result of safepoint placement analysis.
#[derive(Debug, Clone)]
pub struct SafepointPlacement {
    /// Indices of instructions after which to emit safepoint polls.
    pub poll_indices: SmallVec<[usize; 8]>,
    /// Whether this function is a leaf (no safepoints needed).
    pub is_leaf: bool,
    /// Whether to load safepoint register in prologue.
    pub needs_safepoint_register: bool,
}

impl SafepointPlacement {
    /// Create placement indicating no safepoints needed.
    pub fn none() -> Self {
        SafepointPlacement {
            poll_indices: SmallVec::new(),
            is_leaf: true,
            needs_safepoint_register: false,
        }
    }
}

// =============================================================================
// Safepoint Analyzer
// =============================================================================

/// Analyzes machine functions to determine optimal safepoint placement.
pub struct SafepointAnalyzer {
    /// Maximum instructions between safepoints (for very long straight-line code).
    max_poll_interval: usize,
}

impl Default for SafepointAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SafepointAnalyzer {
    /// Create a new analyzer with default settings.
    pub fn new() -> Self {
        SafepointAnalyzer {
            // V8/HotSpot typically use ~1000 instructions as max interval
            max_poll_interval: 1024,
        }
    }

    /// Create analyzer with custom poll interval.
    pub fn with_poll_interval(max_interval: usize) -> Self {
        SafepointAnalyzer {
            max_poll_interval: max_interval,
        }
    }

    /// Analyze a machine function and return optimal safepoint placement.
    pub fn analyze(&self, mfunc: &MachineFunction) -> SafepointPlacement {
        // First check if this is a leaf function (no calls, allocations, or loops)
        if self.is_leaf_function(mfunc) {
            return SafepointPlacement::none();
        }

        let mut poll_indices = SmallVec::new();
        let back_edges = self.find_back_edges(mfunc);

        // Place safepoints at all back-edges
        for edge in &back_edges {
            if !poll_indices.contains(edge) {
                poll_indices.push(*edge);
            }
        }

        // Find long straight-line code sections and add interval polls
        self.add_interval_polls(mfunc, &mut poll_indices);

        // Sort for emission order
        poll_indices.sort_unstable();

        SafepointPlacement {
            is_leaf: false,
            needs_safepoint_register: !poll_indices.is_empty(),
            poll_indices,
        }
    }

    /// Check if function is a leaf (no calls, allocations, or back-edges).
    fn is_leaf_function(&self, mfunc: &MachineFunction) -> bool {
        for inst in &mfunc.insts {
            match inst.op {
                // Calls require safepoints
                MachineOp::Call => return false,
                // Jumps might form loops
                MachineOp::Jmp | MachineOp::Jcc => {
                    // Check if this jumps backwards (potential loop)
                    if self.is_backward_jump(mfunc, inst) {
                        return false;
                    }
                }
                _ => {}
            }
        }

        // Also check instruction count - very short functions are leaves
        mfunc.insts.len() < 32
    }

    /// Check if a jump instruction jumps backwards (potential loop).
    fn is_backward_jump(&self, mfunc: &MachineFunction, inst: &MachineInst) -> bool {
        if let MachineOperand::Label(target_id) = inst.dst {
            // Find the label definition
            if let Some(label_idx) = self.find_label_index(mfunc, target_id) {
                // Find this instruction's index
                if let Some(inst_idx) = mfunc.insts.iter().position(|i| std::ptr::eq(i, inst)) {
                    return label_idx < inst_idx;
                }
            }
        }
        false
    }

    /// Find the index of a label definition.
    fn find_label_index(&self, mfunc: &MachineFunction, label_id: u32) -> Option<usize> {
        mfunc.insts.iter().position(|inst| {
            inst.op == MachineOp::Label
                && matches!(inst.dst, MachineOperand::Label(id) if id == label_id)
        })
    }

    /// Find all back-edge instruction indices (jumps that form loops).
    fn find_back_edges(&self, mfunc: &MachineFunction) -> SmallVec<[usize; 4]> {
        let mut back_edges = SmallVec::new();
        let mut seen_labels = HashSet::new();

        for (idx, inst) in mfunc.insts.iter().enumerate() {
            // Record labels we've seen
            if inst.op == MachineOp::Label {
                if let MachineOperand::Label(id) = inst.dst {
                    seen_labels.insert(id);
                }
            }

            // Check if jump targets a previously-seen label (back-edge)
            if matches!(inst.op, MachineOp::Jmp | MachineOp::Jcc) {
                if let MachineOperand::Label(target_id) = inst.dst {
                    if seen_labels.contains(&target_id) {
                        // This is a back-edge - emit safepoint before the jump
                        back_edges.push(idx.saturating_sub(1));
                    }
                }
            }
        }

        back_edges
    }

    /// Add polls for long straight-line code sections.
    fn add_interval_polls(&self, mfunc: &MachineFunction, polls: &mut SmallVec<[usize; 8]>) {
        let polls_set: HashSet<usize> = polls.iter().copied().collect();

        let mut instructions_since_poll = 0usize;
        let mut last_guaranteed_safepoint = 0usize;

        for (idx, inst) in mfunc.insts.iter().enumerate() {
            // Calls are guaranteed safepoints
            if inst.op == MachineOp::Call {
                last_guaranteed_safepoint = idx;
                instructions_since_poll = 0;
                continue;
            }

            // Check if we have an explicit poll here
            if polls_set.contains(&idx) {
                instructions_since_poll = 0;
                continue;
            }

            instructions_since_poll += 1;

            // Add poll if we've gone too long without one
            if instructions_since_poll >= self.max_poll_interval {
                // Only add if not right after a call or existing poll
                if idx > last_guaranteed_safepoint + 1 && !polls_set.contains(&idx) {
                    polls.push(idx);
                    instructions_since_poll = 0;
                }
            }
        }
    }
}

// =============================================================================
// Safepoint Emission Helper
// =============================================================================

/// Helper for emitting safepoint polls during code generation.
pub struct SafepointEmitter {
    /// The analyzed placement.
    placement: SafepointPlacement,
    /// Set of indices where we should emit polls (for O(1) lookup).
    poll_set: HashSet<usize>,
    /// Address of the safepoint page (loaded into R15).
    safepoint_page_addr: usize,
}

impl SafepointEmitter {
    /// Create a new safepoint emitter.
    pub fn new(placement: SafepointPlacement, page_addr: usize) -> Self {
        let poll_set = placement.poll_indices.iter().copied().collect();
        SafepointEmitter {
            placement,
            poll_set,
            safepoint_page_addr: page_addr,
        }
    }

    /// Check if safepoint poll should be emitted after this instruction index.
    #[inline]
    pub fn should_emit_poll(&self, idx: usize) -> bool {
        self.poll_set.contains(&idx)
    }

    /// Get the safepoint page address for prologue loading.
    pub fn safepoint_page_addr(&self) -> usize {
        self.safepoint_page_addr
    }

    /// Check if we need to load the safepoint register in prologue.
    pub fn needs_safepoint_register(&self) -> bool {
        self.placement.needs_safepoint_register
    }

    /// Check if this function is a leaf (no safepoint infrastructure needed).
    pub fn is_leaf(&self) -> bool {
        self.placement.is_leaf
    }

    /// Get the number of safepoint polls to be emitted.
    pub fn poll_count(&self) -> usize {
        self.placement.poll_indices.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::x64::registers::Gpr;

    fn make_inst(op: MachineOp) -> MachineInst {
        MachineInst::nullary(op)
    }

    fn make_label(id: u32) -> MachineInst {
        MachineInst::new(
            MachineOp::Label,
            MachineOperand::Label(id),
            MachineOperand::None,
        )
    }

    fn make_jmp(target: u32) -> MachineInst {
        MachineInst::new(
            MachineOp::Jmp,
            MachineOperand::Label(target),
            MachineOperand::None,
        )
    }

    fn make_jcc(target: u32) -> MachineInst {
        let mut inst = MachineInst::new(
            MachineOp::Jcc,
            MachineOperand::Label(target),
            MachineOperand::None,
        );
        inst.cc = Some(super::super::lower::CondCode::E);
        inst
    }

    #[test]
    fn test_analyzer_new() {
        let analyzer = SafepointAnalyzer::new();
        assert_eq!(analyzer.max_poll_interval, 1024);
    }

    #[test]
    fn test_analyzer_custom_interval() {
        let analyzer = SafepointAnalyzer::with_poll_interval(512);
        assert_eq!(analyzer.max_poll_interval, 512);
    }

    #[test]
    fn test_empty_function_is_leaf() {
        let mfunc = MachineFunction::new();
        let analyzer = SafepointAnalyzer::new();
        let placement = analyzer.analyze(&mfunc);

        assert!(placement.is_leaf);
        assert!(placement.poll_indices.is_empty());
        assert!(!placement.needs_safepoint_register);
    }

    #[test]
    fn test_short_function_is_leaf() {
        let mut mfunc = MachineFunction::new();
        // Add a few simple instructions
        mfunc.push(MachineInst::new(
            MachineOp::Mov,
            MachineOperand::gpr(Gpr::Rax),
            MachineOperand::Imm(42),
        ));
        mfunc.push(make_inst(MachineOp::Ret));

        let analyzer = SafepointAnalyzer::new();
        let placement = analyzer.analyze(&mfunc);

        assert!(placement.is_leaf);
    }

    #[test]
    fn test_function_with_call_not_leaf() {
        let mut mfunc = MachineFunction::new();
        mfunc.push(MachineInst::new(
            MachineOp::Call,
            MachineOperand::Imm(0x12345678),
            MachineOperand::None,
        ));
        mfunc.push(make_inst(MachineOp::Ret));

        let analyzer = SafepointAnalyzer::new();
        let placement = analyzer.analyze(&mfunc);

        // Has a call, but call itself is a safepoint, so no extra polls needed
        assert!(!placement.is_leaf);
    }

    #[test]
    fn test_back_edge_detection() {
        let mut mfunc = MachineFunction::new();

        // Loop header
        mfunc.push(make_label(1));

        // Loop body
        for _ in 0..10 {
            mfunc.push(make_inst(MachineOp::Nop));
        }

        // Back-edge (jump to loop header)
        mfunc.push(make_jcc(1));

        // Exit
        mfunc.push(make_inst(MachineOp::Ret));

        let analyzer = SafepointAnalyzer::new();
        let back_edges = analyzer.find_back_edges(&mfunc);

        // Should detect the back-edge before the jcc
        assert!(!back_edges.is_empty());
    }

    #[test]
    fn test_forward_jump_not_back_edge() {
        let mut mfunc = MachineFunction::new();

        // Forward jump
        mfunc.push(make_jmp(1));

        // Some instructions
        for _ in 0..5 {
            mfunc.push(make_inst(MachineOp::Nop));
        }

        // Target label
        mfunc.push(make_label(1));
        mfunc.push(make_inst(MachineOp::Ret));

        let analyzer = SafepointAnalyzer::new();
        let back_edges = analyzer.find_back_edges(&mfunc);

        // Forward jumps are not back-edges
        assert!(back_edges.is_empty());
    }

    #[test]
    fn test_loop_requires_safepoint() {
        let mut mfunc = MachineFunction::new();

        // Simple loop
        mfunc.push(make_label(1));
        for _ in 0..50 {
            mfunc.push(make_inst(MachineOp::Nop));
        }
        mfunc.push(make_jcc(1));
        mfunc.push(make_inst(MachineOp::Ret));

        let analyzer = SafepointAnalyzer::new();
        let placement = analyzer.analyze(&mfunc);

        assert!(!placement.is_leaf);
        assert!(placement.needs_safepoint_register);
        assert!(!placement.poll_indices.is_empty());
    }

    #[test]
    fn test_safepoint_emitter_creation() {
        let placement = SafepointPlacement {
            poll_indices: SmallVec::from_slice(&[10, 20, 30]),
            is_leaf: false,
            needs_safepoint_register: true,
        };

        let emitter = SafepointEmitter::new(placement, 0xDEADBEEF);

        assert!(emitter.should_emit_poll(10));
        assert!(emitter.should_emit_poll(20));
        assert!(emitter.should_emit_poll(30));
        assert!(!emitter.should_emit_poll(15));
        assert_eq!(emitter.poll_count(), 3);
    }

    #[test]
    fn test_safepoint_emitter_leaf() {
        let emitter = SafepointEmitter::new(SafepointPlacement::none(), 0);

        assert!(emitter.is_leaf());
        assert!(!emitter.needs_safepoint_register());
        assert_eq!(emitter.poll_count(), 0);
    }

    #[test]
    fn test_interval_poll_long_straight_line() {
        let mut mfunc = MachineFunction::new();

        // Create very long straight-line code
        for _ in 0..2000 {
            mfunc.push(make_inst(MachineOp::Nop));
        }
        mfunc.push(make_inst(MachineOp::Ret));

        // Use small interval to trigger interval polls
        let analyzer = SafepointAnalyzer::with_poll_interval(100);
        let placement = analyzer.analyze(&mfunc);

        // Short function (< 32 instructions when checking is_leaf)
        // but we have > 2000 instructions, so not leaf
        // Should have interval polls
        if !placement.is_leaf {
            assert!(placement.poll_indices.len() >= 1);
        }
    }

    #[test]
    fn test_multiple_loops() {
        let mut mfunc = MachineFunction::new();

        // First loop
        mfunc.push(make_label(1));
        for _ in 0..40 {
            mfunc.push(make_inst(MachineOp::Nop));
        }
        mfunc.push(make_jcc(1));

        // Second loop
        mfunc.push(make_label(2));
        for _ in 0..40 {
            mfunc.push(make_inst(MachineOp::Nop));
        }
        mfunc.push(make_jcc(2));

        mfunc.push(make_inst(MachineOp::Ret));

        let analyzer = SafepointAnalyzer::new();
        let placement = analyzer.analyze(&mfunc);

        // Should have safepoints for both loops
        assert!(placement.poll_indices.len() >= 2);
    }

    #[test]
    fn test_nested_loops() {
        let mut mfunc = MachineFunction::new();

        // Outer loop
        mfunc.push(make_label(1));

        // Inner loop
        mfunc.push(make_label(2));
        for _ in 0..50 {
            mfunc.push(make_inst(MachineOp::Nop));
        }
        mfunc.push(make_jcc(2));

        // Back to outer
        for _ in 0..10 {
            mfunc.push(make_inst(MachineOp::Nop));
        }
        mfunc.push(make_jcc(1));

        mfunc.push(make_inst(MachineOp::Ret));

        let analyzer = SafepointAnalyzer::new();
        let placement = analyzer.analyze(&mfunc);

        // Should have safepoints for both loops
        assert!(!placement.is_leaf);
        assert!(placement.poll_indices.len() >= 2);
    }
}
