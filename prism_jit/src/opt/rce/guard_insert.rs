//! Guard Insertion for Range Check Elimination.
//!
//! This module provides utilities for inserting hoisted and widened guards
//! into loop preheaders. It handles the actual IR graph manipulation required
//! to move guards from loop bodies to preheader blocks.
//!
//! # Guard Types
//!
//! Two types of guard insertion are supported:
//!
//! - **Hoisting**: Move an existing guard to execute once before the loop
//! - **Widening**: Create new guards that check the entire IV range at once
//!
//! # Architecture
//!
//! ```text
//! Original:                       After Hoisting:
//! ┌──────────────────┐            ┌──────────────────┐
//! │ Preheader        │            │ Preheader        │
//! │ (no guards)      │            │ guard(0 < len)   │  ← Hoisted check
//! └────────┬─────────┘            └────────┬─────────┘
//!          │                               │
//!          ▼                               ▼
//! ┌──────────────────┐            ┌──────────────────┐
//! │ Loop Header      │            │ Loop Header      │
//! │ i = phi(0, i+1)  │            │ i = phi(0, i+1)  │
//! └────────┬─────────┘            └────────┬─────────┘
//!          │                               │
//!          ▼                               ▼
//! ┌──────────────────┐            ┌──────────────────┐
//! │ Loop Body        │            │ Loop Body        │
//! │ guard(i < len)   │ ← Per-iter │ (guard removed)  │
//! └──────────────────┘   check    └──────────────────┘
//! ```
//!
//! # Performance Considerations
//!
//! - Guards are inserted at dominating position in preheader
//! - Original guards are replaced with `ConstBool(true)` for DCE
//! - No heap allocations during guard insertion

use super::bounds::{BoundValue, RangeCheck, RangeCheckKind};
use super::elimination::{HoistInfo, WidenBound, WidenInfo};
use super::induction::{InductionAnalysis, InductionInit, InductionVariable};
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{CmpOp, GuardKind, Operator};
use crate::ir::types::ValueType;

// =============================================================================
// Guard Insertion Result
// =============================================================================

/// Result of guard insertion operations.
#[derive(Debug, Clone, Default)]
pub struct InsertionResult {
    /// Number of guards successfully inserted.
    pub inserted: u32,
    /// Number of guards that couldn't be inserted (missing preheader, etc.)
    pub failed: u32,
    /// IDs of newly created guard nodes.
    pub new_guards: Vec<NodeId>,
    /// IDs of newly created comparison nodes.
    pub new_comparisons: Vec<NodeId>,
}

impl InsertionResult {
    /// Create empty result.
    pub const fn new() -> Self {
        InsertionResult {
            inserted: 0,
            failed: 0,
            new_guards: Vec::new(),
            new_comparisons: Vec::new(),
        }
    }

    /// Check if all insertions succeeded.
    #[inline]
    pub fn all_succeeded(&self) -> bool {
        self.failed == 0
    }

    /// Total number of insertion attempts.
    #[inline]
    pub fn total_attempts(&self) -> u32 {
        self.inserted + self.failed
    }
}

// =============================================================================
// Guard Inserter
// =============================================================================

/// Inserts hoisted and widened guards into loop preheaders.
///
/// This is the core engine for transforming loop bounds checks. It works
/// by creating new guard nodes in preheader blocks and connecting them
/// to the existing control flow.
#[derive(Debug)]
pub struct GuardInserter<'a> {
    /// Mutable reference to the IR graph.
    graph: &'a mut Graph,
    /// Induction variable analysis.
    iv_analysis: &'a InductionAnalysis,
    /// Accumulated result.
    result: InsertionResult,
}

impl<'a> GuardInserter<'a> {
    /// Create a new guard inserter.
    pub fn new(graph: &'a mut Graph, iv_analysis: &'a InductionAnalysis) -> Self {
        GuardInserter {
            graph,
            iv_analysis,
            result: InsertionResult::new(),
        }
    }

    /// Insert a hoisted guard in the preheader.
    ///
    /// Hoisting moves a per-iteration guard to execute once before the loop,
    /// checking the initial value instead of the induction variable.
    pub fn insert_hoisted_guard(
        &mut self,
        check: &RangeCheck,
        info: &HoistInfo,
        loop_idx: usize,
        preheader_region: NodeId,
    ) -> Option<NodeId> {
        // Get the induction variable info
        let iv = self.iv_analysis.get_iv(loop_idx, info.iv)?;

        // Get the initial value node
        let init_value = self.get_init_value_node(iv)?;

        // Get the bound value node
        let bound_node = self.get_bound_node(&check.bound)?;

        // Create the comparison based on check kind
        let cmp_node = self.create_comparison(init_value, bound_node, check.kind)?;
        self.result.new_comparisons.push(cmp_node);

        // Create the guard node
        let guard_node = self.create_guard(cmp_node, preheader_region);
        self.result.new_guards.push(guard_node);
        self.result.inserted += 1;

        Some(guard_node)
    }

    /// Insert a widened guard in the preheader.
    ///
    /// Widening creates guards that check both the minimum and maximum values
    /// the induction variable will take, ensuring the entire range is safe.
    pub fn insert_widened_guard(
        &mut self,
        check: &RangeCheck,
        info: &WidenInfo,
        loop_idx: usize,
        preheader_region: NodeId,
    ) -> Option<(Option<NodeId>, Option<NodeId>)> {
        // Verify the IV exists
        let _iv = self.iv_analysis.get_iv(loop_idx, info.iv)?;

        let mut min_guard = None;
        let mut max_guard = None;

        // Insert minimum bound check if needed
        if let Some(min_node) = self.insert_bound_check(
            &info.min_check,
            CmpOp::Ge, // min >= 0
            preheader_region,
        ) {
            min_guard = Some(min_node);
            self.result.new_guards.push(min_node);
        }

        // Insert maximum bound check if needed
        if let Some(max_node) =
            self.insert_max_bound_check(&info.max_check, check, preheader_region)
        {
            max_guard = Some(max_node);
            self.result.new_guards.push(max_node);
        }

        if min_guard.is_some() || max_guard.is_some() {
            self.result.inserted += 1;
        } else {
            self.result.failed += 1;
        }

        Some((min_guard, max_guard))
    }

    /// Record a failed insertion.
    #[inline]
    pub fn record_failure(&mut self) {
        self.result.failed += 1;
    }

    /// Get the insertion result.
    #[inline]
    pub fn into_result(self) -> InsertionResult {
        self.result
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /// Get or create a node representing the initial value.
    fn get_init_value_node(&mut self, iv: &InductionVariable) -> Option<NodeId> {
        match &iv.init {
            InductionInit::Constant(val) => Some(self.graph.const_int(*val)),
            InductionInit::Node(node_id) => Some(*node_id),
        }
    }

    /// Get or create a node representing the bound value.
    fn get_bound_node(&mut self, bound: &BoundValue) -> Option<NodeId> {
        match bound {
            BoundValue::Constant(val) => Some(self.graph.const_int(*val)),
            BoundValue::Node(node_id) => Some(*node_id),
        }
    }

    /// Get or create a node for a widen bound value.
    fn get_widen_bound_node(&mut self, bound: &WidenBound) -> Option<NodeId> {
        match bound {
            WidenBound::Constant(val) => Some(self.graph.const_int(*val)),
            WidenBound::Node(node_id) => Some(*node_id),
            WidenBound::Computed { base, offset } => {
                if *offset == 0 {
                    Some(*base)
                } else {
                    // Create: base + offset
                    let offset_node = self.graph.const_int(*offset);
                    Some(self.graph.int_add(*base, offset_node))
                }
            }
        }
    }

    /// Create a comparison node for the guard.
    fn create_comparison(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        kind: RangeCheckKind,
    ) -> Option<NodeId> {
        let cmp_op = match kind {
            RangeCheckKind::LowerBound => CmpOp::Ge,          // val >= 0
            RangeCheckKind::UpperBound => CmpOp::Lt,          // val < len
            RangeCheckKind::UpperBoundInclusive => CmpOp::Le, // val <= len
        };

        let inputs = InputList::Pair(lhs, rhs);
        let node_id =
            self.graph
                .add_node_with_type(Operator::IntCmp(cmp_op), inputs, ValueType::Bool);

        Some(node_id)
    }

    /// Create a guard node.
    fn create_guard(&mut self, condition: NodeId, control: NodeId) -> NodeId {
        // Guard takes (control, condition) as inputs
        // control: where the guard is attached
        // condition: the boolean comparison result
        let inputs = InputList::Pair(control, condition);
        self.graph
            .add_node(Operator::Guard(GuardKind::Bounds), inputs)
    }

    /// Insert a generic bound check (e.g., init >= 0).
    fn insert_bound_check(
        &mut self,
        bound: &WidenBound,
        cmp_op: CmpOp,
        control: NodeId,
    ) -> Option<NodeId> {
        let bound_node = self.get_widen_bound_node(bound)?;
        let zero = self.graph.const_int(0);
        let cmp = self.graph.add_node_with_type(
            Operator::IntCmp(cmp_op),
            InputList::Pair(bound_node, zero),
            ValueType::Bool,
        );
        self.result.new_comparisons.push(cmp);
        Some(self.create_guard(cmp, control))
    }

    /// Insert a maximum bound check (e.g., max_iv < len).
    fn insert_max_bound_check(
        &mut self,
        max_check: &WidenBound,
        original_check: &RangeCheck,
        control: NodeId,
    ) -> Option<NodeId> {
        let max_node = self.get_widen_bound_node(max_check)?;
        let bound_node = self.get_bound_node(&original_check.bound)?;

        let cmp_op = match original_check.kind {
            RangeCheckKind::UpperBound => CmpOp::Lt,
            RangeCheckKind::UpperBoundInclusive => CmpOp::Le,
            _ => return None, // Lower bound checks don't need max check
        };

        let cmp = self.graph.add_node_with_type(
            Operator::IntCmp(cmp_op),
            InputList::Pair(max_node, bound_node),
            ValueType::Bool,
        );
        self.result.new_comparisons.push(cmp);
        Some(self.create_guard(cmp, control))
    }
}
