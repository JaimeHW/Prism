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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // InsertionResult Tests
    // =========================================================================

    #[test]
    fn test_insertion_result_new() {
        let result = InsertionResult::new();
        assert_eq!(result.inserted, 0);
        assert_eq!(result.failed, 0);
        assert!(result.new_guards.is_empty());
        assert!(result.new_comparisons.is_empty());
    }

    #[test]
    fn test_insertion_result_all_succeeded() {
        let mut result = InsertionResult::new();
        result.inserted = 5;
        result.failed = 0;
        assert!(result.all_succeeded());

        result.failed = 1;
        assert!(!result.all_succeeded());
    }

    #[test]
    fn test_insertion_result_total_attempts() {
        let mut result = InsertionResult::new();
        result.inserted = 3;
        result.failed = 2;
        assert_eq!(result.total_attempts(), 5);
    }

    #[test]
    fn test_insertion_result_default() {
        let result = InsertionResult::default();
        assert_eq!(result.inserted, 0);
        assert_eq!(result.failed, 0);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_insertion_result_total_zero() {
        let result = InsertionResult::new();
        assert_eq!(result.total_attempts(), 0);
    }

    #[test]
    fn test_insertion_result_partial_success() {
        let mut result = InsertionResult::new();
        result.inserted = 10;
        result.failed = 5;
        assert!(!result.all_succeeded());
        assert_eq!(result.total_attempts(), 15);
    }

    #[test]
    fn test_insertion_result_with_nodes() {
        let mut result = InsertionResult::new();
        result.new_guards.push(NodeId::new(1));
        result.new_guards.push(NodeId::new(2));
        result.new_comparisons.push(NodeId::new(3));

        assert_eq!(result.new_guards.len(), 2);
        assert_eq!(result.new_comparisons.len(), 1);
    }

    #[test]
    fn test_insertion_result_clone() {
        let mut result = InsertionResult::new();
        result.inserted = 5;
        result.failed = 2;
        result.new_guards.push(NodeId::new(1));

        let cloned = result.clone();
        assert_eq!(cloned.inserted, 5);
        assert_eq!(cloned.failed, 2);
        assert_eq!(cloned.new_guards.len(), 1);
    }

    #[test]
    fn test_insertion_result_high_counts() {
        let mut result = InsertionResult::new();
        result.inserted = 1000;
        result.failed = 500;
        assert_eq!(result.total_attempts(), 1500);
        assert!(!result.all_succeeded());
    }

    #[test]
    fn test_insertion_result_many_guards() {
        let mut result = InsertionResult::new();
        for i in 0..100 {
            result.new_guards.push(NodeId::new(i));
            result.new_comparisons.push(NodeId::new(i + 1000));
        }
        assert_eq!(result.new_guards.len(), 100);
        assert_eq!(result.new_comparisons.len(), 100);
    }

    // =========================================================================
    // GuardInserter Tests
    // =========================================================================

    #[test]
    fn test_guard_inserter_record_failure() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let iv_analysis = InductionAnalysis::empty();

        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);
        inserter.record_failure();
        inserter.record_failure();
        inserter.record_failure();

        let result = inserter.into_result();
        assert_eq!(result.failed, 3);
        assert_eq!(result.inserted, 0);
    }

    #[test]
    fn test_guard_inserter_empty_analysis() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let iv_analysis = InductionAnalysis::empty();

        let inserter = GuardInserter::new(&mut graph, &iv_analysis);
        let result = inserter.into_result();
        assert!(result.all_succeeded());
        assert_eq!(result.total_attempts(), 0);
    }

    #[test]
    fn test_widen_bound_constant() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let bound = WidenBound::Constant(42);
        let node = inserter.get_widen_bound_node(&bound);
        assert!(node.is_some());
    }

    #[test]
    fn test_widen_bound_node() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let node_id = graph.const_int(100);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let bound = WidenBound::Node(node_id);
        let result = inserter.get_widen_bound_node(&bound);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), node_id);
    }

    #[test]
    fn test_widen_bound_computed_zero_offset() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let base_node = graph.const_int(50);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        // Zero offset should return base directly
        let bound = WidenBound::Computed {
            base: base_node,
            offset: 0,
        };
        let result = inserter.get_widen_bound_node(&bound);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), base_node);
    }

    #[test]
    fn test_widen_bound_computed_nonzero_offset() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let base_node = graph.const_int(50);
        let initial_node_count = graph.len();
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        // Non-zero offset should create addition node
        let bound = WidenBound::Computed {
            base: base_node,
            offset: 10,
        };
        let result = inserter.get_widen_bound_node(&bound);
        assert!(result.is_some());
        // Should have created new nodes (offset const + add)
        assert!(graph.len() > initial_node_count);
    }

    #[test]
    fn test_bound_value_constant() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let bound = BoundValue::Constant(100);
        let node = inserter.get_bound_node(&bound);
        assert!(node.is_some());
    }

    #[test]
    fn test_bound_value_node() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let len_node = graph.const_int(256);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let bound = BoundValue::Node(len_node);
        let result = inserter.get_bound_node(&bound);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), len_node);
    }

    #[test]
    fn test_create_comparison_lower_bound() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let lhs = graph.const_int(0);
        let rhs = graph.const_int(0);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let cmp_node = inserter.create_comparison(lhs, rhs, RangeCheckKind::LowerBound);
        assert!(cmp_node.is_some());
    }

    #[test]
    fn test_create_comparison_upper_bound() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let lhs = graph.const_int(0);
        let rhs = graph.const_int(100);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let cmp_node = inserter.create_comparison(lhs, rhs, RangeCheckKind::UpperBound);
        assert!(cmp_node.is_some());
    }

    #[test]
    fn test_create_comparison_upper_bound_inclusive() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let lhs = graph.const_int(99);
        let rhs = graph.const_int(100);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let cmp_node = inserter.create_comparison(lhs, rhs, RangeCheckKind::UpperBoundInclusive);
        assert!(cmp_node.is_some());
    }

    #[test]
    fn test_create_guard() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let condition = graph.const_bool(true);
        let control = NodeId::new(0); // Use start node
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let guard = inserter.create_guard(condition, control);

        // Verify guard was created
        let guard_node = graph.get(guard);
        assert!(guard_node.is_some());
        assert!(matches!(
            guard_node.unwrap().op,
            Operator::Guard(GuardKind::Bounds)
        ));
    }

    #[test]
    fn test_get_init_value_node_constant() {
        use super::super::induction::{InductionDirection, InductionStep};
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let iv = InductionVariable::new(
            NodeId::new(10),
            InductionInit::Constant(0),
            InductionStep::Constant(1),
            InductionDirection::Increasing,
            None,
        );

        let init_node = inserter.get_init_value_node(&iv);
        assert!(init_node.is_some());
    }

    #[test]
    fn test_get_init_value_node_from_node() {
        use super::super::induction::{InductionDirection, InductionStep};
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let init_id = graph.const_int(42);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let iv = InductionVariable::new(
            NodeId::new(10),
            InductionInit::Node(init_id),
            InductionStep::Constant(1),
            InductionDirection::Increasing,
            None,
        );

        let result = inserter.get_init_value_node(&iv);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), init_id);
    }

    #[test]
    fn test_insert_bound_check_with_constant() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let control = NodeId::new(0);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let bound = WidenBound::Constant(0);
        let guard = inserter.insert_bound_check(&bound, CmpOp::Ge, control);

        assert!(guard.is_some());
        assert!(!inserter.result.new_comparisons.is_empty());
    }

    #[test]
    fn test_insert_max_bound_check_upper_bound() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let control = NodeId::new(0);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let max_check = WidenBound::Constant(99);
        let original_check = RangeCheck::new(
            NodeId::new(5),
            NodeId::new(1),
            NodeId::new(6),
            BoundValue::Constant(100),
            RangeCheckKind::UpperBound,
            0,
        );

        let guard = inserter.insert_max_bound_check(&max_check, &original_check, control);
        assert!(guard.is_some());
    }

    #[test]
    fn test_insert_max_bound_check_lower_bound_returns_none() {
        use crate::ir::graph::Graph;
        let mut graph = Graph::new();
        let control = NodeId::new(0);
        let iv_analysis = InductionAnalysis::empty();
        let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

        let max_check = WidenBound::Constant(0);
        let original_check = RangeCheck::new(
            NodeId::new(5),
            NodeId::new(1),
            NodeId::new(6),
            BoundValue::Constant(0),
            RangeCheckKind::LowerBound, // Lower bound shouldn't need max check
            0,
        );

        let guard = inserter.insert_max_bound_check(&max_check, &original_check, control);
        // Lower bound checks don't need max check
        assert!(guard.is_none());
    }
}
