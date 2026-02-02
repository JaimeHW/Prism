//! Range Check Analysis.
//!
//! This module identifies bounds checks in loop bodies that can potentially
//! be eliminated or hoisted. It analyzes Guard nodes with Bounds kind to
//! find checks on induction variables.
//!
//! # Check Types
//!
//! - **Lower Bound**: `index >= 0`
//! - **Upper Bound**: `index < length`
//! - **Combined**: Both checks (common for array access)
//!
//! # Analysis
//!
//! For each bounds guard, we determine:
//! 1. Which induction variable is being checked
//! 2. What the comparison is (lower/upper bound)
//! 3. What the length/bound value is

use super::induction::{InductionAnalysis, InductionVariable};
use crate::ir::cfg::Loop;
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{CmpOp, GuardKind, Operator};

// =============================================================================
// Range Check Types
// =============================================================================

/// Information about a bounds check that may be eliminable.
#[derive(Debug, Clone)]
pub struct RangeCheck {
    /// The guard node performing the check.
    pub guard: NodeId,

    /// The induction variable being checked.
    pub induction_var: NodeId,

    /// The comparison condition node.
    pub condition: NodeId,

    /// The array/collection length being checked against.
    pub bound: BoundValue,

    /// The kind of check being performed.
    pub kind: RangeCheckKind,

    /// The loop index this check belongs to.
    pub loop_idx: usize,
}

/// The bound value for a range check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundValue {
    /// Constant bound (e.g., array of known size).
    Constant(i64),

    /// Bound from a node (e.g., len(arr)).
    Node(NodeId),
}

/// Kind of range check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RangeCheckKind {
    /// Lower bound check: `index >= 0`.
    LowerBound,

    /// Upper bound check: `index < length`.
    UpperBound,

    /// Upper bound check (less than or equal): `index <= length - 1`.
    UpperBoundInclusive,
}

impl RangeCheck {
    /// Create a new range check.
    #[inline]
    pub fn new(
        guard: NodeId,
        induction_var: NodeId,
        condition: NodeId,
        bound: BoundValue,
        kind: RangeCheckKind,
        loop_idx: usize,
    ) -> Self {
        Self {
            guard,
            induction_var,
            condition,
            bound,
            kind,
            loop_idx,
        }
    }

    /// Check if this is a lower bound check.
    #[inline]
    pub fn is_lower_bound(&self) -> bool {
        matches!(self.kind, RangeCheckKind::LowerBound)
    }

    /// Check if this is an upper bound check.
    #[inline]
    pub fn is_upper_bound(&self) -> bool {
        matches!(
            self.kind,
            RangeCheckKind::UpperBound | RangeCheckKind::UpperBoundInclusive
        )
    }

    /// Check if bound is constant.
    #[inline]
    pub fn has_constant_bound(&self) -> bool {
        matches!(self.bound, BoundValue::Constant(_))
    }

    /// Get constant bound if available.
    #[inline]
    pub fn constant_bound(&self) -> Option<i64> {
        match self.bound {
            BoundValue::Constant(v) => Some(v),
            BoundValue::Node(_) => None,
        }
    }
}

impl BoundValue {
    /// Check if this is a constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, BoundValue::Constant(_))
    }

    /// Get constant value if available.
    #[inline]
    pub fn as_constant(&self) -> Option<i64> {
        match self {
            BoundValue::Constant(v) => Some(*v),
            BoundValue::Node(_) => None,
        }
    }

    /// Get node if available.
    #[inline]
    pub fn as_node(&self) -> Option<NodeId> {
        match self {
            BoundValue::Node(n) => Some(*n),
            BoundValue::Constant(_) => None,
        }
    }
}

// =============================================================================
// Range Check Detector
// =============================================================================

/// Detects range checks in loop bodies.
#[derive(Debug)]
pub struct RangeCheckDetector<'g> {
    graph: &'g Graph,
}

impl<'g> RangeCheckDetector<'g> {
    /// Create a new range check detector.
    #[inline]
    pub fn new(graph: &'g Graph) -> Self {
        Self { graph }
    }

    /// Find all range checks in a loop that involve induction variables.
    pub fn find_range_checks(
        &self,
        loop_info: &Loop,
        loop_idx: usize,
        iv_analysis: &InductionAnalysis,
    ) -> Vec<RangeCheck> {
        let mut checks = Vec::new();

        // Get IVs for this loop
        let Some(ivs) = iv_analysis.get(loop_idx) else {
            return checks;
        };

        // Search for Guard(Bounds) nodes
        for (node_id, node) in self.graph.iter() {
            // Only process bounds guards
            let Operator::Guard(GuardKind::Bounds) = &node.op else {
                continue;
            };

            // Check if this guard is in the loop (simplified check)
            if !self.is_potentially_in_loop(loop_info, node_id) {
                continue;
            }

            // Analyze the guard
            if let Some(check) = self.analyze_bounds_guard(node_id, loop_idx, ivs) {
                checks.push(check);
            }
        }

        checks
    }

    /// Check if a node might be in a loop (conservative).
    fn is_potentially_in_loop(&self, loop_info: &Loop, _node: NodeId) -> bool {
        // For now, if the loop has a body, assume nodes might be in it.
        // Full implementation would check block membership.
        !loop_info.body.is_empty()
    }

    /// Analyze a bounds guard to extract range check information.
    fn analyze_bounds_guard(
        &self,
        guard: NodeId,
        loop_idx: usize,
        ivs: &std::collections::HashMap<NodeId, InductionVariable>,
    ) -> Option<RangeCheck> {
        let guard_node = self.graph.get(guard)?;

        // Guard typically has: control input, condition
        // Condition should be a comparison
        let condition = guard_node.inputs.get(1)?;
        let cond_node = self.graph.get(condition)?;

        match &cond_node.op {
            // index < bound (upper bound)
            Operator::IntCmp(CmpOp::Lt) => {
                self.analyze_lt_pattern(guard, condition, &cond_node.inputs, loop_idx, ivs)
            }

            // index <= bound (upper bound inclusive)
            Operator::IntCmp(CmpOp::Le) => {
                self.analyze_le_pattern(guard, condition, &cond_node.inputs, loop_idx, ivs)
            }

            // index >= 0 (lower bound)
            Operator::IntCmp(CmpOp::Ge) => {
                self.analyze_ge_pattern(guard, condition, &cond_node.inputs, loop_idx, ivs)
            }

            // index > -1 (alternative lower bound)
            Operator::IntCmp(CmpOp::Gt) => {
                self.analyze_gt_pattern(guard, condition, &cond_node.inputs, loop_idx, ivs)
            }

            _ => None,
        }
    }

    /// Analyze `lhs < rhs` pattern.
    fn analyze_lt_pattern(
        &self,
        guard: NodeId,
        condition: NodeId,
        inputs: &InputList,
        loop_idx: usize,
        ivs: &std::collections::HashMap<NodeId, InductionVariable>,
    ) -> Option<RangeCheck> {
        if inputs.len() != 2 {
            return None;
        }

        let lhs = inputs.get(0)?;
        let rhs = inputs.get(1)?;

        // Pattern: iv < bound
        if ivs.contains_key(&lhs) {
            let bound = self.node_to_bound(rhs);
            return Some(RangeCheck::new(
                guard,
                lhs,
                condition,
                bound,
                RangeCheckKind::UpperBound,
                loop_idx,
            ));
        }

        // Pattern: bound > iv (equivalent to iv < bound)
        // This would be caught by LT with operands swapped in normalization
        None
    }

    /// Analyze `lhs <= rhs` pattern.
    fn analyze_le_pattern(
        &self,
        guard: NodeId,
        condition: NodeId,
        inputs: &InputList,
        loop_idx: usize,
        ivs: &std::collections::HashMap<NodeId, InductionVariable>,
    ) -> Option<RangeCheck> {
        if inputs.len() != 2 {
            return None;
        }

        let lhs = inputs.get(0)?;
        let rhs = inputs.get(1)?;

        // Pattern: iv <= bound
        if ivs.contains_key(&lhs) {
            let bound = self.node_to_bound(rhs);
            return Some(RangeCheck::new(
                guard,
                lhs,
                condition,
                bound,
                RangeCheckKind::UpperBoundInclusive,
                loop_idx,
            ));
        }

        None
    }

    /// Analyze `lhs >= rhs` pattern.
    fn analyze_ge_pattern(
        &self,
        guard: NodeId,
        condition: NodeId,
        inputs: &InputList,
        loop_idx: usize,
        ivs: &std::collections::HashMap<NodeId, InductionVariable>,
    ) -> Option<RangeCheck> {
        if inputs.len() != 2 {
            return None;
        }

        let lhs = inputs.get(0)?;
        let rhs = inputs.get(1)?;

        // Pattern: iv >= 0
        if ivs.contains_key(&lhs) {
            if let Some(0) = self.get_constant_value(rhs) {
                return Some(RangeCheck::new(
                    guard,
                    lhs,
                    condition,
                    BoundValue::Constant(0),
                    RangeCheckKind::LowerBound,
                    loop_idx,
                ));
            }
        }

        // Pattern: bound >= iv could represent upper bound
        // but it's less common, skip for now

        None
    }

    /// Analyze `lhs > rhs` pattern.
    fn analyze_gt_pattern(
        &self,
        guard: NodeId,
        condition: NodeId,
        inputs: &InputList,
        loop_idx: usize,
        ivs: &std::collections::HashMap<NodeId, InductionVariable>,
    ) -> Option<RangeCheck> {
        if inputs.len() != 2 {
            return None;
        }

        let lhs = inputs.get(0)?;
        let rhs = inputs.get(1)?;

        // Pattern: iv > -1 (equivalent to iv >= 0)
        if ivs.contains_key(&lhs) {
            if let Some(-1) = self.get_constant_value(rhs) {
                return Some(RangeCheck::new(
                    guard,
                    lhs,
                    condition,
                    BoundValue::Constant(0),
                    RangeCheckKind::LowerBound,
                    loop_idx,
                ));
            }
        }

        None
    }

    /// Convert a node to a bound value.
    fn node_to_bound(&self, node: NodeId) -> BoundValue {
        if let Some(val) = self.get_constant_value(node) {
            BoundValue::Constant(val)
        } else {
            BoundValue::Node(node)
        }
    }

    /// Get constant value from a node.
    #[inline]
    fn get_constant_value(&self, node: NodeId) -> Option<i64> {
        let n = self.graph.get(node)?;
        match &n.op {
            Operator::ConstInt(v) => Some(*v),
            _ => None,
        }
    }
}

// =============================================================================
// Range Check Collection
// =============================================================================

/// Collection of range checks grouped by loop and induction variable.
#[derive(Debug, Default)]
pub struct RangeCheckCollection {
    /// All collected range checks.
    checks: Vec<RangeCheck>,

    /// Checks indexed by loop.
    by_loop: Vec<Vec<usize>>,

    /// Checks indexed by induction variable.
    by_iv: std::collections::HashMap<NodeId, Vec<usize>>,
}

impl RangeCheckCollection {
    /// Create a new empty collection.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with expected capacity.
    #[inline]
    pub fn with_capacity(num_checks: usize, num_loops: usize) -> Self {
        Self {
            checks: Vec::with_capacity(num_checks),
            by_loop: vec![Vec::new(); num_loops],
            by_iv: std::collections::HashMap::new(),
        }
    }

    /// Add a range check.
    pub fn add(&mut self, check: RangeCheck) {
        let idx = self.checks.len();
        let loop_idx = check.loop_idx;
        let iv = check.induction_var;

        // Ensure by_loop has enough capacity
        while self.by_loop.len() <= loop_idx {
            self.by_loop.push(Vec::new());
        }

        self.by_loop[loop_idx].push(idx);
        self.by_iv.entry(iv).or_default().push(idx);
        self.checks.push(check);
    }

    /// Add multiple checks.
    pub fn add_all(&mut self, checks: impl IntoIterator<Item = RangeCheck>) {
        for check in checks {
            self.add(check);
        }
    }

    /// Get all checks.
    #[inline]
    pub fn all(&self) -> &[RangeCheck] {
        &self.checks
    }

    /// Get check by index.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&RangeCheck> {
        self.checks.get(idx)
    }

    /// Get checks for a loop.
    pub fn for_loop(&self, loop_idx: usize) -> impl Iterator<Item = &RangeCheck> {
        self.by_loop
            .get(loop_idx)
            .into_iter()
            .flatten()
            .filter_map(|&idx| self.checks.get(idx))
    }

    /// Get checks for an induction variable.
    pub fn for_iv(&self, iv: NodeId) -> impl Iterator<Item = &RangeCheck> {
        self.by_iv
            .get(&iv)
            .into_iter()
            .flatten()
            .filter_map(|&idx| self.checks.get(idx))
    }

    /// Total number of checks.
    #[inline]
    pub fn len(&self) -> usize {
        self.checks.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }

    /// Count lower bound checks.
    pub fn count_lower_bounds(&self) -> usize {
        self.checks.iter().filter(|c| c.is_lower_bound()).count()
    }

    /// Count upper bound checks.
    pub fn count_upper_bounds(&self) -> usize {
        self.checks.iter().filter(|c| c.is_upper_bound()).count()
    }

    /// Count checks with constant bounds.
    pub fn count_constant_bounds(&self) -> usize {
        self.checks
            .iter()
            .filter(|c| c.has_constant_bound())
            .count()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BoundValue Tests
    // =========================================================================

    #[test]
    fn test_bound_constant() {
        let bound = BoundValue::Constant(100);
        assert!(bound.is_constant());
        assert_eq!(bound.as_constant(), Some(100));
        assert_eq!(bound.as_node(), None);
    }

    #[test]
    fn test_bound_node() {
        let bound = BoundValue::Node(NodeId::new(5));
        assert!(!bound.is_constant());
        assert_eq!(bound.as_constant(), None);
        assert_eq!(bound.as_node(), Some(NodeId::new(5)));
    }

    #[test]
    fn test_bound_equality() {
        assert_eq!(BoundValue::Constant(10), BoundValue::Constant(10));
        assert_ne!(BoundValue::Constant(10), BoundValue::Constant(20));
        assert_ne!(BoundValue::Constant(10), BoundValue::Node(NodeId::new(10)));
    }

    // =========================================================================
    // RangeCheckKind Tests
    // =========================================================================

    #[test]
    fn test_check_kind_variants() {
        assert_ne!(RangeCheckKind::LowerBound, RangeCheckKind::UpperBound);
        assert_ne!(
            RangeCheckKind::LowerBound,
            RangeCheckKind::UpperBoundInclusive
        );
        assert_ne!(
            RangeCheckKind::UpperBound,
            RangeCheckKind::UpperBoundInclusive
        );
    }

    // =========================================================================
    // RangeCheck Tests
    // =========================================================================

    fn make_lower_check() -> RangeCheck {
        RangeCheck::new(
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            BoundValue::Constant(0),
            RangeCheckKind::LowerBound,
            0,
        )
    }

    fn make_upper_check() -> RangeCheck {
        RangeCheck::new(
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            BoundValue::Constant(100),
            RangeCheckKind::UpperBound,
            0,
        )
    }

    #[test]
    fn test_check_is_lower_bound() {
        assert!(make_lower_check().is_lower_bound());
        assert!(!make_upper_check().is_lower_bound());
    }

    #[test]
    fn test_check_is_upper_bound() {
        assert!(!make_lower_check().is_upper_bound());
        assert!(make_upper_check().is_upper_bound());

        let inclusive = RangeCheck::new(
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            BoundValue::Constant(99),
            RangeCheckKind::UpperBoundInclusive,
            0,
        );
        assert!(inclusive.is_upper_bound());
    }

    #[test]
    fn test_check_has_constant_bound() {
        assert!(make_lower_check().has_constant_bound());
        assert!(make_upper_check().has_constant_bound());

        let node_bound = RangeCheck::new(
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            BoundValue::Node(NodeId::new(10)),
            RangeCheckKind::UpperBound,
            0,
        );
        assert!(!node_bound.has_constant_bound());
    }

    #[test]
    fn test_check_constant_bound() {
        assert_eq!(make_lower_check().constant_bound(), Some(0));
        assert_eq!(make_upper_check().constant_bound(), Some(100));

        let node_bound = RangeCheck::new(
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            BoundValue::Node(NodeId::new(10)),
            RangeCheckKind::UpperBound,
            0,
        );
        assert_eq!(node_bound.constant_bound(), None);
    }

    // =========================================================================
    // RangeCheckCollection Tests
    // =========================================================================

    #[test]
    fn test_collection_new() {
        let coll = RangeCheckCollection::new();
        assert!(coll.is_empty());
        assert_eq!(coll.len(), 0);
    }

    #[test]
    fn test_collection_add() {
        let mut coll = RangeCheckCollection::new();
        coll.add(make_lower_check());
        coll.add(make_upper_check());

        assert_eq!(coll.len(), 2);
        assert!(!coll.is_empty());
    }

    #[test]
    fn test_collection_add_all() {
        let mut coll = RangeCheckCollection::new();
        coll.add_all(vec![make_lower_check(), make_upper_check()]);

        assert_eq!(coll.len(), 2);
    }

    #[test]
    fn test_collection_get() {
        let mut coll = RangeCheckCollection::new();
        coll.add(make_lower_check());

        assert!(coll.get(0).is_some());
        assert!(coll.get(1).is_none());
    }

    #[test]
    fn test_collection_for_loop() {
        let mut coll = RangeCheckCollection::new();

        let check1 = RangeCheck::new(
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            BoundValue::Constant(0),
            RangeCheckKind::LowerBound,
            0, // loop 0
        );
        let check2 = RangeCheck::new(
            NodeId::new(3),
            NodeId::new(4),
            NodeId::new(5),
            BoundValue::Constant(100),
            RangeCheckKind::UpperBound,
            1, // loop 1
        );

        coll.add(check1);
        coll.add(check2);

        let loop0_checks: Vec<_> = coll.for_loop(0).collect();
        assert_eq!(loop0_checks.len(), 1);

        let loop1_checks: Vec<_> = coll.for_loop(1).collect();
        assert_eq!(loop1_checks.len(), 1);

        let loop2_checks: Vec<_> = coll.for_loop(2).collect();
        assert_eq!(loop2_checks.len(), 0);
    }

    #[test]
    fn test_collection_for_iv() {
        let mut coll = RangeCheckCollection::new();

        let iv = NodeId::new(1);
        let check1 = RangeCheck::new(
            NodeId::new(0),
            iv,
            NodeId::new(2),
            BoundValue::Constant(0),
            RangeCheckKind::LowerBound,
            0,
        );
        let check2 = RangeCheck::new(
            NodeId::new(3),
            iv,
            NodeId::new(4),
            BoundValue::Constant(100),
            RangeCheckKind::UpperBound,
            0,
        );

        coll.add(check1);
        coll.add(check2);

        let iv_checks: Vec<_> = coll.for_iv(iv).collect();
        assert_eq!(iv_checks.len(), 2);

        let other_checks: Vec<_> = coll.for_iv(NodeId::new(999)).collect();
        assert_eq!(other_checks.len(), 0);
    }

    #[test]
    fn test_collection_count_lower_bounds() {
        let mut coll = RangeCheckCollection::new();
        coll.add(make_lower_check());
        coll.add(make_upper_check());
        coll.add(make_lower_check());

        assert_eq!(coll.count_lower_bounds(), 2);
    }

    #[test]
    fn test_collection_count_upper_bounds() {
        let mut coll = RangeCheckCollection::new();
        coll.add(make_lower_check());
        coll.add(make_upper_check());
        coll.add(make_upper_check());

        assert_eq!(coll.count_upper_bounds(), 2);
    }

    #[test]
    fn test_collection_count_constant_bounds() {
        let mut coll = RangeCheckCollection::new();
        coll.add(make_lower_check());
        coll.add(make_upper_check());

        let node_bound = RangeCheck::new(
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            BoundValue::Node(NodeId::new(10)),
            RangeCheckKind::UpperBound,
            0,
        );
        coll.add(node_bound);

        assert_eq!(coll.count_constant_bounds(), 2);
    }
}
