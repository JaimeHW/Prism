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
