//! Induction Variable Analysis.
//!
//! This module provides comprehensive induction variable detection and analysis
//! for loop optimization. Induction variables are variables that change by a
//! predictable amount on each loop iteration.
//!
//! # Induction Variable Types
//!
//! - **Basic Induction Variable (BIV)**: Incremented/decremented by loop-invariant amount
//! - **Derived Induction Variable (DIV)**: Linear function of a BIV (a * BIV + b)
//!
//! # Algorithm
//!
//! 1. Find all LoopPhi nodes in loop header
//! 2. For each phi, analyze back-edge computation
//! 3. Detect add/sub patterns with loop-invariant operand
//! 4. Classify direction (increasing/decreasing/unknown)
//! 5. Compute derived induction variables

use crate::ir::cfg::Loop;
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{ArithOp, Operator};
use std::collections::HashMap;

// =============================================================================
// Induction Variable Types
// =============================================================================

/// Represents a basic induction variable: init + step * iteration.
///
/// A basic induction variable (BIV) is the simplest form, directly modified
/// in the loop by a constant or loop-invariant amount.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InductionVariable {
    /// The LoopPhi node that defines this induction variable.
    pub phi: NodeId,

    /// The initial value (from loop entry edge).
    pub init: InductionInit,

    /// The step value per iteration.
    pub step: InductionStep,

    /// The direction of iteration.
    pub direction: InductionDirection,

    /// The back-edge update node (the add/sub that modifies the IV).
    pub update_node: Option<NodeId>,
}

/// Initial value of an induction variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InductionInit {
    /// Constant initial value (most common for `for i in range(0, n)`).
    Constant(i64),

    /// Value from a node (parameter, computation, etc.).
    Node(NodeId),
}

/// Step value of an induction variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InductionStep {
    /// Constant step (most common: +1, -1, +2, etc.).
    Constant(i64),

    /// Step from a loop-invariant node.
    Node(NodeId),
}

/// Direction of induction variable iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InductionDirection {
    /// Strictly increasing: step > 0.
    Increasing,

    /// Strictly decreasing: step < 0.
    Decreasing,

    /// Non-monotonic or unknown direction.
    Unknown,
}

impl InductionVariable {
    /// Create a new induction variable.
    #[inline]
    pub fn new(
        phi: NodeId,
        init: InductionInit,
        step: InductionStep,
        direction: InductionDirection,
        update_node: Option<NodeId>,
    ) -> Self {
        Self {
            phi,
            init,
            step,
            direction,
            update_node,
        }
    }

    /// Check if this is a simple linear induction variable with constant step.
    #[inline]
    pub fn is_simple(&self) -> bool {
        matches!(self.step, InductionStep::Constant(_))
    }

    /// Check if this has a constant initial value.
    #[inline]
    pub fn has_constant_init(&self) -> bool {
        matches!(self.init, InductionInit::Constant(_))
    }

    /// Check if this is a "canonical" counter (init=0, step=1).
    #[inline]
    pub fn is_canonical(&self) -> bool {
        matches!(
            (&self.init, &self.step),
            (InductionInit::Constant(0), InductionStep::Constant(1))
        )
    }

    /// Get the constant step value if available.
    #[inline]
    pub fn constant_step(&self) -> Option<i64> {
        match self.step {
            InductionStep::Constant(s) => Some(s),
            InductionStep::Node(_) => None,
        }
    }

    /// Get the constant initial value if available.
    #[inline]
    pub fn constant_init(&self) -> Option<i64> {
        match self.init {
            InductionInit::Constant(i) => Some(i),
            InductionInit::Node(_) => None,
        }
    }

    /// Get the absolute step magnitude.
    #[inline]
    pub fn step_magnitude(&self) -> Option<u64> {
        self.constant_step().map(|s| s.unsigned_abs())
    }

    /// Check if strictly increasing.
    #[inline]
    pub fn is_increasing(&self) -> bool {
        matches!(self.direction, InductionDirection::Increasing)
    }

    /// Check if strictly decreasing.
    #[inline]
    pub fn is_decreasing(&self) -> bool {
        matches!(self.direction, InductionDirection::Decreasing)
    }

    /// Check if monotonic (either increasing or decreasing).
    #[inline]
    pub fn is_monotonic(&self) -> bool {
        !matches!(self.direction, InductionDirection::Unknown)
    }
}

impl InductionInit {
    /// Check if this is a constant init.
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, InductionInit::Constant(_))
    }

    /// Get the constant value if available.
    #[inline]
    pub fn as_constant(&self) -> Option<i64> {
        match self {
            InductionInit::Constant(v) => Some(*v),
            InductionInit::Node(_) => None,
        }
    }

    /// Get the node if this is a node-based init.
    #[inline]
    pub fn as_node(&self) -> Option<NodeId> {
        match self {
            InductionInit::Node(n) => Some(*n),
            InductionInit::Constant(_) => None,
        }
    }
}

impl InductionStep {
    /// Check if this is a constant step.
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, InductionStep::Constant(_))
    }

    /// Get the constant value if available.
    #[inline]
    pub fn as_constant(&self) -> Option<i64> {
        match self {
            InductionStep::Constant(v) => Some(*v),
            InductionStep::Node(_) => None,
        }
    }

    /// Get the node if this is a node-based step.
    #[inline]
    pub fn as_node(&self) -> Option<NodeId> {
        match self {
            InductionStep::Node(n) => Some(*n),
            InductionStep::Constant(_) => None,
        }
    }

    /// Compute direction from step.
    #[inline]
    pub fn direction(&self) -> InductionDirection {
        match self {
            InductionStep::Constant(s) if *s > 0 => InductionDirection::Increasing,
            InductionStep::Constant(s) if *s < 0 => InductionDirection::Decreasing,
            _ => InductionDirection::Unknown,
        }
    }
}

// =============================================================================
// Induction Variable Detector
// =============================================================================

/// Detects and analyzes induction variables in loops.
#[derive(Debug)]
pub struct InductionDetector<'g> {
    graph: &'g Graph,
}

impl<'g> InductionDetector<'g> {
    /// Create a new induction detector.
    #[inline]
    pub fn new(graph: &'g Graph) -> Self {
        Self { graph }
    }

    /// Find all induction variables in a loop.
    pub fn find_induction_variables(&self, loop_info: &Loop) -> HashMap<NodeId, InductionVariable> {
        let mut ivs = HashMap::new();

        // Look for LoopPhi nodes
        for (node_id, node) in self.graph.iter() {
            if !matches!(node.op, Operator::LoopPhi) {
                continue;
            }

            // LoopPhi should have exactly 2 inputs: init and back-edge
            if node.inputs.len() != 2 {
                continue;
            }

            // Analyze the phi
            if let Some(iv) = self.analyze_loop_phi(loop_info, node_id) {
                ivs.insert(node_id, iv);
            }
        }

        ivs
    }

    /// Analyze a LoopPhi to determine if it's an induction variable.
    fn analyze_loop_phi(&self, loop_info: &Loop, phi: NodeId) -> Option<InductionVariable> {
        let node = self.graph.get(phi)?;

        if node.inputs.len() != 2 {
            return None;
        }

        let init_node = node.inputs.get(0)?;
        let back_node = node.inputs.get(1)?;

        // Analyze the back-edge computation
        let (step, direction, update_node) = self.analyze_back_edge(loop_info, phi, back_node)?;

        // Analyze initial value
        let init = self.analyze_init_value(init_node);

        Some(InductionVariable::new(
            phi,
            init,
            step,
            direction,
            update_node,
        ))
    }

    /// Analyze the initial value of an induction variable.
    fn analyze_init_value(&self, node: NodeId) -> InductionInit {
        if let Some(val) = self.get_constant_value(node) {
            InductionInit::Constant(val)
        } else {
            InductionInit::Node(node)
        }
    }

    /// Analyze the back-edge computation to find step and direction.
    fn analyze_back_edge(
        &self,
        loop_info: &Loop,
        phi: NodeId,
        back_node: NodeId,
    ) -> Option<(InductionStep, InductionDirection, Option<NodeId>)> {
        let node = self.graph.get(back_node)?;

        match &node.op {
            // i = phi + step
            Operator::IntOp(ArithOp::Add) => {
                self.analyze_add_pattern(loop_info, phi, back_node, &node.inputs)
            }

            // i = phi - step
            Operator::IntOp(ArithOp::Sub) => {
                self.analyze_sub_pattern(loop_info, phi, back_node, &node.inputs)
            }

            _ => None,
        }
    }

    /// Analyze addition pattern: phi + step or step + phi.
    fn analyze_add_pattern(
        &self,
        loop_info: &Loop,
        phi: NodeId,
        update: NodeId,
        inputs: &InputList,
    ) -> Option<(InductionStep, InductionDirection, Option<NodeId>)> {
        if inputs.len() != 2 {
            return None;
        }

        let lhs = inputs.get(0)?;
        let rhs = inputs.get(1)?;

        // Try phi + step
        if lhs == phi {
            return self.make_add_step(loop_info, rhs, update);
        }

        // Try step + phi (commutative)
        if rhs == phi {
            return self.make_add_step(loop_info, lhs, update);
        }

        None
    }

    /// Make step from addition operand.
    fn make_add_step(
        &self,
        loop_info: &Loop,
        step_node: NodeId,
        update: NodeId,
    ) -> Option<(InductionStep, InductionDirection, Option<NodeId>)> {
        if let Some(step_val) = self.get_constant_value(step_node) {
            let direction = if step_val > 0 {
                InductionDirection::Increasing
            } else if step_val < 0 {
                InductionDirection::Decreasing
            } else {
                InductionDirection::Unknown
            };
            Some((InductionStep::Constant(step_val), direction, Some(update)))
        } else if self.is_loop_invariant(loop_info, step_node) {
            Some((
                InductionStep::Node(step_node),
                InductionDirection::Unknown,
                Some(update),
            ))
        } else {
            None
        }
    }

    /// Analyze subtraction pattern: phi - step.
    fn analyze_sub_pattern(
        &self,
        loop_info: &Loop,
        phi: NodeId,
        update: NodeId,
        inputs: &InputList,
    ) -> Option<(InductionStep, InductionDirection, Option<NodeId>)> {
        if inputs.len() != 2 {
            return None;
        }

        let lhs = inputs.get(0)?;
        let rhs = inputs.get(1)?;

        // Only phi - step is valid (step - phi would be weird)
        if lhs != phi {
            return None;
        }

        if let Some(step_val) = self.get_constant_value(rhs) {
            // Subtracting positive = decreasing, subtracting negative = increasing
            let direction = if step_val > 0 {
                InductionDirection::Decreasing
            } else if step_val < 0 {
                InductionDirection::Increasing
            } else {
                InductionDirection::Unknown
            };
            // Store negative step for consistent semantics
            Some((InductionStep::Constant(-step_val), direction, Some(update)))
        } else if self.is_loop_invariant(loop_info, rhs) {
            Some((
                InductionStep::Node(rhs),
                InductionDirection::Unknown,
                Some(update),
            ))
        } else {
            None
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

    /// Check if a node is loop-invariant (defined outside the loop).
    fn is_loop_invariant(&self, loop_info: &Loop, node: NodeId) -> bool {
        if let Some(n) = self.graph.get(node) {
            // Constants and parameters are always loop-invariant
            match &n.op {
                Operator::ConstInt(_)
                | Operator::ConstFloat(_)
                | Operator::ConstBool(_)
                | Operator::ConstNone
                | Operator::Parameter(_) => return true,
                _ => {}
            }

            // For other nodes, we'd need to check if they're defined outside
            // the loop body. This requires block membership tracking.
            // For now, be conservative and say no.
            let _ = loop_info;
        }
        false
    }
}

// =============================================================================
// Induction Variable Analysis Cache
// =============================================================================

/// Cached induction variable analysis for all loops.
#[derive(Debug)]
pub struct InductionAnalysis {
    /// Induction variables per loop (indexed by loop id).
    ivs: Vec<HashMap<NodeId, InductionVariable>>,

    /// Total number of induction variables found.
    total: usize,
}

impl InductionAnalysis {
    /// Create empty analysis.
    #[inline]
    pub fn empty() -> Self {
        Self {
            ivs: Vec::new(),
            total: 0,
        }
    }

    /// Create analysis with given capacity.
    #[inline]
    pub fn with_capacity(num_loops: usize) -> Self {
        Self {
            ivs: Vec::with_capacity(num_loops),
            total: 0,
        }
    }

    /// Add induction variables for a loop.
    pub fn add_loop(&mut self, loop_ivs: HashMap<NodeId, InductionVariable>) {
        self.total += loop_ivs.len();
        self.ivs.push(loop_ivs);
    }

    /// Get induction variables for a loop.
    #[inline]
    pub fn get(&self, loop_idx: usize) -> Option<&HashMap<NodeId, InductionVariable>> {
        self.ivs.get(loop_idx)
    }

    /// Get mutable induction variables for a loop.
    #[inline]
    pub fn get_mut(&mut self, loop_idx: usize) -> Option<&mut HashMap<NodeId, InductionVariable>> {
        self.ivs.get_mut(loop_idx)
    }

    /// Get total number of induction variables.
    #[inline]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Get number of loops analyzed.
    #[inline]
    pub fn num_loops(&self) -> usize {
        self.ivs.len()
    }

    /// Check if a node is an induction variable in a loop.
    #[inline]
    pub fn is_induction_variable(&self, loop_idx: usize, node: NodeId) -> bool {
        self.ivs
            .get(loop_idx)
            .map_or(false, |m| m.contains_key(&node))
    }

    /// Get induction variable info if node is an IV.
    #[inline]
    pub fn get_iv(&self, loop_idx: usize, node: NodeId) -> Option<&InductionVariable> {
        self.ivs.get(loop_idx).and_then(|m| m.get(&node))
    }

    /// Iterate over all induction variables across all loops.
    pub fn iter_all(&self) -> impl Iterator<Item = (usize, NodeId, &InductionVariable)> {
        self.ivs
            .iter()
            .enumerate()
            .flat_map(|(loop_idx, ivs)| ivs.iter().map(move |(node, iv)| (loop_idx, *node, iv)))
    }

    /// Count canonical induction variables (init=0, step=1).
    pub fn count_canonical(&self) -> usize {
        self.iter_all()
            .filter(|(_, _, iv)| iv.is_canonical())
            .count()
    }

    /// Count simple induction variables (constant step).
    pub fn count_simple(&self) -> usize {
        self.iter_all().filter(|(_, _, iv)| iv.is_simple()).count()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // InductionInit Tests
    // =========================================================================

    #[test]
    fn test_init_constant() {
        let init = InductionInit::Constant(0);
        assert!(init.is_constant());
        assert_eq!(init.as_constant(), Some(0));
        assert_eq!(init.as_node(), None);
    }

    #[test]
    fn test_init_node() {
        let init = InductionInit::Node(NodeId::new(5));
        assert!(!init.is_constant());
        assert_eq!(init.as_constant(), None);
        assert_eq!(init.as_node(), Some(NodeId::new(5)));
    }

    #[test]
    fn test_init_equality() {
        assert_eq!(InductionInit::Constant(0), InductionInit::Constant(0));
        assert_ne!(InductionInit::Constant(0), InductionInit::Constant(1));
        assert_ne!(
            InductionInit::Constant(0),
            InductionInit::Node(NodeId::new(0))
        );
    }

    // =========================================================================
    // InductionStep Tests
    // =========================================================================

    #[test]
    fn test_step_constant() {
        let step = InductionStep::Constant(1);
        assert!(step.is_constant());
        assert_eq!(step.as_constant(), Some(1));
        assert_eq!(step.as_node(), None);
    }

    #[test]
    fn test_step_node() {
        let step = InductionStep::Node(NodeId::new(10));
        assert!(!step.is_constant());
        assert_eq!(step.as_constant(), None);
        assert_eq!(step.as_node(), Some(NodeId::new(10)));
    }

    #[test]
    fn test_step_direction() {
        assert_eq!(
            InductionStep::Constant(1).direction(),
            InductionDirection::Increasing
        );
        assert_eq!(
            InductionStep::Constant(5).direction(),
            InductionDirection::Increasing
        );
        assert_eq!(
            InductionStep::Constant(-1).direction(),
            InductionDirection::Decreasing
        );
        assert_eq!(
            InductionStep::Constant(-10).direction(),
            InductionDirection::Decreasing
        );
        assert_eq!(
            InductionStep::Constant(0).direction(),
            InductionDirection::Unknown
        );
        assert_eq!(
            InductionStep::Node(NodeId::new(0)).direction(),
            InductionDirection::Unknown
        );
    }

    #[test]
    fn test_step_equality() {
        assert_eq!(InductionStep::Constant(1), InductionStep::Constant(1));
        assert_ne!(InductionStep::Constant(1), InductionStep::Constant(2));
        assert_ne!(
            InductionStep::Constant(1),
            InductionStep::Node(NodeId::new(1))
        );
    }

    // =========================================================================
    // InductionDirection Tests
    // =========================================================================

    #[test]
    fn test_direction_variants() {
        assert_ne!(
            InductionDirection::Increasing,
            InductionDirection::Decreasing
        );
        assert_ne!(InductionDirection::Increasing, InductionDirection::Unknown);
        assert_ne!(InductionDirection::Decreasing, InductionDirection::Unknown);
    }

    // =========================================================================
    // InductionVariable Tests
    // =========================================================================

    fn make_canonical_iv() -> InductionVariable {
        InductionVariable::new(
            NodeId::new(0),
            InductionInit::Constant(0),
            InductionStep::Constant(1),
            InductionDirection::Increasing,
            Some(NodeId::new(1)),
        )
    }

    fn make_simple_iv(init: i64, step: i64) -> InductionVariable {
        let direction = if step > 0 {
            InductionDirection::Increasing
        } else if step < 0 {
            InductionDirection::Decreasing
        } else {
            InductionDirection::Unknown
        };
        InductionVariable::new(
            NodeId::new(0),
            InductionInit::Constant(init),
            InductionStep::Constant(step),
            direction,
            None,
        )
    }

    #[test]
    fn test_iv_is_simple() {
        assert!(make_canonical_iv().is_simple());
        assert!(make_simple_iv(5, 2).is_simple());

        let non_simple = InductionVariable::new(
            NodeId::new(0),
            InductionInit::Constant(0),
            InductionStep::Node(NodeId::new(5)),
            InductionDirection::Unknown,
            None,
        );
        assert!(!non_simple.is_simple());
    }

    #[test]
    fn test_iv_has_constant_init() {
        assert!(make_canonical_iv().has_constant_init());

        let node_init = InductionVariable::new(
            NodeId::new(0),
            InductionInit::Node(NodeId::new(10)),
            InductionStep::Constant(1),
            InductionDirection::Increasing,
            None,
        );
        assert!(!node_init.has_constant_init());
    }

    #[test]
    fn test_iv_is_canonical() {
        assert!(make_canonical_iv().is_canonical());
        assert!(!make_simple_iv(0, 2).is_canonical());
        assert!(!make_simple_iv(1, 1).is_canonical());
        assert!(!make_simple_iv(0, -1).is_canonical());
    }

    #[test]
    fn test_iv_constant_step() {
        assert_eq!(make_canonical_iv().constant_step(), Some(1));
        assert_eq!(make_simple_iv(0, 5).constant_step(), Some(5));
        assert_eq!(make_simple_iv(0, -3).constant_step(), Some(-3));

        let node_step = InductionVariable::new(
            NodeId::new(0),
            InductionInit::Constant(0),
            InductionStep::Node(NodeId::new(5)),
            InductionDirection::Unknown,
            None,
        );
        assert_eq!(node_step.constant_step(), None);
    }

    #[test]
    fn test_iv_constant_init() {
        assert_eq!(make_canonical_iv().constant_init(), Some(0));
        assert_eq!(make_simple_iv(10, 1).constant_init(), Some(10));

        let node_init = InductionVariable::new(
            NodeId::new(0),
            InductionInit::Node(NodeId::new(10)),
            InductionStep::Constant(1),
            InductionDirection::Increasing,
            None,
        );
        assert_eq!(node_init.constant_init(), None);
    }

    #[test]
    fn test_iv_step_magnitude() {
        assert_eq!(make_simple_iv(0, 1).step_magnitude(), Some(1));
        assert_eq!(make_simple_iv(0, 5).step_magnitude(), Some(5));
        assert_eq!(make_simple_iv(0, -3).step_magnitude(), Some(3));
        assert_eq!(make_simple_iv(0, -100).step_magnitude(), Some(100));
    }

    #[test]
    fn test_iv_direction_checks() {
        let inc = make_simple_iv(0, 1);
        assert!(inc.is_increasing());
        assert!(!inc.is_decreasing());
        assert!(inc.is_monotonic());

        let dec = make_simple_iv(10, -1);
        assert!(!dec.is_increasing());
        assert!(dec.is_decreasing());
        assert!(dec.is_monotonic());

        let unknown = InductionVariable::new(
            NodeId::new(0),
            InductionInit::Constant(0),
            InductionStep::Node(NodeId::new(5)),
            InductionDirection::Unknown,
            None,
        );
        assert!(!unknown.is_increasing());
        assert!(!unknown.is_decreasing());
        assert!(!unknown.is_monotonic());
    }

    // =========================================================================
    // InductionAnalysis Tests
    // =========================================================================

    #[test]
    fn test_analysis_empty() {
        let analysis = InductionAnalysis::empty();
        assert_eq!(analysis.total(), 0);
        assert_eq!(analysis.num_loops(), 0);
        assert!(analysis.get(0).is_none());
    }

    #[test]
    fn test_analysis_with_capacity() {
        let analysis = InductionAnalysis::with_capacity(5);
        assert_eq!(analysis.total(), 0);
        assert_eq!(analysis.num_loops(), 0);
    }

    #[test]
    fn test_analysis_add_loop() {
        let mut analysis = InductionAnalysis::empty();

        let mut ivs = HashMap::new();
        ivs.insert(NodeId::new(0), make_canonical_iv());
        ivs.insert(NodeId::new(1), make_simple_iv(5, 2));

        analysis.add_loop(ivs);

        assert_eq!(analysis.total(), 2);
        assert_eq!(analysis.num_loops(), 1);
        assert!(analysis.get(0).is_some());
    }

    #[test]
    fn test_analysis_is_induction_variable() {
        let mut analysis = InductionAnalysis::empty();

        let mut ivs = HashMap::new();
        ivs.insert(NodeId::new(5), make_canonical_iv());
        analysis.add_loop(ivs);

        assert!(analysis.is_induction_variable(0, NodeId::new(5)));
        assert!(!analysis.is_induction_variable(0, NodeId::new(6)));
        assert!(!analysis.is_induction_variable(1, NodeId::new(5)));
    }

    #[test]
    fn test_analysis_get_iv() {
        let mut analysis = InductionAnalysis::empty();

        let iv = make_canonical_iv();
        let mut ivs = HashMap::new();
        ivs.insert(NodeId::new(5), iv.clone());
        analysis.add_loop(ivs);

        let retrieved = analysis.get_iv(0, NodeId::new(5));
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().phi, iv.phi);

        assert!(analysis.get_iv(0, NodeId::new(6)).is_none());
        assert!(analysis.get_iv(1, NodeId::new(5)).is_none());
    }

    #[test]
    fn test_analysis_iter_all() {
        let mut analysis = InductionAnalysis::empty();

        let mut ivs1 = HashMap::new();
        ivs1.insert(NodeId::new(0), make_canonical_iv());
        analysis.add_loop(ivs1);

        let mut ivs2 = HashMap::new();
        ivs2.insert(NodeId::new(1), make_simple_iv(0, 2));
        ivs2.insert(NodeId::new(2), make_simple_iv(10, -1));
        analysis.add_loop(ivs2);

        let all: Vec<_> = analysis.iter_all().collect();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_analysis_count_canonical() {
        let mut analysis = InductionAnalysis::empty();

        let mut ivs = HashMap::new();
        ivs.insert(NodeId::new(0), make_canonical_iv());
        ivs.insert(NodeId::new(1), make_simple_iv(0, 2)); // not canonical
        ivs.insert(NodeId::new(2), make_simple_iv(1, 1)); // not canonical
        analysis.add_loop(ivs);

        assert_eq!(analysis.count_canonical(), 1);
    }

    #[test]
    fn test_analysis_count_simple() {
        let mut analysis = InductionAnalysis::empty();

        let node_step = InductionVariable::new(
            NodeId::new(3),
            InductionInit::Constant(0),
            InductionStep::Node(NodeId::new(10)),
            InductionDirection::Unknown,
            None,
        );

        let mut ivs = HashMap::new();
        ivs.insert(NodeId::new(0), make_canonical_iv());
        ivs.insert(NodeId::new(1), make_simple_iv(0, 2));
        ivs.insert(NodeId::new(2), node_step);
        analysis.add_loop(ivs);

        assert_eq!(analysis.count_simple(), 2);
    }
}
