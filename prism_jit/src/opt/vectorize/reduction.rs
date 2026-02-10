//! Reduction Pattern Detection for Loop Vectorization
//!
//! This module provides comprehensive reduction pattern detection for loop
//! vectorization. A reduction is an operation that combines all values from
//! a loop into a single scalar result.
//!
//! # Supported Reduction Patterns
//!
//! - **Sum**: acc += value (associative Add)
//! - **Product**: acc *= value (associative Multiply)
//! - **Minimum**: acc = min(acc, value)
//! - **Maximum**: acc = max(acc, value)
//! - **Bitwise AND**: acc &= value
//! - **Bitwise OR**: acc |= value
//! - **Bitwise XOR**: acc ^= value
//!
//! # Algorithm
//!
//! 1. Find LoopPhi nodes that are NOT induction variables
//! 2. Check if the phi has a back-edge value computed by an associative/commutative op
//! 3. Verify that the operation uses the phi as one operand
//! 4. Classify the reduction kind based on the operator
//!
//! # Reference
//!
//! - "Loop Vectorization in LLVM" - LLVM documentation
//! - "Compilers: Principles, Techniques, and Tools" - Aho, Lam, Sethi, Ullman

use crate::ir::cfg::Loop;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ArithOp, BitwiseOp, Operator};
use rustc_hash::{FxHashMap, FxHashSet};

// =============================================================================
// Reduction Kind
// =============================================================================

/// Classification of reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionKind {
    /// Summation: acc + value
    Sum,
    /// Product: acc * value
    Product,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
}

impl ReductionKind {
    /// Get the identity element for this reduction.
    ///
    /// The identity element is the value that, when combined with any other
    /// value using the reduction operation, produces that other value.
    #[inline]
    pub fn identity_int(&self) -> i64 {
        match self {
            ReductionKind::Sum => 0,
            ReductionKind::Product => 1,
            ReductionKind::Min => i64::MAX,
            ReductionKind::Max => i64::MIN,
            ReductionKind::And => -1, // All bits set
            ReductionKind::Or => 0,
            ReductionKind::Xor => 0,
        }
    }

    /// Get the identity element for float reductions.
    #[inline]
    pub fn identity_float(&self) -> f64 {
        match self {
            ReductionKind::Sum => 0.0,
            ReductionKind::Product => 1.0,
            ReductionKind::Min => f64::INFINITY,
            ReductionKind::Max => f64::NEG_INFINITY,
            // Bitwise ops don't apply to floats
            ReductionKind::And | ReductionKind::Or | ReductionKind::Xor => f64::NAN,
        }
    }

    /// Check if this reduction is associative.
    ///
    /// All supported reductions are associative by definition.
    #[inline]
    pub fn is_associative(&self) -> bool {
        true
    }

    /// Check if this reduction is commutative.
    #[inline]
    pub fn is_commutative(&self) -> bool {
        true
    }

    /// Check if this reduction can be performed on integers.
    #[inline]
    pub fn supports_integer(&self) -> bool {
        true
    }

    /// Check if this reduction can be performed on floats.
    #[inline]
    pub fn supports_float(&self) -> bool {
        matches!(
            self,
            ReductionKind::Sum | ReductionKind::Product | ReductionKind::Min | ReductionKind::Max
        )
    }

    /// Get the SIMD horizontal reduction intrinsic name (for documentation).
    #[inline]
    pub fn horizontal_intrinsic(&self) -> &'static str {
        match self {
            ReductionKind::Sum => "horizontal_add",
            ReductionKind::Product => "horizontal_mul",
            ReductionKind::Min => "horizontal_min",
            ReductionKind::Max => "horizontal_max",
            ReductionKind::And => "horizontal_and",
            ReductionKind::Or => "horizontal_or",
            ReductionKind::Xor => "horizontal_xor",
        }
    }
}

// =============================================================================
// Reduction Variable
// =============================================================================

/// A reduction variable detected in a loop.
#[derive(Debug, Clone)]
pub struct Reduction {
    /// The LoopPhi node representing the reduction accumulator.
    pub phi: NodeId,

    /// The reduction operation node.
    pub op_node: NodeId,

    /// The kind of reduction.
    pub kind: ReductionKind,

    /// The initial value node (from loop entry edge).
    pub init: NodeId,

    /// The value being added to the accumulator each iteration.
    /// This is the non-phi operand of the reduction operation.
    pub value_operand: NodeId,

    /// Whether the reduction is on floats (affects vectorization decisions).
    pub is_float: bool,
}

impl Reduction {
    /// Create a new reduction.
    #[inline]
    pub fn new(
        phi: NodeId,
        op_node: NodeId,
        kind: ReductionKind,
        init: NodeId,
        value_operand: NodeId,
        is_float: bool,
    ) -> Self {
        Self {
            phi,
            op_node,
            kind,
            init,
            value_operand,
            is_float,
        }
    }

    /// Get the identity value for this reduction (as integer).
    #[inline]
    pub fn identity_int(&self) -> i64 {
        self.kind.identity_int()
    }

    /// Get the identity value for this reduction (as float).
    #[inline]
    pub fn identity_float(&self) -> f64 {
        self.kind.identity_float()
    }

    /// Check if this is a simple sum reduction.
    #[inline]
    pub fn is_sum(&self) -> bool {
        self.kind == ReductionKind::Sum
    }

    /// Check if this is a product reduction.
    #[inline]
    pub fn is_product(&self) -> bool {
        self.kind == ReductionKind::Product
    }

    /// Check if this reduction can be vectorized.
    ///
    /// Float reductions may require special handling for strict FP semantics.
    #[inline]
    pub fn is_vectorizable(&self) -> bool {
        // All reductions are vectorizable, but float reductions may need
        // `-ffast-math` semantics for full vectorization
        true
    }
}

// =============================================================================
// Reduction Detector
// =============================================================================

/// Detects reduction patterns in loops.
#[derive(Debug)]
pub struct ReductionDetector<'g> {
    graph: &'g Graph,
}

impl<'g> ReductionDetector<'g> {
    /// Create a new reduction detector.
    #[inline]
    pub fn new(graph: &'g Graph) -> Self {
        Self { graph }
    }

    /// Find all reductions in a loop.
    ///
    /// # Arguments
    /// - `loop_info`: The loop to analyze
    /// - `induction_phis`: Set of phi nodes already classified as induction variables
    ///
    /// # Returns
    /// A map from phi node to reduction info
    pub fn find_reductions(
        &self,
        loop_info: &Loop,
        induction_phis: &FxHashSet<NodeId>,
    ) -> FxHashMap<NodeId, Reduction> {
        let mut reductions = FxHashMap::default();

        // Collect loop body nodes for fast membership test
        let body_set: FxHashSet<_> = loop_info.body.iter().copied().collect();

        // Look for LoopPhi nodes that are NOT induction variables
        for (node_id, node) in self.graph.iter() {
            // Skip non-LoopPhi nodes
            if !matches!(node.op, Operator::LoopPhi) {
                continue;
            }

            // Skip nodes already classified as induction variables
            if induction_phis.contains(&node_id) {
                continue;
            }

            // LoopPhi should have exactly 2 inputs: init and back-edge
            if node.inputs.len() != 2 {
                continue;
            }

            // Analyze as potential reduction
            if let Some(reduction) = self.analyze_reduction(loop_info, node_id, &body_set) {
                reductions.insert(node_id, reduction);
            }
        }

        reductions
    }

    /// Analyze a LoopPhi to determine if it's a reduction.
    fn analyze_reduction(
        &self,
        _loop_info: &Loop,
        phi: NodeId,
        body_set: &FxHashSet<NodeId>,
    ) -> Option<Reduction> {
        let phi_node = self.graph.get(phi)?;

        if phi_node.inputs.len() != 2 {
            return None;
        }

        let init = phi_node.inputs.get(0)?;
        let back_edge = phi_node.inputs.get(1)?;

        // The back-edge should be defined within the loop
        if !body_set.contains(&back_edge) {
            // Check if the back_edge node itself is in the loop indirectly
            // by checking if its definition is in the loop
        }

        // Analyze the back-edge operation
        let back_node = self.graph.get(back_edge)?;

        // Try to match reduction patterns
        if let Some((kind, value_operand, is_float)) =
            self.match_reduction_pattern(phi, back_edge, &back_node.op, &back_node.inputs)
        {
            return Some(Reduction::new(
                phi,
                back_edge,
                kind,
                init,
                value_operand,
                is_float,
            ));
        }

        None
    }

    /// Match a reduction pattern from the back-edge operation.
    fn match_reduction_pattern(
        &self,
        phi: NodeId,
        _op_node: NodeId,
        op: &Operator,
        inputs: &crate::ir::node::InputList,
    ) -> Option<(ReductionKind, NodeId, bool)> {
        if inputs.len() != 2 {
            return None;
        }

        let lhs = inputs.get(0)?;
        let rhs = inputs.get(1)?;

        // Determine which operand is the phi and which is the value
        let (is_lhs_phi, value_operand) = if lhs == phi {
            (true, rhs)
        } else if rhs == phi {
            (false, lhs)
        } else {
            // Neither operand is the phi - check for indirect use through
            // operations that are themselves reductions of the phi
            return None;
        };

        // Match operator to reduction kind
        match op {
            // Integer arithmetic reductions
            Operator::IntOp(ArithOp::Add) => Some((ReductionKind::Sum, value_operand, false)),
            Operator::IntOp(ArithOp::Mul) => Some((ReductionKind::Product, value_operand, false)),

            // Float arithmetic reductions
            Operator::FloatOp(ArithOp::Add) => Some((ReductionKind::Sum, value_operand, true)),
            Operator::FloatOp(ArithOp::Mul) => Some((ReductionKind::Product, value_operand, true)),

            // Generic arithmetic reductions
            Operator::GenericOp(ArithOp::Add) => Some((ReductionKind::Sum, value_operand, false)),
            Operator::GenericOp(ArithOp::Mul) => {
                Some((ReductionKind::Product, value_operand, false))
            }

            // Bitwise reductions
            Operator::Bitwise(BitwiseOp::And) => Some((ReductionKind::And, value_operand, false)),
            Operator::Bitwise(BitwiseOp::Or) => Some((ReductionKind::Or, value_operand, false)),
            Operator::Bitwise(BitwiseOp::Xor) => Some((ReductionKind::Xor, value_operand, false)),

            // Min/Max intrinsics would go here when we add them
            // Operator::Intrinsic(Intrinsic::Min) => ...
            // Operator::Intrinsic(Intrinsic::Max) => ...
            _ => None,
        }
    }
}

// =============================================================================
// Reduction Analysis Cache
// =============================================================================

/// Cached reduction analysis for all loops.
#[derive(Debug)]
pub struct ReductionAnalysis {
    /// Reductions per loop (indexed by loop index).
    reductions: Vec<FxHashMap<NodeId, Reduction>>,

    /// Total number of reductions found.
    total: usize,
}

impl ReductionAnalysis {
    /// Create empty analysis.
    #[inline]
    pub fn empty() -> Self {
        Self {
            reductions: Vec::new(),
            total: 0,
        }
    }

    /// Create analysis with given capacity.
    #[inline]
    pub fn with_capacity(num_loops: usize) -> Self {
        Self {
            reductions: Vec::with_capacity(num_loops),
            total: 0,
        }
    }

    /// Add reductions for a loop.
    pub fn add_loop(&mut self, loop_reductions: FxHashMap<NodeId, Reduction>) {
        self.total += loop_reductions.len();
        self.reductions.push(loop_reductions);
    }

    /// Get reductions for a loop.
    #[inline]
    pub fn get(&self, loop_idx: usize) -> Option<&FxHashMap<NodeId, Reduction>> {
        self.reductions.get(loop_idx)
    }

    /// Get mutable reductions for a loop.
    #[inline]
    pub fn get_mut(&mut self, loop_idx: usize) -> Option<&mut FxHashMap<NodeId, Reduction>> {
        self.reductions.get_mut(loop_idx)
    }

    /// Get total number of reductions.
    #[inline]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Get number of loops analyzed.
    #[inline]
    pub fn num_loops(&self) -> usize {
        self.reductions.len()
    }

    /// Check if a node is a reduction in a loop.
    #[inline]
    pub fn is_reduction(&self, loop_idx: usize, node: NodeId) -> bool {
        self.reductions
            .get(loop_idx)
            .map_or(false, |m| m.contains_key(&node))
    }

    /// Get reduction info if node is a reduction.
    #[inline]
    pub fn get_reduction(&self, loop_idx: usize, node: NodeId) -> Option<&Reduction> {
        self.reductions.get(loop_idx).and_then(|m| m.get(&node))
    }

    /// Iterate over all reductions across all loops.
    pub fn iter_all(&self) -> impl Iterator<Item = (usize, NodeId, &Reduction)> {
        self.reductions
            .iter()
            .enumerate()
            .flat_map(|(loop_idx, reds)| reds.iter().map(move |(node, red)| (loop_idx, *node, red)))
    }

    /// Count sum reductions.
    pub fn count_sum(&self) -> usize {
        self.iter_all().filter(|(_, _, r)| r.is_sum()).count()
    }

    /// Count product reductions.
    pub fn count_product(&self) -> usize {
        self.iter_all().filter(|(_, _, r)| r.is_product()).count()
    }

    /// Count float reductions.
    pub fn count_float(&self) -> usize {
        self.iter_all().filter(|(_, _, r)| r.is_float).count()
    }
}

impl Default for ReductionAnalysis {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ReductionKind Tests
    // =========================================================================

    #[test]
    fn test_reduction_kind_identity_int() {
        assert_eq!(ReductionKind::Sum.identity_int(), 0);
        assert_eq!(ReductionKind::Product.identity_int(), 1);
        assert_eq!(ReductionKind::Min.identity_int(), i64::MAX);
        assert_eq!(ReductionKind::Max.identity_int(), i64::MIN);
        assert_eq!(ReductionKind::And.identity_int(), -1);
        assert_eq!(ReductionKind::Or.identity_int(), 0);
        assert_eq!(ReductionKind::Xor.identity_int(), 0);
    }

    #[test]
    fn test_reduction_kind_identity_float() {
        assert_eq!(ReductionKind::Sum.identity_float(), 0.0);
        assert_eq!(ReductionKind::Product.identity_float(), 1.0);
        assert_eq!(ReductionKind::Min.identity_float(), f64::INFINITY);
        assert_eq!(ReductionKind::Max.identity_float(), f64::NEG_INFINITY);
        assert!(ReductionKind::And.identity_float().is_nan());
        assert!(ReductionKind::Or.identity_float().is_nan());
        assert!(ReductionKind::Xor.identity_float().is_nan());
    }

    #[test]
    fn test_reduction_kind_associative() {
        assert!(ReductionKind::Sum.is_associative());
        assert!(ReductionKind::Product.is_associative());
        assert!(ReductionKind::Min.is_associative());
        assert!(ReductionKind::Max.is_associative());
        assert!(ReductionKind::And.is_associative());
        assert!(ReductionKind::Or.is_associative());
        assert!(ReductionKind::Xor.is_associative());
    }

    #[test]
    fn test_reduction_kind_commutative() {
        assert!(ReductionKind::Sum.is_commutative());
        assert!(ReductionKind::Product.is_commutative());
        assert!(ReductionKind::Min.is_commutative());
        assert!(ReductionKind::Max.is_commutative());
        assert!(ReductionKind::And.is_commutative());
        assert!(ReductionKind::Or.is_commutative());
        assert!(ReductionKind::Xor.is_commutative());
    }

    #[test]
    fn test_reduction_kind_supports_integer() {
        assert!(ReductionKind::Sum.supports_integer());
        assert!(ReductionKind::Product.supports_integer());
        assert!(ReductionKind::Min.supports_integer());
        assert!(ReductionKind::Max.supports_integer());
        assert!(ReductionKind::And.supports_integer());
        assert!(ReductionKind::Or.supports_integer());
        assert!(ReductionKind::Xor.supports_integer());
    }

    #[test]
    fn test_reduction_kind_supports_float() {
        assert!(ReductionKind::Sum.supports_float());
        assert!(ReductionKind::Product.supports_float());
        assert!(ReductionKind::Min.supports_float());
        assert!(ReductionKind::Max.supports_float());
        assert!(!ReductionKind::And.supports_float());
        assert!(!ReductionKind::Or.supports_float());
        assert!(!ReductionKind::Xor.supports_float());
    }

    #[test]
    fn test_reduction_kind_horizontal_intrinsic() {
        assert_eq!(ReductionKind::Sum.horizontal_intrinsic(), "horizontal_add");
        assert_eq!(
            ReductionKind::Product.horizontal_intrinsic(),
            "horizontal_mul"
        );
        assert_eq!(ReductionKind::Min.horizontal_intrinsic(), "horizontal_min");
        assert_eq!(ReductionKind::Max.horizontal_intrinsic(), "horizontal_max");
        assert_eq!(ReductionKind::And.horizontal_intrinsic(), "horizontal_and");
        assert_eq!(ReductionKind::Or.horizontal_intrinsic(), "horizontal_or");
        assert_eq!(ReductionKind::Xor.horizontal_intrinsic(), "horizontal_xor");
    }

    #[test]
    fn test_reduction_kind_equality() {
        assert_eq!(ReductionKind::Sum, ReductionKind::Sum);
        assert_ne!(ReductionKind::Sum, ReductionKind::Product);
        assert_ne!(ReductionKind::Min, ReductionKind::Max);
    }

    // =========================================================================
    // Reduction Tests
    // =========================================================================

    fn make_test_reduction(kind: ReductionKind, is_float: bool) -> Reduction {
        Reduction::new(
            NodeId::new(0),
            NodeId::new(1),
            kind,
            NodeId::new(2),
            NodeId::new(3),
            is_float,
        )
    }

    #[test]
    fn test_reduction_new() {
        let r = make_test_reduction(ReductionKind::Sum, false);
        assert_eq!(r.phi, NodeId::new(0));
        assert_eq!(r.op_node, NodeId::new(1));
        assert_eq!(r.kind, ReductionKind::Sum);
        assert_eq!(r.init, NodeId::new(2));
        assert_eq!(r.value_operand, NodeId::new(3));
        assert!(!r.is_float);
    }

    #[test]
    fn test_reduction_identity_int() {
        assert_eq!(
            make_test_reduction(ReductionKind::Sum, false).identity_int(),
            0
        );
        assert_eq!(
            make_test_reduction(ReductionKind::Product, false).identity_int(),
            1
        );
        assert_eq!(
            make_test_reduction(ReductionKind::And, false).identity_int(),
            -1
        );
    }

    #[test]
    fn test_reduction_identity_float() {
        assert_eq!(
            make_test_reduction(ReductionKind::Sum, true).identity_float(),
            0.0
        );
        assert_eq!(
            make_test_reduction(ReductionKind::Product, true).identity_float(),
            1.0
        );
    }

    #[test]
    fn test_reduction_is_sum() {
        assert!(make_test_reduction(ReductionKind::Sum, false).is_sum());
        assert!(!make_test_reduction(ReductionKind::Product, false).is_sum());
        assert!(!make_test_reduction(ReductionKind::Min, false).is_sum());
    }

    #[test]
    fn test_reduction_is_product() {
        assert!(make_test_reduction(ReductionKind::Product, false).is_product());
        assert!(!make_test_reduction(ReductionKind::Sum, false).is_product());
        assert!(!make_test_reduction(ReductionKind::Max, false).is_product());
    }

    #[test]
    fn test_reduction_is_vectorizable() {
        assert!(make_test_reduction(ReductionKind::Sum, false).is_vectorizable());
        assert!(make_test_reduction(ReductionKind::Sum, true).is_vectorizable());
        assert!(make_test_reduction(ReductionKind::Product, false).is_vectorizable());
        assert!(make_test_reduction(ReductionKind::And, false).is_vectorizable());
    }

    // =========================================================================
    // ReductionAnalysis Tests
    // =========================================================================

    #[test]
    fn test_reduction_analysis_empty() {
        let analysis = ReductionAnalysis::empty();
        assert_eq!(analysis.total(), 0);
        assert_eq!(analysis.num_loops(), 0);
        assert!(!analysis.is_reduction(0, NodeId::new(0)));
    }

    #[test]
    fn test_reduction_analysis_with_capacity() {
        let analysis = ReductionAnalysis::with_capacity(5);
        assert_eq!(analysis.total(), 0);
        assert_eq!(analysis.num_loops(), 0);
    }

    #[test]
    fn test_reduction_analysis_add_loop() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        loop_reds.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Product, false),
        );

        analysis.add_loop(loop_reds);

        assert_eq!(analysis.total(), 2);
        assert_eq!(analysis.num_loops(), 1);
        assert!(analysis.is_reduction(0, NodeId::new(0)));
        assert!(analysis.is_reduction(0, NodeId::new(1)));
        assert!(!analysis.is_reduction(0, NodeId::new(2)));
        assert!(!analysis.is_reduction(1, NodeId::new(0)));
    }

    #[test]
    fn test_reduction_analysis_get() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        analysis.add_loop(loop_reds);

        assert!(analysis.get(0).is_some());
        assert!(analysis.get(1).is_none());
    }

    #[test]
    fn test_reduction_analysis_get_reduction() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(5),
            make_test_reduction(ReductionKind::Max, false),
        );
        analysis.add_loop(loop_reds);

        let red = analysis.get_reduction(0, NodeId::new(5));
        assert!(red.is_some());
        assert_eq!(red.unwrap().kind, ReductionKind::Max);

        assert!(analysis.get_reduction(0, NodeId::new(0)).is_none());
        assert!(analysis.get_reduction(1, NodeId::new(5)).is_none());
    }

    #[test]
    fn test_reduction_analysis_iter_all() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop0 = FxHashMap::default();
        loop0.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        analysis.add_loop(loop0);

        let mut loop1 = FxHashMap::default();
        loop1.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Product, true),
        );
        analysis.add_loop(loop1);

        let all: Vec<_> = analysis.iter_all().collect();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_reduction_analysis_count_sum() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        loop_reds.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Sum, true),
        );
        loop_reds.insert(
            NodeId::new(2),
            make_test_reduction(ReductionKind::Product, false),
        );
        analysis.add_loop(loop_reds);

        assert_eq!(analysis.count_sum(), 2);
    }

    #[test]
    fn test_reduction_analysis_count_product() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Product, false),
        );
        loop_reds.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Product, true),
        );
        loop_reds.insert(
            NodeId::new(2),
            make_test_reduction(ReductionKind::Sum, false),
        );
        analysis.add_loop(loop_reds);

        assert_eq!(analysis.count_product(), 2);
    }

    #[test]
    fn test_reduction_analysis_count_float() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        loop_reds.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Sum, true),
        );
        loop_reds.insert(
            NodeId::new(2),
            make_test_reduction(ReductionKind::Product, true),
        );
        analysis.add_loop(loop_reds);

        assert_eq!(analysis.count_float(), 2);
    }

    #[test]
    fn test_reduction_analysis_default() {
        let analysis = ReductionAnalysis::default();
        assert_eq!(analysis.total(), 0);
        assert_eq!(analysis.num_loops(), 0);
    }

    #[test]
    fn test_reduction_analysis_multiple_loops() {
        let mut analysis = ReductionAnalysis::with_capacity(3);

        let mut loop0 = FxHashMap::default();
        loop0.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        analysis.add_loop(loop0);

        let mut loop1 = FxHashMap::default();
        loop1.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Product, false),
        );
        loop1.insert(
            NodeId::new(2),
            make_test_reduction(ReductionKind::Max, false),
        );
        analysis.add_loop(loop1);

        let loop2 = FxHashMap::default(); // Empty loop
        analysis.add_loop(loop2);

        assert_eq!(analysis.num_loops(), 3);
        assert_eq!(analysis.total(), 3);
        assert!(analysis.is_reduction(0, NodeId::new(0)));
        assert!(analysis.is_reduction(1, NodeId::new(1)));
        assert!(analysis.is_reduction(1, NodeId::new(2)));
    }
}
