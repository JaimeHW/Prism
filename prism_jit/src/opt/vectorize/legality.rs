//! Vectorization Legality Analysis
//!
//! This module determines whether vectorization is legal (safe) for a given
//! loop or code region. It checks for conditions that would make vectorization
//! produce incorrect results.
//!
//! # Legality Conditions
//!
//! Vectorization is legal if:
//!
//! 1. **No unsafe dependences**: All loop-carried dependences are forward-only
//! 2. **Vectorizable operations**: All operations can be converted to SIMD
//! 3. **Affine access patterns**: Memory accesses follow base + stride * i pattern
//! 4. **No complex control flow**: No data-dependent exits (or can be masked)
//! 5. **No function calls**: Unless vectorizable intrinsics
//!
//! # Architecture
//!
//! The legality analyzer performs multi-level checking:
//!
//! 1. Quick rejection for obvious blockers (calls, complex control)
//! 2. Dependence analysis for memory safety
//! 3. Operation-by-operation vectorizability check
//! 4. Access pattern analysis for memory operations

use super::dependence::{Dependence, DependenceGraph, Direction};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{CallKind, ControlOp, MemoryOp, Operator, VectorOp};
use smallvec::SmallVec;

// =============================================================================
// Legality Result
// =============================================================================

/// Result of legality analysis.
#[derive(Debug)]
pub struct LegalityResult {
    /// Whether vectorization is legal.
    pub legal: bool,

    /// Primary reason for illegality (if any).
    pub primary_reason: Option<ViolationKind>,

    /// All violations found.
    pub violations: Vec<LegalityViolation>,

    /// Operations that must remain scalar (cannot vectorize).
    pub scalar_ops: Vec<NodeId>,

    /// Operations that can be vectorized with their target types.
    pub vectorizable_ops: Vec<(NodeId, VectorOp)>,

    /// Required widening operations (scalar -> vector).
    pub widenings: Vec<Widening>,

    /// Maximum legal vector width.
    pub max_legal_width: usize,
}

impl LegalityResult {
    /// Create a result indicating vectorization is legal.
    pub fn legal(max_width: usize) -> Self {
        Self {
            legal: true,
            primary_reason: None,
            violations: Vec::new(),
            scalar_ops: Vec::new(),
            vectorizable_ops: Vec::new(),
            widenings: Vec::new(),
            max_legal_width: max_width,
        }
    }

    /// Create a result indicating vectorization is illegal.
    pub fn illegal(reason: ViolationKind) -> Self {
        Self {
            legal: false,
            primary_reason: Some(reason.clone()),
            violations: vec![LegalityViolation {
                node: NodeId::invalid(),
                kind: reason,
            }],
            scalar_ops: Vec::new(),
            vectorizable_ops: Vec::new(),
            widenings: Vec::new(),
            max_legal_width: 1,
        }
    }

    /// Add a violation.
    pub fn add_violation(&mut self, node: NodeId, kind: ViolationKind) {
        self.violations.push(LegalityViolation { node, kind });
        if self.primary_reason.is_none() {
            self.primary_reason = Some(self.violations.last().unwrap().kind.clone());
        }
        self.legal = false;
    }

    /// Mark an operation as requiring scalar execution.
    pub fn mark_scalar(&mut self, node: NodeId) {
        if !self.scalar_ops.contains(&node) {
            self.scalar_ops.push(node);
        }
    }

    /// Mark an operation as vectorizable.
    pub fn mark_vectorizable(&mut self, node: NodeId, vop: VectorOp) {
        self.vectorizable_ops.push((node, vop));
    }

    /// Add a widening operation.
    pub fn add_widening(&mut self, widening: Widening) {
        self.widenings.push(widening);
    }

    /// Get the percentage of operations that can be vectorized.
    pub fn vectorizable_percentage(&self) -> f32 {
        let total = self.vectorizable_ops.len() + self.scalar_ops.len();
        if total == 0 {
            0.0
        } else {
            self.vectorizable_ops.len() as f32 / total as f32 * 100.0
        }
    }
}

impl Default for LegalityResult {
    fn default() -> Self {
        Self::legal(usize::MAX)
    }
}

// =============================================================================
// Violation Types
// =============================================================================

/// A specific legality violation.
#[derive(Debug, Clone)]
pub struct LegalityViolation {
    /// Node that caused the violation.
    pub node: NodeId,

    /// Type of violation.
    pub kind: ViolationKind,
}

/// Types of legality violations.
#[derive(Debug, Clone)]
pub enum ViolationKind {
    /// Backward loop-carried dependence prevents vectorization.
    BackwardDependence(NodeId, NodeId),

    /// Unknown dependence direction (conservative rejection).
    UnknownDependence(NodeId, NodeId),

    /// Operation cannot be vectorized.
    NonVectorizableOp(Operator),

    /// Memory access pattern is not affine/uniform.
    NonAffineAccess,

    /// Memory access has unknown stride.
    UnknownStride,

    /// Memory access is not contiguous.
    NonContiguousAccess,

    /// Control flow is too complex for vectorization.
    ComplexControlFlow,

    /// Data-dependent loop exit.
    DataDependentExit,

    /// Function call that cannot be vectorized.
    NonVectorizableCall(CallKind),

    /// Exception-throwing operation.
    MayThrow,

    /// Reduction pattern not recognized.
    UnrecognizedReduction,

    /// Induction variable pattern not recognized.
    UnrecognizedInduction,

    /// Loop has multiple exits.
    MultipleExits,

    /// Loop trip count is too small.
    TripCountTooSmall(u64),

    /// Cannot determine loop bounds.
    UnknownBounds,

    /// Vector width exceeds architectural limit.
    WidthExceedsLimit(usize),

    /// Custom reason.
    Custom(String),
}

impl ViolationKind {
    /// Check if this violation is a hard blocker (cannot be worked around).
    pub fn is_hard_blocker(&self) -> bool {
        matches!(
            self,
            ViolationKind::BackwardDependence(_, _)
                | ViolationKind::ComplexControlFlow
                | ViolationKind::MayThrow
                | ViolationKind::MultipleExits
        )
    }

    /// Check if this violation might be resolvable with runtime checks.
    pub fn resolvable_with_runtime_check(&self) -> bool {
        matches!(
            self,
            ViolationKind::UnknownDependence(_, _)
                | ViolationKind::NonContiguousAccess
                | ViolationKind::UnknownBounds
        )
    }

    /// Get a human-readable description.
    pub fn description(&self) -> String {
        match self {
            ViolationKind::BackwardDependence(src, dst) => {
                format!("Backward dependence from {:?} to {:?}", src, dst)
            }
            ViolationKind::UnknownDependence(src, dst) => {
                format!("Unknown dependence between {:?} and {:?}", src, dst)
            }
            ViolationKind::NonVectorizableOp(op) => {
                format!("Non-vectorizable operation: {:?}", op)
            }
            ViolationKind::NonAffineAccess => "Non-affine memory access pattern".to_string(),
            ViolationKind::UnknownStride => "Unknown memory access stride".to_string(),
            ViolationKind::NonContiguousAccess => "Non-contiguous memory access".to_string(),
            ViolationKind::ComplexControlFlow => "Complex control flow in loop".to_string(),
            ViolationKind::DataDependentExit => "Data-dependent loop exit".to_string(),
            ViolationKind::NonVectorizableCall(kind) => {
                format!("Non-vectorizable call: {:?}", kind)
            }
            ViolationKind::MayThrow => "Operation may throw exception".to_string(),
            ViolationKind::UnrecognizedReduction => "Unrecognized reduction pattern".to_string(),
            ViolationKind::UnrecognizedInduction => "Unrecognized induction variable".to_string(),
            ViolationKind::MultipleExits => "Loop has multiple exits".to_string(),
            ViolationKind::TripCountTooSmall(tc) => {
                format!("Trip count {} is too small", tc)
            }
            ViolationKind::UnknownBounds => "Cannot determine loop bounds".to_string(),
            ViolationKind::WidthExceedsLimit(w) => {
                format!("Vector width {} exceeds limit", w)
            }
            ViolationKind::Custom(msg) => msg.clone(),
        }
    }
}

// =============================================================================
// Widening
// =============================================================================

/// A widening operation (scalar to vector conversion).
#[derive(Debug, Clone)]
pub struct Widening {
    /// Scalar node being widened.
    pub scalar: NodeId,

    /// Target vector type.
    pub vector_type: VectorOp,

    /// Kind of widening.
    pub kind: WideningKind,
}

/// Types of widening operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideningKind {
    /// Broadcast scalar to all lanes.
    Broadcast,

    /// Create vector from consecutive scalars (pack).
    Pack,

    /// Splat constant to all lanes.
    Splat,

    /// Load as vector instead of scalar.
    VectorLoad,
}

// =============================================================================
// Legality Analyzer
// =============================================================================

/// Analyzer for vectorization legality.
pub struct LegalityAnalyzer {
    /// Maximum allowed vector width.
    max_width: usize,

    /// Whether to allow gather/scatter for non-contiguous access.
    allow_gather_scatter: bool,

    /// Whether to allow partial vectorization.
    allow_partial: bool,
}

impl LegalityAnalyzer {
    /// Create a new legality analyzer.
    pub fn new(max_width: usize) -> Self {
        Self {
            max_width,
            allow_gather_scatter: false,
            allow_partial: true,
        }
    }

    /// Enable gather/scatter for non-contiguous access.
    pub fn with_gather_scatter(mut self) -> Self {
        self.allow_gather_scatter = true;
        self
    }

    /// Disable partial vectorization.
    pub fn without_partial(mut self) -> Self {
        self.allow_partial = false;
        self
    }

    /// Analyze legality of vectorizing a loop.
    pub fn analyze(
        &self,
        graph: &Graph,
        body_nodes: &[NodeId],
        deps: &DependenceGraph,
    ) -> LegalityResult {
        let mut result = LegalityResult::legal(self.max_width);

        // Phase 1: Check dependences
        self.check_dependences(deps, &mut result);
        if !result.legal && !self.allow_partial {
            return result;
        }

        // Phase 2: Check each operation
        for &node_id in body_nodes {
            self.check_operation(graph, node_id, &mut result);
        }

        // Phase 3: Determine maximum legal width
        result.max_legal_width = result.max_legal_width.min(self.max_width);
        if let Some(dep_width) = self.max_width_from_deps(deps) {
            result.max_legal_width = result.max_legal_width.min(dep_width);
        }

        result
    }

    /// Check dependences for vectorization legality.
    fn check_dependences(&self, deps: &DependenceGraph, result: &mut LegalityResult) {
        for dep in deps.all_dependences() {
            if dep.loop_independent {
                continue; // Loop-independent deps are fine
            }

            // Check innermost loop direction
            match dep.direction_at(0) {
                Direction::Backward => {
                    result.add_violation(
                        dep.src,
                        ViolationKind::BackwardDependence(dep.src, dep.dst),
                    );
                }
                Direction::Unknown => {
                    result
                        .add_violation(dep.src, ViolationKind::UnknownDependence(dep.src, dep.dst));
                }
                Direction::Forward | Direction::Equal => {
                    // OK
                }
            }
        }
    }

    /// Check if an operation is vectorizable.
    fn check_operation(&self, graph: &Graph, node_id: NodeId, result: &mut LegalityResult) {
        let Some(node) = graph.get(node_id) else {
            return;
        };

        match &node.op {
            // Always vectorizable
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone
            | Operator::Parameter(_) => {
                // These become broadcasts or splats
            }

            // Arithmetic - fully vectorizable
            Operator::IntOp(_)
            | Operator::FloatOp(_)
            | Operator::GenericOp(_)
            | Operator::MulHigh
            | Operator::MulHighSigned => {
                result.mark_vectorizable(node_id, VectorOp::V4I64);
            }

            // Comparisons - vectorizable
            Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => {
                result.mark_vectorizable(node_id, VectorOp::V4I64);
            }

            // Bitwise - vectorizable
            Operator::Bitwise(_) | Operator::LogicalNot => {
                result.mark_vectorizable(node_id, VectorOp::V4I64);
            }

            // Memory - need access pattern analysis
            Operator::Memory(MemoryOp::Load)
            | Operator::Memory(MemoryOp::LoadField)
            | Operator::Memory(MemoryOp::LoadElement) => {
                // Assume vectorizable for now, proper analysis in dedicated pass
                result.mark_vectorizable(node_id, VectorOp::V4I64);
            }

            Operator::Memory(MemoryOp::Store)
            | Operator::Memory(MemoryOp::StoreField)
            | Operator::Memory(MemoryOp::StoreElement) => {
                result.mark_vectorizable(node_id, VectorOp::V4I64);
            }

            // SSA nodes - context-dependent
            Operator::Phi | Operator::LoopPhi => {
                // Phis become vector phis
            }

            Operator::Projection(_) => {
                // Projections stay scalar or become vector based on source
            }

            // Control flow - usually blocks vectorization
            Operator::Control(ControlOp::If) => {
                // Could be predicated with masking
                result.mark_scalar(node_id);
            }

            Operator::Control(ControlOp::Return)
            | Operator::Control(ControlOp::Throw)
            | Operator::Control(ControlOp::Deopt) => {
                result.add_violation(node_id, ViolationKind::ComplexControlFlow);
            }

            Operator::Control(ControlOp::Loop)
            | Operator::Control(ControlOp::Region)
            | Operator::Control(ControlOp::Start)
            | Operator::Control(ControlOp::End) => {
                // Loop structure nodes - OK
            }

            // Calls - usually not vectorizable
            Operator::Call(kind) => {
                result.add_violation(node_id, ViolationKind::NonVectorizableCall(*kind));
            }

            // Guards - may block
            Operator::Guard(_) => {
                result.add_violation(node_id, ViolationKind::MayThrow);
            }

            // Container operations - not vectorizable
            Operator::BuildList(_)
            | Operator::BuildTuple(_)
            | Operator::BuildDict(_)
            | Operator::GetIter
            | Operator::IterNext => {
                result.mark_scalar(node_id);
            }

            // Object operations - not vectorizable
            Operator::GetItem
            | Operator::SetItem
            | Operator::GetAttr
            | Operator::SetAttr
            | Operator::Len => {
                result.mark_scalar(node_id);
            }

            // Type operations - not vectorizable
            Operator::TypeCheck | Operator::Box | Operator::Unbox => {
                result.mark_scalar(node_id);
            }

            // Vector operations - already vectorized
            Operator::VectorArith(..)
            | Operator::VectorFma(_)
            | Operator::VectorMemory(..)
            | Operator::VectorBroadcast(_)
            | Operator::VectorExtract(..)
            | Operator::VectorInsert(..)
            | Operator::VectorShuffle(..)
            | Operator::VectorHadd(_)
            | Operator::VectorCmp(..)
            | Operator::VectorBlend(_)
            | Operator::VectorSplat(..) => {
                // Already vector - pass through
            }

            // Memory operations we might not handle
            Operator::Memory(MemoryOp::Alloc) | Operator::Memory(MemoryOp::Free) => {
                result.mark_scalar(node_id);
            }
        }
    }

    /// Get maximum width allowed by dependence distances.
    fn max_width_from_deps(&self, deps: &DependenceGraph) -> Option<usize> {
        if deps.is_vectorizable() {
            Some(deps.max_safe_vector_width())
        } else {
            Some(1) // Not vectorizable
        }
    }

    /// Check if an operator is vectorizable.
    pub fn is_vectorizable_op(op: &Operator) -> bool {
        matches!(
            op,
            Operator::IntOp(_)
                | Operator::FloatOp(_)
                | Operator::GenericOp(_)
                | Operator::IntCmp(_)
                | Operator::FloatCmp(_)
                | Operator::GenericCmp(_)
                | Operator::Bitwise(_)
                | Operator::LogicalNot
                | Operator::MulHigh
                | Operator::MulHighSigned
                | Operator::Memory(MemoryOp::Load)
                | Operator::Memory(MemoryOp::Store)
                | Operator::Memory(MemoryOp::LoadElement)
                | Operator::Memory(MemoryOp::StoreElement)
        )
    }

    /// Check if an operator is inherently scalar (cannot be vectorized).
    pub fn is_inherently_scalar(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Call(_)
                | Operator::Guard(_)
                | Operator::BuildList(_)
                | Operator::BuildTuple(_)
                | Operator::BuildDict(_)
                | Operator::GetIter
                | Operator::IterNext
                | Operator::GetItem
                | Operator::SetItem
                | Operator::GetAttr
                | Operator::SetAttr
                | Operator::TypeCheck
        )
    }
}

impl Default for LegalityAnalyzer {
    fn default() -> Self {
        Self::new(8)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operators::ArithOp;

    // -------------------------------------------------------------------------
    // LegalityResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_legality_result_legal() {
        let result = LegalityResult::legal(8);
        assert!(result.legal);
        assert!(result.primary_reason.is_none());
        assert!(result.violations.is_empty());
        assert_eq!(result.max_legal_width, 8);
    }

    #[test]
    fn test_legality_result_illegal() {
        let result = LegalityResult::illegal(ViolationKind::ComplexControlFlow);
        assert!(!result.legal);
        assert!(result.primary_reason.is_some());
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.max_legal_width, 1);
    }

    #[test]
    fn test_legality_result_add_violation() {
        let mut result = LegalityResult::legal(8);
        assert!(result.legal);

        result.add_violation(NodeId::new(1), ViolationKind::MayThrow);
        assert!(!result.legal);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_legality_result_mark_scalar() {
        let mut result = LegalityResult::legal(8);
        result.mark_scalar(NodeId::new(1));
        result.mark_scalar(NodeId::new(1)); // Duplicate should not add

        assert_eq!(result.scalar_ops.len(), 1);
    }

    #[test]
    fn test_legality_result_mark_vectorizable() {
        let mut result = LegalityResult::legal(8);
        result.mark_vectorizable(NodeId::new(1), VectorOp::V4I64);
        result.mark_vectorizable(NodeId::new(2), VectorOp::V4F64);

        assert_eq!(result.vectorizable_ops.len(), 2);
    }

    #[test]
    fn test_legality_result_vectorizable_percentage() {
        let mut result = LegalityResult::legal(8);

        // 3 vectorizable, 1 scalar = 75%
        result.mark_vectorizable(NodeId::new(1), VectorOp::V4I64);
        result.mark_vectorizable(NodeId::new(2), VectorOp::V4I64);
        result.mark_vectorizable(NodeId::new(3), VectorOp::V4I64);
        result.mark_scalar(NodeId::new(4));

        assert!((result.vectorizable_percentage() - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_legality_result_vectorizable_percentage_empty() {
        let result = LegalityResult::legal(8);
        assert!((result.vectorizable_percentage() - 0.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // ViolationKind Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_violation_kind_is_hard_blocker() {
        assert!(
            ViolationKind::BackwardDependence(NodeId::new(1), NodeId::new(2)).is_hard_blocker()
        );
        assert!(ViolationKind::ComplexControlFlow.is_hard_blocker());
        assert!(ViolationKind::MayThrow.is_hard_blocker());
        assert!(ViolationKind::MultipleExits.is_hard_blocker());

        assert!(!ViolationKind::NonContiguousAccess.is_hard_blocker());
        assert!(!ViolationKind::UnknownBounds.is_hard_blocker());
    }

    #[test]
    fn test_violation_kind_resolvable_with_runtime_check() {
        assert!(
            ViolationKind::UnknownDependence(NodeId::new(1), NodeId::new(2))
                .resolvable_with_runtime_check()
        );
        assert!(ViolationKind::NonContiguousAccess.resolvable_with_runtime_check());
        assert!(ViolationKind::UnknownBounds.resolvable_with_runtime_check());

        assert!(
            !ViolationKind::BackwardDependence(NodeId::new(1), NodeId::new(2))
                .resolvable_with_runtime_check()
        );
        assert!(!ViolationKind::MayThrow.resolvable_with_runtime_check());
    }

    #[test]
    fn test_violation_kind_description() {
        let desc = ViolationKind::ComplexControlFlow.description();
        assert!(desc.contains("Complex control flow"));

        let desc = ViolationKind::TripCountTooSmall(4).description();
        assert!(desc.contains("4"));
    }

    // -------------------------------------------------------------------------
    // Widening Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_widening_kinds() {
        assert_eq!(WideningKind::Broadcast, WideningKind::Broadcast);
        assert_ne!(WideningKind::Broadcast, WideningKind::Pack);
        assert_ne!(WideningKind::Splat, WideningKind::VectorLoad);
    }

    #[test]
    fn test_widening_creation() {
        let widening = Widening {
            scalar: NodeId::new(1),
            vector_type: VectorOp::V4I64,
            kind: WideningKind::Broadcast,
        };

        assert_eq!(widening.scalar, NodeId::new(1));
        assert_eq!(widening.vector_type.lanes, 4);
        assert_eq!(widening.kind, WideningKind::Broadcast);
    }

    // -------------------------------------------------------------------------
    // LegalityAnalyzer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyzer_new() {
        let analyzer = LegalityAnalyzer::new(8);
        assert_eq!(analyzer.max_width, 8);
        assert!(!analyzer.allow_gather_scatter);
        assert!(analyzer.allow_partial);
    }

    #[test]
    fn test_analyzer_with_gather_scatter() {
        let analyzer = LegalityAnalyzer::new(8).with_gather_scatter();
        assert!(analyzer.allow_gather_scatter);
    }

    #[test]
    fn test_analyzer_without_partial() {
        let analyzer = LegalityAnalyzer::new(8).without_partial();
        assert!(!analyzer.allow_partial);
    }

    #[test]
    fn test_is_vectorizable_op() {
        assert!(LegalityAnalyzer::is_vectorizable_op(&Operator::IntOp(
            ArithOp::Add
        )));
        assert!(LegalityAnalyzer::is_vectorizable_op(&Operator::FloatOp(
            ArithOp::Mul
        )));
        assert!(LegalityAnalyzer::is_vectorizable_op(&Operator::Memory(
            MemoryOp::Load
        )));

        assert!(!LegalityAnalyzer::is_vectorizable_op(&Operator::Call(
            CallKind::Direct
        )));
        assert!(!LegalityAnalyzer::is_vectorizable_op(&Operator::GetItem));
    }

    #[test]
    fn test_is_inherently_scalar() {
        assert!(LegalityAnalyzer::is_inherently_scalar(&Operator::Call(
            CallKind::Direct
        )));
        assert!(LegalityAnalyzer::is_inherently_scalar(&Operator::GetItem));
        assert!(LegalityAnalyzer::is_inherently_scalar(&Operator::SetAttr));
        assert!(LegalityAnalyzer::is_inherently_scalar(&Operator::GetIter));

        assert!(!LegalityAnalyzer::is_inherently_scalar(&Operator::IntOp(
            ArithOp::Add
        )));
        assert!(!LegalityAnalyzer::is_inherently_scalar(&Operator::Memory(
            MemoryOp::Load
        )));
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = LegalityAnalyzer::default();
        assert_eq!(analyzer.max_width, 8);
    }

    // -------------------------------------------------------------------------
    // Dependence Checking Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_check_dependences_forward_ok() {
        let analyzer = LegalityAnalyzer::new(8);
        let mut result = LegalityResult::legal(8);

        let mut deps = super::super::dependence::DependenceGraph::new(1);
        let mut dep = super::super::dependence::Dependence::new(
            NodeId::new(1),
            NodeId::new(2),
            super::super::dependence::DependenceKind::RAW,
        );
        dep.set_direction(0, Direction::Forward);
        deps.add_dependence(dep);

        analyzer.check_dependences(&deps, &mut result);
        assert!(result.legal);
    }

    #[test]
    fn test_check_dependences_backward_blocks() {
        let analyzer = LegalityAnalyzer::new(8);
        let mut result = LegalityResult::legal(8);

        let mut deps = super::super::dependence::DependenceGraph::new(1);
        let mut dep = super::super::dependence::Dependence::new(
            NodeId::new(1),
            NodeId::new(2),
            super::super::dependence::DependenceKind::RAW,
        );
        dep.set_direction(0, Direction::Backward);
        deps.add_dependence(dep);

        analyzer.check_dependences(&deps, &mut result);
        assert!(!result.legal);
        assert!(matches!(
            result.primary_reason,
            Some(ViolationKind::BackwardDependence(_, _))
        ));
    }

    #[test]
    fn test_check_dependences_unknown_blocks() {
        let analyzer = LegalityAnalyzer::new(8);
        let mut result = LegalityResult::legal(8);

        let mut deps = super::super::dependence::DependenceGraph::new(1);
        let mut dep = super::super::dependence::Dependence::new(
            NodeId::new(1),
            NodeId::new(2),
            super::super::dependence::DependenceKind::RAW,
        );
        dep.set_direction(0, Direction::Unknown);
        deps.add_dependence(dep);

        analyzer.check_dependences(&deps, &mut result);
        assert!(!result.legal);
        assert!(matches!(
            result.primary_reason,
            Some(ViolationKind::UnknownDependence(_, _))
        ));
    }

    #[test]
    fn test_check_dependences_loop_independent_ok() {
        let analyzer = LegalityAnalyzer::new(8);
        let mut result = LegalityResult::legal(8);

        let mut deps = super::super::dependence::DependenceGraph::new(1);
        let dep = super::super::dependence::Dependence::loop_independent(
            NodeId::new(1),
            NodeId::new(2),
            super::super::dependence::DependenceKind::RAW,
        );
        deps.add_dependence(dep);

        analyzer.check_dependences(&deps, &mut result);
        assert!(result.legal);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_analysis_empty() {
        let analyzer = LegalityAnalyzer::new(8);
        let graph = crate::ir::graph::Graph::new();
        let deps = super::super::dependence::DependenceGraph::new(1);

        let result = analyzer.analyze(&graph, &[], &deps);
        assert!(result.legal);
    }

    #[test]
    fn test_max_width_from_deps() {
        let analyzer = LegalityAnalyzer::new(8);

        let deps = super::super::dependence::DependenceGraph::new(1);
        // Empty deps = fully vectorizable
        let width = analyzer.max_width_from_deps(&deps);
        assert_eq!(width, Some(usize::MAX));
    }
}
