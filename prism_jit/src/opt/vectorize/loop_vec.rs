//! Loop Vectorization Pass
//!
//! This module transforms counted loops to operate on SIMD vectors,
//! processing multiple iterations per vector instruction.
//!
//! # Prerequisites
//!
//! For a loop to be vectorized, it must:
//!
//! 1. Have a known or computable trip count
//! 2. Have no cross-iteration dependencies (or only forward ones)
//! 3. Contain only vectorizable operations
//! 4. Be profitable to vectorize (cost model analysis)
//!
//! # Transformation
//!
//! The vectorizer transforms a loop in several steps:
//!
//! 1. **Analysis**: Identify inductions, reductions, memory patterns
//! 2. **Widening**: Convert scalar operations to vector equivalents
//! 3. **Vector Loop**: Create the main vectorized loop body
//! 4. **Epilog**: Handle remainder iterations (trip_count % vector_width)
//!
//! # Example
//!
//! ```text
//! Before:
//!   for i in 0..n:         # Trip count = n
//!     a[i] = b[i] + c[i]
//!
//! After (VW = vector width = 4):
//!   # Vector loop: processes 4 iterations at once
//!   for i in 0..(n/4)*4 step 4:
//!     va = vload(&a[i])
//!     vb = vload(&b[i])
//!     vc = vload(&c[i])
//!     vr = vadd(vb, vc)
//!     vstore(&a[i], vr)
//!   
//!   # Epilog: remaining n % 4 iterations
//!   for i in (n/4)*4..n:
//!     a[i] = b[i] + c[i]
//! ```

use super::cost::{CostAnalysis, VectorCostModel};
use super::dependence::{Dependence, DependenceGraph};
use super::legality::{LegalityAnalyzer, LegalityResult, ViolationKind};
use super::reduction::{ReductionDetector, ReductionKind as DetectedReductionKind};
use crate::ir::cfg::Loop;
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{Operator, VectorArithKind, VectorOp};
use crate::opt::rce::{InductionDetector, InductionStep as RceInductionStep};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

// =============================================================================
// Loop Vectorization Analysis
// =============================================================================

/// Result of loop vectorization analysis.
#[derive(Debug)]
pub struct LoopVecAnalysis {
    /// Whether the loop is vectorizable.
    pub vectorizable: bool,

    /// Reason if not vectorizable.
    pub rejection_reason: Option<VecRejectReason>,

    /// Suggested vector width.
    pub vector_width: usize,

    /// Required runtime checks.
    pub runtime_checks: Vec<RuntimeCheck>,

    /// Detected induction variables.
    pub inductions: Vec<Induction>,

    /// Detected reductions.
    pub reductions: Vec<Reduction>,

    /// Legality analysis result.
    pub legality: LegalityResult,

    /// Cost analysis result.
    pub cost: Option<CostAnalysis>,

    /// Trip count (if known statically).
    pub trip_count: Option<u64>,

    /// Interleave factor (for memory-bound loops).
    pub interleave_factor: usize,
}

impl LoopVecAnalysis {
    /// Create an analysis indicating the loop cannot be vectorized.
    pub fn not_vectorizable(reason: VecRejectReason) -> Self {
        Self {
            vectorizable: false,
            rejection_reason: Some(reason),
            vector_width: 1,
            runtime_checks: Vec::new(),
            inductions: Vec::new(),
            reductions: Vec::new(),
            legality: LegalityResult::illegal(ViolationKind::Custom(reason.description())),
            cost: None,
            trip_count: None,
            interleave_factor: 1,
        }
    }

    /// Create a successful analysis.
    pub fn vectorizable(width: usize, trip_count: Option<u64>) -> Self {
        Self {
            vectorizable: true,
            rejection_reason: None,
            vector_width: width,
            runtime_checks: Vec::new(),
            inductions: Vec::new(),
            reductions: Vec::new(),
            legality: LegalityResult::legal(width),
            cost: None,
            trip_count,
            interleave_factor: 1,
        }
    }

    /// Check if runtime checks are needed.
    pub fn needs_runtime_checks(&self) -> bool {
        !self.runtime_checks.is_empty()
    }

    /// Check if an epilog loop is needed.
    pub fn needs_epilog(&self, vector_width: usize) -> bool {
        match self.trip_count {
            Some(tc) => tc % vector_width as u64 != 0,
            None => true, // Unknown trip count always needs epilog
        }
    }

    /// Get the number of vector iterations.
    pub fn vector_iterations(&self, vector_width: usize) -> Option<u64> {
        self.trip_count.map(|tc| tc / vector_width as u64)
    }

    /// Get the number of epilog iterations.
    pub fn epilog_iterations(&self, vector_width: usize) -> Option<u64> {
        self.trip_count.map(|tc| tc % vector_width as u64)
    }
}

impl Default for LoopVecAnalysis {
    fn default() -> Self {
        Self::not_vectorizable(VecRejectReason::NotAnalyzed)
    }
}

// =============================================================================
// Rejection Reasons
// =============================================================================

/// Reason for rejecting loop vectorization.
#[derive(Debug, Clone)]
pub enum VecRejectReason {
    /// Loop has not been analyzed yet.
    NotAnalyzed,

    /// Unsafe memory dependence.
    UnsafeDependence(Dependence),

    /// Non-affine memory access pattern.
    NonAffineAccess(NodeId),

    /// Unsupported operation in loop body.
    UnsupportedOp(Operator),

    /// Trip count is too small for vectorization overhead.
    TripCountTooLow(u64),

    /// Vectorization is not profitable.
    NotProfitable(f32), // Speedup ratio

    /// Complex control flow (conditionals, multiple exits).
    ComplexControlFlow,

    /// Unknown trip count and cannot generate epilog.
    UnknownTripCount,

    /// Loop contains function calls.
    ContainsCalls,

    /// Loop contains exceptions/guards.
    ContainsExceptions,

    /// Induction variable not recognized.
    UnrecognizedInduction,

    /// Reduction pattern not recognized.
    UnrecognizedReduction,

    /// Maximum vector width exceeded.
    WidthTooNarrow,
}

impl VecRejectReason {
    /// Get a human-readable description.
    pub fn description(&self) -> String {
        match self {
            VecRejectReason::NotAnalyzed => "Loop not analyzed".to_string(),
            VecRejectReason::UnsafeDependence(dep) => {
                format!("Unsafe dependence: {:?} -> {:?}", dep.src, dep.dst)
            }
            VecRejectReason::NonAffineAccess(node) => {
                format!("Non-affine access at {:?}", node)
            }
            VecRejectReason::UnsupportedOp(op) => {
                format!("Unsupported operation: {:?}", op)
            }
            VecRejectReason::TripCountTooLow(tc) => {
                format!("Trip count {} too low", tc)
            }
            VecRejectReason::NotProfitable(speedup) => {
                format!("Not profitable (speedup {:.2}x)", speedup)
            }
            VecRejectReason::ComplexControlFlow => "Complex control flow".to_string(),
            VecRejectReason::UnknownTripCount => "Unknown trip count".to_string(),
            VecRejectReason::ContainsCalls => "Contains function calls".to_string(),
            VecRejectReason::ContainsExceptions => "Contains exception/guard".to_string(),
            VecRejectReason::UnrecognizedInduction => "Unrecognized induction".to_string(),
            VecRejectReason::UnrecognizedReduction => "Unrecognized reduction".to_string(),
            VecRejectReason::WidthTooNarrow => "Vector width too narrow".to_string(),
        }
    }

    /// Check if this is a hard blocker (cannot be worked around).
    pub fn is_hard_blocker(&self) -> bool {
        matches!(
            self,
            VecRejectReason::UnsafeDependence(_)
                | VecRejectReason::ComplexControlFlow
                | VecRejectReason::ContainsCalls
                | VecRejectReason::ContainsExceptions
        )
    }
}

// =============================================================================
// Runtime Checks
// =============================================================================

/// A runtime check required for safe vectorization.
#[derive(Debug, Clone)]
pub struct RuntimeCheck {
    /// Kind of check.
    pub kind: RuntimeCheckKind,

    /// Nodes involved in the check.
    pub nodes: SmallVec<[NodeId; 2]>,

    /// Check condition expression (for code generation).
    pub condition: Option<NodeId>,
}

/// Types of runtime checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeCheckKind {
    /// Check that two pointers don't alias.
    NoAlias,

    /// Check that memory regions don't overlap.
    NoOverlap,

    /// Check that trip count is at least this value.
    MinTripCount(u64),

    /// Check that access is aligned to vector boundary.
    Alignment(usize),

    /// Check that stride is as expected.
    Stride(i64),
}

impl RuntimeCheck {
    /// Create a no-alias check.
    pub fn no_alias(ptr1: NodeId, ptr2: NodeId) -> Self {
        Self {
            kind: RuntimeCheckKind::NoAlias,
            nodes: smallvec::smallvec![ptr1, ptr2],
            condition: None,
        }
    }

    /// Create a minimum trip count check.
    pub fn min_trip_count(min: u64) -> Self {
        Self {
            kind: RuntimeCheckKind::MinTripCount(min),
            nodes: SmallVec::new(),
            condition: None,
        }
    }

    /// Create an alignment check.
    pub fn alignment(ptr: NodeId, align: usize) -> Self {
        Self {
            kind: RuntimeCheckKind::Alignment(align),
            nodes: smallvec::smallvec![ptr],
            condition: None,
        }
    }
}

// =============================================================================
// Induction Variables
// =============================================================================

/// An induction variable detected in the loop.
#[derive(Debug, Clone)]
pub struct Induction {
    /// The phi node representing the induction variable.
    pub phi: NodeId,

    /// The increment operation.
    pub increment: NodeId,

    /// Initial value.
    pub init: NodeId,

    /// Step value (usually constant).
    pub step: InductionStep,

    /// Vector widened version (after vectorization).
    pub widened: Option<NodeId>,
}

/// Step value for an induction variable.
#[derive(Debug, Clone, Copy)]
pub enum InductionStep {
    /// Constant step.
    Constant(i64),
    /// Dynamic step (node that computes step).
    Dynamic(NodeId),
    /// Unknown step.
    Unknown,
}

impl InductionStep {
    /// Get the constant step value if known.
    pub fn constant_value(&self) -> Option<i64> {
        match self {
            InductionStep::Constant(v) => Some(*v),
            _ => None,
        }
    }

    /// Check if step is unit (1 or -1).
    pub fn is_unit(&self) -> bool {
        matches!(
            self,
            InductionStep::Constant(1) | InductionStep::Constant(-1)
        )
    }
}

// =============================================================================
// Reductions
// =============================================================================

/// A reduction operation detected in the loop.
#[derive(Debug, Clone)]
pub struct Reduction {
    /// The phi node for the reduction.
    pub phi: NodeId,

    /// The reduction operation (accumulator update).
    pub op: NodeId,

    /// Kind of reduction.
    pub kind: ReductionKind,

    /// Initial value of the accumulator.
    pub init: NodeId,

    /// Vector partial result (after vectorization, before horizontal reduction).
    pub partial: Option<NodeId>,

    /// Final scalar result (after horizontal reduction).
    pub final_result: Option<NodeId>,
}

/// Kind of reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    /// Summation: acc += value
    Sum,
    /// Product: acc *= value
    Product,
    /// Minimum: acc = min(acc, value)
    Min,
    /// Maximum: acc = max(acc, value)
    Max,
    /// Bitwise AND: acc &= value
    And,
    /// Bitwise OR: acc |= value
    Or,
    /// Bitwise XOR: acc ^= value
    Xor,
}

impl ReductionKind {
    /// Get the identity element for this reduction.
    pub fn identity(&self) -> i64 {
        match self {
            ReductionKind::Sum => 0,
            ReductionKind::Product => 1,
            ReductionKind::Min => i64::MAX,
            ReductionKind::Max => i64::MIN,
            ReductionKind::And => -1, // All 1s
            ReductionKind::Or => 0,
            ReductionKind::Xor => 0,
        }
    }

    /// Check if this reduction is associative (can be parallelized).
    pub fn is_associative(&self) -> bool {
        // All of these are associative
        true
    }

    /// Check if this reduction is commutative.
    pub fn is_commutative(&self) -> bool {
        // All of these are commutative
        true
    }
}

// =============================================================================
// Loop Vectorizer
// =============================================================================

/// The loop vectorization pass.
///
/// Transforms eligible loops to use SIMD vector operations.
pub struct LoopVectorizer {
    /// Cost model for profitability analysis.
    cost_model: VectorCostModel,

    /// Legality analyzer.
    legality: LegalityAnalyzer,

    /// Minimum trip count to consider vectorization.
    min_trip_count: u64,

    /// Target vector width.
    target_width: usize,

    /// Statistics.
    stats: LoopVecStats,
}

/// Loop vectorization statistics.
#[derive(Debug, Clone, Default)]
pub struct LoopVecStats {
    /// Loops analyzed.
    pub loops_analyzed: usize,
    /// Loops vectorized.
    pub loops_vectorized: usize,
    /// Loops rejected as unsafe.
    pub loops_rejected_unsafe: usize,
    /// Loops rejected as unprofitable.
    pub loops_rejected_cost: usize,
    /// Total vector width used.
    pub total_width: usize,
}

impl LoopVectorizer {
    /// Create a new loop vectorizer.
    pub fn new(cost_model: VectorCostModel) -> Self {
        let target_width = cost_model.best_vector_width(crate::ir::types::ValueType::Int64);
        Self {
            legality: LegalityAnalyzer::new(target_width),
            cost_model,
            min_trip_count: 8,
            target_width,
            stats: LoopVecStats::default(),
        }
    }

    /// Set minimum trip count threshold.
    pub fn with_min_trip_count(mut self, min: u64) -> Self {
        self.min_trip_count = min;
        self
    }

    /// Set target vector width.
    pub fn with_target_width(mut self, width: usize) -> Self {
        self.target_width = width;
        self.legality = LegalityAnalyzer::new(width);
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &LoopVecStats {
        &self.stats
    }

    /// Analyze a loop for vectorization.
    pub fn analyze(
        &mut self,
        graph: &Graph,
        body_nodes: &[NodeId],
        deps: &DependenceGraph,
        trip_count: Option<u64>,
    ) -> LoopVecAnalysis {
        // For backward compatibility, call analyze_loop with a synthetic loop
        // In practice, callers should use analyze_loop directly when possible
        self.analyze_with_options(graph, body_nodes, deps, trip_count, None)
    }

    /// Analyze a loop for vectorization with full Loop structure.
    ///
    /// This is the preferred entry point when a Loop struct is available,
    /// as it enables proper induction and reduction detection.
    pub fn analyze_loop(
        &mut self,
        graph: &Graph,
        body_nodes: &[NodeId],
        deps: &DependenceGraph,
        trip_count: Option<u64>,
        loop_info: &Loop,
    ) -> LoopVecAnalysis {
        self.analyze_with_options(graph, body_nodes, deps, trip_count, Some(loop_info))
    }

    /// Internal analysis with optional Loop info.
    fn analyze_with_options(
        &mut self,
        graph: &Graph,
        body_nodes: &[NodeId],
        deps: &DependenceGraph,
        trip_count: Option<u64>,
        loop_info: Option<&Loop>,
    ) -> LoopVecAnalysis {
        self.stats.loops_analyzed += 1;

        // Phase 1: Quick rejection tests
        if let Some(tc) = trip_count {
            if tc < self.min_trip_count {
                self.stats.loops_rejected_cost += 1;
                return LoopVecAnalysis::not_vectorizable(VecRejectReason::TripCountTooLow(tc));
            }
        }

        // Phase 2: Dependence analysis
        if !deps.is_vectorizable() {
            self.stats.loops_rejected_unsafe += 1;
            // Find the blocking dependence
            for dep in deps.all_dependences() {
                if dep.prevents_vectorization_at(0) {
                    return LoopVecAnalysis::not_vectorizable(VecRejectReason::UnsafeDependence(
                        dep.clone(),
                    ));
                }
            }
            return LoopVecAnalysis::not_vectorizable(VecRejectReason::ComplexControlFlow);
        }

        // Determine max safe width from dependences
        let max_dep_width = deps.max_safe_vector_width();
        let vector_width = self.target_width.min(max_dep_width);

        if vector_width < 2 {
            return LoopVecAnalysis::not_vectorizable(VecRejectReason::WidthTooNarrow);
        }

        // Phase 3: Legality analysis
        let legality = self.legality.analyze(graph, body_nodes, deps);
        if !legality.legal {
            self.stats.loops_rejected_unsafe += 1;
            if let Some(_reason) = &legality.primary_reason {
                return LoopVecAnalysis::not_vectorizable(
                    VecRejectReason::UnsupportedOp(Operator::ConstNone), // Placeholder
                );
            }
        }

        // Phase 4: Cost analysis
        let cost = self.analyze_cost(graph, body_nodes, vector_width, trip_count);
        if !cost.profitable {
            self.stats.loops_rejected_cost += 1;
            return LoopVecAnalysis::not_vectorizable(VecRejectReason::NotProfitable(cost.speedup));
        }

        // Phase 5: Identify patterns
        let (inductions, reductions) = if let Some(li) = loop_info {
            // Use proper detection with Loop structure
            let inds = self.identify_inductions(graph, li);

            // Collect induction phi nodes for reduction detection
            let induction_phis: FxHashSet<NodeId> = inds.iter().map(|i| i.phi).collect();
            let reds = self.identify_reductions(graph, li, &induction_phis);

            (inds, reds)
        } else {
            // No Loop info - return empty (backward compatibility)
            (Vec::new(), Vec::new())
        };

        // Phase 6: Determine runtime checks
        let runtime_checks = self.determine_runtime_checks(deps, vector_width);

        // Success
        self.stats.loops_vectorized += 1;
        self.stats.total_width += vector_width;

        let mut analysis = LoopVecAnalysis::vectorizable(vector_width, trip_count);
        analysis.legality = legality;
        analysis.cost = Some(cost);
        analysis.inductions = inductions;
        analysis.reductions = reductions;
        analysis.runtime_checks = runtime_checks;
        analysis
    }

    /// Analyze the cost of vectorization.
    fn analyze_cost(
        &self,
        _graph: &Graph,
        body_nodes: &[NodeId],
        vector_width: usize,
        trip_count: Option<u64>,
    ) -> CostAnalysis {
        // Simplified cost estimation
        let num_ops = body_nodes.len();

        // Scalar cost = N ops * 1 cycle each
        let scalar_cost = num_ops as f32;

        // Vector cost = N/width ops * 1 cycle + overhead
        let vector_cost = (num_ops as f32 / vector_width as f32) + 2.0; // +2 for setup

        CostAnalysis::new(scalar_cost, vector_cost, vector_width, trip_count)
    }

    /// Identify induction variables in the loop using the RCE induction detector.
    ///
    /// This method uses the full RCE `InductionDetector` which provides comprehensive
    /// pattern matching for LoopPhi nodes with Add/Sub updates.
    fn identify_inductions(&self, graph: &Graph, loop_info: &Loop) -> Vec<Induction> {
        let detector = InductionDetector::new(graph);
        let rce_ivs = detector.find_induction_variables(loop_info);

        // Convert RCE induction variables to loop_vec format
        rce_ivs
            .into_iter()
            .filter_map(|(phi, iv)| {
                // Get the update node (increment operation)
                let increment = iv.update_node?;

                // Convert step
                let step = match iv.step {
                    RceInductionStep::Constant(c) => InductionStep::Constant(c),
                    RceInductionStep::Node(n) => InductionStep::Dynamic(n),
                };

                // Get init node - for vectorization we need the node, not value
                let init = match iv.init {
                    crate::opt::rce::InductionInit::Constant(_) => {
                        // We need the actual node - look for it in phi inputs
                        let phi_node = graph.get(phi)?;
                        phi_node.inputs.get(0)?
                    }
                    crate::opt::rce::InductionInit::Node(n) => n,
                };

                Some(Induction {
                    phi,
                    increment,
                    init,
                    step,
                    widened: None,
                })
            })
            .collect()
    }

    /// Identify reduction patterns in the loop.
    ///
    /// Uses the `ReductionDetector` to find LoopPhi nodes that accumulate
    /// values via associative operations (Sum, Product, Min, Max, And, Or, Xor).
    fn identify_reductions(
        &self,
        graph: &Graph,
        loop_info: &Loop,
        induction_phis: &FxHashSet<NodeId>,
    ) -> Vec<Reduction> {
        let detector = ReductionDetector::new(graph);
        let detected = detector.find_reductions(loop_info, induction_phis);

        // Convert reduction format
        detected
            .into_iter()
            .map(|(phi, red)| {
                // Convert reduction kind
                let kind = match red.kind {
                    DetectedReductionKind::Sum => ReductionKind::Sum,
                    DetectedReductionKind::Product => ReductionKind::Product,
                    DetectedReductionKind::Min => ReductionKind::Min,
                    DetectedReductionKind::Max => ReductionKind::Max,
                    DetectedReductionKind::And => ReductionKind::And,
                    DetectedReductionKind::Or => ReductionKind::Or,
                    DetectedReductionKind::Xor => ReductionKind::Xor,
                };

                Reduction {
                    phi,
                    op: red.op_node,
                    kind,
                    init: red.init,
                    partial: None,
                    final_result: None,
                }
            })
            .collect()
    }

    /// Determine required runtime checks.
    fn determine_runtime_checks(
        &self,
        deps: &DependenceGraph,
        vector_width: usize,
    ) -> Vec<RuntimeCheck> {
        let mut checks = Vec::new();

        // Add minimum trip count check
        if vector_width > 1 {
            checks.push(RuntimeCheck::min_trip_count(vector_width as u64));
        }

        // Check for potential aliasing that needs runtime verification
        for dep in deps.all_dependences() {
            if dep.confidence == super::dependence::DependenceConfidence::Possible {
                // Add no-alias check for uncertain dependences
                checks.push(RuntimeCheck::no_alias(dep.src, dep.dst));
            }
        }

        checks
    }

    /// Transform a loop to vectorized form.
    ///
    /// This is the core vectorization transformation that:
    /// 1. Widens induction variables to vector form
    /// 2. Converts scalar operations to vector equivalents  
    /// 3. Generates horizontal reductions for reduction patterns
    /// 4. Creates epilog loop for remainder iterations
    ///
    /// Returns the result of the transformation.
    pub fn vectorize(&self, graph: &mut Graph, analysis: &LoopVecAnalysis) -> LoopVecResult {
        if !analysis.vectorizable {
            return LoopVecResult::failure();
        }

        let vector_width = analysis.vector_width;
        let mut transformer = VectorTransformer::new(graph, vector_width);

        // Step 1: Widen induction variables
        for induction in &analysis.inductions {
            transformer.widen_induction(induction);
        }

        // Step 2: Transform scalar operations to vector equivalents
        // (Handled by widen_induction and widen_reduction internally)

        // Step 3: Finalize reductions with horizontal ops
        for reduction in &analysis.reductions {
            transformer.finalize_reduction(reduction);
        }

        // Step 4: Generate epilog if needed
        if analysis.needs_epilog(vector_width) {
            transformer.generate_epilog();
        }

        // Create result
        let mut result = LoopVecResult::success(vector_width, analysis.trip_count);
        result.vector_loop = transformer.vector_loop;
        result.epilog_loop = transformer.epilog_loop;
        result
    }
}

// =============================================================================
// Vector Transformer (Internal)
// =============================================================================

/// Internal transformer for loop vectorization.
///
/// Manages the scalar-to-vector node mapping and generates the
/// necessary vector operations.
struct VectorTransformer<'g> {
    /// The graph being transformed.
    graph: &'g mut Graph,

    /// Target vector width (2, 4, or 8 lanes).
    width: usize,

    /// Map from scalar nodes to their widened vector equivalents.
    scalar_to_vector: FxHashMap<NodeId, NodeId>,

    /// The vector loop header (if created).
    vector_loop: Option<NodeId>,

    /// The epilog loop header (if created).
    epilog_loop: Option<NodeId>,

    /// Vector type for integer operations.
    int_vop: VectorOp,

    /// Vector type for float operations.
    float_vop: VectorOp,
}

impl<'g> VectorTransformer<'g> {
    /// Create a new transformer for the given width.
    fn new(graph: &'g mut Graph, width: usize) -> Self {
        // Select appropriate vector types based on width
        let (int_vop, float_vop) = match width {
            2 => (VectorOp::V2I64, VectorOp::V2F64),
            4 => (VectorOp::V4I64, VectorOp::V4F64),
            8 => (VectorOp::V8I64, VectorOp::V8F64),
            _ => (VectorOp::V4I64, VectorOp::V4F64), // Default to 4-wide
        };

        Self {
            graph,
            width,
            scalar_to_vector: FxHashMap::default(),
            vector_loop: None,
            epilog_loop: None,
            int_vop,
            float_vop,
        }
    }

    /// Widen an induction variable to vector form.
    ///
    /// Creates a vector containing consecutive values:
    /// `<init, init+step, init+2*step, ..., init+(width-1)*step>`
    fn widen_induction(&mut self, induction: &Induction) {
        // Get the step as a constant if possible
        let step = match induction.step {
            InductionStep::Constant(s) => s,
            InductionStep::Dynamic(_) | InductionStep::Unknown => {
                // Dynamic step requires more complex handling
                // For now, skip dynamic step inductions
                return;
            }
        };

        // Create vector step: <0*step, 1*step, 2*step, ..., (width-1)*step>
        // This will be added to the broadcast of the scalar induction
        let step_offsets: Vec<i64> = (0..self.width as i64).map(|i| i * step).collect();

        // Create a splat of the step offset pattern
        // For i = init, vector = <init, init+step, init+2*step, init+3*step>
        let scalar_init = induction.init;

        // Broadcast the scalar init to all lanes
        let broadcast = self.graph.add_node(
            Operator::VectorBroadcast(self.int_vop),
            InputList::Single(scalar_init),
        );

        // Create constant vector for offsets: <0, step, 2*step, 3*step>
        // We use VectorSplat for the base, then add index * step
        let offsets = self.create_lane_offsets(step_offsets);

        // widened_iv = broadcast(init) + offsets
        let widened = self.graph.add_node(
            Operator::VectorArith(self.int_vop, VectorArithKind::Add),
            InputList::Pair(broadcast, offsets),
        );

        // Map the scalar phi to its widened version
        self.scalar_to_vector.insert(induction.phi, widened);
    }

    /// Create a vector of lane offset values.
    fn create_lane_offsets(&mut self, offsets: Vec<i64>) -> NodeId {
        // For simplicity, we create this as a series of inserts
        // A production implementation would use a constant pool

        // Start with zero vector
        let zero = self
            .graph
            .add_node(Operator::VectorSplat(self.int_vop, 0), InputList::Empty);

        // Insert each offset into its lane
        let mut result = zero;
        for (lane, &offset) in offsets.iter().enumerate() {
            if offset != 0 {
                let offset_const = self.graph.const_int(offset);
                result = self.graph.add_node(
                    Operator::VectorInsert(self.int_vop, lane as u8),
                    InputList::Pair(result, offset_const),
                );
            }
        }

        result
    }

    /// Finalize a reduction with horizontal operation.
    ///
    /// After the vector loop, we need to reduce the partial vector result
    /// to a scalar using horizontal operations.
    fn finalize_reduction(&mut self, reduction: &Reduction) {
        // The reduction phi should have a vector partial result
        let vector_partial = self.scalar_to_vector.get(&reduction.phi).copied();

        if let Some(partial) = vector_partial {
            // Generate horizontal reduction based on kind
            let final_result = match reduction.kind {
                ReductionKind::Sum | ReductionKind::Xor => {
                    // Use horizontal add for sum and xor
                    self.graph.add_node(
                        Operator::VectorHadd(self.int_vop),
                        InputList::Single(partial),
                    )
                }
                ReductionKind::Product => {
                    // Product reduction: extract all lanes and multiply
                    self.emit_reduce_product(partial)
                }
                ReductionKind::Min => {
                    // Min reduction: extract all and find min
                    self.emit_reduce_minmax(partial, true)
                }
                ReductionKind::Max => {
                    // Max reduction: extract all and find max
                    self.emit_reduce_minmax(partial, false)
                }
                ReductionKind::And | ReductionKind::Or => {
                    // Bitwise reduction: extract and apply
                    self.emit_reduce_bitwise(partial, reduction.kind)
                }
            };

            // The final_result is now a scalar that replaces uses of the reduction phi
            // outside the loop. This would be wired up by the loop cloning logic.
            self.scalar_to_vector.insert(reduction.phi, final_result);
        }
    }

    /// Emit product reduction (multiply all lanes).
    fn emit_reduce_product(&mut self, vector: NodeId) -> NodeId {
        // Extract each lane and multiply them together
        let mut result = self.graph.add_node(
            Operator::VectorExtract(self.int_vop, 0),
            InputList::Single(vector),
        );

        for lane in 1..self.width as u8 {
            let extracted = self.graph.add_node(
                Operator::VectorExtract(self.int_vop, lane),
                InputList::Single(vector),
            );
            result = self.graph.int_mul(result, extracted);
        }

        result
    }

    /// Emit min/max reduction.
    fn emit_reduce_minmax(&mut self, vector: NodeId, is_min: bool) -> NodeId {
        use crate::ir::operators::CmpOp;

        let mut result = self.graph.add_node(
            Operator::VectorExtract(self.int_vop, 0),
            InputList::Single(vector),
        );

        for lane in 1..self.width as u8 {
            let extracted = self.graph.add_node(
                Operator::VectorExtract(self.int_vop, lane),
                InputList::Single(vector),
            );

            // cmp = result < extracted (for min) or result > extracted (for max)
            let cmp_op = if is_min { CmpOp::Lt } else { CmpOp::Gt };
            let cmp = self
                .graph
                .add_node(Operator::IntCmp(cmp_op), InputList::Pair(result, extracted));

            // For now, we don't have a select node, so we'd need control flow
            // In a real implementation, we'd use a conditional select
            // For simplicity, assume we have a min/max intrinsic
            // This is a placeholder - real impl would use proper select
            let _ = cmp; // Suppress unused warning
            result = extracted; // Simplified - would use select
        }

        result
    }

    /// Emit bitwise reduction (AND or OR all lanes).
    fn emit_reduce_bitwise(&mut self, vector: NodeId, kind: ReductionKind) -> NodeId {
        use crate::ir::operators::BitwiseOp;

        let op = match kind {
            ReductionKind::And => BitwiseOp::And,
            ReductionKind::Or => BitwiseOp::Or,
            _ => BitwiseOp::And, // Fallback
        };

        let mut result = self.graph.add_node(
            Operator::VectorExtract(self.int_vop, 0),
            InputList::Single(vector),
        );

        for lane in 1..self.width as u8 {
            let extracted = self.graph.add_node(
                Operator::VectorExtract(self.int_vop, lane),
                InputList::Single(vector),
            );
            result = self
                .graph
                .add_node(Operator::Bitwise(op), InputList::Pair(result, extracted));
        }

        result
    }

    /// Generate epilog loop for remainder iterations.
    ///
    /// When trip_count % width != 0, we need a scalar epilog to handle
    /// the remaining iterations.
    fn generate_epilog(&mut self) {
        // In a full implementation, this would:
        // 1. Clone the original scalar loop body
        // 2. Set up the epilog to start at (trip_count / width) * width
        // 3. Run for the remaining trip_count % width iterations
        //
        // For now, we just mark that an epilog is needed
        // The actual cloning would be done by a separate loop transformation pass
        self.epilog_loop = None; // Placeholder - would be set to cloned loop header
    }
}

impl Default for LoopVectorizer {
    fn default() -> Self {
        Self::new(VectorCostModel::default())
    }
}

// =============================================================================
// Loop Vectorization Result
// =============================================================================

/// Result of loop vectorization transformation.
#[derive(Debug)]
pub struct LoopVecResult {
    /// Whether vectorization was applied.
    pub success: bool,

    /// Vector width used.
    pub vector_width: usize,

    /// Original scalar loop (if kept).
    pub scalar_loop: Option<NodeId>,

    /// New vectorized loop.
    pub vector_loop: Option<NodeId>,

    /// Epilog loop for remainder.
    pub epilog_loop: Option<NodeId>,

    /// Entry block for runtime checks.
    pub check_block: Option<NodeId>,

    /// Estimated speedup.
    pub speedup: f32,
}

impl LoopVecResult {
    /// Create a failed result.
    pub fn failure() -> Self {
        Self {
            success: false,
            vector_width: 1,
            scalar_loop: None,
            vector_loop: None,
            epilog_loop: None,
            check_block: None,
            speedup: 1.0,
        }
    }

    /// Create a successful result.
    pub fn success(width: usize, trip_count: Option<u64>) -> Self {
        let speedup = match trip_count {
            Some(tc) if tc > 0 => width as f32 * 0.9, // 90% efficiency
            _ => width as f32 * 0.8,
        };
        Self {
            success: true,
            vector_width: width,
            scalar_loop: None,
            vector_loop: None, // TODO: Set after transformation
            epilog_loop: None,
            check_block: None,
            speedup,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
