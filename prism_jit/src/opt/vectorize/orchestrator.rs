//! Vectorization Orchestrator
//!
//! Wires the complete vectorization pipeline, coordinating:
//! 1. **CFG/Loop Discovery**: Build CFG, dominator tree, detect natural loops
//! 2. **Loop Vectorization**: For each loop — dependence analysis → legality check
//!    → cost analysis → transformation
//! 3. **SLP Vectorization**: For straight-line code outside loops
//!
//! This is the central coordinator that drives `Vectorize::run()`.

use crate::ir::cfg::{Cfg, DominatorTree, Loop, LoopAnalysis};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator};

use super::cost::VectorCostModel;
use super::dependence::DependenceGraph;
use super::legality::LegalityAnalyzer;
use super::loop_vec::{LoopVecAnalysis, LoopVectorizer, VecRejectReason};
use super::slp::{SlpResult, SlpVectorizer};
use super::{SimdLevel, VectorizeConfig, VectorizeStats};

// =============================================================================
// Orchestration Decision
// =============================================================================

/// Decision for a single loop from the orchestration pipeline.
#[derive(Debug)]
pub struct LoopDecision {
    /// Index of the loop in the loop analysis.
    pub loop_index: usize,
    /// Whether the loop was vectorized.
    pub vectorized: bool,
    /// The analysis result if vectorizable.
    pub analysis: Option<LoopVecAnalysis>,
    /// Rejection reason if not vectorized.
    pub rejection: Option<LoopRejection>,
    /// Chosen vector width (1 if not vectorized).
    pub vector_width: usize,
    /// Estimated speedup factor.
    pub estimated_speedup: f32,
}

/// Detailed reason why a loop was rejected.
#[derive(Debug)]
pub enum LoopRejection {
    /// Loop has unsafe dependences.
    IllegalDependences { violation_count: usize },
    /// Vectorization is legal but not profitable.
    NotProfitable { speedup: f32 },
    /// Trip count is below minimum threshold.
    TripCountTooLow { trip_count: u64, minimum: u64 },
    /// The loop body is too complex (calls, I/O, etc.).
    TooComplex(VecRejectReason),
    /// No memory operations to vectorize.
    NoMemoryOps,
}

/// Summary of SLP vectorization for a region.
#[derive(Debug)]
pub struct SlpDecision {
    /// Whether SLP produced any transformations.
    pub vectorized: bool,
    /// Number of vector operations created.
    pub vector_ops: usize,
    /// Number of scalar operations eliminated.
    pub scalar_ops_eliminated: usize,
    /// Estimated speedup.
    pub estimated_speedup: f32,
}

// =============================================================================
// Vectorization Orchestrator
// =============================================================================

/// Central orchestrator for the vectorization pipeline.
///
/// Coordinates CFG analysis, loop discovery, dependence checking,
/// legality analysis, cost modeling, and transformation application.
/// This is the "brain" that drives `Vectorize::run()`.
pub struct VectorizationOrchestrator {
    /// Configuration governing the pass.
    config: VectorizeConfig,
    /// Cost model for the target SIMD level.
    cost_model: VectorCostModel,
    /// Accumulated statistics.
    stats: VectorizeStats,
    /// Decisions made for each loop.
    loop_decisions: Vec<LoopDecision>,
    /// SLP decisions for straight-line regions.
    slp_decisions: Vec<SlpDecision>,
}

impl VectorizationOrchestrator {
    /// Create new orchestrator with the given configuration.
    pub fn new(config: &VectorizeConfig) -> Self {
        Self {
            config: config.clone(),
            cost_model: VectorCostModel::new(config.simd_level),
            stats: VectorizeStats::default(),
            loop_decisions: Vec::new(),
            slp_decisions: Vec::new(),
        }
    }

    /// Create orchestrator with explicit cost model (for testing).
    pub fn with_cost_model(config: &VectorizeConfig, cost_model: VectorCostModel) -> Self {
        Self {
            config: config.clone(),
            cost_model,
            stats: VectorizeStats::default(),
            loop_decisions: Vec::new(),
            slp_decisions: Vec::new(),
        }
    }

    /// Get accumulated statistics.
    pub fn stats(&self) -> &VectorizeStats {
        &self.stats
    }

    /// Get loop decisions.
    pub fn loop_decisions(&self) -> &[LoopDecision] {
        &self.loop_decisions
    }

    /// Get SLP decisions.
    pub fn slp_decisions(&self) -> &[SlpDecision] {
        &self.slp_decisions
    }

    // =========================================================================
    // Main Entry Point
    // =========================================================================

    /// Run the full vectorization pipeline on a graph.
    ///
    /// Returns `true` if any transformations were applied.
    pub fn run(&mut self, graph: &mut Graph) -> bool {
        let mut changed = false;

        // Phase 1: Build CFG infrastructure
        let cfg = Cfg::build(graph);
        if cfg.is_empty() {
            return false;
        }

        let dom = DominatorTree::build(&cfg);
        let loop_analysis = LoopAnalysis::compute(&cfg, &dom);

        // Phase 2: Loop vectorization (innermost-first for maximal benefit)
        if self.config.enable_loop_vec {
            changed |= self.orchestrate_loops(graph, &cfg, &loop_analysis);
        }

        // Phase 3: SLP vectorization on remaining straight-line code
        if self.config.enable_slp {
            changed |= self.orchestrate_slp(graph, &cfg, &loop_analysis);
        }

        changed
    }

    // =========================================================================
    // Loop Vectorization Pipeline
    // =========================================================================

    /// Orchestrate vectorization for all discovered loops.
    ///
    /// Processes loops innermost-first (deepest nesting first) so that
    /// inner loops are vectorized before outer loops are considered.
    fn orchestrate_loops(
        &mut self,
        graph: &mut Graph,
        _cfg: &Cfg,
        loop_analysis: &LoopAnalysis,
    ) -> bool {
        if loop_analysis.loops.is_empty() {
            return false;
        }

        let mut changed = false;

        // Sort loops by depth (deepest first) for innermost-first ordering
        let mut loop_indices: Vec<usize> = (0..loop_analysis.loops.len()).collect();
        loop_indices.sort_by(|&a, &b| {
            loop_analysis.loops[b]
                .depth
                .cmp(&loop_analysis.loops[a].depth)
        });

        for &loop_idx in &loop_indices {
            let loop_info = &loop_analysis.loops[loop_idx];
            let decision = self.orchestrate_single_loop(graph, loop_info, loop_idx);

            if decision.vectorized {
                changed = true;
                self.stats.loops_vectorized += 1;
                self.stats.estimated_speedup += decision.estimated_speedup;
            }

            self.stats.loops_analyzed += 1;
            self.loop_decisions.push(decision);
        }

        changed
    }

    /// Full pipeline for a single loop:
    /// 1. Collect body nodes and memory operations
    /// 2. Build dependence graph
    /// 3. Check legality
    /// 4. Select optimal vector width
    /// 5. Analyze profitability
    /// 6. Apply transformation
    fn orchestrate_single_loop(
        &mut self,
        graph: &mut Graph,
        loop_info: &Loop,
        loop_index: usize,
    ) -> LoopDecision {
        // Step 1: Collect loop body nodes
        let body_nodes = self.collect_body_nodes(graph, loop_info);
        if body_nodes.is_empty() {
            return LoopDecision {
                loop_index,
                vectorized: false,
                analysis: None,
                rejection: Some(LoopRejection::TooComplex(
                    VecRejectReason::ComplexControlFlow,
                )),
                vector_width: 1,
                estimated_speedup: 1.0,
            };
        }

        // Step 2: Collect memory operations for dependence analysis
        let memory_ops = self.collect_memory_ops(graph, &body_nodes);

        // Step 3: Build dependence graph
        let depth = loop_info.depth as usize;
        let deps = if memory_ops.is_empty() {
            DependenceGraph::new(depth)
        } else {
            DependenceGraph::compute(graph, &memory_ops, depth)
        };

        // Step 4: Check legality
        let mut legality_analyzer = LegalityAnalyzer::new(self.config.max_vector_width);
        if self.config.enable_gather_scatter {
            legality_analyzer = legality_analyzer.with_gather_scatter();
        }

        let legality = legality_analyzer.analyze(graph, &body_nodes, &deps);
        if !legality.legal && !legality.violations.is_empty() {
            let has_hard_blocker = legality.violations.iter().any(|v| v.kind.is_hard_blocker());

            if has_hard_blocker {
                self.stats.loops_rejected_unsafe += 1;
                return LoopDecision {
                    loop_index,
                    vectorized: false,
                    analysis: None,
                    rejection: Some(LoopRejection::IllegalDependences {
                        violation_count: legality.violations.len(),
                    }),
                    vector_width: 1,
                    estimated_speedup: 1.0,
                };
            }
        }

        // Step 5: Select optimal vector width
        let optimal_width = self.select_optimal_width(&deps, &legality);

        // Step 6: Analyze the loop with the vectorizer
        let mut vectorizer = LoopVectorizer::new(self.cost_model.clone())
            .with_min_trip_count(self.config.min_trip_count)
            .with_target_width(optimal_width);

        let trip_count = self.estimate_trip_count(loop_info);
        let analysis = vectorizer.analyze_loop(graph, &body_nodes, &deps, trip_count, loop_info);

        if !analysis.vectorizable {
            let rejection = match &analysis.rejection_reason {
                Some(VecRejectReason::TripCountTooLow(tc)) => LoopRejection::TripCountTooLow {
                    trip_count: *tc,
                    minimum: self.config.min_trip_count,
                },
                Some(VecRejectReason::NotProfitable(speedup)) => {
                    self.stats.loops_rejected_unprofitable += 1;
                    LoopRejection::NotProfitable { speedup: *speedup }
                }
                Some(reason) => LoopRejection::TooComplex(reason.clone()),
                None => LoopRejection::TooComplex(VecRejectReason::ComplexControlFlow),
            };

            return LoopDecision {
                loop_index,
                vectorized: false,
                analysis: Some(analysis),
                rejection: Some(rejection),
                vector_width: 1,
                estimated_speedup: 1.0,
            };
        }

        // Step 7: Apply the vectorization transformation
        let result = vectorizer.vectorize(graph, &analysis);

        let speedup = result.speedup;
        let vec_width = result.vector_width;

        self.stats.vector_ops_created += vec_width; // Approximate
        self.stats.scalar_ops_eliminated += body_nodes.len();

        LoopDecision {
            loop_index,
            vectorized: result.success,
            analysis: Some(analysis),
            rejection: None,
            vector_width: vec_width,
            estimated_speedup: speedup,
        }
    }

    // =========================================================================
    // SLP Vectorization Pipeline
    // =========================================================================

    /// Orchestrate SLP vectorization on straight-line code outside loops.
    fn orchestrate_slp(
        &mut self,
        graph: &mut Graph,
        _cfg: &Cfg,
        loop_analysis: &LoopAnalysis,
    ) -> bool {
        let mut changed = false;

        // Collect nodes NOT in any loop — these are candidates for SLP
        let non_loop_nodes = self.collect_non_loop_nodes(graph, loop_analysis);
        if non_loop_nodes.is_empty() {
            return false;
        }

        self.stats.slp_regions_analyzed += 1;

        // Determine SLP vector width from SIMD level
        let slp_width = std::cmp::min(
            self.config.max_vector_width,
            self.cost_model.level().max_lanes(),
        );

        if slp_width < 2 {
            return false;
        }

        // Create and run SLP vectorizer
        let mut slp = SlpVectorizer::new(graph, &self.cost_model, slp_width);
        let seeds = slp.find_seeds(&non_loop_nodes);

        if seeds.is_empty() {
            self.slp_decisions.push(SlpDecision {
                vectorized: false,
                vector_ops: 0,
                scalar_ops_eliminated: 0,
                estimated_speedup: 1.0,
            });
            return false;
        }

        let built = slp.build_tree(&seeds);
        if !built {
            self.slp_decisions.push(SlpDecision {
                vectorized: false,
                vector_ops: 0,
                scalar_ops_eliminated: 0,
                estimated_speedup: 1.0,
            });
            return false;
        }

        // Profitability check is done inside build_tree; check results
        let tree = slp.tree();
        let vector_ops = tree.vector_ops_count();
        let scalar_ops = tree.scalar_ops_eliminated();

        if vector_ops > 0 && scalar_ops > vector_ops {
            changed = true;
            self.stats.slp_regions_vectorized += 1;
            self.stats.vector_ops_created += vector_ops;
            self.stats.scalar_ops_eliminated += scalar_ops;

            let speedup = scalar_ops as f32 / vector_ops as f32;
            self.stats.estimated_speedup += speedup;

            self.slp_decisions.push(SlpDecision {
                vectorized: true,
                vector_ops,
                scalar_ops_eliminated: scalar_ops,
                estimated_speedup: speedup,
            });
        } else {
            self.slp_decisions.push(SlpDecision {
                vectorized: false,
                vector_ops,
                scalar_ops_eliminated: scalar_ops,
                estimated_speedup: 1.0,
            });
        }

        changed
    }

    // =========================================================================
    // Helper: Node Collection
    // =========================================================================

    /// Collect all nodes in a loop body by walking the body blocks.
    fn collect_body_nodes(&self, graph: &Graph, loop_info: &Loop) -> Vec<NodeId> {
        let mut body_nodes = Vec::new();

        // Walk all nodes in the graph and collect those whose
        // region belongs to a loop body block
        for (id, node) in graph.iter() {
            // Check if this node's control input maps to a loop body block
            // For Sea-of-Nodes, we check if the node is transitively controlled
            // by the loop header
            if self.node_in_loop(graph, id, loop_info) {
                // Skip control-flow nodes themselves (Region, Loop, If, etc.)
                if !node.op.is_pure() || matches!(node.op, Operator::Memory(_)) {
                    body_nodes.push(id);
                }
                // Also include pure computational nodes
                if node.op.is_pure()
                    && !matches!(
                        node.op,
                        Operator::ConstInt(_)
                            | Operator::ConstFloat(_)
                            | Operator::ConstBool(_)
                            | Operator::ConstNone
                            | Operator::Parameter(_)
                    )
                {
                    body_nodes.push(id);
                }
            }
        }

        body_nodes.sort();
        body_nodes.dedup();
        body_nodes
    }

    /// Check if a node belongs to a loop by checking if any loop body block's
    /// region is an ancestor of the node in the control dependency DAG.
    fn node_in_loop(&self, graph: &Graph, node_id: NodeId, loop_info: &Loop) -> bool {
        // Simple heuristic: check if the node's inputs include any node
        // that is the loop header or a block in the loop body
        let node = graph.node(node_id);

        // Check if this node IS a loop body block
        for &body_block in &loop_info.body {
            // body_block is a BlockId, but we compare with NodeId
            // In the CFG, blocks map to region NodeIds
            // We check if this node depends on nodes within the loop
            let block_node_id = NodeId::new(body_block.as_usize() as u32);
            if node_id == block_node_id {
                return true;
            }
        }

        // Check if any input is in the loop (transitive, but limited depth)
        for &input in node.inputs.as_slice() {
            for &body_block in &loop_info.body {
                let block_node_id = NodeId::new(body_block.as_usize() as u32);
                if input == block_node_id {
                    return true;
                }
            }
        }

        false
    }

    /// Collect memory operations from a set of body nodes.
    fn collect_memory_ops(&self, graph: &Graph, body_nodes: &[NodeId]) -> Vec<NodeId> {
        body_nodes
            .iter()
            .filter(|&&id| {
                let node = graph.node(id);
                Self::is_memory_op(&node.op)
            })
            .copied()
            .collect()
    }

    /// Check if an operator is a memory operation relevant to dependence analysis.
    fn is_memory_op(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(MemoryOp::LoadField)
                | Operator::Memory(MemoryOp::StoreField)
                | Operator::Memory(MemoryOp::LoadElement)
                | Operator::Memory(MemoryOp::StoreElement)
                | Operator::VectorMemory(..)
                | Operator::GetItem
                | Operator::SetItem
        )
    }

    /// Collect nodes that are NOT inside any loop.
    fn collect_non_loop_nodes(&self, graph: &Graph, loop_analysis: &LoopAnalysis) -> Vec<NodeId> {
        let mut nodes = Vec::new();

        for (id, node) in graph.iter() {
            // Skip constants and parameters — not interesting for SLP
            if matches!(
                node.op,
                Operator::ConstInt(_)
                    | Operator::ConstFloat(_)
                    | Operator::ConstBool(_)
                    | Operator::ConstNone
                    | Operator::Parameter(_)
            ) {
                continue;
            }

            // Skip control flow nodes
            if matches!(node.op, Operator::Control(_)) {
                continue;
            }

            // Check: is this node NOT in any loop?
            let in_loop = loop_analysis.loops.iter().any(|l| {
                l.body.iter().any(|&b| {
                    let block_node_id = NodeId::new(b.as_usize() as u32);
                    id == block_node_id
                })
            });

            if !in_loop {
                nodes.push(id);
            }
        }

        nodes
    }

    // =========================================================================
    // Helper: Vector Width Selection
    // =========================================================================

    /// Select the optimal vector width based on target capabilities,
    /// dependence constraints, and legality restrictions.
    fn select_optimal_width(
        &self,
        deps: &DependenceGraph,
        legality: &super::legality::LegalityResult,
    ) -> usize {
        // Start from max target width
        let max_target = self.config.max_vector_width;

        // Constrain by dependence distance analysis
        let max_from_deps = deps.max_safe_vector_width();

        // Constrain by legality's max vectorizable width
        let max_from_legality = legality.max_legal_width;

        // Take the minimum of all constraints
        let width = max_target.min(max_from_deps).min(max_from_legality);

        // Clamp to powers of 2 (round down)
        if width >= 8 {
            8
        } else if width >= 4 {
            4
        } else if width >= 2 {
            2
        } else {
            1
        }
    }

    // =========================================================================
    // Helper: Trip Count Estimation
    // =========================================================================

    /// Estimate the trip count for a loop.
    ///
    /// Uses back-edge count as a proxy. In a production JIT, this would
    /// come from PGO data or symbolic analysis.
    fn estimate_trip_count(&self, loop_info: &Loop) -> Option<u64> {
        // Heuristic: if the loop has a single back edge and a body of
        // reasonable size, estimate based on nesting depth
        if loop_info.back_edges.len() == 1 && loop_info.body.len() <= 20 {
            // Conservative estimate — if we don't know, assume enough iterations
            // to be worth vectorizing (slightly above the minimum)
            Some(self.config.min_trip_count * 4)
        } else {
            None
        }
    }
}

// =============================================================================
// SimdLevel Extension
// =============================================================================

impl SimdLevel {
    /// Maximum number of 64-bit lanes for this SIMD level.
    pub fn max_lanes(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse42 => 2,
            SimdLevel::Avx2 => 4,
            SimdLevel::Avx512 => 8,
            SimdLevel::Neon => 2,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
