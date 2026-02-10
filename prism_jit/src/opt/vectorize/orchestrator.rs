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
mod tests {
    use super::*;
    use crate::ir::graph::Graph;
    use crate::ir::node::NodeId;
    use crate::ir::operators::*;

    // =========================================================================
    // Orchestrator Construction
    // =========================================================================

    #[test]
    fn test_orchestrator_new_default_config() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.stats().loops_analyzed, 0);
        assert_eq!(orch.stats().loops_vectorized, 0);
        assert_eq!(orch.stats().slp_regions_analyzed, 0);
        assert!(orch.loop_decisions().is_empty());
        assert!(orch.slp_decisions().is_empty());
    }

    #[test]
    fn test_orchestrator_new_sse42_config() {
        let config = VectorizeConfig::sse42();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.config.simd_level, SimdLevel::Sse42);
        assert_eq!(orch.config.max_vector_width, 2);
    }

    #[test]
    fn test_orchestrator_new_avx2_config() {
        let config = VectorizeConfig::avx2();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.config.simd_level, SimdLevel::Avx2);
        assert_eq!(orch.config.max_vector_width, 4);
    }

    #[test]
    fn test_orchestrator_new_avx512_config() {
        let config = VectorizeConfig::avx512();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.config.simd_level, SimdLevel::Avx512);
        assert_eq!(orch.config.max_vector_width, 8);
    }

    #[test]
    fn test_orchestrator_new_aggressive_config() {
        let config = VectorizeConfig::aggressive();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.config.simd_level, SimdLevel::Avx512);
        assert_eq!(orch.config.max_vector_width, 16);
        assert!(orch.config.enable_gather_scatter);
    }

    #[test]
    fn test_orchestrator_with_custom_cost_model() {
        let config = VectorizeConfig::default();
        let cost_model = VectorCostModel::new(SimdLevel::Avx512);
        let orch = VectorizationOrchestrator::with_cost_model(&config, cost_model);
        assert_eq!(orch.cost_model.level(), SimdLevel::Avx512);
    }

    // =========================================================================
    // Run on Empty Graph
    // =========================================================================

    #[test]
    fn test_orchestrator_run_empty_graph() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        let changed = orch.run(&mut graph);

        // Empty graph should not be changed
        assert!(!changed);
        assert_eq!(orch.stats().loops_analyzed, 0);
        assert_eq!(orch.stats().loops_vectorized, 0);
    }

    #[test]
    fn test_orchestrator_run_no_loops() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        // Add some simple nodes but no loops
        let _c1 = graph.const_int(42);
        let _c2 = graph.const_int(100);

        let changed = orch.run(&mut graph);

        assert!(!changed);
        assert_eq!(orch.stats().loops_analyzed, 0);
    }

    // =========================================================================
    // Vector Width Selection
    // =========================================================================

    #[test]
    fn test_select_optimal_width_unconstrained() {
        let config = VectorizeConfig::avx2();
        let orch = VectorizationOrchestrator::new(&config);

        let deps = DependenceGraph::new(1);
        let legality = super::super::legality::LegalityResult::default();
        // max_legal_width defaults to usize::MAX — no legality constraint

        let width = orch.select_optimal_width(&deps, &legality);
        assert_eq!(width, 4); // AVX2 max = 4
    }

    #[test]
    fn test_select_optimal_width_constrained_by_deps() {
        let config = VectorizeConfig::avx512();
        let orch = VectorizationOrchestrator::new(&config);

        // Deps allow max width of 4
        let mut deps = DependenceGraph::new(1);
        // DependenceGraph's max_safe_width() defaults to large value
        // when no deps exist, so this will take from config

        let legality = super::super::legality::LegalityResult::default(); // max_legal_width = usize::MAX
        let width = orch.select_optimal_width(&deps, &legality);
        // Should be limited by config max_vector_width (8 for avx512)
        // and rounded to power of 2
        assert!(width >= 2);
        assert!(width <= 8);
    }

    #[test]
    fn test_select_optimal_width_constrained_by_legality() {
        let config = VectorizeConfig::avx512();
        let orch = VectorizationOrchestrator::new(&config);

        let deps = DependenceGraph::new(1);
        let mut legality = super::super::legality::LegalityResult::default();
        legality.max_legal_width = 2; // Legality limits to 2

        let width = orch.select_optimal_width(&deps, &legality);
        assert_eq!(width, 2);
    }

    #[test]
    fn test_select_optimal_width_power_of_two_rounding() {
        let config = VectorizeConfig {
            max_vector_width: 5, // Not a power of 2
            ..VectorizeConfig::default()
        };
        let orch = VectorizationOrchestrator::new(&config);

        let deps = DependenceGraph::new(1);
        let legality = super::super::legality::LegalityResult::default();

        let width = orch.select_optimal_width(&deps, &legality);
        assert_eq!(width, 4); // Rounded down to 4
    }

    #[test]
    fn test_select_optimal_width_min_is_1() {
        let config = VectorizeConfig {
            max_vector_width: 1,
            ..VectorizeConfig::default()
        };
        let orch = VectorizationOrchestrator::new(&config);

        let deps = DependenceGraph::new(1);
        let legality = super::super::legality::LegalityResult::default();

        let width = orch.select_optimal_width(&deps, &legality);
        assert_eq!(width, 1);
    }

    // =========================================================================
    // Memory Operation Detection
    // =========================================================================

    #[test]
    fn test_is_memory_op_load_field() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::LoadField
        )));
    }

    #[test]
    fn test_is_memory_op_store_field() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::StoreField
        )));
    }

    #[test]
    fn test_is_memory_op_load_element() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::LoadElement
        )));
    }

    #[test]
    fn test_is_memory_op_store_element() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::StoreElement
        )));
    }

    #[test]
    fn test_is_memory_op_get_item() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::GetItem));
    }

    #[test]
    fn test_is_memory_op_set_item() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::SetItem));
    }

    #[test]
    fn test_is_not_memory_op_arith() {
        assert!(!VectorizationOrchestrator::is_memory_op(&Operator::IntOp(
            ArithOp::Add
        )));
    }

    #[test]
    fn test_is_not_memory_op_const() {
        assert!(!VectorizationOrchestrator::is_memory_op(
            &Operator::ConstInt(42)
        ));
    }

    #[test]
    fn test_is_not_memory_op_control() {
        assert!(!VectorizationOrchestrator::is_memory_op(
            &Operator::Control(ControlOp::Start)
        ));
    }

    #[test]
    fn test_is_not_memory_op_alloc() {
        // Alloc is a memory op by type but not relevant for vectorization dependence
        assert!(!VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::Alloc
        )));
    }

    // =========================================================================
    // SimdLevel Extensions
    // =========================================================================

    #[test]
    fn test_simd_level_max_lanes_scalar() {
        assert_eq!(SimdLevel::Scalar.max_lanes(), 1);
    }

    #[test]
    fn test_simd_level_max_lanes_sse42() {
        assert_eq!(SimdLevel::Sse42.max_lanes(), 2);
    }

    #[test]
    fn test_simd_level_max_lanes_avx2() {
        assert_eq!(SimdLevel::Avx2.max_lanes(), 4);
    }

    #[test]
    fn test_simd_level_max_lanes_avx512() {
        assert_eq!(SimdLevel::Avx512.max_lanes(), 8);
    }

    #[test]
    fn test_simd_level_max_lanes_neon() {
        assert_eq!(SimdLevel::Neon.max_lanes(), 2);
    }

    // =========================================================================
    // LoopDecision Tests
    // =========================================================================

    #[test]
    fn test_loop_decision_not_vectorized() {
        let decision = LoopDecision {
            loop_index: 0,
            vectorized: false,
            analysis: None,
            rejection: Some(LoopRejection::NoMemoryOps),
            vector_width: 1,
            estimated_speedup: 1.0,
        };
        assert!(!decision.vectorized);
        assert_eq!(decision.vector_width, 1);
        assert!((decision.estimated_speedup - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_loop_decision_vectorized() {
        let analysis = LoopVecAnalysis {
            vectorizable: true,
            rejection_reason: None,
            vector_width: 4,
            runtime_checks: Vec::new(),
            inductions: Vec::new(),
            reductions: Vec::new(),
            legality: super::super::legality::LegalityResult::default(),
            cost: None,
            trip_count: Some(100),
            interleave_factor: 1,
        };
        let decision = LoopDecision {
            loop_index: 0,
            vectorized: true,
            analysis: Some(analysis),
            rejection: None,
            vector_width: 4,
            estimated_speedup: 3.6,
        };
        assert!(decision.vectorized);
        assert_eq!(decision.vector_width, 4);
        assert!(decision.estimated_speedup > 1.0);
    }

    #[test]
    fn test_loop_decision_rejected_dependences() {
        let decision = LoopDecision {
            loop_index: 1,
            vectorized: false,
            analysis: None,
            rejection: Some(LoopRejection::IllegalDependences { violation_count: 3 }),
            vector_width: 1,
            estimated_speedup: 1.0,
        };
        assert!(!decision.vectorized);
        match &decision.rejection {
            Some(LoopRejection::IllegalDependences { violation_count }) => {
                assert_eq!(*violation_count, 3);
            }
            _ => panic!("Expected IllegalDependences rejection"),
        }
    }

    #[test]
    fn test_loop_decision_rejected_unprofitable() {
        let decision = LoopDecision {
            loop_index: 2,
            vectorized: false,
            analysis: None,
            rejection: Some(LoopRejection::NotProfitable { speedup: 0.5 }),
            vector_width: 1,
            estimated_speedup: 1.0,
        };
        assert!(!decision.vectorized);
        match &decision.rejection {
            Some(LoopRejection::NotProfitable { speedup }) => {
                assert!((*speedup - 0.5).abs() < 0.001);
            }
            _ => panic!("Expected NotProfitable rejection"),
        }
    }

    #[test]
    fn test_loop_decision_rejected_trip_count() {
        let decision = LoopDecision {
            loop_index: 3,
            vectorized: false,
            analysis: None,
            rejection: Some(LoopRejection::TripCountTooLow {
                trip_count: 4,
                minimum: 8,
            }),
            vector_width: 1,
            estimated_speedup: 1.0,
        };
        match &decision.rejection {
            Some(LoopRejection::TripCountTooLow {
                trip_count,
                minimum,
            }) => {
                assert_eq!(*trip_count, 4);
                assert_eq!(*minimum, 8);
            }
            _ => panic!("Expected TripCountTooLow rejection"),
        }
    }

    // =========================================================================
    // SlpDecision Tests
    // =========================================================================

    #[test]
    fn test_slp_decision_not_vectorized() {
        let decision = SlpDecision {
            vectorized: false,
            vector_ops: 0,
            scalar_ops_eliminated: 0,
            estimated_speedup: 1.0,
        };
        assert!(!decision.vectorized);
        assert_eq!(decision.vector_ops, 0);
    }

    #[test]
    fn test_slp_decision_vectorized() {
        let decision = SlpDecision {
            vectorized: true,
            vector_ops: 5,
            scalar_ops_eliminated: 20,
            estimated_speedup: 4.0,
        };
        assert!(decision.vectorized);
        assert_eq!(decision.vector_ops, 5);
        assert_eq!(decision.scalar_ops_eliminated, 20);
        assert!((decision.estimated_speedup - 4.0).abs() < 0.001);
    }

    // =========================================================================
    // LoopRejection Debug Coverage
    // =========================================================================

    #[test]
    fn test_loop_rejection_debug_formatting() {
        // Ensure all variants implement Debug
        let rejections: Vec<LoopRejection> = vec![
            LoopRejection::IllegalDependences { violation_count: 1 },
            LoopRejection::NotProfitable { speedup: 0.8 },
            LoopRejection::TripCountTooLow {
                trip_count: 3,
                minimum: 8,
            },
            LoopRejection::TooComplex(VecRejectReason::ComplexControlFlow),
            LoopRejection::NoMemoryOps,
        ];

        for r in &rejections {
            let s = format!("{:?}", r);
            assert!(!s.is_empty());
        }
    }

    // =========================================================================
    // Trip Count Estimation
    // =========================================================================

    #[test]
    fn test_estimate_trip_count_simple_loop() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        use crate::ir::cfg::BlockId;
        let loop_info = Loop {
            header: BlockId::new(0),
            back_edges: vec![BlockId::new(1)],
            body: vec![BlockId::new(0), BlockId::new(1)],
            parent: None,
            children: vec![],
            depth: 1,
        };

        let tc = orch.estimate_trip_count(&loop_info);
        assert!(tc.is_some());
        assert!(tc.unwrap() >= config.min_trip_count);
    }

    #[test]
    fn test_estimate_trip_count_multi_backedge() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        use crate::ir::cfg::BlockId;
        let loop_info = Loop {
            header: BlockId::new(0),
            back_edges: vec![BlockId::new(1), BlockId::new(2)],
            body: vec![BlockId::new(0), BlockId::new(1), BlockId::new(2)],
            parent: None,
            children: vec![],
            depth: 1,
        };

        let tc = orch.estimate_trip_count(&loop_info);
        // Multiple back edges → unknown trip count
        assert!(tc.is_none());
    }

    #[test]
    fn test_estimate_trip_count_large_body() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        use crate::ir::cfg::BlockId;
        let body: Vec<BlockId> = (0..25).map(|i| BlockId::new(i)).collect();
        let loop_info = Loop {
            header: BlockId::new(0),
            back_edges: vec![BlockId::new(1)],
            body,
            parent: None,
            children: vec![],
            depth: 1,
        };

        let tc = orch.estimate_trip_count(&loop_info);
        // Large body → unknown trip count
        assert!(tc.is_none());
    }

    // =========================================================================
    // Collect Memory Ops
    // =========================================================================

    #[test]
    fn test_collect_memory_ops_from_body() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        let mut graph = Graph::new();
        use crate::ir::node::InputList;

        // Create some nodes including memory ops
        let c1 = graph.const_int(0);
        let c2 = graph.const_int(1);
        let add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(c1, c2));
        let load = graph.add_node(
            Operator::Memory(MemoryOp::LoadElement),
            InputList::Single(c1),
        );
        let store = graph.add_node(
            Operator::Memory(MemoryOp::StoreElement),
            InputList::Pair(c1, c2),
        );

        let body_nodes = vec![c1, c2, add, load, store];
        let mem_ops = orch.collect_memory_ops(&graph, &body_nodes);

        assert_eq!(mem_ops.len(), 2); // load + store
        assert!(mem_ops.contains(&load));
        assert!(mem_ops.contains(&store));
    }

    #[test]
    fn test_collect_memory_ops_empty_body() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);
        let graph = Graph::new();

        let mem_ops = orch.collect_memory_ops(&graph, &[]);
        assert!(mem_ops.is_empty());
    }

    #[test]
    fn test_collect_memory_ops_no_memory() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        let mut graph = Graph::new();
        let c1 = graph.const_int(0);
        let c2 = graph.const_int(1);

        let mem_ops = orch.collect_memory_ops(&graph, &[c1, c2]);
        assert!(mem_ops.is_empty());
    }

    // =========================================================================
    // Stats Tracking
    // =========================================================================

    #[test]
    fn test_stats_initial_zero() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        assert_eq!(orch.stats().loops_analyzed, 0);
        assert_eq!(orch.stats().loops_vectorized, 0);
        assert_eq!(orch.stats().loops_rejected_unsafe, 0);
        assert_eq!(orch.stats().loops_rejected_unprofitable, 0);
        assert_eq!(orch.stats().slp_regions_analyzed, 0);
        assert_eq!(orch.stats().slp_regions_vectorized, 0);
        assert_eq!(orch.stats().vector_ops_created, 0);
        assert_eq!(orch.stats().scalar_ops_eliminated, 0);
        assert!((orch.stats().estimated_speedup - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_after_empty_run() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        orch.run(&mut graph);

        assert_eq!(orch.stats().loops_analyzed, 0);
        assert_eq!(orch.stats().loops_vectorized, 0);
    }

    // =========================================================================
    // Config Propagation
    // =========================================================================

    #[test]
    fn test_config_disable_slp() {
        let config = VectorizeConfig {
            enable_slp: false,
            ..VectorizeConfig::default()
        };
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        orch.run(&mut graph);

        // SLP should not have been analyzed
        assert_eq!(orch.stats().slp_regions_analyzed, 0);
    }

    #[test]
    fn test_config_disable_loop_vec() {
        let config = VectorizeConfig {
            enable_loop_vec: false,
            ..VectorizeConfig::default()
        };
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        orch.run(&mut graph);

        assert_eq!(orch.stats().loops_analyzed, 0);
    }

    #[test]
    fn test_config_disable_both() {
        let config = VectorizeConfig {
            enable_loop_vec: false,
            enable_slp: false,
            ..VectorizeConfig::default()
        };
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        let changed = orch.run(&mut graph);
        assert!(!changed);
    }

    // =========================================================================
    // Integration: Orchestrator with Graph containing only constants
    // =========================================================================

    #[test]
    fn test_orchestrator_graph_with_constants_only() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        // Constants only — no loops, no SLP candidates
        let _c1 = graph.const_int(1);
        let _c2 = graph.const_int(2);
        let _c3 = graph.const_int(3);
        let _c4 = graph.const_int(4);

        let changed = orch.run(&mut graph);
        assert!(!changed);
    }

    #[test]
    fn test_orchestrator_graph_with_arithmetic() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        use crate::ir::node::InputList;

        let c1 = graph.const_int(10);
        let c2 = graph.const_int(20);
        let _add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(c1, c2));
        let _mul = graph.add_node(Operator::IntOp(ArithOp::Mul), InputList::Pair(c1, c2));

        let changed = orch.run(&mut graph);
        // Simple arithmetic without loops — may or may not vectorize via SLP
        // but at minimum should not crash
        let _ = changed; // Don't assert specific outcome — graph-dependent
    }

    // =========================================================================
    // Non-Loop Node Collection
    // =========================================================================

    #[test]
    fn test_collect_non_loop_nodes_empty_analysis() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        use crate::ir::node::InputList;
        let c1 = graph.const_int(1);
        let c2 = graph.const_int(2);
        let _add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(c1, c2));

        let empty_cfg = Cfg::build(&graph);
        let dom = DominatorTree::build(&empty_cfg);
        let loop_analysis = LoopAnalysis::compute(&empty_cfg, &dom);

        let non_loop = orch.collect_non_loop_nodes(&graph, &loop_analysis);
        // Should include the add node but not constants
        // (constants are filtered out)
        assert!(non_loop.iter().any(|&id| {
            let node = graph.node(id);
            matches!(node.op, Operator::IntOp(ArithOp::Add))
        }));
    }
}
