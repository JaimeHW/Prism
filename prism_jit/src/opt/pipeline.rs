//! Optimization Pipeline.
//!
//! Orchestrates multiple optimization passes with proper ordering,
//! fixed-point iteration, and comprehensive statistics.
//!
//! # Pass Phases
//!
//! 1. **Canonicalization**: Simplify, constant folding
//! 2. **ProfileGuided**: Branch probability, hot/cold splitting
//! 3. **Local**: GVN, copy propagation
//! 4. **Loop**: LICM, loop unrolling
//! 5. **Interprocedural**: Inlining, escape analysis
//! 6. **Cleanup**: DCE, CFG simplification
//!
//! # Fixed-Point Iteration
//!
//! The pipeline runs passes until no changes occur or a maximum
//! iteration count is reached. Some passes trigger re-running
//! of earlier passes (e.g., inlining enables more GVN).

use super::OptimizationPass;
use super::branch_probability::BranchProbabilityPass;
use super::copy_prop::CopyProp;
use super::dce::Dce;
use super::dse::Dse;
use super::escape::Escape;
use super::gvn::Gvn;
use super::hot_cold::HotColdPass;
use super::inline::Inline;
use super::instcombine::InstCombine;
use super::licm::Licm;
use super::pre::Pre;
use super::rce::RangeCheckElimination;
use super::sccp::Sccp;
use super::simplify::Simplify;
use super::strength_reduce::StrengthReduce;
use super::tailcall::TailCallOpt;
use super::unroll::Unroll;
use crate::ir::graph::Graph;

use std::time::{Duration, Instant};

// =============================================================================
// Pass Phase
// =============================================================================

/// Phase of the optimization pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassPhase {
    /// Early passes: canonicalization, constant folding.
    Canonicalization,
    /// Profile-guided optimizations: branch probability, hot/cold splitting.
    ProfileGuided,
    /// Local optimizations: GVN, copy propagation.
    Local,
    /// Loop optimizations: LICM, unrolling.
    Loop,
    /// Interprocedural: inlining, escape analysis.
    Interprocedural,
    /// Cleanup passes: DCE, CFG simplification.
    Cleanup,
}

// =============================================================================
// Pass Entry
// =============================================================================

/// A registered pass in the pipeline.
struct PassEntry {
    /// The pass (boxed for polymorphism).
    pass: Box<dyn OptimizationPass>,
    /// Which phase this pass belongs to.
    phase: PassPhase,
    /// Whether this pass is enabled.
    enabled: bool,
    /// Pass-specific statistics.
    runs: usize,
    changes: usize,
    time: Duration,
}

impl PassEntry {
    fn new<P: OptimizationPass + 'static>(pass: P, phase: PassPhase) -> Self {
        Self {
            pass: Box::new(pass),
            phase,
            enabled: true,
            runs: 0,
            changes: 0,
            time: Duration::ZERO,
        }
    }
}

// =============================================================================
// Pipeline Configuration
// =============================================================================

/// Configuration for the optimization pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum iterations per phase.
    pub max_iterations_per_phase: usize,

    /// Maximum total iterations across all phases.
    pub max_total_iterations: usize,

    // =========================================================================
    // Canonicalization Phase
    // =========================================================================
    /// Enable simplification (constant folding, algebraic identities).
    pub enable_simplify: bool,

    /// Enable Sparse Conditional Constant Propagation.
    pub enable_sccp: bool,

    /// Enable Instruction Combining (peephole optimization).
    pub enable_instcombine: bool,

    // =========================================================================
    // ProfileGuided Phase
    // =========================================================================
    /// Enable branch probability annotation (PGO).
    pub enable_branch_probability: bool,

    /// Enable hot/cold code splitting (PGO).
    pub enable_hot_cold: bool,

    // =========================================================================
    // Local Phase
    // =========================================================================
    /// Enable Copy Propagation.
    pub enable_copy_prop: bool,

    /// Enable Global Value Numbering.
    pub enable_gvn: bool,

    /// Enable Dead Store Elimination.
    pub enable_dse: bool,

    /// Enable Partial Redundancy Elimination.
    pub enable_pre: bool,

    /// Enable Strength Reduction (magic number division, etc.).
    pub enable_strength_reduce: bool,

    // =========================================================================
    // Loop Phase
    // =========================================================================
    /// Enable Loop Invariant Code Motion.
    pub enable_licm: bool,

    /// Enable Loop Unrolling.
    pub enable_unroll: bool,

    /// Enable Range Check Elimination.
    pub enable_rce: bool,

    // =========================================================================
    // Interprocedural Phase
    // =========================================================================
    /// Enable function inlining.
    pub enable_inline: bool,

    /// Enable escape analysis for stack allocation.
    pub enable_escape: bool,

    /// Enable Tail Call Optimization.
    pub enable_tco: bool,

    // =========================================================================
    // Cleanup Phase
    // =========================================================================
    /// Enable Dead Code Elimination.
    pub enable_dce: bool,

    // =========================================================================
    // Diagnostics
    // =========================================================================
    /// Collect timing statistics.
    pub collect_timing: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_iterations_per_phase: 5,
            max_total_iterations: 20,
            // Canonicalization
            enable_simplify: true,
            enable_sccp: true,
            enable_instcombine: true,
            // ProfileGuided
            enable_branch_probability: true,
            enable_hot_cold: true,
            // Local
            enable_copy_prop: true,
            enable_gvn: true,
            enable_dse: true,
            enable_pre: true,
            enable_strength_reduce: true,
            // Loop
            enable_licm: true,
            enable_unroll: true,
            enable_rce: true,
            // Interprocedural
            enable_inline: true,
            enable_escape: true,
            enable_tco: true,
            // Cleanup
            enable_dce: true,
            // Diagnostics
            collect_timing: true,
        }
    }
}

impl PipelineConfig {
    /// Create a minimal configuration (fewer passes for faster compile).
    /// This is suitable for Tier-1 JIT compilation where compile time is critical.
    pub fn minimal() -> Self {
        Self {
            max_iterations_per_phase: 2,
            max_total_iterations: 8,
            // Canonicalization - keep cheap passes
            enable_simplify: true,
            enable_sccp: false,       // Skip expensive dataflow
            enable_instcombine: true, // Cheap and effective
            // ProfileGuided - skip PGO in Tier-1 for compile speed
            enable_branch_probability: false,
            enable_hot_cold: false,
            // Local - only essential
            enable_copy_prop: true, // Cheap and improves code quality
            enable_gvn: true,       // Essential for code quality
            enable_dse: false,      // Skip for faster compile
            enable_pre: false,      // Skip expensive LCM
            enable_strength_reduce: false,
            // Loop - skip all
            enable_licm: false,
            enable_unroll: false,
            enable_rce: false,
            // Interprocedural - skip all
            enable_inline: false,
            enable_escape: false,
            enable_tco: false,
            // Cleanup - always run
            enable_dce: true,
            // Diagnostics
            collect_timing: false,
        }
    }

    /// Create a full optimization configuration.
    pub fn full() -> Self {
        Self {
            max_iterations_per_phase: 10,
            max_total_iterations: 50,
            ..Default::default()
        }
    }
}

// =============================================================================
// Optimization Pipeline
// =============================================================================

/// The main optimization pipeline.
pub struct OptPipeline {
    /// Configuration.
    config: PipelineConfig,

    /// Registered passes in order.
    passes: Vec<PassEntry>,

    /// Total iterations run.
    total_iterations: usize,

    /// Total time spent.
    total_time: Duration,
}

impl OptPipeline {
    /// Create a new pipeline with default configuration.
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    /// Create a pipeline with custom configuration.
    pub fn with_config(config: PipelineConfig) -> Self {
        let mut pipeline = Self {
            config,
            passes: Vec::new(),
            total_iterations: 0,
            total_time: Duration::ZERO,
        };

        pipeline.register_default_passes();
        pipeline
    }

    /// Create a pipeline with profile data for PGO-guided optimization.
    ///
    /// This is the primary entry point for Tier 2 compilation with PGO.
    /// Profile data is injected into the `BranchProbabilityPass` so that
    /// measured branch frequencies override static heuristics.
    pub fn with_profile(
        config: PipelineConfig,
        profile: crate::runtime::profile_data::ProfileData,
    ) -> Self {
        let mut pipeline = Self::with_config(config);
        pipeline.inject_profile(profile);
        pipeline
    }

    /// Inject profile data into the `BranchProbabilityPass`.
    ///
    /// Scans the registered passes and injects the profile data into
    /// any `BranchProbabilityPass` found. This allows downstream passes
    /// (e.g., `HotColdPass`) to consume PGO-annotated probabilities.
    pub fn inject_profile(&mut self, profile: crate::runtime::profile_data::ProfileData) {
        for entry in &mut self.passes {
            if entry.pass.name() == "BranchProbability" {
                // Replace the existing BranchProbabilityPass with one carrying profile data
                entry.pass = Box::new(BranchProbabilityPass::with_profile(profile));
                return;
            }
        }

        // If branch probability was disabled, register it now with profile data
        self.passes.push(PassEntry::new(
            BranchProbabilityPass::with_profile(profile),
            PassPhase::ProfileGuided,
        ));
    }

    /// Register the default set of optimization passes.
    ///
    /// Pass ordering is critical for effectiveness:
    /// - SCCP runs early to propagate constants before other passes
    /// - Copy propagation runs before GVN to maximize redundancy detection
    /// - GVN runs before DSE/PRE to normalize expressions
    /// - DSE runs before PRE (dead stores don't need redundancy elimination)
    /// - Strength reduce runs after GVN when patterns are normalized
    /// - LICM runs before unrolling to hoist invariants first
    /// - Unrolling runs before RCE to expose range check patterns
    /// - Inlining runs early in interprocedural to expose more optimization
    /// - TCO runs after inlining (inlined calls can become tail calls)
    /// - DCE runs last to clean up dead code from all passes
    fn register_default_passes(&mut self) {
        // =====================================================================
        // Canonicalization phase - normalize and simplify the IR
        // =====================================================================

        // SCCP first: propagates constants and eliminates unreachable code
        if self.config.enable_sccp {
            self.register(Sccp::new(), PassPhase::Canonicalization);
        }

        // Simplify: algebraic identities, strength reduction, etc.
        if self.config.enable_simplify {
            self.register(Simplify::new(), PassPhase::Canonicalization);
        }

        // InstCombine: peephole optimizations on instruction sequences
        if self.config.enable_instcombine {
            self.register(InstCombine::new(), PassPhase::Canonicalization);
        }

        // =====================================================================
        // ProfileGuided phase - PGO-driven optimizations
        // =====================================================================

        // Branch probability: annotate branches with measured/estimated weights
        if self.config.enable_branch_probability {
            self.register(BranchProbabilityPass::new(), PassPhase::ProfileGuided);
        }

        // Hot/cold splitting: partition code by execution temperature
        if self.config.enable_hot_cold {
            self.register(HotColdPass::new(), PassPhase::ProfileGuided);
        }

        // =====================================================================
        // Local phase - single basic block optimizations
        // =====================================================================

        // Copy propagation first: simplifies use-def chains for subsequent passes
        if self.config.enable_copy_prop {
            self.register(CopyProp::new(), PassPhase::Local);
        }

        // GVN: eliminate redundant computations
        if self.config.enable_gvn {
            self.register(Gvn::new(), PassPhase::Local);
        }

        // DSE: eliminate stores that are never read
        if self.config.enable_dse {
            self.register(Dse::new(), PassPhase::Local);
        }

        // PRE: eliminate partially redundant expressions via code motion
        if self.config.enable_pre {
            self.register(Pre::new(), PassPhase::Local);
        }

        // Strength reduction: convert expensive ops to cheaper sequences
        if self.config.enable_strength_reduce {
            self.register(StrengthReduce::new(), PassPhase::Local);
        }

        // =====================================================================
        // Loop phase - loop-level optimizations
        // =====================================================================

        // LICM first: hoist invariants before unrolling
        if self.config.enable_licm {
            self.register(Licm::new(), PassPhase::Loop);
        }

        // Unrolling: replicate loop bodies to reduce overhead
        if self.config.enable_unroll {
            self.register(Unroll::new(), PassPhase::Loop);
        }

        // RCE: eliminate array bounds checks after unrolling exposes patterns
        if self.config.enable_rce {
            self.register(RangeCheckElimination::new(), PassPhase::Loop);
        }

        // =====================================================================
        // Interprocedural phase - cross-function optimizations
        // =====================================================================

        // Inlining first: exposes more optimization opportunities
        if self.config.enable_inline {
            self.register(Inline::new(), PassPhase::Interprocedural);
        }

        // Escape analysis: identify objects that don't escape for stack allocation
        if self.config.enable_escape {
            self.register(Escape::new(), PassPhase::Interprocedural);
        }

        // TCO: convert eligible tail calls to jumps
        if self.config.enable_tco {
            self.register(TailCallOpt::new(), PassPhase::Interprocedural);
        }

        // =====================================================================
        // Cleanup phase - final cleanup
        // =====================================================================

        // DCE: remove dead code generated by other passes
        if self.config.enable_dce {
            self.register(Dce::new(), PassPhase::Cleanup);
        }
    }

    /// Register a custom pass.
    pub fn register<P: OptimizationPass + 'static>(&mut self, pass: P, phase: PassPhase) {
        self.passes.push(PassEntry::new(pass, phase));
    }

    /// Run the optimization pipeline on a graph.
    pub fn run(&mut self, graph: &mut Graph) -> PipelineStats {
        let start = Instant::now();
        let initial_size = graph.len();

        let mut stats = PipelineStats::default();
        let mut iterations = 0;

        // Run phases in order
        for phase in &[
            PassPhase::Canonicalization,
            PassPhase::ProfileGuided,
            PassPhase::Local,
            PassPhase::Loop,
            PassPhase::Interprocedural,
            PassPhase::Cleanup,
        ] {
            let phase_changed = self.run_phase(graph, *phase, &mut stats);
            iterations += 1;

            if iterations >= self.config.max_total_iterations {
                break;
            }
        }

        self.total_iterations = iterations;
        self.total_time = start.elapsed();

        stats.total_iterations = iterations;
        stats.total_time = self.total_time;
        stats.initial_size = initial_size;
        stats.final_size = graph.len();

        stats
    }

    /// Run all passes in a specific phase.
    fn run_phase(
        &mut self,
        graph: &mut Graph,
        phase: PassPhase,
        stats: &mut PipelineStats,
    ) -> bool {
        let mut phase_changed = false;

        for iteration in 0..self.config.max_iterations_per_phase {
            let mut iter_changed = false;

            for entry in &mut self.passes {
                if entry.phase != phase || !entry.enabled {
                    continue;
                }

                let start = if self.config.collect_timing {
                    Some(Instant::now())
                } else {
                    None
                };

                let changed = entry.pass.run(graph);

                if let Some(start) = start {
                    entry.time += start.elapsed();
                }

                entry.runs += 1;
                if changed {
                    entry.changes += 1;
                    iter_changed = true;
                }
            }

            if iter_changed {
                phase_changed = true;
            } else {
                // Fixed point reached for this phase
                break;
            }
        }

        stats.phases_run += 1;
        phase_changed
    }

    /// Get pass statistics.
    pub fn pass_stats(&self) -> Vec<PassStat> {
        self.passes
            .iter()
            .map(|e| PassStat {
                name: e.pass.name().to_string(),
                phase: e.phase,
                runs: e.runs,
                changes: e.changes,
                time: e.time,
            })
            .collect()
    }

    /// Get total iterations run.
    #[inline]
    pub fn iterations(&self) -> usize {
        self.total_iterations
    }

    /// Get total time spent.
    #[inline]
    pub fn total_time(&self) -> Duration {
        self.total_time
    }
}

impl Default for OptPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics from a single pass.
#[derive(Debug, Clone)]
pub struct PassStat {
    /// Pass name.
    pub name: String,
    /// Pass phase.
    pub phase: PassPhase,
    /// Number of times run.
    pub runs: usize,
    /// Number of times it made changes.
    pub changes: usize,
    /// Total time spent in this pass.
    pub time: Duration,
}

/// Statistics from the entire pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total iterations across all phases.
    pub total_iterations: usize,
    /// Number of phases run.
    pub phases_run: usize,
    /// Total time spent.
    pub total_time: Duration,
    /// Initial graph size.
    pub initial_size: usize,
    /// Final graph size.
    pub final_size: usize,
}

impl PipelineStats {
    /// Get size reduction ratio.
    pub fn size_reduction(&self) -> f64 {
        if self.initial_size == 0 {
            1.0
        } else {
            self.final_size as f64 / self.initial_size as f64
        }
    }
}

// =============================================================================
// Quick Optimize Functions
// =============================================================================

/// Run full optimization pipeline on a graph.
pub fn optimize_full(graph: &mut Graph) -> PipelineStats {
    let mut pipeline = OptPipeline::with_config(PipelineConfig::full());
    pipeline.run(graph)
}

/// Run minimal optimization pipeline on a graph.
pub fn optimize_minimal(graph: &mut Graph) -> PipelineStats {
    let mut pipeline = OptPipeline::with_config(PipelineConfig::minimal());
    pipeline.run(graph)
}

/// Run default optimization pipeline on a graph.
pub fn optimize(graph: &mut Graph) -> PipelineStats {
    let mut pipeline = OptPipeline::new();
    pipeline.run(graph)
}

/// Run full optimization pipeline with PGO profile data.
///
/// This is the primary entry point for Tier 2 compilation. Profile data
/// from Tier 1 execution is used to guide branch probability estimation,
/// hot/cold splitting, and loop frequency calculations.
pub fn optimize_with_profile(
    graph: &mut Graph,
    profile: crate::runtime::profile_data::ProfileData,
) -> PipelineStats {
    let mut pipeline = OptPipeline::with_profile(PipelineConfig::full(), profile);
    pipeline.run(graph)
}
