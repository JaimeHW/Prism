//! Optimization Pipeline.
//!
//! Orchestrates multiple optimization passes with proper ordering,
//! fixed-point iteration, and comprehensive statistics.
//!
//! # Pass Phases
//!
//! 1. **Canonicalization**: Simplify, constant folding
//! 2. **Local**: GVN, copy propagation
//! 3. **Loop**: LICM, loop unrolling
//! 4. **Interprocedural**: Inlining, escape analysis
//! 5. **Cleanup**: DCE, CFG simplification
//!
//! # Fixed-Point Iteration
//!
//! The pipeline runs passes until no changes occur or a maximum
//! iteration count is reached. Some passes trigger re-running
//! of earlier passes (e.g., inlining enables more GVN).

use super::dce::Dce;
use super::escape::Escape;
use super::gvn::Gvn;
use super::inline::Inline;
use super::licm::Licm;
use super::simplify::Simplify;
use super::OptimizationPass;
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

    /// Enable LICM.
    pub enable_licm: bool,

    /// Enable inlining.
    pub enable_inline: bool,

    /// Enable escape analysis.
    pub enable_escape: bool,

    /// Enable GVN.
    pub enable_gvn: bool,

    /// Enable DCE.
    pub enable_dce: bool,

    /// Enable simplification.
    pub enable_simplify: bool,

    /// Collect timing statistics.
    pub collect_timing: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_iterations_per_phase: 5,
            max_total_iterations: 20,
            enable_licm: true,
            enable_inline: true,
            enable_escape: true,
            enable_gvn: true,
            enable_dce: true,
            enable_simplify: true,
            collect_timing: true,
        }
    }
}

impl PipelineConfig {
    /// Create a minimal configuration (fewer passes for faster compile).
    pub fn minimal() -> Self {
        Self {
            max_iterations_per_phase: 2,
            max_total_iterations: 8,
            enable_licm: false,
            enable_inline: false,
            enable_escape: false,
            enable_gvn: true,
            enable_dce: true,
            enable_simplify: true,
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

    /// Register the default set of optimization passes.
    fn register_default_passes(&mut self) {
        // Canonicalization phase
        if self.config.enable_simplify {
            self.register(Simplify::new(), PassPhase::Canonicalization);
        }

        // Local phase
        if self.config.enable_gvn {
            self.register(Gvn::new(), PassPhase::Local);
        }

        // Loop phase
        if self.config.enable_licm {
            self.register(Licm::new(), PassPhase::Loop);
        }

        // Interprocedural phase
        if self.config.enable_inline {
            self.register(Inline::new(), PassPhase::Interprocedural);
        }
        if self.config.enable_escape {
            self.register(Escape::new(), PassPhase::Interprocedural);
        }

        // Cleanup phase
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

    #[test]
    fn test_pass_phase_ordering() {
        assert!(PassPhase::Canonicalization < PassPhase::Local);
        assert!(PassPhase::Local < PassPhase::Loop);
        assert!(PassPhase::Loop < PassPhase::Interprocedural);
        assert!(PassPhase::Interprocedural < PassPhase::Cleanup);
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.enable_gvn);
        assert!(config.enable_dce);
        assert!(config.enable_licm);
    }

    #[test]
    fn test_pipeline_config_minimal() {
        let config = PipelineConfig::minimal();
        assert!(!config.enable_licm);
        assert!(!config.enable_inline);
        assert!(config.enable_gvn);
    }

    #[test]
    fn test_pipeline_new() {
        let pipeline = OptPipeline::new();
        assert!(!pipeline.passes.is_empty());
    }

    #[test]
    fn test_pipeline_run_empty() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let mut pipeline = OptPipeline::new();
        let stats = pipeline.run(&mut graph);

        assert!(stats.total_iterations > 0);
    }

    #[test]
    fn test_pipeline_run_simple() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let initial = graph.len();

        let mut pipeline = OptPipeline::new();
        let stats = pipeline.run(&mut graph);

        // Should run at least one iteration
        assert!(stats.total_iterations >= 1);
        // Size shouldn't dramatically change for simple graph
        assert!(stats.final_size <= initial + 5);
    }

    #[test]
    fn test_optimize_functions() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let stats = optimize(&mut graph);
        assert!(stats.total_iterations >= 1);

        let mut graph2 = GraphBuilder::new(0, 0).finish();
        let stats2 = optimize_minimal(&mut graph2);
        assert!(stats2.total_iterations >= 1);

        let mut graph3 = GraphBuilder::new(0, 0).finish();
        let stats3 = optimize_full(&mut graph3);
        assert!(stats3.total_iterations >= 1);
    }

    #[test]
    fn test_pipeline_stats() {
        let stats = PipelineStats {
            total_iterations: 5,
            phases_run: 5,
            total_time: Duration::from_millis(100),
            initial_size: 100,
            final_size: 80,
        };

        assert_eq!(stats.size_reduction(), 0.8);
    }

    #[test]
    fn test_pipeline_stats_zero_size() {
        let stats = PipelineStats {
            initial_size: 0,
            final_size: 0,
            ..Default::default()
        };

        assert_eq!(stats.size_reduction(), 1.0);
    }

    #[test]
    fn test_pass_stats() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let mut pipeline = OptPipeline::new();
        pipeline.run(&mut graph);

        let stats = pipeline.pass_stats();
        assert!(!stats.is_empty());

        for stat in stats {
            assert!(!stat.name.is_empty());
        }
    }
}
