//! Vectorization Optimization Pass
//!
//! This module provides comprehensive SIMD vectorization for the Prism JIT:
//!
//! - **Memory Dependence Analysis**: Safety analysis for reordering/vectorization
//! - **Cost Model**: Target-aware profitability decisions
//! - **SLP Vectorization**: Superword-Level Parallelism for straight-line code
//! - **Loop Vectorization**: Transform counted loops to vector operations
//!
//! # Module Structure
//!
//! - `dependence`: Memory dependence graph construction and analysis
//! - `legality`: Vectorization safety checking
//! - `cost`: Target-specific cost model
//! - `slp`: SLP vectorizer
//! - `loop_vec`: Loop vectorization pass
//!
//! # Architecture
//!
//! The vectorization pipeline follows this order:
//!
//! 1. **Dependence Analysis**: Build memory dependence graph
//! 2. **Legality Checking**: Determine if vectorization is safe
//! 3. **Cost Analysis**: Determine if vectorization is profitable
//! 4. **Transformation**: Apply SLP or loop vectorization
//!
//! # Example
//!
//! ```text
//! Before:
//!   for i in range(n):
//!       a[i] = b[i] + c[i]
//!
//! After (vectorized with 4-wide SIMD):
//!   for i in range(0, n, 4):
//!       a[i:i+4] = b[i:i+4] + c[i:i+4]
//!   # Epilog handles remainder
//! ```

pub mod cost;
pub mod dependence;
pub mod legality;
pub mod loop_vec;
pub mod orchestrator;
pub mod reduction;
pub mod slp;

use crate::ir::graph::Graph;
use crate::opt::OptimizationPass;

pub use cost::{CostAnalysis, OpCost, SimdLevel, VectorCostModel};
pub use dependence::{Dependence, DependenceGraph, DependenceKind, Direction, Distance};
pub use legality::{LegalityResult, LegalityViolation, ViolationKind};
pub use loop_vec::{
    LoopVecAnalysis, LoopVectorizer, RuntimeCheck, RuntimeCheckKind, VecRejectReason,
};
pub use orchestrator::{LoopDecision, LoopRejection, SlpDecision, VectorizationOrchestrator};
pub use reduction::{Reduction, ReductionAnalysis, ReductionDetector, ReductionKind};
pub use slp::{Pack, SlpTree, SlpVectorizer};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for vectorization passes.
#[derive(Debug, Clone)]
pub struct VectorizeConfig {
    /// Target SIMD level.
    pub simd_level: SimdLevel,
    /// Minimum trip count for loop vectorization.
    pub min_trip_count: u64,
    /// Maximum vector width to attempt (in elements).
    pub max_vector_width: usize,
    /// Enable SLP vectorization.
    pub enable_slp: bool,
    /// Enable loop vectorization.
    pub enable_loop_vec: bool,
    /// Enable gather/scatter for non-contiguous access.
    pub enable_gather_scatter: bool,
    /// Cost threshold - vectorize only if savings exceed this.
    pub cost_threshold: f32,
    /// Enable interleaving for memory-bound loops.
    pub enable_interleaving: bool,
    /// Maximum interleave factor.
    pub max_interleave_factor: usize,
}

impl Default for VectorizeConfig {
    fn default() -> Self {
        Self {
            simd_level: SimdLevel::Avx2,
            min_trip_count: 8,
            max_vector_width: 8,
            enable_slp: true,
            enable_loop_vec: true,
            enable_gather_scatter: false, // Conservative default
            cost_threshold: 1.0,
            enable_interleaving: true,
            max_interleave_factor: 4,
        }
    }
}

impl VectorizeConfig {
    /// Create configuration for SSE4.2 targets.
    pub fn sse42() -> Self {
        Self {
            simd_level: SimdLevel::Sse42,
            max_vector_width: 2,
            enable_gather_scatter: false,
            ..Default::default()
        }
    }

    /// Create configuration for AVX2 targets.
    pub fn avx2() -> Self {
        Self {
            simd_level: SimdLevel::Avx2,
            max_vector_width: 4,
            enable_gather_scatter: false,
            ..Default::default()
        }
    }

    /// Create configuration for AVX-512 targets.
    pub fn avx512() -> Self {
        Self {
            simd_level: SimdLevel::Avx512,
            max_vector_width: 8,
            enable_gather_scatter: true,
            ..Default::default()
        }
    }

    /// Create aggressive configuration for maximum vectorization.
    pub fn aggressive() -> Self {
        Self {
            simd_level: SimdLevel::Avx512,
            min_trip_count: 4,
            max_vector_width: 16,
            enable_slp: true,
            enable_loop_vec: true,
            enable_gather_scatter: true,
            cost_threshold: 0.5,
            enable_interleaving: true,
            max_interleave_factor: 8,
        }
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics from vectorization passes.
#[derive(Debug, Clone, Default)]
pub struct VectorizeStats {
    /// Number of loops analyzed.
    pub loops_analyzed: usize,
    /// Number of loops successfully vectorized.
    pub loops_vectorized: usize,
    /// Number of loops rejected (with reason breakdown).
    pub loops_rejected_unsafe: usize,
    /// Number of loops rejected as unprofitable.
    pub loops_rejected_unprofitable: usize,
    /// Number of SLP regions analyzed.
    pub slp_regions_analyzed: usize,
    /// Number of SLP regions vectorized.
    pub slp_regions_vectorized: usize,
    /// Total vector operations created.
    pub vector_ops_created: usize,
    /// Total scalar operations eliminated.
    pub scalar_ops_eliminated: usize,
    /// Total estimated speedup factor.
    pub estimated_speedup: f32,
}

impl VectorizeStats {
    /// Merge statistics from another instance.
    pub fn merge(&mut self, other: &VectorizeStats) {
        self.loops_analyzed += other.loops_analyzed;
        self.loops_vectorized += other.loops_vectorized;
        self.loops_rejected_unsafe += other.loops_rejected_unsafe;
        self.loops_rejected_unprofitable += other.loops_rejected_unprofitable;
        self.slp_regions_analyzed += other.slp_regions_analyzed;
        self.slp_regions_vectorized += other.slp_regions_vectorized;
        self.vector_ops_created += other.vector_ops_created;
        self.scalar_ops_eliminated += other.scalar_ops_eliminated;
        self.estimated_speedup += other.estimated_speedup;
    }

    /// Get the vectorization success rate.
    pub fn success_rate(&self) -> f32 {
        if self.loops_analyzed == 0 {
            0.0
        } else {
            self.loops_vectorized as f32 / self.loops_analyzed as f32
        }
    }
}

// =============================================================================
// Main Pass
// =============================================================================

/// Vectorization optimization pass.
///
/// This pass attempts to vectorize loops and straight-line code using SIMD
/// instructions. It integrates with the cost model to ensure profitable
/// transformations.
pub struct Vectorize {
    /// Configuration.
    config: VectorizeConfig,
    /// Accumulated statistics.
    stats: VectorizeStats,
}

impl Vectorize {
    /// Create vectorization pass with default configuration.
    pub fn new() -> Self {
        Self {
            config: VectorizeConfig::default(),
            stats: VectorizeStats::default(),
        }
    }

    /// Create vectorization pass with custom configuration.
    pub fn with_config(config: VectorizeConfig) -> Self {
        Self {
            config,
            stats: VectorizeStats::default(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &VectorizeConfig {
        &self.config
    }

    /// Get accumulated statistics.
    pub fn stats(&self) -> &VectorizeStats {
        &self.stats
    }

    /// Get number of loops vectorized.
    pub fn loops_vectorized(&self) -> usize {
        self.stats.loops_vectorized
    }

    /// Get number of SLP regions vectorized.
    pub fn slp_regions_vectorized(&self) -> usize {
        self.stats.slp_regions_vectorized
    }
}

impl Default for Vectorize {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Vectorize {
    fn name(&self) -> &'static str {
        "vectorize"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        let mut orchestrator = VectorizationOrchestrator::new(&self.config);
        let changed = orchestrator.run(graph);
        self.stats.merge(orchestrator.stats());
        changed
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = VectorizeConfig::default();
        assert_eq!(config.simd_level, SimdLevel::Avx2);
        assert_eq!(config.min_trip_count, 8);
        assert!(config.enable_slp);
        assert!(config.enable_loop_vec);
    }

    #[test]
    fn test_config_sse42() {
        let config = VectorizeConfig::sse42();
        assert_eq!(config.simd_level, SimdLevel::Sse42);
        assert_eq!(config.max_vector_width, 2);
        assert!(!config.enable_gather_scatter);
    }

    #[test]
    fn test_config_avx2() {
        let config = VectorizeConfig::avx2();
        assert_eq!(config.simd_level, SimdLevel::Avx2);
        assert_eq!(config.max_vector_width, 4);
    }

    #[test]
    fn test_config_avx512() {
        let config = VectorizeConfig::avx512();
        assert_eq!(config.simd_level, SimdLevel::Avx512);
        assert_eq!(config.max_vector_width, 8);
        assert!(config.enable_gather_scatter);
    }

    #[test]
    fn test_config_aggressive() {
        let config = VectorizeConfig::aggressive();
        assert_eq!(config.simd_level, SimdLevel::Avx512);
        assert_eq!(config.max_vector_width, 16);
        assert_eq!(config.min_trip_count, 4);
        assert!(config.enable_gather_scatter);
    }

    #[test]
    fn test_stats_default() {
        let stats = VectorizeStats::default();
        assert_eq!(stats.loops_analyzed, 0);
        assert_eq!(stats.loops_vectorized, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_stats_merge() {
        let mut stats1 = VectorizeStats {
            loops_analyzed: 10,
            loops_vectorized: 5,
            ..Default::default()
        };
        let stats2 = VectorizeStats {
            loops_analyzed: 20,
            loops_vectorized: 15,
            ..Default::default()
        };
        stats1.merge(&stats2);
        assert_eq!(stats1.loops_analyzed, 30);
        assert_eq!(stats1.loops_vectorized, 20);
    }

    #[test]
    fn test_stats_success_rate() {
        let stats = VectorizeStats {
            loops_analyzed: 10,
            loops_vectorized: 7,
            ..Default::default()
        };
        assert!((stats.success_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_vectorize_pass_new() {
        let pass = Vectorize::new();
        assert_eq!(pass.name(), "vectorize");
        assert_eq!(pass.loops_vectorized(), 0);
    }

    #[test]
    fn test_vectorize_pass_with_config() {
        let config = VectorizeConfig::avx512();
        let pass = Vectorize::with_config(config.clone());
        assert_eq!(pass.config().simd_level, SimdLevel::Avx512);
    }
}
