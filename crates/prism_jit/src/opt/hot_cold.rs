//! Hot/cold code splitting and function layout optimization.
//!
//! Uses branch probability and block frequency data to partition code
//! into hot and cold regions. Hot code is placed on the fall-through
//! path while cold code is moved out-of-line to improve instruction
//! cache utilization.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────┐    ┌──────────────┐    ┌──────────────────┐
//! │ BranchAnnota-  │───▶│  HotColdSplit │───▶│  Split Layout    │
//! │ tions (probs)  │    │  (partition)  │    │  (hot + cold)    │
//! └────────────────┘    └──────────────┘    └──────────────────┘
//! ```

use super::branch_probability::{BlockFrequency, BranchAnnotations, BranchProbability};
use rustc_hash::{FxHashMap, FxHashSet};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for hot/cold splitting.
#[derive(Debug, Clone)]
pub struct HotColdConfig {
    /// Minimum block frequency to be considered "hot".
    pub hot_threshold: f64,
    /// Maximum block frequency to be considered "cold".
    pub cold_threshold: f64,
    /// Minimum function execution count to enable splitting.
    pub min_execution_count: u64,
    /// Maximum number of cold regions to create (limits fragmentation).
    pub max_cold_regions: usize,
    /// Branch probability threshold below which the target is cold.
    pub cold_branch_probability: f64,
}

impl Default for HotColdConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 1.0,
            cold_threshold: 0.01,
            min_execution_count: 100,
            max_cold_regions: 16,
            cold_branch_probability: 0.05,
        }
    }
}

impl HotColdConfig {
    /// Aggressive splitting for maximum I-cache benefit.
    pub fn aggressive() -> Self {
        Self {
            hot_threshold: 0.5,
            cold_threshold: 0.1,
            min_execution_count: 50,
            max_cold_regions: 32,
            cold_branch_probability: 0.1,
        }
    }

    /// Conservative splitting (only obvious cold code).
    pub fn conservative() -> Self {
        Self {
            hot_threshold: 2.0,
            cold_threshold: 0.001,
            min_execution_count: 500,
            max_cold_regions: 8,
            cold_branch_probability: 0.01,
        }
    }
}

// =============================================================================
// Block Classification
// =============================================================================

/// Temperature classification for a code block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockTemperature {
    /// Hot — on the fast path, should be in-line.
    Hot,
    /// Warm — moderate frequency, keep near hot code.
    Warm,
    /// Cold — rarely executed, move out-of-line.
    Cold,
    /// Frozen — never/almost never executed (error paths, etc).
    Frozen,
}

impl BlockTemperature {
    /// Whether this block should remain in-line.
    pub fn is_inline(self) -> bool {
        matches!(self, Self::Hot | Self::Warm)
    }

    /// Whether this block should be moved out-of-line.
    pub fn is_out_of_line(self) -> bool {
        matches!(self, Self::Cold | Self::Frozen)
    }
}

// =============================================================================
// Split Region
// =============================================================================

/// A contiguous region of blocks with the same temperature.
#[derive(Debug, Clone)]
pub struct SplitRegion {
    /// Block IDs in this region (in layout order).
    blocks: Vec<u32>,
    /// Temperature of this region.
    temperature: BlockTemperature,
    /// Aggregate frequency of this region.
    frequency: f64,
}

impl SplitRegion {
    /// Create a new region.
    pub fn new(temperature: BlockTemperature) -> Self {
        Self {
            blocks: Vec::new(),
            temperature,
            frequency: 0.0,
        }
    }

    /// Add a block to this region.
    pub fn add_block(&mut self, block_id: u32, freq: f64) {
        self.blocks.push(block_id);
        self.frequency += freq;
    }

    /// Blocks in this region.
    pub fn blocks(&self) -> &[u32] {
        &self.blocks
    }

    /// Temperature of this region.
    pub fn temperature(&self) -> BlockTemperature {
        self.temperature
    }

    /// Number of blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Whether this region is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Aggregate frequency.
    pub fn frequency(&self) -> f64 {
        self.frequency
    }
}

// =============================================================================
// Split Layout
// =============================================================================

/// The result of hot/cold splitting: an ordered layout with temperature zones.
#[derive(Debug, Clone)]
pub struct SplitLayout {
    /// Hot regions (placed first in layout).
    hot_regions: Vec<SplitRegion>,
    /// Cold regions (placed after hot code).
    cold_regions: Vec<SplitRegion>,
    /// Per-block temperature classification.
    block_temps: FxHashMap<u32, BlockTemperature>,
    /// Statistics.
    stats: SplitStats,
}

/// Statistics from the splitting pass.
#[derive(Debug, Clone, Default)]
pub struct SplitStats {
    /// Total blocks analyzed.
    pub total_blocks: usize,
    /// Blocks classified as hot.
    pub hot_blocks: usize,
    /// Blocks classified as warm.
    pub warm_blocks: usize,
    /// Blocks classified as cold.
    pub cold_blocks: usize,
    /// Blocks classified as frozen.
    pub frozen_blocks: usize,
    /// Number of cold regions created.
    pub cold_regions: usize,
}

impl SplitLayout {
    /// Create a new empty layout.
    pub fn new() -> Self {
        Self {
            hot_regions: Vec::new(),
            cold_regions: Vec::new(),
            block_temps: FxHashMap::default(),
            stats: SplitStats::default(),
        }
    }

    /// Get the temperature of a block.
    pub fn temperature(&self, block_id: u32) -> Option<BlockTemperature> {
        self.block_temps.get(&block_id).copied()
    }

    /// Whether a block is hot.
    pub fn is_hot(&self, block_id: u32) -> bool {
        self.block_temps
            .get(&block_id)
            .map_or(false, |t| matches!(t, BlockTemperature::Hot))
    }

    /// Whether a block is cold.
    pub fn is_cold(&self, block_id: u32) -> bool {
        self.block_temps
            .get(&block_id)
            .map_or(false, |t| t.is_out_of_line())
    }

    /// Get hot regions.
    pub fn hot_regions(&self) -> &[SplitRegion] {
        &self.hot_regions
    }

    /// Get cold regions.
    pub fn cold_regions(&self) -> &[SplitRegion] {
        &self.cold_regions
    }

    /// Get splitting statistics.
    pub fn stats(&self) -> &SplitStats {
        &self.stats
    }

    /// Total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.stats.total_blocks
    }

    /// Iterate blocks in layout order (hot first, then cold).
    pub fn layout_order(&self) -> Vec<u32> {
        let mut order = Vec::new();
        for region in &self.hot_regions {
            order.extend_from_slice(&region.blocks);
        }
        for region in &self.cold_regions {
            order.extend_from_slice(&region.blocks);
        }
        order
    }
}

impl Default for SplitLayout {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Hot/Cold Splitter
// =============================================================================

/// Partitions code into hot and cold regions based on profile data.
pub struct HotColdSplitter {
    config: HotColdConfig,
}

impl HotColdSplitter {
    /// Create a new splitter with default configuration.
    pub fn new() -> Self {
        Self {
            config: HotColdConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: HotColdConfig) -> Self {
        Self { config }
    }

    /// Classify blocks by temperature using branch annotations.
    pub fn classify_blocks(
        &self,
        block_ids: &[u32],
        annotations: &BranchAnnotations,
    ) -> SplitLayout {
        let mut layout = SplitLayout::new();
        layout.stats.total_blocks = block_ids.len();

        // Phase 1: Classify each block
        for &block_id in block_ids {
            let freq = annotations
                .get_block_freq(block_id)
                .unwrap_or(BlockFrequency::ENTRY);
            let temp = self.classify_temperature(freq);
            layout.block_temps.insert(block_id, temp);

            match temp {
                BlockTemperature::Hot => layout.stats.hot_blocks += 1,
                BlockTemperature::Warm => layout.stats.warm_blocks += 1,
                BlockTemperature::Cold => layout.stats.cold_blocks += 1,
                BlockTemperature::Frozen => layout.stats.frozen_blocks += 1,
            }
        }

        // Phase 2: Build contiguous regions
        let mut current_hot = SplitRegion::new(BlockTemperature::Hot);
        let mut current_cold = SplitRegion::new(BlockTemperature::Cold);

        for &block_id in block_ids {
            let temp = layout.block_temps[&block_id];
            let freq = annotations
                .get_block_freq(block_id)
                .map(|f| f.value())
                .unwrap_or(1.0);

            if temp.is_inline() {
                // Flush any accumulated cold region
                if !current_cold.is_empty() {
                    if layout.cold_regions.len() < self.config.max_cold_regions {
                        layout.cold_regions.push(current_cold);
                    } else {
                        // Too many cold regions, merge into hot
                        for &b in current_cold.blocks() {
                            current_hot.add_block(b, 0.0);
                        }
                    }
                    current_cold = SplitRegion::new(BlockTemperature::Cold);
                }
                current_hot.add_block(block_id, freq);
            } else {
                // Flush any accumulated hot region
                if !current_hot.is_empty() {
                    layout.hot_regions.push(current_hot);
                    current_hot = SplitRegion::new(BlockTemperature::Hot);
                }
                current_cold.add_block(block_id, freq);
            }
        }

        // Flush remaining regions
        if !current_hot.is_empty() {
            layout.hot_regions.push(current_hot);
        }
        if !current_cold.is_empty() {
            if layout.cold_regions.len() < self.config.max_cold_regions {
                layout.cold_regions.push(current_cold);
            }
        }

        layout.stats.cold_regions = layout.cold_regions.len();
        layout
    }

    /// Classify a block's temperature based on its frequency.
    fn classify_temperature(&self, freq: BlockFrequency) -> BlockTemperature {
        let v = freq.value();
        if v >= self.config.hot_threshold {
            BlockTemperature::Hot
        } else if v >= self.config.cold_threshold {
            BlockTemperature::Warm
        } else if v > 0.0 {
            BlockTemperature::Cold
        } else {
            BlockTemperature::Frozen
        }
    }

    /// Identify cold branches — branches whose targets are cold.
    pub fn identify_cold_branches(&self, annotations: &BranchAnnotations) -> FxHashSet<u32> {
        let mut cold_branches = FxHashSet::default();
        let threshold = BranchProbability::from_f64(self.config.cold_branch_probability);

        for (offset, prob) in annotations.iter_branches() {
            if prob.numerator() <= threshold.numerator() {
                cold_branches.insert(offset);
            }
        }

        cold_branches
    }
}

impl Default for HotColdSplitter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Hot/Cold Pass (Pipeline Adapter)
// =============================================================================

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::operators::{ControlOp, Operator};

/// Pipeline-integrated hot/cold splitting pass.
///
/// Analyzes the IR graph to classify blocks by execution temperature,
/// then produces a `SplitLayout` that downstream passes can use for
/// code placement decisions (hot code inline, cold code out-of-line).
///
/// # Pipeline Phase
///
/// Runs in the `ProfileGuided` phase *after* `BranchProbabilityPass`,
/// consuming branch annotations to inform temperature classification.
pub struct HotColdPass {
    /// Configuration for splitting thresholds.
    config: HotColdConfig,
    /// The computed split layout from the last run.
    layout: SplitLayout,
    /// Whether splitting was performed.
    did_split: bool,
}

impl HotColdPass {
    /// Create a new hot/cold splitting pass.
    pub fn new() -> Self {
        Self {
            config: HotColdConfig::default(),
            layout: SplitLayout::new(),
            did_split: false,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: HotColdConfig) -> Self {
        Self {
            config,
            layout: SplitLayout::new(),
            did_split: false,
        }
    }

    /// Get the computed split layout.
    pub fn layout(&self) -> &SplitLayout {
        &self.layout
    }

    /// Whether splitting was performed in the last run.
    pub fn did_split(&self) -> bool {
        self.did_split
    }
}

impl Default for HotColdPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for HotColdPass {
    fn name(&self) -> &'static str {
        "HotColdSplit"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        let splitter = HotColdSplitter::with_config(self.config.clone());

        // Collect control-flow block IDs from the graph.
        // In Sea-of-Nodes, "blocks" correspond to Region, Loop, Start, and
        // If control nodes.
        let mut block_ids = Vec::new();
        let mut annotations = BranchAnnotations::new();

        for (id, node) in graph.iter() {
            let offset = id.index() as u32;
            match node.op {
                Operator::Control(ControlOp::Region) => {
                    block_ids.push(offset);
                    // Default entry-level frequency
                    annotations.set_block_freq(offset, BlockFrequency::ENTRY);
                }
                Operator::Control(ControlOp::Loop) => {
                    block_ids.push(offset);
                    // Loops get elevated frequency
                    annotations.set_block_freq(offset, BlockFrequency::for_loop(1.0, 10.0));
                }
                Operator::Control(ControlOp::Start) => {
                    block_ids.push(offset);
                    annotations.set_block_freq(offset, BlockFrequency::ENTRY);
                }
                Operator::Control(ControlOp::If) => {
                    // If nodes generate branch probabilities
                    annotations.set_branch(offset, BranchProbability::EVEN);
                }
                Operator::Control(ControlOp::Throw) | Operator::Control(ControlOp::Deopt) => {
                    // Exception/deopt paths are cold
                    block_ids.push(offset);
                    annotations.set_block_freq(offset, BlockFrequency::COLD);
                }
                _ => {}
            }
        }

        if block_ids.is_empty() {
            self.did_split = false;
            return false;
        }

        let layout = splitter.classify_blocks(&block_ids, &annotations);
        let has_cold = !layout.cold_regions().is_empty();

        self.layout = layout;
        self.did_split = has_cold;
        has_cold
    }
}
