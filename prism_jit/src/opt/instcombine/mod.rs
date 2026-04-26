//! Instruction Combining (InstCombine) optimization pass.
//!
//! Combines and simplifies sequences of instructions using pattern matching.
//!
//! # Algorithm Overview
//!
//! 1. Build worklist of all instructions
//! 2. Pop instruction from worklist
//! 3. Try all applicable patterns
//! 4. If transformed, add uses to worklist
//! 5. Repeat until worklist empty
//!
//! # Pattern Categories
//!
//! - Arithmetic: x + 0, x * 1, x - x, etc.
//! - Bitwise: x & 0, x | -1, x ^ x, etc.
//! - Comparison: x < x, x == x, etc.
//! - Memory: load after store, redundant stores
//! - Control: branch on constant, dead paths

mod patterns;
mod worklist;

pub use patterns::{Pattern, PatternMatch, PatternRegistry};
pub use worklist::Worklist;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::opt::OptimizationPass;

// =============================================================================
// InstCombine Statistics
// =============================================================================

/// Statistics from instruction combining.
#[derive(Debug, Clone, Default)]
pub struct InstCombineStats {
    /// Patterns successfully applied.
    pub patterns_applied: usize,
    /// Instructions eliminated.
    pub instructions_eliminated: usize,
    /// Instructions simplified.
    pub instructions_simplified: usize,
    /// New instructions created.
    pub instructions_created: usize,
    /// Total instructions analyzed.
    pub instructions_analyzed: usize,
}

impl InstCombineStats {
    /// Net instruction reduction.
    pub fn net_reduction(&self) -> isize {
        self.instructions_eliminated as isize - self.instructions_created as isize
    }

    /// Merge statistics.
    pub fn merge(&mut self, other: &InstCombineStats) {
        self.patterns_applied += other.patterns_applied;
        self.instructions_eliminated += other.instructions_eliminated;
        self.instructions_simplified += other.instructions_simplified;
        self.instructions_created += other.instructions_created;
        self.instructions_analyzed += other.instructions_analyzed;
    }
}

// =============================================================================
// InstCombine Configuration
// =============================================================================

/// Configuration for instruction combining.
#[derive(Debug, Clone)]
pub struct InstCombineConfig {
    /// Maximum worklist iterations (prevent infinite loops).
    pub max_iterations: usize,
    /// Enable arithmetic patterns.
    pub enable_arithmetic: bool,
    /// Enable bitwise patterns.
    pub enable_bitwise: bool,
    /// Enable comparison patterns.
    pub enable_comparison: bool,
    /// Enable memory patterns.
    pub enable_memory: bool,
    /// Enable control flow patterns.
    pub enable_control: bool,
}

impl Default for InstCombineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            enable_arithmetic: true,
            enable_bitwise: true,
            enable_comparison: true,
            enable_memory: true,
            enable_control: true,
        }
    }
}

// =============================================================================
// InstCombine Pass
// =============================================================================

/// Instruction combining optimization pass.
#[derive(Debug)]
pub struct InstCombine {
    /// Configuration.
    config: InstCombineConfig,
    /// Statistics from last run.
    stats: InstCombineStats,
    /// Pattern registry.
    patterns: PatternRegistry,
}

impl InstCombine {
    /// Create a new instruction combine pass.
    pub fn new() -> Self {
        Self {
            config: InstCombineConfig::default(),
            stats: InstCombineStats::default(),
            patterns: PatternRegistry::new(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: InstCombineConfig) -> Self {
        Self {
            config,
            stats: InstCombineStats::default(),
            patterns: PatternRegistry::new(),
        }
    }

    /// Get statistics from last run.
    pub fn stats(&self) -> &InstCombineStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &InstCombineConfig {
        &self.config
    }

    /// Run instruction combining.
    fn run_instcombine(&mut self, graph: &mut Graph) -> bool {
        self.stats = InstCombineStats::default();

        // Build initial worklist
        let mut worklist = Worklist::new();
        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if graph.get(id).is_some() {
                worklist.push(id);
            }
        }

        self.stats.instructions_analyzed = worklist.len();

        let mut changed = false;
        let mut iterations = 0;

        // Process worklist
        while let Some(node_id) = worklist.pop() {
            if iterations >= self.config.max_iterations {
                break;
            }
            iterations += 1;

            // Try to match patterns
            if let Some(matched) = self.try_patterns(graph, node_id) {
                // Apply the transformation
                if self.apply_pattern(graph, &matched, &mut worklist) {
                    changed = true;
                    self.stats.patterns_applied += 1;
                }
            }
        }

        changed
    }

    /// Try to match any pattern on a node.
    fn try_patterns(&self, graph: &Graph, node_id: NodeId) -> Option<PatternMatch> {
        let node = graph.get(node_id)?;

        // Try each enabled pattern category
        for pattern in self.patterns.iter() {
            // Check if pattern category is enabled
            if !self.is_pattern_enabled(pattern) {
                continue;
            }

            // Try to match
            if let Some(matched) = pattern.try_match(graph, node_id, &node.op) {
                return Some(matched);
            }
        }

        None
    }

    /// Check if a pattern category is enabled.
    fn is_pattern_enabled(&self, pattern: &Pattern) -> bool {
        match pattern.category() {
            PatternCategory::Arithmetic => self.config.enable_arithmetic,
            PatternCategory::Bitwise => self.config.enable_bitwise,
            PatternCategory::Comparison => self.config.enable_comparison,
            PatternCategory::Memory => self.config.enable_memory,
            PatternCategory::Control => self.config.enable_control,
        }
    }

    /// Apply a pattern match to the graph.
    fn apply_pattern(
        &mut self,
        graph: &mut Graph,
        matched: &PatternMatch,
        worklist: &mut Worklist,
    ) -> bool {
        // Apply the transformation
        matched.apply(graph);

        // Add affected nodes to worklist
        if let Some(replacement) = matched.replacement() {
            // Add uses of the replacement
            for &use_id in graph.uses(replacement) {
                worklist.push(use_id);
            }
            self.stats.instructions_simplified += 1;
        }

        if matched.eliminated() {
            self.stats.instructions_eliminated += 1;
        }

        if matched.created_new() {
            self.stats.instructions_created += 1;
        }

        true
    }
}

impl Default for InstCombine {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for InstCombine {
    fn name(&self) -> &'static str {
        "instcombine"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_instcombine(graph)
    }
}

// =============================================================================
// Pattern Category
// =============================================================================

/// Categories of patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternCategory {
    /// Arithmetic patterns (add, sub, mul, div).
    Arithmetic,
    /// Bitwise patterns (and, or, xor, shifts).
    Bitwise,
    /// Comparison patterns.
    Comparison,
    /// Memory patterns.
    Memory,
    /// Control flow patterns.
    Control,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
