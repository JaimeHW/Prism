//! Remainder Handling for Loop Unrolling.
//!
//! When a loop is partially unrolled, the trip count may not be evenly
//! divisible by the unroll factor. This module handles the "remainder"
//! iterations that don't fit in the main unrolled loop.
//!
//! # Strategies
//!
//! - **Epilog Loop**: A simple loop handles remainder iterations
//! - **Unrolled Remainder**: Fully unroll the remainder (for small factors)
//! - **Duff's Device**: Jump table to the correct iteration
//! - **Prolog Loop**: Handle remainder before the main loop
//!
//! # Example
//!
//! For a loop with trip count N and unroll factor 4:
//! - Main loop runs N/4 iterations (each doing 4 original iterations)
//! - Remainder loop runs N%4 iterations

use crate::ir::cfg::Cfg;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;

use rustc_hash::FxHashMap;

// =============================================================================
// Remainder Strategy
// =============================================================================

/// Strategy for handling remainder iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemainderStrategy {
    /// No remainder handling (trip count evenly divides).
    None,

    /// Epilog loop: simple loop after the main unrolled loop.
    EpilogLoop,

    /// Prolog loop: handle remainder before main loop.
    PrologLoop,

    /// Fully unroll remainder iterations with guards.
    UnrolledRemainder,

    /// Duff's device style jump table.
    DuffsDevice,
}

impl RemainderStrategy {
    /// Check if this strategy adds a loop.
    pub fn adds_loop(&self) -> bool {
        matches!(
            self,
            RemainderStrategy::EpilogLoop | RemainderStrategy::PrologLoop
        )
    }

    /// Check if this strategy adds straight-line code.
    pub fn adds_straight_line(&self) -> bool {
        matches!(
            self,
            RemainderStrategy::UnrolledRemainder | RemainderStrategy::DuffsDevice
        )
    }

    /// Estimate code growth factor.
    pub fn code_growth(&self, unroll_factor: u32, body_size: usize) -> usize {
        match self {
            RemainderStrategy::None => 0,
            RemainderStrategy::EpilogLoop => body_size + 10, // Body + loop overhead
            RemainderStrategy::PrologLoop => body_size + 10,
            RemainderStrategy::UnrolledRemainder => {
                // Up to (factor - 1) copies with guards
                (unroll_factor as usize - 1) * (body_size + 2)
            }
            RemainderStrategy::DuffsDevice => {
                // All copies + jump table
                (unroll_factor as usize - 1) * body_size + 10
            }
        }
    }
}

impl Default for RemainderStrategy {
    fn default() -> Self {
        RemainderStrategy::EpilogLoop
    }
}

// =============================================================================
// Remainder Generator
// =============================================================================

/// Generates remainder handling code.
#[derive(Debug)]
pub struct RemainderGenerator<'a> {
    #[allow(dead_code)]
    graph: &'a mut Graph,
    #[allow(dead_code)]
    cfg: &'a mut Cfg,
    /// Node mapping from original to cloned.
    #[allow(dead_code)]
    node_map: FxHashMap<NodeId, NodeId>,
}

impl<'a> RemainderGenerator<'a> {
    /// Create a new remainder generator.
    pub fn new(graph: &'a mut Graph, cfg: &'a mut Cfg) -> Self {
        Self {
            graph,
            cfg,
            node_map: FxHashMap::default(),
        }
    }

    /// Generate remainder handling for a partially unrolled loop.
    pub fn generate(
        &mut self,
        strategy: RemainderStrategy,
        _unroll_factor: u32,
        _loop_body: &[NodeId],
        _induction_var: NodeId,
        _limit: NodeId,
    ) -> RemainderResult {
        match strategy {
            RemainderStrategy::None => RemainderResult::none(),
            RemainderStrategy::EpilogLoop => self.generate_epilog_loop(),
            RemainderStrategy::PrologLoop => self.generate_prolog_loop(),
            RemainderStrategy::UnrolledRemainder => self.generate_unrolled_remainder(),
            RemainderStrategy::DuffsDevice => self.generate_duffs_device(),
        }
    }

    /// Generate an epilog loop for remainder iterations.
    fn generate_epilog_loop(&mut self) -> RemainderResult {
        // The epilog loop is a simplified version of the original loop
        // that runs for the remaining iterations.
        //
        // Structure:
        //   main_loop_exit:
        //     current_iv = ... (from main loop)
        //     goto epilog_check
        //
        //   epilog_check:
        //     if current_iv < limit:
        //       goto epilog_body
        //     else:
        //       goto epilog_exit
        //
        //   epilog_body:
        //     ... original body ...
        //     current_iv = current_iv + step
        //     goto epilog_check
        //
        //   epilog_exit:
        //     ... continue ...

        // For now, return a placeholder - actual implementation requires
        // cloning the loop body and adjusting control flow
        RemainderResult {
            entry: None,
            exit: None,
            nodes_added: 0,
        }
    }

    /// Generate a prolog loop for remainder iterations.
    fn generate_prolog_loop(&mut self) -> RemainderResult {
        // The prolog loop handles remainder iterations BEFORE the main loop.
        // This is sometimes better for alignment purposes.
        //
        // Structure:
        //   prolog_check:
        //     remainder = trip_count % factor
        //     if remainder > 0:
        //       goto prolog_body
        //     else:
        //       goto main_loop
        //
        //   prolog_body:
        //     ... original body ...
        //     i = i + 1
        //     remainder = remainder - 1
        //     if remainder > 0:
        //       goto prolog_body
        //     else:
        //       goto main_loop

        RemainderResult {
            entry: None,
            exit: None,
            nodes_added: 0,
        }
    }

    /// Generate unrolled remainder with guards.
    fn generate_unrolled_remainder(&mut self) -> RemainderResult {
        // Generate straight-line code for all possible remainders.
        //
        // For factor = 4:
        //   remainder = trip_count % 4
        //   if remainder >= 3: body()
        //   if remainder >= 2: body()
        //   if remainder >= 1: body()
        //
        // This avoids the overhead of a loop for small remainders.

        RemainderResult {
            entry: None,
            exit: None,
            nodes_added: 0,
        }
    }

    /// Generate Duff's device style jump table.
    fn generate_duffs_device(&mut self) -> RemainderResult {
        // Duff's device uses a computed goto (switch) to jump into the
        // middle of an unrolled loop.
        //
        // switch (trip_count % 4) {
        //   case 3: body(); // fallthrough
        //   case 2: body(); // fallthrough
        //   case 1: body(); // fallthrough
        //   case 0: break;
        // }
        //
        // This is efficient but requires careful handling of phi nodes.

        RemainderResult {
            entry: None,
            exit: None,
            nodes_added: 0,
        }
    }
}

// =============================================================================
// Remainder Result
// =============================================================================

/// Result of remainder code generation.
#[derive(Debug, Clone)]
pub struct RemainderResult {
    /// Entry point of remainder code (if any).
    pub entry: Option<NodeId>,
    /// Exit point of remainder code (if any).
    pub exit: Option<NodeId>,
    /// Number of nodes added.
    pub nodes_added: usize,
}

impl RemainderResult {
    /// Create a result for no remainder handling.
    pub fn none() -> Self {
        Self {
            entry: None,
            exit: None,
            nodes_added: 0,
        }
    }

    /// Check if remainder handling was generated.
    pub fn has_remainder(&self) -> bool {
        self.entry.is_some()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
