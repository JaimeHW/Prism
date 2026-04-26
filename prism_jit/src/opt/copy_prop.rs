//! Copy Propagation optimization pass.
//!
//! Copy Propagation replaces uses of a copied value with the original value,
//! eliminating redundant copies and enabling more optimization opportunities.
//!
//! # Algorithm
//!
//! 1. Identify all copy-like operations (Phi with single value input, identity operations)
//! 2. For each copy `x = y`, replace all uses of `x` with `y`
//! 3. Mark the copy as dead for subsequent DCE
//!
//! # Performance
//!
//! - O(n) where n = number of nodes
//! - Enables more GVN opportunities by exposing identical values
//! - Reduces register pressure by eliminating unnecessary moves

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::Operator;

use rustc_hash::FxHashMap;

// =============================================================================
// Copy Propagation Pass
// =============================================================================

/// Copy Propagation statistics.
#[derive(Debug, Clone, Default)]
pub struct CopyPropStats {
    /// Number of copies identified.
    pub copies_found: usize,
    /// Number of uses rewritten.
    pub uses_rewritten: usize,
    /// Number of copies eliminated (became dead).
    pub copies_eliminated: usize,
    /// Number of phi nodes simplified.
    pub phis_simplified: usize,
}

/// Copy Propagation optimization pass.
///
/// This pass eliminates redundant copies by replacing uses of copied values
/// with the original value. It handles:
///
/// - Direct copies (move operations)
/// - Phi nodes with a single non-self value input
/// - Phi nodes where all value inputs are identical
#[derive(Debug)]
pub struct CopyProp {
    /// Statistics from the last run.
    stats: CopyPropStats,
    /// Map from copied node to original.
    copies: FxHashMap<NodeId, NodeId>,
    /// Aggressive mode for more thorough analysis.
    aggressive: bool,
}

impl CopyProp {
    /// Create a new copy propagation pass.
    pub fn new() -> Self {
        Self {
            stats: CopyPropStats::default(),
            copies: FxHashMap::default(),
            aggressive: false,
        }
    }

    /// Create an aggressive copy propagation pass.
    ///
    /// Aggressive mode performs:
    /// - Transitive copy chain resolution
    /// - Cross-block copy propagation
    pub fn aggressive() -> Self {
        Self {
            stats: CopyPropStats::default(),
            copies: FxHashMap::default(),
            aggressive: true,
        }
    }

    /// Get statistics from the last run.
    pub fn stats(&self) -> &CopyPropStats {
        &self.stats
    }

    /// Get number of copies found.
    #[inline]
    pub fn copies_found(&self) -> usize {
        self.stats.copies_found
    }

    /// Get number of uses rewritten.
    #[inline]
    pub fn uses_rewritten(&self) -> usize {
        self.stats.uses_rewritten
    }

    /// Run copy propagation on the graph.
    fn run_copy_prop(&mut self, graph: &mut Graph) -> bool {
        self.stats = CopyPropStats::default();
        self.copies.clear();

        // Phase 1: Identify all copies
        self.identify_copies(graph);

        if self.copies.is_empty() {
            return false;
        }

        // Phase 2: Resolve transitive copy chains
        if self.aggressive {
            self.resolve_copy_chains();
        }

        // Phase 3: Rewrite uses
        let changed = self.rewrite_uses(graph);

        changed
    }

    /// Identify all copy-like operations in the graph.
    ///
    /// For Phi nodes, the first input is typically the control region,
    /// so we only look at value inputs starting from index 1.
    fn identify_copies(&mut self, graph: &Graph) {
        for (node_id, node) in graph.iter() {
            let original = match &node.op {
                // Phi with single value input (after region) is a copy
                // Phi structure: inputs[0] = region, inputs[1..] = values
                Operator::Phi if node.inputs.len() == 2 => {
                    // Single value input after region
                    self.stats.phis_simplified += 1;
                    match node.inputs.get(1) {
                        Some(val) => val,
                        None => continue,
                    }
                }

                // Phi where all value inputs are the same (excluding region and self-references)
                Operator::Phi if node.inputs.len() > 2 => {
                    let first = match node.inputs.get(1) {
                        Some(val) => val,
                        None => continue,
                    };

                    // Check if all value inputs (from index 1) are the same
                    let mut all_same = true;
                    for i in 2..node.inputs.len() {
                        if let Some(input) = node.inputs.get(i) {
                            if input != first && input != node_id {
                                all_same = false;
                                break;
                            }
                        }
                    }

                    if all_same {
                        self.stats.phis_simplified += 1;
                        first
                    } else {
                        continue;
                    }
                }

                // LoopPhi has: inputs[0] = loop header, inputs[1] = initial, inputs[2] = back edge
                // If initial == back edge (excluding self-references), it's a copy
                Operator::LoopPhi if node.inputs.len() >= 2 => {
                    let initial = match node.inputs.get(1) {
                        Some(val) => val,
                        None => continue,
                    };

                    if node.inputs.len() == 2 {
                        // No back edge yet
                        self.stats.phis_simplified += 1;
                        initial
                    } else if node.inputs.len() == 3 {
                        let back = match node.inputs.get(2) {
                            Some(val) => val,
                            None => continue,
                        };
                        if initial == back || back == node_id {
                            self.stats.phis_simplified += 1;
                            initial
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                // Projection(0) with single input is identity
                Operator::Projection(0) if node.inputs.len() == 1 => match node.inputs.get(0) {
                    Some(val) => val,
                    None => continue,
                },

                _ => continue,
            };

            // Don't create self-loops
            if original != node_id {
                self.copies.insert(node_id, original);
                self.stats.copies_found += 1;
            }
        }
    }

    /// Resolve transitive copy chains.
    fn resolve_copy_chains(&mut self) {
        let keys: Vec<NodeId> = self.copies.keys().copied().collect();

        for key in keys {
            let mut current = key;
            let mut chain = Vec::new();

            // Walk the chain
            while let Some(&next) = self.copies.get(&current) {
                if chain.contains(&next) {
                    break;
                }
                chain.push(current);
                current = next;
            }

            // Update all entries in the chain to point to the root
            for node in chain {
                self.copies.insert(node, current);
            }
        }
    }

    /// Rewrite uses of copied values to use originals.
    fn rewrite_uses(&mut self, graph: &mut Graph) -> bool {
        let mut changed = false;
        let node_ids: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();

        for node_id in node_ids {
            if let Some(node) = graph.get(node_id) {
                let mut new_inputs = Vec::with_capacity(node.inputs.len());
                let mut any_changed = false;

                for input in node.inputs.iter() {
                    let resolved = self.resolve_copy(input);
                    if resolved != input {
                        self.stats.uses_rewritten += 1;
                        any_changed = true;
                    }
                    new_inputs.push(resolved);
                }

                if any_changed {
                    if let Some(node_mut) = graph.get_mut(node_id) {
                        node_mut.inputs = InputList::from_slice(&new_inputs);
                    }
                    changed = true;
                }
            }
        }

        // Count eliminated copies
        for &copy in self.copies.keys() {
            if graph.uses(copy).is_empty() {
                self.stats.copies_eliminated += 1;
            }
        }

        changed
    }

    /// Resolve a potentially copied value to its original.
    fn resolve_copy(&self, node: NodeId) -> NodeId {
        self.copies.get(&node).copied().unwrap_or(node)
    }
}

impl Default for CopyProp {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for CopyProp {
    fn name(&self) -> &'static str {
        "copy_prop"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_copy_prop(graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
