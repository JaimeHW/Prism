//! Loop Body Transformation for Unrolling.
//!
//! This module implements the actual transformation logic for loop unrolling:
//! - **Graph cloning**: Duplicate loop body nodes
//! - **Phi resolution**: Update phi nodes for multiple iterations
//! - **Induction update**: Modify induction variable updates
//! - **Control flow rewiring**: Connect unrolled iterations
//!
//! # Full Unrolling
//!
//! Complete elimination of the loop by replicating the body N times
//! and replacing the loop header with straight-line code.
//!
//! # Partial Unrolling
//!
//! Replicate the body M times while keeping the loop structure,
//! reducing the number of back edges executed.

use super::analysis::UnrollabilityAnalysis;
use super::remainder::{RemainderGenerator, RemainderResult, RemainderStrategy};

use crate::ir::cfg::Cfg;
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{ArithOp, Operator};

use rustc_hash::FxHashMap;

// =============================================================================
// Loop Unroller
// =============================================================================

/// Main loop unroller implementation.
#[derive(Debug)]
pub struct LoopUnroller<'a> {
    graph: &'a mut Graph,
    cfg: &'a mut Cfg,
    /// Node mapping: original -> copies per iteration
    node_copies: Vec<FxHashMap<NodeId, NodeId>>,
    /// Statistics
    nodes_added: usize,
    nodes_removed: usize,
}

impl<'a> LoopUnroller<'a> {
    /// Create a new loop unroller.
    pub fn new(graph: &'a mut Graph, cfg: &'a mut Cfg) -> Self {
        Self {
            graph,
            cfg,
            node_copies: Vec::new(),
            nodes_added: 0,
            nodes_removed: 0,
        }
    }

    /// Fully unroll a loop.
    pub fn full_unroll(&mut self, analysis: &UnrollabilityAnalysis, trip_count: u32) -> bool {
        if trip_count == 0 {
            return self.eliminate_loop(analysis);
        }

        // Collect loop body nodes
        let body_nodes: Vec<NodeId> = analysis.body_nodes.iter().copied().collect();
        if body_nodes.is_empty() {
            return false;
        }

        // Create copies for each iteration
        self.node_copies.clear();
        for _ in 0..trip_count {
            let copy_map = self.clone_body(&body_nodes);
            self.node_copies.push(copy_map);
        }

        // Chain the iterations together
        self.chain_full_unroll_iterations(trip_count);

        // Remove the original loop structure
        self.remove_loop_structure(analysis);

        true
    }

    /// Partially unroll a loop.
    pub fn partial_unroll(
        &mut self,
        analysis: &UnrollabilityAnalysis,
        factor: u32,
        remainder: RemainderStrategy,
    ) -> bool {
        if factor <= 1 {
            return false;
        }

        // Collect loop body nodes
        let body_nodes: Vec<NodeId> = analysis.body_nodes.iter().copied().collect();
        if body_nodes.is_empty() {
            return false;
        }

        // Create copies for the unroll factor
        self.node_copies.clear();

        // First iteration reuses original body (implicit copy at index 0)
        let mut identity_map = FxHashMap::default();
        for &node_id in &body_nodes {
            identity_map.insert(node_id, node_id);
        }
        self.node_copies.push(identity_map);

        // Additional iterations are cloned
        for _ in 1..factor {
            let copy_map = self.clone_body(&body_nodes);
            self.node_copies.push(copy_map);
        }

        // Chain the iterations
        self.chain_partial_unroll_iterations(factor, analysis);

        // Update the loop condition
        self.update_loop_condition(factor, analysis);

        // Generate remainder handling
        if remainder != RemainderStrategy::None {
            self.generate_remainder(analysis, factor, remainder);
        }

        true
    }

    /// Eliminate a zero-trip loop.
    fn eliminate_loop(&mut self, analysis: &UnrollabilityAnalysis) -> bool {
        // Simply remove all loop body nodes and redirect control flow
        self.remove_loop_structure(analysis);
        true
    }

    /// Clone the loop body nodes.
    fn clone_body(&mut self, body_nodes: &[NodeId]) -> FxHashMap<NodeId, NodeId> {
        let mut node_map = FxHashMap::default();

        // First pass: create all nodes with placeholder inputs
        for &orig_id in body_nodes {
            if let Some(orig_node) = self.graph.get(orig_id) {
                // Clone the operator
                let new_op = orig_node.op.clone();
                // Create with same inputs initially (will fix up in second pass)
                let new_id = self.graph.add_node(new_op, orig_node.inputs.clone());
                node_map.insert(orig_id, new_id);
                self.nodes_added += 1;
            }
        }

        // Second pass: collect input remappings, then apply them
        let mut input_replacements: Vec<(NodeId, usize, NodeId)> = Vec::new();

        for &orig_id in body_nodes {
            if let Some(&new_id) = node_map.get(&orig_id) {
                if let Some(orig_node) = self.graph.get(orig_id) {
                    // Map inputs to their copies (or keep original if outside loop)
                    for (idx, inp) in orig_node.inputs.iter().enumerate() {
                        let mapped_inp = *node_map.get(&inp).unwrap_or(&inp);
                        if mapped_inp != inp {
                            input_replacements.push((new_id, idx, mapped_inp));
                        }
                    }
                }
            }
        }

        // Apply all collected replacements
        for (node_id, idx, new_input) in input_replacements {
            self.graph.replace_input(node_id, idx, new_input);
        }

        node_map
    }

    /// Chain iterations for full unrolling.
    fn chain_full_unroll_iterations(&mut self, trip_count: u32) {
        // Connect each iteration's outputs to the next iteration's inputs
        // For phi nodes, replace with direct values

        // Collect all replacements first to avoid borrow issues
        let mut replacements: Vec<(NodeId, NodeId)> = Vec::new();

        for iter in 1..trip_count as usize {
            let prev_map = &self.node_copies[iter - 1];
            let curr_map = &self.node_copies[iter];

            // For each node in the current iteration
            for (&orig_id, &curr_id) in curr_map {
                if let Some(curr_node) = self.graph.get(curr_id) {
                    if matches!(curr_node.op, Operator::LoopPhi) {
                        // Replace phi with value from previous iteration
                        if let Some(&prev_id) = prev_map.get(&orig_id) {
                            replacements.push((curr_id, prev_id));
                        }
                    }
                }
            }
        }

        // Apply all replacements
        for (phi_id, value_id) in replacements {
            self.replace_phi_with_value(phi_id, value_id);
        }
    }

    /// Chain iterations for partial unrolling.
    fn chain_partial_unroll_iterations(&mut self, factor: u32, _analysis: &UnrollabilityAnalysis) {
        // Similar to full unroll, but we keep the loop structure

        // Collect all input replacements first
        let mut input_replacements: Vec<(NodeId, usize, NodeId)> = Vec::new();

        for iter in 1..factor as usize {
            let prev_map = &self.node_copies[iter - 1];
            let curr_map = &self.node_copies[iter];

            for (&_orig_id, &curr_id) in curr_map {
                if let Some(curr_node) = self.graph.get(curr_id) {
                    // Collect inputs to check
                    let inputs: Vec<(usize, NodeId)> = curr_node
                        .inputs
                        .iter()
                        .enumerate()
                        .map(|(idx, inp)| (idx, inp))
                        .collect();

                    for (idx, inp) in inputs {
                        // If input is a phi in the original, get value from prev iteration
                        let original = self.find_original(&inp);
                        if let Some(&prev_val) = prev_map.get(&original) {
                            if let Some(inp_node) = self.graph.get(inp) {
                                if matches!(inp_node.op, Operator::LoopPhi) {
                                    input_replacements.push((curr_id, idx, prev_val));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply all input replacements
        for (node_id, idx, new_input) in input_replacements {
            self.graph.replace_input(node_id, idx, new_input);
        }
    }

    /// Find the original node ID given a potentially copied ID.
    fn find_original(&self, id: &NodeId) -> NodeId {
        for map in &self.node_copies {
            for (&orig, &copy) in map {
                if copy == *id {
                    return orig;
                }
            }
        }
        *id
    }

    /// Replace a phi node with a direct value.
    fn replace_phi_with_value(&mut self, phi_id: NodeId, value_id: NodeId) {
        // Replace all uses of phi with the value
        self.graph.replace_all_uses(phi_id, value_id);

        // Mark phi as dead
        self.graph.kill(phi_id);
        self.nodes_removed += 1;
    }

    /// Update the loop condition for partial unrolling.
    fn update_loop_condition(&mut self, factor: u32, analysis: &UnrollabilityAnalysis) {
        // Find the induction variable and its update
        if analysis.induction_vars.is_empty() {
            return;
        }

        let iv = analysis.induction_vars[0];

        // Collect the step update info first
        let mut step_update: Option<(NodeId, i64)> = None;

        let uses: Vec<NodeId> = self.graph.uses(iv).to_vec();
        for use_id in uses {
            if let Some(use_node) = self.graph.get(use_id) {
                if let Operator::IntOp(ArithOp::Add) = &use_node.op {
                    // Found the increment - check for constant step
                    if let Some(step_id) = use_node.inputs.get(1) {
                        if let Some(step_node) = self.graph.get(step_id) {
                            if let Operator::ConstInt(step) = &step_node.op {
                                step_update = Some((use_id, *step));
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Apply the step update
        if let Some((use_id, step)) = step_update {
            let new_step = step * factor as i64;
            let new_step_id = self
                .graph
                .add_node(Operator::ConstInt(new_step), InputList::Empty);
            self.graph.replace_input(use_id, 1, new_step_id);
        }
    }

    /// Generate remainder handling.
    fn generate_remainder(
        &mut self,
        analysis: &UnrollabilityAnalysis,
        factor: u32,
        strategy: RemainderStrategy,
    ) -> RemainderResult {
        let body_nodes: Vec<NodeId> = analysis.body_nodes.iter().copied().collect();
        let iv = if analysis.induction_vars.is_empty() {
            self.graph.start
        } else {
            analysis.induction_vars[0]
        };

        // Find the limit (for now, use a placeholder)
        let limit = self.graph.start;

        let mut remainder_gen = RemainderGenerator::new(self.graph, self.cfg);
        remainder_gen.generate(strategy, factor, &body_nodes, iv, limit)
    }

    /// Remove the original loop structure after full unrolling.
    fn remove_loop_structure(&mut self, analysis: &UnrollabilityAnalysis) {
        // Mark all loop control nodes as dead
        for node_id in &analysis.body_nodes {
            if let Some(node) = self.graph.get(*node_id) {
                if matches!(node.op, Operator::Control(_) | Operator::LoopPhi) {
                    self.graph.kill(*node_id);
                    self.nodes_removed += 1;
                }
            }
        }
    }

    /// Get the number of nodes added.
    pub fn nodes_added(&self) -> usize {
        self.nodes_added
    }

    /// Get the number of nodes removed.
    pub fn nodes_removed(&self) -> usize {
        self.nodes_removed
    }
}

// =============================================================================
// Unroll Transform (High-Level API)
// =============================================================================

/// High-level interface for loop unrolling transformations.
pub struct UnrollTransform<'a> {
    unroller: LoopUnroller<'a>,
    analysis: &'a UnrollabilityAnalysis,
}

impl<'a> UnrollTransform<'a> {
    /// Create a new unroll transform.
    pub fn new(
        graph: &'a mut Graph,
        cfg: &'a mut Cfg,
        analysis: &'a UnrollabilityAnalysis,
    ) -> Self {
        Self {
            unroller: LoopUnroller::new(graph, cfg),
            analysis,
        }
    }

    /// Fully unroll the loop.
    pub fn full_unroll(mut self, trip_count: u32) -> bool {
        self.unroller.full_unroll(self.analysis, trip_count)
    }

    /// Partially unroll the loop.
    pub fn partial_unroll(mut self, factor: u32, remainder: RemainderStrategy) -> bool {
        self.unroller
            .partial_unroll(self.analysis, factor, remainder)
    }

    /// Runtime unroll with trip count check.
    pub fn runtime_unroll(
        mut self,
        _min_trip: u32,
        factor: u32,
        remainder: RemainderStrategy,
    ) -> bool {
        // For runtime unrolling, we generate:
        // 1. A trip count check
        // 2. Fast path with partial unroll
        // 3. Slow path with original loop

        // For now, just do partial unroll (full implementation would add the check)
        self.unroller
            .partial_unroll(self.analysis, factor, remainder)
    }
}
