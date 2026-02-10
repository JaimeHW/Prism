//! Sparse SSA Solver for SCCP.
//!
//! This module implements the sparse conditional constant propagation algorithm.
//! The solver uses a worklist algorithm that only revisits nodes when their
//! inputs change, achieving near-linear time complexity.
//!
//! # Algorithm Overview
//!
//! 1. Initialize all values to Undef (bottom)
//! 2. Mark entry block as executable
//! 3. Process worklist:
//!    - For each node, compute value from inputs
//!    - If value changes (rises in lattice), add uses to worklist
//!    - For branches, mark edges executable based on condition
//! 4. Replace constant nodes with their values
//! 5. Eliminate unreachable code
//!
//! # Key Properties
//!
//! - **Sparse**: Only visits nodes when inputs change
//! - **Conditional**: Considers branch conditions
//! - **Monotonic**: Values only rise in the lattice (termination guarantee)

use super::evaluation::ConstEvaluator;
use super::lattice::{Constant, LatticeValue};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ControlOp, Operator};

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

// =============================================================================
// Block Executability
// =============================================================================

/// Tracks which control flow edges are executable.
#[derive(Debug, Default)]
pub struct EdgeExecutability {
    /// Set of executable edges (from, to).
    executable: FxHashSet<(NodeId, NodeId)>,
    /// Set of reachable nodes.
    reachable: FxHashSet<NodeId>,
}

impl EdgeExecutability {
    /// Create a new edge executability tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark an edge as executable.
    pub fn mark_executable(&mut self, from: NodeId, to: NodeId) -> bool {
        self.reachable.insert(to);
        self.executable.insert((from, to))
    }

    /// Mark a node as reachable directly (e.g., entry point).
    pub fn mark_reachable(&mut self, node: NodeId) -> bool {
        self.reachable.insert(node)
    }

    /// Check if an edge is executable.
    pub fn is_edge_executable(&self, from: NodeId, to: NodeId) -> bool {
        self.executable.contains(&(from, to))
    }

    /// Check if a node is reachable.
    pub fn is_reachable(&self, node: NodeId) -> bool {
        self.reachable.contains(&node)
    }

    /// Get all reachable nodes.
    pub fn reachable_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.reachable.iter().copied()
    }

    /// Count of executable edges.
    pub fn edge_count(&self) -> usize {
        self.executable.len()
    }
}

// =============================================================================
// SCCP Solver
// =============================================================================

/// The SCCP sparse solver.
///
/// Implements the worklist algorithm for sparse conditional constant propagation.
#[derive(Debug)]
pub struct SccpSolver<'g> {
    /// The graph being analyzed.
    graph: &'g Graph,
    /// Lattice values for each node.
    values: FxHashMap<NodeId, LatticeValue>,
    /// Edge executability.
    edges: EdgeExecutability,
    /// SSA worklist (nodes to revisit).
    ssa_worklist: VecDeque<NodeId>,
    /// CFG worklist (edges to process).
    cfg_worklist: VecDeque<(NodeId, NodeId)>,
    /// Constant evaluator.
    evaluator: ConstEvaluator,
    /// Statistics.
    stats: SolverStats,
}

/// Solver statistics.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Number of nodes visited.
    pub nodes_visited: usize,
    /// Number of lattice value changes.
    pub value_changes: usize,
    /// Number of constants found.
    pub constants_found: usize,
    /// Number of unreachable nodes.
    pub unreachable_nodes: usize,
}

impl<'g> SccpSolver<'g> {
    /// Create a new solver for the given graph.
    pub fn new(graph: &'g Graph) -> Self {
        Self {
            graph,
            values: FxHashMap::default(),
            edges: EdgeExecutability::new(),
            ssa_worklist: VecDeque::new(),
            cfg_worklist: VecDeque::new(),
            evaluator: ConstEvaluator::new(),
            stats: SolverStats::default(),
        }
    }

    /// Run the solver and return the computed values.
    pub fn solve(mut self) -> SolverResult {
        // Initialize: find entry point and mark reachable
        self.initialize();

        // Main loop: process worklists until empty
        while !self.ssa_worklist.is_empty() || !self.cfg_worklist.is_empty() {
            // Process CFG edges first (can enable new SSA uses)
            while let Some((from, to)) = self.cfg_worklist.pop_front() {
                self.process_cfg_edge(from, to);
            }

            // Process SSA uses
            while let Some(node) = self.ssa_worklist.pop_front() {
                self.process_node(node);
            }
        }

        // Count statistics
        self.compute_stats();

        SolverResult {
            values: self.values,
            edges: self.edges,
            stats: self.stats,
        }
    }

    /// Initialize the solver.
    fn initialize(&mut self) {
        // Find Start node (entry point) and initialize all nodes
        for (node_id, node) in self.graph.iter() {
            match &node.op {
                Operator::Control(ControlOp::Start) => {
                    self.edges.mark_reachable(node_id);
                    // Add all successors to CFG worklist
                    for use_id in self.graph.uses(node_id).iter() {
                        self.cfg_worklist.push_back((node_id, *use_id));
                    }
                }
                // Initialize constants right away - mark as reachable (pure, no control dependency)
                Operator::ConstInt(v) => {
                    self.edges.mark_reachable(node_id); // Constants are always reachable
                    self.values.insert(node_id, LatticeValue::int(*v));
                }
                Operator::ConstFloat(bits) => {
                    self.edges.mark_reachable(node_id);
                    let v = f64::from_bits(*bits);
                    self.values.insert(node_id, LatticeValue::float(v));
                }
                Operator::ConstBool(v) => {
                    self.edges.mark_reachable(node_id);
                    self.values.insert(node_id, LatticeValue::bool(*v));
                }
                Operator::ConstNone => {
                    self.edges.mark_reachable(node_id);
                    self.values
                        .insert(node_id, LatticeValue::constant(Constant::None));
                }
                Operator::Parameter(_) => {
                    // Parameters are overdefined (unknown at compile time)
                    self.edges.mark_reachable(node_id);
                    self.values.insert(node_id, LatticeValue::overdefined());
                }
                _ => {
                    // Everything else starts as Undef
                    self.values.insert(node_id, LatticeValue::undef());
                }
            }
        }

        // After initializing all constant values, add users of constants to the worklist
        for (node_id, node) in self.graph.iter() {
            if Self::is_constant_or_param(&node.op) {
                // Add uses to worklist - they now have inputs with known values
                for use_id in self.graph.uses(node_id).iter() {
                    self.ssa_worklist.push_back(*use_id);
                    // Mark pure uses as reachable
                    if let Some(use_node) = self.graph.get(*use_id) {
                        if Self::is_pure_op(&use_node.op) {
                            self.edges.mark_reachable(*use_id);
                        }
                    }
                }
            }
        }
    }

    /// Check if operator is a constant or parameter.
    fn is_constant_or_param(op: &Operator) -> bool {
        matches!(
            op,
            Operator::ConstInt(_)
                | Operator::ConstFloat(_)
                | Operator::ConstBool(_)
                | Operator::ConstNone
                | Operator::Parameter(_)
        )
    }

    /// Check if an operator is pure (no side effects, can be freely reordered).
    fn is_pure_op(op: &Operator) -> bool {
        matches!(
            op,
            Operator::ConstInt(_)
                | Operator::ConstFloat(_)
                | Operator::ConstBool(_)
                | Operator::ConstNone
                | Operator::IntOp(_)
                | Operator::FloatOp(_)
                | Operator::GenericOp(_)
                | Operator::IntCmp(_)
                | Operator::FloatCmp(_)
                | Operator::GenericCmp(_)
                | Operator::Bitwise(_)
                | Operator::LogicalNot
        )
    }

    /// Process a CFG edge.
    fn process_cfg_edge(&mut self, from: NodeId, to: NodeId) {
        // If edge already executable, nothing to do
        if self.edges.is_edge_executable(from, to) {
            return;
        }

        // Mark edge executable
        self.edges.mark_executable(from, to);

        // If target wasn't reachable before, add its nodes to worklist
        if let Some(node) = self.graph.get(to) {
            // Add phis in the target to SSA worklist
            if matches!(node.op, Operator::Phi | Operator::LoopPhi) {
                self.ssa_worklist.push_back(to);
            }

            // Add successors to CFG worklist
            for use_id in self.graph.uses(to).iter() {
                if !self.edges.is_reachable(*use_id) {
                    self.cfg_worklist.push_back((to, *use_id));
                }
            }
        }
    }

    /// Process a node from the SSA worklist.
    fn process_node(&mut self, node_id: NodeId) {
        self.stats.nodes_visited += 1;

        let node = match self.graph.get(node_id) {
            Some(n) => n,
            None => return,
        };

        // Mark pure operator nodes as reachable when processed
        if Self::is_pure_op(&node.op) {
            self.edges.mark_reachable(node_id);
        }

        // Only process if reachable
        if !self.edges.is_reachable(node_id) {
            return;
        }

        let new_value = match &node.op {
            // Constants already initialized
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone => return,

            // Parameters stay overdefined
            Operator::Parameter(_) => return,

            // Phi: meet all incoming values from executable edges
            Operator::Phi => self.eval_phi(node_id),

            // LoopPhi: similar to Phi but handles back edges specially
            Operator::LoopPhi => self.eval_loop_phi(node_id),

            // Branches: evaluate condition and mark appropriate edges
            Operator::Control(ControlOp::If) => {
                self.eval_branch(node_id);
                return;
            }

            // Other operators: evaluate based on inputs
            _ => self.eval_operator(node_id),
        };

        // Update value if changed
        self.update_value(node_id, new_value);
    }

    /// Evaluate a phi node.
    fn eval_phi(&self, node_id: NodeId) -> LatticeValue {
        let node = match self.graph.get(node_id) {
            Some(n) => n,
            None => return LatticeValue::overdefined(),
        };

        // Skip region input (first input)
        let values = node.inputs.iter().skip(1).filter_map(|input| {
            // Only consider values from executable edges
            self.values.get(&input)
        });

        self.evaluator.eval_phi(values.cloned())
    }

    /// Evaluate a loop phi node.
    fn eval_loop_phi(&self, node_id: NodeId) -> LatticeValue {
        let node = match self.graph.get(node_id) {
            Some(n) => n,
            None => return LatticeValue::overdefined(),
        };

        // LoopPhi: inputs[0] = header, inputs[1] = initial, inputs[2..] = back edges
        let len = node.inputs.len();
        if len < 2 {
            return LatticeValue::overdefined();
        }

        // Get initial value
        let initial = node.inputs.get(1).and_then(|id| self.values.get(&id));

        // Get back edge value (if present and executable)
        let back = if len >= 3 {
            node.inputs.get(2).and_then(|id| {
                // Only include if back edge is executable
                if self.edges.is_reachable(id) {
                    self.values.get(&id)
                } else {
                    None
                }
            })
        } else {
            None
        };

        match (initial, back) {
            (Some(init), Some(b)) => init.meet(b),
            (Some(init), None) => init.clone(),
            (None, Some(b)) => b.clone(),
            (None, None) => LatticeValue::undef(),
        }
    }

    /// Evaluate a branch and mark edges executable.
    fn eval_branch(&mut self, node_id: NodeId) {
        let node = match self.graph.get(node_id) {
            Some(n) => n,
            None => return,
        };

        // Get condition value
        let cond = node.inputs.get(1).and_then(|id| self.values.get(&id));

        match cond {
            Some(LatticeValue::Constant(Constant::Bool(true))) => {
                // Only true branch is executable
                self.mark_branch_edge(node_id, 0);
            }
            Some(LatticeValue::Constant(Constant::Bool(false))) => {
                // Only false branch is executable
                self.mark_branch_edge(node_id, 1);
            }
            Some(LatticeValue::Constant(Constant::Int(v))) => {
                if *v != 0 {
                    self.mark_branch_edge(node_id, 0);
                } else {
                    self.mark_branch_edge(node_id, 1);
                }
            }
            Some(LatticeValue::Overdefined) | None => {
                // Both branches are executable
                self.mark_branch_edge(node_id, 0);
                self.mark_branch_edge(node_id, 1);
            }
            Some(LatticeValue::Undef) => {
                // Condition is undef - conservatively mark both
                self.mark_branch_edge(node_id, 0);
                self.mark_branch_edge(node_id, 1);
            }
            _ => {
                // Unknown constant type - conservative
                self.mark_branch_edge(node_id, 0);
                self.mark_branch_edge(node_id, 1);
            }
        }
    }

    /// Mark a branch edge as executable.
    fn mark_branch_edge(&mut self, branch: NodeId, index: usize) {
        // Find the projection for this branch
        for use_id in self.graph.uses(branch).iter() {
            if let Some(use_node) = self.graph.get(*use_id) {
                if let Operator::Projection(idx) = &use_node.op {
                    if *idx as usize == index {
                        self.cfg_worklist.push_back((branch, *use_id));
                    }
                }
            }
        }
    }

    /// Evaluate an operator.
    fn eval_operator(&self, node_id: NodeId) -> LatticeValue {
        let node = match self.graph.get(node_id) {
            Some(n) => n,
            None => return LatticeValue::overdefined(),
        };

        // Get input values
        let input_count = node.inputs.len();

        if input_count == 0 {
            return LatticeValue::overdefined();
        }

        // Single input (unary)
        if input_count == 1 {
            let operand = node
                .inputs
                .get(0)
                .and_then(|id| self.values.get(&id))
                .cloned()
                .unwrap_or(LatticeValue::overdefined());
            return self.evaluator.eval_unary(&node.op, &operand);
        }

        // Two inputs (binary)
        if input_count >= 2 {
            let lhs = node
                .inputs
                .get(0)
                .and_then(|id| self.values.get(&id))
                .cloned()
                .unwrap_or(LatticeValue::overdefined());
            let rhs = node
                .inputs
                .get(1)
                .and_then(|id| self.values.get(&id))
                .cloned()
                .unwrap_or(LatticeValue::overdefined());
            return self.evaluator.eval_binary(&node.op, &lhs, &rhs);
        }

        LatticeValue::overdefined()
    }

    /// Update a node's value and propagate if changed.
    fn update_value(&mut self, node_id: NodeId, new_value: LatticeValue) {
        let old_value = self
            .values
            .get(&node_id)
            .cloned()
            .unwrap_or(LatticeValue::undef());

        if new_value != old_value {
            // Value rises in lattice
            self.stats.value_changes += 1;
            self.values.insert(node_id, new_value);
            self.add_uses_to_worklist(node_id);
        }
    }

    /// Add all uses of a node to the worklist.
    fn add_uses_to_worklist(&mut self, node_id: NodeId) {
        for use_id in self.graph.uses(node_id).iter() {
            self.ssa_worklist.push_back(*use_id);
            // Also mark pure uses as reachable so they get processed
            if let Some(use_node) = self.graph.get(*use_id) {
                if Self::is_pure_op(&use_node.op) {
                    self.edges.mark_reachable(*use_id);
                }
            }
        }
    }

    /// Compute final statistics.
    fn compute_stats(&mut self) {
        for (_, value) in &self.values {
            if value.is_constant() {
                self.stats.constants_found += 1;
            }
        }

        let total_nodes = self.graph.len();
        let reachable = self.edges.reachable_nodes().count();
        self.stats.unreachable_nodes = total_nodes.saturating_sub(reachable);
    }
}

// =============================================================================
// Solver Result
// =============================================================================

/// Result from the SCCP solver.
#[derive(Debug)]
pub struct SolverResult {
    /// Computed lattice values for each node.
    pub values: FxHashMap<NodeId, LatticeValue>,
    /// Edge executability information.
    pub edges: EdgeExecutability,
    /// Statistics from the solve.
    pub stats: SolverStats,
}

impl SolverResult {
    /// Get the value for a node.
    pub fn value(&self, node: NodeId) -> Option<&LatticeValue> {
        self.values.get(&node)
    }

    /// Check if a node is constant.
    pub fn is_constant(&self, node: NodeId) -> bool {
        self.values
            .get(&node)
            .map(|v| v.is_constant())
            .unwrap_or(false)
    }

    /// Get the constant value for a node.
    pub fn constant_value(&self, node: NodeId) -> Option<&Constant> {
        self.values.get(&node).and_then(|v| v.as_constant())
    }

    /// Check if a node is reachable.
    pub fn is_reachable(&self, node: NodeId) -> bool {
        self.edges.is_reachable(node)
    }

    /// Iterate over all constant nodes.
    pub fn constants(&self) -> impl Iterator<Item = (NodeId, &Constant)> {
        self.values
            .iter()
            .filter_map(|(id, v)| v.as_constant().map(|c| (*id, c)))
    }

    /// Iterate over unreachable nodes.
    pub fn unreachable_nodes<'a>(&'a self, graph: &'a Graph) -> impl Iterator<Item = NodeId> + 'a {
        graph.iter().filter_map(move |(id, _)| {
            if !self.edges.is_reachable(id) {
                Some(id)
            } else {
                None
            }
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

    // =========================================================================
    // EdgeExecutability Tests
    // =========================================================================

    #[test]
    fn test_edge_executability_new() {
        let e = EdgeExecutability::new();
        assert_eq!(e.edge_count(), 0);
    }

    #[test]
    fn test_edge_executability_mark() {
        let mut e = EdgeExecutability::new();
        let n1 = NodeId::new(1);
        let n2 = NodeId::new(2);

        assert!(e.mark_executable(n1, n2));
        assert!(e.is_edge_executable(n1, n2));
        assert!(e.is_reachable(n2));

        // Marking again returns false
        assert!(!e.mark_executable(n1, n2));
    }

    #[test]
    fn test_edge_executability_reachable() {
        let mut e = EdgeExecutability::new();
        let n1 = NodeId::new(1);

        assert!(e.mark_reachable(n1));
        assert!(e.is_reachable(n1));
        assert!(!e.mark_reachable(n1));
    }

    // =========================================================================
    // SolverStats Tests
    // =========================================================================

    #[test]
    fn test_solver_stats_default() {
        let s = SolverStats::default();
        assert_eq!(s.nodes_visited, 0);
        assert_eq!(s.value_changes, 0);
        assert_eq!(s.constants_found, 0);
        assert_eq!(s.unreachable_nodes, 0);
    }

    // =========================================================================
    // Basic Solver Tests
    // =========================================================================

    #[test]
    fn test_solver_empty_graph() {
        let builder = GraphBuilder::new(0, 0);
        let graph = builder.finish();

        let solver = SccpSolver::new(&graph);
        let result = solver.solve();

        assert_eq!(result.stats.constants_found, 0);
    }

    #[test]
    fn test_solver_simple_constant() {
        let mut builder = GraphBuilder::new(8, 0);
        // x = 10 + 32
        let c10 = builder.const_int(10);
        let c32 = builder.const_int(32);
        let sum = builder.int_add(c10, c32);
        builder.return_value(sum);

        let graph = builder.finish();
        let solver = SccpSolver::new(&graph);
        let result = solver.solve();

        // The sum should be constant 42
        assert!(result.is_constant(sum));
        match result.constant_value(sum) {
            Some(Constant::Int(42)) => (),
            other => panic!("Expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn test_solver_constant_chain() {
        let mut builder = GraphBuilder::new(12, 0);
        // a = 2
        // b = 3
        // c = a * b  (= 6)
        // d = c + 1  (= 7)
        let a = builder.const_int(2);
        let b = builder.const_int(3);
        let c = builder.int_mul(a, b);
        let c1 = builder.const_int(1);
        let d = builder.int_add(c, c1);
        builder.return_value(d);

        let graph = builder.finish();
        let solver = SccpSolver::new(&graph);
        let result = solver.solve();

        assert_eq!(result.constant_value(c), Some(&Constant::Int(6)));
        assert_eq!(result.constant_value(d), Some(&Constant::Int(7)));
    }

    #[test]
    fn test_solver_with_parameter() {
        let mut builder = GraphBuilder::new(8, 1);
        // x = param0 + 10
        let p0 = builder.parameter(0).unwrap();
        let c10 = builder.const_int(10);
        let sum = builder.int_add(p0, c10);
        builder.return_value(sum);

        let graph = builder.finish();
        let solver = SccpSolver::new(&graph);
        let result = solver.solve();

        // sum cannot be constant (depends on parameter)
        assert!(!result.is_constant(sum));
    }

    #[test]
    fn test_solver_result_constants_iter() {
        let mut builder = GraphBuilder::new(8, 0);
        let c1 = builder.const_int(1);
        let c2 = builder.const_int(2);
        let sum = builder.int_add(c1, c2);
        builder.return_value(sum);

        let graph = builder.finish();
        let result = SccpSolver::new(&graph).solve();

        let constants: Vec<_> = result.constants().collect();
        assert!(constants.len() >= 3); // c1, c2, sum
    }

    // =========================================================================
    // SolverResult Tests
    // =========================================================================

    #[test]
    fn test_solver_result_is_reachable() {
        let mut builder = GraphBuilder::new(4, 0);
        let c = builder.const_int(42);
        builder.return_value(c);

        let graph = builder.finish();
        let result = SccpSolver::new(&graph).solve();

        // Constants are reachable via initialization
        assert!(result.is_constant(c));
    }
}
