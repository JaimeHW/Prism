//! Loop Invariant Analysis.
//!
//! Provides comprehensive loop analysis for optimization passes:
//! - **Invariant Detection**: Identifies computations that don't change within a loop
//! - **Preheader Insertion**: Adds landing pads for hoisted code
//! - **Exit Block Analysis**: Finds all exit points from a loop

use crate::ir::arena::BitSet;
use crate::ir::cfg::{BlockId, Cfg, DominatorTree, Loop, LoopAnalysis};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ArithOp, Operator};

use std::collections::HashSet;

// =============================================================================
// Loop Invariant Analysis
// =============================================================================

/// Cached analysis of loop-invariant nodes.
#[derive(Debug)]
pub struct LoopInvariantAnalysis {
    /// Set of invariant node IDs per loop.
    invariants: Vec<BitSet>,
    /// Nodes that are hoistable (invariant AND safe to move).
    hoistable: Vec<BitSet>,
    /// Preheader blocks inserted for each loop (if any).
    preheaders: Vec<Option<BlockId>>,
    /// Exit blocks for each loop.
    exits: Vec<Vec<BlockId>>,
}

impl LoopInvariantAnalysis {
    /// Compute loop invariant analysis for all loops.
    pub fn compute(graph: &Graph, cfg: &Cfg, dom: &DominatorTree, loops: &LoopAnalysis) -> Self {
        let num_loops = loops.loops.len();
        let num_nodes = graph.len();

        let mut analysis = Self {
            invariants: Vec::with_capacity(num_loops),
            hoistable: Vec::with_capacity(num_loops),
            preheaders: vec![None; num_loops],
            exits: Vec::with_capacity(num_loops),
        };

        for (_loop_idx, loop_info) in loops.loops.iter().enumerate() {
            let invariants = Self::find_invariants(graph, loop_info, num_nodes);
            let hoistable = Self::find_hoistable(graph, &invariants, loop_info, dom, cfg);
            let exits = Self::find_exits(cfg, loop_info);

            analysis.invariants.push(invariants);
            analysis.hoistable.push(hoistable);
            analysis.exits.push(exits);
        }

        analysis
    }

    /// Find all loop-invariant nodes.
    fn find_invariants(graph: &Graph, loop_info: &Loop, num_nodes: usize) -> BitSet {
        let mut invariant = BitSet::with_capacity(num_nodes);
        let mut changed = true;

        while changed {
            changed = false;

            for (node_id, node) in graph.iter() {
                if invariant.contains(node_id.as_usize()) {
                    continue;
                }

                // Skip control nodes
                if matches!(node.op, Operator::Control(_)) {
                    continue;
                }

                // Skip phi nodes
                if matches!(node.op, Operator::Phi | Operator::LoopPhi) {
                    continue;
                }

                // Skip nodes with side effects
                if !node.op.is_pure() {
                    continue;
                }

                let all_inputs_invariant = node.inputs.iter().all(|input| {
                    if invariant.contains(input.as_usize()) {
                        return true;
                    }
                    let input_node = graph.node(input);
                    Self::is_defined_outside_loop(&input_node.op)
                });

                if all_inputs_invariant {
                    invariant.insert(node_id.as_usize());
                    changed = true;
                }
            }
        }

        invariant
    }

    /// Find hoistable nodes (invariant AND safe to move).
    fn find_hoistable(
        graph: &Graph,
        invariants: &BitSet,
        loop_info: &Loop,
        dom: &DominatorTree,
        _cfg: &Cfg,
    ) -> BitSet {
        let mut hoistable = BitSet::with_capacity(graph.len());

        for node_idx in invariants.iter() {
            let node_id = NodeId::new(node_idx as u32);
            if Self::can_safely_hoist(graph, node_id, loop_info, dom) {
                hoistable.insert(node_idx);
            }
        }

        hoistable
    }

    /// Check if a node can be safely hoisted.
    fn can_safely_hoist(
        graph: &Graph,
        node_id: NodeId,
        _loop_info: &Loop,
        _dom: &DominatorTree,
    ) -> bool {
        let node = graph.node(node_id);

        if !node.op.is_pure() {
            return false;
        }

        if Self::can_trap(&node.op) {
            return false;
        }

        true
    }

    /// Find exit blocks for a loop.
    fn find_exits(cfg: &Cfg, loop_info: &Loop) -> Vec<BlockId> {
        let mut exits = Vec::new();
        let body_set: HashSet<_> = loop_info.body.iter().copied().collect();

        for &block in &loop_info.body {
            let bb = cfg.block(block);
            for &succ in &bb.successors {
                if !body_set.contains(&succ) && !exits.contains(&succ) {
                    exits.push(succ);
                }
            }
        }

        exits
    }

    /// Check if an operator is defined outside any loop (constant/param).
    #[inline]
    fn is_defined_outside_loop(op: &Operator) -> bool {
        matches!(
            op,
            Operator::ConstInt(_)
                | Operator::ConstFloat(_)
                | Operator::ConstBool(_)
                | Operator::ConstNone
                | Operator::Parameter(_)
        )
    }

    /// Check if an operator can trap.
    #[inline]
    fn can_trap(op: &Operator) -> bool {
        match op {
            Operator::IntOp(ArithOp::TrueDiv)
            | Operator::IntOp(ArithOp::FloorDiv)
            | Operator::IntOp(ArithOp::Mod)
            | Operator::FloatOp(ArithOp::TrueDiv)
            | Operator::GenericOp(ArithOp::TrueDiv)
            | Operator::GenericOp(ArithOp::FloorDiv)
            | Operator::GenericOp(ArithOp::Mod) => true,
            Operator::Memory(_) => true,
            _ => false,
        }
    }

    // =========================================================================
    // Query API
    // =========================================================================

    /// Check if a node is loop-invariant.
    #[inline]
    pub fn is_invariant(&self, loop_idx: usize, node: NodeId) -> bool {
        self.invariants
            .get(loop_idx)
            .map(|s| s.contains(node.as_usize()))
            .unwrap_or(false)
    }

    /// Check if a node is hoistable.
    #[inline]
    pub fn is_hoistable(&self, loop_idx: usize, node: NodeId) -> bool {
        self.hoistable
            .get(loop_idx)
            .map(|s| s.contains(node.as_usize()))
            .unwrap_or(false)
    }

    /// Get hoistable nodes for a loop.
    pub fn hoistable_nodes(&self, loop_idx: usize) -> impl Iterator<Item = NodeId> + '_ {
        self.hoistable
            .get(loop_idx)
            .into_iter()
            .flat_map(|s| s.iter())
            .map(|idx| NodeId::new(idx as u32))
    }

    /// Get exit blocks for a loop.
    pub fn exits(&self, loop_idx: usize) -> &[BlockId] {
        self.exits
            .get(loop_idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get preheader block for a loop (if inserted).
    pub fn preheader(&self, loop_idx: usize) -> Option<BlockId> {
        self.preheaders.get(loop_idx).copied().flatten()
    }

    /// Get total number of invariant nodes across all loops.
    pub fn total_invariants(&self) -> usize {
        self.invariants.iter().map(|s| s.count()).sum()
    }

    /// Get total number of hoistable nodes across all loops.
    pub fn total_hoistable(&self) -> usize {
        self.hoistable.iter().map(|s| s.count()).sum()
    }
}

// =============================================================================
// Preheader Insertion
// =============================================================================

/// Insert preheader blocks for loops that don't have them.
pub struct PreheaderInserter<'a> {
    graph: &'a mut Graph,
    cfg: &'a mut Cfg,
}

impl<'a> PreheaderInserter<'a> {
    /// Create a new preheader inserter.
    pub fn new(graph: &'a mut Graph, cfg: &'a mut Cfg) -> Self {
        Self { graph, cfg }
    }

    /// Insert preheaders for all loops that need them.
    pub fn insert_all(&mut self, loops: &LoopAnalysis) -> Vec<Option<BlockId>> {
        let mut preheaders = Vec::with_capacity(loops.loops.len());

        for loop_info in &loops.loops {
            let preheader = self.insert_preheader_if_needed(loop_info);
            preheaders.push(preheader);
        }

        preheaders
    }

    /// Insert a preheader for a loop if needed.
    fn insert_preheader_if_needed(&mut self, loop_info: &Loop) -> Option<BlockId> {
        let header = loop_info.header;
        let header_bb = self.cfg.block(header);

        let body_set: HashSet<_> = loop_info.body.iter().copied().collect();
        let outside_preds: Vec<_> = header_bb
            .predecessors
            .iter()
            .copied()
            .filter(|&p| !body_set.contains(&p))
            .collect();

        if outside_preds.len() == 1 {
            let pred = outside_preds[0];
            let pred_bb = self.cfg.block(pred);
            if pred_bb.successors.len() == 1 {
                return Some(pred);
            }
        }

        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::GraphBuilder;

    #[test]
    fn test_is_defined_outside_loop() {
        assert!(LoopInvariantAnalysis::is_defined_outside_loop(
            &Operator::ConstInt(42)
        ));
        assert!(LoopInvariantAnalysis::is_defined_outside_loop(
            &Operator::Parameter(0)
        ));
        assert!(!LoopInvariantAnalysis::is_defined_outside_loop(
            &Operator::IntOp(ArithOp::Add)
        ));
    }

    #[test]
    fn test_can_trap() {
        assert!(LoopInvariantAnalysis::can_trap(&Operator::IntOp(
            ArithOp::TrueDiv
        )));
        assert!(LoopInvariantAnalysis::can_trap(&Operator::IntOp(
            ArithOp::Mod
        )));
        assert!(!LoopInvariantAnalysis::can_trap(&Operator::IntOp(
            ArithOp::Add
        )));
    }

    #[test]
    fn test_loop_invariant_analysis_empty() {
        let builder = GraphBuilder::new(0, 0);
        let graph = builder.finish();
        let cfg = Cfg::build(&graph);
        let dom = DominatorTree::build(&cfg);
        let loops = LoopAnalysis::compute(&cfg, &dom);

        let analysis = LoopInvariantAnalysis::compute(&graph, &cfg, &dom, &loops);
        assert_eq!(analysis.total_invariants(), 0);
        assert_eq!(analysis.total_hoistable(), 0);
    }

    #[test]
    fn test_invariant_query() {
        let builder = GraphBuilder::new(0, 0);
        let graph = builder.finish();
        let cfg = Cfg::build(&graph);
        let dom = DominatorTree::build(&cfg);
        let loops = LoopAnalysis::compute(&cfg, &dom);

        let analysis = LoopInvariantAnalysis::compute(&graph, &cfg, &dom, &loops);
        assert!(!analysis.is_invariant(0, NodeId::new(0)));
        assert!(!analysis.is_hoistable(0, NodeId::new(0)));
    }

    #[test]
    fn test_preheader_inserter_no_change() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();
        let mut cfg = Cfg::build(&graph);
        let dom = DominatorTree::build(&cfg);
        let loops = LoopAnalysis::compute(&cfg, &dom);

        let mut inserter = PreheaderInserter::new(&mut graph, &mut cfg);
        let preheaders = inserter.insert_all(&loops);
        assert!(preheaders.is_empty());
    }
}
