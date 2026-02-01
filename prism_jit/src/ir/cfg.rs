//! Control Flow Graph (CFG) analysis for the Sea-of-Nodes IR.
//!
//! While Sea-of-Nodes unifies data and control flow, we sometimes need
//! explicit CFG structure for:
//! - **Dominator tree computation**: Required for SSA correctness
//! - **Loop analysis**: Detecting natural loops for LICM
//! - **Scheduling**: Placing floating nodes in basic blocks
//!
//! # Structure
//!
//! The CFG is built on-demand from control edges in the graph.
//! Each basic block corresponds to a Region node (control merge point).

use super::arena::{BitSet, Id, SecondaryMap};
use super::graph::Graph;
use super::node::NodeId;
use super::operators::{ControlOp, Operator};

use std::collections::{HashMap, VecDeque};

// =============================================================================
// Basic Block
// =============================================================================

/// A basic block in the CFG.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// The region node that defines this block (control merge).
    pub region: NodeId,

    /// Predecessor blocks.
    pub predecessors: Vec<BlockId>,

    /// Successor blocks.
    pub successors: Vec<BlockId>,

    /// Nodes scheduled in this block (filled by scheduler).
    pub nodes: Vec<NodeId>,

    /// The terminator node (branch, return, etc.).
    pub terminator: Option<NodeId>,

    /// Loop depth (0 = not in loop).
    pub loop_depth: u32,
}

impl BasicBlock {
    /// Create a new basic block.
    fn new(region: NodeId) -> Self {
        BasicBlock {
            region,
            predecessors: Vec::new(),
            successors: Vec::new(),
            nodes: Vec::new(),
            terminator: None,
            loop_depth: 0,
        }
    }
}

/// Block identifier.
pub type BlockId = Id<BasicBlock>;

// =============================================================================
// CFG
// =============================================================================

/// Control Flow Graph extracted from Sea-of-Nodes.
#[derive(Debug, Clone)]
pub struct Cfg {
    /// All basic blocks.
    blocks: Vec<BasicBlock>,

    /// Mapping from region nodes to block IDs.
    region_to_block: HashMap<NodeId, BlockId>,

    /// Entry block (corresponds to Start).
    pub entry: BlockId,

    /// Exit block (corresponds to End).
    pub exit: BlockId,

    /// Reverse postorder traversal (for dataflow).
    pub rpo: Vec<BlockId>,

    /// Postorder numbers for dominance computation.
    pub postorder: SecondaryMap<BasicBlock, u32>,
}

impl Cfg {
    /// Build CFG from a Sea-of-Nodes graph.
    pub fn build(graph: &Graph) -> Self {
        let mut cfg = Cfg {
            blocks: Vec::new(),
            region_to_block: HashMap::new(),
            entry: BlockId::INVALID,
            exit: BlockId::INVALID,
            rpo: Vec::new(),
            postorder: SecondaryMap::new(),
        };

        // Find all control merge points (Start, Region, Loop)
        for (id, node) in graph.iter() {
            if Self::is_block_start(&node.op) {
                let block_id = cfg.add_block(id);
                if matches!(node.op, Operator::Control(ControlOp::Start)) {
                    cfg.entry = block_id;
                }
                if matches!(node.op, Operator::Control(ControlOp::End)) {
                    cfg.exit = block_id;
                }
            }
        }

        // If no explicit End block, create one
        if !cfg.exit.is_valid() {
            cfg.exit = cfg.add_block(graph.end);
        }

        // Build edges by walking control successors
        for (id, node) in graph.iter() {
            if let Some(block_id) = cfg.region_to_block.get(&id).copied() {
                // Find successors by looking at control users
                for &user in graph.uses(id) {
                    let user_node = graph.node(user);
                    if Self::is_block_start(&user_node.op) {
                        if let Some(&succ_id) = cfg.region_to_block.get(&user) {
                            // Add edge
                            cfg.add_edge(block_id, succ_id);
                        }
                    }
                }
            }
        }

        // Compute reverse postorder for dataflow
        cfg.compute_rpo();

        cfg
    }

    /// Check if an operator starts a new block.
    fn is_block_start(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Control(ControlOp::Start)
                | Operator::Control(ControlOp::Region)
                | Operator::Control(ControlOp::Loop)
                | Operator::Control(ControlOp::End)
        )
    }

    /// Add a new block.
    fn add_block(&mut self, region: NodeId) -> BlockId {
        let id = BlockId::new(self.blocks.len() as u32);
        self.blocks.push(BasicBlock::new(region));
        self.region_to_block.insert(region, id);
        id
    }

    /// Add an edge between blocks.
    fn add_edge(&mut self, from: BlockId, to: BlockId) {
        if !self.blocks[from.as_usize()].successors.contains(&to) {
            self.blocks[from.as_usize()].successors.push(to);
        }
        if !self.blocks[to.as_usize()].predecessors.contains(&from) {
            self.blocks[to.as_usize()].predecessors.push(from);
        }
    }

    /// Compute reverse postorder traversal.
    fn compute_rpo(&mut self) {
        if !self.entry.is_valid() {
            return;
        }

        let mut visited = BitSet::with_capacity(self.blocks.len());
        let mut postorder = Vec::with_capacity(self.blocks.len());

        self.dfs_postorder(self.entry, &mut visited, &mut postorder);

        // Assign postorder numbers
        for (i, &block) in postorder.iter().enumerate() {
            self.postorder.set(block, i as u32);
        }

        // Reverse for RPO
        postorder.reverse();
        self.rpo = postorder;
    }

    /// DFS for postorder computation.
    fn dfs_postorder(&self, block: BlockId, visited: &mut BitSet, postorder: &mut Vec<BlockId>) {
        if visited.contains(block.as_usize()) {
            return;
        }
        visited.insert(block.as_usize());

        for &succ in &self.blocks[block.as_usize()].successors {
            self.dfs_postorder(succ, visited, postorder);
        }

        postorder.push(block);
    }

    /// Get a block by ID.
    #[inline]
    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.blocks[id.as_usize()]
    }

    /// Get a mutable block.
    #[inline]
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        &mut self.blocks[id.as_usize()]
    }

    /// Get block for a region node.
    pub fn block_for_region(&self, region: NodeId) -> Option<BlockId> {
        self.region_to_block.get(&region).copied()
    }

    /// Get the number of blocks.
    #[inline]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Iterate over blocks in reverse postorder.
    pub fn iter_rpo(&self) -> impl Iterator<Item = (BlockId, &BasicBlock)> {
        self.rpo
            .iter()
            .copied()
            .map(move |id| (id, &self.blocks[id.as_usize()]))
    }

    /// Iterate over all blocks.
    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &BasicBlock)> {
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (BlockId::new(i as u32), b))
    }
}

// =============================================================================
// Dominator Tree
// =============================================================================

/// Dominator tree for the CFG.
///
/// The dominator tree is used for:
/// - SSA validation
/// - Loop detection
/// - Optimization scheduling
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// Immediate dominator for each block (INVALID for entry).
    idom: SecondaryMap<BasicBlock, BlockId>,

    /// Dominator tree children.
    children: SecondaryMap<BasicBlock, Vec<BlockId>>,

    /// Dominance frontier.
    frontier: SecondaryMap<BasicBlock, Vec<BlockId>>,

    /// Dominator tree depth.
    depth: SecondaryMap<BasicBlock, u32>,
}

impl DominatorTree {
    /// Build dominator tree using the Lengauer-Tarjan algorithm.
    pub fn build(cfg: &Cfg) -> Self {
        let n = cfg.len();

        let mut dom = DominatorTree {
            idom: SecondaryMap::with_capacity(n),
            children: SecondaryMap::with_capacity(n),
            frontier: SecondaryMap::with_capacity(n),
            depth: SecondaryMap::with_capacity(n),
        };

        if n == 0 || !cfg.entry.is_valid() {
            return dom;
        }

        // Initialize idom
        for i in 0..n {
            dom.idom.set(BlockId::new(i as u32), BlockId::INVALID);
        }
        dom.idom.set(cfg.entry, cfg.entry); // Entry dominates itself

        // Cooper-Harvey-Kennedy algorithm (simple, fast enough for JIT)
        let mut changed = true;
        while changed {
            changed = false;

            for &block in &cfg.rpo {
                if block == cfg.entry {
                    continue;
                }

                let preds = &cfg.block(block).predecessors;
                if preds.is_empty() {
                    continue;
                }

                // Find first processed predecessor
                let mut new_idom = BlockId::INVALID;
                for &pred in preds {
                    if dom.idom[pred].is_valid() {
                        new_idom = pred;
                        break;
                    }
                }

                if !new_idom.is_valid() {
                    continue;
                }

                // Intersect with other predecessors
                for &pred in preds {
                    if pred == new_idom {
                        continue;
                    }
                    if dom.idom[pred].is_valid() {
                        new_idom = dom.intersect(pred, new_idom, cfg);
                    }
                }

                if dom.idom[block] != new_idom {
                    dom.idom.set(block, new_idom);
                    changed = true;
                }
            }
        }

        // Build tree children
        for i in 0..n {
            let block = BlockId::new(i as u32);
            if block != cfg.entry {
                let idom = dom.idom[block];
                if idom.is_valid() && idom != block {
                    dom.children.resize(idom.as_usize() + 1);
                    if let Some(children) = dom.children.get_mut(idom) {
                        children.push(block);
                    } else {
                        dom.children.set(idom, vec![block]);
                    }
                }
            }
        }

        // Compute depths
        dom.compute_depths(cfg.entry, 0);

        // Compute dominance frontier
        dom.compute_frontier(cfg);

        dom
    }

    /// Intersect for dominator computation.
    fn intersect(&self, mut b1: BlockId, mut b2: BlockId, cfg: &Cfg) -> BlockId {
        while b1 != b2 {
            let po1 = cfg.postorder.get(b1).copied().unwrap_or(0);
            let po2 = cfg.postorder.get(b2).copied().unwrap_or(0);

            while po1 < po2 {
                b1 = self.idom[b1];
                if !b1.is_valid() {
                    return b2;
                }
            }
            while po2 < po1 {
                b2 = self.idom[b2];
                if !b2.is_valid() {
                    return b1;
                }
            }
        }
        b1
    }

    /// Compute tree depths.
    fn compute_depths(&mut self, block: BlockId, depth: u32) {
        self.depth.set(block, depth);

        if let Some(children) = self.children.get(block).cloned() {
            for child in children {
                self.compute_depths(child, depth + 1);
            }
        }
    }

    /// Compute dominance frontier.
    fn compute_frontier(&mut self, cfg: &Cfg) {
        for (block, bb) in cfg.iter() {
            if bb.predecessors.len() >= 2 {
                for &pred in &bb.predecessors {
                    let mut runner = pred;
                    while runner.is_valid() && runner != self.idom[block] {
                        self.frontier.resize(runner.as_usize() + 1);
                        if let Some(frontier) = self.frontier.get_mut(runner) {
                            if !frontier.contains(&block) {
                                frontier.push(block);
                            }
                        } else {
                            self.frontier.set(runner, vec![block]);
                        }
                        runner = self.idom[runner];
                    }
                }
            }
        }
    }

    /// Get immediate dominator.
    pub fn idom(&self, block: BlockId) -> Option<BlockId> {
        let idom = self.idom.get(block).copied().unwrap_or(BlockId::INVALID);
        if idom.is_valid() && idom != block {
            Some(idom)
        } else {
            None
        }
    }

    /// Get dominator tree children.
    pub fn children(&self, block: BlockId) -> &[BlockId] {
        self.children
            .get(block)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get dominance frontier.
    pub fn frontier(&self, block: BlockId) -> &[BlockId] {
        self.frontier
            .get(block)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get dominator depth.
    pub fn depth(&self, block: BlockId) -> u32 {
        self.depth.get(block).copied().unwrap_or(0)
    }

    /// Check if `a` dominates `b`.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }

        let mut current = b;
        while let Some(idom) = self.idom(current) {
            if idom == a {
                return true;
            }
            current = idom;
        }
        false
    }

    /// Check if `a` strictly dominates `b`.
    pub fn strictly_dominates(&self, a: BlockId, b: BlockId) -> bool {
        a != b && self.dominates(a, b)
    }
}

// =============================================================================
// Loop Analysis
// =============================================================================

/// A natural loop in the CFG.
#[derive(Debug, Clone)]
pub struct Loop {
    /// The loop header block.
    pub header: BlockId,

    /// Back edge sources (blocks that jump back to header).
    pub back_edges: Vec<BlockId>,

    /// All blocks in the loop body.
    pub body: Vec<BlockId>,

    /// Parent loop (if nested).
    pub parent: Option<usize>,

    /// Child loops (nested).
    pub children: Vec<usize>,

    /// Loop depth (1 = outermost).
    pub depth: u32,
}

/// Loop analysis results.
#[derive(Debug, Clone)]
pub struct LoopAnalysis {
    /// All detected loops.
    pub loops: Vec<Loop>,

    /// Map from header block to loop index.
    pub header_to_loop: HashMap<BlockId, usize>,

    /// Map from block to innermost containing loop.
    pub block_to_loop: HashMap<BlockId, usize>,
}

impl LoopAnalysis {
    /// Compute loop analysis.
    pub fn compute(cfg: &Cfg, dom: &DominatorTree) -> Self {
        let mut analysis = LoopAnalysis {
            loops: Vec::new(),
            header_to_loop: HashMap::new(),
            block_to_loop: HashMap::new(),
        };

        // Find back edges (edges where target dominates source)
        for (block, bb) in cfg.iter() {
            for &succ in &bb.successors {
                if dom.dominates(succ, block) {
                    // Back edge found: block -> succ (header)
                    analysis.add_loop(succ, block, cfg, dom);
                }
            }
        }

        // Compute parent-child relationships and depths
        analysis.compute_nesting();

        analysis
    }

    /// Add a loop with the given header and back edge source.
    fn add_loop(&mut self, header: BlockId, back_edge: BlockId, cfg: &Cfg, _dom: &DominatorTree) {
        // Check if loop already exists for this header
        if let Some(&loop_idx) = self.header_to_loop.get(&header) {
            // Add back edge to existing loop
            if !self.loops[loop_idx].back_edges.contains(&back_edge) {
                self.loops[loop_idx].back_edges.push(back_edge);
            }
            return;
        }

        // Create new loop
        let loop_idx = self.loops.len();
        let mut loop_info = Loop {
            header,
            back_edges: vec![back_edge],
            body: Vec::new(),
            parent: None,
            children: Vec::new(),
            depth: 1,
        };

        // Find loop body using reverse DFS from back edge
        let mut body = BitSet::with_capacity(cfg.len());
        body.insert(header.as_usize());

        let mut worklist = VecDeque::new();
        worklist.push_back(back_edge);

        while let Some(block) = worklist.pop_front() {
            if !body.contains(block.as_usize()) {
                body.insert(block.as_usize());
                for &pred in &cfg.block(block).predecessors {
                    worklist.push_back(pred);
                }
            }
        }

        // Collect body blocks
        for block_idx in body.iter() {
            let block = BlockId::new(block_idx as u32);
            loop_info.body.push(block);
            self.block_to_loop.insert(block, loop_idx);
        }

        self.header_to_loop.insert(header, loop_idx);
        self.loops.push(loop_info);
    }

    /// Compute loop nesting.
    fn compute_nesting(&mut self) {
        let n = self.loops.len();

        for i in 0..n {
            // Find smallest enclosing loop
            let header = self.loops[i].header;
            let mut smallest_parent: Option<usize> = None;
            let mut smallest_size = usize::MAX;

            for j in 0..n {
                if i != j && self.loops[j].body.iter().any(|&b| b == header) {
                    let size = self.loops[j].body.len();
                    if size < smallest_size {
                        smallest_size = size;
                        smallest_parent = Some(j);
                    }
                }
            }

            if let Some(parent) = smallest_parent {
                self.loops[i].parent = Some(parent);
                self.loops[parent].children.push(i);
            }
        }

        // Compute depths
        for i in 0..n {
            self.loops[i].depth = self.compute_loop_depth(i);
        }
    }

    /// Compute loop depth.
    fn compute_loop_depth(&self, loop_idx: usize) -> u32 {
        let mut depth = 1;
        let mut current = self.loops[loop_idx].parent;
        while let Some(parent) = current {
            depth += 1;
            current = self.loops[parent].parent;
        }
        depth
    }

    /// Get the innermost loop containing a block.
    pub fn loop_for_block(&self, block: BlockId) -> Option<&Loop> {
        self.block_to_loop.get(&block).map(|&idx| &self.loops[idx])
    }

    /// Check if a block is in any loop.
    pub fn is_in_loop(&self, block: BlockId) -> bool {
        self.block_to_loop.contains_key(&block)
    }

    /// Get loop depth for a block (0 if not in loop).
    pub fn loop_depth(&self, block: BlockId) -> u32 {
        self.block_to_loop
            .get(&block)
            .map(|&idx| self.loops[idx].depth)
            .unwrap_or(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfg_build() {
        let g = Graph::new();
        let cfg = Cfg::build(&g);

        assert!(cfg.entry.is_valid());
        assert!(!cfg.is_empty());
    }

    #[test]
    fn test_dominator_tree() {
        let g = Graph::new();
        let cfg = Cfg::build(&g);
        let dom = DominatorTree::build(&cfg);

        // Entry dominates everything
        if cfg.exit.is_valid() {
            assert!(dom.dominates(cfg.entry, cfg.exit));
        }
    }

    #[test]
    fn test_dominator_self() {
        let g = Graph::new();
        let cfg = Cfg::build(&g);
        let dom = DominatorTree::build(&cfg);

        assert!(dom.dominates(cfg.entry, cfg.entry));
    }
}
