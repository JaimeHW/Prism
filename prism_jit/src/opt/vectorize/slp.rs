//! Superword-Level Parallelism (SLP) Vectorizer
//!
//! This module implements bottom-up SLP vectorization for straight-line code.
//! SLP groups isomorphic scalar operations into vector packs that execute
//! in parallel using SIMD instructions.
//!
//! # Algorithm Overview
//!
//! 1. **Seed Selection**: Find vectorization opportunities from:
//!    - Stores to consecutive memory locations
//!    - Reductions with associative operators
//!
//! 2. **Tree Building**: Recursively build packs by:
//!    - Grouping isomorphic operations (same opcode, same pattern)
//!    - Following operand chains upward
//!    - Stopping at non-vectorizable operations
//!
//! 3. **Scheduling**: Order pack execution to minimize cross-pack dependencies
//!
//! 4. **Cost Analysis**: Determine if vectorization is profitable
//!
//! 5. **Code Generation**: Replace scalar ops with vector equivalents
//!
//! # Example
//!
//! ```text
//! Before:
//!   a0 = b0 + c0
//!   a1 = b1 + c1
//!   a2 = b2 + c2
//!   a3 = b3 + c3
//!
//! After (SLP with 4-wide vector):
//!   v_b = pack(b0, b1, b2, b3)
//!   v_c = pack(c0, c1, c2, c3)
//!   v_a = vadd(v_b, v_c)
//!   unpack(v_a) -> a0, a1, a2, a3
//! ```
//!
//! # References
//!
//! - "Exploiting Superword Level Parallelism with Multimedia Instruction Sets"
//!   (Larsen & Amarasinghe, PLDI 2000)

use super::cost::{CostAnalysis, OpCost, VectorCostModel};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ArithOp, Operator, VectorArithKind, VectorOp};
use crate::ir::types::ValueType;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

// =============================================================================
// Pack
// =============================================================================

/// A pack of isomorphic scalar operations to be vectorized together.
///
/// Each pack represents a group of scalar operations that:
/// - Have the same opcode/operation
/// - Process independent data
/// - Can be combined into a single SIMD instruction
#[derive(Debug, Clone)]
pub struct Pack {
    /// Scalar operations in lane order (index = lane number).
    pub scalars: SmallVec<[NodeId; 8]>,

    /// Vector operation that will replace the scalars (after vectorization).
    pub vector_node: Option<NodeId>,

    /// Vector type for this pack.
    pub vector_type: VectorOp,

    /// The operation being performed (for quick lookup).
    pub op_kind: PackOpKind,

    /// Cost of scalar execution.
    pub scalar_cost: f32,

    /// Cost of vector execution.
    pub vector_cost: f32,

    /// Whether this pack is profitable to vectorize.
    pub profitable: bool,

    /// Pack generation (for build order).
    pub generation: usize,
}

/// Kind of operation in a pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackOpKind {
    /// Arithmetic operation.
    Arith(ArithOp),
    /// Comparison operation.
    Comparison,
    /// Memory load.
    Load,
    /// Memory store.
    Store,
    /// Constant/splat.
    Constant,
    /// Unknown/mixed.
    Unknown,
}

impl Pack {
    /// Create a new pack with given scalars.
    pub fn new(scalars: SmallVec<[NodeId; 8]>, op_kind: PackOpKind) -> Self {
        let lanes = scalars.len() as u8;
        Self {
            scalars,
            vector_node: None,
            vector_type: VectorOp {
                element: ValueType::Int64,
                lanes,
            },
            op_kind,
            scalar_cost: 0.0,
            vector_cost: 0.0,
            profitable: false,
            generation: 0,
        }
    }

    /// Create an empty pack.
    pub fn empty() -> Self {
        Self::new(SmallVec::new(), PackOpKind::Unknown)
    }

    /// Get the number of lanes in this pack.
    #[inline]
    pub fn lanes(&self) -> usize {
        self.scalars.len()
    }

    /// Check if the pack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.scalars.is_empty()
    }

    /// Check if all pack slots are filled.
    #[inline]
    pub fn is_complete(&self) -> bool {
        !self.scalars.is_empty() && self.scalars.iter().all(|n| n.is_valid())
    }

    /// Get scalar at a specific lane.
    #[inline]
    pub fn scalar_at(&self, lane: usize) -> Option<NodeId> {
        self.scalars.get(lane).copied().filter(|n| n.is_valid())
    }

    /// Set scalar at a specific lane.
    pub fn set_scalar(&mut self, lane: usize, node: NodeId) {
        while self.scalars.len() <= lane {
            self.scalars.push(NodeId::invalid());
        }
        self.scalars[lane] = node;
    }

    /// Check if a node is in this pack.
    pub fn contains(&self, node: NodeId) -> bool {
        self.scalars.contains(&node)
    }

    /// Get the lane number for a node.
    pub fn lane_of(&self, node: NodeId) -> Option<usize> {
        self.scalars.iter().position(|&n| n == node)
    }

    /// Compute cost savings from vectorization.
    pub fn savings(&self) -> f32 {
        if self.profitable {
            self.scalar_cost - self.vector_cost
        } else {
            0.0
        }
    }

    /// Set element type for the pack.
    pub fn set_element_type(&mut self, element: ValueType) {
        self.vector_type.element = element;
    }
}

// =============================================================================
// SLP Tree
// =============================================================================

/// The SLP tree representing all vector packs found.
///
/// The tree structure captures dependencies between packs,
/// allowing proper scheduling during code generation.
pub struct SlpTree {
    /// All packs discovered (indexed by pack ID).
    packs: Vec<Pack>,

    /// Mapping from scalar node to containing pack index.
    scalar_to_pack: FxHashMap<NodeId, usize>,

    /// Pack dependencies: (producer_pack, consumer_pack).
    pack_deps: Vec<(usize, usize)>,

    /// Root packs (starting points for vectorization).
    roots: Vec<usize>,

    /// Current build generation.
    generation: usize,
}

impl SlpTree {
    /// Create a new empty SLP tree.
    pub fn new() -> Self {
        Self {
            packs: Vec::new(),
            scalar_to_pack: FxHashMap::default(),
            pack_deps: Vec::new(),
            roots: Vec::new(),
            generation: 0,
        }
    }

    /// Get the number of packs.
    #[inline]
    pub fn num_packs(&self) -> usize {
        self.packs.len()
    }

    /// Check if the tree is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.packs.is_empty()
    }

    /// Get a pack by index.
    pub fn pack(&self, idx: usize) -> Option<&Pack> {
        self.packs.get(idx)
    }

    /// Get a mutable pack by index.
    pub fn pack_mut(&mut self, idx: usize) -> Option<&mut Pack> {
        self.packs.get_mut(idx)
    }

    /// Add a new pack, returning its index.
    pub fn add_pack(&mut self, mut pack: Pack) -> usize {
        let idx = self.packs.len();
        pack.generation = self.generation;

        // Register scalars
        for &scalar in &pack.scalars {
            if scalar.is_valid() {
                self.scalar_to_pack.insert(scalar, idx);
            }
        }

        self.packs.push(pack);
        idx
    }

    /// Mark a pack as a root.
    pub fn mark_root(&mut self, pack_idx: usize) {
        if !self.roots.contains(&pack_idx) {
            self.roots.push(pack_idx);
        }
    }

    /// Add a dependency between packs.
    pub fn add_dependency(&mut self, producer: usize, consumer: usize) {
        self.pack_deps.push((producer, consumer));
    }

    /// Get the pack containing a scalar node.
    pub fn find_pack(&self, scalar: NodeId) -> Option<usize> {
        self.scalar_to_pack.get(&scalar).copied()
    }

    /// Check if a scalar is already in a pack.
    pub fn is_packed(&self, scalar: NodeId) -> bool {
        self.scalar_to_pack.contains_key(&scalar)
    }

    /// Get all root packs.
    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    /// Iterate over all packs.
    pub fn iter(&self) -> impl Iterator<Item = &Pack> {
        self.packs.iter()
    }

    /// Get packs in topological order (producers before consumers).
    pub fn topological_order(&self) -> Vec<usize> {
        let n = self.packs.len();
        if n == 0 {
            return Vec::new();
        }

        // Build in-degree count
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for &(prod, cons) in &self.pack_deps {
            if prod < n && cons < n {
                adj[prod].push(cons);
                in_degree[cons] += 1;
            }
        }

        // Kahn's algorithm
        let mut result = Vec::with_capacity(n);
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();

        while let Some(node) = queue.pop() {
            result.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push(next);
                }
            }
        }

        result
    }

    /// Compute total vector ops that will be created.
    pub fn vector_ops_count(&self) -> usize {
        self.packs.iter().filter(|p| p.profitable).count()
    }

    /// Compute total scalar ops that will be eliminated.
    pub fn scalar_ops_eliminated(&self) -> usize {
        self.packs
            .iter()
            .filter(|p| p.profitable)
            .map(|p| p.lanes())
            .sum()
    }

    /// Increment the build generation.
    pub fn next_generation(&mut self) {
        self.generation += 1;
    }

    /// Clear all packs.
    pub fn clear(&mut self) {
        self.packs.clear();
        self.scalar_to_pack.clear();
        self.pack_deps.clear();
        self.roots.clear();
    }
}

impl Default for SlpTree {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SlpTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlpTree")
            .field("num_packs", &self.packs.len())
            .field("num_roots", &self.roots.len())
            .field("num_deps", &self.pack_deps.len())
            .finish()
    }
}

// =============================================================================
// SLP Vectorizer
// =============================================================================

/// The SLP vectorizer.
///
/// Builds and schedules vector packs from straight-line scalar code.
pub struct SlpVectorizer<'a> {
    /// Reference to the IR graph.
    graph: &'a Graph,

    /// Cost model for profitability analysis.
    cost_model: &'a VectorCostModel,

    /// The SLP tree being built.
    tree: SlpTree,

    /// Target vector width (number of lanes).
    width: usize,

    /// Nodes already visited in current build.
    visited: FxHashSet<NodeId>,

    /// Statistics.
    stats: SlpStats,
}

/// SLP vectorization statistics.
#[derive(Debug, Clone, Default)]
pub struct SlpStats {
    /// Number of seeds found.
    pub seeds_found: usize,
    /// Number of packs built.
    pub packs_built: usize,
    /// Number of profitable packs.
    pub profitable_packs: usize,
    /// Total scalar ops eliminated.
    pub scalar_ops_eliminated: usize,
    /// Total vector ops created.
    pub vector_ops_created: usize,
}

impl<'a> SlpVectorizer<'a> {
    /// Create a new SLP vectorizer.
    pub fn new(graph: &'a Graph, cost_model: &'a VectorCostModel, width: usize) -> Self {
        Self {
            graph,
            cost_model,
            tree: SlpTree::new(),
            width,
            visited: FxHashSet::default(),
            stats: SlpStats::default(),
        }
    }

    /// Get the built SLP tree.
    pub fn tree(&self) -> &SlpTree {
        &self.tree
    }

    /// Get the consumed SLP tree
    pub fn into_tree(self) -> SlpTree {
        self.tree
    }

    /// Get statistics.
    pub fn stats(&self) -> &SlpStats {
        &self.stats
    }

    /// Find seed groups for vectorization.
    ///
    /// Seeds are starting points for SLP tree building:
    /// - Stores to consecutive addresses
    /// - Groups of isomorphic operations
    pub fn find_seeds(&self, nodes: &[NodeId]) -> Vec<SmallVec<[NodeId; 8]>> {
        let mut seeds = Vec::new();

        // Group stores by base
        let mut stores_by_base: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();

        for &node_id in nodes {
            if let Some(node) = self.graph.get(node_id) {
                if Self::is_store_op(&node.op) && !node.inputs.is_empty() {
                    let base = node.inputs[0];
                    stores_by_base.entry(base).or_default().push(node_id);
                }
            }
        }

        // Create seed groups from stores
        for (_base, store_group) in stores_by_base {
            if store_group.len() >= self.width {
                // Take first `width` stores as a seed
                let seed: SmallVec<[NodeId; 8]> =
                    store_group.into_iter().take(self.width).collect();
                seeds.push(seed);
            }
        }

        // Also look for isomorphic arithmetic groups
        let mut op_groups: FxHashMap<PackOpKind, Vec<NodeId>> = FxHashMap::default();

        for &node_id in nodes {
            if let Some(node) = self.graph.get(node_id) {
                if let Some(kind) = Self::classify_op(&node.op) {
                    op_groups.entry(kind).or_default().push(node_id);
                }
            }
        }

        for (_kind, group) in op_groups {
            if group.len() >= self.width {
                let seed: SmallVec<[NodeId; 8]> = group.into_iter().take(self.width).collect();
                seeds.push(seed);
            }
        }

        seeds
    }

    /// Build the SLP tree from seeds.
    ///
    /// Returns true if any packs were built.
    pub fn build_tree(&mut self, seeds: &[SmallVec<[NodeId; 8]>]) -> bool {
        self.tree.clear();
        self.visited.clear();
        self.stats.seeds_found = seeds.len();

        for seed in seeds {
            if seed.len() < 2 {
                continue;
            }

            // Check isomorphism
            if !self.are_isomorphic(seed) {
                continue;
            }

            // Build pack for this seed
            let op_kind = self.determine_op_kind(seed);
            let pack = Pack::new(seed.clone(), op_kind);
            let pack_idx = self.tree.add_pack(pack);
            self.tree.mark_root(pack_idx);

            // Mark visited
            for &scalar in seed {
                self.visited.insert(scalar);
            }

            // Recursively build operand packs
            self.build_operand_packs(pack_idx);
        }

        // Analyze profitability
        self.analyze_profitability();

        self.tree.num_packs() > 0
    }

    /// Check if a group of nodes are isomorphic (same operation shape).
    pub fn are_isomorphic(&self, nodes: &[NodeId]) -> bool {
        if nodes.len() < 2 {
            return false;
        }

        let first = match self.graph.get(nodes[0]) {
            Some(n) => n,
            None => return false,
        };

        let first_op_kind = Self::classify_op(&first.op);
        let first_num_inputs = first.inputs.len();

        for &node_id in &nodes[1..] {
            let node = match self.graph.get(node_id) {
                Some(n) => n,
                None => return false,
            };

            // Same operation kind
            if Self::classify_op(&node.op) != first_op_kind {
                return false;
            }

            // Same number of inputs
            if node.inputs.len() != first_num_inputs {
                return false;
            }

            // Could add more checks: same constant, same target, etc.
        }

        true
    }

    /// Recursively build packs for operands.
    fn build_operand_packs(&mut self, pack_idx: usize) {
        let pack = match self.tree.pack(pack_idx) {
            Some(p) => p.clone(),
            None => return,
        };

        if pack.is_empty() {
            return;
        }

        // Get first scalar to determine number of operands
        let first_node = match self.graph.get(pack.scalars[0]) {
            Some(n) => n,
            None => return,
        };

        let num_operands = first_node.inputs.len();

        // For each operand position, try to build an operand pack
        for operand_idx in 0..num_operands {
            let mut operands: SmallVec<[NodeId; 8]> = SmallVec::new();

            // Gather operands from all scalars in the pack
            for &scalar in &pack.scalars {
                if let Some(node) = self.graph.get(scalar) {
                    if operand_idx < node.inputs.len() {
                        operands.push(node.inputs[operand_idx]);
                    }
                }
            }

            // Skip if any operand already packed or visited
            if operands.iter().any(|&n| self.visited.contains(&n)) {
                continue;
            }

            // Check if operands are isomorphic
            if operands.len() == pack.lanes() && self.are_isomorphic(&operands) {
                let op_kind = self.determine_op_kind(&operands);
                let operand_pack = Pack::new(operands.clone(), op_kind);
                let operand_pack_idx = self.tree.add_pack(operand_pack);

                // Mark visited
                for &op in &operands {
                    self.visited.insert(op);
                }

                // Add dependency
                self.tree.add_dependency(operand_pack_idx, pack_idx);

                // Recurse
                self.build_operand_packs(operand_pack_idx);
            }
        }
    }

    /// Determine the operation kind for a group of nodes.
    fn determine_op_kind(&self, nodes: &[NodeId]) -> PackOpKind {
        if let Some(first) = nodes.first() {
            if let Some(node) = self.graph.get(*first) {
                return Self::classify_op(&node.op).unwrap_or(PackOpKind::Unknown);
            }
        }
        PackOpKind::Unknown
    }

    /// Analyze profitability of all packs.
    fn analyze_profitability(&mut self) {
        for pack_idx in 0..self.tree.num_packs() {
            let (scalar_cost, vector_cost) = self.compute_pack_costs(pack_idx);

            if let Some(pack) = self.tree.pack_mut(pack_idx) {
                pack.scalar_cost = scalar_cost;
                pack.vector_cost = vector_cost;
                pack.profitable = vector_cost < scalar_cost * 0.9; // 10% margin

                if pack.profitable {
                    self.stats.profitable_packs += 1;
                }
            }
        }

        self.stats.packs_built = self.tree.num_packs();
        self.stats.scalar_ops_eliminated = self.tree.scalar_ops_eliminated();
        self.stats.vector_ops_created = self.tree.vector_ops_count();
    }

    /// Compute costs for a pack.
    fn compute_pack_costs(&self, pack_idx: usize) -> (f32, f32) {
        let pack = match self.tree.pack(pack_idx) {
            Some(p) => p,
            None => return (0.0, 0.0),
        };

        let lanes = pack.lanes();
        if lanes == 0 {
            return (0.0, 0.0);
        }

        // Scalar cost = N * single op cost
        let scalar_op_cost = match pack.op_kind {
            PackOpKind::Arith(ArithOp::Add) | PackOpKind::Arith(ArithOp::Sub) => {
                OpCost::alu().total_cost()
            }
            PackOpKind::Arith(ArithOp::Mul) => OpCost::mul().total_cost(),
            PackOpKind::Arith(ArithOp::TrueDiv) => OpCost::div().total_cost(),
            PackOpKind::Load => OpCost::load().total_cost(),
            PackOpKind::Store => OpCost::store().total_cost(),
            _ => OpCost::alu().total_cost(),
        };
        let scalar_cost = scalar_op_cost * lanes as f32;

        // Vector cost = single vector op + pack/unpack overhead
        let vop = VectorOp {
            element: ValueType::Int64,
            lanes: lanes as u8,
        };
        let vector_kind = match pack.op_kind {
            PackOpKind::Arith(ArithOp::Add) => VectorArithKind::Add,
            PackOpKind::Arith(ArithOp::Sub) => VectorArithKind::Sub,
            PackOpKind::Arith(ArithOp::Mul) => VectorArithKind::Mul,
            PackOpKind::Arith(ArithOp::TrueDiv) => VectorArithKind::Div,
            _ => VectorArithKind::Add,
        };
        let vector_cost = self.cost_model.arith_cost(vector_kind, vop).total_cost();

        (scalar_cost, vector_cost)
    }

    /// Check if an operation is a store.
    fn is_store_op(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(crate::ir::operators::MemoryOp::Store)
                | Operator::Memory(crate::ir::operators::MemoryOp::StoreField)
                | Operator::Memory(crate::ir::operators::MemoryOp::StoreElement)
                | Operator::SetItem
        )
    }

    /// Classify an operator into a pack kind.
    fn classify_op(op: &Operator) -> Option<PackOpKind> {
        match op {
            Operator::IntOp(arith) | Operator::FloatOp(arith) | Operator::GenericOp(arith) => {
                Some(PackOpKind::Arith(*arith))
            }
            Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => {
                Some(PackOpKind::Comparison)
            }
            Operator::Memory(crate::ir::operators::MemoryOp::Load)
            | Operator::Memory(crate::ir::operators::MemoryOp::LoadField)
            | Operator::Memory(crate::ir::operators::MemoryOp::LoadElement) => {
                Some(PackOpKind::Load)
            }
            Operator::Memory(crate::ir::operators::MemoryOp::Store)
            | Operator::Memory(crate::ir::operators::MemoryOp::StoreField)
            | Operator::Memory(crate::ir::operators::MemoryOp::StoreElement)
            | Operator::SetItem => Some(PackOpKind::Store),
            Operator::ConstInt(_) | Operator::ConstFloat(_) => Some(PackOpKind::Constant),
            _ => None,
        }
    }
}

// =============================================================================
// SLP Result
// =============================================================================

/// Result of SLP vectorization.
#[derive(Debug)]
pub struct SlpResult {
    /// Whether any vectorization occurred.
    pub changed: bool,

    /// Number of vector operations created.
    pub vector_ops: usize,

    /// Number of scalar operations eliminated.
    pub scalar_ops: usize,

    /// Estimated speedup.
    pub speedup: f32,

    /// Cost analysis.
    pub cost: Option<CostAnalysis>,
}

impl SlpResult {
    /// Create a result indicating no vectorization.
    pub fn unchanged() -> Self {
        Self {
            changed: false,
            vector_ops: 0,
            scalar_ops: 0,
            speedup: 1.0,
            cost: None,
        }
    }

    /// Create a result indicating successful vectorization.
    pub fn success(vector_ops: usize, scalar_ops: usize, speedup: f32) -> Self {
        Self {
            changed: true,
            vector_ops,
            scalar_ops,
            speedup,
            cost: None,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
