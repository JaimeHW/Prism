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
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Pack Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pack_new() {
        let scalars: SmallVec<[NodeId; 8]> = smallvec::smallvec![
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
            NodeId::new(4),
        ];
        let pack = Pack::new(scalars.clone(), PackOpKind::Arith(ArithOp::Add));

        assert_eq!(pack.lanes(), 4);
        assert!(!pack.is_empty());
        assert!(pack.is_complete());
        assert_eq!(pack.op_kind, PackOpKind::Arith(ArithOp::Add));
    }

    #[test]
    fn test_pack_empty() {
        let pack = Pack::empty();
        assert!(pack.is_empty());
        assert!(!pack.is_complete());
        assert_eq!(pack.lanes(), 0);
    }

    #[test]
    fn test_pack_scalar_at() {
        let scalars: SmallVec<[NodeId; 8]> = smallvec::smallvec![NodeId::new(1), NodeId::new(2),];
        let pack = Pack::new(scalars, PackOpKind::Load);

        assert_eq!(pack.scalar_at(0), Some(NodeId::new(1)));
        assert_eq!(pack.scalar_at(1), Some(NodeId::new(2)));
        assert_eq!(pack.scalar_at(2), None);
    }

    #[test]
    fn test_pack_set_scalar() {
        let mut pack = Pack::empty();
        pack.set_scalar(0, NodeId::new(1));
        pack.set_scalar(2, NodeId::new(3));

        assert_eq!(pack.lanes(), 3);
        assert_eq!(pack.scalar_at(0), Some(NodeId::new(1)));
        assert_eq!(pack.scalar_at(1), None); // Invalid node
        assert_eq!(pack.scalar_at(2), Some(NodeId::new(3)));
    }

    #[test]
    fn test_pack_contains() {
        let scalars: SmallVec<[NodeId; 8]> = smallvec::smallvec![NodeId::new(1), NodeId::new(2),];
        let pack = Pack::new(scalars, PackOpKind::Load);

        assert!(pack.contains(NodeId::new(1)));
        assert!(pack.contains(NodeId::new(2)));
        assert!(!pack.contains(NodeId::new(3)));
    }

    #[test]
    fn test_pack_lane_of() {
        let scalars: SmallVec<[NodeId; 8]> =
            smallvec::smallvec![NodeId::new(10), NodeId::new(20), NodeId::new(30),];
        let pack = Pack::new(scalars, PackOpKind::Store);

        assert_eq!(pack.lane_of(NodeId::new(10)), Some(0));
        assert_eq!(pack.lane_of(NodeId::new(20)), Some(1));
        assert_eq!(pack.lane_of(NodeId::new(30)), Some(2));
        assert_eq!(pack.lane_of(NodeId::new(99)), None);
    }

    #[test]
    fn test_pack_savings() {
        let mut pack = Pack::new(
            smallvec::smallvec![NodeId::new(1), NodeId::new(2)],
            PackOpKind::Arith(ArithOp::Add),
        );

        // Not profitable
        pack.profitable = false;
        pack.scalar_cost = 4.0;
        pack.vector_cost = 5.0;
        assert!((pack.savings() - 0.0).abs() < 0.001);

        // Profitable
        pack.profitable = true;
        assert!((pack.savings() - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_pack_set_element_type() {
        let mut pack = Pack::new(
            smallvec::smallvec![NodeId::new(1), NodeId::new(2)],
            PackOpKind::Arith(ArithOp::Add),
        );

        pack.set_element_type(ValueType::Float64);
        assert_eq!(pack.vector_type.element, ValueType::Float64);
    }

    // -------------------------------------------------------------------------
    // PackOpKind Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pack_op_kind_equality() {
        assert_eq!(
            PackOpKind::Arith(ArithOp::Add),
            PackOpKind::Arith(ArithOp::Add)
        );
        assert_ne!(
            PackOpKind::Arith(ArithOp::Add),
            PackOpKind::Arith(ArithOp::Sub)
        );
        assert_eq!(PackOpKind::Load, PackOpKind::Load);
    }

    #[test]
    fn test_pack_op_kind_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        PackOpKind::Load.hash(&mut hasher1);
        PackOpKind::Load.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    // -------------------------------------------------------------------------
    // SlpTree Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slp_tree_new() {
        let tree = SlpTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.num_packs(), 0);
    }

    #[test]
    fn test_slp_tree_add_pack() {
        let mut tree = SlpTree::new();
        let pack = Pack::new(
            smallvec::smallvec![NodeId::new(1), NodeId::new(2)],
            PackOpKind::Load,
        );

        let idx = tree.add_pack(pack);
        assert_eq!(idx, 0);
        assert_eq!(tree.num_packs(), 1);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_slp_tree_find_pack() {
        let mut tree = SlpTree::new();
        let pack = Pack::new(
            smallvec::smallvec![NodeId::new(1), NodeId::new(2)],
            PackOpKind::Load,
        );
        let idx = tree.add_pack(pack);

        assert_eq!(tree.find_pack(NodeId::new(1)), Some(idx));
        assert_eq!(tree.find_pack(NodeId::new(2)), Some(idx));
        assert_eq!(tree.find_pack(NodeId::new(99)), None);
    }

    #[test]
    fn test_slp_tree_is_packed() {
        let mut tree = SlpTree::new();
        let pack = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Constant);
        tree.add_pack(pack);

        assert!(tree.is_packed(NodeId::new(1)));
        assert!(!tree.is_packed(NodeId::new(2)));
    }

    #[test]
    fn test_slp_tree_mark_root() {
        let mut tree = SlpTree::new();
        let pack = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Store);
        let idx = tree.add_pack(pack);

        tree.mark_root(idx);
        tree.mark_root(idx); // Duplicate should not add

        assert_eq!(tree.roots().len(), 1);
        assert_eq!(tree.roots()[0], idx);
    }

    #[test]
    fn test_slp_tree_add_dependency() {
        let mut tree = SlpTree::new();
        let pack1 = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Load);
        let pack2 = Pack::new(smallvec::smallvec![NodeId::new(2)], PackOpKind::Store);

        let idx1 = tree.add_pack(pack1);
        let idx2 = tree.add_pack(pack2);

        tree.add_dependency(idx1, idx2);
        assert_eq!(tree.pack_deps.len(), 1);
    }

    #[test]
    fn test_slp_tree_topological_order() {
        let mut tree = SlpTree::new();

        // Create chain: 0 -> 1 -> 2
        let pack0 = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Load);
        let pack1 = Pack::new(
            smallvec::smallvec![NodeId::new(2)],
            PackOpKind::Arith(ArithOp::Add),
        );
        let pack2 = Pack::new(smallvec::smallvec![NodeId::new(3)], PackOpKind::Store);

        tree.add_pack(pack0);
        tree.add_pack(pack1);
        tree.add_pack(pack2);
        tree.add_dependency(0, 1);
        tree.add_dependency(1, 2);

        let order = tree.topological_order();
        assert_eq!(order.len(), 3);

        // 0 should come before 1, and 1 before 2
        let pos_0 = order.iter().position(|&x| x == 0).unwrap();
        let pos_1 = order.iter().position(|&x| x == 1).unwrap();
        let pos_2 = order.iter().position(|&x| x == 2).unwrap();

        assert!(pos_0 < pos_1);
        assert!(pos_1 < pos_2);
    }

    #[test]
    fn test_slp_tree_clear() {
        let mut tree = SlpTree::new();
        tree.add_pack(Pack::new(
            smallvec::smallvec![NodeId::new(1)],
            PackOpKind::Load,
        ));
        tree.mark_root(0);

        tree.clear();
        assert!(tree.is_empty());
        assert!(tree.roots().is_empty());
    }

    #[test]
    fn test_slp_tree_vector_ops_count() {
        let mut tree = SlpTree::new();

        let mut pack1 = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Load);
        pack1.profitable = true;
        tree.add_pack(pack1);

        let mut pack2 = Pack::new(smallvec::smallvec![NodeId::new(2)], PackOpKind::Load);
        pack2.profitable = false;
        tree.add_pack(pack2);

        assert_eq!(tree.vector_ops_count(), 1);
    }

    #[test]
    fn test_slp_tree_debug() {
        let tree = SlpTree::new();
        let debug_str = format!("{:?}", tree);
        assert!(debug_str.contains("SlpTree"));
    }

    // -------------------------------------------------------------------------
    // SlpVectorizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slp_vectorizer_new() {
        let graph = Graph::new();
        let cost_model = VectorCostModel::default();
        let vectorizer = SlpVectorizer::new(&graph, &cost_model, 4);

        assert!(vectorizer.tree().is_empty());
        assert_eq!(vectorizer.statistics().seeds_found, 0);
    }

    impl SlpVectorizer<'_> {
        fn statistics(&self) -> &SlpStats {
            &self.stats
        }
    }

    #[test]
    fn test_slp_find_seeds_empty() {
        let graph = Graph::new();
        let cost_model = VectorCostModel::default();
        let vectorizer = SlpVectorizer::new(&graph, &cost_model, 4);

        let seeds = vectorizer.find_seeds(&[]);
        assert!(seeds.is_empty());
    }

    #[test]
    fn test_slp_are_isomorphic_empty() {
        let graph = Graph::new();
        let cost_model = VectorCostModel::default();
        let vectorizer = SlpVectorizer::new(&graph, &cost_model, 4);

        assert!(!vectorizer.are_isomorphic(&[]));
        assert!(!vectorizer.are_isomorphic(&[NodeId::new(1)]));
    }

    #[test]
    fn test_slp_classify_op() {
        assert_eq!(
            SlpVectorizer::classify_op(&Operator::IntOp(ArithOp::Add)),
            Some(PackOpKind::Arith(ArithOp::Add))
        );
        assert_eq!(
            SlpVectorizer::classify_op(&Operator::ConstInt(42)),
            Some(PackOpKind::Constant)
        );
        assert_eq!(SlpVectorizer::classify_op(&Operator::Phi), None);
    }

    #[test]
    fn test_slp_is_store_op() {
        assert!(SlpVectorizer::is_store_op(&Operator::SetItem));
        assert!(SlpVectorizer::is_store_op(&Operator::Memory(
            crate::ir::operators::MemoryOp::Store
        )));
        assert!(!SlpVectorizer::is_store_op(&Operator::IntOp(ArithOp::Add)));
    }

    // -------------------------------------------------------------------------
    // SlpResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slp_result_unchanged() {
        let result = SlpResult::unchanged();
        assert!(!result.changed);
        assert_eq!(result.vector_ops, 0);
        assert_eq!(result.scalar_ops, 0);
        assert!((result.speedup - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_slp_result_success() {
        let result = SlpResult::success(4, 16, 3.5);
        assert!(result.changed);
        assert_eq!(result.vector_ops, 4);
        assert_eq!(result.scalar_ops, 16);
        assert!((result.speedup - 3.5).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // SlpStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slp_stats_default() {
        let stats = SlpStats::default();
        assert_eq!(stats.seeds_found, 0);
        assert_eq!(stats.packs_built, 0);
        assert_eq!(stats.profitable_packs, 0);
    }
}
