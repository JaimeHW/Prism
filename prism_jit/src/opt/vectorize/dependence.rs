//! Memory Dependence Analysis for Vectorization
//!
//! This module provides comprehensive memory dependence analysis to determine
//! whether memory operations can be safely reordered or vectorized. It computes:
//!
//! - **Direction Vectors**: Forward, Backward, or Unknown for each loop level
//! - **Distance Vectors**: Constant offsets or unknown distances
//! - **Dependence Classification**: RAW (true), WAR (anti), WAW (output)
//!
//! # Algorithm
//!
//! The analysis uses a constraint-based approach:
//!
//! 1. For each pair of memory operations (load/store), determine if they may alias
//! 2. If aliasing is possible, compute the access functions (base + index * stride)
//! 3. Solve the dependence equation to find if accesses overlap across iterations
//! 4. Compute direction and distance vectors from the solution
//!
//! # Vectorization Safety
//!
//! A loop is safe to vectorize if:
//!
//! - All loop-carried dependences have Forward direction (no cycles)
//! - The distance of any dependence is >= vector width, OR
//! - Dependences can be satisfied within a single vector iteration
//!
//! # References
//!
//! - "Optimizing Compilers for Modern Architectures" - Allen & Kennedy
//! - "Dependence Analysis for Supercomputing" - Banerjee

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator, VectorMemoryKind};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

// =============================================================================
// Direction and Distance
// =============================================================================

/// Direction of a loop-carried dependence.
///
/// For a dependence from operation A at iteration i to operation B at iteration j:
/// - Forward: i < j (A executes before B in source order, B depends on earlier A)
/// - Backward: i > j (A depends on later B - prevents vectorization)
/// - Equal: i = j (same iteration - loop independent)
/// - Unknown: Cannot determine statically
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Dependence flows forward (i < j) - safe for vectorization
    Forward,
    /// Dependence flows backward (i > j) - prevents vectorization
    Backward,
    /// Same iteration (i = j) - loop independent
    Equal,
    /// Direction unknown - must assume worst case
    Unknown,
}

impl Direction {
    /// Check if this direction is safe for vectorization.
    ///
    /// A direction is safe if dependences only flow forward or are loop-independent.
    #[inline]
    pub const fn is_safe(self) -> bool {
        matches!(self, Direction::Forward | Direction::Equal)
    }

    /// Check if this direction definitely prevents vectorization.
    #[inline]
    pub const fn prevents_vectorization(self) -> bool {
        matches!(self, Direction::Backward)
    }

    /// Merge two directions (for multiple paths).
    ///
    /// The result is the most conservative combination.
    pub const fn merge(self, other: Direction) -> Direction {
        match (self, other) {
            (Direction::Unknown, _) | (_, Direction::Unknown) => Direction::Unknown,
            (Direction::Equal, d) | (d, Direction::Equal) => d,
            (Direction::Forward, Direction::Forward) => Direction::Forward,
            (Direction::Backward, Direction::Backward) => Direction::Backward,
            (Direction::Forward, Direction::Backward)
            | (Direction::Backward, Direction::Forward) => Direction::Unknown,
        }
    }
}

impl Default for Direction {
    fn default() -> Self {
        Direction::Unknown
    }
}

/// Distance of a loop-carried dependence.
///
/// Represents the iteration distance between dependent operations.
/// A distance of k means operation B at iteration i depends on operation A
/// at iteration i-k.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Distance {
    /// Known constant distance (e.g., A[i] depends on A[i-1] has distance 1)
    Constant(i64),
    /// Distance is a known multiple of a stride
    Stride { base: i64, stride: i64 },
    /// Distance is positive but exact value unknown
    Positive,
    /// Distance is negative but exact value unknown
    Negative,
    /// Distance is completely unknown
    Unknown,
    /// No dependence (infinite distance)
    Infinity,
}

impl Distance {
    /// Get the minimum distance if known.
    pub const fn min_distance(self) -> Option<i64> {
        match self {
            Distance::Constant(d) => Some(d),
            Distance::Stride { base, stride } => {
                // Minimum is base (when multiplier is 0)
                if stride >= 0 {
                    Some(base)
                } else {
                    None // Could be arbitrarily negative
                }
            }
            Distance::Positive => Some(1),
            Distance::Negative => None,
            Distance::Unknown => None,
            Distance::Infinity => None,
        }
    }

    /// Check if this distance allows vectorization with given width.
    ///
    /// Vectorization with width W is safe if distance >= W (no overlap in vector).
    pub fn allows_vector_width(self, width: usize) -> bool {
        match self.min_distance() {
            Some(d) if d >= 0 => d as usize >= width,
            _ => false,
        }
    }

    /// Merge two distances (for multiple paths).
    pub fn merge(self, other: Distance) -> Distance {
        match (self, other) {
            (Distance::Infinity, d) | (d, Distance::Infinity) => d,
            (Distance::Unknown, _) | (_, Distance::Unknown) => Distance::Unknown,
            (Distance::Constant(a), Distance::Constant(b)) if a == b => Distance::Constant(a),
            (Distance::Constant(a), Distance::Constant(b)) => {
                // Different constants - use GCD for stride
                let gcd = gcd(a.unsigned_abs(), b.unsigned_abs()) as i64;
                if gcd > 1 {
                    Distance::Stride {
                        base: a.min(b),
                        stride: gcd,
                    }
                } else {
                    Distance::Unknown
                }
            }
            (Distance::Positive, Distance::Positive) => Distance::Positive,
            (Distance::Negative, Distance::Negative) => Distance::Negative,
            _ => Distance::Unknown,
        }
    }
}

impl Default for Distance {
    fn default() -> Self {
        Distance::Unknown
    }
}

/// Compute GCD using Euclidean algorithm.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// =============================================================================
// Dependence Types
// =============================================================================

/// Type of memory dependence between two operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependenceKind {
    /// Read-After-Write (true dependence / flow dependence)
    ///
    /// A store is followed by a load that reads the stored value.
    /// Essential to preserve - cannot reorder.
    RAW,

    /// Write-After-Read (anti dependence)
    ///
    /// A load is followed by a store to the same location.
    /// Can sometimes be eliminated via renaming.
    WAR,

    /// Write-After-Write (output dependence)
    ///
    /// A store is followed by another store to the same location.
    /// Only the last store matters for final result.
    WAW,
}

impl DependenceKind {
    /// Check if source is a write operation.
    #[inline]
    pub const fn source_is_write(self) -> bool {
        matches!(self, DependenceKind::RAW | DependenceKind::WAW)
    }

    /// Check if destination is a write operation.
    #[inline]
    pub const fn dest_is_write(self) -> bool {
        matches!(self, DependenceKind::WAR | DependenceKind::WAW)
    }

    /// Check if this dependence can be removed by variable renaming.
    #[inline]
    pub const fn removable_by_renaming(self) -> bool {
        matches!(self, DependenceKind::WAR | DependenceKind::WAW)
    }
}

// =============================================================================
// Memory Access Description
// =============================================================================

/// Description of a memory access pattern.
///
/// Represents an affine access: base + sum(coeff[i] * loop_var[i]) + offset
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Base pointer node
    pub base: NodeId,
    /// Offset from base (constant)
    pub offset: i64,
    /// Coefficients for each loop induction variable (innermost first)
    pub coefficients: SmallVec<[i64; 4]>,
    /// Element size in bytes
    pub element_size: usize,
    /// Whether the access is a store
    pub is_store: bool,
    /// Original memory operation node
    pub node: NodeId,
}

impl AccessPattern {
    /// Create a new access pattern.
    pub fn new(
        base: NodeId,
        offset: i64,
        element_size: usize,
        is_store: bool,
        node: NodeId,
    ) -> Self {
        Self {
            base,
            offset,
            coefficients: SmallVec::new(),
            element_size,
            is_store,
            node,
        }
    }

    /// Set coefficient for a loop level.
    pub fn set_coefficient(&mut self, level: usize, coeff: i64) {
        while self.coefficients.len() <= level {
            self.coefficients.push(0);
        }
        self.coefficients[level] = coeff;
    }

    /// Get coefficient for a loop level.
    pub fn coefficient(&self, level: usize) -> i64 {
        self.coefficients.get(level).copied().unwrap_or(0)
    }

    /// Check if this access is loop-invariant at a given level.
    pub fn is_invariant_at(&self, level: usize) -> bool {
        self.coefficient(level) == 0
    }

    /// Check if two accesses are to the same base array.
    pub fn same_base(&self, other: &AccessPattern) -> bool {
        self.base == other.base
    }

    /// Check if this access is definitely before another in memory.
    ///
    /// Returns true only if we can prove non-overlapping addresses.
    pub fn definitely_before(&self, other: &AccessPattern) -> bool {
        if !self.same_base(other) {
            return false; // Cannot prove ordering for different bases
        }

        // Same coefficients - compare offsets
        if self.coefficients == other.coefficients {
            let self_end = self.offset + self.element_size as i64;
            self_end <= other.offset
        } else {
            false
        }
    }
}

// =============================================================================
// Dependence Edge
// =============================================================================

/// A single dependence between two memory operations.
#[derive(Debug, Clone)]
pub struct Dependence {
    /// Source memory operation (writes for RAW/WAW, reads for WAR)
    pub src: NodeId,
    /// Destination memory operation
    pub dst: NodeId,
    /// Type of dependence
    pub kind: DependenceKind,
    /// Direction vector (one entry per loop nest level, innermost first)
    pub direction: SmallVec<[Direction; 4]>,
    /// Distance vector (one entry per loop nest level, innermost first)
    pub distance: SmallVec<[Distance; 4]>,
    /// Whether this is a loop-independent dependence
    pub loop_independent: bool,
    /// Confidence level (for approximate analysis)
    pub confidence: DependenceConfidence,
}

/// Confidence level of dependence analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependenceConfidence {
    /// Dependence is certain (proven by exact analysis)
    Proven,
    /// Dependence is likely (heuristic/approximate analysis)
    Likely,
    /// Dependence is possible but unproven (conservative)
    Possible,
}

impl Dependence {
    /// Create a new dependence.
    pub fn new(src: NodeId, dst: NodeId, kind: DependenceKind) -> Self {
        Self {
            src,
            dst,
            kind,
            direction: SmallVec::new(),
            distance: SmallVec::new(),
            loop_independent: false,
            confidence: DependenceConfidence::Possible,
        }
    }

    /// Create a loop-independent dependence.
    pub fn loop_independent(src: NodeId, dst: NodeId, kind: DependenceKind) -> Self {
        Self {
            src,
            dst,
            kind,
            direction: SmallVec::new(),
            distance: SmallVec::new(),
            loop_independent: true,
            confidence: DependenceConfidence::Proven,
        }
    }

    /// Set direction for a loop level.
    pub fn set_direction(&mut self, level: usize, dir: Direction) {
        while self.direction.len() <= level {
            self.direction.push(Direction::Unknown);
        }
        self.direction[level] = dir;
    }

    /// Set distance for a loop level.
    pub fn set_distance(&mut self, level: usize, dist: Distance) {
        while self.distance.len() <= level {
            self.distance.push(Distance::Unknown);
        }
        self.distance[level] = dist;
    }

    /// Get direction for a loop level.
    pub fn direction_at(&self, level: usize) -> Direction {
        self.direction
            .get(level)
            .copied()
            .unwrap_or(Direction::Unknown)
    }

    /// Get distance for a loop level.
    pub fn distance_at(&self, level: usize) -> Distance {
        self.distance
            .get(level)
            .copied()
            .unwrap_or(Distance::Unknown)
    }

    /// Check if this dependence prevents vectorization at a given loop level.
    pub fn prevents_vectorization_at(&self, level: usize) -> bool {
        if self.loop_independent {
            return false; // Loop-independent deps don't prevent vectorization
        }
        self.direction_at(level).prevents_vectorization()
    }

    /// Check if this dependence is safe for vectorization with given width.
    pub fn allows_vector_width(&self, level: usize, width: usize) -> bool {
        if self.loop_independent {
            return true;
        }
        match self.direction_at(level) {
            Direction::Equal => true,
            Direction::Forward => self.distance_at(level).allows_vector_width(width),
            Direction::Backward | Direction::Unknown => false,
        }
    }
}

// =============================================================================
// Dependence Graph
// =============================================================================

/// Memory dependence graph for a loop or code region.
///
/// Represents all memory dependences as a directed graph where nodes are
/// memory operations and edges are dependences.
pub struct DependenceGraph {
    /// All memory operations in the analyzed region.
    memory_ops: Vec<NodeId>,
    /// Store operations only (for RAW/WAW analysis)
    stores: Vec<NodeId>,
    /// Load operations only (for WAR analysis)
    loads: Vec<NodeId>,
    /// Dependence edges: src -> list of dependences
    dependences: FxHashMap<NodeId, Vec<Dependence>>,
    /// Reverse edges: dst -> list of dependences
    reverse_deps: FxHashMap<NodeId, Vec<Dependence>>,
    /// Access patterns for each memory operation
    patterns: FxHashMap<NodeId, AccessPattern>,
    /// Loop nesting depth of analysis
    depth: usize,
    /// Whether the region is vectorizable (no backward loop-carried deps)
    vectorizable: bool,
    /// Maximum safe vector width based on distance analysis
    max_safe_width: usize,
}

impl DependenceGraph {
    /// Create a new empty dependence graph.
    pub fn new(depth: usize) -> Self {
        Self {
            memory_ops: Vec::new(),
            stores: Vec::new(),
            loads: Vec::new(),
            dependences: FxHashMap::default(),
            reverse_deps: FxHashMap::default(),
            patterns: FxHashMap::default(),
            depth,
            vectorizable: true,
            max_safe_width: usize::MAX,
        }
    }

    /// Compute dependence graph for memory operations in a graph.
    ///
    /// This is the main entry point for dependence analysis.
    pub fn compute(graph: &Graph, memory_ops: &[NodeId], depth: usize) -> Self {
        let mut dg = Self::new(depth);

        // Classify memory operations
        for &node_id in memory_ops {
            if let Some(node) = graph.get(node_id) {
                dg.memory_ops.push(node_id);
                let is_store = Self::is_store_op(&node.op);
                if is_store {
                    dg.stores.push(node_id);
                } else {
                    dg.loads.push(node_id);
                }

                // Extract access pattern
                let pattern = Self::extract_pattern(graph, node_id, is_store);
                dg.patterns.insert(node_id, pattern);
            }
        }

        // Compute dependences
        dg.compute_all_dependences(graph);
        dg.compute_vectorizability();

        dg
    }

    /// Check if an operator is a store operation.
    fn is_store_op(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(MemoryOp::Store)
                | Operator::Memory(MemoryOp::StoreField)
                | Operator::Memory(MemoryOp::StoreElement)
                | Operator::SetItem
                | Operator::SetAttr
                | Operator::VectorMemory(_, VectorMemoryKind::StoreAligned)
                | Operator::VectorMemory(_, VectorMemoryKind::StoreUnaligned)
                | Operator::VectorMemory(_, VectorMemoryKind::Scatter)
        )
    }

    /// Extract access pattern from a memory operation.
    fn extract_pattern(graph: &Graph, node_id: NodeId, is_store: bool) -> AccessPattern {
        let node = graph.node(node_id);

        // Get base pointer (first input for most memory ops)
        let base = if !node.inputs.is_empty() {
            node.inputs[0]
        } else {
            node_id // Fallback
        };

        // Try to extract offset from second input if it's a constant
        let offset = if node.inputs.len() > 1 {
            let idx_node = graph.get(node.inputs[1]);
            if let Some(n) = idx_node {
                if let Operator::ConstInt(v) = n.op {
                    v * 8 // Assume 8-byte (64-bit) elements
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        };

        AccessPattern::new(base, offset, 8, is_store, node_id)
    }

    /// Compute all dependences between memory operations.
    fn compute_all_dependences(&mut self, _graph: &Graph) {
        // RAW: Store -> Load
        for &store in &self.stores {
            for &load in &self.loads {
                if let Some(dep) = self.check_dependence(store, load, DependenceKind::RAW) {
                    self.add_dependence(dep);
                }
            }
        }

        // WAR: Load -> Store
        for &load in &self.loads {
            for &store in &self.stores {
                if let Some(dep) = self.check_dependence(load, store, DependenceKind::WAR) {
                    self.add_dependence(dep);
                }
            }
        }

        // WAW: Store -> Store
        for i in 0..self.stores.len() {
            for j in (i + 1)..self.stores.len() {
                let store1 = self.stores[i];
                let store2 = self.stores[j];
                if let Some(dep) = self.check_dependence(store1, store2, DependenceKind::WAW) {
                    self.add_dependence(dep);
                }
            }
        }
    }

    /// Check for dependence between two operations.
    fn check_dependence(
        &self,
        src: NodeId,
        dst: NodeId,
        kind: DependenceKind,
    ) -> Option<Dependence> {
        let src_pattern = self.patterns.get(&src)?;
        let dst_pattern = self.patterns.get(&dst)?;

        // Different base pointers - may alias, be conservative
        if !src_pattern.same_base(dst_pattern) {
            // Could use alias analysis here for better precision
            // For now, assume may-alias
            let mut dep = Dependence::new(src, dst, kind);
            dep.confidence = DependenceConfidence::Possible;
            for level in 0..self.depth {
                dep.set_direction(level, Direction::Unknown);
                dep.set_distance(level, Distance::Unknown);
            }
            return Some(dep);
        }

        // Same base - analyze access patterns
        self.analyze_same_base_dependence(src_pattern, dst_pattern, kind)
    }

    /// Analyze dependence for same-base accesses.
    fn analyze_same_base_dependence(
        &self,
        src: &AccessPattern,
        dst: &AccessPattern,
        kind: DependenceKind,
    ) -> Option<Dependence> {
        let mut dep = Dependence::new(src.node, dst.node, kind);
        dep.confidence = DependenceConfidence::Proven;

        // Check each loop level
        for level in 0..self.depth {
            let src_coeff = src.coefficient(level);
            let dst_coeff = dst.coefficient(level);

            if src_coeff == 0 && dst_coeff == 0 {
                // Both loop-invariant at this level
                dep.set_direction(level, Direction::Equal);
                dep.set_distance(level, Distance::Constant(0));
            } else if src_coeff == dst_coeff {
                // Same stride - check offset difference
                let offset_diff = dst.offset - src.offset;
                let elem_diff = offset_diff / src.element_size as i64;

                if elem_diff == 0 {
                    // Same element in each iteration
                    dep.set_direction(level, Direction::Equal);
                    dep.set_distance(level, Distance::Constant(0));
                    dep.loop_independent = true;
                } else if elem_diff > 0 {
                    // Forward dependence
                    dep.set_direction(level, Direction::Forward);
                    dep.set_distance(level, Distance::Constant(elem_diff));
                } else {
                    // Backward dependence
                    dep.set_direction(level, Direction::Backward);
                    dep.set_distance(level, Distance::Constant(-elem_diff));
                }
            } else {
                // Different strides - complex analysis needed
                // Use GCD test for quick rejection
                let gcd_val = gcd(src_coeff.unsigned_abs(), dst_coeff.unsigned_abs());
                let offset_diff = (dst.offset - src.offset).unsigned_abs();

                if gcd_val > 0 && offset_diff % gcd_val != 0 {
                    // GCD test proves independence
                    return None;
                }

                // Cannot determine precisely
                dep.set_direction(level, Direction::Unknown);
                dep.set_distance(level, Distance::Unknown);
                dep.confidence = DependenceConfidence::Likely;
            }
        }

        Some(dep)
    }

    /// Add a dependence to the graph.
    fn add_dependence(&mut self, dep: Dependence) {
        let src = dep.src;
        let dst = dep.dst;

        self.reverse_deps.entry(dst).or_default().push(dep.clone());
        self.dependences.entry(src).or_default().push(dep);
    }

    /// Compute vectorizability based on dependences.
    fn compute_vectorizability(&mut self) {
        self.vectorizable = true;
        self.max_safe_width = usize::MAX;

        for deps in self.dependences.values() {
            for dep in deps {
                if dep.loop_independent {
                    continue;
                }

                // Check innermost loop (level 0)
                match dep.direction_at(0) {
                    Direction::Backward => {
                        self.vectorizable = false;
                        self.max_safe_width = 1;
                        return;
                    }
                    Direction::Unknown => {
                        // Be conservative
                        self.vectorizable = false;
                        self.max_safe_width = 1;
                        return;
                    }
                    Direction::Forward => {
                        // Check distance
                        if let Some(min_dist) = dep.distance_at(0).min_distance() {
                            if min_dist >= 0 {
                                self.max_safe_width = self.max_safe_width.min(min_dist as usize);
                            }
                        } else {
                            // Unknown distance with forward direction
                            // Could still be vectorizable with runtime check
                            self.max_safe_width = self.max_safe_width.min(1);
                        }
                    }
                    Direction::Equal => {
                        // Loop-independent, fine
                    }
                }
            }
        }
    }

    // =========================================================================
    // Query API
    // =========================================================================

    /// Check if the region is safe to vectorize.
    #[inline]
    pub fn is_vectorizable(&self) -> bool {
        self.vectorizable
    }

    /// Get maximum safe vector width.
    ///
    /// Returns the largest width that can be used without violating dependences.
    #[inline]
    pub fn max_safe_vector_width(&self) -> usize {
        self.max_safe_width
    }

    /// Check if two specific operations have a dependence.
    pub fn has_dependence(&self, src: NodeId, dst: NodeId) -> bool {
        self.dependences
            .get(&src)
            .map(|deps| deps.iter().any(|d| d.dst == dst))
            .unwrap_or(false)
    }

    /// Get all dependences from a source node.
    pub fn dependences_from(&self, src: NodeId) -> &[Dependence] {
        self.dependences
            .get(&src)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all dependences to a destination node.
    pub fn dependences_to(&self, dst: NodeId) -> &[Dependence] {
        self.reverse_deps
            .get(&dst)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all dependences in the graph.
    pub fn all_dependences(&self) -> impl Iterator<Item = &Dependence> {
        self.dependences.values().flat_map(|deps| deps.iter())
    }

    /// Get number of dependences.
    pub fn num_dependences(&self) -> usize {
        self.dependences.values().map(|v| v.len()).sum()
    }

    /// Get all memory operations.
    pub fn memory_ops(&self) -> &[NodeId] {
        &self.memory_ops
    }

    /// Get all store operations.
    pub fn stores(&self) -> &[NodeId] {
        &self.stores
    }

    /// Get all load operations.
    pub fn loads(&self) -> &[NodeId] {
        &self.loads
    }

    /// Get access pattern for a node.
    pub fn pattern(&self, node: NodeId) -> Option<&AccessPattern> {
        self.patterns.get(&node)
    }

    /// Get loop nesting depth.
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Find all backward dependences (blocking vectorization).
    pub fn backward_dependences(&self) -> Vec<&Dependence> {
        self.all_dependences()
            .filter(|dep| dep.direction_at(0) == Direction::Backward)
            .collect()
    }

    /// Find all loop-independent dependences.
    pub fn loop_independent_dependences(&self) -> Vec<&Dependence> {
        self.all_dependences()
            .filter(|dep| dep.loop_independent)
            .collect()
    }
}

impl std::fmt::Debug for DependenceGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DependenceGraph")
            .field("memory_ops", &self.memory_ops.len())
            .field("stores", &self.stores.len())
            .field("loads", &self.loads.len())
            .field("dependences", &self.num_dependences())
            .field("vectorizable", &self.vectorizable)
            .field("max_safe_width", &self.max_safe_width)
            .field("depth", &self.depth)
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
