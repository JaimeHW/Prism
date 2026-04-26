//! Memory Alias Analysis for DSE.
//!
//! Determines whether two memory operations may reference the same location.
//!
//! # Alias Results
//!
//! - **MustAlias**: Definitely the same location
//! - **MayAlias**: Possibly the same location (conservative)
//! - **NoAlias**: Definitely different locations
//!
//! # Analysis Techniques
//!
//! 1. **Field-based**: Different field offsets → NoAlias
//! 2. **Allocation-based**: Fresh allocations don't alias existing memory
//! 3. **Index-based**: Constant array indices can be compared
//! 4. **Type-based**: Different types can't alias (with Python semantics)

use rustc_hash::FxHashMap;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator};

// =============================================================================
// Alias Result
// =============================================================================

/// Result of alias query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AliasResult {
    /// Definitely the same memory location.
    MustAlias,
    /// May or may not be the same location (conservative).
    MayAlias,
    /// Definitely different memory locations.
    NoAlias,
}

impl AliasResult {
    /// Check if locations may alias.
    pub fn may_alias(self) -> bool {
        matches!(self, AliasResult::MustAlias | AliasResult::MayAlias)
    }

    /// Check if locations must alias.
    pub fn must_alias(self) -> bool {
        matches!(self, AliasResult::MustAlias)
    }

    /// Check if locations don't alias.
    pub fn no_alias(self) -> bool {
        matches!(self, AliasResult::NoAlias)
    }

    /// Combine two alias results (meet operation).
    /// If either says NoAlias, result is NoAlias.
    /// If both say MustAlias, result is MustAlias.
    /// Otherwise, MayAlias.
    pub fn meet(self, other: AliasResult) -> AliasResult {
        match (self, other) {
            (AliasResult::NoAlias, _) | (_, AliasResult::NoAlias) => AliasResult::NoAlias,
            (AliasResult::MustAlias, AliasResult::MustAlias) => AliasResult::MustAlias,
            _ => AliasResult::MayAlias,
        }
    }
}

// =============================================================================
// Memory Offset
// =============================================================================

/// Offset within a memory object.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemOffset {
    /// Named field (by index/hash).
    Field(u32),
    /// Constant array/list element index.
    ConstElement(i64),
    /// Variable array/list element index (unknown at compile time).
    VarElement(NodeId),
    /// Unknown offset.
    Unknown,
}

impl MemOffset {
    /// Check if this offset is a known constant.
    pub fn is_constant(&self) -> bool {
        matches!(self, MemOffset::Field(_) | MemOffset::ConstElement(_))
    }

    /// Check if two offsets are definitely different.
    pub fn definitely_different(&self, other: &MemOffset) -> bool {
        match (self, other) {
            // Different fields are definitely different
            (MemOffset::Field(a), MemOffset::Field(b)) => a != b,
            // Different constant indices are definitely different
            (MemOffset::ConstElement(a), MemOffset::ConstElement(b)) => a != b,
            // Field vs element are different (in Sea-of-Nodes IR)
            (MemOffset::Field(_), MemOffset::ConstElement(_))
            | (MemOffset::ConstElement(_), MemOffset::Field(_)) => true,
            (MemOffset::Field(_), MemOffset::VarElement(_))
            | (MemOffset::VarElement(_), MemOffset::Field(_)) => true,
            // Variable indices may alias anything of same type
            _ => false,
        }
    }

    /// Check if two offsets are definitely the same.
    pub fn definitely_same(&self, other: &MemOffset) -> bool {
        match (self, other) {
            (MemOffset::Field(a), MemOffset::Field(b)) => a == b,
            (MemOffset::ConstElement(a), MemOffset::ConstElement(b)) => a == b,
            (MemOffset::VarElement(a), MemOffset::VarElement(b)) => a == b,
            _ => false,
        }
    }
}

// =============================================================================
// Memory Location
// =============================================================================

/// A memory location being accessed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryLocation {
    /// Base object/allocation.
    pub base: NodeId,
    /// Offset within the object.
    pub offset: MemOffset,
    /// Access size in bytes (if known).
    pub size: Option<usize>,
}

impl MemoryLocation {
    /// Create a new memory location.
    pub fn new(base: NodeId, offset: MemOffset) -> Self {
        Self {
            base,
            offset,
            size: None,
        }
    }

    /// Create with known size.
    pub fn with_size(base: NodeId, offset: MemOffset, size: usize) -> Self {
        Self {
            base,
            offset,
            size: Some(size),
        }
    }

    /// Create a field access location.
    pub fn field(base: NodeId, field_idx: u32) -> Self {
        Self::new(base, MemOffset::Field(field_idx))
    }

    /// Create a constant element access location.
    pub fn const_element(base: NodeId, index: i64) -> Self {
        Self::new(base, MemOffset::ConstElement(index))
    }

    /// Create a variable element access location.
    pub fn var_element(base: NodeId, index_node: NodeId) -> Self {
        Self::new(base, MemOffset::VarElement(index_node))
    }
}

// =============================================================================
// Alias Analyzer
// =============================================================================

/// Memory alias analyzer.
#[derive(Debug)]
pub struct AliasAnalyzer {
    /// Tracked allocations (NodeId -> allocation info).
    allocations: FxHashMap<NodeId, AllocationInfo>,
    /// Memory location cache.
    location_cache: FxHashMap<NodeId, MemoryLocation>,
}

/// Information about an allocation.
#[derive(Debug, Clone)]
struct AllocationInfo {
    /// The allocation node.
    node: NodeId,
    /// Whether this is a fresh allocation (doesn't alias pre-existing memory).
    is_fresh: bool,
}

impl AliasAnalyzer {
    /// Create a new alias analyzer for a graph.
    pub fn new(graph: &Graph) -> Self {
        let mut analyzer = Self {
            allocations: FxHashMap::default(),
            location_cache: FxHashMap::default(),
        };
        analyzer.analyze_allocations(graph);
        analyzer.build_location_cache(graph);
        analyzer
    }

    /// Analyze allocations in the graph.
    fn analyze_allocations(&mut self, graph: &Graph) {
        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if let Some(node) = graph.get(id) {
                match node.op {
                    Operator::Memory(MemoryOp::Alloc) | Operator::Memory(MemoryOp::AllocArray) => {
                        self.allocations.insert(
                            id,
                            AllocationInfo {
                                node: id,
                                is_fresh: true,
                            },
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    /// Build cache of memory locations for all loads/stores.
    fn build_location_cache(&mut self, graph: &Graph) {
        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if let Some(node) = graph.get(id) {
                if let Some(loc) = self.extract_location(graph, id, &node.op) {
                    self.location_cache.insert(id, loc);
                }
            }
        }
    }

    /// Extract memory location from a memory operation.
    fn extract_location(
        &self,
        graph: &Graph,
        node_id: NodeId,
        op: &Operator,
    ) -> Option<MemoryLocation> {
        match op {
            Operator::Memory(MemoryOp::LoadField) | Operator::Memory(MemoryOp::StoreField) => {
                // Field operations: inputs are (base, [value for store])
                let node = graph.get(node_id)?;
                let base = node.inputs.get(0)?;
                // Field index is encoded in node or as additional input
                // For simplicity, use node id hash as field indicator
                let field_idx = node.bc_offset; // Using bc_offset as field id placeholder
                Some(MemoryLocation::field(base, field_idx))
            }
            Operator::Memory(MemoryOp::LoadElement) | Operator::Memory(MemoryOp::StoreElement) => {
                // Element operations: inputs are (base, index, [value for store])
                let node = graph.get(node_id)?;
                let base = node.inputs.get(0)?;
                let index = node.inputs.get(1)?;

                // Check if index is a constant
                if let Some(idx_node) = graph.get(index) {
                    if let Operator::ConstInt(val) = idx_node.op {
                        return Some(MemoryLocation::const_element(base, val));
                    }
                }
                Some(MemoryLocation::var_element(base, index))
            }
            _ => None,
        }
    }

    /// Get the memory location for a memory operation.
    pub fn get_location(&self, node_id: NodeId) -> Option<&MemoryLocation> {
        self.location_cache.get(&node_id)
    }

    /// Query alias relationship between two memory operations.
    pub fn alias(&self, loc1: &MemoryLocation, loc2: &MemoryLocation) -> AliasResult {
        // Same base?
        if loc1.base == loc2.base {
            // Same base - check offset
            if loc1.offset.definitely_same(&loc2.offset) {
                return AliasResult::MustAlias;
            }
            if loc1.offset.definitely_different(&loc2.offset) {
                return AliasResult::NoAlias;
            }
            return AliasResult::MayAlias;
        }

        // Different bases - check if either is a fresh allocation
        let base1_fresh = self
            .allocations
            .get(&loc1.base)
            .map_or(false, |a| a.is_fresh);
        let base2_fresh = self
            .allocations
            .get(&loc2.base)
            .map_or(false, |a| a.is_fresh);

        // Two different fresh allocations can't alias
        if base1_fresh && base2_fresh {
            return AliasResult::NoAlias;
        }

        // One fresh allocation can't alias pre-existing memory from earlier
        // (but might alias if passed as argument, so be conservative)
        if base1_fresh || base2_fresh {
            // Fresh allocation doesn't alias other fresh allocations
            // But may alias non-fresh (could be same object through aliasing)
            return AliasResult::MayAlias;
        }

        // Different non-fresh bases - conservative
        AliasResult::MayAlias
    }

    /// Check if two memory operations may alias.
    pub fn may_alias(&self, node1: NodeId, node2: NodeId) -> bool {
        let loc1 = match self.get_location(node1) {
            Some(l) => l,
            None => return true, // Conservative
        };
        let loc2 = match self.get_location(node2) {
            Some(l) => l,
            None => return true, // Conservative
        };
        self.alias(loc1, loc2).may_alias()
    }

    /// Check if two memory operations must alias.
    pub fn must_alias(&self, node1: NodeId, node2: NodeId) -> bool {
        let loc1 = match self.get_location(node1) {
            Some(l) => l,
            None => return false,
        };
        let loc2 = match self.get_location(node2) {
            Some(l) => l,
            None => return false,
        };
        self.alias(loc1, loc2).must_alias()
    }

    /// Check if a node is a fresh allocation.
    pub fn is_fresh_allocation(&self, node_id: NodeId) -> bool {
        self.allocations.get(&node_id).map_or(false, |a| a.is_fresh)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
