//! Field Access Tracking for Scalar Replacement
//!
//! This module tracks all field accesses (loads and stores) on allocated objects
//! to determine if scalar replacement is possible. For scalar replacement to work:
//!
//! 1. All accessed fields must be known at compile time (constant field indices)
//! 2. All field accesses must be trackable loads/stores
//! 3. No aliasing through unknown pointers
//!
//! # Algorithm
//!
//! For each non-escaping allocation:
//! 1. Collect all StoreField operations (writes to the object)
//! 2. Collect all LoadField operations (reads from the object)
//! 3. Build a mapping of field index -> SSA values at each program point
//! 4. For each load, find the corresponding store(s) and create phi if needed
//!
//! # Data Structures
//!
//! - `FieldAccess`: Represents a single field load or store
//! - `FieldMap`: Maps field indices to their current SSA values
//! - `FieldTracker`: Collects and analyzes all field accesses for an allocation

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator};
use crate::ir::types::ValueType;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

// =============================================================================
// Field Index
// =============================================================================

/// A field index - either a constant offset or dynamic (unknown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldIndex {
    /// Constant field index (known at compile time).
    Constant(u32),
    /// Dynamic index (computed at runtime) - prevents scalar replacement.
    Dynamic,
}

impl FieldIndex {
    /// Check if this is a constant index.
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, FieldIndex::Constant(_))
    }

    /// Get the constant value if available.
    #[inline]
    pub fn as_constant(&self) -> Option<u32> {
        match self {
            FieldIndex::Constant(idx) => Some(*idx),
            FieldIndex::Dynamic => None,
        }
    }
}

// =============================================================================
// Field Access
// =============================================================================

/// Kind of field access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldAccessKind {
    /// Load from field.
    Load,
    /// Store to field.
    Store,
}

/// A single field access operation.
#[derive(Debug, Clone)]
pub struct FieldAccess {
    /// The node performing the access.
    pub node: NodeId,
    /// The allocation being accessed.
    pub object: NodeId,
    /// The field being accessed.
    pub field: FieldIndex,
    /// Kind of access (load/store).
    pub kind: FieldAccessKind,
    /// For stores: the value being stored. For loads: None.
    pub value: Option<NodeId>,
    /// The type of the field value.
    pub value_type: ValueType,
    /// Control dependency (for ordering).
    pub control: Option<NodeId>,
}

impl FieldAccess {
    /// Check if this is a load.
    #[inline]
    pub fn is_load(&self) -> bool {
        self.kind == FieldAccessKind::Load
    }

    /// Check if this is a store.
    #[inline]
    pub fn is_store(&self) -> bool {
        self.kind == FieldAccessKind::Store
    }
}

// =============================================================================
// Field State
// =============================================================================

/// State of a field at a particular program point.
#[derive(Debug, Clone)]
pub enum FieldState {
    /// Field has not been written yet (uninitialized).
    Uninitialized,
    /// Field has a known SSA value.
    Value(NodeId),
    /// Field has multiple possible values (needs phi).
    Phi(SmallVec<[NodeId; 4]>),
}

impl FieldState {
    /// Get the SSA value if uniquely known.
    pub fn as_value(&self) -> Option<NodeId> {
        match self {
            FieldState::Value(v) => Some(*v),
            _ => None,
        }
    }

    /// Get all possible values.
    pub fn values(&self) -> SmallVec<[NodeId; 4]> {
        match self {
            FieldState::Uninitialized => SmallVec::new(),
            FieldState::Value(v) => {
                let mut sv = SmallVec::new();
                sv.push(*v);
                sv
            }
            FieldState::Phi(values) => values.clone(),
        }
    }

    /// Merge two field states.
    pub fn merge(&self, other: &FieldState) -> FieldState {
        match (self, other) {
            (FieldState::Uninitialized, other) => other.clone(),
            (this, FieldState::Uninitialized) => this.clone(),
            (FieldState::Value(v1), FieldState::Value(v2)) => {
                if v1 == v2 {
                    FieldState::Value(*v1)
                } else {
                    let mut values = SmallVec::new();
                    values.push(*v1);
                    values.push(*v2);
                    FieldState::Phi(values)
                }
            }
            (FieldState::Value(v), FieldState::Phi(values))
            | (FieldState::Phi(values), FieldState::Value(v)) => {
                let mut new_values = values.clone();
                if !new_values.contains(v) {
                    new_values.push(*v);
                }
                FieldState::Phi(new_values)
            }
            (FieldState::Phi(v1), FieldState::Phi(v2)) => {
                let mut new_values = v1.clone();
                for v in v2 {
                    if !new_values.contains(v) {
                        new_values.push(*v);
                    }
                }
                FieldState::Phi(new_values)
            }
        }
    }
}

// =============================================================================
// Field Map
// =============================================================================

/// Maps field indices to their current SSA values.
#[derive(Debug, Clone, Default)]
pub struct FieldMap {
    /// Field index -> current state.
    fields: FxHashMap<u32, FieldState>,
    /// Whether any dynamic access was seen (prevents scalar replacement).
    has_dynamic_access: bool,
    /// Maximum field index seen (for sizing).
    max_field: u32,
}

impl FieldMap {
    /// Create a new empty field map.
    pub fn new() -> Self {
        Self {
            fields: FxHashMap::default(),
            has_dynamic_access: false,
            max_field: 0,
        }
    }

    /// Record a store to a field.
    pub fn store(&mut self, field: FieldIndex, value: NodeId) {
        match field {
            FieldIndex::Constant(idx) => {
                self.fields.insert(idx, FieldState::Value(value));
                self.max_field = self.max_field.max(idx);
            }
            FieldIndex::Dynamic => {
                self.has_dynamic_access = true;
            }
        }
    }

    /// Get the current value of a field.
    pub fn load(&self, field: FieldIndex) -> Option<FieldState> {
        match field {
            FieldIndex::Constant(idx) => self.fields.get(&idx).cloned(),
            FieldIndex::Dynamic => None,
        }
    }

    /// Check if scalar replacement is possible.
    #[inline]
    pub fn can_scalar_replace(&self) -> bool {
        !self.has_dynamic_access
    }

    /// Get all field indices.
    pub fn field_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.fields.keys().copied()
    }

    /// Get the number of fields.
    #[inline]
    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    /// Get maximum field index.
    #[inline]
    pub fn max_field_index(&self) -> u32 {
        self.max_field
    }

    /// Merge with another field map (for control flow merge points).
    pub fn merge(&mut self, other: &FieldMap) {
        self.has_dynamic_access |= other.has_dynamic_access;
        self.max_field = self.max_field.max(other.max_field);

        // Merge field states
        for (&idx, other_state) in &other.fields {
            let new_state = match self.fields.get(&idx) {
                Some(self_state) => self_state.merge(other_state),
                None => other_state.clone(),
            };
            self.fields.insert(idx, new_state);
        }
    }

    /// Get field state.
    pub fn get(&self, idx: u32) -> Option<&FieldState> {
        self.fields.get(&idx)
    }

    /// Check if a field has been written.
    pub fn is_initialized(&self, idx: u32) -> bool {
        self.fields.contains_key(&idx)
    }
}

// =============================================================================
// Field Tracker
// =============================================================================

/// Tracks all field accesses for an allocation.
#[derive(Debug)]
pub struct FieldTracker {
    /// The allocation being tracked.
    pub allocation: NodeId,
    /// All field accesses.
    pub accesses: Vec<FieldAccess>,
    /// Stores grouped by field index.
    pub stores_by_field: FxHashMap<u32, SmallVec<[usize; 4]>>,
    /// Loads grouped by field index.
    pub loads_by_field: FxHashMap<u32, SmallVec<[usize; 4]>>,
    /// Whether any dynamic access exists.
    pub has_dynamic_access: bool,
    /// Whether the object has unknown uses.
    pub has_unknown_uses: bool,
    /// Field types (inferred from stores).
    pub field_types: FxHashMap<u32, ValueType>,
}

impl FieldTracker {
    /// Create a new field tracker for an allocation.
    pub fn new(allocation: NodeId) -> Self {
        Self {
            allocation,
            accesses: Vec::new(),
            stores_by_field: FxHashMap::default(),
            loads_by_field: FxHashMap::default(),
            has_dynamic_access: false,
            has_unknown_uses: false,
            field_types: FxHashMap::default(),
        }
    }

    /// Analyze an allocation to track all field accesses.
    pub fn analyze(graph: &Graph, allocation: NodeId) -> Self {
        let mut tracker = Self::new(allocation);
        tracker.collect_accesses(graph);
        tracker
    }

    /// Collect all field accesses to this allocation.
    fn collect_accesses(&mut self, graph: &Graph) {
        // Find all uses of the allocation
        let uses: Vec<NodeId> = graph.uses(self.allocation).to_vec();

        for use_id in uses {
            let user = match graph.get(use_id) {
                Some(n) => n,
                None => continue,
            };

            match user.op {
                Operator::Memory(MemoryOp::LoadField) => {
                    self.process_load_field(graph, use_id);
                }
                Operator::Memory(MemoryOp::StoreField) => {
                    self.process_store_field(graph, use_id);
                }
                Operator::Memory(MemoryOp::LoadElement) => {
                    self.process_load_element(graph, use_id);
                }
                Operator::Memory(MemoryOp::StoreElement) => {
                    self.process_store_element(graph, use_id);
                }
                Operator::Phi | Operator::LoopPhi => {
                    // Phi nodes are OK - the allocation flows through
                }
                Operator::Guard(_) | Operator::TypeCheck => {
                    // Guards are OK - just checking the object
                }
                Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => {
                    // Comparisons are OK (e.g., is None check)
                }
                _ => {
                    // Unknown use - might escape or be modified in unknown ways
                    self.has_unknown_uses = true;
                }
            }
        }
    }

    /// Process a LoadField operation.
    fn process_load_field(&mut self, graph: &Graph, node: NodeId) {
        let node_data = match graph.get(node) {
            Some(n) => n,
            None => return,
        };

        // LoadField inputs: [object, field_index]
        let object = node_data.inputs.get(0);
        let field_idx_node = node_data.inputs.get(1);

        if object != Some(self.allocation) {
            return;
        }

        let field = self.extract_field_index(graph, field_idx_node);
        let access_idx = self.accesses.len();

        let access = FieldAccess {
            node,
            object: self.allocation,
            field,
            kind: FieldAccessKind::Load,
            value: None,
            value_type: node_data.ty,
            control: node_data
                .inputs
                .get(0)
                .filter(|&n| graph.get(n).map(|nd| nd.is_control()).unwrap_or(false)),
        };

        self.accesses.push(access);

        match field {
            FieldIndex::Constant(idx) => {
                self.loads_by_field.entry(idx).or_default().push(access_idx);
            }
            FieldIndex::Dynamic => {
                self.has_dynamic_access = true;
            }
        }
    }

    /// Process a StoreField operation.
    fn process_store_field(&mut self, graph: &Graph, node: NodeId) {
        let node_data = match graph.get(node) {
            Some(n) => n,
            None => return,
        };

        // StoreField inputs: [control, object, field_index, value]
        // Or: [object, field_index, value] depending on IR design
        let inputs: Vec<NodeId> = node_data.inputs.iter().collect();

        if inputs.len() < 3 {
            return;
        }

        // Try to find object in inputs
        let (object, field_idx_node, value) = if inputs.len() >= 4 {
            // [control, object, field_index, value]
            (inputs[1], Some(inputs[2]), inputs[3])
        } else {
            // [object, field_index, value]
            (inputs[0], Some(inputs[1]), inputs[2])
        };

        if object != self.allocation {
            return;
        }

        let field = self.extract_field_index(graph, field_idx_node);
        let access_idx = self.accesses.len();

        // Get value type from the stored value
        let value_type = graph.get(value).map(|n| n.ty).unwrap_or(ValueType::Top);

        let access = FieldAccess {
            node,
            object: self.allocation,
            field,
            kind: FieldAccessKind::Store,
            value: Some(value),
            value_type,
            control: if inputs.len() >= 4 {
                Some(inputs[0])
            } else {
                None
            },
        };

        self.accesses.push(access);

        match field {
            FieldIndex::Constant(idx) => {
                self.stores_by_field
                    .entry(idx)
                    .or_default()
                    .push(access_idx);
                // Record field type
                self.field_types.insert(idx, value_type);
            }
            FieldIndex::Dynamic => {
                self.has_dynamic_access = true;
            }
        }
    }

    /// Process a LoadElement operation (array access).
    fn process_load_element(&mut self, graph: &Graph, node: NodeId) {
        let node_data = match graph.get(node) {
            Some(n) => n,
            None => return,
        };

        // LoadElement inputs: [array, index]
        let object = node_data.inputs.get(0);
        let idx_node = node_data.inputs.get(1);

        if object != Some(self.allocation) {
            return;
        }

        let field = self.extract_field_index(graph, idx_node);
        let access_idx = self.accesses.len();

        let access = FieldAccess {
            node,
            object: self.allocation,
            field,
            kind: FieldAccessKind::Load,
            value: None,
            value_type: node_data.ty,
            control: None,
        };

        self.accesses.push(access);

        match field {
            FieldIndex::Constant(idx) => {
                self.loads_by_field.entry(idx).or_default().push(access_idx);
            }
            FieldIndex::Dynamic => {
                self.has_dynamic_access = true;
            }
        }
    }

    /// Process a StoreElement operation (array access).
    fn process_store_element(&mut self, graph: &Graph, node: NodeId) {
        let node_data = match graph.get(node) {
            Some(n) => n,
            None => return,
        };

        // StoreElement inputs: [array, index, value] or [control, array, index, value]
        let inputs: Vec<NodeId> = node_data.inputs.iter().collect();

        if inputs.len() < 3 {
            return;
        }

        let (object, idx_node, value) = if inputs.len() >= 4 {
            (inputs[1], Some(inputs[2]), inputs[3])
        } else {
            (inputs[0], Some(inputs[1]), inputs[2])
        };

        if object != self.allocation {
            return;
        }

        let field = self.extract_field_index(graph, idx_node);
        let access_idx = self.accesses.len();

        let value_type = graph.get(value).map(|n| n.ty).unwrap_or(ValueType::Top);

        let access = FieldAccess {
            node,
            object: self.allocation,
            field,
            kind: FieldAccessKind::Store,
            value: Some(value),
            value_type,
            control: if inputs.len() >= 4 {
                Some(inputs[0])
            } else {
                None
            },
        };

        self.accesses.push(access);

        match field {
            FieldIndex::Constant(idx) => {
                self.stores_by_field
                    .entry(idx)
                    .or_default()
                    .push(access_idx);
                self.field_types.insert(idx, value_type);
            }
            FieldIndex::Dynamic => {
                self.has_dynamic_access = true;
            }
        }
    }

    /// Extract field index from a node (constant or dynamic).
    fn extract_field_index(&self, graph: &Graph, node: Option<NodeId>) -> FieldIndex {
        let node_id = match node {
            Some(n) => n,
            None => return FieldIndex::Dynamic,
        };

        let node_data = match graph.get(node_id) {
            Some(n) => n,
            None => return FieldIndex::Dynamic,
        };

        match node_data.op {
            Operator::ConstInt(val) if val >= 0 && val <= u32::MAX as i64 => {
                FieldIndex::Constant(val as u32)
            }
            _ => FieldIndex::Dynamic,
        }
    }

    /// Check if scalar replacement is possible.
    pub fn can_scalar_replace(&self) -> bool {
        !self.has_dynamic_access && !self.has_unknown_uses && !self.accesses.is_empty()
    }

    /// Get all field indices that are accessed.
    pub fn accessed_fields(&self) -> impl Iterator<Item = u32> + '_ {
        self.stores_by_field
            .keys()
            .chain(self.loads_by_field.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
    }

    /// Get number of distinct fields.
    pub fn num_fields(&self) -> usize {
        self.accessed_fields().count()
    }

    /// Get stores for a field.
    pub fn stores_for_field(&self, field: u32) -> impl Iterator<Item = &FieldAccess> {
        self.stores_by_field
            .get(&field)
            .into_iter()
            .flat_map(|indices| indices.iter().map(|&i| &self.accesses[i]))
    }

    /// Get loads for a field.
    pub fn loads_for_field(&self, field: u32) -> impl Iterator<Item = &FieldAccess> {
        self.loads_by_field
            .get(&field)
            .into_iter()
            .flat_map(|indices| indices.iter().map(|&i| &self.accesses[i]))
    }

    /// Get type for a field (inferred from stores).
    pub fn field_type(&self, field: u32) -> ValueType {
        self.field_types
            .get(&field)
            .copied()
            .unwrap_or(ValueType::Top)
    }
}
