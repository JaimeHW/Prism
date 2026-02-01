//! Guard node builder for speculative type guards.
//!
//! This module provides the `GuardBuilder` extension for constructing
//! type guards, bounds checks, and null checks in the IR. Guards are
//! essential for speculative optimization:
//!
//! ```text
//!     ┌──────────┐
//!     │ value    │
//!     └────┬─────┘
//!          │
//!     ╔════▼════╗
//!     ║TypeGuard║ expected_type = int
//!     ╚════╤════╝
//!          │
//!     ┌────▼─────┐           ┌────────────┐
//!     │ success  │           │ deopt stub │
//!     └──────────┘           └────────────┘
//! ```
//!
//! # Guard Semantics
//!
//! - **Type Guard**: Checks that a value has the expected runtime type.
//!   On failure, triggers deoptimization back to the interpreter.
//!
//! - **Bounds Guard**: Checks that an index is within valid range.
//!   JIT can eliminate these when index is known at compile time.
//!
//! - **NotNull Guard**: Checks that a value is not None.
//!   Enables safe attribute access and method calls.
//!
//! # Performance
//!
//! Guards are designed to be as cheap as possible on the fast path:
//! - Type check: Single compare + predicted branch (1-2 cycles)
//! - Bounds check: Compare + branch (1-2 cycles)
//! - NotNull: Tag check or pointer compare (1 cycle)

use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{GuardKind, Operator};
use crate::ir::types::ValueType;

// =============================================================================
// Guard Reason
// =============================================================================

/// Reason for a guard to deoptimize.
///
/// These values are encoded in the deopt metadata to help the profiler
/// understand why speculation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GuardReason {
    /// Type mismatch (expected type X, got type Y).
    TypeMismatch = 0,
    /// Index out of bounds.
    OutOfBounds = 1,
    /// Unexpected None/null value.
    UnexpectedNull = 2,
    /// Arithmetic overflow.
    Overflow = 3,
    /// Division by zero.
    DivisionByZero = 4,
    /// IC miss (inline cache prediction failed).
    ICMiss = 5,
    /// OSR state mismatch.
    OsrMismatch = 6,
    /// Class/shape changed (hidden class invalidation).
    ShapeChange = 7,
    /// Generic deoptimization.
    Generic = 255,
}

impl GuardReason {
    /// Get a human-readable name for this reason.
    pub const fn name(self) -> &'static str {
        match self {
            GuardReason::TypeMismatch => "type_mismatch",
            GuardReason::OutOfBounds => "out_of_bounds",
            GuardReason::UnexpectedNull => "unexpected_null",
            GuardReason::Overflow => "overflow",
            GuardReason::DivisionByZero => "division_by_zero",
            GuardReason::ICMiss => "ic_miss",
            GuardReason::OsrMismatch => "osr_mismatch",
            GuardReason::ShapeChange => "shape_change",
            GuardReason::Generic => "generic",
        }
    }
}

// =============================================================================
// Expected Type
// =============================================================================

/// Expected type for type guards.
///
/// This is the speculation target - what type we expect the value to be.
/// Uses the same representation as the runtime TypeId for fast comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExpectedType {
    /// Integer (i64 in NaN-boxing).
    Int = 1,
    /// Float (f64).
    Float = 2,
    /// Boolean.
    Bool = 3,
    /// String (interned).
    String = 4,
    /// List object.
    List = 5,
    /// Tuple object.
    Tuple = 6,
    /// Dict object.
    Dict = 7,
    /// Set object.
    Set = 8,
    /// Function object.
    Function = 9,
    /// Any object (non-primitive).
    Object = 10,
    /// Numeric (int or float).
    Numeric = 11,
    /// Sequence (list, tuple, or string).
    Sequence = 12,
    /// Container (list, dict, set, tuple).
    Container = 13,
}

impl ExpectedType {
    /// Convert to ValueType for IR type inference.
    pub fn to_value_type(self) -> ValueType {
        match self {
            ExpectedType::Int => ValueType::Int64,
            ExpectedType::Float => ValueType::Float64,
            ExpectedType::Bool => ValueType::Bool,
            ExpectedType::String => ValueType::String,
            ExpectedType::List => ValueType::List,
            ExpectedType::Tuple => ValueType::Tuple,
            ExpectedType::Dict => ValueType::Dict,
            ExpectedType::Numeric => ValueType::Numeric,
            _ => ValueType::Object,
        }
    }

    /// Get the runtime TypeId value for this expected type.
    pub fn type_id(self) -> u32 {
        self as u32
    }
}

// =============================================================================
// Guard Metadata
// =============================================================================

/// Metadata for a guard node.
///
/// This is attached to guard nodes for use during lowering and deoptimization.
#[derive(Debug, Clone, Copy)]
pub struct GuardMetadata {
    /// Expected type (for type guards).
    pub expected_type: Option<ExpectedType>,
    /// Bytecode offset where this guard was inserted.
    pub bc_offset: u32,
    /// Reason code for deoptimization.
    pub reason: GuardReason,
    /// Whether this guard is "hoisted" (moved out of a loop).
    pub is_hoisted: bool,
    /// Counter for how many times this guard has deoptimized.
    /// Used to decide when to stop speculating on this path.
    pub deopt_count: u32,
}

impl Default for GuardMetadata {
    fn default() -> Self {
        Self {
            expected_type: None,
            bc_offset: 0,
            reason: GuardReason::Generic,
            is_hoisted: false,
            deopt_count: 0,
        }
    }
}

// =============================================================================
// Guard Builder Trait
// =============================================================================

/// Extension trait for building guard nodes.
pub trait GuardBuilder {
    /// Create a type guard that checks if a value has the expected type.
    ///
    /// Returns the guarded value (with narrowed type) on success.
    /// On failure, control flows to a deoptimization stub.
    fn type_guard(
        &mut self,
        value: NodeId,
        expected: ExpectedType,
        control: NodeId,
        bc_offset: u32,
    ) -> (NodeId, NodeId); // (guarded_value, new_control)

    /// Create a bounds guard that checks if index is within [0, length).
    ///
    /// Returns the verified index on success.
    fn bounds_guard(
        &mut self,
        index: NodeId,
        length: NodeId,
        control: NodeId,
        bc_offset: u32,
    ) -> (NodeId, NodeId);

    /// Create a not-null guard that checks if value is not None.
    ///
    /// Returns the non-null value on success.
    fn not_null_guard(
        &mut self,
        value: NodeId,
        control: NodeId,
        bc_offset: u32,
    ) -> (NodeId, NodeId);

    /// Create an overflow guard for arithmetic operations.
    fn overflow_guard(
        &mut self,
        result: NodeId,
        control: NodeId,
        bc_offset: u32,
    ) -> (NodeId, NodeId);

    /// Create a non-zero divisor guard.
    fn non_zero_guard(
        &mut self,
        divisor: NodeId,
        control: NodeId,
        bc_offset: u32,
    ) -> (NodeId, NodeId);
}

impl GuardBuilder for Graph {
    fn type_guard(
        &mut self,
        value: NodeId,
        expected: ExpectedType,
        control: NodeId,
        _bc_offset: u32,
    ) -> (NodeId, NodeId) {
        // Create the guard node with type check semantics
        let guard = self.add_node_with_type(
            Operator::Guard(GuardKind::Type),
            InputList::Pair(control, value),
            expected.to_value_type(),
        );

        // The guard node itself serves as both the control output
        // (for scheduling) and the value output (with narrowed type).
        // In practice, lowering will emit:
        //   cmp [value+0], expected_type_id  ; check type tag
        //   jne deopt_stub
        //
        // For now, return the guard as both the value and control.
        // The deopt path is implicit - lowering handles it.
        (guard, guard)
    }

    fn bounds_guard(
        &mut self,
        index: NodeId,
        length: NodeId,
        control: NodeId,
        _bc_offset: u32,
    ) -> (NodeId, NodeId) {
        // Create bounds check guard
        // Inputs: control, index, length
        let guard = self.add_node_with_type(
            Operator::Guard(GuardKind::Bounds),
            InputList::from_slice(&[control, index, length]),
            ValueType::Int64, // Index is always int
        );

        (guard, guard)
    }

    fn not_null_guard(
        &mut self,
        value: NodeId,
        control: NodeId,
        _bc_offset: u32,
    ) -> (NodeId, NodeId) {
        // Create null check guard
        let guard = self.add_node_with_type(
            Operator::Guard(GuardKind::NotNull),
            InputList::Pair(control, value),
            self.node(value).ty, // Preserve input type
        );

        (guard, guard)
    }

    fn overflow_guard(
        &mut self,
        result: NodeId,
        control: NodeId,
        _bc_offset: u32,
    ) -> (NodeId, NodeId) {
        let guard = self.add_node_with_type(
            Operator::Guard(GuardKind::Overflow),
            InputList::Pair(control, result),
            ValueType::Int64,
        );

        (guard, guard)
    }

    fn non_zero_guard(
        &mut self,
        divisor: NodeId,
        control: NodeId,
        _bc_offset: u32,
    ) -> (NodeId, NodeId) {
        let guard = self.add_node_with_type(
            Operator::Guard(GuardKind::NonZeroDivisor),
            InputList::Pair(control, divisor),
            self.node(divisor).ty,
        );

        (guard, guard)
    }
}

// =============================================================================
// Guard Elimination
// =============================================================================

/// Analysis for eliminating redundant guards.
///
/// Guards can be eliminated when:
/// 1. The type is already proven by a dominating guard
/// 2. The bounds are statically known to be safe
/// 3. A value is known to be non-null from context
pub struct GuardEliminator {
    /// Known types per value (from dominating guards).
    known_types: std::collections::HashMap<NodeId, ExpectedType>,
    /// Guards that have been marked as redundant.
    redundant: Vec<NodeId>,
}

impl GuardEliminator {
    /// Create a new guard eliminator.
    pub fn new() -> Self {
        Self {
            known_types: Default::default(),
            redundant: Vec::new(),
        }
    }

    /// Record that a value has a known type after a guard.
    pub fn record_type(&mut self, value: NodeId, ty: ExpectedType) {
        self.known_types.insert(value, ty);
    }

    /// Check if a type guard is redundant.
    pub fn is_type_guard_redundant(&self, value: NodeId, expected: ExpectedType) -> bool {
        self.known_types
            .get(&value)
            .map(|&known| known == expected)
            .unwrap_or(false)
    }

    /// Get all redundant guards.
    pub fn redundant_guards(&self) -> &[NodeId] {
        &self.redundant
    }
}

impl Default for GuardEliminator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Speculative Type State
// =============================================================================

/// Tracks speculative type information for values.
///
/// This is used during IR construction to track what types we're speculating
/// about, and to insert appropriate guards.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeTypeState {
    /// Map from value to speculated type.
    speculation: std::collections::HashMap<NodeId, ExpectedType>,
    /// Values that have been guarded.
    guarded: std::collections::HashSet<NodeId>,
}

impl SpeculativeTypeState {
    /// Create a new speculative state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a speculation about a value's type.
    pub fn speculate(&mut self, value: NodeId, ty: ExpectedType) {
        self.speculation.insert(value, ty);
    }

    /// Record that a value has been guarded.
    pub fn mark_guarded(&mut self, value: NodeId) {
        self.guarded.insert(value);
    }

    /// Check if a value needs a guard.
    pub fn needs_guard(&self, value: NodeId) -> bool {
        self.speculation.contains_key(&value) && !self.guarded.contains(&value)
    }

    /// Get the speculated type for a value.
    pub fn get_speculation(&self, value: NodeId) -> Option<ExpectedType> {
        self.speculation.get(&value).copied()
    }

    /// Clear all speculation (e.g., at a merge point).
    pub fn clear(&mut self) {
        self.speculation.clear();
        self.guarded.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_reason_names() {
        assert_eq!(GuardReason::TypeMismatch.name(), "type_mismatch");
        assert_eq!(GuardReason::OutOfBounds.name(), "out_of_bounds");
        assert_eq!(GuardReason::DivisionByZero.name(), "division_by_zero");
    }

    #[test]
    fn test_expected_type_conversion() {
        assert_eq!(ExpectedType::Int.to_value_type(), ValueType::Int64);
        assert_eq!(ExpectedType::Float.to_value_type(), ValueType::Float64);
        assert_eq!(ExpectedType::Bool.to_value_type(), ValueType::Bool);
        assert_eq!(ExpectedType::List.to_value_type(), ValueType::List);
    }

    #[test]
    fn test_type_guard_creation() {
        let mut graph = Graph::new();

        // Create a parameter to guard
        let param = graph.add_node(Operator::Parameter(0), InputList::Single(graph.start));

        // Create type guard
        let (guarded, _control) = graph.type_guard(param, ExpectedType::Int, graph.start, 0);

        // Check that guard was created
        assert!(guarded.is_valid());
        assert!(matches!(
            graph.node(guarded).op,
            Operator::Guard(GuardKind::Type)
        ));
    }

    #[test]
    fn test_guard_eliminator() {
        let mut eliminator = GuardEliminator::new();

        let value = NodeId::new(1);

        // Initially no known type
        assert!(!eliminator.is_type_guard_redundant(value, ExpectedType::Int));

        // Record known type from guard
        eliminator.record_type(value, ExpectedType::Int);

        // Same type guard is now redundant
        assert!(eliminator.is_type_guard_redundant(value, ExpectedType::Int));

        // Different type guard is not redundant
        assert!(!eliminator.is_type_guard_redundant(value, ExpectedType::Float));
    }

    #[test]
    fn test_speculative_state() {
        let mut state = SpeculativeTypeState::new();

        let value = NodeId::new(1);

        // Speculate about type
        state.speculate(value, ExpectedType::Int);
        assert!(state.needs_guard(value));

        // Mark as guarded
        state.mark_guarded(value);
        assert!(!state.needs_guard(value));

        // Can still get speculation
        assert_eq!(state.get_speculation(value), Some(ExpectedType::Int));
    }
}
