//! Escape Analysis.
//!
//! Escape analysis determines which objects can be allocated on the stack
//! instead of the heap, and which can be eliminated entirely through
//! scalar replacement.

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ControlOp, MemoryOp, Operator};

use std::collections::{HashMap, HashSet, VecDeque};

// =============================================================================
// Escape State
// =============================================================================

/// The escape state of an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EscapeState {
    /// Object does not escape the function.
    NoEscape,
    /// Object escapes via arguments to called methods.
    ArgEscape,
    /// Object escapes globally (stored to heap, returned, etc.).
    GlobalEscape,
}

impl EscapeState {
    /// Check if the object can be stack-allocated.
    #[inline]
    pub fn can_stack_allocate(self) -> bool {
        matches!(self, EscapeState::NoEscape | EscapeState::ArgEscape)
    }

    /// Check if the object can be scalar-replaced.
    #[inline]
    pub fn can_scalar_replace(self) -> bool {
        self == EscapeState::NoEscape
    }

    /// Merge two escape states (takes the more conservative one).
    #[inline]
    pub fn merge(self, other: EscapeState) -> EscapeState {
        std::cmp::max(self, other)
    }
}

impl Default for EscapeState {
    fn default() -> Self {
        EscapeState::NoEscape
    }
}

// =============================================================================
// Allocation Info
// =============================================================================

/// Information about an allocation site.
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Node ID of the allocation.
    pub node: NodeId,
    /// Computed escape state.
    pub escape_state: EscapeState,
    /// Type of the allocated object (if known).
    pub object_type: Option<ObjectType>,
    /// Whether all uses are known.
    pub all_uses_known: bool,
}

/// Type of allocated object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Array,
    Dict,
    Object,
    Closure,
    Tuple,
}

// =============================================================================
// Escape Analysis
// =============================================================================

/// Escape analysis result.
#[derive(Debug)]
pub struct EscapeAnalysis {
    /// Escape state per allocation node.
    states: HashMap<NodeId, EscapeState>,
    /// Detailed info per allocation.
    allocations: Vec<AllocationInfo>,
    /// Number of non-escaping allocations.
    non_escaping: usize,
    /// Number of arg-escaping allocations.
    arg_escaping: usize,
    /// Number of globally-escaping allocations.
    global_escaping: usize,
}

impl EscapeAnalysis {
    /// Compute escape analysis for a graph.
    pub fn compute(graph: &Graph) -> Self {
        let mut analysis = Self {
            states: HashMap::new(),
            allocations: Vec::new(),
            non_escaping: 0,
            arg_escaping: 0,
            global_escaping: 0,
        };

        // Find all allocation sites
        let alloc_sites: Vec<NodeId> = graph
            .iter()
            .filter_map(|(id, node)| {
                if Self::is_allocation(&node.op) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();

        // Analyze each allocation
        for alloc in alloc_sites {
            let (state, info) = Self::analyze_allocation(graph, alloc);

            match state {
                EscapeState::NoEscape => analysis.non_escaping += 1,
                EscapeState::ArgEscape => analysis.arg_escaping += 1,
                EscapeState::GlobalEscape => analysis.global_escaping += 1,
            }

            analysis.states.insert(alloc, state);
            analysis.allocations.push(info);
        }

        analysis
    }

    /// Check if an operator is an allocation.
    #[inline]
    fn is_allocation(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(MemoryOp::Alloc) | Operator::Memory(MemoryOp::AllocArray)
        )
    }

    /// Analyze a single allocation site.
    fn analyze_allocation(graph: &Graph, alloc: NodeId) -> (EscapeState, AllocationInfo) {
        let mut state = EscapeState::NoEscape;
        let mut all_uses_known = true;

        // BFS through all uses
        let mut visited = HashSet::new();
        let mut worklist = VecDeque::new();

        worklist.push_back(alloc);
        visited.insert(alloc);

        while let Some(node_id) = worklist.pop_front() {
            for &user_id in graph.uses(node_id) {
                if visited.contains(&user_id) {
                    continue;
                }
                visited.insert(user_id);

                let user = graph.node(user_id);
                let use_escape = Self::classify_use(&user.op);

                state = state.merge(use_escape);

                // If used as input, propagate to users of this node
                if use_escape != EscapeState::GlobalEscape && Self::should_propagate(&user.op) {
                    worklist.push_back(user_id);
                }

                if !Self::is_known_use(&user.op) {
                    all_uses_known = false;
                }
            }
        }

        let info = AllocationInfo {
            node: alloc,
            escape_state: state,
            object_type: Self::infer_object_type(graph, alloc),
            all_uses_known,
        };

        (state, info)
    }

    /// Classify a use of an allocated object.
    fn classify_use(op: &Operator) -> EscapeState {
        match op {
            // Field load doesn't cause escape
            Operator::Memory(MemoryOp::LoadField) | Operator::Memory(MemoryOp::LoadElement) => {
                EscapeState::NoEscape
            }
            // Field store - conservative
            Operator::Memory(MemoryOp::StoreField) | Operator::Memory(MemoryOp::StoreElement) => {
                EscapeState::GlobalEscape
            }
            // Return - escapes globally
            Operator::Control(ControlOp::Return) => EscapeState::GlobalEscape,
            // Phi - propagate
            Operator::Phi | Operator::LoopPhi => EscapeState::NoEscape,
            // Call - arg escape
            Operator::Call(_) => EscapeState::ArgEscape,
            // Guards/type checks - don't cause escape
            Operator::Guard(_) | Operator::TypeCheck => EscapeState::NoEscape,
            // Comparison - doesn't cause escape
            Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => {
                EscapeState::NoEscape
            }
            // Unknown - conservative
            _ => EscapeState::GlobalEscape,
        }
    }

    /// Check if we should propagate through an operation.
    fn should_propagate(op: &Operator) -> bool {
        matches!(op, Operator::Phi | Operator::LoopPhi)
    }

    /// Check if a use is known/understood.
    fn is_known_use(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(_)
                | Operator::Control(_)
                | Operator::IntCmp(_)
                | Operator::FloatCmp(_)
                | Operator::GenericCmp(_)
                | Operator::Guard(_)
                | Operator::TypeCheck
                | Operator::Call(_)
                | Operator::Phi
                | Operator::LoopPhi
        )
    }

    /// Infer object type from allocation.
    fn infer_object_type(graph: &Graph, alloc: NodeId) -> Option<ObjectType> {
        let node = graph.node(alloc);
        match node.op {
            Operator::Memory(MemoryOp::AllocArray) => Some(ObjectType::Array),
            Operator::Memory(MemoryOp::Alloc) => Some(ObjectType::Object),
            Operator::BuildList(_) => Some(ObjectType::Array),
            Operator::BuildTuple(_) => Some(ObjectType::Tuple),
            Operator::BuildDict(_) => Some(ObjectType::Dict),
            _ => None,
        }
    }

    // =========================================================================
    // Query API
    // =========================================================================

    /// Get escape state for an allocation.
    #[inline]
    pub fn escape_state(&self, node: NodeId) -> Option<EscapeState> {
        self.states.get(&node).copied()
    }

    /// Check if an allocation can be stack-allocated.
    #[inline]
    pub fn can_stack_allocate(&self, node: NodeId) -> bool {
        self.states
            .get(&node)
            .map(|s| s.can_stack_allocate())
            .unwrap_or(false)
    }

    /// Check if an allocation can be scalar-replaced.
    #[inline]
    pub fn can_scalar_replace(&self, node: NodeId) -> bool {
        self.states
            .get(&node)
            .map(|s| s.can_scalar_replace())
            .unwrap_or(false)
    }

    /// Get all allocations that can be stack-allocated.
    pub fn stack_allocatable(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.states
            .iter()
            .filter(|(_, s)| s.can_stack_allocate())
            .map(|(n, _)| *n)
    }

    /// Get all allocations that can be scalar-replaced.
    pub fn scalar_replaceable(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.states
            .iter()
            .filter(|(_, s)| s.can_scalar_replace())
            .map(|(n, _)| *n)
    }

    /// Get detailed allocation info.
    pub fn allocations(&self) -> &[AllocationInfo] {
        &self.allocations
    }

    /// Get number of non-escaping allocations.
    #[inline]
    pub fn non_escaping_count(&self) -> usize {
        self.non_escaping
    }

    /// Get number of arg-escaping allocations.
    #[inline]
    pub fn arg_escaping_count(&self) -> usize {
        self.arg_escaping
    }

    /// Get number of globally-escaping allocations.
    #[inline]
    pub fn global_escaping_count(&self) -> usize {
        self.global_escaping
    }
}

// =============================================================================
// Escape Analysis Pass
// =============================================================================

/// Escape analysis optimization pass.
#[derive(Debug)]
pub struct Escape {
    /// Number of allocations stack-allocated.
    stack_allocated: usize,
    /// Number of allocations scalar-replaced.
    scalar_replaced: usize,
    /// Enable aggressive optimization.
    aggressive: bool,
}

impl Escape {
    /// Create a new escape analysis pass.
    pub fn new() -> Self {
        Self {
            stack_allocated: 0,
            scalar_replaced: 0,
            aggressive: false,
        }
    }

    /// Create an aggressive escape analysis pass.
    pub fn aggressive() -> Self {
        Self {
            stack_allocated: 0,
            scalar_replaced: 0,
            aggressive: true,
        }
    }

    /// Get number of stack allocations.
    #[inline]
    pub fn stack_allocated(&self) -> usize {
        self.stack_allocated
    }

    /// Get number of scalar replacements.
    #[inline]
    pub fn scalar_replaced(&self) -> usize {
        self.scalar_replaced
    }

    /// Run escape analysis pass.
    fn run_escape(&mut self, graph: &mut Graph) -> bool {
        let analysis = EscapeAnalysis::compute(graph);

        let mut changed = false;

        for alloc in analysis.scalar_replaceable() {
            if self.try_scalar_replace(graph, alloc, &analysis) {
                self.scalar_replaced += 1;
                changed = true;
            }
        }

        for alloc in analysis.stack_allocatable() {
            if self.try_stack_allocate(graph, alloc, &analysis) {
                self.stack_allocated += 1;
                changed = true;
            }
        }

        changed
    }

    fn try_scalar_replace(
        &self,
        _graph: &mut Graph,
        _alloc: NodeId,
        _analysis: &EscapeAnalysis,
    ) -> bool {
        false
    }

    fn try_stack_allocate(
        &self,
        _graph: &mut Graph,
        _alloc: NodeId,
        _analysis: &EscapeAnalysis,
    ) -> bool {
        false
    }
}

impl Default for Escape {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Escape {
    fn name(&self) -> &'static str {
        "escape"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_escape(graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ControlBuilder, GraphBuilder};

    #[test]
    fn test_escape_state_ordering() {
        assert!(EscapeState::NoEscape < EscapeState::ArgEscape);
        assert!(EscapeState::ArgEscape < EscapeState::GlobalEscape);
    }

    #[test]
    fn test_escape_state_merge() {
        assert_eq!(
            EscapeState::NoEscape.merge(EscapeState::NoEscape),
            EscapeState::NoEscape
        );
        assert_eq!(
            EscapeState::NoEscape.merge(EscapeState::ArgEscape),
            EscapeState::ArgEscape
        );
        assert_eq!(
            EscapeState::ArgEscape.merge(EscapeState::GlobalEscape),
            EscapeState::GlobalEscape
        );
    }

    #[test]
    fn test_escape_state_can_stack_allocate() {
        assert!(EscapeState::NoEscape.can_stack_allocate());
        assert!(EscapeState::ArgEscape.can_stack_allocate());
        assert!(!EscapeState::GlobalEscape.can_stack_allocate());
    }

    #[test]
    fn test_escape_state_can_scalar_replace() {
        assert!(EscapeState::NoEscape.can_scalar_replace());
        assert!(!EscapeState::ArgEscape.can_scalar_replace());
        assert!(!EscapeState::GlobalEscape.can_scalar_replace());
    }

    #[test]
    fn test_escape_analysis_empty() {
        let builder = GraphBuilder::new(0, 0);
        let graph = builder.finish();

        let analysis = EscapeAnalysis::compute(&graph);
        assert_eq!(analysis.non_escaping_count(), 0);
        assert_eq!(analysis.arg_escaping_count(), 0);
        assert_eq!(analysis.global_escaping_count(), 0);
    }

    #[test]
    fn test_escape_pass_new() {
        let escape = Escape::new();
        assert_eq!(escape.stack_allocated(), 0);
        assert_eq!(escape.scalar_replaced(), 0);
        assert!(!escape.aggressive);
    }

    #[test]
    fn test_escape_pass_aggressive() {
        let escape = Escape::aggressive();
        assert!(escape.aggressive);
    }

    #[test]
    fn test_escape_pass_name() {
        let escape = Escape::new();
        assert_eq!(escape.name(), "escape");
    }

    #[test]
    fn test_escape_no_allocs() {
        let mut builder = GraphBuilder::new(1, 1);
        let p0 = builder.parameter(0).unwrap();
        builder.return_value(p0);

        let mut graph = builder.finish();
        let mut escape = Escape::new();

        let changed = escape.run(&mut graph);
        assert!(!changed);
    }

    #[test]
    fn test_object_type() {
        assert_eq!(ObjectType::Array, ObjectType::Array);
        assert_ne!(ObjectType::Array, ObjectType::Dict);
    }
}
