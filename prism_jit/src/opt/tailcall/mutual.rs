//! Mutual Recursion Tail Call Optimization
//!
//! Handles tail calls between mutually recursive functions.
//! Uses SCC detection and trampoline pattern for optimization.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{ControlOp, Operator};

use super::detection::TailCallInfo;

// =============================================================================
// SCC Info
// =============================================================================

/// A strongly connected component of mutually recursive functions.
#[derive(Debug, Clone)]
pub struct SccInfo {
    /// Functions in this SCC.
    pub functions: Vec<u64>,
    /// Tail calls between functions in the SCC.
    pub internal_calls: Vec<(u64, u64)>,
    /// Size of the SCC.
    pub size: usize,
    /// Whether this SCC can be optimized.
    pub optimizable: bool,
}

impl SccInfo {
    /// Create a new SCC with a single function.
    pub fn singleton(func_id: u64) -> Self {
        Self {
            functions: vec![func_id],
            internal_calls: Vec::new(),
            size: 1,
            optimizable: true,
        }
    }

    /// Check if this is a self-recursion SCC (single function).
    pub fn is_self_recursion(&self) -> bool {
        self.size == 1 && !self.internal_calls.is_empty()
    }

    /// Check if this is true mutual recursion (multiple functions).
    pub fn is_mutual_recursion(&self) -> bool {
        self.size > 1
    }

    /// Check if a function is in this SCC.
    pub fn contains(&self, func_id: u64) -> bool {
        self.functions.contains(&func_id)
    }
}

// =============================================================================
// Call Graph
// =============================================================================

/// Simplified call graph for SCC detection.
#[derive(Debug, Default)]
pub struct CallGraph {
    /// Edges: caller -> callees (only tail calls).
    edges: FxHashMap<u64, FxHashSet<u64>>,
    /// Reverse edges: callee -> callers.
    reverse_edges: FxHashMap<u64, FxHashSet<u64>>,
    /// All functions.
    functions: FxHashSet<u64>,
}

impl CallGraph {
    /// Create an empty call graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function to the graph.
    pub fn add_function(&mut self, func_id: u64) {
        self.functions.insert(func_id);
    }

    /// Add a tail call edge.
    pub fn add_tail_call(&mut self, caller: u64, callee: u64) {
        self.functions.insert(caller);
        self.functions.insert(callee);
        self.edges.entry(caller).or_default().insert(callee);
        self.reverse_edges.entry(callee).or_default().insert(caller);
    }

    /// Get callees of a function.
    pub fn callees(&self, func_id: u64) -> impl Iterator<Item = u64> + '_ {
        self.edges
            .get(&func_id)
            .into_iter()
            .flat_map(|s| s.iter().copied())
    }

    /// Get callers of a function.
    pub fn callers(&self, func_id: u64) -> impl Iterator<Item = u64> + '_ {
        self.reverse_edges
            .get(&func_id)
            .into_iter()
            .flat_map(|s| s.iter().copied())
    }

    /// Get all functions.
    pub fn all_functions(&self) -> impl Iterator<Item = u64> + '_ {
        self.functions.iter().copied()
    }

    /// Detect strongly connected components using Tarjan's algorithm.
    pub fn find_sccs(&self) -> Vec<SccInfo> {
        let mut result = Vec::new();
        let mut index_map: FxHashMap<u64, usize> = FxHashMap::default();
        let mut lowlink_map: FxHashMap<u64, usize> = FxHashMap::default();
        let mut on_stack: FxHashSet<u64> = FxHashSet::default();
        let mut stack: Vec<u64> = Vec::new();
        let mut index = 0;

        for &func in &self.functions {
            if !index_map.contains_key(&func) {
                self.tarjan_scc(
                    func,
                    &mut index,
                    &mut index_map,
                    &mut lowlink_map,
                    &mut on_stack,
                    &mut stack,
                    &mut result,
                );
            }
        }

        result
    }

    fn tarjan_scc(
        &self,
        v: u64,
        index: &mut usize,
        index_map: &mut FxHashMap<u64, usize>,
        lowlink_map: &mut FxHashMap<u64, usize>,
        on_stack: &mut FxHashSet<u64>,
        stack: &mut Vec<u64>,
        result: &mut Vec<SccInfo>,
    ) {
        index_map.insert(v, *index);
        lowlink_map.insert(v, *index);
        *index += 1;
        stack.push(v);
        on_stack.insert(v);

        for w in self.callees(v) {
            if !index_map.contains_key(&w) {
                self.tarjan_scc(w, index, index_map, lowlink_map, on_stack, stack, result);
                let v_low = lowlink_map[&v];
                let w_low = lowlink_map[&w];
                lowlink_map.insert(v, v_low.min(w_low));
            } else if on_stack.contains(&w) {
                let v_low = lowlink_map[&v];
                let w_idx = index_map[&w];
                lowlink_map.insert(v, v_low.min(w_idx));
            }
        }

        if lowlink_map[&v] == index_map[&v] {
            let mut scc_funcs = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc_funcs.push(w);
                if w == v {
                    break;
                }
            }

            let scc_set: FxHashSet<_> = scc_funcs.iter().copied().collect();
            let mut internal_calls = Vec::new();
            for &func in &scc_funcs {
                for callee in self.callees(func) {
                    if scc_set.contains(&callee) {
                        internal_calls.push((func, callee));
                    }
                }
            }

            result.push(SccInfo {
                size: scc_funcs.len(),
                functions: scc_funcs,
                internal_calls,
                optimizable: true,
            });
        }
    }
}

// =============================================================================
// Trampoline
// =============================================================================

/// Trampoline state for bouncing between functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrampolineState {
    /// Continue with next function.
    Continue { next_func: u64 },
    /// Return final result.
    Done,
}

/// Trampoline for executing mutually recursive tail calls.
#[derive(Debug)]
pub struct Trampoline {
    /// Map from function ID to dispatch index.
    dispatch_table: FxHashMap<u64, usize>,
    /// Functions in the trampoline.
    functions: Vec<u64>,
}

impl Trampoline {
    /// Create a new trampoline for an SCC.
    pub fn new(scc: &SccInfo) -> Self {
        let dispatch_table: FxHashMap<_, _> = scc
            .functions
            .iter()
            .enumerate()
            .map(|(i, &f)| (f, i))
            .collect();

        Self {
            dispatch_table,
            functions: scc.functions.clone(),
        }
    }

    /// Get the dispatch index for a function.
    pub fn dispatch_index(&self, func_id: u64) -> Option<usize> {
        self.dispatch_table.get(&func_id).copied()
    }

    /// Get the function for a dispatch index.
    pub fn function_at(&self, index: usize) -> Option<u64> {
        self.functions.get(index).copied()
    }

    /// Get the number of functions in the trampoline.
    pub fn size(&self) -> usize {
        self.functions.len()
    }
}

// =============================================================================
// Mutual Recursion Transformer
// =============================================================================

/// Configuration for mutual recursion transformation.
#[derive(Debug, Clone)]
pub struct MutualRecursionConfig {
    /// Maximum SCC size to optimize.
    pub max_scc_size: usize,
    /// Use trampoline vs fusion.
    pub use_trampoline: bool,
}

impl Default for MutualRecursionConfig {
    fn default() -> Self {
        Self {
            max_scc_size: 8,
            use_trampoline: true,
        }
    }
}

/// Result of mutual recursion transformation.
#[derive(Debug)]
pub struct MutualTransformResult {
    /// Whether transformation succeeded.
    pub success: bool,
    /// Number of functions transformed.
    pub functions_transformed: usize,
    /// Trampoline created (if any).
    pub trampoline: Option<Trampoline>,
    /// Error message.
    pub error: Option<String>,
}

impl MutualTransformResult {
    fn success(funcs: usize, trampoline: Option<Trampoline>) -> Self {
        Self {
            success: true,
            functions_transformed: funcs,
            trampoline,
            error: None,
        }
    }

    fn failure(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            functions_transformed: 0,
            trampoline: None,
            error: Some(msg.into()),
        }
    }

    fn no_candidates() -> Self {
        Self {
            success: true,
            functions_transformed: 0,
            trampoline: None,
            error: None,
        }
    }
}

/// Transforms mutually recursive tail calls.
#[derive(Debug)]
pub struct MutualRecursionTransformer {
    /// Configuration.
    config: MutualRecursionConfig,
}

impl MutualRecursionTransformer {
    /// Create a new transformer.
    pub fn new() -> Self {
        Self::with_config(MutualRecursionConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: MutualRecursionConfig) -> Self {
        Self { config }
    }

    /// Analyze call graph and find optimizable SCCs.
    pub fn analyze(&self, call_graph: &CallGraph) -> Vec<SccInfo> {
        let sccs = call_graph.find_sccs();
        sccs.into_iter()
            .filter(|scc| {
                scc.is_mutual_recursion()
                    && scc.size <= self.config.max_scc_size
                    && !scc.internal_calls.is_empty()
            })
            .collect()
    }

    /// Transform an SCC using trampoline pattern.
    pub fn transform_scc(&self, scc: &SccInfo) -> MutualTransformResult {
        if !scc.optimizable {
            return MutualTransformResult::failure("SCC not optimizable");
        }

        if scc.size > self.config.max_scc_size {
            return MutualTransformResult::failure("SCC too large");
        }

        if scc.internal_calls.is_empty() {
            return MutualTransformResult::no_candidates();
        }

        let trampoline = Trampoline::new(scc);
        MutualTransformResult::success(scc.size, Some(trampoline))
    }

    /// Create a dispatch loop for the SCC.
    pub fn create_dispatch_loop(&self, graph: &mut Graph, scc: &SccInfo) -> Option<NodeId> {
        if scc.functions.is_empty() {
            return None;
        }

        let header = graph.add_node(Operator::Control(ControlOp::Loop), InputList::Empty);
        let _dispatch_phi = graph.add_node(Operator::LoopPhi, InputList::Single(header));
        Some(header)
    }
}

impl Default for MutualRecursionTransformer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Build a call graph from tail call infos.
pub fn build_call_graph(current_func: u64, tail_calls: &[TailCallInfo]) -> CallGraph {
    let mut graph = CallGraph::new();
    graph.add_function(current_func);

    for tc in tail_calls {
        if tc.status.is_optimizable() && !tc.is_self_call {
            graph.add_tail_call(current_func, current_func + 1);
        }
    }

    graph
}

/// Find mutual recursion SCCs in the call graph.
pub fn find_mutual_recursion(call_graph: &CallGraph) -> Vec<SccInfo> {
    let transformer = MutualRecursionTransformer::new();
    transformer.analyze(call_graph)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SccInfo Tests
    // =========================================================================

    #[test]
    fn test_scc_singleton() {
        let scc = SccInfo::singleton(42);
        assert_eq!(scc.size, 1);
        assert!(scc.contains(42));
        assert!(!scc.is_mutual_recursion());
    }

    #[test]
    fn test_scc_is_self_recursion() {
        let mut scc = SccInfo::singleton(42);
        scc.internal_calls.push((42, 42));
        assert!(scc.is_self_recursion());
        assert!(!scc.is_mutual_recursion());
    }

    #[test]
    fn test_scc_is_mutual_recursion() {
        let scc = SccInfo {
            functions: vec![1, 2],
            internal_calls: vec![(1, 2), (2, 1)],
            size: 2,
            optimizable: true,
        };
        assert!(scc.is_mutual_recursion());
        assert!(!scc.is_self_recursion());
    }

    // =========================================================================
    // CallGraph Tests
    // =========================================================================

    #[test]
    fn test_call_graph_empty() {
        let graph = CallGraph::new();
        assert_eq!(graph.all_functions().count(), 0);
    }

    #[test]
    fn test_call_graph_add_function() {
        let mut graph = CallGraph::new();
        graph.add_function(42);
        assert_eq!(graph.all_functions().count(), 1);
    }

    #[test]
    fn test_call_graph_add_edge() {
        let mut graph = CallGraph::new();
        graph.add_tail_call(1, 2);

        let callees: Vec<_> = graph.callees(1).collect();
        assert_eq!(callees, vec![2]);

        let callers: Vec<_> = graph.callers(2).collect();
        assert_eq!(callers, vec![1]);
    }

    // =========================================================================
    // SCC Detection Tests
    // =========================================================================

    #[test]
    fn test_find_sccs_empty() {
        let graph = CallGraph::new();
        let sccs = graph.find_sccs();
        assert!(sccs.is_empty());
    }

    #[test]
    fn test_find_sccs_mutual_two() {
        let mut graph = CallGraph::new();
        graph.add_tail_call(1, 2);
        graph.add_tail_call(2, 1);

        let sccs = graph.find_sccs();
        let mutual_sccs: Vec<_> = sccs.iter().filter(|s| s.size == 2).collect();
        assert_eq!(mutual_sccs.len(), 1);
    }

    // =========================================================================
    // Trampoline Tests
    // =========================================================================

    #[test]
    fn test_trampoline_new() {
        let scc = SccInfo {
            functions: vec![1, 2, 3],
            internal_calls: vec![],
            size: 3,
            optimizable: true,
        };

        let trampoline = Trampoline::new(&scc);
        assert_eq!(trampoline.size(), 3);
    }

    #[test]
    fn test_trampoline_dispatch_index() {
        let scc = SccInfo {
            functions: vec![10, 20, 30],
            internal_calls: vec![],
            size: 3,
            optimizable: true,
        };

        let trampoline = Trampoline::new(&scc);
        assert_eq!(trampoline.dispatch_index(10), Some(0));
        assert_eq!(trampoline.dispatch_index(20), Some(1));
        assert_eq!(trampoline.dispatch_index(99), None);
    }

    // =========================================================================
    // Transformer Tests
    // =========================================================================

    #[test]
    fn test_transformer_analyze() {
        let mut graph = CallGraph::new();
        graph.add_tail_call(1, 2);
        graph.add_tail_call(2, 1);

        let transformer = MutualRecursionTransformer::new();
        let sccs = transformer.analyze(&graph);

        assert!(!sccs.is_empty());
    }

    #[test]
    fn test_transform_scc() {
        let scc = SccInfo {
            functions: vec![1, 2],
            internal_calls: vec![(1, 2), (2, 1)],
            size: 2,
            optimizable: true,
        };

        let transformer = MutualRecursionTransformer::new();
        let result = transformer.transform_scc(&scc);

        assert!(result.success);
        assert_eq!(result.functions_transformed, 2);
    }
}

