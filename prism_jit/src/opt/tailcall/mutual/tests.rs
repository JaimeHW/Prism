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
