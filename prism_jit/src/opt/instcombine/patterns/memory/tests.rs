use super::*;
use crate::ir::node::InputList;

#[test]
fn test_is_fresh_alloc() {
    let mut graph = Graph::new();
    let alloc = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);

    assert!(MemoryPatterns::is_fresh_alloc(&graph, alloc));
}

#[test]
fn test_is_fresh_alloc_array() {
    let mut graph = Graph::new();
    let alloc = graph.add_node(Operator::Memory(MemoryOp::AllocArray), InputList::Empty);

    assert!(MemoryPatterns::is_fresh_alloc(&graph, alloc));
}

#[test]
fn test_is_not_fresh_alloc() {
    let mut graph = Graph::new();
    let c = graph.const_int(42);

    assert!(!MemoryPatterns::is_fresh_alloc(&graph, c));
}

#[test]
fn test_memory_patterns_defer_to_dse() {
    // Memory patterns are handled by DSE, not InstCombine
    let mut graph = Graph::new();
    let alloc = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let val = graph.const_int(42);
    let store = graph.add_node(
        Operator::Memory(MemoryOp::StoreField),
        InputList::Pair(alloc, val),
    );

    // InstCombine defers memory patterns to DSE
    let m = MemoryPatterns::try_match(&graph, store, &Operator::Memory(MemoryOp::StoreField));
    assert!(m.is_none());
}

#[test]
fn test_load_no_match() {
    let mut graph = Graph::new();
    let alloc = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let load = graph.add_node(
        Operator::Memory(MemoryOp::LoadField),
        InputList::Single(alloc),
    );

    let m = MemoryPatterns::try_match(&graph, load, &Operator::Memory(MemoryOp::LoadField));
    assert!(m.is_none());
}
