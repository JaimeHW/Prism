use super::*;
use crate::ir::node::InputList;

// =========================================================================
// Configuration Tests
// =========================================================================

#[test]
fn test_dse_config_default() {
    let config = DseConfig::default();
    assert!(config.enable_must_alias_dse);
    assert!(config.enable_dead_alloc_dse);
    assert_eq!(config.max_iterations, 10);
}

// =========================================================================
// Statistics Tests
// =========================================================================

#[test]
fn test_dse_stats_default() {
    let stats = DseStats::default();
    assert_eq!(stats.redundant_stores, 0);
    assert_eq!(stats.dead_allocation_stores, 0);
    assert_eq!(stats.partial_dead_stores, 0);
    assert_eq!(stats.total_eliminated(), 0);
}

#[test]
fn test_dse_stats_total() {
    let stats = DseStats {
        redundant_stores: 3,
        dead_allocation_stores: 2,
        partial_dead_stores: 1,
        stores_analyzed: 10,
    };
    assert_eq!(stats.total_eliminated(), 6);
}

#[test]
fn test_dse_stats_merge() {
    let mut stats1 = DseStats {
        redundant_stores: 2,
        dead_allocation_stores: 1,
        partial_dead_stores: 0,
        stores_analyzed: 5,
    };
    let stats2 = DseStats {
        redundant_stores: 3,
        dead_allocation_stores: 2,
        partial_dead_stores: 1,
        stores_analyzed: 8,
    };
    stats1.merge(&stats2);
    assert_eq!(stats1.redundant_stores, 5);
    assert_eq!(stats1.dead_allocation_stores, 3);
    assert_eq!(stats1.partial_dead_stores, 1);
    assert_eq!(stats1.stores_analyzed, 13);
}

// =========================================================================
// Pass Creation Tests
// =========================================================================

#[test]
fn test_dse_new() {
    let dse = Dse::new();
    assert_eq!(dse.removed(), 0);
    assert_eq!(dse.stats().total_eliminated(), 0);
}

#[test]
fn test_dse_with_config() {
    let config = DseConfig {
        enable_must_alias_dse: false,
        enable_dead_alloc_dse: true,
        max_iterations: 5,
    };
    let dse = Dse::with_config(config);
    assert_eq!(dse.removed(), 0);
}

#[test]
fn test_dse_name() {
    let dse = Dse::new();
    assert_eq!(dse.name(), "dse");
}

// =========================================================================
// Store/Load Detection Tests
// =========================================================================

#[test]
fn test_is_store_field() {
    assert!(Dse::is_store(&Operator::Memory(MemoryOp::StoreField)));
}

#[test]
fn test_is_store_element() {
    assert!(Dse::is_store(&Operator::Memory(MemoryOp::StoreElement)));
}

#[test]
fn test_is_store_non_store() {
    assert!(!Dse::is_store(&Operator::Memory(MemoryOp::LoadField)));
    assert!(!Dse::is_store(&Operator::ConstInt(42)));
}

#[test]
fn test_is_load_field() {
    assert!(Dse::is_load(&Operator::Memory(MemoryOp::LoadField)));
}

#[test]
fn test_is_load_element() {
    assert!(Dse::is_load(&Operator::Memory(MemoryOp::LoadElement)));
}

#[test]
fn test_is_load_non_load() {
    assert!(!Dse::is_load(&Operator::Memory(MemoryOp::StoreField)));
    assert!(!Dse::is_load(&Operator::ConstInt(42)));
}

// =========================================================================
// Collection Tests
// =========================================================================

#[test]
fn test_collect_stores_empty() {
    let graph = Graph::new();
    let dse = Dse::new();
    let stores = dse.collect_stores(&graph);
    assert!(stores.is_empty());
}

#[test]
fn test_collect_stores_with_stores() {
    let mut graph = Graph::new();
    let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let val = graph.const_int(42);
    let _store = graph.add_node(
        Operator::Memory(MemoryOp::StoreField),
        InputList::Pair(obj, val),
    );

    let dse = Dse::new();
    let stores = dse.collect_stores(&graph);
    assert_eq!(stores.len(), 1);
}

#[test]
fn test_collect_loads_empty() {
    let graph = Graph::new();
    let dse = Dse::new();
    let loads = dse.collect_loads(&graph);
    assert!(loads.is_empty());
}

#[test]
fn test_collect_loads_with_loads() {
    let mut graph = Graph::new();
    let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let _load = graph.add_node(
        Operator::Memory(MemoryOp::LoadField),
        InputList::Single(obj),
    );

    let dse = Dse::new();
    let loads = dse.collect_loads(&graph);
    assert_eq!(loads.len(), 1);
}

// =========================================================================
// DSE Run Tests
// =========================================================================

#[test]
fn test_dse_no_stores() {
    let mut graph = Graph::new();
    graph.const_int(42);

    let mut dse = Dse::new();
    let changed = dse.run(&mut graph);

    assert!(!changed);
    assert_eq!(dse.removed(), 0);
}

#[test]
fn test_dse_stats_after_run() {
    let mut graph = Graph::new();
    graph.const_int(42);

    let mut dse = Dse::new();
    dse.run(&mut graph);

    assert_eq!(dse.stats().stores_analyzed, 0);
}
