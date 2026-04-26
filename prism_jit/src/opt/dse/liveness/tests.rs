use super::*;
use crate::ir::node::InputList;

// =========================================================================
// KillInfo Tests
// =========================================================================

#[test]
fn test_kill_info_new() {
    let victim = NodeId::new(1);
    let killer = NodeId::new(2);
    let info = KillInfo::new(victim, killer, true);

    assert_eq!(info.victim, victim);
    assert_eq!(info.killer, killer);
    assert!(info.is_must_kill);
}

#[test]
fn test_kill_info_may_kill() {
    let info = KillInfo::new(NodeId::new(1), NodeId::new(2), false);
    assert!(!info.is_must_kill);
}

// =========================================================================
// StoreLiveness Basic Tests
// =========================================================================

#[test]
fn test_store_liveness_empty_graph() {
    let graph = Graph::new();
    let alias = AliasAnalyzer::new(&graph);
    let liveness = StoreLiveness::compute(&graph, &alias);

    assert_eq!(liveness.dead_count(), 0);
    assert_eq!(liveness.live_count(), 0);
}

#[test]
fn test_store_liveness_single_store() {
    let mut graph = Graph::new();
    let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let val = graph.const_int(42);
    let store = graph.add_node(
        Operator::Memory(MemoryOp::StoreField),
        InputList::Pair(obj, val),
    );

    let alias = AliasAnalyzer::new(&graph);
    let liveness = StoreLiveness::compute(&graph, &alias);

    // Single store is live (not killed)
    assert!(liveness.is_live(store));
    assert!(!liveness.is_dead(store));
}

#[test]
fn test_store_liveness_is_dead() {
    let mut liveness = StoreLiveness {
        dead_stores: FxHashSet::default(),
        killers: FxHashMap::default(),
        live_stores: FxHashSet::default(),
    };

    let store = NodeId::new(1);
    liveness.dead_stores.insert(store);

    assert!(liveness.is_dead(store));
    assert!(!liveness.is_dead(NodeId::new(2)));
}

#[test]
fn test_store_liveness_is_live() {
    let mut liveness = StoreLiveness {
        dead_stores: FxHashSet::default(),
        killers: FxHashMap::default(),
        live_stores: FxHashSet::default(),
    };

    let store = NodeId::new(1);
    liveness.live_stores.insert(store);

    assert!(liveness.is_live(store));
    assert!(!liveness.is_live(NodeId::new(2)));
}

#[test]
fn test_store_liveness_get_killer() {
    let mut liveness = StoreLiveness {
        dead_stores: FxHashSet::default(),
        killers: FxHashMap::default(),
        live_stores: FxHashSet::default(),
    };

    let victim = NodeId::new(1);
    let killer = NodeId::new(2);
    liveness.killers.insert(victim, killer);

    assert_eq!(liveness.get_killer(victim), Some(killer));
    assert_eq!(liveness.get_killer(NodeId::new(3)), None);
}

#[test]
fn test_store_liveness_dead_stores() {
    let mut liveness = StoreLiveness {
        dead_stores: FxHashSet::default(),
        killers: FxHashMap::default(),
        live_stores: FxHashSet::default(),
    };

    liveness.dead_stores.insert(NodeId::new(1));
    liveness.dead_stores.insert(NodeId::new(2));

    let dead = liveness.dead_stores();
    assert_eq!(dead.len(), 2);
}

#[test]
fn test_store_liveness_counts() {
    let mut liveness = StoreLiveness {
        dead_stores: FxHashSet::default(),
        killers: FxHashMap::default(),
        live_stores: FxHashSet::default(),
    };

    liveness.dead_stores.insert(NodeId::new(1));
    liveness.live_stores.insert(NodeId::new(2));
    liveness.live_stores.insert(NodeId::new(3));

    assert_eq!(liveness.dead_count(), 1);
    assert_eq!(liveness.live_count(), 2);
}

// =========================================================================
// Store Killing Tests
// =========================================================================

#[test]
fn test_store_not_killed_no_later_store() {
    let mut graph = Graph::new();
    let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let val = graph.const_int(42);
    let store = graph.add_node(
        Operator::Memory(MemoryOp::StoreField),
        InputList::Pair(obj, val),
    );

    let alias = AliasAnalyzer::new(&graph);
    let liveness = StoreLiveness::compute(&graph, &alias);

    assert!(liveness.is_live(store));
    assert!(!liveness.is_dead(store));
}

#[test]
fn test_store_not_killed_different_bases() {
    let mut graph = Graph::new();

    // Two different objects
    let obj1 = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let obj2 = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);

    let val = graph.const_int(42);

    // Store to obj1
    let store1 = graph.add_node(
        Operator::Memory(MemoryOp::StoreField),
        InputList::Pair(obj1, val),
    );

    // Store to obj2 (different object)
    let _store2 = graph.add_node(
        Operator::Memory(MemoryOp::StoreField),
        InputList::Pair(obj2, val),
    );

    let alias = AliasAnalyzer::new(&graph);
    let liveness = StoreLiveness::compute(&graph, &alias);

    // store1 is NOT killed because obj1 != obj2
    assert!(liveness.is_live(store1));
}

// =========================================================================
// Load Intervening Tests
// =========================================================================

#[test]
fn test_has_intervening_load_empty() {
    let liveness = StoreLiveness {
        dead_stores: FxHashSet::default(),
        killers: FxHashMap::default(),
        live_stores: FxHashSet::default(),
    };

    let graph = Graph::new();
    let alias = AliasAnalyzer::new(&graph);
    let loc = super::super::alias::MemoryLocation::field(NodeId::new(1), 0);

    let has_load = liveness.has_intervening_load(&[], NodeId::new(1), NodeId::new(2), &alias, &loc);

    assert!(!has_load);
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_store_with_load() {
    let mut graph = Graph::new();

    let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let val = graph.const_int(42);

    let store = graph.add_node(
        Operator::Memory(MemoryOp::StoreField),
        InputList::Pair(obj, val),
    );

    let _load = graph.add_node(
        Operator::Memory(MemoryOp::LoadField),
        InputList::Single(obj),
    );

    let alias = AliasAnalyzer::new(&graph);
    let liveness = StoreLiveness::compute(&graph, &alias);

    // Store followed by load should be live
    assert!(liveness.is_live(store) || !liveness.is_dead(store));
}
