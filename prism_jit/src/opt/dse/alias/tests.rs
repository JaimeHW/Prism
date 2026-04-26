use super::*;
use crate::ir::node::InputList;

// =========================================================================
// AliasResult Tests
// =========================================================================

#[test]
fn test_alias_result_may_alias() {
    assert!(AliasResult::MustAlias.may_alias());
    assert!(AliasResult::MayAlias.may_alias());
    assert!(!AliasResult::NoAlias.may_alias());
}

#[test]
fn test_alias_result_must_alias() {
    assert!(AliasResult::MustAlias.must_alias());
    assert!(!AliasResult::MayAlias.must_alias());
    assert!(!AliasResult::NoAlias.must_alias());
}

#[test]
fn test_alias_result_no_alias() {
    assert!(!AliasResult::MustAlias.no_alias());
    assert!(!AliasResult::MayAlias.no_alias());
    assert!(AliasResult::NoAlias.no_alias());
}

#[test]
fn test_alias_result_meet_no_alias() {
    assert_eq!(
        AliasResult::MustAlias.meet(AliasResult::NoAlias),
        AliasResult::NoAlias
    );
    assert_eq!(
        AliasResult::NoAlias.meet(AliasResult::MayAlias),
        AliasResult::NoAlias
    );
}

#[test]
fn test_alias_result_meet_must_alias() {
    assert_eq!(
        AliasResult::MustAlias.meet(AliasResult::MustAlias),
        AliasResult::MustAlias
    );
}

#[test]
fn test_alias_result_meet_may_alias() {
    assert_eq!(
        AliasResult::MustAlias.meet(AliasResult::MayAlias),
        AliasResult::MayAlias
    );
    assert_eq!(
        AliasResult::MayAlias.meet(AliasResult::MayAlias),
        AliasResult::MayAlias
    );
}

// =========================================================================
// MemOffset Tests
// =========================================================================

#[test]
fn test_mem_offset_is_constant() {
    assert!(MemOffset::Field(0).is_constant());
    assert!(MemOffset::ConstElement(5).is_constant());
    assert!(!MemOffset::VarElement(NodeId::new(1)).is_constant());
    assert!(!MemOffset::Unknown.is_constant());
}

#[test]
fn test_mem_offset_definitely_different_fields() {
    let f1 = MemOffset::Field(0);
    let f2 = MemOffset::Field(1);
    let f3 = MemOffset::Field(0);

    assert!(f1.definitely_different(&f2));
    assert!(!f1.definitely_different(&f3));
}

#[test]
fn test_mem_offset_definitely_different_elements() {
    let e1 = MemOffset::ConstElement(0);
    let e2 = MemOffset::ConstElement(1);
    let e3 = MemOffset::ConstElement(0);

    assert!(e1.definitely_different(&e2));
    assert!(!e1.definitely_different(&e3));
}

#[test]
fn test_mem_offset_definitely_different_mixed() {
    let f = MemOffset::Field(0);
    let e = MemOffset::ConstElement(0);

    assert!(f.definitely_different(&e));
    assert!(e.definitely_different(&f));
}

#[test]
fn test_mem_offset_definitely_different_var_element() {
    let v1 = MemOffset::VarElement(NodeId::new(1));
    let v2 = MemOffset::VarElement(NodeId::new(2));
    let c = MemOffset::ConstElement(0);

    // Variable elements may alias other elements
    assert!(!v1.definitely_different(&v2));
    assert!(!v1.definitely_different(&c));
}

#[test]
fn test_mem_offset_definitely_same() {
    let f1 = MemOffset::Field(5);
    let f2 = MemOffset::Field(5);
    let f3 = MemOffset::Field(6);

    assert!(f1.definitely_same(&f2));
    assert!(!f1.definitely_same(&f3));
}

#[test]
fn test_mem_offset_definitely_same_var_element() {
    let v1 = MemOffset::VarElement(NodeId::new(10));
    let v2 = MemOffset::VarElement(NodeId::new(10));
    let v3 = MemOffset::VarElement(NodeId::new(11));

    assert!(v1.definitely_same(&v2));
    assert!(!v1.definitely_same(&v3));
}

// =========================================================================
// MemoryLocation Tests
// =========================================================================

#[test]
fn test_memory_location_new() {
    let base = NodeId::new(1);
    let loc = MemoryLocation::new(base, MemOffset::Field(0));
    assert_eq!(loc.base, base);
    assert_eq!(loc.offset, MemOffset::Field(0));
    assert!(loc.size.is_none());
}

#[test]
fn test_memory_location_with_size() {
    let base = NodeId::new(1);
    let loc = MemoryLocation::with_size(base, MemOffset::Field(0), 8);
    assert_eq!(loc.size, Some(8));
}

#[test]
fn test_memory_location_field() {
    let base = NodeId::new(1);
    let loc = MemoryLocation::field(base, 42);
    assert_eq!(loc.offset, MemOffset::Field(42));
}

#[test]
fn test_memory_location_const_element() {
    let base = NodeId::new(1);
    let loc = MemoryLocation::const_element(base, 7);
    assert_eq!(loc.offset, MemOffset::ConstElement(7));
}

#[test]
fn test_memory_location_var_element() {
    let base = NodeId::new(1);
    let idx = NodeId::new(2);
    let loc = MemoryLocation::var_element(base, idx);
    assert_eq!(loc.offset, MemOffset::VarElement(idx));
}

// =========================================================================
// AliasAnalyzer Creation Tests
// =========================================================================

#[test]
fn test_alias_analyzer_empty_graph() {
    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);
    assert!(analyzer.allocations.is_empty());
}

#[test]
fn test_alias_analyzer_tracks_allocations() {
    let mut graph = Graph::new();
    let alloc = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);

    let analyzer = AliasAnalyzer::new(&graph);
    assert!(analyzer.is_fresh_allocation(alloc));
}

#[test]
fn test_alias_analyzer_tracks_array_allocations() {
    let mut graph = Graph::new();
    let size = graph.const_int(10);
    let alloc = graph.add_node(
        Operator::Memory(MemoryOp::AllocArray),
        InputList::Single(size),
    );

    let analyzer = AliasAnalyzer::new(&graph);
    assert!(analyzer.is_fresh_allocation(alloc));
}

#[test]
fn test_alias_analyzer_non_alloc_not_fresh() {
    let mut graph = Graph::new();
    let const_node = graph.const_int(42);

    let analyzer = AliasAnalyzer::new(&graph);
    assert!(!analyzer.is_fresh_allocation(const_node));
}

// =========================================================================
// Alias Query Tests
// =========================================================================

#[test]
fn test_alias_same_base_same_field() {
    let base = NodeId::new(1);
    let loc1 = MemoryLocation::field(base, 0);
    let loc2 = MemoryLocation::field(base, 0);

    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);

    assert_eq!(analyzer.alias(&loc1, &loc2), AliasResult::MustAlias);
}

#[test]
fn test_alias_same_base_different_fields() {
    let base = NodeId::new(1);
    let loc1 = MemoryLocation::field(base, 0);
    let loc2 = MemoryLocation::field(base, 1);

    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);

    assert_eq!(analyzer.alias(&loc1, &loc2), AliasResult::NoAlias);
}

#[test]
fn test_alias_same_base_same_const_element() {
    let base = NodeId::new(1);
    let loc1 = MemoryLocation::const_element(base, 5);
    let loc2 = MemoryLocation::const_element(base, 5);

    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);

    assert_eq!(analyzer.alias(&loc1, &loc2), AliasResult::MustAlias);
}

#[test]
fn test_alias_same_base_different_const_elements() {
    let base = NodeId::new(1);
    let loc1 = MemoryLocation::const_element(base, 0);
    let loc2 = MemoryLocation::const_element(base, 1);

    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);

    assert_eq!(analyzer.alias(&loc1, &loc2), AliasResult::NoAlias);
}

#[test]
fn test_alias_same_base_var_element_may_alias() {
    let base = NodeId::new(1);
    let idx = NodeId::new(2);
    let loc1 = MemoryLocation::var_element(base, idx);
    let loc2 = MemoryLocation::const_element(base, 0);

    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);

    assert_eq!(analyzer.alias(&loc1, &loc2), AliasResult::MayAlias);
}

#[test]
fn test_alias_different_fresh_allocations() {
    let mut graph = Graph::new();
    let alloc1 = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
    let alloc2 = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);

    let analyzer = AliasAnalyzer::new(&graph);

    let loc1 = MemoryLocation::field(alloc1, 0);
    let loc2 = MemoryLocation::field(alloc2, 0);

    assert_eq!(analyzer.alias(&loc1, &loc2), AliasResult::NoAlias);
}

#[test]
fn test_alias_different_non_fresh_bases() {
    // Non-fresh bases (e.g., parameters) may alias
    let base1 = NodeId::new(100);
    let base2 = NodeId::new(101);

    let loc1 = MemoryLocation::field(base1, 0);
    let loc2 = MemoryLocation::field(base2, 0);

    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);

    assert_eq!(analyzer.alias(&loc1, &loc2), AliasResult::MayAlias);
}

// =========================================================================
// may_alias/must_alias Convenience Tests
// =========================================================================

#[test]
fn test_may_alias_missing_locations() {
    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);

    // Nodes without locations are conservative (may alias)
    assert!(analyzer.may_alias(NodeId::new(1), NodeId::new(2)));
}

#[test]
fn test_must_alias_missing_locations() {
    let graph = Graph::new();
    let analyzer = AliasAnalyzer::new(&graph);

    // Nodes without locations can't be proven to must-alias
    assert!(!analyzer.must_alias(NodeId::new(1), NodeId::new(2)));
}
