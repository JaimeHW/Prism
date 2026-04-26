use super::*;
use crate::ir::builder::{ControlBuilder, GraphBuilder};

// -------------------------------------------------------------------------
// EscapeState Tests
// -------------------------------------------------------------------------

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
    assert_eq!(
        EscapeState::GlobalEscape.merge(EscapeState::NoEscape),
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

// -------------------------------------------------------------------------
// AllocationInfo Tests
// -------------------------------------------------------------------------

#[test]
fn test_allocation_info_can_optimize() {
    let info = AllocationInfo {
        node: NodeId::new(0),
        escape_state: EscapeState::NoEscape,
        object_type: ObjectType::Object,
        all_uses_known: true,
        estimated_size: Some(64),
        access_count: 5,
    };
    assert!(info.can_optimize());

    let info2 = AllocationInfo {
        escape_state: EscapeState::GlobalEscape,
        ..info.clone()
    };
    assert!(!info2.can_optimize());

    let info3 = AllocationInfo {
        all_uses_known: false,
        ..info.clone()
    };
    assert!(!info3.can_optimize());
}

// -------------------------------------------------------------------------
// EscapeAnalysis Tests
// -------------------------------------------------------------------------

#[test]
fn test_escape_analysis_empty() {
    let builder = GraphBuilder::new(0, 0);
    let graph = builder.finish();

    let analysis = EscapeAnalysis::compute(&graph);
    assert_eq!(analysis.non_escaping_count(), 0);
    assert_eq!(analysis.arg_escaping_count(), 0);
    assert_eq!(analysis.global_escaping_count(), 0);
    assert_eq!(analysis.total_allocations(), 0);
}

#[test]
fn test_escape_analysis_query() {
    let analysis = EscapeAnalysis {
        states: {
            let mut m = FxHashMap::default();
            m.insert(NodeId::new(0), EscapeState::NoEscape);
            m.insert(NodeId::new(1), EscapeState::ArgEscape);
            m.insert(NodeId::new(2), EscapeState::GlobalEscape);
            m
        },
        allocations: vec![],
        non_escaping: 1,
        arg_escaping: 1,
        global_escaping: 1,
    };

    assert_eq!(
        analysis.escape_state(NodeId::new(0)),
        Some(EscapeState::NoEscape)
    );
    assert!(analysis.can_scalar_replace(NodeId::new(0)));
    assert!(analysis.can_stack_allocate(NodeId::new(0)));

    assert!(analysis.can_stack_allocate(NodeId::new(1)));
    assert!(!analysis.can_scalar_replace(NodeId::new(1)));

    assert!(!analysis.can_stack_allocate(NodeId::new(2)));
    assert!(!analysis.can_scalar_replace(NodeId::new(2)));
}

#[test]
fn test_escape_analysis_iterators() {
    let analysis = EscapeAnalysis {
        states: {
            let mut m = FxHashMap::default();
            m.insert(NodeId::new(0), EscapeState::NoEscape);
            m.insert(NodeId::new(1), EscapeState::NoEscape);
            m.insert(NodeId::new(2), EscapeState::ArgEscape);
            m.insert(NodeId::new(3), EscapeState::GlobalEscape);
            m
        },
        allocations: vec![],
        non_escaping: 2,
        arg_escaping: 1,
        global_escaping: 1,
    };

    let scalar: Vec<_> = analysis.scalar_replaceable().collect();
    assert_eq!(scalar.len(), 2);

    let stack: Vec<_> = analysis.stack_allocatable().collect();
    assert_eq!(stack.len(), 3);
}

// -------------------------------------------------------------------------
// Escape Pass Tests
// -------------------------------------------------------------------------

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
fn test_escape_pass_no_allocs() {
    let mut builder = GraphBuilder::new(1, 1);
    let p0 = builder.parameter(0).unwrap();
    builder.return_value(p0);

    let mut graph = builder.finish();
    let mut escape = Escape::new();

    let changed = escape.run(&mut graph);
    assert!(!changed);
    assert_eq!(escape.stats().allocations_analyzed, 0);
}

#[test]
fn test_escape_pass_name() {
    use super::super::OptimizationPass;
    let escape = Escape::new();
    assert_eq!(OptimizationPass::name(&escape), "escape");
}

// -------------------------------------------------------------------------
// EscapeStats Tests
// -------------------------------------------------------------------------

#[test]
fn test_escape_stats_default() {
    let stats = EscapeStats::default();
    assert_eq!(stats.allocations_analyzed, 0);
    assert_eq!(stats.non_escaping, 0);
    assert_eq!(stats.scalar_replaced, 0);
    assert_eq!(stats.stack_allocated, 0);
    assert_eq!(stats.nodes_eliminated, 0);
}

// -------------------------------------------------------------------------
// Integration Tests
// -------------------------------------------------------------------------

#[test]
fn test_full_analysis_pipeline() {
    // Create a simple graph
    let builder = GraphBuilder::new(4, 2);
    let graph = builder.finish();

    let analysis = EscapeAnalysis::compute(&graph);

    // Should have counted allocations correctly
    let total = analysis.non_escaping_count()
        + analysis.arg_escaping_count()
        + analysis.global_escaping_count();
    assert_eq!(total, analysis.total_allocations());
}

#[test]
fn test_is_allocation() {
    assert!(EscapeAnalysis::is_allocation(&Operator::Memory(
        MemoryOp::Alloc
    )));
    assert!(EscapeAnalysis::is_allocation(&Operator::Memory(
        MemoryOp::AllocArray
    )));
    assert!(EscapeAnalysis::is_allocation(&Operator::BuildList(5)));
    assert!(EscapeAnalysis::is_allocation(&Operator::BuildTuple(3)));
    assert!(EscapeAnalysis::is_allocation(&Operator::BuildDict(2)));

    assert!(!EscapeAnalysis::is_allocation(&Operator::ConstInt(42)));
    assert!(!EscapeAnalysis::is_allocation(&Operator::Phi));
}

#[test]
fn test_classify_use() {
    // Non-escaping uses
    assert_eq!(
        EscapeAnalysis::classify_use(&Operator::Memory(MemoryOp::LoadField)),
        EscapeState::NoEscape
    );
    assert_eq!(
        EscapeAnalysis::classify_use(&Operator::GetItem),
        EscapeState::NoEscape
    );
    assert_eq!(
        EscapeAnalysis::classify_use(&Operator::Guard(crate::ir::operators::GuardKind::Type)),
        EscapeState::NoEscape
    );

    // Arg escape
    assert_eq!(
        EscapeAnalysis::classify_use(&Operator::Call(crate::ir::operators::CallKind::Direct)),
        EscapeState::ArgEscape
    );

    // Global escape
    assert_eq!(
        EscapeAnalysis::classify_use(&Operator::Control(ControlOp::Return)),
        EscapeState::GlobalEscape
    );
}

#[test]
fn test_infer_object_type() {
    use crate::ir::graph::Graph;
    use crate::ir::node::InputList;

    let mut graph = Graph::new();

    // Create different allocation types
    let list_alloc = graph.add_node_with_type(
        Operator::BuildList(5),
        InputList::Empty,
        crate::ir::types::ValueType::List,
    );
    let tuple_alloc = graph.add_node_with_type(
        Operator::BuildTuple(3),
        InputList::Empty,
        crate::ir::types::ValueType::Tuple,
    );
    let dict_alloc = graph.add_node_with_type(
        Operator::BuildDict(2),
        InputList::Empty,
        crate::ir::types::ValueType::Dict,
    );

    assert_eq!(
        EscapeAnalysis::infer_object_type(&graph, list_alloc),
        ObjectType::Array
    );
    assert_eq!(
        EscapeAnalysis::infer_object_type(&graph, tuple_alloc),
        ObjectType::Tuple
    );
    assert_eq!(
        EscapeAnalysis::infer_object_type(&graph, dict_alloc),
        ObjectType::Dict
    );
}

#[test]
fn test_estimate_size() {
    use crate::ir::graph::Graph;
    use crate::ir::node::InputList;

    let mut graph = Graph::new();

    let list_alloc = graph.add_node_with_type(
        Operator::BuildList(10),
        InputList::Empty,
        crate::ir::types::ValueType::List,
    );
    let tuple_alloc = graph.add_node_with_type(
        Operator::BuildTuple(5),
        InputList::Empty,
        crate::ir::types::ValueType::Tuple,
    );

    // List: 64 + 10*8 = 144
    assert_eq!(EscapeAnalysis::estimate_size(&graph, list_alloc), Some(144));
    // Tuple: 32 + 5*8 = 72
    assert_eq!(EscapeAnalysis::estimate_size(&graph, tuple_alloc), Some(72));
}
