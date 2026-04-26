use super::*;

// =========================================================================
// InsertionResult Tests
// =========================================================================

#[test]
fn test_insertion_result_new() {
    let result = InsertionResult::new();
    assert_eq!(result.inserted, 0);
    assert_eq!(result.failed, 0);
    assert!(result.new_guards.is_empty());
    assert!(result.new_comparisons.is_empty());
}

#[test]
fn test_insertion_result_all_succeeded() {
    let mut result = InsertionResult::new();
    result.inserted = 5;
    result.failed = 0;
    assert!(result.all_succeeded());

    result.failed = 1;
    assert!(!result.all_succeeded());
}

#[test]
fn test_insertion_result_total_attempts() {
    let mut result = InsertionResult::new();
    result.inserted = 3;
    result.failed = 2;
    assert_eq!(result.total_attempts(), 5);
}

#[test]
fn test_insertion_result_default() {
    let result = InsertionResult::default();
    assert_eq!(result.inserted, 0);
    assert_eq!(result.failed, 0);
    assert!(result.all_succeeded());
}

#[test]
fn test_insertion_result_total_zero() {
    let result = InsertionResult::new();
    assert_eq!(result.total_attempts(), 0);
}

#[test]
fn test_insertion_result_partial_success() {
    let mut result = InsertionResult::new();
    result.inserted = 10;
    result.failed = 5;
    assert!(!result.all_succeeded());
    assert_eq!(result.total_attempts(), 15);
}

#[test]
fn test_insertion_result_with_nodes() {
    let mut result = InsertionResult::new();
    result.new_guards.push(NodeId::new(1));
    result.new_guards.push(NodeId::new(2));
    result.new_comparisons.push(NodeId::new(3));

    assert_eq!(result.new_guards.len(), 2);
    assert_eq!(result.new_comparisons.len(), 1);
}

#[test]
fn test_insertion_result_high_counts() {
    let mut result = InsertionResult::new();
    result.inserted = 1000;
    result.failed = 500;
    assert_eq!(result.total_attempts(), 1500);
    assert!(!result.all_succeeded());
}

#[test]
fn test_insertion_result_many_guards() {
    let mut result = InsertionResult::new();
    for i in 0..100 {
        result.new_guards.push(NodeId::new(i));
        result.new_comparisons.push(NodeId::new(i + 1000));
    }
    assert_eq!(result.new_guards.len(), 100);
    assert_eq!(result.new_comparisons.len(), 100);
}

// =========================================================================
// GuardInserter Tests
// =========================================================================

#[test]
fn test_guard_inserter_record_failure() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let iv_analysis = InductionAnalysis::empty();

    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);
    inserter.record_failure();
    inserter.record_failure();
    inserter.record_failure();

    let result = inserter.into_result();
    assert_eq!(result.failed, 3);
    assert_eq!(result.inserted, 0);
}

#[test]
fn test_guard_inserter_empty_analysis() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let iv_analysis = InductionAnalysis::empty();

    let inserter = GuardInserter::new(&mut graph, &iv_analysis);
    let result = inserter.into_result();
    assert!(result.all_succeeded());
    assert_eq!(result.total_attempts(), 0);
}

#[test]
fn test_widen_bound_constant() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let bound = WidenBound::Constant(42);
    let node = inserter.get_widen_bound_node(&bound);
    assert!(node.is_some());
}

#[test]
fn test_widen_bound_node() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let node_id = graph.const_int(100);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let bound = WidenBound::Node(node_id);
    let result = inserter.get_widen_bound_node(&bound);
    assert!(result.is_some());
    assert_eq!(result.unwrap(), node_id);
}

#[test]
fn test_widen_bound_computed_zero_offset() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let base_node = graph.const_int(50);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    // Zero offset should return base directly
    let bound = WidenBound::Computed {
        base: base_node,
        offset: 0,
    };
    let result = inserter.get_widen_bound_node(&bound);
    assert!(result.is_some());
    assert_eq!(result.unwrap(), base_node);
}

#[test]
fn test_widen_bound_computed_nonzero_offset() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let base_node = graph.const_int(50);
    let initial_node_count = graph.len();
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    // Non-zero offset should create addition node
    let bound = WidenBound::Computed {
        base: base_node,
        offset: 10,
    };
    let result = inserter.get_widen_bound_node(&bound);
    assert!(result.is_some());
    // Should have created new nodes (offset const + add)
    assert!(graph.len() > initial_node_count);
}

#[test]
fn test_bound_value_constant() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let bound = BoundValue::Constant(100);
    let node = inserter.get_bound_node(&bound);
    assert!(node.is_some());
}

#[test]
fn test_bound_value_node() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let len_node = graph.const_int(256);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let bound = BoundValue::Node(len_node);
    let result = inserter.get_bound_node(&bound);
    assert!(result.is_some());
    assert_eq!(result.unwrap(), len_node);
}

#[test]
fn test_create_comparison_lower_bound() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let lhs = graph.const_int(0);
    let rhs = graph.const_int(0);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let cmp_node = inserter.create_comparison(lhs, rhs, RangeCheckKind::LowerBound);
    assert!(cmp_node.is_some());
}

#[test]
fn test_create_comparison_upper_bound() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let lhs = graph.const_int(0);
    let rhs = graph.const_int(100);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let cmp_node = inserter.create_comparison(lhs, rhs, RangeCheckKind::UpperBound);
    assert!(cmp_node.is_some());
}

#[test]
fn test_create_comparison_upper_bound_inclusive() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let lhs = graph.const_int(99);
    let rhs = graph.const_int(100);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let cmp_node = inserter.create_comparison(lhs, rhs, RangeCheckKind::UpperBoundInclusive);
    assert!(cmp_node.is_some());
}

#[test]
fn test_create_guard() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let condition = graph.const_bool(true);
    let control = NodeId::new(0); // Use start node
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let guard = inserter.create_guard(condition, control);

    // Verify guard was created
    let guard_node = graph.get(guard);
    assert!(guard_node.is_some());
    assert!(matches!(
        guard_node.unwrap().op,
        Operator::Guard(GuardKind::Bounds)
    ));
}

#[test]
fn test_get_init_value_node_constant() {
    use super::super::induction::{InductionDirection, InductionStep};
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let iv = InductionVariable::new(
        NodeId::new(10),
        InductionInit::Constant(0),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    );

    let init_node = inserter.get_init_value_node(&iv);
    assert!(init_node.is_some());
}

#[test]
fn test_get_init_value_node_from_node() {
    use super::super::induction::{InductionDirection, InductionStep};
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let init_id = graph.const_int(42);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let iv = InductionVariable::new(
        NodeId::new(10),
        InductionInit::Node(init_id),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    );

    let result = inserter.get_init_value_node(&iv);
    assert!(result.is_some());
    assert_eq!(result.unwrap(), init_id);
}

#[test]
fn test_insert_bound_check_with_constant() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let control = NodeId::new(0);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let bound = WidenBound::Constant(0);
    let guard = inserter.insert_bound_check(&bound, CmpOp::Ge, control);

    assert!(guard.is_some());
    assert!(!inserter.result.new_comparisons.is_empty());
}

#[test]
fn test_insert_max_bound_check_upper_bound() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let control = NodeId::new(0);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let max_check = WidenBound::Constant(99);
    let original_check = RangeCheck::new(
        NodeId::new(5),
        NodeId::new(1),
        NodeId::new(6),
        BoundValue::Constant(100),
        RangeCheckKind::UpperBound,
        0,
    );

    let guard = inserter.insert_max_bound_check(&max_check, &original_check, control);
    assert!(guard.is_some());
}

#[test]
fn test_insert_max_bound_check_lower_bound_returns_none() {
    use crate::ir::graph::Graph;
    let mut graph = Graph::new();
    let control = NodeId::new(0);
    let iv_analysis = InductionAnalysis::empty();
    let mut inserter = GuardInserter::new(&mut graph, &iv_analysis);

    let max_check = WidenBound::Constant(0);
    let original_check = RangeCheck::new(
        NodeId::new(5),
        NodeId::new(1),
        NodeId::new(6),
        BoundValue::Constant(0),
        RangeCheckKind::LowerBound, // Lower bound shouldn't need max check
        0,
    );

    let guard = inserter.insert_max_bound_check(&max_check, &original_check, control);
    // Lower bound checks don't need max check
    assert!(guard.is_none());
}
