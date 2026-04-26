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
