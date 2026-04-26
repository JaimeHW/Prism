use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};
use crate::ir::operators::CallKind;

fn make_simple_callee() -> Graph {
    // Create: fn(a, b) { return a + b }
    let mut builder = GraphBuilder::new(8, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    builder.return_value(sum);
    builder.finish()
}

fn make_identity_callee() -> Graph {
    // Create: fn(x) { return x }
    let mut builder = GraphBuilder::new(4, 1);
    let p0 = builder.parameter(0).unwrap();
    builder.return_value(p0);
    builder.finish()
}

#[test]
fn test_inline_error_display() {
    assert_eq!(
        InlineError::InvalidCallSite.to_string(),
        "Invalid call site"
    );
    assert_eq!(
        InlineError::NoReturns.to_string(),
        "Callee has no return nodes"
    );
}

#[test]
fn test_handle_single_return() {
    let mut caller = Graph::new();
    let val = caller.const_int(42);
    let control = caller.start;

    // Create a return node manually
    let ret = caller.add_node_with_type(
        Operator::Control(ControlOp::Return),
        InputList::Pair(control, val),
        ValueType::Control,
    );

    let result = InlineTransform::handle_returns(&mut caller, &[ret]);
    assert!(result.is_ok());

    let (result_node, exit_control) = result.unwrap();
    assert!(result_node.is_some());
    assert_ne!(exit_control, caller.start);
}

#[test]
fn test_handle_multiple_returns() {
    let mut caller = Graph::new();
    let val1 = caller.const_int(1);
    let val2 = caller.const_int(2);

    // Create a branch with two returns
    let if_cond = caller.const_bool(true);
    let if_node = caller.add_node_with_type(
        Operator::Control(ControlOp::If),
        InputList::Pair(caller.start, if_cond),
        ValueType::Control,
    );

    let true_proj = caller.add_node_with_type(
        Operator::Projection(0),
        InputList::Single(if_node),
        ValueType::Control,
    );
    let false_proj = caller.add_node_with_type(
        Operator::Projection(1),
        InputList::Single(if_node),
        ValueType::Control,
    );

    let ret1 = caller.add_node_with_type(
        Operator::Control(ControlOp::Return),
        InputList::Pair(true_proj, val1),
        ValueType::Control,
    );
    let ret2 = caller.add_node_with_type(
        Operator::Control(ControlOp::Return),
        InputList::Pair(false_proj, val2),
        ValueType::Control,
    );

    let result = InlineTransform::handle_returns(&mut caller, &[ret1, ret2]);
    assert!(result.is_ok());

    let (result_node, exit_control) = result.unwrap();
    // Should have created a phi for the merged values
    assert!(result_node.is_some());
    // The exit control should be a region
    let exit_node = caller.get(exit_control).unwrap();
    assert!(matches!(exit_node.op, Operator::Control(ControlOp::Region)));
}

#[test]
fn test_handle_no_returns() {
    let caller = &mut Graph::new();
    let result = InlineTransform::handle_returns(caller, &[]);
    assert!(matches!(result, Err(InlineError::NoReturns)));
}

#[test]
fn test_inline_info() {
    let info = InlineInfo {
        nodes_added: 10,
        result_node: Some(NodeId::new(5)),
        exit_control: NodeId::new(6),
    };

    assert_eq!(info.nodes_added, 10);
    assert!(info.result_node.is_some());
}

// Integration test for full inlining
#[test]
fn test_basic_inline_integration() {
    // This tests the overall structure without actually performing
    // the full inline (which requires proper callee provider setup)
    let callee = make_simple_callee();
    let mut caller = Graph::new();

    // Create arguments in caller
    let arg0 = caller.const_int(10);
    let arg1 = caller.const_int(20);

    // The actual inline would be done via Inline::run with proper setup
    // Here we just verify the callee is valid
    assert!(callee.len() > 2); // More than just start/end
}
