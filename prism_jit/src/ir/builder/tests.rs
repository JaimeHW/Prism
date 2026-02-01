//! Comprehensive unit tests for the IR Graph Builder.
//!
//! Tests are organized by builder trait:
//! - ArithmeticBuilder: Constants, int/float/generic ops, comparisons
//! - ControlBuilder: Regions, branches, loops, phi nodes
//! - ObjectBuilder: Item/attribute access, calls
//! - ContainerBuilder: List/tuple building, iteration

use super::*;
use crate::ir::operators::{ArithOp, CmpOp, ControlOp, Operator};
use crate::ir::types::ValueType;

// =============================================================================
// Basic Builder Tests
// =============================================================================

#[test]
fn test_builder_basic() {
    let mut builder = GraphBuilder::new(4, 2);

    // Get parameters
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // Add them
    let sum = builder.int_add(p0, p1);

    // Return
    let _ret = builder.return_value(sum);

    let graph = builder.finish();
    assert!(graph.len() > 4); // start, end, 2 params, add, return
}

#[test]
fn test_builder_constants() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(10);
    let b = builder.const_int(20);
    let sum = builder.int_add(a, b);

    builder.set_register(0, sum);

    assert_eq!(builder.get_register(0), sum);
}

#[test]
fn test_builder_register_state() {
    let mut builder = GraphBuilder::new(8, 0);

    let c1 = builder.const_int(42);
    let c2 = builder.const_int(100);

    builder.set_register(0, c1);
    builder.set_register(3, c2);

    assert_eq!(builder.get_register(0), c1);
    assert_eq!(builder.get_register(3), c2);
    assert!(!builder.get_register(1).is_valid()); // Uninitialized
}

// =============================================================================
// ArithmeticBuilder Tests - Constants
// =============================================================================

#[test]
fn test_const_int_values() {
    let mut builder = GraphBuilder::new(4, 0);

    let zero = builder.const_int(0);
    let positive = builder.const_int(42);
    let negative = builder.const_int(-100);
    let max = builder.const_int(i64::MAX);
    let min = builder.const_int(i64::MIN);

    let graph = builder.finish();

    assert_eq!(graph.node(zero).ty, ValueType::Int64);
    assert_eq!(graph.node(positive).ty, ValueType::Int64);
    assert_eq!(graph.node(negative).ty, ValueType::Int64);
    assert_eq!(graph.node(max).ty, ValueType::Int64);
    assert_eq!(graph.node(min).ty, ValueType::Int64);
}

#[test]
fn test_const_float_values() {
    let mut builder = GraphBuilder::new(4, 0);

    let zero = builder.const_float(0.0);
    let positive = builder.const_float(3.14159);
    let negative = builder.const_float(-2.718);
    let inf = builder.const_float(f64::INFINITY);
    let neg_inf = builder.const_float(f64::NEG_INFINITY);

    let graph = builder.finish();

    assert_eq!(graph.node(zero).ty, ValueType::Float64);
    assert_eq!(graph.node(positive).ty, ValueType::Float64);
    assert_eq!(graph.node(negative).ty, ValueType::Float64);
    assert_eq!(graph.node(inf).ty, ValueType::Float64);
    assert_eq!(graph.node(neg_inf).ty, ValueType::Float64);
}

#[test]
fn test_const_bool_values() {
    let mut builder = GraphBuilder::new(4, 0);

    let t = builder.const_bool(true);
    let f = builder.const_bool(false);

    let graph = builder.finish();

    assert_eq!(graph.node(t).ty, ValueType::Bool);
    assert_eq!(graph.node(f).ty, ValueType::Bool);

    // Verify they produce ConstBool operators
    assert!(matches!(graph.node(t).op, Operator::ConstBool(true)));
    assert!(matches!(graph.node(f).op, Operator::ConstBool(false)));
}

#[test]
fn test_const_none() {
    let mut builder = GraphBuilder::new(4, 0);

    let none = builder.const_none();

    let graph = builder.finish();

    assert_eq!(graph.node(none).ty, ValueType::None);
    assert!(matches!(graph.node(none).op, Operator::ConstNone));
}

// =============================================================================
// ArithmeticBuilder Tests - Integer Operations
// =============================================================================

#[test]
fn test_int_add_sub_basic() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(10);
    let b = builder.const_int(3);

    let sum = builder.int_add(a, b);
    let diff = builder.int_sub(a, b);

    let graph = builder.finish();

    // Verify operators
    assert!(matches!(graph.node(sum).op, Operator::IntOp(ArithOp::Add)));
    assert!(matches!(graph.node(diff).op, Operator::IntOp(ArithOp::Sub)));

    // Verify types
    assert_eq!(graph.node(sum).ty, ValueType::Int64);
    assert_eq!(graph.node(diff).ty, ValueType::Int64);

    // Verify inputs
    assert_eq!(graph.node(sum).inputs.len(), 2);
    assert_eq!(graph.node(diff).inputs.len(), 2);
}

#[test]
fn test_int_mul_div_mod() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(20);
    let b = builder.const_int(6);

    let prod = builder.int_mul(a, b);
    let quot = builder.int_div(a, b);
    let rem = builder.int_mod(a, b);

    let graph = builder.finish();

    assert!(matches!(graph.node(prod).op, Operator::IntOp(ArithOp::Mul)));
    assert!(matches!(
        graph.node(quot).op,
        Operator::IntOp(ArithOp::FloorDiv)
    ));
    assert!(matches!(graph.node(rem).op, Operator::IntOp(ArithOp::Mod)));
}

#[test]
fn test_int_neg() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(42);
    let neg_a = builder.int_neg(a);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(neg_a).op,
        Operator::IntOp(ArithOp::Neg)
    ));
    // Unary operation has 1 input
    assert_eq!(graph.node(neg_a).inputs.len(), 1);
}

#[test]
fn test_int_chained_operations() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(2);
    let b = builder.const_int(3);
    let c = builder.const_int(4);

    // (a + b) * c
    let sum = builder.int_add(a, b);
    let result = builder.int_mul(sum, c);

    let graph = builder.finish();

    // result should have sum as one of its inputs
    let inputs: Vec<_> = graph.node(result).inputs.iter().collect();
    assert!(inputs.contains(&sum));
}

// =============================================================================
// ArithmeticBuilder Tests - Float Operations
// =============================================================================

#[test]
fn test_float_add_sub() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_float(3.14);
    let b = builder.const_float(2.71);

    let sum = builder.float_add(a, b);
    let diff = builder.float_sub(a, b);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(sum).op,
        Operator::FloatOp(ArithOp::Add)
    ));
    assert!(matches!(
        graph.node(diff).op,
        Operator::FloatOp(ArithOp::Sub)
    ));
    assert_eq!(graph.node(sum).ty, ValueType::Float64);
}

#[test]
fn test_float_mul_div() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_float(10.0);
    let b = builder.const_float(3.0);

    let prod = builder.float_mul(a, b);
    let quot = builder.float_div(a, b);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(prod).op,
        Operator::FloatOp(ArithOp::Mul)
    ));
    assert!(matches!(
        graph.node(quot).op,
        Operator::FloatOp(ArithOp::TrueDiv)
    ));
}

#[test]
fn test_float_neg() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_float(3.14);
    let neg_a = builder.float_neg(a);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(neg_a).op,
        Operator::FloatOp(ArithOp::Neg)
    ));
}

// =============================================================================
// ArithmeticBuilder Tests - Generic (Polymorphic) Operations
// =============================================================================

#[test]
fn test_generic_arithmetic_ops() {
    let mut builder = GraphBuilder::new(4, 2);

    // Use parameters (unknown type at compile time)
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    let add = builder.generic_add(p0, p1);
    let sub = builder.generic_sub(p0, p1);
    let mul = builder.generic_mul(p0, p1);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(add).op,
        Operator::GenericOp(ArithOp::Add)
    ));
    assert!(matches!(
        graph.node(sub).op,
        Operator::GenericOp(ArithOp::Sub)
    ));
    assert!(matches!(
        graph.node(mul).op,
        Operator::GenericOp(ArithOp::Mul)
    ));

    // Generic ops initially have Numeric type (refined during optimization)
    assert_eq!(graph.node(add).ty, ValueType::Numeric);
}

#[test]
fn test_generic_comparison_ops() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    let lt = builder.generic_lt(p0, p1);
    let le = builder.generic_le(p0, p1);
    let eq = builder.generic_eq(p0, p1);
    let ne = builder.generic_ne(p0, p1);
    let gt = builder.generic_gt(p0, p1);
    let ge = builder.generic_ge(p0, p1);

    let graph = builder.finish();

    assert!(matches!(graph.node(lt).op, Operator::GenericCmp(CmpOp::Lt)));
    assert!(matches!(graph.node(le).op, Operator::GenericCmp(CmpOp::Le)));
    assert!(matches!(graph.node(eq).op, Operator::GenericCmp(CmpOp::Eq)));
    assert!(matches!(graph.node(ne).op, Operator::GenericCmp(CmpOp::Ne)));
    assert!(matches!(graph.node(gt).op, Operator::GenericCmp(CmpOp::Gt)));
    assert!(matches!(graph.node(ge).op, Operator::GenericCmp(CmpOp::Ge)));
}

// =============================================================================
// ArithmeticBuilder Tests - Integer Comparisons
// =============================================================================

#[test]
fn test_builder_comparison() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(5);
    let b = builder.const_int(10);

    let lt = builder.int_lt(a, b);
    let eq = builder.int_eq(a, b);

    let graph = builder.finish();

    assert_eq!(graph.node(lt).ty, ValueType::Bool);
    assert_eq!(graph.node(eq).ty, ValueType::Bool);
}

#[test]
fn test_int_comparison_all_ops() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(10);
    let b = builder.const_int(20);

    let lt = builder.int_lt(a, b);
    let le = builder.int_le(a, b);
    let eq = builder.int_eq(a, b);
    let ne = builder.int_ne(a, b);
    let gt = builder.int_gt(a, b);
    let ge = builder.int_ge(a, b);

    let graph = builder.finish();

    assert!(matches!(graph.node(lt).op, Operator::IntCmp(CmpOp::Lt)));
    assert!(matches!(graph.node(le).op, Operator::IntCmp(CmpOp::Le)));
    assert!(matches!(graph.node(eq).op, Operator::IntCmp(CmpOp::Eq)));
    assert!(matches!(graph.node(ne).op, Operator::IntCmp(CmpOp::Ne)));
    assert!(matches!(graph.node(gt).op, Operator::IntCmp(CmpOp::Gt)));
    assert!(matches!(graph.node(ge).op, Operator::IntCmp(CmpOp::Ge)));

    // All comparisons return bool
    for node_id in [lt, le, eq, ne, gt, ge] {
        assert_eq!(graph.node(node_id).ty, ValueType::Bool);
    }
}

// =============================================================================
// ControlBuilder Tests
// =============================================================================

#[test]
fn test_control_region_merge() {
    let mut builder = GraphBuilder::new(4, 0);

    let start = builder.control();

    // Create a simple branch
    let cond = builder.const_bool(true);
    let if_node = builder.branch(cond);

    // Create projections
    let true_proj = builder.graph.add_node_with_type(
        Operator::Projection(0),
        crate::ir::node::InputList::Single(if_node),
        ValueType::Control,
    );
    let false_proj = builder.graph.add_node_with_type(
        Operator::Projection(1),
        crate::ir::node::InputList::Single(if_node),
        ValueType::Control,
    );

    // Merge them
    let region = builder.region(&[true_proj, false_proj]);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(region).op,
        Operator::Control(ControlOp::Region)
    ));
    assert_eq!(graph.node(region).ty, ValueType::Control);
}

#[test]
fn test_control_branch_creates_if() {
    let mut builder = GraphBuilder::new(4, 0);

    let cond = builder.const_bool(true);
    let if_node = builder.branch(cond);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(if_node).op,
        Operator::Control(ControlOp::If)
    ));
}

#[test]
fn test_builder_loop() {
    let mut builder = GraphBuilder::new(4, 0);

    let entry = builder.control();
    let loop_head = builder.loop_header(entry);

    // Create loop phi for counter
    let initial = builder.const_int(0);
    let counter = builder.loop_phi(loop_head, initial, ValueType::Int64);

    // Increment
    let one = builder.const_int(1);
    let next = builder.int_add(counter, one);

    // Update back edge
    builder.set_loop_phi_back(counter, next);

    let graph = builder.finish();
    assert!(graph.node(counter).is_phi());
}

#[test]
fn test_control_loop_header() {
    let mut builder = GraphBuilder::new(4, 0);

    let entry = builder.control();
    let loop_head = builder.loop_header(entry);

    // Check that loop_header updates control before consuming builder
    assert_eq!(builder.control(), loop_head);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(loop_head).op,
        Operator::Control(ControlOp::Loop)
    ));
}

#[test]
fn test_control_return_value() {
    let mut builder = GraphBuilder::new(4, 0);

    let value = builder.const_int(42);
    let ret = builder.return_value(value);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(ret).op,
        Operator::Control(ControlOp::Return)
    ));
}

#[test]
fn test_control_return_none() {
    let mut builder = GraphBuilder::new(4, 0);

    let ret = builder.return_none();

    let graph = builder.finish();

    // return_none creates a ConstNone and returns it
    assert!(matches!(
        graph.node(ret).op,
        Operator::Control(ControlOp::Return)
    ));
}

#[test]
fn test_control_phi_node() {
    let mut builder = GraphBuilder::new(4, 0);

    // Create two values
    let v1 = builder.const_int(10);
    let v2 = builder.const_int(20);

    // Create control merge
    let start = builder.control();
    let region = builder.region(&[start, start]); // Dummy for test

    // Create phi
    let phi = builder.phi(region, &[v1, v2], ValueType::Int64);

    let graph = builder.finish();

    assert!(matches!(graph.node(phi).op, Operator::Phi));
    assert_eq!(graph.node(phi).ty, ValueType::Int64);
}

#[test]
fn test_control_loop_phi() {
    let mut builder = GraphBuilder::new(4, 0);

    let entry = builder.control();
    let loop_head = builder.loop_header(entry);

    let initial = builder.const_int(0);
    let phi = builder.loop_phi(loop_head, initial, ValueType::Int64);

    // Set back edge
    let one = builder.const_int(1);
    let next = builder.int_add(phi, one);
    builder.set_loop_phi_back(phi, next);

    let graph = builder.finish();

    assert!(matches!(graph.node(phi).op, Operator::LoopPhi));
    // LoopPhi should have 3 inputs: loop_header, initial, back_edge
    assert_eq!(graph.node(phi).inputs.len(), 3);
}

#[test]
fn test_translate_jump_saves_state() {
    let mut builder = GraphBuilder::new(4, 0);

    let c = builder.const_int(42);
    builder.set_register(0, c);

    // Jump to target 10
    builder.translate_jump(10);

    // State should be saved at offset 10
    assert!(builder.states_at_offset.contains_key(&10));
}

#[test]
fn test_translate_branch_creates_projections() {
    let mut builder = GraphBuilder::new(4, 0);

    let cond = builder.const_bool(true);
    builder.translate_branch(cond, 5, 10);

    // Both targets should have states saved
    assert!(builder.states_at_offset.contains_key(&5));
    assert!(builder.states_at_offset.contains_key(&10));
}

// =============================================================================
// ObjectBuilder Tests
// =============================================================================

#[test]
fn test_object_get_item() {
    let mut builder = GraphBuilder::new(4, 0);

    let obj = builder.const_int(0); // Placeholder
    let key = builder.const_int(1);

    let result = builder.get_item(obj, key);

    let graph = builder.finish();

    assert!(matches!(graph.node(result).op, Operator::GetItem));
    assert_eq!(graph.node(result).inputs.len(), 2);
}

#[test]
fn test_object_set_item() {
    let mut builder = GraphBuilder::new(4, 0);

    let obj = builder.const_int(0); // Placeholder
    let key = builder.const_int(1);
    let value = builder.const_int(42);

    let result = builder.set_item(obj, key, value);

    let graph = builder.finish();

    assert!(matches!(graph.node(result).op, Operator::SetItem));
    assert_eq!(graph.node(result).inputs.len(), 3);
}

#[test]
fn test_object_get_attr() {
    let mut builder = GraphBuilder::new(4, 0);

    let obj = builder.const_int(0); // Placeholder
    let name = builder.const_int(1); // Attribute name node

    let result = builder.get_attr(obj, name);

    let graph = builder.finish();

    assert!(matches!(graph.node(result).op, Operator::GetAttr));
    assert_eq!(graph.node(result).ty, ValueType::Top);
}

#[test]
fn test_object_set_attr() {
    let mut builder = GraphBuilder::new(4, 0);

    let obj = builder.const_int(0); // Placeholder
    let name = builder.const_int(1); // Attribute name node
    let value = builder.const_int(42);

    let result = builder.set_attr(obj, name, value);

    let graph = builder.finish();

    assert!(matches!(graph.node(result).op, Operator::SetAttr));
    assert_eq!(graph.node(result).inputs.len(), 3);
}

#[test]
fn test_object_call_direct() {
    let mut builder = GraphBuilder::new(4, 0);

    let func = builder.const_int(0); // Placeholder function
    let arg1 = builder.const_int(10);
    let arg2 = builder.const_int(20);

    let result = builder.call(func, &[arg1, arg2]);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(result).op,
        Operator::Call(crate::ir::operators::CallKind::Direct)
    ));
    // Call node inputs: [function, arg1, arg2]
    assert_eq!(graph.node(result).inputs.len(), 3);
}

#[test]
fn test_object_call_no_args() {
    let mut builder = GraphBuilder::new(4, 0);

    let func = builder.const_int(0); // Placeholder function

    let result = builder.call(func, &[]);

    let graph = builder.finish();

    assert!(matches!(
        graph.node(result).op,
        Operator::Call(crate::ir::operators::CallKind::Direct)
    ));
    // Call node inputs: [function]
    assert_eq!(graph.node(result).inputs.len(), 1);
}

#[test]
fn test_object_call_method() {
    let mut builder = GraphBuilder::new(4, 0);

    let obj = builder.const_int(0); // Placeholder object
    let method_name = builder.const_int(1); // Method name node
    let arg = builder.const_int(42);

    let result = builder.call_method(method_name, obj, &[arg]);

    let graph = builder.finish();

    // call_method creates GetAttr + Call
    // The final result should be a Call node
    assert!(matches!(
        graph.node(result).op,
        Operator::Call(crate::ir::operators::CallKind::Direct)
    ));
}

// =============================================================================
// ContainerBuilder Tests
// =============================================================================

#[test]
fn test_container_build_list_empty() {
    let mut builder = GraphBuilder::new(4, 0);

    let list = builder.build_list(&[]);

    let graph = builder.finish();

    assert!(matches!(graph.node(list).op, Operator::BuildList(0)));
    assert_eq!(graph.node(list).ty, ValueType::List);
}

#[test]
fn test_container_build_list_with_elements() {
    let mut builder = GraphBuilder::new(4, 0);

    let e1 = builder.const_int(1);
    let e2 = builder.const_int(2);
    let e3 = builder.const_int(3);

    let list = builder.build_list(&[e1, e2, e3]);

    let graph = builder.finish();

    assert!(matches!(graph.node(list).op, Operator::BuildList(3)));
    assert_eq!(graph.node(list).inputs.len(), 3);
    assert_eq!(graph.node(list).ty, ValueType::List);
}

#[test]
fn test_container_build_tuple_empty() {
    let mut builder = GraphBuilder::new(4, 0);

    let tuple = builder.build_tuple(&[]);

    let graph = builder.finish();

    assert!(matches!(graph.node(tuple).op, Operator::BuildTuple(0)));
    assert_eq!(graph.node(tuple).ty, ValueType::Tuple);
}

#[test]
fn test_container_build_tuple_with_elements() {
    let mut builder = GraphBuilder::new(4, 0);

    let e1 = builder.const_int(10);
    let e2 = builder.const_float(3.14);

    let tuple = builder.build_tuple(&[e1, e2]);

    let graph = builder.finish();

    assert!(matches!(graph.node(tuple).op, Operator::BuildTuple(2)));
    assert_eq!(graph.node(tuple).inputs.len(), 2);
}

#[test]
fn test_container_get_iter() {
    let mut builder = GraphBuilder::new(4, 0);

    let iterable = builder.const_int(0); // Placeholder

    let iter = builder.get_iter(iterable);

    let graph = builder.finish();

    assert!(matches!(graph.node(iter).op, Operator::GetIter));
    assert_eq!(graph.node(iter).inputs.len(), 1);
}

#[test]
fn test_container_iter_next() {
    let mut builder = GraphBuilder::new(4, 0);

    let iterator = builder.const_int(0); // Placeholder

    let next_val = builder.iter_next(iterator);

    let graph = builder.finish();

    assert!(matches!(graph.node(next_val).op, Operator::IterNext));
}

#[test]
fn test_container_len() {
    let mut builder = GraphBuilder::new(4, 0);

    let obj = builder.const_int(0); // Placeholder

    let length = builder.len(obj);

    let graph = builder.finish();

    assert!(matches!(graph.node(length).op, Operator::Len));
    assert_eq!(graph.node(length).ty, ValueType::Int64);
}

// =============================================================================
// State Management Tests
// =============================================================================

#[test]
fn test_state_save_and_merge() {
    let mut builder = GraphBuilder::new(4, 0);

    let c1 = builder.const_int(10);
    builder.set_register(0, c1);

    // Save state at offset 5
    builder.save_state(5);

    // Change register
    let c2 = builder.const_int(20);
    builder.set_register(0, c2);

    // Save again at offset 5 (should merge)
    builder.save_state(5);

    // Verify state exists
    assert!(builder.states_at_offset.contains_key(&5));
}

#[test]
fn test_merge_state_at_offset() {
    let mut builder = GraphBuilder::new(4, 0);

    let c1 = builder.const_int(10);
    builder.set_register(0, c1);

    // Save at target
    builder.save_state(10);

    // Change value
    let c2 = builder.const_int(20);
    builder.set_register(0, c2);

    // Merge at target
    builder.merge_state(10);

    // After merge, we should have a phi or merged control
    // The region should be created since values differ
    let graph = builder.finish();
    assert!(graph.len() > 3); // start, end, plus merged nodes
}

#[test]
fn test_set_bc_offset() {
    let mut builder = GraphBuilder::new(4, 0);

    builder.set_bc_offset(42);

    // Add a node
    let c = builder.const_int(10);

    let graph = builder.finish();

    // The node should have bc_offset set (implementation detail)
    // Just verify no crash
    assert!(c.is_valid());
    let _ = graph;
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_parameter_beyond_count() {
    let builder = GraphBuilder::new(4, 2);

    // Only 2 parameters
    assert!(builder.parameter(0).is_some());
    assert!(builder.parameter(1).is_some());
    assert!(builder.parameter(2).is_none());
    assert!(builder.parameter(100).is_none());
}

#[test]
fn test_register_out_of_bounds() {
    let mut builder = GraphBuilder::new(4, 0);

    let c = builder.const_int(42);

    // Setting beyond bounds should not crash (silently ignored)
    builder.set_register(100, c);

    // Getting beyond bounds returns invalid
    assert!(!builder.get_register(100).is_valid());
}

#[test]
fn test_many_registers() {
    let mut builder = GraphBuilder::new(256, 0);

    for i in 0..256 {
        let c = builder.const_int(i as i64);
        builder.set_register(i as u16, c);
    }

    for i in 0..256 {
        assert!(builder.get_register(i as u16).is_valid());
    }
}

#[test]
fn test_complex_expression_tree() {
    let mut builder = GraphBuilder::new(4, 0);

    // Build: ((a + b) * (c - d)) / e
    let a = builder.const_int(10);
    let b = builder.const_int(5);
    let c = builder.const_int(20);
    let d = builder.const_int(3);
    let e = builder.const_int(2);

    let ab = builder.int_add(a, b);
    let cd = builder.int_sub(c, d);
    let mul = builder.int_mul(ab, cd);
    let result = builder.int_div(mul, e);

    let graph = builder.finish();

    // Verify the result is connected
    assert!(result.is_valid());
    assert!(matches!(
        graph.node(result).op,
        Operator::IntOp(ArithOp::FloorDiv)
    ));
}
