use super::*;
use crate::ir::builder::{
    ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
};

#[test]
fn test_simplify_constant_fold_add() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(3);
    let b = builder.const_int(4);
    let sum = builder.int_add(a, b);
    let _ret = builder.return_value(sum);

    let mut graph = builder.finish();

    let mut simplify = Simplify::new();
    let changed = simplify.run(&mut graph);

    assert!(changed);
    assert_eq!(simplify.simplifications(), 1);
}

#[test]
fn test_simplify_identity_add_zero() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let zero = builder.const_int(0);
    let sum = builder.int_add(p0, zero);
    let _ret = builder.return_value(sum);

    let mut graph = builder.finish();

    let mut simplify = Simplify::new();
    let changed = simplify.run(&mut graph);

    assert!(changed);
    assert_eq!(simplify.simplifications(), 1);
}

#[test]
fn test_simplify_identity_mul_one() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let one = builder.const_int(1);
    let product = builder.int_mul(p0, one);
    let _ret = builder.return_value(product);

    let mut graph = builder.finish();

    let mut simplify = Simplify::new();
    let changed = simplify.run(&mut graph);

    assert!(changed);
    assert_eq!(simplify.simplifications(), 1);
}

#[test]
fn test_simplify_absorbing_mul_zero() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let zero = builder.const_int(0);
    let product = builder.int_mul(p0, zero);
    let _ret = builder.return_value(product);

    let mut graph = builder.finish();

    let mut simplify = Simplify::new();
    let changed = simplify.run(&mut graph);

    assert!(changed);
    assert_eq!(simplify.simplifications(), 1);
}

#[test]
fn test_simplify_sub_self() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let diff = builder.int_sub(p0, p0);
    let _ret = builder.return_value(diff);

    let mut graph = builder.finish();

    let mut simplify = Simplify::new();
    let changed = simplify.run(&mut graph);

    assert!(changed);
    assert_eq!(simplify.simplifications(), 1);
}

#[test]
fn test_simplify_cmp_self_eq() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let eq = builder.int_eq(p0, p0);
    let _ret = builder.return_value(eq);

    let mut graph = builder.finish();

    let mut simplify = Simplify::new();
    let changed = simplify.run(&mut graph);

    assert!(changed);
    assert_eq!(simplify.simplifications(), 1);
}

#[test]
fn test_simplify_cmp_constants() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(5);
    let b = builder.const_int(10);
    let lt = builder.int_lt(a, b); // 5 < 10 = true
    let _ret = builder.return_value(lt);

    let mut graph = builder.finish();

    let mut simplify = Simplify::new();
    let changed = simplify.run(&mut graph);

    assert!(changed);
    assert_eq!(simplify.simplifications(), 1);
}

#[test]
fn test_simplify_no_change() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1); // Can't simplify
    let _ret = builder.return_value(sum);

    let mut graph = builder.finish();

    let mut simplify = Simplify::new();
    let changed = simplify.run(&mut graph);

    assert!(!changed);
    assert_eq!(simplify.simplifications(), 0);
}
