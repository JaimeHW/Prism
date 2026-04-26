use super::*;
use crate::ir::builder::{
    ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
};

#[test]
fn test_gvn_duplicate_constants() {
    let mut builder = GraphBuilder::new(4, 0);

    // Create duplicate constants
    let a = builder.const_int(42);
    let b = builder.const_int(42);
    let sum = builder.int_add(a, b);
    let _ret = builder.return_value(sum);

    let mut graph = builder.finish();
    let initial_len = graph.len();

    let mut gvn = Gvn::new();
    let changed = gvn.run(&mut graph);

    assert!(changed);
    assert!(gvn.deduplicated() >= 1);
    // The duplicate constant should be deduplicated
    assert!(graph.len() <= initial_len);
}

#[test]
fn test_gvn_duplicate_arithmetic() {
    let mut builder = GraphBuilder::new(4, 2);

    // p0 + p1
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum1 = builder.int_add(p0, p1);

    // p0 + p1 again (duplicate)
    let sum2 = builder.int_add(p0, p1);

    // Use both
    let result = builder.int_add(sum1, sum2);
    let _ret = builder.return_value(result);

    let mut graph = builder.finish();

    let mut gvn = Gvn::new();
    let changed = gvn.run(&mut graph);

    // sum2 should be replaced with sum1
    assert!(changed);
    assert_eq!(gvn.deduplicated(), 1);
}

#[test]
fn test_gvn_different_ops_not_merged() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // Different operations should not be merged
    let sum = builder.int_add(p0, p1);
    let sub = builder.int_sub(p0, p1);

    let result = builder.int_add(sum, sub);
    let _ret = builder.return_value(result);

    let mut graph = builder.finish();

    let mut gvn = Gvn::new();
    let changed = gvn.run(&mut graph);

    // Nothing to deduplicate
    assert!(!changed);
    assert_eq!(gvn.deduplicated(), 0);
}

#[test]
fn test_gvn_comparison() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // Same comparison twice
    let lt1 = builder.int_lt(p0, p1);
    let lt2 = builder.int_lt(p0, p1);

    let _ = builder.const_bool(true); // Use both somehow
    builder.set_register(0, lt1);
    builder.set_register(1, lt2);

    let mut graph = builder.finish();

    let mut gvn = Gvn::new();
    let changed = gvn.run(&mut graph);

    assert!(changed);
    assert_eq!(gvn.deduplicated(), 1);
}

#[test]
fn test_gvn_preserves_side_effects() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // GetItem has side effects, should not be merged
    let get1 = builder.get_item(p0, p1);
    let get2 = builder.get_item(p0, p1);

    let result = builder.int_add(get1, get2);
    let _ret = builder.return_value(result);

    let mut graph = builder.finish();

    let mut gvn = Gvn::new();
    let changed = gvn.run(&mut graph);

    // get_item should not be deduplicated (has side effects)
    assert!(!changed);
    assert_eq!(gvn.deduplicated(), 0);
}
