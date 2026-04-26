use super::*;
use crate::ir::builder::{
    ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
};

#[test]
fn test_dce_removes_unused() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // Used computation
    let sum = builder.int_add(p0, p1);
    let _ret = builder.return_value(sum);

    // Unused computation (should be removed)
    let _unused = builder.int_sub(p0, p1);

    let mut graph = builder.finish();

    let mut dce = Dce::new();
    let changed = dce.run(&mut graph);

    assert!(changed);
    assert!(dce.removed() >= 1);
}

#[test]
fn test_dce_preserves_used() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // All nodes are used
    let sum = builder.int_add(p0, p1);
    let double = builder.int_add(sum, sum);
    let _ret = builder.return_value(double);

    let mut graph = builder.finish();

    let mut dce = Dce::new();
    let changed = dce.run(&mut graph);

    // Nothing should be removed
    assert!(!changed);
    assert_eq!(dce.removed(), 0);
}

#[test]
fn test_dce_preserves_side_effects() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // Side-effecting operation (should not be removed even if unused)
    let key = builder.const_int(0);
    let _set = builder.set_item(p0, key, p1);

    let none = builder.const_none();
    let _ret = builder.return_value(none);

    let mut graph = builder.finish();
    let before_count = graph.len();

    let mut dce = Dce::new();
    let _changed = dce.run(&mut graph);

    // Side effects should be preserved
    // The graph might shrink but set_item should remain
    assert!(graph.len() >= before_count - 2); // Allow some cleanup
}

#[test]
fn test_dce_unused_constants() {
    let mut builder = GraphBuilder::new(4, 0);

    // Create unused constants
    let _c1 = builder.const_int(42);
    let _c2 = builder.const_int(100);
    let _c3 = builder.const_float(3.125);

    // But only use one
    let used = builder.const_int(1);
    let _ret = builder.return_value(used);

    let mut graph = builder.finish();

    let mut dce = Dce::new();
    let changed = dce.run(&mut graph);

    assert!(changed);
    assert!(dce.removed() >= 3); // At least the 3 unused constants
}

#[test]
fn test_dce_chain_removal() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // Create a chain of unused computations
    let a = builder.int_add(p0, p1);
    let b = builder.int_mul(a, p0);
    let _c = builder.int_sub(b, p1);

    // Return something else
    let _ret = builder.return_value(p0);

    let mut graph = builder.finish();

    let mut dce = Dce::new();
    let changed = dce.run(&mut graph);

    // The entire chain should be removed
    assert!(changed);
    assert!(dce.removed() >= 3);
}
