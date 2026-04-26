use super::*;
use crate::ir::builder::{
    ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
};

#[test]
fn test_default_config() {
    let config = OptConfig::default();
    assert!(config.enable_gvn);
    assert!(config.enable_dce);
    assert!(config.enable_simplify);
    assert_eq!(config.max_iterations, 10);
}

#[test]
fn test_optimize_simple() {
    let mut builder = GraphBuilder::new(4, 2);

    // Create a simple computation: p0 + p1
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    let _ret = builder.return_value(sum);

    let mut graph = builder.finish();
    let config = OptConfig::default();

    let stats = optimize(&mut graph, &config);
    assert!(stats.iterations >= 1);
}
