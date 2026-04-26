use super::*;
use crate::ir::builder::{ArithmeticBuilder, BitwiseBuilder, ControlBuilder, GraphBuilder};

// =========================================================================
// Basic Tests
// =========================================================================

#[test]
fn test_strength_reduce_new() {
    let sr = StrengthReduce::new();
    assert_eq!(sr.stats().total, 0);
}

#[test]
fn test_strength_reduce_default() {
    let sr = StrengthReduce::default();
    assert!(sr.config.decompose_multiply);
    assert!(sr.config.optimize_division);
}

#[test]
fn test_config_conservative() {
    let config = StrengthReduceConfig::conservative();
    assert_eq!(config.multiply_config.max_ops, 2);
}

#[test]
fn test_config_aggressive() {
    let config = StrengthReduceConfig::aggressive();
    assert!(config.multiply_config.max_ops >= 6);
}

// =========================================================================
// Multiplication Tests
// =========================================================================

#[test]
fn test_mul_by_power_of_two() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c4 = builder.const_int(4);
    let mul = builder.int_mul(p0, c4);
    builder.return_value(mul);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().mul_to_shift, 1);
}

#[test]
fn test_mul_by_negative_power_of_two() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c_neg4 = builder.const_int(-4);
    let mul = builder.int_mul(p0, c_neg4);
    builder.return_value(mul);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().mul_to_shift, 1);
}

#[test]
fn test_mul_by_three() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c3 = builder.const_int(3);
    let mul = builder.int_mul(p0, c3);
    builder.return_value(mul);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().mul_decomposed, 1);
}

// =========================================================================
// Division Tests
// =========================================================================

#[test]
fn test_div_by_power_of_two() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c8 = builder.const_int(8);
    let div = builder.int_div(p0, c8);
    builder.return_value(div);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert!(sr.stats().div_signed_replaced > 0 || sr.stats().div_unsigned_replaced > 0);
}

// =========================================================================
// Modulo Tests
// =========================================================================

#[test]
fn test_mod_by_power_of_two() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c16 = builder.const_int(16);
    let modulo = builder.int_mod(p0, c16);
    builder.return_value(modulo);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().mod_replaced, 1);
}

// =========================================================================
// Bitwise Tests
// =========================================================================

#[test]
fn test_and_with_zero() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c0 = builder.const_int(0);
    let and = builder.bitwise_and(p0, c0);
    builder.return_value(and);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().bitwise_simplified, 1);
}

#[test]
fn test_and_with_neg_one() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c_neg1 = builder.const_int(-1);
    let and = builder.bitwise_and(p0, c_neg1);
    builder.return_value(and);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().bitwise_simplified, 1);
}

#[test]
fn test_or_with_zero() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c0 = builder.const_int(0);
    let or = builder.bitwise_or(p0, c0);
    builder.return_value(or);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().bitwise_simplified, 1);
}

#[test]
fn test_or_with_neg_one() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let c_neg1 = builder.const_int(-1);
    let or = builder.bitwise_or(p0, c_neg1);
    builder.return_value(or);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().bitwise_simplified, 1);
}

#[test]
fn test_xor_with_self() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let xor = builder.bitwise_xor(p0, p0);
    builder.return_value(xor);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().bitwise_simplified, 1);
}

#[test]
fn test_and_with_self() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let and = builder.bitwise_and(p0, p0);
    builder.return_value(and);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(changed);
    assert_eq!(sr.stats().bitwise_simplified, 1);
}

// =========================================================================
// Stats Tests
// =========================================================================

#[test]
fn test_stats_reset() {
    let mut stats = StrengthReduceStats::default();
    stats.total = 10;
    stats.mul_to_shift = 5;

    stats.reset();

    assert_eq!(stats.total, 0);
    assert_eq!(stats.mul_to_shift, 0);
}

// =========================================================================
// No-op Tests
// =========================================================================

#[test]
fn test_no_transform_needed() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let mul = builder.int_mul(p0, p1); // Variable * variable, can't reduce
    builder.return_value(mul);

    let mut graph = builder.finish();
    let mut sr = StrengthReduce::new();

    let changed = sr.run(&mut graph);

    assert!(!changed);
    assert_eq!(sr.stats().total, 0);
}
