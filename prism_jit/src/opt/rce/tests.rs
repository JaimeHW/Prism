use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

#[test]
fn test_rce_new() {
    let rce = RangeCheckElimination::new();
    assert!(!rce.aggressive);
    assert_eq!(rce.checks_eliminated(), 0);
    assert_eq!(rce.checks_hoisted(), 0);
    assert_eq!(rce.induction_vars_found(), 0);
}

#[test]
fn test_rce_aggressive() {
    let rce = RangeCheckElimination::aggressive();
    assert!(rce.aggressive);
}

#[test]
fn test_rce_default() {
    let rce = RangeCheckElimination::default();
    assert!(!rce.aggressive);
}

#[test]
fn test_rce_name() {
    let rce = RangeCheckElimination::new();
    assert_eq!(rce.name(), "rce");
}

#[test]
fn test_rce_no_loops() {
    let mut builder = GraphBuilder::new(2, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut rce = RangeCheckElimination::new();

    let changed = rce.run(&mut graph);
    assert!(!changed);
    assert_eq!(rce.stats().loops_analyzed, 0);
}

#[test]
fn test_rce_stats_default() {
    let stats = RceStats::default();
    assert_eq!(stats.loops_analyzed, 0);
    assert_eq!(stats.induction_vars_found, 0);
    assert_eq!(stats.range_checks_found, 0);
    assert_eq!(stats.checks_eliminated, 0);
    assert_eq!(stats.checks_hoisted, 0);
    assert_eq!(stats.checks_widened, 0);
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_induction_analysis_integration() {
    // Create a simple graph and verify IV analysis works
    let builder = GraphBuilder::new(0, 0);
    let graph = builder.finish();

    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);

    // No loops in empty graph
    assert!(loops.loops.is_empty());
}

#[test]
fn test_range_check_collection_integration() {
    let mut collection = RangeCheckCollection::new();
    assert!(collection.is_empty());

    // Add some checks
    let check1 = RangeCheck::new(
        crate::ir::node::NodeId::new(0),
        crate::ir::node::NodeId::new(1),
        crate::ir::node::NodeId::new(2),
        BoundValue::Constant(0),
        RangeCheckKind::LowerBound,
        0,
    );
    let check2 = RangeCheck::new(
        crate::ir::node::NodeId::new(3),
        crate::ir::node::NodeId::new(1),
        crate::ir::node::NodeId::new(4),
        BoundValue::Constant(100),
        RangeCheckKind::UpperBound,
        0,
    );

    collection.add(check1);
    collection.add(check2);

    assert_eq!(collection.len(), 2);
    assert_eq!(collection.count_lower_bounds(), 1);
    assert_eq!(collection.count_upper_bounds(), 1);
}

#[test]
fn test_elimination_analyzer_integration() {
    let mut analyzer = EliminationAnalyzer::new();

    let iv = InductionVariable::new(
        crate::ir::node::NodeId::new(0),
        InductionInit::Constant(0),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    );

    let check = RangeCheck::new(
        crate::ir::node::NodeId::new(10),
        crate::ir::node::NodeId::new(0),
        crate::ir::node::NodeId::new(11),
        BoundValue::Constant(0),
        RangeCheckKind::LowerBound,
        0,
    );

    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Eliminate);
}
