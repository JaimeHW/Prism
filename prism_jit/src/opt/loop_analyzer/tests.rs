use super::*;
use crate::ir::builder::GraphBuilder;

#[test]
fn test_is_defined_outside_loop() {
    assert!(LoopInvariantAnalysis::is_defined_outside_loop(
        &Operator::ConstInt(42)
    ));
    assert!(LoopInvariantAnalysis::is_defined_outside_loop(
        &Operator::Parameter(0)
    ));
    assert!(!LoopInvariantAnalysis::is_defined_outside_loop(
        &Operator::IntOp(ArithOp::Add)
    ));
}

#[test]
fn test_can_trap() {
    assert!(LoopInvariantAnalysis::can_trap(&Operator::IntOp(
        ArithOp::TrueDiv
    )));
    assert!(LoopInvariantAnalysis::can_trap(&Operator::IntOp(
        ArithOp::Mod
    )));
    assert!(!LoopInvariantAnalysis::can_trap(&Operator::IntOp(
        ArithOp::Add
    )));
}

#[test]
fn test_loop_invariant_analysis_empty() {
    let builder = GraphBuilder::new(0, 0);
    let graph = builder.finish();
    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);

    let analysis = LoopInvariantAnalysis::compute(&graph, &cfg, &dom, &loops);
    assert_eq!(analysis.total_invariants(), 0);
    assert_eq!(analysis.total_hoistable(), 0);
}

#[test]
fn test_invariant_query() {
    let builder = GraphBuilder::new(0, 0);
    let graph = builder.finish();
    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);

    let analysis = LoopInvariantAnalysis::compute(&graph, &cfg, &dom, &loops);
    assert!(!analysis.is_invariant(0, NodeId::new(0)));
    assert!(!analysis.is_hoistable(0, NodeId::new(0)));
}

#[test]
fn test_preheader_inserter_no_change() {
    let builder = GraphBuilder::new(0, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);

    let mut inserter = PreheaderInserter::new(&mut graph, &mut cfg);
    let preheaders = inserter.insert_all(&loops);
    assert!(preheaders.is_empty());
}
