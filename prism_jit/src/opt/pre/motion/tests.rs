use super::*;
use crate::ir::graph::Graph;
use crate::opt::pre::ExpressionTable;
use crate::opt::pre::anticipation::AnticipationAnalysis;
use crate::opt::pre::availability::AvailabilityAnalysis;

#[test]
fn test_code_motion_empty() {
    let mut graph = Graph::new();
    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);
    let placement = PlacementAnalysis::compute(&antic, &avail, &expr_table);

    let mut motion = CodeMotionEngine::new(&mut graph, &placement);
    let changed = motion.apply();

    assert!(!changed);
    assert_eq!(motion.inserted(), 0);
    assert_eq!(motion.eliminated(), 0);
}

#[test]
fn test_code_motion_graph_access() {
    let mut graph = Graph::new();
    let c = graph.const_int(42);

    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);
    let placement = PlacementAnalysis::compute(&antic, &avail, &expr_table);

    let motion = CodeMotionEngine::new(&mut graph, &placement);
    let g = motion.graph();

    // Should be able to access the graph
    assert!(g.get(c).is_some());
}
