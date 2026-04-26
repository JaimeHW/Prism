use super::*;
use crate::ir::cfg::DominatorTree;

// =========================================================================
// LoopTripCount Tests
// =========================================================================

#[test]
fn test_trip_count_constant() {
    let tc = LoopTripCount::Constant(10);
    assert!(tc.is_constant());
    assert_eq!(tc.as_constant(), Some(10));
    assert_eq!(tc.upper_bound(), Some(10));
    assert_eq!(tc.lower_bound(), Some(10));
}

#[test]
fn test_trip_count_bounded() {
    let tc = LoopTripCount::Bounded { min: 5, max: 100 };
    assert!(!tc.is_constant());
    assert_eq!(tc.as_constant(), None);
    assert_eq!(tc.upper_bound(), Some(100));
    assert_eq!(tc.lower_bound(), Some(5));
}

#[test]
fn test_trip_count_parameter() {
    let tc = LoopTripCount::Parameter {
        param: 0,
        offset: 0,
        scale: 1,
    };
    assert!(!tc.is_constant());
    assert_eq!(tc.upper_bound(), Some(1000)); // Heuristic
}

#[test]
fn test_trip_count_unknown() {
    let tc = LoopTripCount::Unknown;
    assert!(!tc.is_constant());
    assert_eq!(tc.as_constant(), None);
    assert!(tc.upper_bound().is_none());
    assert_eq!(tc.lower_bound(), Some(0));
}

#[test]
fn test_trip_count_default() {
    let tc = LoopTripCount::default();
    assert_eq!(tc, LoopTripCount::Unknown);
}

// =========================================================================
// UnrollabilityAnalysis Tests
// =========================================================================

fn make_canonical_analysis() -> UnrollabilityAnalysis {
    UnrollabilityAnalysis {
        loop_idx: 0,
        trip_count: LoopTripCount::Constant(4),
        body_size: 10,
        has_single_entry: true,
        has_single_exit: true,
        contains_calls: false,
        has_memory_effects: false,
        has_early_exits: false,
        nesting_depth: 0,
        induction_vars: vec![NodeId::new(5)],
        register_pressure: 4,
        is_canonical: true,
        body_nodes: FxHashSet::default(),
    }
}

#[test]
fn test_can_fully_unroll_yes() {
    let analysis = make_canonical_analysis();
    assert!(analysis.can_fully_unroll(16, 100));
}

#[test]
fn test_can_fully_unroll_trip_too_large() {
    let analysis = make_canonical_analysis();
    assert!(!analysis.can_fully_unroll(2, 100)); // max_trip = 2, trip = 4
}

#[test]
fn test_can_fully_unroll_size_too_large() {
    let analysis = make_canonical_analysis();
    assert!(!analysis.can_fully_unroll(16, 20)); // 4 * 10 = 40 > 20
}

#[test]
fn test_can_fully_unroll_not_canonical() {
    let mut analysis = make_canonical_analysis();
    analysis.is_canonical = false;
    assert!(!analysis.can_fully_unroll(16, 100));
}

#[test]
fn test_can_partial_unroll_yes() {
    let analysis = make_canonical_analysis();
    assert!(analysis.can_partial_unroll(2));
}

#[test]
fn test_can_partial_unroll_trip_too_small() {
    let analysis = make_canonical_analysis();
    assert!(!analysis.can_partial_unroll(10)); // trip = 4 < 10
}

#[test]
fn test_can_partial_unroll_no_iv() {
    let mut analysis = make_canonical_analysis();
    analysis.induction_vars.clear();
    assert!(!analysis.can_partial_unroll(2));
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_analyzer_empty_loops() {
    use crate::ir::builder::GraphBuilder;

    let builder = GraphBuilder::new(4, 0);
    let graph = builder.finish();

    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);

    let analyzer = UnrollabilityAnalyzer::new(&graph, &loops, &cfg);
    assert!(analyzer.analyze(0).is_none()); // No loops
}

#[test]
fn test_trip_count_analyzer_no_loops() {
    use crate::ir::builder::GraphBuilder;

    let builder = GraphBuilder::new(4, 0);
    let graph = builder.finish();

    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);

    let analyzer = TripCountAnalyzer::new(&graph, &loops, &cfg);
    assert_eq!(analyzer.analyze(0), LoopTripCount::Unknown);
}
