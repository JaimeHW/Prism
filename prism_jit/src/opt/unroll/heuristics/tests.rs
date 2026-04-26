use super::*;
use crate::ir::node::NodeId;
use crate::opt::unroll::analysis::LoopTripCount;
use rustc_hash::FxHashSet;

fn make_analysis(trip: LoopTripCount, body_size: usize, canonical: bool) -> UnrollabilityAnalysis {
    UnrollabilityAnalysis {
        loop_idx: 0,
        trip_count: trip,
        body_size,
        has_single_entry: canonical,
        has_single_exit: canonical,
        contains_calls: false,
        has_memory_effects: false,
        has_early_exits: !canonical,
        nesting_depth: 0,
        induction_vars: if canonical {
            vec![NodeId::new(5)]
        } else {
            vec![]
        },
        register_pressure: 4,
        is_canonical: canonical,
        body_nodes: FxHashSet::default(),
    }
}

// =========================================================================
// Cost Model Tests
// =========================================================================

#[test]
fn test_cost_model_default() {
    let model = UnrollCostModel::default();
    assert_eq!(model.loop_iteration_cost, 3.0);
    assert_eq!(model.target_register_count, 16);
}

#[test]
fn test_cost_model_server() {
    let model = UnrollCostModel::server();
    assert_eq!(model.target_icache_size, 65536);
}

#[test]
fn test_cost_model_mobile() {
    let model = UnrollCostModel::mobile();
    assert_eq!(model.target_icache_size, 16384);
    assert!(model.instruction_cost > UnrollCostModel::default().instruction_cost);
}

#[test]
fn test_cost_model_baseline_cost() {
    let model = UnrollCostModel::default();
    let analysis = make_analysis(LoopTripCount::Constant(10), 10, true);
    let cost = model.baseline_cost(&analysis);
    assert!(cost > 0.0);
}

#[test]
fn test_cost_model_full_unroll_cost() {
    let model = UnrollCostModel::default();
    let analysis = make_analysis(LoopTripCount::Constant(4), 10, true);
    let cost = model.full_unroll_cost(&analysis, 4);
    assert!(cost >= 0.0);
}

#[test]
fn test_cost_model_partial_unroll_cost() {
    let model = UnrollCostModel::default();
    let analysis = make_analysis(LoopTripCount::Constant(100), 10, true);
    let cost = model.partial_unroll_cost(&analysis, 4);
    assert!(cost > 0.0);
}

#[test]
fn test_cost_model_unroll_benefit() {
    let model = UnrollCostModel::default();
    let analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
    let benefit = model.unroll_benefit(&analysis, 4);
    // Full unroll of small loop should be beneficial
    assert!(benefit > 0.0);
}

// =========================================================================
// Unroll Decision Tests
// =========================================================================

#[test]
fn test_unroll_decision_no_unroll() {
    let decision = UnrollDecision::no_unroll(NoUnrollReason::BodyTooLarge);
    assert!(matches!(decision.strategy, UnrollStrategy::NoUnroll { .. }));
    assert_eq!(decision.estimated_speedup, 1.0);
}

// =========================================================================
// Heuristics Tests
// =========================================================================

#[test]
fn test_heuristics_full_unroll_small_loop() {
    let config = UnrollConfig::default();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
    let strategy = heuristics.determine_strategy(&analysis);

    assert!(matches!(
        strategy,
        UnrollStrategy::FullUnroll { trip_count: 4 }
    ));
}

#[test]
fn test_heuristics_no_unroll_non_canonical() {
    let config = UnrollConfig::default();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let analysis = make_analysis(LoopTripCount::Constant(4), 5, false);
    let strategy = heuristics.determine_strategy(&analysis);

    assert!(matches!(
        strategy,
        UnrollStrategy::NoUnroll {
            reason: NoUnrollReason::NotCanonical
        }
    ));
}

#[test]
fn test_heuristics_partial_unroll_large_trip() {
    let config = UnrollConfig::default();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let analysis = make_analysis(LoopTripCount::Constant(1000), 10, true);
    let strategy = heuristics.determine_strategy(&analysis);

    assert!(matches!(
        strategy,
        UnrollStrategy::PartialUnroll { .. } | UnrollStrategy::NoUnroll { .. }
    ));
}

#[test]
fn test_heuristics_no_unroll_too_large_trip() {
    let config = UnrollConfig::default();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let analysis = make_analysis(LoopTripCount::Constant(32), 100, true);
    let strategy = heuristics.determine_strategy(&analysis);

    // Body too large for full unroll, should try partial or skip
    assert!(!matches!(strategy, UnrollStrategy::FullUnroll { .. }));
}

#[test]
fn test_heuristics_chooses_epilog_for_large() {
    let config = UnrollConfig::default();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let remainder = heuristics
        .choose_remainder_strategy(&make_analysis(LoopTripCount::Constant(100), 50, true), 8);

    assert_eq!(remainder, RemainderStrategy::EpilogLoop);
}

#[test]
fn test_heuristics_chooses_unrolled_for_small() {
    let config = UnrollConfig::default();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let remainder = heuristics
        .choose_remainder_strategy(&make_analysis(LoopTripCount::Constant(100), 8, true), 4);

    assert_eq!(remainder, RemainderStrategy::UnrolledRemainder);
}

#[test]
fn test_heuristics_no_calls_block() {
    let config = UnrollConfig::default();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let mut analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
    analysis.contains_calls = true;

    let strategy = heuristics.determine_strategy(&analysis);

    assert!(matches!(
        strategy,
        UnrollStrategy::NoUnroll {
            reason: NoUnrollReason::ContainsCalls
        }
    ));
}

#[test]
fn test_heuristics_aggressive_allows_calls() {
    let config = UnrollConfig::aggressive();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let mut analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
    analysis.contains_calls = true;

    let strategy = heuristics.determine_strategy(&analysis);

    // Aggressive config allows calls
    assert!(!matches!(
        strategy,
        UnrollStrategy::NoUnroll {
            reason: NoUnrollReason::ContainsCalls
        }
    ));
}

#[test]
fn test_heuristics_nesting_limit() {
    let config = UnrollConfig::default();
    let model = UnrollCostModel::default();
    let heuristics = UnrollHeuristics::new(&config, &model);

    let mut analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
    analysis.nesting_depth = 5; // Too deep

    let strategy = heuristics.determine_strategy(&analysis);

    assert!(matches!(
        strategy,
        UnrollStrategy::NoUnroll {
            reason: NoUnrollReason::NestingTooDeep
        }
    ));
}
