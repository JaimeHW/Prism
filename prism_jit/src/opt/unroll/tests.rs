use super::*;

// =========================================================================
// Configuration Tests
// =========================================================================

#[test]
fn test_unroll_config_default() {
    let config = UnrollConfig::default();
    assert_eq!(config.max_full_unroll_trip, 16);
    assert_eq!(config.max_full_unroll_size, 64);
    assert_eq!(config.default_unroll_factor, 4);
    assert_eq!(config.max_unroll_factor, 16);
    assert!(config.enable_runtime_unroll);
    assert!(config.enable_remainder);
}

#[test]
fn test_unroll_config_conservative() {
    let config = UnrollConfig::conservative();
    assert_eq!(config.max_full_unroll_trip, 8);
    assert_eq!(config.max_full_unroll_size, 32);
    assert_eq!(config.default_unroll_factor, 2);
    assert!(!config.enable_runtime_unroll);
}

#[test]
fn test_unroll_config_aggressive() {
    let config = UnrollConfig::aggressive();
    assert_eq!(config.max_full_unroll_trip, 32);
    assert_eq!(config.max_full_unroll_size, 128);
    assert_eq!(config.default_unroll_factor, 8);
    assert!(config.aggressive_inner_loops);
}

#[test]
fn test_unroll_config_tier1() {
    let config = UnrollConfig::tier1();
    assert_eq!(config.max_full_unroll_trip, 4);
    assert!(!config.enable_runtime_unroll);
    assert!(!config.enable_remainder);
}

#[test]
fn test_unroll_config_tier2() {
    let config = UnrollConfig::tier2();
    assert_eq!(config.max_full_unroll_trip, 16);
    assert!(config.enable_runtime_unroll);
}

// =========================================================================
// Strategy Tests
// =========================================================================

#[test]
fn test_unroll_strategy_full() {
    let strategy = UnrollStrategy::FullUnroll { trip_count: 4 };
    assert!(matches!(strategy, UnrollStrategy::FullUnroll { .. }));
}

#[test]
fn test_unroll_strategy_partial() {
    let strategy = UnrollStrategy::PartialUnroll {
        factor: 4,
        remainder: RemainderStrategy::EpilogLoop,
    };
    if let UnrollStrategy::PartialUnroll { factor, remainder } = strategy {
        assert_eq!(factor, 4);
        assert_eq!(remainder, RemainderStrategy::EpilogLoop);
    } else {
        panic!("Wrong strategy type");
    }
}

#[test]
fn test_unroll_strategy_runtime() {
    let strategy = UnrollStrategy::RuntimeUnroll {
        min_trip: 8,
        factor: 4,
        remainder: RemainderStrategy::UnrolledRemainder,
    };
    if let UnrollStrategy::RuntimeUnroll {
        min_trip,
        factor,
        remainder,
    } = strategy
    {
        assert_eq!(min_trip, 8);
        assert_eq!(factor, 4);
        assert_eq!(remainder, RemainderStrategy::UnrolledRemainder);
    } else {
        panic!("Wrong strategy type");
    }
}

#[test]
fn test_unroll_strategy_no_unroll() {
    let strategy = UnrollStrategy::NoUnroll {
        reason: NoUnrollReason::BodyTooLarge,
    };
    if let UnrollStrategy::NoUnroll { reason } = strategy {
        assert_eq!(reason, NoUnrollReason::BodyTooLarge);
    } else {
        panic!("Wrong strategy type");
    }
}

// =========================================================================
// NoUnrollReason Tests
// =========================================================================

#[test]
fn test_no_unroll_reason_display() {
    assert_eq!(
        NoUnrollReason::BodyTooLarge.to_string(),
        "loop body too large"
    );
    assert_eq!(
        NoUnrollReason::UnknownTripCount.to_string(),
        "unknown trip count"
    );
    assert_eq!(
        NoUnrollReason::ComplexControlFlow.to_string(),
        "complex control flow"
    );
    assert_eq!(
        NoUnrollReason::ContainsCalls.to_string(),
        "contains function calls"
    );
    assert_eq!(NoUnrollReason::SideEffects.to_string(), "has side effects");
    assert_eq!(
        NoUnrollReason::RegisterPressure.to_string(),
        "register pressure too high"
    );
    assert_eq!(
        NoUnrollReason::CodeGrowthLimit.to_string(),
        "code growth limit exceeded"
    );
    assert_eq!(
        NoUnrollReason::NotCanonical.to_string(),
        "not in canonical form"
    );
    assert_eq!(
        NoUnrollReason::NestingTooDeep.to_string(),
        "nesting too deep"
    );
    assert_eq!(NoUnrollReason::NotProfitable.to_string(), "not profitable");
    assert_eq!(
        NoUnrollReason::AlreadyUnrolled.to_string(),
        "already unrolled"
    );
}

// =========================================================================
// Statistics Tests
// =========================================================================

#[test]
fn test_unroll_stats_default() {
    let stats = UnrollStats::default();
    assert_eq!(stats.loops_analyzed, 0);
    assert_eq!(stats.loops_fully_unrolled, 0);
    assert_eq!(stats.loops_partially_unrolled, 0);
    assert_eq!(stats.loops_runtime_unrolled, 0);
    assert_eq!(stats.loops_not_unrolled, 0);
    assert_eq!(stats.nodes_added, 0);
    assert_eq!(stats.nodes_removed, 0);
}

#[test]
fn test_unroll_stats_record_no_unroll() {
    let mut stats = UnrollStats::default();
    stats.record_no_unroll(NoUnrollReason::BodyTooLarge);
    assert_eq!(stats.loops_not_unrolled, 1);
    assert_eq!(
        stats.no_unroll_reasons[NoUnrollReason::BodyTooLarge as usize],
        1
    );
}

#[test]
fn test_unroll_stats_total_loops() {
    let mut stats = UnrollStats::default();
    stats.loops_fully_unrolled = 2;
    stats.loops_partially_unrolled = 3;
    stats.loops_runtime_unrolled = 1;
    stats.loops_not_unrolled = 4;
    assert_eq!(stats.total_loops(), 10);
}

#[test]
fn test_unroll_stats_success_rate() {
    let mut stats = UnrollStats::default();
    stats.loops_fully_unrolled = 3;
    stats.loops_partially_unrolled = 2;
    stats.loops_not_unrolled = 5;
    assert!((stats.success_rate() - 0.5).abs() < 0.01);
}

#[test]
fn test_unroll_stats_success_rate_empty() {
    let stats = UnrollStats::default();
    assert_eq!(stats.success_rate(), 0.0);
}

// =========================================================================
// Unroll Pass Tests
// =========================================================================

#[test]
fn test_unroll_new() {
    let unroll = Unroll::new();
    assert_eq!(unroll.config().max_full_unroll_trip, 16);
}

#[test]
fn test_unroll_with_config() {
    let config = UnrollConfig::conservative();
    let unroll = Unroll::with_config(config.clone());
    assert_eq!(
        unroll.config().max_full_unroll_trip,
        config.max_full_unroll_trip
    );
}

#[test]
fn test_unroll_conservative() {
    let unroll = Unroll::conservative();
    assert_eq!(unroll.config().max_full_unroll_trip, 8);
}

#[test]
fn test_unroll_aggressive() {
    let unroll = Unroll::aggressive();
    assert_eq!(unroll.config().max_full_unroll_trip, 32);
}

#[test]
fn test_unroll_name() {
    let unroll = Unroll::new();
    assert_eq!(unroll.name(), "Unroll");
}

#[test]
fn test_unroll_no_loops() {
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

    let mut builder = GraphBuilder::new(4, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut unroll = Unroll::new();
    let changed = unroll.run(&mut graph);

    assert!(!changed);
    assert_eq!(unroll.stats().loops_analyzed, 0);
}

#[test]
fn test_unroll_stats_after_run() {
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

    let mut builder = GraphBuilder::new(4, 0);
    let c = builder.const_int(42);
    builder.return_value(c);

    let mut graph = builder.finish();
    let mut unroll = Unroll::new();
    unroll.run(&mut graph);

    // Stats should be reset each run
    assert_eq!(unroll.stats().loops_analyzed, 0);
}
