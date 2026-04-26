use super::*;

// =========================================================================
// Helper Functions
// =========================================================================

fn make_iv(init: i64, step: i64, direction: InductionDirection) -> InductionVariable {
    InductionVariable {
        phi: NodeId::new(0),
        init: InductionInit::Constant(init),
        step: InductionStep::Constant(step),
        direction,
        update_node: None,
    }
}

fn make_lower_check(iv_node: NodeId) -> RangeCheck {
    RangeCheck::new(
        NodeId::new(10),
        iv_node,
        NodeId::new(11),
        BoundValue::Constant(0),
        RangeCheckKind::LowerBound,
        0,
    )
}

fn make_upper_check(iv_node: NodeId, bound: i64) -> RangeCheck {
    RangeCheck::new(
        NodeId::new(10),
        iv_node,
        NodeId::new(11),
        BoundValue::Constant(bound),
        RangeCheckKind::UpperBound,
        0,
    )
}

// =========================================================================
// EliminationAnalyzer Tests
// =========================================================================

#[test]
fn test_analyzer_new() {
    let analyzer = EliminationAnalyzer::new();
    assert!(!analyzer.aggressive);
    assert_eq!(analyzer.stats().analyzed, 0);
}

#[test]
fn test_analyzer_aggressive() {
    let analyzer = EliminationAnalyzer::aggressive();
    assert!(analyzer.aggressive);
}

#[test]
fn test_lower_bound_eliminable_positive_init_increasing() {
    let mut analyzer = EliminationAnalyzer::new();
    let iv = make_iv(0, 1, InductionDirection::Increasing);
    let check = make_lower_check(NodeId::new(0));

    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Eliminate);
    assert_eq!(analyzer.stats().eliminable, 1);
}

#[test]
fn test_lower_bound_eliminable_positive_init_positive_step() {
    let mut analyzer = EliminationAnalyzer::new();
    let iv = make_iv(5, 2, InductionDirection::Increasing);
    let check = make_lower_check(NodeId::new(0));

    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Eliminate);
}

#[test]
fn test_lower_bound_kept_negative_init() {
    let mut analyzer = EliminationAnalyzer::new();
    let iv = make_iv(-5, 1, InductionDirection::Increasing);
    let check = make_lower_check(NodeId::new(0));

    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Keep);
    assert_eq!(analyzer.stats().kept, 1);
}

#[test]
fn test_lower_bound_kept_decreasing() {
    let mut analyzer = EliminationAnalyzer::new();
    let iv = make_iv(10, -1, InductionDirection::Decreasing);
    let check = make_lower_check(NodeId::new(0));

    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Keep);
}

#[test]
fn test_upper_bound_eliminable_decreasing_below() {
    let mut analyzer = EliminationAnalyzer::new();
    let iv = make_iv(50, -1, InductionDirection::Decreasing);
    let check = make_upper_check(NodeId::new(0), 100);

    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Eliminate);
}

#[test]
fn test_upper_bound_kept_increasing_no_bound_info() {
    let mut analyzer = EliminationAnalyzer::new();
    let iv = make_iv(0, 1, InductionDirection::Increasing);
    let check = make_upper_check(NodeId::new(0), 100);

    let decision = analyzer.analyze(&check, &iv);
    // Without trip count info, we can't eliminate
    assert_eq!(decision, EliminationDecision::Keep);
}

#[test]
fn test_aggressive_hoists_lower_bound() {
    let mut analyzer = EliminationAnalyzer::aggressive();
    let iv = make_iv(-5, 1, InductionDirection::Increasing);
    let check = make_lower_check(NodeId::new(0));

    let decision = analyzer.analyze(&check, &iv);
    assert!(matches!(decision, EliminationDecision::Hoist(_)));
    assert_eq!(analyzer.stats().hoistable, 1);
}

#[test]
fn test_aggressive_widens_upper_bound() {
    let mut analyzer = EliminationAnalyzer::aggressive();
    let iv = make_iv(0, 1, InductionDirection::Increasing);
    let check = make_upper_check(NodeId::new(0), 100);

    let decision = analyzer.analyze(&check, &iv);
    assert!(matches!(decision, EliminationDecision::Widen(_)));
    assert_eq!(analyzer.stats().widenable, 1);
}

#[test]
fn test_stats_reset() {
    let mut analyzer = EliminationAnalyzer::new();
    let iv = make_iv(0, 1, InductionDirection::Increasing);
    let check = make_lower_check(NodeId::new(0));

    analyzer.analyze(&check, &iv);
    assert_eq!(analyzer.stats().analyzed, 1);

    analyzer.reset_stats();
    assert_eq!(analyzer.stats().analyzed, 0);
}

// =========================================================================
// EliminationResult Tests
// =========================================================================

#[test]
fn test_result_new() {
    let result = EliminationResult::new();
    assert!(result.is_empty());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_result_add() {
    let mut result = EliminationResult::new();
    let check = make_lower_check(NodeId::new(0));
    result.add(check, EliminationDecision::Eliminate);

    assert!(!result.is_empty());
    assert_eq!(result.len(), 1);
    assert_eq!(result.count_eliminable(), 1);
}

#[test]
fn test_result_eliminable_iter() {
    let mut result = EliminationResult::new();
    result.add(
        make_lower_check(NodeId::new(0)),
        EliminationDecision::Eliminate,
    );
    result.add(
        make_upper_check(NodeId::new(0), 100),
        EliminationDecision::Keep,
    );
    result.add(
        make_lower_check(NodeId::new(1)),
        EliminationDecision::Eliminate,
    );

    let eliminable: Vec<_> = result.eliminable().collect();
    assert_eq!(eliminable.len(), 2);
}

#[test]
fn test_result_hoistable_iter() {
    let mut result = EliminationResult::new();
    result.add(
        make_lower_check(NodeId::new(0)),
        EliminationDecision::Hoist(HoistInfo {
            guard: NodeId::new(10),
            iv: NodeId::new(0),
        }),
    );
    result.add(
        make_upper_check(NodeId::new(0), 100),
        EliminationDecision::Keep,
    );

    let hoistable: Vec<_> = result.hoistable().collect();
    assert_eq!(hoistable.len(), 1);
}

#[test]
fn test_result_widenable_iter() {
    let mut result = EliminationResult::new();
    result.add(
        make_upper_check(NodeId::new(0), 100),
        EliminationDecision::Widen(WidenInfo {
            guard: NodeId::new(10),
            iv: NodeId::new(0),
            min_check: WidenBound::Constant(0),
            max_check: WidenBound::Constant(99),
        }),
    );

    let widenable: Vec<_> = result.widenable().collect();
    assert_eq!(widenable.len(), 1);
}

// =========================================================================
// WidenBound Tests
// =========================================================================

#[test]
fn test_widen_bound_constant() {
    let bound = WidenBound::Constant(100);
    assert!(matches!(bound, WidenBound::Constant(100)));
}

#[test]
fn test_widen_bound_node() {
    let bound = WidenBound::Node(NodeId::new(5));
    assert!(matches!(bound, WidenBound::Node(_)));
}

#[test]
fn test_widen_bound_computed() {
    let bound = WidenBound::Computed {
        base: NodeId::new(0),
        offset: 10,
    };
    assert!(matches!(bound, WidenBound::Computed { .. }));
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_zero_step_iv() {
    let mut analyzer = EliminationAnalyzer::new();
    let iv = make_iv(0, 0, InductionDirection::Unknown);
    let check = make_lower_check(NodeId::new(0));

    // Zero step with init >= 0 should still be eliminable for lower bound
    // Actually no, because direction is Unknown
    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Keep);
}

#[test]
fn test_negative_step_positive_init_lower_bound() {
    let mut analyzer = EliminationAnalyzer::new();
    // Init = 100, step = -1 (decreasing)
    // Lower bound check NOT safe - will eventually go negative
    let iv = make_iv(100, -1, InductionDirection::Decreasing);
    let check = make_lower_check(NodeId::new(0));

    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Keep);
}

#[test]
fn test_upper_bound_at_exact_limit() {
    let mut analyzer = EliminationAnalyzer::new();
    // Init = 99, decreasing, bound = 100
    // 99 < 100 is true, and we're decreasing, so safe
    let iv = make_iv(99, -1, InductionDirection::Decreasing);
    let check = make_upper_check(NodeId::new(0), 100);

    let decision = analyzer.analyze(&check, &iv);
    assert_eq!(decision, EliminationDecision::Eliminate);
}

#[test]
fn test_upper_bound_starting_at_limit() {
    let mut analyzer = EliminationAnalyzer::new();
    // Init = 100, decreasing, bound = 100
    // 100 < 100 is FALSE, so not safe
    let iv = make_iv(100, -1, InductionDirection::Decreasing);
    let check = make_upper_check(NodeId::new(0), 100);

    let decision = analyzer.analyze(&check, &iv);
    // init < bound check fails (100 < 100 is false)
    assert_eq!(decision, EliminationDecision::Keep);
}
