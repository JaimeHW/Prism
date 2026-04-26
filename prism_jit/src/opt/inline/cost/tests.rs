use super::*;
use crate::ir::node::NodeId;
use crate::ir::operators::CallKind;

fn make_site(loop_depth: u32, is_hot: bool, arg_count: usize) -> CallSite {
    CallSite {
        call_node: NodeId::new(0),
        call_kind: CallKind::Direct,
        callee: None,
        loop_depth,
        is_hot,
        priority: 0,
        arguments: (0..arg_count).map(|i| NodeId::new(i as u32)).collect(),
        control_input: None,
    }
}

fn make_callee(size: usize, has_loops: bool) -> CalleeInfo {
    CalleeInfo {
        func_id: 1,
        size,
        param_count: 2,
        is_recursive: false,
        has_loops,
        always_inline: false,
        never_inline: false,
        call_count: 0,
        is_intrinsic: false,
    }
}

#[test]
fn test_inline_cost_should_inline() {
    // Beneficial (negative score)
    let cost = InlineCost::new(10, 20, 25, 10);
    assert!(cost.should_inline());
    assert!(cost.net_benefit() > 0);

    // Not beneficial (high positive score)
    let cost = InlineCost::new(200, 10, 25, 0);
    assert!(!cost.should_inline());
}

#[test]
fn test_inline_cost_forced() {
    assert!(InlineCost::FORCED_INLINE.should_inline());
    assert!(InlineCost::FORCED_INLINE.forced);
}

#[test]
fn test_inline_cost_blocked() {
    assert!(!InlineCost::BLOCKED.should_inline());
    assert!(InlineCost::BLOCKED.blocked);
}

#[test]
fn test_cost_model_small_function() {
    let model = InlineCostModel::new();
    let small_callee = make_callee(5, false);
    let site = make_site(0, false, 2);

    let cost = model.compute_cost(&small_callee, &site);

    // Small functions should have low or negative cost
    assert!(cost.size_cost < 50);
}

#[test]
fn test_cost_model_large_function() {
    let model = InlineCostModel::new();
    let large_callee = make_callee(200, false);
    let site = make_site(0, false, 2);

    let cost = model.compute_cost(&large_callee, &site);

    // Large functions should have high cost
    assert!(cost.size_cost > 500);
    assert!(!cost.should_inline());
}

#[test]
fn test_cost_model_loop_benefit() {
    let model = InlineCostModel::new();
    let callee = make_callee(20, false);

    let site_no_loop = make_site(0, false, 2);
    let site_in_loop = make_site(2, false, 2);

    let cost_no_loop = model.compute_cost(&callee, &site_no_loop);
    let cost_in_loop = model.compute_cost(&callee, &site_in_loop);

    // Inlining in loops should be more beneficial
    assert!(cost_in_loop.loop_benefit > cost_no_loop.loop_benefit);
    assert!(cost_in_loop.score < cost_no_loop.score);
}

#[test]
fn test_cost_model_hot_site() {
    let model = InlineCostModel::new();
    let callee = make_callee(30, false);

    let cold_site = make_site(0, false, 2);
    let hot_site = make_site(0, true, 2);

    let cold_cost = model.compute_cost(&callee, &cold_site);
    let hot_cost = model.compute_cost(&callee, &hot_site);

    // Hot sites should have more optimization benefit
    assert!(hot_cost.optimization_benefit > cold_cost.optimization_benefit);
}

#[test]
fn test_cost_model_always_inline() {
    let model = InlineCostModel::new();
    let mut callee = make_callee(500, true);
    callee.always_inline = true;

    let site = make_site(0, false, 2);
    let cost = model.compute_cost(&callee, &site);

    assert!(cost.forced);
    assert!(cost.should_inline());
}

#[test]
fn test_cost_model_never_inline() {
    let model = InlineCostModel::new();
    let mut callee = make_callee(5, false);
    callee.never_inline = true;

    let site = make_site(0, false, 2);
    let cost = model.compute_cost(&callee, &site);

    assert!(cost.blocked);
    assert!(!cost.should_inline());
}

#[test]
fn test_cost_model_intrinsic() {
    let model = InlineCostModel::new();
    let mut callee = make_callee(10, false);
    callee.is_intrinsic = true;

    let site = make_site(0, false, 2);
    let cost = model.compute_cost(&callee, &site);

    // Intrinsics should have high optimization benefit
    assert!(cost.optimization_benefit >= 50);
}

#[test]
fn test_cost_model_loop_penalty() {
    let model = InlineCostModel::new();
    let callee_no_loop = make_callee(20, false);
    let callee_with_loop = make_callee(20, true);

    let site = make_site(0, false, 2);

    let cost_no_loop = model.compute_cost(&callee_no_loop, &site);
    let cost_with_loop = model.compute_cost(&callee_with_loop, &site);

    // Functions with loops should have higher size cost
    assert!(cost_with_loop.size_cost > cost_no_loop.size_cost);
}

#[test]
fn test_cost_model_high_call_count() {
    let model = InlineCostModel::new();
    let mut cold_callee = make_callee(20, false);
    cold_callee.call_count = 10;

    let mut hot_callee = make_callee(20, false);
    hot_callee.call_count = 5000;

    let site = make_site(0, false, 2);

    let cold_cost = model.compute_cost(&cold_callee, &site);
    let hot_cost = model.compute_cost(&hot_callee, &site);

    // High call count should increase benefit
    assert!(hot_cost.optimization_benefit > cold_cost.optimization_benefit);
}
