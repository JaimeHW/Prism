use super::super::induction::{
    InductionDirection, InductionInit, InductionStep, InductionVariable,
};
use super::*;
use crate::ir::cfg::DominatorTree;
use crate::ir::node::InputList;
use crate::ir::operators::ControlOp;

// =========================================================================
// TransformationResult Tests
// =========================================================================

#[test]
fn test_result_new() {
    let result = TransformationResult::new();
    assert_eq!(result.eliminated, 0);
    assert_eq!(result.hoisted, 0);
    assert_eq!(result.widened, 0);
    assert!(result.dead_guards.is_empty());
    assert!(result.new_guards.is_empty());
}

#[test]
fn test_result_has_changes() {
    let mut result = TransformationResult::new();
    assert!(!result.has_changes());

    result.eliminated = 1;
    assert!(result.has_changes());
}

#[test]
fn test_result_has_changes_hoisted() {
    let mut result = TransformationResult::new();
    result.hoisted = 1;
    assert!(result.has_changes());
}

#[test]
fn test_result_has_changes_widened() {
    let mut result = TransformationResult::new();
    result.widened = 1;
    assert!(result.has_changes());
}

#[test]
fn test_result_total() {
    let mut result = TransformationResult::new();
    result.eliminated = 3;
    result.hoisted = 2;
    result.widened = 1;
    assert_eq!(result.total(), 6);
}

#[test]
fn test_result_merge() {
    let mut result1 = TransformationResult::new();
    result1.eliminated = 3;
    result1.hoisted = 2;
    result1.dead_guards.push(NodeId::new(1));

    let mut result2 = TransformationResult::new();
    result2.eliminated = 1;
    result2.widened = 4;
    result2.dead_guards.push(NodeId::new(2));
    result2.new_guards.push(NodeId::new(3));

    result1.merge(&result2);

    assert_eq!(result1.eliminated, 4);
    assert_eq!(result1.hoisted, 2);
    assert_eq!(result1.widened, 4);
    assert_eq!(result1.dead_guards.len(), 2);
    assert_eq!(result1.new_guards.len(), 1);
}

#[test]
fn test_result_merge_empty() {
    let mut result1 = TransformationResult::new();
    result1.eliminated = 5;

    let result2 = TransformationResult::new();
    result1.merge(&result2);

    assert_eq!(result1.eliminated, 5);
    assert_eq!(result1.total(), 5);
}

// =========================================================================
// WidenCalculator Tests
// =========================================================================

#[test]
fn test_lower_bound_widen_increasing_from_zero() {
    let iv = make_canonical_iv();
    let check = make_lower_bound_check();

    let widen = WidenCalculator::compute_lower_bound_widen(&iv, &check);
    assert!(widen.is_some());
    let bounds = widen.unwrap();
    assert_eq!(bounds.min_bound, BoundValue::Constant(0));
}

#[test]
fn test_lower_bound_widen_increasing_from_positive() {
    let iv = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(5),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    );
    let check = make_lower_bound_check();

    let widen = WidenCalculator::compute_lower_bound_widen(&iv, &check);
    assert!(widen.is_some());
    let bounds = widen.unwrap();
    assert_eq!(bounds.min_bound, BoundValue::Constant(5));
}

#[test]
fn test_lower_bound_widen_increasing_from_negative() {
    let iv = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(-5),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    );
    let check = make_lower_bound_check();

    let widen = WidenCalculator::compute_lower_bound_widen(&iv, &check);
    // Cannot widen because min value is negative
    assert!(widen.is_none());
}

#[test]
fn test_lower_bound_widen_decreasing() {
    let iv = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(10),
        InductionStep::Constant(-1),
        InductionDirection::Decreasing,
        None,
    );
    let check = make_lower_bound_check();

    let widen = WidenCalculator::compute_lower_bound_widen(&iv, &check);
    // Cannot widen decreasing IV for lower bound
    assert!(widen.is_none());
}

#[test]
fn test_upper_bound_widen_constant_safe() {
    let iv = make_canonical_iv();
    let check = RangeCheck::new(
        NodeId::new(10),
        NodeId::new(0),
        NodeId::new(11),
        BoundValue::Constant(100),
        RangeCheckKind::UpperBound,
        0,
    );
    let trip_count = TripCount::Constant(50);

    let widen = WidenCalculator::compute_upper_bound_widen(&iv, &check, &trip_count);
    assert!(widen.is_some());
    let bounds = widen.unwrap();
    assert_eq!(bounds.min_bound, BoundValue::Constant(0));
    assert_eq!(bounds.max_bound, BoundValue::Constant(49)); // max = 0 + 1*(50-1) = 49
}

#[test]
fn test_upper_bound_widen_constant_unsafe() {
    let iv = make_canonical_iv();
    let check = RangeCheck::new(
        NodeId::new(10),
        NodeId::new(0),
        NodeId::new(11),
        BoundValue::Constant(10), // Bound is only 10
        RangeCheckKind::UpperBound,
        0,
    );
    let trip_count = TripCount::Constant(50); // Max IV will be 49

    let widen = WidenCalculator::compute_upper_bound_widen(&iv, &check, &trip_count);
    // Cannot widen because max (49) >= bound (10)
    assert!(widen.is_none());
}

#[test]
fn test_upper_bound_widen_inclusive_at_limit() {
    let iv = make_canonical_iv();
    let check = RangeCheck::new(
        NodeId::new(10),
        NodeId::new(0),
        NodeId::new(11),
        BoundValue::Constant(49), // i <= 49
        RangeCheckKind::UpperBoundInclusive,
        0,
    );
    let trip_count = TripCount::Constant(50); // Max IV will be 49

    let widen = WidenCalculator::compute_upper_bound_widen(&iv, &check, &trip_count);
    assert!(widen.is_some());
}

#[test]
fn test_upper_bound_widen_at_most() {
    let iv = make_canonical_iv();
    let check = RangeCheck::new(
        NodeId::new(10),
        NodeId::new(0),
        NodeId::new(11),
        BoundValue::Constant(100),
        RangeCheckKind::UpperBound,
        0,
    );
    let trip_count = TripCount::AtMost(50);

    let widen = WidenCalculator::compute_upper_bound_widen(&iv, &check, &trip_count);
    assert!(widen.is_some());
}

#[test]
fn test_upper_bound_widen_unknown_trip() {
    let iv = make_canonical_iv();
    let check = RangeCheck::new(
        NodeId::new(10),
        NodeId::new(0),
        NodeId::new(11),
        BoundValue::Constant(100),
        RangeCheckKind::UpperBound,
        0,
    );
    let trip_count = TripCount::Unknown;

    let widen = WidenCalculator::compute_upper_bound_widen(&iv, &check, &trip_count);
    assert!(widen.is_none());
}

// =========================================================================
// PreheaderUtils Tests
// =========================================================================

#[test]
fn test_is_valid_preheader_in_body_returns_false() {
    // Test that a block in the loop body is NOT a valid preheader
    let block = BlockId::new(2);
    let loop_info = Loop {
        header: BlockId::new(1),
        back_edges: vec![BlockId::new(3)],
        body: vec![BlockId::new(1), BlockId::new(2), BlockId::new(3)],
        parent: None,
        children: vec![],
        depth: 1,
    };

    // Block 2 is in body, cannot be preheader
    let graph = Graph::new();
    let cfg = Cfg::build(&graph);

    // Should return false because block is in body
    // (The CFG check for successors will also fail but body check comes first)
    assert!(!PreheaderUtils::is_valid_preheader(&cfg, block, &loop_info));
}

#[test]
fn test_is_valid_preheader_is_header_returns_false() {
    // Test that the loop header itself is NOT a valid preheader
    let block = BlockId::new(1);
    let loop_info = Loop {
        header: BlockId::new(1),
        back_edges: vec![BlockId::new(3)],
        body: vec![BlockId::new(1), BlockId::new(2), BlockId::new(3)],
        parent: None,
        children: vec![],
        depth: 1,
    };

    // Header is in body, cannot be preheader
    let graph = Graph::new();
    let cfg = Cfg::build(&graph);
    assert!(!PreheaderUtils::is_valid_preheader(&cfg, block, &loop_info));
}

#[test]
fn test_find_preheader_empty_cfg_returns_none() {
    // Test that find_preheader returns None for an empty CFG
    let graph = Graph::new();
    let cfg = Cfg::build(&graph);
    let loop_info = Loop {
        header: BlockId::new(1),
        back_edges: vec![BlockId::new(3)],
        body: vec![BlockId::new(1), BlockId::new(2), BlockId::new(3)],
        parent: None,
        children: vec![],
        depth: 1,
    };

    // Empty CFG has no blocks to look up, should return None gracefully
    // Note: This may panic on block lookup, but the test shows expected behavior
    // For now we just verify it doesn't crash by using the entry block
    let minimal_loop = Loop {
        header: cfg.entry,
        back_edges: vec![],
        body: vec![cfg.entry],
        parent: None,
        children: vec![],
        depth: 1,
    };

    // Entry block with no predecessors should have no preheader
    assert!(PreheaderUtils::find_preheader(&cfg, &minimal_loop).is_none());
}

#[test]
fn test_create_preheader_returns_none() {
    // Test that create_preheader is not yet implemented
    let mut graph = Graph::new();
    let cfg = Cfg::build(&graph);
    let loop_info = Loop {
        header: BlockId::new(1),
        back_edges: vec![BlockId::new(3)],
        body: vec![BlockId::new(1), BlockId::new(2), BlockId::new(3)],
        parent: None,
        children: vec![],
        depth: 1,
    };

    // Not yet implemented, should return None
    assert!(PreheaderUtils::create_preheader(&mut graph, &cfg, &loop_info).is_none());
}

// =========================================================================
// Helper Functions
// =========================================================================

fn make_canonical_iv() -> InductionVariable {
    InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(0),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    )
}

fn make_lower_bound_check() -> RangeCheck {
    RangeCheck::new(
        NodeId::new(10),
        NodeId::new(0),
        NodeId::new(11),
        BoundValue::Constant(0),
        RangeCheckKind::LowerBound,
        0,
    )
}

// =========================================================================
// RceTransformContext Tests
// =========================================================================

#[test]
fn test_context_creation() {
    // Test that RceTransformContext can be created with valid inputs
    let graph = Graph::new();
    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);
    let iv_analysis = InductionAnalysis::empty();

    let ctx = RceTransformContext::new(&cfg, &loops, &iv_analysis);

    // Verify the context is initialized correctly
    assert!(ctx.get_loop(0).is_none()); // No loops in empty graph
}

#[test]
fn test_context_get_loop_out_of_bounds() {
    // Test that get_loop returns None for out-of-bounds index
    let graph = Graph::new();
    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);
    let iv_analysis = InductionAnalysis::empty();

    let ctx = RceTransformContext::new(&cfg, &loops, &iv_analysis);

    // Should return None for any loop index in empty graph
    assert!(ctx.get_loop(0).is_none());
    assert!(ctx.get_loop(100).is_none());
}

#[test]
fn test_context_preheader_control_missing_loop() {
    // Test that get_preheader_control returns None for non-existent loop
    let graph = Graph::new();
    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);
    let iv_analysis = InductionAnalysis::empty();

    let mut ctx = RceTransformContext::new(&cfg, &loops, &iv_analysis);

    // No loops exist, should return None
    assert!(ctx.get_preheader_control(0, &graph).is_none());
    assert!(ctx.get_preheader_control(5, &graph).is_none());
}

#[test]
fn test_transformation_result_merge() {
    // Test that TransformationResult merge works correctly
    let mut r1 = TransformationResult::new();
    r1.eliminated = 5;
    r1.hoisted = 3;
    r1.widened = 2;
    r1.dead_guards.push(NodeId::new(1));
    r1.new_guards.push(NodeId::new(100));
    r1.insertion_failures = 1;

    let mut r2 = TransformationResult::new();
    r2.eliminated = 2;
    r2.hoisted = 1;
    r2.widened = 1;
    r2.dead_guards.push(NodeId::new(2));
    r2.new_guards.push(NodeId::new(101));
    r2.insertion_failures = 2;

    r1.merge(&r2);

    assert_eq!(r1.eliminated, 7);
    assert_eq!(r1.hoisted, 4);
    assert_eq!(r1.widened, 3);
    assert_eq!(r1.dead_guards.len(), 2);
    assert_eq!(r1.new_guards.len(), 2);
    assert_eq!(r1.insertion_failures, 3);
}

#[test]
fn test_transformation_result_total() {
    // Test that total() returns correct sum
    let mut r = TransformationResult::new();
    r.eliminated = 5;
    r.hoisted = 3;
    r.widened = 2;

    assert_eq!(r.total(), 10);
}

#[test]
fn test_transformation_result_has_changes() {
    // Test has_changes with various combinations
    let empty = TransformationResult::new();
    assert!(!empty.has_changes());

    let mut eliminated = TransformationResult::new();
    eliminated.eliminated = 1;
    assert!(eliminated.has_changes());

    let mut hoisted = TransformationResult::new();
    hoisted.hoisted = 1;
    assert!(hoisted.has_changes());

    let mut widened = TransformationResult::new();
    widened.widened = 1;
    assert!(widened.has_changes());
}

#[test]
fn test_transformer_legacy_apply() {
    // Test that legacy apply() method works (eliminates guards only)
    let mut graph = Graph::new();
    let decisions = EliminationResult::with_capacity(0);

    let mut transformer = RceTransformer::new(&mut graph);
    let result = transformer.apply(&decisions);

    // Empty decisions should produce empty result
    assert!(!result.has_changes());
    assert_eq!(result.eliminated, 0);
    assert_eq!(result.hoisted, 0);
    assert_eq!(result.widened, 0);
}

#[test]
fn test_transformer_apply_with_context_empty() {
    // Test apply_with_context with empty decisions
    let mut graph = Graph::new();
    let cfg = Cfg::build(&graph);
    let dom = DominatorTree::build(&cfg);
    let loops = LoopAnalysis::compute(&cfg, &dom);
    let iv_analysis = InductionAnalysis::empty();

    let mut ctx = RceTransformContext::new(&cfg, &loops, &iv_analysis);
    let decisions = EliminationResult::with_capacity(0);

    let mut transformer = RceTransformer::new(&mut graph);
    let result = transformer.apply_with_context(&decisions, &mut ctx);

    assert!(!result.has_changes());
}

#[test]
fn test_context_block_to_region_uses_region_field() {
    // Test that block_to_region returns the block's region field
    let mut graph = Graph::new();

    // Add a control node to the graph
    let control = graph.add_node(Operator::Control(ControlOp::Start), InputList::Empty);

    let cfg = Cfg::build(&graph);

    // The entry block should have the control node as its region
    if cfg.len() > 0 {
        let block = cfg.block(cfg.entry);
        // The region field should be the control node
        assert_eq!(block.region, control);
    }
}

#[test]
fn test_transformer_result_getter() {
    // Test that result() returns correct reference
    let mut graph = Graph::new();
    let mut transformer = RceTransformer::new(&mut graph);

    let result = transformer.result();
    assert_eq!(result.eliminated, 0);
    assert!(!result.has_changes());
}
