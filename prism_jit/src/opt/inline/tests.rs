use super::*;
use crate::ir::builder::{ControlBuilder, GraphBuilder};

#[test]
fn test_inline_config_default() {
    let config = InlineConfig::default();
    assert_eq!(config.max_callee_size, 100);
    assert_eq!(config.max_depth, 4);
    assert!(config.enable_speculative);
}

#[test]
fn test_inline_config_conservative() {
    let config = InlineConfig::conservative();
    assert!(config.max_callee_size < InlineConfig::default().max_callee_size);
    assert!(config.max_depth < InlineConfig::default().max_depth);
    assert!(!config.enable_speculative);
}

#[test]
fn test_inline_config_aggressive() {
    let config = InlineConfig::aggressive();
    assert!(config.max_callee_size > InlineConfig::default().max_callee_size);
    assert!(config.max_depth > InlineConfig::default().max_depth);
}

#[test]
fn test_inline_config_tier1() {
    let config = InlineConfig::tier1();
    assert_eq!(config.max_depth, 1);
    assert!(!config.enable_speculative);
}

#[test]
fn test_inline_new() {
    let inline = Inline::new();
    assert_eq!(inline.inlined(), 0);
    assert_eq!(inline.growth(), 0);
}

#[test]
fn test_inline_name() {
    let inline = Inline::new();
    assert_eq!(inline.name(), "inline");
}

#[test]
fn test_inline_no_calls() {
    let mut builder = GraphBuilder::new(2, 1);
    let p0 = builder.parameter(0).unwrap();
    builder.return_value(p0);

    let mut graph = builder.finish();
    let mut inline = Inline::new();

    let changed = inline.run(&mut graph);
    assert!(!changed);
    assert_eq!(inline.inlined(), 0);
}

#[test]
fn test_callee_info_default() {
    let info = CalleeInfo::default();
    assert_eq!(info.size, 0);
    assert!(!info.is_recursive);
    assert!(!info.always_inline);
    assert!(!info.has_loops);
}

#[test]
fn test_inline_stats_default() {
    let stats = InlineStats::default();
    assert_eq!(stats.sites_examined, 0);
    assert_eq!(stats.sites_inlined, 0);
    assert_eq!(stats.nodes_added, 0);
}

#[test]
fn test_priority_calculation() {
    let inline = Inline::new();

    // Base priority
    let priority = inline.compute_priority(CallKind::Direct, 0, false, &None);
    assert_eq!(priority, 0);

    // Hot call bonus
    let priority_hot = inline.compute_priority(CallKind::Direct, 0, true, &None);
    assert_eq!(priority_hot, inline.config.hot_call_bonus);

    // Loop depth bonus
    let priority_loop = inline.compute_priority(CallKind::Direct, 2, false, &None);
    assert_eq!(priority_loop, 40); // 2 * 20

    // Small callee bonus
    let small_callee = CalleeInfo {
        size: 5,
        ..Default::default()
    };
    let priority_small = inline.compute_priority(CallKind::Direct, 0, false, &Some(small_callee));
    assert!(priority_small > 0);
}

#[test]
fn test_should_inline_never_inline() {
    let mut inline = Inline::new();
    let callee = CalleeInfo {
        never_inline: true,
        size: 10,
        ..Default::default()
    };
    let site = CallSite {
        call_node: NodeId::new(0),
        call_kind: CallKind::Direct,
        callee: Some(callee),
        loop_depth: 0,
        is_hot: false,
        priority: 100,
        arguments: vec![],
        control_input: None,
    };

    assert_eq!(inline.should_inline(&site), Some(false));
}

#[test]
fn test_should_inline_always_inline() {
    let mut inline = Inline::new();
    let callee = CalleeInfo {
        always_inline: true,
        size: 1000, // Would normally be rejected
        ..Default::default()
    };
    let site = CallSite {
        call_node: NodeId::new(0),
        call_kind: CallKind::Direct,
        callee: Some(callee),
        loop_depth: 0,
        is_hot: false,
        priority: 100,
        arguments: vec![],
        control_input: None,
    };

    assert_eq!(inline.should_inline(&site), Some(true));
}

#[test]
fn test_should_inline_too_large() {
    let mut inline = Inline::new();
    let callee = CalleeInfo {
        size: 1000, // Exceeds max_callee_size
        ..Default::default()
    };
    let site = CallSite {
        call_node: NodeId::new(0),
        call_kind: CallKind::Direct,
        callee: Some(callee),
        loop_depth: 0,
        is_hot: false,
        priority: 100,
        arguments: vec![],
        control_input: None,
    };

    assert_eq!(inline.should_inline(&site), Some(false));
    assert_eq!(inline.stats.rejected_size, 1);
}
