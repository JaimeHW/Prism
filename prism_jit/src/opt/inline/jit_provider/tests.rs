use super::*;
use crate::ir::graph::Graph;
use std::thread;

fn make_test_graph() -> CalleeGraph {
    let graph = Graph::new();
    CalleeGraph::new(graph, 2)
}

// =========================================================================
// CompilationTier Tests
// =========================================================================

#[test]
fn test_tier_ordering() {
    assert!(CompilationTier::Interpreted < CompilationTier::Tier1);
    assert!(CompilationTier::Tier1 < CompilationTier::Tier2);
}

#[test]
fn test_tier_inline_hint() {
    assert_eq!(CompilationTier::Interpreted.inline_hint(), InlineHint::Cold);
    assert_eq!(CompilationTier::Tier1.inline_hint(), InlineHint::Default);
    assert_eq!(CompilationTier::Tier2.inline_hint(), InlineHint::Hot);
}

#[test]
fn test_tier_is_optimized() {
    assert!(!CompilationTier::Interpreted.is_optimized());
    assert!(!CompilationTier::Tier1.is_optimized());
    assert!(CompilationTier::Tier2.is_optimized());
}

#[test]
fn test_tier_priority() {
    assert_eq!(CompilationTier::Interpreted.priority_multiplier(), 0.0);
    assert_eq!(CompilationTier::Tier1.priority_multiplier(), 0.5);
    assert_eq!(CompilationTier::Tier2.priority_multiplier(), 1.0);
}

// =========================================================================
// Provider Basic Tests
// =========================================================================

#[test]
fn test_provider_new() {
    let provider = JitCalleeProvider::new();
    assert_eq!(provider.cached_count(), 0);
    assert_eq!(provider.function_count(), 0);
}

#[test]
fn test_register_compiled() {
    let provider = JitCalleeProvider::new();
    let graph = make_test_graph();

    let version = provider.register_compiled(1, graph, CompilationTier::Tier1);

    assert_eq!(version, 0);
    assert_eq!(provider.cached_count(), 1);
    assert!(provider.has_function(1));
}

#[test]
fn test_get_cached() {
    let provider = JitCalleeProvider::new();
    let graph = make_test_graph();

    provider.register_compiled(1, graph, CompilationTier::Tier2);

    let cached = provider.get_cached(1);
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().param_count, 2);
}

#[test]
fn test_get_tier() {
    let provider = JitCalleeProvider::new();
    let graph = make_test_graph();

    provider.register_compiled(1, graph, CompilationTier::Tier2);

    assert_eq!(provider.get_tier(1), Some(CompilationTier::Tier2));
    assert_eq!(provider.get_tier(999), None);
}

#[test]
fn test_get_state() {
    let provider = JitCalleeProvider::new();

    assert_eq!(provider.get_state(1), CompilationState::NotCompiled);

    provider.mark_compiling(1, CompilationTier::Tier1);
    assert_eq!(
        provider.get_state(1),
        CompilationState::Compiling {
            tier: CompilationTier::Tier1
        }
    );

    let graph = make_test_graph();
    provider.register_compiled(1, graph, CompilationTier::Tier1);

    match provider.get_state(1) {
        CompilationState::Compiled { tier, .. } => {
            assert_eq!(tier, CompilationTier::Tier1);
        }
        _ => panic!("Expected Compiled state"),
    }
}

// =========================================================================
// Tier Upgrade Tests
// =========================================================================

#[test]
fn test_upgrade_tier() {
    let provider = JitCalleeProvider::new();

    let graph1 = make_test_graph();
    provider.register_compiled(1, graph1, CompilationTier::Tier1);

    let graph2 = make_test_graph();
    let new_version = provider.upgrade_tier(1, graph2, CompilationTier::Tier2);

    assert!(new_version > 0);
    assert_eq!(provider.get_tier(1), Some(CompilationTier::Tier2));
}

#[test]
fn test_upgrade_invalidates_old_version() {
    let provider = JitCalleeProvider::new();

    let graph1 = make_test_graph();
    let v1 = provider.register_compiled(1, graph1, CompilationTier::Tier1);

    let graph2 = make_test_graph();
    let v2 = provider.upgrade_tier(1, graph2, CompilationTier::Tier2);

    assert!(provider.is_stale(1, v1));
    assert!(!provider.is_stale(1, v2));
}

// =========================================================================
// Call Recording Tests
// =========================================================================

#[test]
fn test_record_call() {
    let provider = JitCalleeProvider::new();

    provider.record_call(1);
    provider.record_call(1);
    provider.record_call(1);

    assert!(provider.hotness.get_call_count(1) >= 3);
}

#[test]
fn test_record_calls_batch() {
    let provider = JitCalleeProvider::new();

    provider.record_calls(1, 100);

    assert!(provider.hotness.get_call_count(1) >= 100);
}

#[test]
fn test_get_hotness() {
    let config = JitProviderConfig {
        hotness_config: HotnessConfig {
            cold_threshold: 5,
            hot_threshold: 50,
            very_hot_threshold: 500,
            ..Default::default()
        },
        ..Default::default()
    };
    let provider = JitCalleeProvider::with_config(config);

    provider.record_calls(1, 3);
    assert_eq!(provider.get_hotness(1), HotnessLevel::Cold);

    provider.record_calls(2, 100);
    assert_eq!(provider.get_hotness(2), HotnessLevel::Hot);
}

#[test]
fn test_get_hot_functions() {
    let config = JitProviderConfig {
        hotness_config: HotnessConfig {
            hot_threshold: 10,
            ..Default::default()
        },
        ..Default::default()
    };
    let provider = JitCalleeProvider::with_config(config);

    provider.record_calls(1, 5); // Cold
    provider.record_calls(2, 50); // Hot
    provider.record_calls(3, 100); // Hot

    let hot = provider.get_hot_functions();
    assert_eq!(hot.len(), 2);
}

// =========================================================================
// Inline Priority Tests
// =========================================================================

#[test]
fn test_inline_priority_combines_factors() {
    let provider = JitCalleeProvider::new();

    // No calls, no graph -> 0
    assert_eq!(provider.get_inline_priority(1), 0.0);

    // Calls but no graph
    provider.record_calls(2, 100);
    let p2 = provider.get_inline_priority(2);
    assert!(p2 > 0.0);

    // Graph and calls
    let graph = make_test_graph();
    provider.register_compiled(3, graph, CompilationTier::Tier2);
    provider.record_calls(3, 100);
    let p3 = provider.get_inline_priority(3);
    assert!(p3 > p2); // Tier adds priority
}

// =========================================================================
// Deoptimization Tests
// =========================================================================

#[test]
fn test_handle_deoptimization() {
    let provider = JitCalleeProvider::new();

    let graph = make_test_graph();
    provider.register_compiled(1, graph, CompilationTier::Tier2);

    provider.handle_deoptimization(1, DeoptReason::TypeCheckFailed);

    assert!(provider.get_cached(1).is_none());
    assert!(!provider.has_function(1));

    match provider.get_state(1) {
        CompilationState::Deoptimized { reason, .. } => {
            assert_eq!(reason, DeoptReason::TypeCheckFailed);
        }
        _ => panic!("Expected Deoptimized state"),
    }
}

#[test]
fn test_deoptimization_cascades() {
    let provider = JitCalleeProvider::new();

    let graph1 = make_test_graph();
    let v1 = provider.register_compiled(1, graph1, CompilationTier::Tier2);

    let graph2 = make_test_graph();
    let v2 = provider.register_compiled(2, graph2, CompilationTier::Tier2);

    // 1 inlined 2
    provider.record_inline(1, 2);

    // Deopt 2
    provider.handle_deoptimization(2, DeoptReason::TypeCheckFailed);

    // 1 should be stale
    assert!(provider.is_stale(1, v1));
}

// =========================================================================
// Inlining Dependency Tests
// =========================================================================

#[test]
fn test_record_inline() {
    let provider = JitCalleeProvider::new();

    provider.record_inline(1, 2);

    let dependents = provider.invalidation.get_dependents(2);
    assert!(dependents.contains(&1));
}

#[test]
fn test_clear_inline_dependencies() {
    let provider = JitCalleeProvider::new();

    provider.record_inline(1, 10);
    provider.record_inline(1, 11);

    provider.clear_inline_dependencies(1);

    assert!(provider.invalidation.get_callees(1).is_empty());
}

// =========================================================================
// Eviction Tests
// =========================================================================

#[test]
fn test_evict_cold() {
    let config = JitProviderConfig {
        hotness_config: HotnessConfig {
            cold_threshold: 10,
            ..Default::default()
        },
        ..Default::default()
    };
    let provider = JitCalleeProvider::with_config(config);

    let graph1 = make_test_graph();
    provider.register_compiled(1, graph1, CompilationTier::Tier1);

    let graph2 = make_test_graph();
    provider.register_compiled(2, graph2, CompilationTier::Tier1);
    provider.record_calls(2, 100); // Make hot

    provider.evict_cold();

    assert!(!provider.has_function(1)); // Cold - evicted
    assert!(provider.has_function(2)); // Hot - kept
}

// =========================================================================
// CalleeProvider Trait Tests
// =========================================================================

#[test]
fn test_callee_provider_get_graph() {
    let provider = JitCalleeProvider::new();
    let graph = make_test_graph();
    provider.register_compiled(1, graph, CompilationTier::Tier2);

    let result = <JitCalleeProvider as CalleeProvider>::get_graph(&provider, 1);
    assert!(result.is_some());
}

#[test]
fn test_callee_provider_has_function() {
    let provider = JitCalleeProvider::new();
    let graph = make_test_graph();
    provider.register_compiled(1, graph, CompilationTier::Tier2);

    assert!(<JitCalleeProvider as CalleeProvider>::has_function(
        &provider, 1
    ));
    assert!(!<JitCalleeProvider as CalleeProvider>::has_function(
        &provider, 2
    ));
}

#[test]
fn test_callee_provider_param_count() {
    let provider = JitCalleeProvider::new();
    let graph = make_test_graph();
    provider.register_compiled(1, graph, CompilationTier::Tier2);

    assert_eq!(
        <JitCalleeProvider as CalleeProvider>::param_count(&provider, 1),
        Some(2)
    );
}

#[test]
fn test_callee_provider_inline_hint() {
    let config = JitProviderConfig {
        hotness_config: HotnessConfig {
            very_hot_threshold: 100,
            ..Default::default()
        },
        ..Default::default()
    };
    let provider = JitCalleeProvider::with_config(config);

    let graph = make_test_graph();
    provider.register_compiled(1, graph, CompilationTier::Tier2);
    provider.record_calls(1, 500);

    let hint = <JitCalleeProvider as CalleeProvider>::inline_hint(&provider, 1);
    assert_eq!(hint, InlineHint::Always);
}

// =========================================================================
// Summary Tests
// =========================================================================

#[test]
fn test_summary() {
    let provider = JitCalleeProvider::new();

    let graph1 = make_test_graph();
    provider.register_compiled(1, graph1, CompilationTier::Tier1);

    let graph2 = make_test_graph();
    provider.register_compiled(2, graph2, CompilationTier::Tier2);

    provider.record_calls(1, 100);
    provider.record_calls(2, 100);

    let summary = provider.summary();
    assert_eq!(summary.cached_graphs, 2);
    assert_eq!(summary.tier1_count, 1);
    assert_eq!(summary.tier2_count, 1);
}

// =========================================================================
// Cleanup Tests
// =========================================================================

#[test]
fn test_remove() {
    let provider = JitCalleeProvider::new();

    let graph = make_test_graph();
    provider.register_compiled(1, graph, CompilationTier::Tier2);
    provider.record_calls(1, 100);

    provider.remove(1);

    assert!(!provider.has_function(1));
    assert_eq!(provider.hotness.get_call_count(1), 0);
}

#[test]
fn test_clear() {
    let provider = JitCalleeProvider::new();

    let graph = make_test_graph();
    provider.register_compiled(1, graph, CompilationTier::Tier2);

    provider.clear();

    assert_eq!(provider.cached_count(), 0);
    assert_eq!(provider.function_count(), 0);
}

// =========================================================================
// Thread Safety Tests
// =========================================================================

#[test]
fn test_concurrent_registration() {
    let provider = Arc::new(JitCalleeProvider::new());
    let mut handles = vec![];

    for i in 0..8 {
        let p = provider.clone();
        handles.push(thread::spawn(move || {
            for j in 0..50 {
                let graph = make_test_graph();
                p.register_compiled(i * 100 + j, graph, CompilationTier::Tier1);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(provider.cached_count(), 400);
}

#[test]
fn test_concurrent_calls_and_invalidation() {
    let provider = Arc::new(JitCalleeProvider::new());

    // Pre-register some functions
    for i in 0..10 {
        let graph = make_test_graph();
        provider.register_compiled(i, graph, CompilationTier::Tier2);
    }

    let mut handles = vec![];

    // Record calls concurrently
    let p1 = provider.clone();
    handles.push(thread::spawn(move || {
        for _ in 0..1000 {
            for i in 0..10 {
                p1.record_call(i);
            }
        }
    }));

    // Upgrade tiers concurrently
    let p2 = provider.clone();
    handles.push(thread::spawn(move || {
        for i in 0..5 {
            let graph = make_test_graph();
            p2.upgrade_tier(i, graph, CompilationTier::Tier2);
        }
    }));

    // Deopt concurrently
    let p3 = provider.clone();
    handles.push(thread::spawn(move || {
        for i in 5..10 {
            p3.handle_deoptimization(i, DeoptReason::TypeCheckFailed);
        }
    }));

    for h in handles {
        h.join().unwrap();
    }

    // Should complete without panicking
    let _ = provider.summary();
}

// =========================================================================
// Configuration Tests
// =========================================================================

#[test]
fn test_tier1_config() {
    let config = JitProviderConfig::tier1();
    assert!(config.max_inline_depth < JitProviderConfig::default().max_inline_depth);
}

#[test]
fn test_tier2_config() {
    let config = JitProviderConfig::tier2();
    assert_eq!(
        config.max_inline_depth,
        JitProviderConfig::default().max_inline_depth
    );
}

#[test]
fn test_aggressive_config() {
    let config = JitProviderConfig::aggressive();
    assert!(config.max_inline_depth > JitProviderConfig::default().max_inline_depth);
}

#[test]
fn test_conservative_config() {
    let config = JitProviderConfig::conservative();
    assert!(config.max_cached_graphs < JitProviderConfig::default().max_cached_graphs);
}
