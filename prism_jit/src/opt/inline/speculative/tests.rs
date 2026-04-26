use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};
use crate::ir::operators::CallKind;

// -------------------------------------------------------------------------
// TypeId Tests
// -------------------------------------------------------------------------

#[test]
fn test_type_id_primitives() {
    assert!(TypeId::INT.is_primitive());
    assert!(TypeId::FLOAT.is_primitive());
    assert!(TypeId::STRING.is_primitive());
    assert!(TypeId::LIST.is_primitive());
    assert!(TypeId::DICT.is_primitive());
    assert!(TypeId::NONE.is_primitive());
    assert!(TypeId::BOOL.is_primitive());
    assert!(TypeId::TUPLE.is_primitive());

    assert!(!TypeId::UNKNOWN.is_primitive());
}

#[test]
fn test_type_id_user_types() {
    let user_type = TypeId::new(TypeId::USER_TYPE_START);
    assert!(user_type.is_user_type());
    assert!(!user_type.is_primitive());

    let user_type2 = TypeId::new(TypeId::USER_TYPE_START + 100);
    assert!(user_type2.is_user_type());
}

#[test]
fn test_type_id_from_u64() {
    let type_id: TypeId = 42u64.into();
    assert_eq!(type_id.raw(), 42);
}

// -------------------------------------------------------------------------
// TypeGuardInfo Tests
// -------------------------------------------------------------------------

#[test]
fn test_type_guard_info_new() {
    let guard = TypeGuardInfo::new(TypeId::INT);
    assert_eq!(guard.expected_type, TypeId::INT);
    assert!(guard.guard_node.is_none());
    assert_eq!(guard.hit_count, 0);
    assert_eq!(guard.total_calls, 0);
}

#[test]
fn test_type_guard_info_with_profile() {
    let guard = TypeGuardInfo::with_profile(TypeId::STRING, 950, 1000);
    assert_eq!(guard.expected_type, TypeId::STRING);
    assert_eq!(guard.hit_count, 950);
    assert_eq!(guard.total_calls, 1000);
}

#[test]
fn test_type_guard_hit_rate() {
    let guard = TypeGuardInfo::with_profile(TypeId::INT, 750, 1000);
    assert!((guard.hit_rate() - 0.75).abs() < 0.001);

    let empty_guard = TypeGuardInfo::new(TypeId::INT);
    assert_eq!(empty_guard.hit_rate(), 0.0);
}

#[test]
fn test_type_guard_is_monomorphic() {
    let mono = TypeGuardInfo::with_profile(TypeId::INT, 960, 1000);
    assert!(mono.is_monomorphic());

    let poly = TypeGuardInfo::with_profile(TypeId::INT, 800, 1000);
    assert!(!poly.is_monomorphic());
}

#[test]
fn test_type_guard_is_worth_inlining() {
    let worthy = TypeGuardInfo::with_profile(TypeId::INT, 100, 1000); // 10%
    assert!(worthy.is_worth_inlining());

    let unworthy = TypeGuardInfo::with_profile(TypeId::INT, 40, 1000); // 4%
    assert!(!unworthy.is_worth_inlining());
}

// -------------------------------------------------------------------------
// SpeculativeTarget Tests
// -------------------------------------------------------------------------

#[test]
fn test_speculative_target_new() {
    let callee = CalleeInfo::default();
    let target = SpeculativeTarget::new(TypeId::INT, 42, callee);

    assert_eq!(target.receiver_type, TypeId::INT);
    assert_eq!(target.target_func, 42);
    assert!(!target.is_fallback);
}

#[test]
fn test_speculative_target_fallback() {
    let target = SpeculativeTarget::fallback(99);

    assert_eq!(target.receiver_type, TypeId::UNKNOWN);
    assert_eq!(target.target_func, 99);
    assert!(target.is_fallback);
}

#[test]
fn test_speculative_target_with_count() {
    let callee = CalleeInfo::default();
    let target = SpeculativeTarget::new(TypeId::INT, 42, callee).with_count(1000);

    assert_eq!(target.invocation_count, 1000);
}

// -------------------------------------------------------------------------
// TypeProfile Tests
// -------------------------------------------------------------------------

#[test]
fn test_type_profile_new() {
    let profile = TypeProfile::new(100);
    assert_eq!(profile.bc_offset, 100);
    assert_eq!(profile.total_calls(), 0);
    assert_eq!(profile.type_diversity(), 0);
}

#[test]
fn test_type_profile_record() {
    let profile = TypeProfile::new(0);

    profile.record(TypeId::INT);
    profile.record(TypeId::INT);
    profile.record(TypeId::FLOAT);

    assert_eq!(profile.total_calls(), 3);
    assert_eq!(profile.type_count(TypeId::INT), 2);
    assert_eq!(profile.type_count(TypeId::FLOAT), 1);
    assert_eq!(profile.type_diversity(), 2);
}

#[test]
fn test_type_profile_monomorphic() {
    let profile = TypeProfile::new(0);

    // Record enough calls to pass minimum threshold
    for _ in 0..MIN_CALL_COUNT_FOR_SPECULATION + 10 {
        profile.record(TypeId::INT);
    }

    assert!(profile.is_monomorphic());
    assert!(!profile.is_polymorphic());
    assert!(!profile.is_megamorphic());
}

#[test]
fn test_type_profile_polymorphic() {
    let profile = TypeProfile::new(0);

    // Record 3 different types
    for _ in 0..50 {
        profile.record(TypeId::INT);
        profile.record(TypeId::FLOAT);
        profile.record(TypeId::STRING);
    }

    assert!(!profile.is_monomorphic());
    assert!(profile.is_polymorphic());
    assert!(!profile.is_megamorphic());
}

#[test]
fn test_type_profile_megamorphic() {
    let profile = TypeProfile::new(0);

    // Record more than MAX_POLYMORPHIC_TYPES types
    for i in 0..10 {
        profile.record(TypeId::new(i));
    }

    assert!(profile.is_megamorphic());
}

#[test]
fn test_type_profile_dominant_types() {
    let profile = TypeProfile::new(0);

    // Record with different frequencies
    for _ in 0..100 {
        profile.record(TypeId::INT);
    }
    for _ in 0..50 {
        profile.record(TypeId::FLOAT);
    }
    for _ in 0..25 {
        profile.record(TypeId::STRING);
    }

    let dominant = profile.get_dominant_types(2);
    assert_eq!(dominant.len(), 2);
    assert_eq!(dominant[0].0, TypeId::INT);
    assert_eq!(dominant[0].1, 100);
    assert_eq!(dominant[1].0, TypeId::FLOAT);
    assert_eq!(dominant[1].1, 50);
}

#[test]
fn test_type_profile_get_guard_info() {
    let profile = TypeProfile::new(0);

    for _ in 0..100 {
        profile.record(TypeId::INT);
    }
    for _ in 0..50 {
        profile.record(TypeId::FLOAT);
    }
    // This one is below MIN_TYPE_PERCENTAGE
    for _ in 0..5 {
        profile.record(TypeId::STRING);
    }

    let guards = profile.get_guard_info();

    // STRING should be filtered out (5/155 < 5%)
    assert_eq!(guards.len(), 2);
    assert!(guards.iter().any(|g| g.expected_type == TypeId::INT));
    assert!(guards.iter().any(|g| g.expected_type == TypeId::FLOAT));
}

#[test]
fn test_type_profile_clone() {
    let profile = TypeProfile::new(42);
    profile.record(TypeId::INT);
    profile.record(TypeId::INT);
    profile.record(TypeId::FLOAT);

    let cloned = profile.clone();

    assert_eq!(cloned.bc_offset, 42);
    assert_eq!(cloned.total_calls(), 3);
    assert_eq!(cloned.type_count(TypeId::INT), 2);
    assert_eq!(cloned.type_count(TypeId::FLOAT), 1);
}

// -------------------------------------------------------------------------
// TypeProfileRegistry Tests
// -------------------------------------------------------------------------

#[test]
fn test_profile_registry_new() {
    let registry = TypeProfileRegistry::new();
    assert!(registry.get(0, 0).is_none());
}

#[test]
fn test_profile_registry_get_or_create() {
    let registry = TypeProfileRegistry::new();

    let profile1 = registry.get_or_create(1, 100);
    let profile2 = registry.get_or_create(1, 100);

    // Should return the same profile
    assert_eq!(profile1.bc_offset, profile2.bc_offset);

    // Record on one should be visible on the other
    profile1.record(TypeId::INT);
    assert_eq!(profile2.type_count(TypeId::INT), 1);
}

#[test]
fn test_profile_registry_different_sites() {
    let registry = TypeProfileRegistry::new();

    let profile1 = registry.get_or_create(1, 100);
    let profile2 = registry.get_or_create(1, 200);
    let profile3 = registry.get_or_create(2, 100);

    profile1.record(TypeId::INT);
    profile2.record(TypeId::FLOAT);
    profile3.record(TypeId::STRING);

    assert_eq!(profile1.type_count(TypeId::INT), 1);
    assert_eq!(profile2.type_count(TypeId::FLOAT), 1);
    assert_eq!(profile3.type_count(TypeId::STRING), 1);

    assert_eq!(profile1.type_count(TypeId::FLOAT), 0);
}

#[test]
fn test_profile_registry_clear() {
    let registry = TypeProfileRegistry::new();

    registry.get_or_create(1, 100);
    registry.get_or_create(2, 200);

    registry.clear();

    assert!(registry.get(1, 100).is_none());
    assert!(registry.get(2, 200).is_none());
}

#[test]
fn test_profile_registry_profiles_for_function() {
    let registry = TypeProfileRegistry::new();

    registry.get_or_create(1, 100);
    registry.get_or_create(1, 200);
    registry.get_or_create(1, 300);
    registry.get_or_create(2, 100);

    let func1_profiles = registry.profiles_for_function(1);
    assert_eq!(func1_profiles.len(), 3);

    let func2_profiles = registry.profiles_for_function(2);
    assert_eq!(func2_profiles.len(), 1);
}

// -------------------------------------------------------------------------
// SpeculativeStats Tests
// -------------------------------------------------------------------------

#[test]
fn test_speculative_stats_default() {
    let stats = SpeculativeStats::default();
    assert_eq!(stats.monomorphic_inlines, 0);
    assert_eq!(stats.polymorphic_inlines, 0);
    assert_eq!(stats.megamorphic_skipped, 0);
    assert_eq!(stats.guards_inserted, 0);
    assert_eq!(stats.deopt_paths, 0);
    assert_eq!(stats.nodes_added, 0);
}

// -------------------------------------------------------------------------
// SpeculativeInliner Tests
// -------------------------------------------------------------------------

#[test]
fn test_speculative_inliner_new() {
    let config = InlineConfig::default();
    let profiles = Arc::new(TypeProfileRegistry::new());
    let inliner = SpeculativeInliner::new(config, profiles);

    let stats = inliner.stats();
    assert_eq!(stats.monomorphic_inlines, 0);
}

#[test]
fn test_speculative_inliner_no_profile() {
    let config = InlineConfig::default();
    let profiles = Arc::new(TypeProfileRegistry::new());
    let mut inliner = SpeculativeInliner::new(config, profiles);

    let mut graph = Graph::new();
    let site = CallSite {
        call_node: NodeId::new(0),
        call_kind: CallKind::Direct,
        callee: None,
        loop_depth: 0,
        is_hot: false,
        priority: 0,
        arguments: vec![],
        control_input: None,
    };

    let result = inliner.try_speculative_inline(&mut graph, &site, 1);
    assert!(result.is_err());
}

#[test]
fn test_speculative_inliner_insufficient_calls() {
    let config = InlineConfig::default();
    let profiles = Arc::new(TypeProfileRegistry::new());

    // Record fewer calls than minimum
    let profile = profiles.get_or_create(1, 0);
    for _ in 0..10 {
        profile.record(TypeId::INT);
    }

    let mut inliner = SpeculativeInliner::new(config, Arc::clone(&profiles));

    let mut graph = Graph::new();
    let site = CallSite {
        call_node: NodeId::new(0),
        call_kind: CallKind::Direct,
        callee: None,
        loop_depth: 0,
        is_hot: false,
        priority: 0,
        arguments: vec![],
        control_input: None,
    };

    let result = inliner.try_speculative_inline(&mut graph, &site, 1);
    assert!(result.is_err());
}

#[test]
fn test_speculative_inliner_megamorphic_skip() {
    let config = InlineConfig::default();
    let profiles = Arc::new(TypeProfileRegistry::new());

    // Record many different types to make it megamorphic
    let profile = profiles.get_or_create(1, 0);
    for i in 0..20 {
        for _ in 0..10 {
            profile.record(TypeId::new(i));
        }
    }

    let mut inliner = SpeculativeInliner::new(config, Arc::clone(&profiles));

    let mut graph = Graph::new();
    let site = CallSite {
        call_node: NodeId::new(0),
        call_kind: CallKind::Direct,
        callee: None,
        loop_depth: 0,
        is_hot: false,
        priority: 0,
        arguments: vec![],
        control_input: None,
    };

    let result = inliner.try_speculative_inline(&mut graph, &site, 1);
    assert!(result.is_err());
    assert_eq!(inliner.stats().megamorphic_skipped, 1);
}

// -------------------------------------------------------------------------
// Integration Tests
// -------------------------------------------------------------------------

#[allow(dead_code)]
fn make_simple_callee() -> Graph {
    let mut builder = GraphBuilder::new(8, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    builder.return_value(sum);
    builder.finish()
}

#[test]
fn test_guard_info_serialization() {
    let guard = TypeGuardInfo {
        expected_type: TypeId::INT,
        guard_node: Some(NodeId::new(10)),
        true_branch: Some(NodeId::new(11)),
        false_branch: Some(NodeId::new(12)),
        hit_count: 1000,
        total_calls: 1100,
    };

    assert_eq!(guard.expected_type, TypeId::INT);
    assert!(guard.guard_node.is_some());
    assert!((guard.hit_rate() - 0.909).abs() < 0.01);
}

#[test]
fn test_speculative_inline_info() {
    let info = SpeculativeInlineInfo {
        nodes_added: 50,
        guards_inserted: 2,
        result_node: Some(NodeId::new(100)),
        exit_control: NodeId::new(101),
        guard_info: TypeGuardInfo::new(TypeId::INT),
    };

    assert_eq!(info.nodes_added, 50);
    assert_eq!(info.guards_inserted, 2);
    assert!(info.result_node.is_some());
}

// -------------------------------------------------------------------------
// Edge Case Tests
// -------------------------------------------------------------------------

#[test]
fn test_type_profile_single_call() {
    let profile = TypeProfile::new(0);
    profile.record(TypeId::INT);

    assert_eq!(profile.total_calls(), 1);
    assert!(!profile.is_monomorphic()); // Below minimum threshold
}

#[test]
fn test_dominant_types_empty() {
    let profile = TypeProfile::new(0);
    let dominant = profile.get_dominant_types(5);
    assert!(dominant.is_empty());
}

#[test]
fn test_dominant_types_caching() {
    let profile = TypeProfile::new(0);
    for _ in 0..100 {
        profile.record(TypeId::INT);
    }

    // First call computes and caches
    let dominant1 = profile.get_dominant_types(2);

    // Second call should return cached value
    let dominant2 = profile.get_dominant_types(2);

    assert_eq!(dominant1, dominant2);
}

#[test]
fn test_guard_info_edge_cases() {
    // Zero total calls
    let guard = TypeGuardInfo::with_profile(TypeId::INT, 0, 0);
    assert_eq!(guard.hit_rate(), 0.0);
    assert!(!guard.is_monomorphic());
    assert!(!guard.is_worth_inlining());

    // All hits
    let guard = TypeGuardInfo::with_profile(TypeId::INT, 1000, 1000);
    assert_eq!(guard.hit_rate(), 1.0);
    assert!(guard.is_monomorphic());
    assert!(guard.is_worth_inlining());
}
