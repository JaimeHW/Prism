use super::*;

// =========================================================================
// Helper
// =========================================================================

fn make_profile_with_types(offset: u32, type_samples: &[(u8, u64)]) -> ProfileData {
    let mut profile = ProfileData::new(1);
    for &(type_id, count) in type_samples {
        for _ in 0..count {
            profile.record_type(offset, type_id);
        }
    }
    profile
}

fn make_profile_with_calls(offset: u32, call_samples: &[(u32, u64)]) -> ProfileData {
    let mut profile = ProfileData::new(1);
    for &(target_id, count) in call_samples {
        for _ in 0..count {
            profile.record_call(offset, target_id);
        }
    }
    profile
}

// =========================================================================
// ObservedType Tests
// =========================================================================

#[test]
fn test_observed_type_from_type_id_valid() {
    assert_eq!(ObservedType::from_type_id(0), ObservedType::Int);
    assert_eq!(ObservedType::from_type_id(1), ObservedType::Float);
    assert_eq!(ObservedType::from_type_id(2), ObservedType::Bool);
    assert_eq!(ObservedType::from_type_id(3), ObservedType::None);
    assert_eq!(ObservedType::from_type_id(4), ObservedType::String);
    assert_eq!(ObservedType::from_type_id(5), ObservedType::List);
    assert_eq!(ObservedType::from_type_id(6), ObservedType::Dict);
    assert_eq!(ObservedType::from_type_id(7), ObservedType::Tuple);
    assert_eq!(ObservedType::from_type_id(8), ObservedType::Set);
    assert_eq!(ObservedType::from_type_id(9), ObservedType::Object);
    assert_eq!(ObservedType::from_type_id(10), ObservedType::Callable);
}

#[test]
fn test_observed_type_from_type_id_unknown() {
    assert_eq!(ObservedType::from_type_id(11), ObservedType::Unknown);
    assert_eq!(ObservedType::from_type_id(100), ObservedType::Unknown);
    assert_eq!(ObservedType::from_type_id(255), ObservedType::Unknown);
}

#[test]
fn test_observed_type_is_numeric() {
    assert!(ObservedType::Int.is_numeric());
    assert!(ObservedType::Float.is_numeric());
    assert!(!ObservedType::String.is_numeric());
    assert!(!ObservedType::Bool.is_numeric());
    assert!(!ObservedType::None.is_numeric());
    assert!(!ObservedType::List.is_numeric());
}

#[test]
fn test_observed_type_is_collection() {
    assert!(ObservedType::List.is_collection());
    assert!(ObservedType::Dict.is_collection());
    assert!(ObservedType::Tuple.is_collection());
    assert!(ObservedType::Set.is_collection());
    assert!(!ObservedType::Int.is_collection());
    assert!(!ObservedType::String.is_collection());
    assert!(!ObservedType::Object.is_collection());
}

#[test]
fn test_observed_type_is_heap_allocated() {
    assert!(ObservedType::String.is_heap_allocated());
    assert!(ObservedType::List.is_heap_allocated());
    assert!(ObservedType::Dict.is_heap_allocated());
    assert!(ObservedType::Tuple.is_heap_allocated());
    assert!(ObservedType::Set.is_heap_allocated());
    assert!(ObservedType::Object.is_heap_allocated());
    assert!(ObservedType::Callable.is_heap_allocated());
    assert!(!ObservedType::Int.is_heap_allocated());
    assert!(!ObservedType::Float.is_heap_allocated());
    assert!(!ObservedType::Bool.is_heap_allocated());
    assert!(!ObservedType::None.is_heap_allocated());
}

// =========================================================================
// TypeStability Tests
// =========================================================================

#[test]
fn test_type_stability_can_speculate() {
    assert!(TypeStability::Monomorphic(ObservedType::Int).can_speculate());
    assert!(!TypeStability::Polymorphic.can_speculate());
    assert!(!TypeStability::Megamorphic.can_speculate());
    assert!(!TypeStability::Unstable.can_speculate());
}

#[test]
fn test_type_stability_can_type_switch() {
    assert!(TypeStability::Monomorphic(ObservedType::Int).can_type_switch());
    assert!(TypeStability::Polymorphic.can_type_switch());
    assert!(!TypeStability::Megamorphic.can_type_switch());
    assert!(!TypeStability::Unstable.can_type_switch());
}

// =========================================================================
// OracleConfig Tests
// =========================================================================

#[test]
fn test_oracle_config_default() {
    let config = OracleConfig::default();
    assert_eq!(config.min_samples, 30);
    assert!((config.monomorphic_threshold - 0.90).abs() < f64::EPSILON);
    assert!((config.polymorphic_threshold - 0.70).abs() < f64::EPSILON);
    assert_eq!(config.max_polymorphic_types, 4);
    assert!((config.call_monomorphic_threshold - 0.95).abs() < f64::EPSILON);
}

#[test]
fn test_oracle_config_conservative() {
    let config = OracleConfig::conservative();
    assert_eq!(config.min_samples, 100);
    assert!(config.monomorphic_threshold > OracleConfig::default().monomorphic_threshold);
}

#[test]
fn test_oracle_config_aggressive() {
    let config = OracleConfig::aggressive();
    assert_eq!(config.min_samples, 10);
    assert!(config.monomorphic_threshold < OracleConfig::default().monomorphic_threshold);
}

// =========================================================================
// Oracle Type Query Tests
// =========================================================================

#[test]
fn test_oracle_query_no_profile() {
    let profile = ProfileData::new(1);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(42);
    assert_eq!(
        decision,
        SpeculationDecision::Generic {
            reason: GenericReason::NoProfile
        }
    );
}

#[test]
fn test_oracle_query_insufficient_data() {
    // Only 5 samples (below default threshold of 30)
    let profile = make_profile_with_types(10, &[(0, 5)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    assert_eq!(
        decision,
        SpeculationDecision::Generic {
            reason: GenericReason::InsufficientData
        }
    );
}

#[test]
fn test_oracle_query_monomorphic_int() {
    // 95 ints, 5 floats = 95% int → monomorphic
    let profile = make_profile_with_types(10, &[(0, 95), (1, 5)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    match decision {
        SpeculationDecision::Speculate {
            guard_type,
            confidence,
            deopt_count_estimate,
        } => {
            assert_eq!(guard_type, ObservedType::Int);
            assert!(confidence >= 0.90);
            assert!(deopt_count_estimate <= 0.10);
        }
        other => panic!("Expected Speculate, got {:?}", other),
    }
}

#[test]
fn test_oracle_query_monomorphic_string() {
    // 100 strings → 100% monomorphic
    let profile = make_profile_with_types(5, &[(4, 100)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(5);
    match decision {
        SpeculationDecision::Speculate {
            guard_type,
            confidence,
            ..
        } => {
            assert_eq!(guard_type, ObservedType::String);
            assert!((confidence - 1.0).abs() < f64::EPSILON);
        }
        other => panic!("Expected Speculate, got {:?}", other),
    }
}

#[test]
fn test_oracle_query_polymorphic() {
    // 40 ints, 35 floats, 25 strings → polymorphic
    let profile = make_profile_with_types(10, &[(0, 40), (1, 35), (4, 25)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    match decision {
        SpeculationDecision::TypeSwitch { types } => {
            assert!(types.len() >= 2);
            assert!(types.len() <= 4);
            // All types should be present
            let has_int = types.iter().any(|(t, _)| *t == ObservedType::Int);
            let has_float = types.iter().any(|(t, _)| *t == ObservedType::Float);
            assert!(has_int);
            assert!(has_float);
        }
        other => panic!("Expected TypeSwitch, got {:?}", other),
    }
}

#[test]
fn test_oracle_query_megamorphic() {
    // 6 types with roughly equal distribution → megamorphic
    let profile =
        make_profile_with_types(10, &[(0, 20), (1, 18), (4, 16), (5, 14), (6, 12), (7, 10)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    assert_eq!(
        decision,
        SpeculationDecision::Generic {
            reason: GenericReason::Megamorphic
        }
    );
}

#[test]
fn test_oracle_dominant_type_at() {
    let profile = make_profile_with_types(7, &[(0, 100)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    assert_eq!(oracle.dominant_type_at(7), Some(ObservedType::Int));
    assert_eq!(oracle.dominant_type_at(99), None); // no data
}

#[test]
fn test_oracle_dominant_type_insufficient_data() {
    let profile = make_profile_with_types(7, &[(0, 5)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    assert_eq!(oracle.dominant_type_at(7), None); // below threshold
}

#[test]
fn test_oracle_is_monomorphic_at() {
    // 98% int → monomorphic
    let profile = make_profile_with_types(10, &[(0, 98), (1, 2)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    assert!(oracle.is_monomorphic_at(10));
    assert!(!oracle.is_monomorphic_at(99)); // no data
}

#[test]
fn test_oracle_stability_at() {
    let profile = make_profile_with_types(10, &[(0, 100)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    assert_eq!(
        oracle.stability_at(10),
        TypeStability::Monomorphic(ObservedType::Int)
    );
    assert_eq!(oracle.stability_at(99), TypeStability::Unstable);
}

#[test]
fn test_oracle_confidence_at() {
    let profile = make_profile_with_types(10, &[(0, 80), (1, 20)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let conf = oracle.confidence_at(10);
    assert!((conf - 0.80).abs() < f64::EPSILON);
    assert!((oracle.confidence_at(99) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_oracle_conservative_config_raises_threshold() {
    // 92% int — passes default but not conservative threshold
    let profile = make_profile_with_types(10, &[(0, 92), (1, 8)]);

    let default_oracle = TypeFeedbackOracle::new(&profile);
    let conservative_oracle =
        TypeFeedbackOracle::with_config(&profile, OracleConfig::conservative());

    // Default should speculate (>90%)
    match default_oracle.query_type(10) {
        SpeculationDecision::Speculate { .. } => {} // expected
        other => panic!("Expected Speculate with default, got {:?}", other),
    }

    // Conservative should NOT speculate monomorphically (>98% needed)
    // but may produce TypeSwitch since confidence is > polymorphic_threshold
    match conservative_oracle.query_type(10) {
        SpeculationDecision::Speculate { .. } => {
            panic!("Conservative should not speculate at 92%")
        }
        _ => {} // any non-speculate is correct
    }
}

#[test]
fn test_oracle_aggressive_config_lowers_threshold() {
    // 82% int — fails default but passes aggressive threshold
    let profile = make_profile_with_types(10, &[(0, 82), (1, 18)]);

    let aggressive_oracle = TypeFeedbackOracle::with_config(&profile, OracleConfig::aggressive());

    match aggressive_oracle.query_type(10) {
        SpeculationDecision::Speculate {
            guard_type,
            confidence,
            ..
        } => {
            assert_eq!(guard_type, ObservedType::Int);
            assert!(confidence >= 0.80);
        }
        other => panic!("Expected Speculate with aggressive, got {:?}", other),
    }
}

// =========================================================================
// Call Site Query Tests
// =========================================================================

#[test]
fn test_oracle_call_no_profile() {
    let profile = ProfileData::new(1);
    let oracle = TypeFeedbackOracle::new(&profile);
    assert_eq!(oracle.query_call(42), CallSpeculation::Unknown);
}

#[test]
fn test_oracle_call_monomorphic() {
    // 98% calls to target 42
    let profile = make_profile_with_calls(10, &[(42, 98), (43, 2)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    match oracle.query_call(10) {
        CallSpeculation::Monomorphic {
            target_id,
            confidence,
        } => {
            assert_eq!(target_id, 42);
            assert!(confidence >= 0.95);
        }
        other => panic!("Expected Monomorphic call, got {:?}", other),
    }
}

#[test]
fn test_oracle_call_polymorphic() {
    // 60% target 1, 30% target 2, 10% target 3
    let profile = make_profile_with_calls(10, &[(1, 60), (2, 30), (3, 10)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    match oracle.query_call(10) {
        CallSpeculation::Polymorphic { targets } => {
            assert!(targets.len() >= 2);
            assert!(targets.len() <= 4);
        }
        other => panic!("Expected Polymorphic call, got {:?}", other),
    }
}

#[test]
fn test_oracle_call_megamorphic() {
    // Many targets with low individual frequency
    let profile =
        make_profile_with_calls(10, &[(1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    assert_eq!(oracle.query_call(10), CallSpeculation::Megamorphic);
}

#[test]
fn test_oracle_is_monomorphic_call_at() {
    let profile = make_profile_with_calls(10, &[(42, 100)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    assert!(oracle.is_monomorphic_call_at(10));
    assert!(!oracle.is_monomorphic_call_at(99));
}

// =========================================================================
// Loop Query Tests
// =========================================================================

#[test]
fn test_oracle_loop_trip_count() {
    let mut profile = ProfileData::new(1);
    for _ in 0..500 {
        profile.record_loop_iteration(20);
    }
    let oracle = TypeFeedbackOracle::new(&profile);
    assert_eq!(oracle.loop_trip_count(20), 500);
    assert_eq!(oracle.loop_trip_count(99), 0);
}

#[test]
fn test_oracle_is_hot_loop() {
    let mut profile = ProfileData::new(1);
    for _ in 0..1000 {
        profile.record_loop_iteration(20);
    }
    let oracle = TypeFeedbackOracle::new(&profile);
    assert!(oracle.is_hot_loop(20, 100));
    assert!(!oracle.is_hot_loop(20, 2000));
    assert!(!oracle.is_hot_loop(99, 1));
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_oracle_query_exactly_at_threshold() {
    // Exactly 30 samples (min_samples default) — should be processed
    let profile = make_profile_with_types(10, &[(0, 30)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    match decision {
        SpeculationDecision::Speculate { guard_type, .. } => {
            assert_eq!(guard_type, ObservedType::Int);
        }
        other => panic!("Expected Speculate at exact threshold, got {:?}", other),
    }
}

#[test]
fn test_oracle_query_just_below_threshold() {
    // 29 samples — just below threshold
    let profile = make_profile_with_types(10, &[(0, 29)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    assert_eq!(
        decision,
        SpeculationDecision::Generic {
            reason: GenericReason::InsufficientData
        }
    );
}

#[test]
fn test_oracle_multiple_offsets_independent() {
    let mut profile = ProfileData::new(1);
    // Offset 10: 100% int
    for _ in 0..100 {
        profile.record_type(10, 0);
    }
    // Offset 20: 100% string
    for _ in 0..100 {
        profile.record_type(20, 4);
    }

    let oracle = TypeFeedbackOracle::new(&profile);

    assert_eq!(oracle.dominant_type_at(10), Some(ObservedType::Int));
    assert_eq!(oracle.dominant_type_at(20), Some(ObservedType::String));
}

#[test]
fn test_oracle_polymorphic_exactly_two_types() {
    // 55% int, 45% float → polymorphic
    let profile = make_profile_with_types(10, &[(0, 55), (1, 45)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    match decision {
        SpeculationDecision::TypeSwitch { types } => {
            assert_eq!(types.len(), 2);
        }
        other => panic!("Expected TypeSwitch with 2 types, got {:?}", other),
    }
}

#[test]
fn test_oracle_type_switch_ordering() {
    // Types should be ordered by frequency (descending)
    let profile = make_profile_with_types(10, &[(1, 20), (0, 50), (4, 30)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    match decision {
        SpeculationDecision::TypeSwitch { types } => {
            // First should be Int (50%), then String (30%), then Float (20%)
            assert_eq!(types[0].0, ObservedType::Int);
            assert_eq!(types[1].0, ObservedType::String);
            assert_eq!(types[2].0, ObservedType::Float);
        }
        other => panic!("Expected TypeSwitch, got {:?}", other),
    }
}

#[test]
fn test_oracle_single_type_below_mono_threshold_as_speculate() {
    // 85% int — below monomorphic threshold (90%) but only 1 significant type
    // Should still produce a Speculate decision if the polymorphic path
    // detects only 1 significant type
    let profile = make_profile_with_types(10, &[(0, 85), (1, 15)]);
    let oracle = TypeFeedbackOracle::new(&profile);
    let decision = oracle.query_type(10);
    // With 85%, below mono threshold but polymorphic should still work
    // since 2 types are significant
    match decision {
        SpeculationDecision::TypeSwitch { types } => {
            assert!(types.len() >= 2);
        }
        SpeculationDecision::Speculate { .. } => {
            // Also acceptable if oracle determines single type is dominant
        }
        other => panic!("Expected TypeSwitch or Speculate, got {:?}", other),
    }
}
