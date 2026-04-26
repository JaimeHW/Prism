use super::*;

// =========================================================================
// TypeHint Tests
// =========================================================================

#[test]
fn test_type_hint_is_int() {
    assert!(TypeHint::IntInt.is_int());
    assert!(!TypeHint::FloatFloat.is_int());
    assert!(!TypeHint::StrStr.is_int());
    assert!(!TypeHint::None.is_int());
}

#[test]
fn test_type_hint_is_float() {
    assert!(TypeHint::FloatFloat.is_float());
    assert!(TypeHint::IntFloat.is_float());
    assert!(TypeHint::FloatInt.is_float());
    assert!(!TypeHint::IntInt.is_float());
    assert!(!TypeHint::StrStr.is_float());
}

#[test]
fn test_type_hint_is_string() {
    assert!(TypeHint::StrStr.is_string());
    assert!(TypeHint::StrInt.is_string());
    assert!(TypeHint::IntStr.is_string());
    assert!(!TypeHint::IntInt.is_string());
    assert!(!TypeHint::FloatFloat.is_string());
}

#[test]
fn test_type_hint_is_valid() {
    assert!(!TypeHint::None.is_valid());
    assert!(TypeHint::IntInt.is_valid());
    assert!(TypeHint::FloatFloat.is_valid());
    assert!(TypeHint::StrStr.is_valid());
}

#[test]
fn test_type_hint_repr() {
    assert_eq!(TypeHint::None as u8, 0);
    assert_eq!(TypeHint::IntInt as u8, 1);
    assert_eq!(TypeHint::ListList as u8, 8);
}

// =========================================================================
// PgoBranchHint Tests
// =========================================================================

#[test]
fn test_branch_hint_new() {
    let hint = PgoBranchHint::new(42, u32::MAX / 2);
    assert_eq!(hint.offset, 42);
    assert_eq!(hint.taken_numer, u32::MAX / 2);
}

#[test]
fn test_branch_hint_from_counts_balanced() {
    let hint = PgoBranchHint::from_counts(10, 100, 100);
    let prob = hint.taken_probability();
    assert!((prob - 0.5).abs() < 0.01);
}

#[test]
fn test_branch_hint_from_counts_always_taken() {
    let hint = PgoBranchHint::from_counts(0, 1000, 0);
    assert!(hint.taken_probability() > 0.99);
}

#[test]
fn test_branch_hint_from_counts_never_taken() {
    let hint = PgoBranchHint::from_counts(0, 0, 1000);
    assert!(hint.taken_probability() < 0.01);
}

#[test]
fn test_branch_hint_from_counts_zero_total() {
    let hint = PgoBranchHint::from_counts(5, 0, 0);
    // Should default to ~50/50
    let prob = hint.taken_probability();
    assert!((prob - 0.5).abs() < 0.01);
}

#[test]
fn test_branch_hint_from_f64() {
    let hint = PgoBranchHint::from_f64(0, 0.75);
    let prob = hint.taken_probability();
    assert!((prob - 0.75).abs() < 0.01);
}

#[test]
fn test_branch_hint_from_f64_clamped() {
    let hint_over = PgoBranchHint::from_f64(0, 1.5);
    assert!(hint_over.taken_probability() <= 1.0);

    let hint_under = PgoBranchHint::from_f64(0, -0.5);
    assert!(hint_under.taken_probability() >= 0.0);
}

#[test]
fn test_branch_hint_not_taken_probability() {
    let hint = PgoBranchHint::from_f64(0, 0.3);
    let not_taken = hint.not_taken_probability();
    assert!((not_taken - 0.7).abs() < 0.01);
}

#[test]
fn test_branch_hint_is_biased() {
    // Heavily biased
    let hint_biased = PgoBranchHint::from_f64(0, 0.95);
    assert!(hint_biased.is_biased());

    // Not biased
    let hint_even = PgoBranchHint::from_f64(0, 0.5);
    assert!(!hint_even.is_biased());

    // Slightly biased but within threshold
    let hint_slight = PgoBranchHint::from_f64(0, 0.6);
    assert!(!hint_slight.is_biased());
}

#[test]
fn test_branch_hint_is_likely_taken() {
    let likely = PgoBranchHint::from_f64(0, 0.9);
    assert!(likely.is_likely_taken());

    let unlikely = PgoBranchHint::from_f64(0, 0.3);
    assert!(!unlikely.is_likely_taken());
}

#[test]
fn test_branch_hint_is_unlikely_taken() {
    let unlikely = PgoBranchHint::from_f64(0, 0.1);
    assert!(unlikely.is_unlikely_taken());

    let likely = PgoBranchHint::from_f64(0, 0.7);
    assert!(!likely.is_unlikely_taken());
}

#[test]
fn test_branch_hint_complement() {
    let hint = PgoBranchHint::from_f64(42, 0.8);
    let comp = hint.complement();
    assert_eq!(comp.offset, 42);
    let comp_prob = comp.taken_probability();
    assert!((comp_prob - 0.2).abs() < 0.01);
}

#[test]
fn test_branch_hint_complement_roundtrip() {
    let hint = PgoBranchHint::new(0, 1000000);
    let double_comp = hint.complement().complement();
    // Not exact roundtrip due to wrapping arithmetic, but close
    assert!((hint.taken_numer as i64 - double_comp.taken_numer as i64).unsigned_abs() <= 1,);
}

#[test]
fn test_branch_hint_from_counts_large_numbers() {
    let hint = PgoBranchHint::from_counts(0, u64::MAX / 2, u64::MAX / 2);
    let prob = hint.taken_probability();
    assert!((prob - 0.5).abs() < 0.01);
}

#[test]
fn test_branch_hint_from_counts_saturating_add() {
    // Very large counts that would overflow without saturating add
    let hint = PgoBranchHint::from_counts(0, u64::MAX, 1);
    // Should not panic, probability should be near 1.0
    assert!(hint.taken_probability() > 0.99);
}

#[test]
fn test_branch_hint_extreme_probabilities() {
    let always = PgoBranchHint::new(0, u32::MAX);
    assert!(always.taken_probability() > 0.99);
    assert!(always.is_likely_taken());
    assert!(!always.is_unlikely_taken());

    let never = PgoBranchHint::new(0, 0);
    assert!(never.taken_probability() < 0.01);
    assert!(!never.is_likely_taken());
    assert!(never.is_unlikely_taken());
}

// =========================================================================
// PgoCallTarget Tests
// =========================================================================

#[test]
fn test_call_target_new() {
    let target = PgoCallTarget::new(10, 42, 100);
    assert_eq!(target.offset, 10);
    assert_eq!(target.target_id, 42);
    assert_eq!(target.frequency, 100);
}

#[test]
fn test_call_target_is_frequent() {
    let target = PgoCallTarget::new(0, 1, 50);
    assert!(target.is_frequent(50));
    assert!(target.is_frequent(10));
    assert!(!target.is_frequent(51));
}

#[test]
fn test_call_target_zero_frequency() {
    let target = PgoCallTarget::new(0, 1, 0);
    assert!(!target.is_frequent(1));
}

// =========================================================================
// CallSiteProfile Tests
// =========================================================================

#[test]
fn test_call_site_profile_new() {
    let profile = CallSiteProfile::new(10);
    assert_eq!(profile.offset, 10);
    assert_eq!(profile.target_count(), 0);
    assert_eq!(profile.total_calls, 0);
}

#[test]
fn test_call_site_profile_add_target() {
    let mut profile = CallSiteProfile::new(10);
    profile.add_target(1, 100);
    assert_eq!(profile.target_count(), 1);
    assert_eq!(profile.total_calls, 100);
}

#[test]
fn test_call_site_profile_sorted_by_frequency() {
    let mut profile = CallSiteProfile::new(10);
    profile.add_target(1, 10);
    profile.add_target(2, 100);
    profile.add_target(3, 50);

    assert_eq!(profile.targets[0].target_id, 2); // Most frequent
    assert_eq!(profile.targets[1].target_id, 3);
    assert_eq!(profile.targets[2].target_id, 1); // Least frequent
}

#[test]
fn test_call_site_profile_morphism_cold() {
    let profile = CallSiteProfile::new(10);
    assert_eq!(profile.morphism(), CallMorphism::Cold);
}

#[test]
fn test_call_site_profile_morphism_monomorphic() {
    let mut profile = CallSiteProfile::new(10);
    profile.add_target(1, 100);
    assert_eq!(profile.morphism(), CallMorphism::Monomorphic);
}

#[test]
fn test_call_site_profile_morphism_polymorphic() {
    let mut profile = CallSiteProfile::new(10);
    profile.add_target(1, 100);
    profile.add_target(2, 50);
    assert_eq!(profile.morphism(), CallMorphism::Polymorphic);

    profile.add_target(3, 30);
    profile.add_target(4, 10);
    assert_eq!(profile.morphism(), CallMorphism::Polymorphic);
}

#[test]
fn test_call_site_profile_morphism_megamorphic() {
    let mut profile = CallSiteProfile::new(10);
    for i in 0..5 {
        profile.add_target(i, 10);
    }
    assert_eq!(profile.morphism(), CallMorphism::Megamorphic);
}

#[test]
fn test_call_site_profile_dominant_target() {
    let mut profile = CallSiteProfile::new(10);
    profile.add_target(1, 100);
    profile.add_target(2, 10);
    profile.add_target(3, 5);

    let dom = profile.dominant_target().unwrap();
    assert_eq!(dom.target_id, 1);
    assert_eq!(dom.frequency, 100);
}

#[test]
fn test_call_site_profile_dominant_target_none() {
    let profile = CallSiteProfile::new(10);
    assert!(profile.dominant_target().is_none());
}

#[test]
fn test_call_site_profile_has_dominant_target() {
    let mut profile = CallSiteProfile::new(10);
    profile.add_target(1, 90);
    profile.add_target(2, 10);

    assert!(profile.has_dominant_target(85)); // 90%
    assert!(profile.has_dominant_target(90));
    assert!(!profile.has_dominant_target(91));
}

#[test]
fn test_call_site_profile_no_dominant_when_empty() {
    let profile = CallSiteProfile::new(10);
    assert!(!profile.has_dominant_target(50));
}

#[test]
fn test_call_site_profile_total_calls_accumulates() {
    let mut profile = CallSiteProfile::new(10);
    profile.add_target(1, 100);
    profile.add_target(2, 200);
    profile.add_target(3, 300);
    assert_eq!(profile.total_calls, 600);
}

// =========================================================================
// CallMorphism Tests
// =========================================================================

// =========================================================================
// SpeculationProvider Tests
// =========================================================================

#[test]
fn test_no_speculation_provider() {
    let provider = NoSpeculation;
    assert_eq!(provider.get_type_hint(1, 0), TypeHint::None);
    assert_eq!(provider.get_type_hint(100, 50), TypeHint::None);
    assert!(provider.get_branch_hint(1, 0).is_none());
    assert!(provider.get_all_branch_hints(1).is_empty());
    assert!(provider.get_call_targets(1, 0).is_none());
    assert!(!provider.has_profile_data(1));
    assert_eq!(provider.execution_count(1), 0);
}

// =========================================================================
// StaticSpeculation Tests
// =========================================================================

#[test]
fn test_static_speculation_new() {
    let spec = StaticSpeculation::new();
    assert_eq!(spec.get_type_hint(1, 0), TypeHint::None);
    assert!(!spec.has_profile_data(1));
}

#[test]
fn test_static_speculation_type_hints() {
    let mut spec = StaticSpeculation::new();
    spec.add_type_hint(1, 10, TypeHint::IntInt);
    spec.add_type_hint(1, 20, TypeHint::FloatFloat);
    spec.add_type_hint(2, 10, TypeHint::StrStr);

    assert_eq!(spec.get_type_hint(1, 10), TypeHint::IntInt);
    assert_eq!(spec.get_type_hint(1, 20), TypeHint::FloatFloat);
    assert_eq!(spec.get_type_hint(2, 10), TypeHint::StrStr);
    assert_eq!(spec.get_type_hint(1, 30), TypeHint::None); // Missing
    assert_eq!(spec.get_type_hint(3, 10), TypeHint::None); // Missing code_id
}

#[test]
fn test_static_speculation_branch_hints() {
    let mut spec = StaticSpeculation::new();
    let hint1 = PgoBranchHint::from_f64(10, 0.9);
    let hint2 = PgoBranchHint::from_f64(20, 0.1);
    spec.add_branch_hint(1, hint1);
    spec.add_branch_hint(1, hint2);

    let retrieved = spec.get_branch_hint(1, 10).unwrap();
    assert!((retrieved.taken_probability() - 0.9).abs() < 0.01);

    let retrieved2 = spec.get_branch_hint(1, 20).unwrap();
    assert!((retrieved2.taken_probability() - 0.1).abs() < 0.01);

    assert!(spec.get_branch_hint(1, 30).is_none());
    assert!(spec.get_branch_hint(2, 10).is_none());
}

#[test]
fn test_static_speculation_all_branch_hints() {
    let mut spec = StaticSpeculation::new();
    spec.add_branch_hint(1, PgoBranchHint::from_f64(10, 0.9));
    spec.add_branch_hint(1, PgoBranchHint::from_f64(20, 0.1));
    spec.add_branch_hint(2, PgoBranchHint::from_f64(10, 0.5));

    let hints = spec.get_all_branch_hints(1);
    assert_eq!(hints.len(), 2);

    let hints2 = spec.get_all_branch_hints(2);
    assert_eq!(hints2.len(), 1);

    let hints3 = spec.get_all_branch_hints(3);
    assert!(hints3.is_empty());
}

#[test]
fn test_static_speculation_call_targets() {
    let mut spec = StaticSpeculation::new();
    let mut profile = CallSiteProfile::new(10);
    profile.add_target(42, 100);
    profile.add_target(43, 50);
    spec.add_call_profile(1, profile);

    let retrieved = spec.get_call_targets(1, 10).unwrap();
    assert_eq!(retrieved.target_count(), 2);
    assert_eq!(retrieved.dominant_target().unwrap().target_id, 42);

    assert!(spec.get_call_targets(1, 20).is_none());
    assert!(spec.get_call_targets(2, 10).is_none());
}

#[test]
fn test_static_speculation_has_profile_data() {
    let mut spec = StaticSpeculation::new();
    assert!(!spec.has_profile_data(1));

    spec.add_branch_hint(1, PgoBranchHint::from_f64(10, 0.9));
    assert!(spec.has_profile_data(1));
    assert!(!spec.has_profile_data(2));
}

#[test]
fn test_static_speculation_execution_count() {
    let mut spec = StaticSpeculation::new();
    assert_eq!(spec.execution_count(1), 0);

    spec.set_execution_count(1, 5000);
    assert_eq!(spec.execution_count(1), 5000);
    assert_eq!(spec.execution_count(2), 0);
}

#[test]
fn test_static_speculation_mixed_data() {
    let mut spec = StaticSpeculation::new();

    // Add all types of data for code unit 1
    spec.add_type_hint(1, 10, TypeHint::IntInt);
    spec.add_branch_hint(1, PgoBranchHint::from_f64(20, 0.9));

    let mut call_profile = CallSiteProfile::new(30);
    call_profile.add_target(99, 500);
    spec.add_call_profile(1, call_profile);

    spec.set_execution_count(1, 10000);

    // Verify all data
    assert_eq!(spec.get_type_hint(1, 10), TypeHint::IntInt);
    assert!(spec.get_branch_hint(1, 20).is_some());
    assert!(spec.get_call_targets(1, 30).is_some());
    assert_eq!(spec.execution_count(1), 10000);
    assert!(spec.has_profile_data(1));
}

#[test]
fn test_static_speculation_multiple_code_units() {
    let mut spec = StaticSpeculation::new();

    spec.add_branch_hint(1, PgoBranchHint::from_f64(10, 0.9));
    spec.add_branch_hint(2, PgoBranchHint::from_f64(10, 0.1));

    let h1 = spec.get_branch_hint(1, 10).unwrap();
    let h2 = spec.get_branch_hint(2, 10).unwrap();

    assert!(h1.is_likely_taken());
    assert!(h2.is_unlikely_taken());
}

#[test]
fn test_static_speculation_marks_profiled() {
    let mut spec = StaticSpeculation::new();

    // Adding call profile should mark as profiled
    spec.add_call_profile(5, CallSiteProfile::new(0));
    assert!(spec.has_profile_data(5));

    // Setting exec count should also mark as profiled
    spec.set_execution_count(7, 100);
    assert!(spec.has_profile_data(7));
}

#[test]
fn test_branch_hint_probability_bounds() {
    // All probabilities should be in [0.0, 1.0]
    for numer in [0, 1, u32::MAX / 4, u32::MAX / 2, u32::MAX - 1, u32::MAX] {
        let hint = PgoBranchHint::new(0, numer);
        let prob = hint.taken_probability();
        assert!(prob >= 0.0, "probability {} should be >= 0", prob);
        assert!(prob <= 1.0, "probability {} should be <= 1", prob);
    }
}

#[test]
fn test_branch_hint_from_counts_precision() {
    // 90% taken
    let hint = PgoBranchHint::from_counts(0, 900, 100);
    let prob = hint.taken_probability();
    assert!((prob - 0.9).abs() < 0.001, "Expected ~0.9, got {}", prob);

    // 10% taken
    let hint = PgoBranchHint::from_counts(0, 10, 90);
    let prob = hint.taken_probability();
    assert!((prob - 0.1).abs() < 0.001, "Expected ~0.1, got {}", prob);
}

#[test]
fn test_call_site_profile_single_target_fully_dominant() {
    let mut profile = CallSiteProfile::new(0);
    profile.add_target(1, 1000);
    assert!(profile.has_dominant_target(100)); // 100%
}

#[test]
fn test_call_site_profile_equal_targets_not_dominant() {
    let mut profile = CallSiteProfile::new(0);
    profile.add_target(1, 50);
    profile.add_target(2, 50);
    assert!(!profile.has_dominant_target(51)); // Each is 50%
    assert!(profile.has_dominant_target(50)); // 50% exactly
}
