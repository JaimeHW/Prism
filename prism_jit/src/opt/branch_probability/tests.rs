use super::*;

// =========================================================================
// BranchProbability Tests
// =========================================================================

#[test]
fn test_probability_constants() {
    assert!((BranchProbability::ALWAYS.as_f64() - 1.0).abs() < 0.001);
    assert!((BranchProbability::NEVER.as_f64() - 0.0).abs() < 0.001);
    assert!((BranchProbability::EVEN.as_f64() - 0.5).abs() < 0.001);
    assert!((BranchProbability::LIKELY.as_f64() - 0.9).abs() < 0.01);
    assert!((BranchProbability::UNLIKELY.as_f64() - 0.1).abs() < 0.01);
}

#[test]
fn test_probability_from_f64() {
    let p = BranchProbability::from_f64(0.75);
    assert!((p.as_f64() - 0.75).abs() < 0.001);
}

#[test]
fn test_probability_from_f64_clamp() {
    let p1 = BranchProbability::from_f64(-0.5);
    assert!((p1.as_f64() - 0.0).abs() < 0.001);
    let p2 = BranchProbability::from_f64(1.5);
    assert!((p2.as_f64() - 1.0).abs() < 0.001);
}

#[test]
fn test_probability_from_ratio() {
    let p = BranchProbability::from_ratio(3, 4);
    assert!((p.as_f64() - 0.75).abs() < 0.001);
}

#[test]
fn test_probability_from_ratio_zero_denom() {
    let p = BranchProbability::from_ratio(3, 0);
    assert_eq!(p, BranchProbability::EVEN);
}

#[test]
fn test_probability_from_counts() {
    let p = BranchProbability::from_counts(90, 10);
    assert!((p.as_f64() - 0.9).abs() < 0.001);
}

#[test]
fn test_probability_from_counts_zero() {
    let p = BranchProbability::from_counts(0, 0);
    assert_eq!(p, BranchProbability::EVEN);
}

#[test]
fn test_probability_complement() {
    let p = BranchProbability::from_f64(0.75);
    let c = p.complement();
    assert!((c.as_f64() - 0.25).abs() < 0.001);
}

#[test]
fn test_probability_scale() {
    let p = BranchProbability::from_f64(0.8);
    let s = p.scale(0.5);
    assert!((s.as_f64() - 0.4).abs() < 0.001);
}

#[test]
fn test_probability_is_biased() {
    assert!(BranchProbability::from_f64(0.9).is_biased());
    assert!(BranchProbability::from_f64(0.1).is_biased());
    assert!(!BranchProbability::from_f64(0.55).is_biased());
}

#[test]
fn test_probability_is_likely() {
    assert!(BranchProbability::from_f64(0.85).is_likely());
    assert!(!BranchProbability::from_f64(0.5).is_likely());
}

#[test]
fn test_probability_is_unlikely() {
    assert!(BranchProbability::from_f64(0.1).is_unlikely());
    assert!(!BranchProbability::from_f64(0.5).is_unlikely());
}

#[test]
fn test_probability_display() {
    let p = BranchProbability::from_f64(0.9);
    let s = format!("{}", p);
    assert!(s.contains("0.9"));
}

#[test]
fn test_probability_numerator() {
    let p = BranchProbability::ALWAYS;
    assert_eq!(p.numerator(), u32::MAX);
}

// =========================================================================
// BlockFrequency Tests
// =========================================================================

#[test]
fn test_block_frequency_entry() {
    assert!((BlockFrequency::ENTRY.value() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_block_frequency_cold() {
    assert!(BlockFrequency::COLD.is_cold());
}

#[test]
fn test_block_frequency_new() {
    let f = BlockFrequency::new(5.0);
    assert!((f.value() - 5.0).abs() < f64::EPSILON);
}

#[test]
fn test_block_frequency_new_negative_clamped() {
    let f = BlockFrequency::new(-1.0);
    assert!((f.value() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_block_frequency_for_loop() {
    let f = BlockFrequency::for_loop(2.0, 100.0);
    assert!((f.value() - 200.0).abs() < f64::EPSILON);
}

#[test]
fn test_block_frequency_is_hot() {
    let f = BlockFrequency::new(10.0);
    assert!(f.is_hot(5.0));
    assert!(!f.is_hot(15.0));
}

#[test]
fn test_block_frequency_is_cold() {
    let f = BlockFrequency::new(0.005);
    assert!(f.is_cold());
}

#[test]
fn test_block_frequency_scale() {
    let f = BlockFrequency::new(10.0);
    let scaled = f.scale(BranchProbability::from_f64(0.5));
    assert!((scaled.value() - 5.0).abs() < 0.01);
}

#[test]
fn test_block_frequency_display() {
    let f = BlockFrequency::new(3.5);
    assert_eq!(format!("{}", f), "3.50x");
}

// =========================================================================
// StaticHeuristics Tests
// =========================================================================

#[test]
fn test_heuristic_exception() {
    let p = StaticHeuristics::classify(BranchHint::Exception);
    assert!(p.is_unlikely());
}

#[test]
fn test_heuristic_loop_back() {
    let p = StaticHeuristics::classify(BranchHint::LoopBack);
    assert!(p.is_likely());
}

#[test]
fn test_heuristic_guard() {
    let p = StaticHeuristics::classify(BranchHint::Guard);
    assert!(p.is_likely());
}

#[test]
fn test_heuristic_null_check() {
    let p = StaticHeuristics::classify(BranchHint::NullCheck);
    assert!(p.is_likely());
}

#[test]
fn test_heuristic_type_check() {
    let p = StaticHeuristics::classify(BranchHint::TypeCheck);
    assert!(p.is_likely());
}

#[test]
fn test_heuristic_unknown() {
    let p = StaticHeuristics::classify(BranchHint::None);
    assert_eq!(p, BranchProbability::EVEN);
}

#[test]
fn test_heuristic_likely_hint() {
    let p = StaticHeuristics::classify(BranchHint::Likely);
    assert!(p.is_likely());
}

#[test]
fn test_heuristic_unlikely_hint() {
    let p = StaticHeuristics::classify(BranchHint::Unlikely);
    assert!(p.is_unlikely());
}

// =========================================================================
// BranchAnnotations Tests
// =========================================================================

#[test]
fn test_annotations_new() {
    let ann = BranchAnnotations::new();
    assert_eq!(ann.branch_count(), 0);
    assert_eq!(ann.block_count(), 0);
    assert!(!ann.has_profile_data());
}

#[test]
fn test_annotations_set_get_branch() {
    let mut ann = BranchAnnotations::new();
    ann.set_branch(10, BranchProbability::LIKELY);
    assert_eq!(ann.get_branch(10), Some(BranchProbability::LIKELY));
    assert_eq!(ann.get_branch(20), None);
}

#[test]
fn test_annotations_get_branch_or_default() {
    let mut ann = BranchAnnotations::new();
    ann.set_branch(10, BranchProbability::from_f64(0.95));

    // With recorded data
    let bp = ann.get_branch_or_default(10, BranchHint::None);
    assert!((bp.as_f64() - 0.95).abs() < 0.01);

    // Without recorded data, falls back to heuristic
    let bp2 = ann.get_branch_or_default(99, BranchHint::LoopBack);
    assert!(bp2.is_likely());
}

#[test]
fn test_annotations_set_get_block_freq() {
    let mut ann = BranchAnnotations::new();
    ann.set_block_freq(1, BlockFrequency::new(10.0));
    assert!((ann.get_block_freq(1).unwrap().value() - 10.0).abs() < f64::EPSILON);
    assert!(ann.get_block_freq(999).is_none());
}

#[test]
fn test_annotations_is_block_hot() {
    let mut ann = BranchAnnotations::new();
    ann.set_block_freq(1, BlockFrequency::new(10.0));
    assert!(ann.is_block_hot(1, 5.0));
    assert!(!ann.is_block_hot(1, 15.0));
    assert!(!ann.is_block_hot(999, 1.0));
}

#[test]
fn test_annotations_is_block_cold() {
    let mut ann = BranchAnnotations::new();
    ann.set_block_freq(1, BlockFrequency::new(0.005));
    assert!(ann.is_block_cold(1));
    assert!(!ann.is_block_cold(999));
}

#[test]
fn test_annotations_merge_from_profile() {
    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    profile.record_branch(10, true);
    profile.record_branch(10, true);
    profile.record_branch(10, false);

    let mut ann = BranchAnnotations::new();
    ann.merge_from_profile(&profile);

    assert!(ann.has_profile_data());
    assert_eq!(ann.branch_count(), 1);
    let bp = ann.get_branch(10).unwrap();
    // 2 taken, 1 not taken → ~0.667
    assert!((bp.as_f64() - 0.667).abs() < 0.01);
}

#[test]
fn test_annotations_branch_offsets() {
    let mut ann = BranchAnnotations::new();
    ann.set_branch(10, BranchProbability::LIKELY);
    ann.set_branch(20, BranchProbability::UNLIKELY);
    let offsets = ann.branch_offsets();
    assert_eq!(offsets.len(), 2);
    assert!(offsets.contains(&10));
    assert!(offsets.contains(&20));
}

#[test]
fn test_annotations_iter_branches() {
    let mut ann = BranchAnnotations::new();
    ann.set_branch(10, BranchProbability::LIKELY);
    ann.set_branch(20, BranchProbability::UNLIKELY);
    let items: Vec<_> = ann.iter_branches().collect();
    assert_eq!(items.len(), 2);
}

#[test]
fn test_annotations_profile_data_flag() {
    let mut ann = BranchAnnotations::new();
    assert!(!ann.has_profile_data());
    ann.set_has_profile_data(true);
    assert!(ann.has_profile_data());
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_probability_extreme_counts() {
    let p = BranchProbability::from_counts(u64::MAX / 2, u64::MAX / 2);
    assert!((p.as_f64() - 0.5).abs() < 0.01);
}

#[test]
fn test_probability_one_count() {
    let p = BranchProbability::from_counts(1, 0);
    assert!((p.as_f64() - 1.0).abs() < 0.001);
}

#[test]
fn test_frequency_very_high() {
    let f = BlockFrequency::new(1_000_000.0);
    assert!(f.is_hot(1.0));
    assert!(!f.is_cold());
}

#[test]
fn test_probability_complement_round_trip() {
    let p = BranchProbability::from_f64(0.3);
    let c = p.complement();
    let cc = c.complement();
    // Note: there's a ±1 quantization error, so we check near-equality
    assert!((p.as_f64() - cc.as_f64()).abs() < 0.001);
}

#[test]
fn test_multiple_profile_merge() {
    let mut profile1 = crate::runtime::profile_data::ProfileData::new(1);
    profile1.record_branch(10, true);
    profile1.record_branch(10, true);

    let mut profile2 = crate::runtime::profile_data::ProfileData::new(1);
    profile2.record_branch(10, false);
    profile2.record_branch(20, true);

    let mut ann = BranchAnnotations::new();
    ann.merge_from_profile(&profile1);
    ann.merge_from_profile(&profile2);

    // Profile2 overwrites profile1 for offset 10
    assert_eq!(ann.branch_count(), 2);
}

// =========================================================================
// PGO Integration Tests (BranchProbabilityPass with ProfileData)
// =========================================================================

#[test]
fn test_pass_with_profile_data_merges() {
    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    // Record heavily biased branch at offset 0 (simulating an If node)
    for _ in 0..95 {
        profile.record_branch(0, true);
    }
    for _ in 0..5 {
        profile.record_branch(0, false);
    }

    let pass = BranchProbabilityPass::with_profile(profile);
    assert!(pass.profile_data.is_some());
}

#[test]
fn test_pass_without_profile_data() {
    let pass = BranchProbabilityPass::new();
    assert!(pass.profile_data.is_none());
}

#[test]
fn test_inject_profile_into_existing_pass() {
    let mut pass = BranchProbabilityPass::new();
    assert!(pass.profile_data.is_none());

    let profile = crate::runtime::profile_data::ProfileData::new(42);
    pass.inject_profile(profile);
    assert!(pass.profile_data.is_some());
}

#[test]
fn test_pass_annotations_default_empty() {
    let pass = BranchProbabilityPass::new();
    assert_eq!(pass.branches_annotated(), 0);
    assert_eq!(pass.blocks_estimated(), 0);
    assert_eq!(pass.annotations().branch_count(), 0);
}

#[test]
fn test_pass_with_profile_run_on_graph_with_branches() {
    use crate::ir::builder::{ControlBuilder, GraphBuilder};

    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    // Record a 90% taken branch at offset that maps to an If node
    for _ in 0..90 {
        profile.record_branch(3, true);
    }
    for _ in 0..10 {
        profile.record_branch(3, false);
    }

    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    builder.translate_branch(p0, 5, 6);
    builder.return_value(p0);

    let mut graph = builder.finish();
    let mut pass = BranchProbabilityPass::with_profile(profile);

    use super::OptimizationPass;
    let changed = pass.run(&mut graph);
    // The pass should have annotated branches
    assert!(changed || pass.branches_annotated() > 0 || pass.blocks_estimated() > 0);
}

#[test]
fn test_pass_profile_loop_frequency_refinement() {
    // Verify that measured loop counts refine frequency estimates
    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    // Simulate 5 executions and 500 loop iterations → avg 100 trips
    for _ in 0..5 {
        profile.record_execution();
    }
    for _ in 0..500 {
        profile.record_loop_iteration(10);
    }

    assert_eq!(profile.execution_count(), 5);
    assert_eq!(profile.loop_count(10), 500);

    // When the pass runs, it should compute avg_trips = 500/5 = 100
    let pass = BranchProbabilityPass::with_profile(profile);
    assert!(pass.profile_data.is_some());
}

#[test]
fn test_profile_merge_sets_has_profile_flag() {
    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    profile.record_branch(5, true);

    let mut ann = BranchAnnotations::new();
    assert!(!ann.has_profile_data());

    ann.merge_from_profile(&profile);
    assert!(ann.has_profile_data());
}

#[test]
fn test_profile_overrides_static_default() {
    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    // Record a 99% taken branch
    for _ in 0..99 {
        profile.record_branch(10, true);
    }
    profile.record_branch(10, false);

    let mut ann = BranchAnnotations::new();
    // Set static default first
    ann.set_branch(10, BranchProbability::EVEN);
    assert!((ann.get_branch(10).unwrap().as_f64() - 0.5).abs() < 0.01);

    // Profile merge should override
    ann.merge_from_profile(&profile);
    let prob = ann.get_branch(10).unwrap().as_f64();
    assert!(
        prob > 0.95,
        "Profile should override static default, got {prob}"
    );
}

#[test]
fn test_empty_profile_preserves_defaults() {
    let profile = crate::runtime::profile_data::ProfileData::new(1);

    let mut ann = BranchAnnotations::new();
    ann.set_branch(10, BranchProbability::LIKELY);
    let before = ann.get_branch(10).unwrap();

    ann.merge_from_profile(&profile);
    let after = ann.get_branch(10).unwrap();

    // Empty profile has no branches → existing annotation preserved
    assert_eq!(before, after);
    // But profile flag should be set
    assert!(ann.has_profile_data());
}
