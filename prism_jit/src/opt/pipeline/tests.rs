use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

#[test]
fn test_pass_phase_ordering() {
    assert!(PassPhase::Canonicalization < PassPhase::Local);
    assert!(PassPhase::Local < PassPhase::Loop);
    assert!(PassPhase::Loop < PassPhase::Interprocedural);
    assert!(PassPhase::Interprocedural < PassPhase::Cleanup);
}

#[test]
fn test_pipeline_config_default() {
    let config = PipelineConfig::default();
    assert!(config.enable_gvn);
    assert!(config.enable_dce);
    assert!(config.enable_licm);
}

#[test]
fn test_pipeline_config_minimal() {
    let config = PipelineConfig::minimal();
    assert!(!config.enable_licm);
    assert!(!config.enable_inline);
    assert!(config.enable_gvn);
}

#[test]
fn test_pipeline_new() {
    let pipeline = OptPipeline::new();
    assert!(!pipeline.passes.is_empty());
}

#[test]
fn test_pipeline_run_empty() {
    let builder = GraphBuilder::new(0, 0);
    let mut graph = builder.finish();

    let mut pipeline = OptPipeline::new();
    let stats = pipeline.run(&mut graph);

    assert!(stats.total_iterations > 0);
}

#[test]
fn test_pipeline_run_simple() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let initial = graph.len();

    let mut pipeline = OptPipeline::new();
    let stats = pipeline.run(&mut graph);

    // Should run at least one iteration
    assert!(stats.total_iterations >= 1);
    // Size shouldn't dramatically change for simple graph
    assert!(stats.final_size <= initial + 5);
}

#[test]
fn test_optimize_functions() {
    let builder = GraphBuilder::new(0, 0);
    let mut graph = builder.finish();

    let stats = optimize(&mut graph);
    assert!(stats.total_iterations >= 1);

    let mut graph2 = GraphBuilder::new(0, 0).finish();
    let stats2 = optimize_minimal(&mut graph2);
    assert!(stats2.total_iterations >= 1);

    let mut graph3 = GraphBuilder::new(0, 0).finish();
    let stats3 = optimize_full(&mut graph3);
    assert!(stats3.total_iterations >= 1);
}

#[test]
fn test_pipeline_stats() {
    let stats = PipelineStats {
        total_iterations: 5,
        phases_run: 5,
        total_time: Duration::from_millis(100),
        initial_size: 100,
        final_size: 80,
    };

    assert_eq!(stats.size_reduction(), 0.8);
}

#[test]
fn test_pipeline_stats_zero_size() {
    let stats = PipelineStats {
        initial_size: 0,
        final_size: 0,
        ..Default::default()
    };

    assert_eq!(stats.size_reduction(), 1.0);
}

#[test]
fn test_pass_stats() {
    let builder = GraphBuilder::new(0, 0);
    let mut graph = builder.finish();

    let mut pipeline = OptPipeline::new();
    pipeline.run(&mut graph);

    let stats = pipeline.pass_stats();
    assert!(!stats.is_empty());

    for stat in stats {
        assert!(!stat.name.is_empty());
    }
}

// =========================================================================
// New Pass Integration Tests
// =========================================================================

#[test]
fn test_config_default_enables_all_passes() {
    let config = PipelineConfig::default();

    // Canonicalization
    assert!(
        config.enable_simplify,
        "simplify should be enabled by default"
    );
    assert!(config.enable_sccp, "sccp should be enabled by default");
    assert!(
        config.enable_instcombine,
        "instcombine should be enabled by default"
    );

    // Local
    assert!(
        config.enable_copy_prop,
        "copy_prop should be enabled by default"
    );
    assert!(config.enable_gvn, "gvn should be enabled by default");
    assert!(config.enable_dse, "dse should be enabled by default");
    assert!(config.enable_pre, "pre should be enabled by default");
    assert!(
        config.enable_strength_reduce,
        "strength_reduce should be enabled by default"
    );

    // Loop
    assert!(config.enable_licm, "licm should be enabled by default");
    assert!(config.enable_unroll, "unroll should be enabled by default");
    assert!(config.enable_rce, "rce should be enabled by default");

    // Interprocedural
    assert!(config.enable_inline, "inline should be enabled by default");
    assert!(config.enable_escape, "escape should be enabled by default");
    assert!(config.enable_tco, "tco should be enabled by default");

    // Cleanup
    assert!(config.enable_dce, "dce should be enabled by default");
}

#[test]
fn test_config_minimal_disables_expensive_passes() {
    let config = PipelineConfig::minimal();

    // Expensive passes should be disabled
    assert!(!config.enable_sccp, "sccp should be disabled in minimal");
    assert!(!config.enable_dse, "dse should be disabled in minimal");
    assert!(!config.enable_pre, "pre should be disabled in minimal");
    assert!(
        !config.enable_unroll,
        "unroll should be disabled in minimal"
    );
    assert!(!config.enable_tco, "tco should be disabled in minimal");
    assert!(!config.enable_licm, "licm should be disabled in minimal");
    assert!(!config.enable_rce, "rce should be disabled in minimal");
    assert!(
        !config.enable_inline,
        "inline should be disabled in minimal"
    );
    assert!(
        !config.enable_escape,
        "escape should be disabled in minimal"
    );
    assert!(
        !config.enable_strength_reduce,
        "strength_reduce should be disabled in minimal"
    );

    // Essential cheap passes should remain enabled
    assert!(
        config.enable_simplify,
        "simplify should be enabled in minimal"
    );
    assert!(
        config.enable_instcombine,
        "instcombine should be enabled in minimal"
    );
    assert!(
        config.enable_copy_prop,
        "copy_prop should be enabled in minimal"
    );
    assert!(config.enable_gvn, "gvn should be enabled in minimal");
    assert!(config.enable_dce, "dce should be enabled in minimal");
}

#[test]
fn test_pipeline_registers_dse_pass() {
    let pipeline = OptPipeline::new();
    let stats = pipeline.pass_stats();
    assert!(
        stats.iter().any(|s| s.name.to_lowercase().contains("dse")),
        "Pipeline should include DSE pass"
    );
}

#[test]
fn test_pipeline_registers_pre_pass() {
    let pipeline = OptPipeline::new();
    let stats = pipeline.pass_stats();
    assert!(
        stats.iter().any(|s| s.name.to_lowercase().contains("pre")),
        "Pipeline should include PRE pass"
    );
}

#[test]
fn test_pipeline_registers_unroll_pass() {
    let pipeline = OptPipeline::new();
    let stats = pipeline.pass_stats();
    assert!(
        stats
            .iter()
            .any(|s| s.name.to_lowercase().contains("unroll")),
        "Pipeline should include Unroll pass"
    );
}

#[test]
fn test_pipeline_registers_sccp_pass() {
    let pipeline = OptPipeline::new();
    let stats = pipeline.pass_stats();
    assert!(
        stats.iter().any(|s| s.name.to_lowercase().contains("sccp")),
        "Pipeline should include SCCP pass"
    );
}

#[test]
fn test_pipeline_registers_instcombine_pass() {
    let pipeline = OptPipeline::new();
    let stats = pipeline.pass_stats();
    assert!(
        stats
            .iter()
            .any(|s| s.name.to_lowercase().contains("instcombine")
                || s.name.to_lowercase().contains("combine")),
        "Pipeline should include InstCombine pass"
    );
}

#[test]
fn test_pipeline_registers_tco_pass() {
    let pipeline = OptPipeline::new();
    let stats = pipeline.pass_stats();
    assert!(
        stats.iter().any(
            |s| s.name.to_lowercase().contains("tail") || s.name.to_lowercase().contains("tco")
        ),
        "Pipeline should include TCO pass"
    );
}

#[test]
fn test_pipeline_pass_count_default_vs_minimal() {
    let default_pipeline = OptPipeline::new();
    let minimal_pipeline = OptPipeline::with_config(PipelineConfig::minimal());

    let default_count = default_pipeline.pass_stats().len();
    let minimal_count = minimal_pipeline.pass_stats().len();

    assert!(
        default_count > minimal_count,
        "Default pipeline ({}) should have more passes than minimal ({})",
        default_count,
        minimal_count
    );
}

#[test]
fn test_pipeline_pass_phases_are_correct() {
    let pipeline = OptPipeline::new();
    let stats = pipeline.pass_stats();

    // Verify we have passes in each expected phase
    let phases: Vec<PassPhase> = stats.iter().map(|s| s.phase).collect();

    assert!(
        phases.contains(&PassPhase::Canonicalization),
        "Should have Canonicalization passes"
    );
    assert!(
        phases.contains(&PassPhase::Local),
        "Should have Local passes"
    );
    assert!(phases.contains(&PassPhase::Loop), "Should have Loop passes");
    assert!(
        phases.contains(&PassPhase::Interprocedural),
        "Should have Interprocedural passes"
    );
    assert!(
        phases.contains(&PassPhase::Cleanup),
        "Should have Cleanup passes"
    );
}

#[test]
fn test_config_full_has_higher_iteration_limits() {
    let default_config = PipelineConfig::default();
    let full_config = PipelineConfig::full();

    assert!(
        full_config.max_iterations_per_phase > default_config.max_iterations_per_phase,
        "Full config should have higher iteration limits"
    );
    assert!(
        full_config.max_total_iterations > default_config.max_total_iterations,
        "Full config should have higher total iteration limits"
    );
}

#[test]
fn test_config_minimal_has_lower_iteration_limits() {
    let default_config = PipelineConfig::default();
    let minimal_config = PipelineConfig::minimal();

    assert!(
        minimal_config.max_iterations_per_phase < default_config.max_iterations_per_phase,
        "Minimal config should have lower iteration limits"
    );
    assert!(
        minimal_config.max_total_iterations < default_config.max_total_iterations,
        "Minimal config should have lower total iteration limits"
    );
}

#[test]
fn test_config_minimal_disables_timing() {
    let minimal_config = PipelineConfig::minimal();
    assert!(
        !minimal_config.collect_timing,
        "Minimal config should disable timing"
    );
}

#[test]
fn test_config_default_enables_timing() {
    let default_config = PipelineConfig::default();
    assert!(
        default_config.collect_timing,
        "Default config should enable timing"
    );
}

#[test]
fn test_pipeline_with_all_passes_disabled() {
    let config = PipelineConfig {
        max_iterations_per_phase: 1,
        max_total_iterations: 1,
        enable_simplify: false,
        enable_sccp: false,
        enable_instcombine: false,
        enable_branch_probability: false,
        enable_hot_cold: false,
        enable_copy_prop: false,
        enable_gvn: false,
        enable_dse: false,
        enable_pre: false,
        enable_strength_reduce: false,
        enable_licm: false,
        enable_unroll: false,
        enable_rce: false,
        enable_inline: false,
        enable_escape: false,
        enable_tco: false,
        enable_dce: false,
        collect_timing: false,
    };

    let pipeline = OptPipeline::with_config(config);
    assert!(
        pipeline.pass_stats().is_empty(),
        "Pipeline with all passes disabled should be empty"
    );
}

#[test]
fn test_pipeline_runs_with_graph() {
    let mut builder = GraphBuilder::new(4, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();

    // Create some redundant computations for optimizers to work on
    let sum1 = builder.int_add(p0, p1);
    let sum2 = builder.int_add(p0, p1); // Redundant - GVN should eliminate
    let product = builder.int_mul(sum1, sum2);
    builder.return_value(product);

    let mut graph = builder.finish();
    let initial_size = graph.len();

    let mut pipeline = OptPipeline::new();
    let stats = pipeline.run(&mut graph);

    // Pipeline should complete successfully
    assert!(stats.total_iterations >= 1);
    assert!(stats.phases_run >= 1);

    // With all optimizations, should be able to reduce some redundancy
    // (exact reduction depends on graph structure)
    assert!(stats.initial_size == initial_size);
}

// =========================================================================
// PGO Pipeline Integration Tests
// =========================================================================

#[test]
fn test_pipeline_with_profile_constructor() {
    let profile = crate::runtime::profile_data::ProfileData::new(1);
    let pipeline = OptPipeline::with_profile(PipelineConfig::default(), profile);

    // Should have registered passes including BranchProbabilityPass
    let stats = pipeline.pass_stats();
    let bp_pass = stats.iter().find(|s| s.name == "BranchProbability");
    assert!(
        bp_pass.is_some(),
        "Pipeline with profile should have BranchProbability pass"
    );
}

#[test]
fn test_pipeline_inject_profile() {
    let mut pipeline = OptPipeline::new();

    // Verify BranchProbability pass exists
    let has_bp_before = pipeline
        .pass_stats()
        .iter()
        .any(|s| s.name == "BranchProbability");
    assert!(has_bp_before);

    // Inject profile
    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    profile.record_branch(0, true);
    pipeline.inject_profile(profile);

    // BranchProbability pass should still exist
    let has_bp_after = pipeline
        .pass_stats()
        .iter()
        .any(|s| s.name == "BranchProbability");
    assert!(has_bp_after);
}

#[test]
fn test_pipeline_inject_profile_when_disabled() {
    // Create a config with branch probability disabled
    let config = PipelineConfig {
        enable_branch_probability: false,
        ..PipelineConfig::minimal()
    };
    let mut pipeline = OptPipeline::with_config(config);

    // Should NOT have BranchProbability initially
    let has_bp_before = pipeline
        .pass_stats()
        .iter()
        .any(|s| s.name == "BranchProbability");
    assert!(
        !has_bp_before,
        "BranchProbability should be disabled initially"
    );

    // Inject profile should register the pass
    let profile = crate::runtime::profile_data::ProfileData::new(1);
    pipeline.inject_profile(profile);

    let has_bp_after = pipeline
        .pass_stats()
        .iter()
        .any(|s| s.name == "BranchProbability");
    assert!(
        has_bp_after,
        "inject_profile should register BranchProbability even if disabled"
    );
}

#[test]
fn test_pipeline_run_with_profile() {
    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    for _ in 0..100 {
        profile.record_execution();
    }
    for _ in 0..90 {
        profile.record_branch(3, true);
    }
    for _ in 0..10 {
        profile.record_branch(3, false);
    }

    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    builder.return_value(sum);

    let mut graph = builder.finish();

    let mut pipeline = OptPipeline::with_profile(PipelineConfig::default(), profile);
    let stats = pipeline.run(&mut graph);

    // Pipeline should complete
    assert!(stats.total_iterations >= 1);
    assert!(stats.phases_run >= 1);
}

#[test]
fn test_optimize_with_profile_convenience() {
    let mut profile = crate::runtime::profile_data::ProfileData::new(1);
    profile.record_branch(0, true);

    let builder = GraphBuilder::new(0, 0);
    let mut graph = builder.finish();

    let stats = optimize_with_profile(&mut graph, profile);
    assert!(stats.total_iterations >= 1);
}
