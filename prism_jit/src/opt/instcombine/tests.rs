use super::*;

// =========================================================================
// InstCombineStats Tests
// =========================================================================

#[test]
fn test_stats_default() {
    let stats = InstCombineStats::default();
    assert_eq!(stats.patterns_applied, 0);
    assert_eq!(stats.instructions_eliminated, 0);
}

#[test]
fn test_stats_net_reduction_positive() {
    let stats = InstCombineStats {
        patterns_applied: 1,
        instructions_eliminated: 5,
        instructions_simplified: 2,
        instructions_created: 2,
        instructions_analyzed: 10,
    };
    assert_eq!(stats.net_reduction(), 3);
}

#[test]
fn test_stats_net_reduction_negative() {
    let stats = InstCombineStats {
        patterns_applied: 1,
        instructions_eliminated: 1,
        instructions_simplified: 0,
        instructions_created: 3,
        instructions_analyzed: 10,
    };
    assert_eq!(stats.net_reduction(), -2);
}

#[test]
fn test_stats_merge() {
    let mut stats1 = InstCombineStats {
        patterns_applied: 3,
        instructions_eliminated: 5,
        instructions_simplified: 2,
        instructions_created: 1,
        instructions_analyzed: 20,
    };
    let stats2 = InstCombineStats {
        patterns_applied: 2,
        instructions_eliminated: 3,
        instructions_simplified: 1,
        instructions_created: 1,
        instructions_analyzed: 10,
    };
    stats1.merge(&stats2);
    assert_eq!(stats1.patterns_applied, 5);
    assert_eq!(stats1.instructions_eliminated, 8);
    assert_eq!(stats1.instructions_analyzed, 30);
}

// =========================================================================
// InstCombineConfig Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = InstCombineConfig::default();
    assert_eq!(config.max_iterations, 10000);
    assert!(config.enable_arithmetic);
    assert!(config.enable_bitwise);
    assert!(config.enable_comparison);
    assert!(config.enable_memory);
    assert!(config.enable_control);
}

#[test]
fn test_config_custom() {
    let config = InstCombineConfig {
        max_iterations: 100,
        enable_arithmetic: false,
        enable_bitwise: true,
        enable_comparison: true,
        enable_memory: false,
        enable_control: false,
    };
    assert_eq!(config.max_iterations, 100);
    assert!(!config.enable_arithmetic);
}

// =========================================================================
// InstCombine Pass Tests
// =========================================================================

#[test]
fn test_instcombine_new() {
    let ic = InstCombine::new();
    assert_eq!(ic.stats().patterns_applied, 0);
}

#[test]
fn test_instcombine_with_config() {
    let config = InstCombineConfig {
        max_iterations: 500,
        ..Default::default()
    };
    let ic = InstCombine::with_config(config);
    assert_eq!(ic.config().max_iterations, 500);
}

#[test]
fn test_instcombine_name() {
    let ic = InstCombine::new();
    assert_eq!(ic.name(), "instcombine");
}

#[test]
fn test_instcombine_empty_graph() {
    let mut graph = Graph::new();
    let mut ic = InstCombine::new();
    let changed = ic.run(&mut graph);
    assert!(!changed);
}

#[test]
fn test_instcombine_simple_graph() {
    let mut graph = Graph::new();
    graph.const_int(42);

    let mut ic = InstCombine::new();
    ic.run(&mut graph);

    assert!(ic.stats().instructions_analyzed >= 1);
}

#[test]
fn test_is_pattern_enabled() {
    let mut config = InstCombineConfig::default();
    config.enable_arithmetic = false;

    let ic = InstCombine::with_config(config);

    // Create a dummy arithmetic pattern
    let pattern = Pattern::new(PatternCategory::Arithmetic);
    assert!(!ic.is_pattern_enabled(&pattern));

    let pattern = Pattern::new(PatternCategory::Bitwise);
    assert!(ic.is_pattern_enabled(&pattern));
}
