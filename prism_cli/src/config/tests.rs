use super::*;
use crate::args::PrismArgs;

#[test]
fn test_default_config_from_default_args() {
    let args = PrismArgs::default();
    // Set PYTHONINSPECT=0 to avoid env pollution.
    let config = RuntimeConfig::from_args(&args);
    assert_eq!(config.optimize, OptimizationLevel::None);
    assert!(!config.quiet);
    assert!(!config.debug);
    assert!(!config.isolated);
    assert!(!config.no_site);
    assert!(config.warnings.is_empty());
    assert!(config.x_options.is_empty());
    assert_eq!(config.execution_step_limit, None);
}

#[test]
fn test_config_inherits_optimize_from_args() {
    let args = PrismArgs {
        optimize: OptimizationLevel::Full,
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert_eq!(config.optimize, OptimizationLevel::Full);
}

#[test]
fn test_config_inherits_verbose_from_args() {
    let args = PrismArgs {
        verbose: 3,
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert_eq!(config.verbose, 3);
}

#[test]
fn test_config_ignore_environment_blocks_env_vars() {
    let args = PrismArgs {
        ignore_environment: true,
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert!(config.ignore_environment);
    // Even if env vars are set, they should be ignored.
    assert_eq!(config.hash_seed, None);
}

#[test]
fn test_config_isolated_mode_implies_environment_and_user_site_isolation() {
    let args = PrismArgs {
        isolated: true,
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert!(config.isolated);
    assert!(config.ignore_environment);
    assert!(config.no_user_site);
    assert_eq!(config.hash_seed, None);
}

#[test]
fn test_config_inherits_all_boolean_flags() {
    let args = PrismArgs {
        inspect: true,
        unbuffered: true,
        quiet: true,
        dont_write_bytecode: true,
        ignore_environment: true,
        isolated: true,
        no_user_site: true,
        no_site: true,
        debug: true,
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert!(config.inspect);
    assert!(config.unbuffered);
    assert!(config.quiet);
    assert!(config.dont_write_bytecode);
    assert!(config.ignore_environment);
    assert!(config.isolated);
    assert!(config.no_user_site);
    assert!(config.no_site);
    assert!(config.debug);
}

#[test]
fn test_config_inherits_warnings() {
    let args = PrismArgs {
        warnings: vec![
            WarningFilter {
                spec: "error".to_string(),
            },
            WarningFilter {
                spec: "ignore".to_string(),
            },
        ],
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert_eq!(config.warnings.len(), 2);
    assert_eq!(config.warnings[0].spec, "error");
    assert_eq!(config.warnings[1].spec, "ignore");
}

#[test]
fn test_config_inherits_x_options() {
    let args = PrismArgs {
        x_options: vec!["utf8".to_string(), "dev".to_string()],
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert_eq!(config.x_options, vec!["utf8", "dev"]);
}

#[test]
fn test_config_parses_execution_step_limit_from_x_options() {
    let args = PrismArgs {
        x_options: vec!["max-steps=4096".to_string(), "dev".to_string()],
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert_eq!(config.execution_step_limit, Some(4096));
}

#[test]
fn test_config_ignores_invalid_execution_step_limit_values() {
    let zero_args = PrismArgs {
        x_options: vec!["max-steps=0".to_string()],
        ..Default::default()
    };
    let invalid_args = PrismArgs {
        x_options: vec!["max_steps=abc".to_string()],
        ..Default::default()
    };

    assert_eq!(
        RuntimeConfig::from_args(&zero_args).execution_step_limit,
        None
    );
    assert_eq!(
        RuntimeConfig::from_args(&invalid_args).execution_step_limit,
        None
    );
}

#[test]
fn test_env_bool_empty_is_false() {
    // env_bool internally: empty string is considered false.
    // We can test the helper directly.
    assert!(!RuntimeConfig::env_bool("PRISM_TEST_NONEXISTENT_21398721"));
}

#[test]
fn test_jit_enabled_default_true() {
    let config = RuntimeConfig::from_args(&PrismArgs::default());
    assert!(config.jit_enabled());
}

#[test]
fn test_jit_enabled_can_be_disabled_with_x_option() {
    let args = PrismArgs {
        x_options: vec!["jit=off".to_string()],
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert!(!config.jit_enabled());
}

#[test]
fn test_jit_enabled_nojit_alias_disables() {
    let args = PrismArgs {
        x_options: vec!["nojit".to_string()],
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert!(!config.jit_enabled());
}

#[test]
fn test_jit_enabled_last_option_wins() {
    let args = PrismArgs {
        x_options: vec![
            "jit=off".to_string(),
            "jit=on".to_string(),
            "jit=0".to_string(),
            "jit=1".to_string(),
        ],
        ..Default::default()
    };
    let config = RuntimeConfig::from_args(&args);
    assert!(config.jit_enabled());
}
