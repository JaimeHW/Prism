//! Runtime configuration aggregated from CLI flags and environment variables.
//!
//! Mirrors CPython's `PyConfig` — a single struct that captures all runtime
//! settings, resolved once at startup for zero-cost access during execution.

use crate::args::{OptimizationLevel, PrismArgs, WarningFilter};

// =============================================================================
// Runtime Configuration
// =============================================================================

/// Complete runtime configuration resolved from CLI args + environment.
///
/// This struct is immutable after construction — all settings are resolved
/// once during startup. The VM and pipeline read from this without any
/// per-operation cost.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Optimization level (`-O`, `-OO`).
    pub optimize: OptimizationLevel,

    /// Whether to enter interactive mode after script execution (`-i`).
    pub inspect: bool,

    /// Force unbuffered stdout/stderr (`-u`).
    pub unbuffered: bool,

    /// Verbose import tracing level (`-v`, `-vv`, etc.).
    pub verbose: u32,

    /// Suppress version banner in REPL (`-q`).
    pub quiet: bool,

    /// Don't write `.pyc` files (`-B`). No-op for Prism.
    pub dont_write_bytecode: bool,

    /// Ignore `PYTHON*` environment variables (`-E`).
    pub ignore_environment: bool,

    /// Don't add user site-packages to `sys.path` (`-s`).
    pub no_user_site: bool,

    /// Don't import the `site` module on initialization (`-S`).
    pub no_site: bool,

    /// Warning filters in order of specification.
    pub warnings: Vec<WarningFilter>,

    /// Implementation-specific `-X` options.
    pub x_options: Vec<String>,

    /// Parser debugging mode (`-d`).
    pub debug: bool,

    /// Hash seed from environment (`PYTHONHASHSEED`).
    pub hash_seed: Option<u64>,
}

impl RuntimeConfig {
    /// Resolve configuration from parsed CLI args and environment variables.
    ///
    /// Environment variables are only consulted if `-E` was NOT specified.
    pub fn from_args(args: &PrismArgs) -> Self {
        let ignore_env = args.ignore_environment;

        // Resolve optimization level: CLI flag OR `PYTHONOPTIMIZE` env var.
        let optimize = if args.optimize != OptimizationLevel::None {
            args.optimize
        } else if !ignore_env {
            Self::env_optimize()
        } else {
            OptimizationLevel::None
        };

        // Resolve verbose: CLI flag OR `PYTHONVERBOSE` env var.
        let verbose = if args.verbose > 0 {
            args.verbose
        } else if !ignore_env {
            Self::env_verbose()
        } else {
            0
        };

        // Resolve unbuffered: CLI flag OR `PYTHONUNBUFFERED` env var.
        let unbuffered = args.unbuffered || (!ignore_env && Self::env_bool("PYTHONUNBUFFERED"));

        // Resolve dont_write_bytecode: CLI flag OR `PYTHONDONTWRITEBYTECODE` env var.
        let dont_write_bytecode =
            args.dont_write_bytecode || (!ignore_env && Self::env_bool("PYTHONDONTWRITEBYTECODE"));

        // Resolve inspect: CLI flag OR `PYTHONINSPECT` env var.
        let inspect = args.inspect || (!ignore_env && Self::env_bool("PYTHONINSPECT"));

        // Resolve no_user_site: CLI flag OR `PYTHONNOUSERSITE` env var.
        let no_user_site = args.no_user_site || (!ignore_env && Self::env_bool("PYTHONNOUSERSITE"));

        // Resolve hash seed from `PYTHONHASHSEED` env var.
        let hash_seed = if !ignore_env {
            Self::env_hash_seed()
        } else {
            None
        };

        Self {
            optimize,
            inspect,
            unbuffered,
            verbose,
            quiet: args.quiet,
            dont_write_bytecode,
            ignore_environment: ignore_env,
            no_user_site,
            no_site: args.no_site,
            warnings: args.warnings.clone(),
            x_options: args.x_options.clone(),
            debug: args.debug,
            hash_seed,
        }
    }

    /// Whether the VM should run with JIT enabled.
    ///
    /// Defaults to enabled. `-X` options can force behavior:
    /// - disable: `jit=off`, `jit=0`, `nojit`
    /// - enable: `jit=on`, `jit=1`
    pub fn jit_enabled(&self) -> bool {
        let mut enabled = true;

        for opt in &self.x_options {
            match opt.as_str() {
                "jit=off" | "jit=0" | "nojit" => enabled = false,
                "jit=on" | "jit=1" => enabled = true,
                _ => {}
            }
        }

        enabled
    }

    /// Check if an environment variable is set to a non-empty, truthy value.
    #[inline]
    fn env_bool(var: &str) -> bool {
        std::env::var(var)
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    }

    /// Get optimization level from `PYTHONOPTIMIZE` env var.
    fn env_optimize() -> OptimizationLevel {
        match std::env::var("PYTHONOPTIMIZE") {
            Ok(val) => match val.as_str() {
                "2" | "OO" => OptimizationLevel::Full,
                "1" | "O" | "" => OptimizationLevel::Basic,
                v if v.parse::<u32>().map(|n| n >= 2).unwrap_or(false) => OptimizationLevel::Full,
                v if v.parse::<u32>().map(|n| n >= 1).unwrap_or(false) => OptimizationLevel::Basic,
                _ => OptimizationLevel::None,
            },
            Err(_) => OptimizationLevel::None,
        }
    }

    /// Get verbose level from `PYTHONVERBOSE` env var.
    fn env_verbose() -> u32 {
        match std::env::var("PYTHONVERBOSE") {
            Ok(val) => val
                .parse::<u32>()
                .unwrap_or(if val.is_empty() { 0 } else { 1 }),
            Err(_) => 0,
        }
    }

    /// Get hash seed from `PYTHONHASHSEED` env var.
    fn env_hash_seed() -> Option<u64> {
        match std::env::var("PYTHONHASHSEED") {
            Ok(val) if val == "random" => None,
            Ok(val) => val.parse::<u64>().ok(),
            Err(_) => None,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
        assert!(!config.no_site);
        assert!(config.warnings.is_empty());
        assert!(config.x_options.is_empty());
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
    fn test_config_inherits_all_boolean_flags() {
        let args = PrismArgs {
            inspect: true,
            unbuffered: true,
            quiet: true,
            dont_write_bytecode: true,
            ignore_environment: true,
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
    fn test_config_clone() {
        let args = PrismArgs {
            optimize: OptimizationLevel::Basic,
            verbose: 2,
            quiet: true,
            ..Default::default()
        };
        let config = RuntimeConfig::from_args(&args);
        let config2 = config.clone();
        assert_eq!(config2.optimize, OptimizationLevel::Basic);
        assert_eq!(config2.verbose, 2);
        assert!(config2.quiet);
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
}
