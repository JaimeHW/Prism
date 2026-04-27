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

    /// Isolated mode (`-I`, implies `-E` and `-s`).
    pub isolated: bool,

    /// Don't add user site-packages to `sys.path` (`-s`).
    pub no_user_site: bool,

    /// Don't import the `site` module on initialization (`-S`).
    pub no_site: bool,

    /// Warning filters in order of specification.
    pub warnings: Vec<WarningFilter>,

    /// Implementation-specific `-X` options.
    pub x_options: Vec<String>,

    /// Deterministic interpreter step budget (`-X max-steps=N`).
    pub execution_step_limit: Option<u64>,

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
        let isolated = args.isolated;
        let ignore_env = args.ignore_environment || isolated;

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
        let no_user_site =
            args.no_user_site || isolated || (!ignore_env && Self::env_bool("PYTHONNOUSERSITE"));

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
            isolated,
            no_user_site,
            no_site: args.no_site,
            warnings: args.warnings.clone(),
            x_options: args.x_options.clone(),
            execution_step_limit: Self::parse_execution_step_limit(&args.x_options),
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

    fn parse_execution_step_limit(x_options: &[String]) -> Option<u64> {
        let mut limit = None;

        for opt in x_options {
            let value = opt
                .strip_prefix("max-steps=")
                .or_else(|| opt.strip_prefix("max_steps="));

            if let Some(raw) = value {
                limit = raw.parse::<u64>().ok().filter(|parsed| *parsed > 0);
            }
        }

        limit
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
