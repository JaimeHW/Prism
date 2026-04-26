use std::path::PathBuf;

use thiserror::Error;

/// Errors raised while constructing or emitting an AOT build plan.
#[derive(Debug, Error)]
pub enum AotError {
    /// A filesystem read or write failed.
    #[error("I/O error for '{path}': {message}")]
    Io { path: PathBuf, message: String },

    /// A module could not be parsed.
    #[error("failed to parse module '{module}' from '{path}': {message}")]
    Parse {
        module: String,
        path: PathBuf,
        message: String,
    },

    /// A module could not be compiled to bytecode.
    #[error("failed to compile module '{module}' from '{path}': {message}")]
    Compile {
        module: String,
        path: PathBuf,
        message: String,
    },

    /// A module reference could not be resolved from the configured search paths.
    #[error("unable to resolve module '{module}'{importer_suffix}")]
    ModuleNotFound {
        module: String,
        importer_suffix: String,
    },

    /// The requested entrypoint is not valid.
    #[error("{message}")]
    InvalidEntrypoint { message: String },

    /// An AOT artifact could not be encoded or decoded.
    #[error("{message}")]
    InvalidArtifact { message: String },

    /// The requested target is not supported by the selected emit step.
    #[error("target '{target}' does not support {feature} yet")]
    UnsupportedTarget { target: String, feature: String },

    /// A module could not satisfy the selected native lowering policy.
    #[error("module '{module}' does not satisfy native policy '{policy}': {message}")]
    NativeLowering {
        /// Module that failed native lowering.
        module: String,
        /// Source path for diagnostics.
        path: PathBuf,
        /// Selected native policy.
        policy: String,
        /// Lowering diagnostic.
        message: String,
    },

    /// A native object or linker-facing artifact could not be emitted.
    #[error("failed to emit {artifact}: {message}")]
    EmitArtifact { artifact: String, message: String },
}

impl AotError {
    /// Construct a module resolution error with optional importer context.
    pub fn module_not_found(module: impl Into<String>, importer: Option<&str>) -> Self {
        let importer_suffix = importer
            .map(|name| format!(" imported from '{name}'"))
            .unwrap_or_default();
        Self::ModuleNotFound {
            module: module.into(),
            importer_suffix,
        }
    }
}
