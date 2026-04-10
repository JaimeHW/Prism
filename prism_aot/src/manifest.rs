use std::path::Path;

use serde::Serialize;

use crate::error::AotError;
use crate::planner::{BuildPlan, ModuleKind};

/// Serializable manifest for Prism's native build pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildManifest {
    /// Schema version for compatibility with future native backends.
    pub format_version: u32,
    /// Compilation target identifier.
    pub target: String,
    /// Optimization level selected for compilation.
    pub optimization: String,
    /// Entry module and bootstrap metadata.
    pub entry: EntryManifest,
    /// All compiled source and stdlib modules in deterministic order.
    pub modules: Vec<ModuleManifest>,
}

/// Entry metadata used by the native bootstrap stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryManifest {
    /// How the build was invoked.
    pub invocation: InvocationManifest,
    /// Canonical module name in the frozen module graph.
    pub canonical_module: String,
    /// Runtime execution name. Module and package entrypoints execute as `__main__`.
    pub execution_name: String,
    /// Package context seen by relative imports.
    pub package: String,
    /// Source file used for the entrypoint.
    pub source_path: String,
}

/// How the compiler was invoked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InvocationManifest {
    /// Invocation kind: `script` or `module`.
    pub kind: String,
    /// Invocation value: script path or module name.
    pub value: String,
}

/// One module in the deterministic AOT build graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModuleManifest {
    /// Canonical module name.
    pub name: String,
    /// Source or stdlib.
    pub kind: String,
    /// Backing source file path, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    /// `__package__` value for the module.
    pub package: String,
    /// Whether the module is backed by `__init__.py`.
    pub is_package: bool,
    /// Number of bytecode instructions generated for the module.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction_count: Option<usize>,
    /// Number of constants in the code object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constant_count: Option<usize>,
    /// Nested code object count for planning future native lowering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nested_code_object_count: Option<usize>,
    /// Static imports discovered syntactically in the module.
    pub static_imports: Vec<String>,
    /// Candidate submodules referenced through `from ... import ...`.
    pub from_import_candidates: Vec<String>,
    /// Whether the current native lowering subset can emit a module-init stub.
    pub native_init_supported: bool,
    /// Stable symbol name for the native init stub when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_init_symbol: Option<String>,
    /// Diagnostic describing why native lowering is not yet available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_init_diagnostic: Option<String>,
}

impl BuildManifest {
    /// Convert to pretty JSON.
    pub fn to_pretty_json(&self) -> Result<String, AotError> {
        serde_json::to_string_pretty(self).map_err(|err| AotError::InvalidEntrypoint {
            message: format!("failed to encode build manifest: {err}"),
        })
    }

    /// Write the manifest to disk, creating parent directories as needed.
    pub fn write_to_path(&self, path: &Path) -> Result<(), AotError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| AotError::Io {
                path: parent.to_path_buf(),
                message: err.to_string(),
            })?;
        }

        let json = self.to_pretty_json()?;
        std::fs::write(path, json).map_err(|err| AotError::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        })
    }
}

impl From<&BuildPlan> for BuildManifest {
    fn from(plan: &BuildPlan) -> Self {
        Self {
            format_version: plan.format_version,
            target: plan.target.clone(),
            optimization: plan.optimization_label(),
            entry: EntryManifest {
                invocation: InvocationManifest {
                    kind: plan.entry.invocation_kind.clone(),
                    value: plan.entry.invocation_value.clone(),
                },
                canonical_module: plan.entry.canonical_module.clone(),
                execution_name: plan.entry.execution_name.clone(),
                package: plan.entry.package_name.clone(),
                source_path: plan.entry.source_path.display().to_string(),
            },
            modules: plan
                .modules
                .iter()
                .map(|module| ModuleManifest {
                    name: module.name.clone(),
                    kind: match module.kind {
                        ModuleKind::Source => "source".to_string(),
                        ModuleKind::Stdlib => "stdlib".to_string(),
                    },
                    source_path: module
                        .source_path
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    package: module.package_name.clone(),
                    is_package: module.is_package,
                    instruction_count: module.instruction_count,
                    constant_count: module.constant_count,
                    nested_code_object_count: module.nested_code_object_count,
                    static_imports: module.static_imports.clone(),
                    from_import_candidates: module.from_import_candidates.clone(),
                    native_init_supported: module.native_init.is_some(),
                    native_init_symbol: module
                        .native_init
                        .as_ref()
                        .map(|plan| plan.symbol_name.clone()),
                    native_init_diagnostic: module.native_init_diagnostic.clone(),
                })
                .collect(),
        }
    }
}
