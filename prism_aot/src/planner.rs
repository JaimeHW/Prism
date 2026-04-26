use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use prism_compiler::{OptimizationLevel, SourceCompileError, compile_source_module};

use crate::bundle::CodeImage;
use crate::error::AotError;
use crate::imports::collect_static_imports;
use crate::native::NativeModuleInitPlan;

/// Initial entrypoint for an AOT build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildEntry {
    /// Build from a concrete script path.
    Script(PathBuf),
    /// Build from a module or package name.
    Module(String),
}

/// Compiler options for whole-program AOT planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildOptions {
    /// Additional search paths used to resolve imports.
    pub search_paths: Vec<PathBuf>,
    /// Optimization level forwarded to the bytecode compiler.
    pub optimize: OptimizationLevel,
    /// Target identifier for downstream native code generation.
    pub target: String,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            search_paths: default_search_paths(),
            optimize: OptimizationLevel::None,
            target: default_target_triple(),
        }
    }
}

/// Planned entrypoint metadata for native bootstrapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedEntry {
    /// Invocation kind: `script` or `module`.
    pub invocation_kind: String,
    /// Invocation value passed to the compiler driver.
    pub invocation_value: String,
    /// Canonical module name in the build graph.
    pub canonical_module: String,
    /// Runtime execution name.
    pub execution_name: String,
    /// Package context for relative imports.
    pub package_name: String,
    /// Source file backing the entrypoint.
    pub source_path: PathBuf,
}

/// Module classification inside the AOT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    /// Source-backed Python module compiled from `.py`.
    Source,
    /// Built-in stdlib module provided by Prism.
    Stdlib,
}

/// One compiled module in the whole-program build graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedModule {
    /// Canonical module name.
    pub name: String,
    /// Module origin.
    pub kind: ModuleKind,
    /// Path to the source file, when source-backed.
    pub source_path: Option<PathBuf>,
    /// Package context for relative imports.
    pub package_name: String,
    /// Whether the module is backed by `__init__.py`.
    pub is_package: bool,
    /// Bytecode instruction count.
    pub instruction_count: Option<usize>,
    /// Constant pool size.
    pub constant_count: Option<usize>,
    /// Nested code object count.
    pub nested_code_object_count: Option<usize>,
    /// Static imports discovered syntactically in the module.
    pub static_imports: Vec<String>,
    /// Candidate submodules referenced by `from ... import ...`.
    pub from_import_candidates: Vec<String>,
    /// Deterministic serialized code image for future native lowering.
    pub code_image: Option<CodeImage>,
    /// Native top-level init plan when the source module fits the current AOT subset.
    pub native_init: Option<NativeModuleInitPlan>,
    /// Lowering diagnostic when native init emission is not yet supported.
    pub native_init_diagnostic: Option<String>,
}

/// Entire build plan for a standalone native executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildPlan {
    /// Manifest schema version.
    pub format_version: u32,
    /// Target identifier.
    pub target: String,
    /// Entrypoint metadata.
    pub entry: PlannedEntry,
    /// Deterministic module list.
    pub modules: Vec<PlannedModule>,
    optimize: OptimizationLevel,
}

impl BuildPlan {
    /// Convert the optimization level into a stable manifest label.
    pub fn optimization_label(&self) -> String {
        match self.optimize {
            OptimizationLevel::None => "none",
            OptimizationLevel::Basic => "basic",
            OptimizationLevel::Full => "full",
        }
        .to_string()
    }
}

/// Planner for Prism's whole-program native build pipeline.
pub struct BuildPlanner {
    options: BuildOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedSource {
    module_name: String,
    path: PathBuf,
    package_name: String,
    is_package: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedEntry {
    invocation_kind: String,
    invocation_value: String,
    canonical_module: String,
    execution_name: String,
    package_name: String,
    source_path: PathBuf,
    source: ResolvedSource,
}

impl BuildPlanner {
    /// Create a new planner with explicit build options.
    pub fn new(options: BuildOptions) -> Self {
        Self { options }
    }

    /// Plan a whole-program build from the provided entrypoint.
    pub fn plan(&self, entry: BuildEntry) -> Result<BuildPlan, AotError> {
        let resolved_entry = self.resolve_entry(entry)?;
        let effective_search_paths = self.effective_search_paths(&resolved_entry);
        let mut planned_modules = BTreeMap::<String, PlannedModule>::new();
        let mut pending = VecDeque::<String>::new();
        let mut queued = BTreeSet::<String>::new();

        let entry_source = self.compile_resolved_source(&resolved_entry.source)?;
        enqueue_with_prefixes(&mut pending, &mut queued, &entry_source.name);
        enqueue_many(&mut pending, &mut queued, &entry_source.static_imports);
        self.enqueue_candidate_modules(
            &mut pending,
            &mut queued,
            &entry_source.from_import_candidates,
            &effective_search_paths,
        );
        planned_modules.insert(entry_source.name.clone(), entry_source);

        while let Some(module_name) = pending.pop_front() {
            if planned_modules.contains_key(&module_name) {
                continue;
            }

            if prism_stdlib::is_native_stdlib_module(&module_name) {
                planned_modules.insert(
                    module_name.clone(),
                    PlannedModule {
                        name: module_name,
                        kind: ModuleKind::Stdlib,
                        source_path: None,
                        package_name: String::new(),
                        is_package: false,
                        instruction_count: None,
                        constant_count: None,
                        nested_code_object_count: None,
                        static_imports: Vec::new(),
                        from_import_candidates: Vec::new(),
                        code_image: None,
                        native_init: None,
                        native_init_diagnostic: None,
                    },
                );
                continue;
            }

            let resolved = self
                .resolve_dependency_in_paths(&module_name, &effective_search_paths)
                .ok_or_else(|| AotError::module_not_found(module_name.clone(), None))?;
            let compiled = self.compile_resolved_source(&resolved)?;
            enqueue_many(&mut pending, &mut queued, &compiled.static_imports);
            self.enqueue_candidate_modules(
                &mut pending,
                &mut queued,
                &compiled.from_import_candidates,
                &effective_search_paths,
            );
            planned_modules.insert(compiled.name.clone(), compiled);
        }

        Ok(BuildPlan {
            format_version: 1,
            target: self.options.target.clone(),
            entry: PlannedEntry {
                invocation_kind: resolved_entry.invocation_kind,
                invocation_value: resolved_entry.invocation_value,
                canonical_module: resolved_entry.canonical_module,
                execution_name: resolved_entry.execution_name,
                package_name: resolved_entry.package_name,
                source_path: resolved_entry.source_path,
            },
            modules: planned_modules.into_values().collect(),
            optimize: self.options.optimize,
        })
    }

    fn resolve_entry(&self, entry: BuildEntry) -> Result<ResolvedEntry, AotError> {
        match entry {
            BuildEntry::Script(path) => {
                let path = canonicalize_or_clone(&path);
                if !path.is_file() {
                    return Err(AotError::InvalidEntrypoint {
                        message: format!("script entry '{}' does not exist", path.display()),
                    });
                }

                Ok(ResolvedEntry {
                    invocation_kind: "script".to_string(),
                    invocation_value: path.display().to_string(),
                    canonical_module: "__main__".to_string(),
                    execution_name: "__main__".to_string(),
                    package_name: String::new(),
                    source_path: path.clone(),
                    source: ResolvedSource {
                        module_name: "__main__".to_string(),
                        path,
                        package_name: String::new(),
                        is_package: false,
                    },
                })
            }
            BuildEntry::Module(name) => {
                let entry = self.resolve_module_entry(&name).ok_or_else(|| {
                    AotError::module_not_found(name.clone(), Some("compiler entrypoint"))
                })?;

                Ok(ResolvedEntry {
                    invocation_kind: "module".to_string(),
                    invocation_value: name,
                    canonical_module: entry.module_name.clone(),
                    execution_name: "__main__".to_string(),
                    package_name: entry.package_name.clone(),
                    source_path: entry.path.clone(),
                    source: entry,
                })
            }
        }
    }

    fn resolve_module_entry(&self, module: &str) -> Option<ResolvedSource> {
        let parts: Vec<&str> = module.split('.').collect();
        if parts.is_empty()
            || parts
                .iter()
                .any(|segment| !is_valid_module_segment(segment))
        {
            return None;
        }

        for base in &self.options.search_paths {
            let mut module_base = base.clone();
            for part in &parts {
                module_base.push(part);
            }

            let module_file = module_base.with_extension("py");
            if module_file.is_file() {
                return Some(ResolvedSource {
                    module_name: module.to_string(),
                    path: canonicalize_or_clone(&module_file),
                    package_name: parent_package(module).unwrap_or("").to_string(),
                    is_package: false,
                });
            }

            let package_main = module_base.join("__main__.py");
            if package_main.is_file() {
                return Some(ResolvedSource {
                    module_name: format!("{module}.__main__"),
                    path: canonicalize_or_clone(&package_main),
                    package_name: module.to_string(),
                    is_package: false,
                });
            }
        }

        None
    }

    fn resolve_dependency_in_paths(
        &self,
        module: &str,
        search_paths: &[PathBuf],
    ) -> Option<ResolvedSource> {
        let parts: Vec<&str> = module.split('.').collect();
        if parts.is_empty()
            || parts
                .iter()
                .any(|segment| !is_valid_module_segment(segment))
        {
            return None;
        }

        for base in search_paths {
            if parts.len() == 1 {
                let package_init = base.join(parts[0]).join("__init__.py");
                if package_init.is_file() {
                    return Some(ResolvedSource {
                        module_name: module.to_string(),
                        path: canonicalize_or_clone(&package_init),
                        package_name: module.to_string(),
                        is_package: true,
                    });
                }

                let module_file = base.join(format!("{}.py", parts[0]));
                if module_file.is_file() {
                    return Some(ResolvedSource {
                        module_name: module.to_string(),
                        path: canonicalize_or_clone(&module_file),
                        package_name: String::new(),
                        is_package: false,
                    });
                }

                continue;
            }

            let mut dir = base.clone();
            let mut valid_prefix = true;
            for part in &parts[..parts.len() - 1] {
                dir.push(part);
                if !dir.join("__init__.py").is_file() {
                    valid_prefix = false;
                    break;
                }
            }

            if !valid_prefix {
                continue;
            }

            let leaf = parts.last().expect("dotted module missing leaf");
            let package_init = dir.join(leaf).join("__init__.py");
            if package_init.is_file() {
                return Some(ResolvedSource {
                    module_name: module.to_string(),
                    path: canonicalize_or_clone(&package_init),
                    package_name: module.to_string(),
                    is_package: true,
                });
            }

            let module_file = dir.join(format!("{leaf}.py"));
            if module_file.is_file() {
                return Some(ResolvedSource {
                    module_name: module.to_string(),
                    path: canonicalize_or_clone(&module_file),
                    package_name: parent_package(module).unwrap_or("").to_string(),
                    is_package: false,
                });
            }
        }

        None
    }

    fn compile_resolved_source(
        &self,
        resolved: &ResolvedSource,
    ) -> Result<PlannedModule, AotError> {
        let source = std::fs::read_to_string(&resolved.path).map_err(|err| AotError::Io {
            path: resolved.path.clone(),
            message: err.to_string(),
        })?;
        let compilation = compile_source_module(
            &source,
            &resolved.path.display().to_string(),
            self.options.optimize,
        )
        .map_err(|err| match err {
            SourceCompileError::Parse(err) => AotError::Parse {
                module: resolved.module_name.clone(),
                path: resolved.path.clone(),
                message: err.to_string(),
            },
            SourceCompileError::Compile(err) => AotError::Compile {
                module: resolved.module_name.clone(),
                path: resolved.path.clone(),
                message: err.to_string(),
            },
        })?;
        let static_imports = collect_static_imports(&compilation.module, &resolved.package_name)?;
        let code_image =
            CodeImage::from_code_object(&resolved.module_name, compilation.code.as_ref())?;
        let (native_init, native_init_diagnostic) =
            match NativeModuleInitPlan::lower(&resolved.module_name, &compilation.module) {
                Ok(plan) => (Some(plan), None),
                Err(err) => (None, Some(err.to_string())),
            };

        Ok(PlannedModule {
            name: resolved.module_name.clone(),
            kind: ModuleKind::Source,
            source_path: Some(resolved.path.clone()),
            package_name: resolved.package_name.clone(),
            is_package: resolved.is_package,
            instruction_count: Some(compilation.code.instructions.len()),
            constant_count: Some(compilation.code.constants.len()),
            nested_code_object_count: Some(compilation.code.nested_code_objects.len()),
            static_imports: static_imports.required_modules,
            from_import_candidates: static_imports.from_import_candidates,
            code_image: Some(code_image),
            native_init,
            native_init_diagnostic,
        })
    }

    fn enqueue_candidate_modules(
        &self,
        pending: &mut VecDeque<String>,
        queued: &mut BTreeSet<String>,
        candidates: &[String],
        search_paths: &[PathBuf],
    ) {
        for candidate in candidates {
            if self.module_exists(candidate, search_paths) {
                enqueue_with_prefixes(pending, queued, candidate);
            }
        }
    }

    fn module_exists(&self, module: &str, search_paths: &[PathBuf]) -> bool {
        prism_stdlib::is_native_stdlib_module(module)
            || self
                .resolve_dependency_in_paths(module, search_paths)
                .is_some()
    }

    fn effective_search_paths(&self, entry: &ResolvedEntry) -> Vec<PathBuf> {
        let mut search_paths = Vec::new();

        if entry.invocation_kind == "script" {
            if let Some(parent) = entry.source_path.parent() {
                push_unique_path(&mut search_paths, canonicalize_or_clone(parent));
            }
        }

        for path in &self.options.search_paths {
            push_unique_path(&mut search_paths, canonicalize_or_clone(path));
        }

        search_paths
    }
}

fn enqueue_many(pending: &mut VecDeque<String>, queued: &mut BTreeSet<String>, imports: &[String]) {
    for import in imports {
        enqueue_with_prefixes(pending, queued, import);
    }
}

fn enqueue_with_prefixes(
    pending: &mut VecDeque<String>,
    queued: &mut BTreeSet<String>,
    module_name: &str,
) {
    let mut prefix = String::new();
    for (index, segment) in module_name.split('.').enumerate() {
        if index > 0 {
            prefix.push('.');
        }
        prefix.push_str(segment);
        if queued.insert(prefix.clone()) {
            pending.push_back(prefix.clone());
        }
    }
}

fn default_target_triple() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

fn default_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd);
    }

    if let Some(pythonpath) = std::env::var_os("PYTHONPATH") {
        for path in std::env::split_paths(&pythonpath) {
            push_unique_path(&mut paths, path);
        }
    }

    paths
}

fn push_unique_path(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|existing| existing == &candidate) {
        paths.push(candidate);
    }
}

fn parent_package(name: &str) -> Option<&str> {
    name.rsplit_once('.').map(|(parent, _)| parent)
}

fn canonicalize_or_clone(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn is_valid_module_segment(segment: &str) -> bool {
    let mut chars = segment.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }

    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}
