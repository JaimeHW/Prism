//! Execution pipeline: source → parse → compile → VM → result.
//!
//! Provides the core execution functions used by every CLI mode
//! (script, command string, stdin, REPL).

use crate::args::OptimizationLevel as CliOptimizationLevel;
use crate::config::RuntimeConfig;
use crate::error;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct ModuleExecutionSpec {
    module_name: Arc<str>,
    package_name: Arc<str>,
    filename: Arc<str>,
}

impl ModuleExecutionSpec {
    fn main(filename: impl Into<Arc<str>>) -> Self {
        Self {
            module_name: Arc::from("__main__"),
            package_name: Arc::from(""),
            filename: filename.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedModuleEntry {
    path: PathBuf,
    package_name: Arc<str>,
}

// =============================================================================
// Public Pipeline Functions
// =============================================================================

/// Run a Python source file.
///
/// Reads the file, parses, compiles, and executes it through the VM.
/// Returns the process exit code.
pub fn run_file(path: &Path, config: &RuntimeConfig) -> ExitCode {
    let script_args = vec![path.display().to_string()];
    run_file_with_args(path, config, &script_args)
}

/// Run a Python source file with explicit `sys.argv` values.
pub fn run_file_with_args(path: &Path, config: &RuntimeConfig, script_args: &[String]) -> ExitCode {
    let filename = path.display().to_string();

    // Read source file.
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "prism: can't open file '{}': [Errno {}] {}",
                filename,
                e.raw_os_error().unwrap_or(0),
                e,
            );
            return ExitCode::from(error::EXIT_USAGE_ERROR);
        }
    };

    let search_paths = script_search_paths(path);
    execute_source_entry(
        &source,
        config,
        script_args,
        &search_paths,
        ModuleExecutionSpec::main(filename),
    )
}

/// Run a command string (from `-c` flag).
///
/// Parses and executes the string as if it were a module.
pub fn run_string(code: &str, config: &RuntimeConfig) -> ExitCode {
    let script_args = vec!["-c".to_string()];
    run_string_with_args(code, config, &script_args)
}

/// Run a command string with explicit `sys.argv` values.
pub fn run_string_with_args(
    code: &str,
    config: &RuntimeConfig,
    script_args: &[String],
) -> ExitCode {
    let search_paths = module_search_paths();
    execute_source_entry(
        code,
        config,
        script_args,
        &search_paths,
        ModuleExecutionSpec::main("<string>"),
    )
}

/// Run from stdin.
///
/// Reads all of stdin, then parses and executes.
pub fn run_stdin(config: &RuntimeConfig) -> ExitCode {
    let script_args = vec!["-".to_string()];
    run_stdin_with_args(config, &script_args)
}

/// Run from stdin with explicit `sys.argv` values.
pub fn run_stdin_with_args(config: &RuntimeConfig, script_args: &[String]) -> ExitCode {
    let mut source = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut source) {
        eprintln!("prism: error reading stdin: {}", e);
        return ExitCode::from(error::EXIT_ERROR);
    }
    let search_paths = module_search_paths();
    execute_source_entry(
        &source,
        config,
        script_args,
        &search_paths,
        ModuleExecutionSpec::main("<stdin>"),
    )
}

/// Run a module by dotted name (from `-m` flag).
///
/// Supports:
/// - `package.module` -> `<search_path>/package/module.py`
/// - `package` -> `<search_path>/package/__main__.py` (package entry point)
pub fn run_module(module: &str, config: &RuntimeConfig) -> ExitCode {
    run_module_with_args(module, config, &[])
}

/// Run a module by dotted name with explicit `sys.argv` values.
pub fn run_module_with_args(
    module: &str,
    config: &RuntimeConfig,
    script_args: &[String],
) -> ExitCode {
    let search_paths = module_search_paths();
    run_module_with_search_paths_with_args(module, config, &search_paths, script_args)
}

// =============================================================================
// Core Execution
// =============================================================================

/// Execute source code through the full pipeline.
///
/// Pipeline: parse → compile → VM execute.
///
/// Returns the process exit code (0 on success, 1 on error).
fn execute_source(source: &str, filename: &str, config: &RuntimeConfig) -> ExitCode {
    execute_source_with_args(source, filename, config, &[])
}

fn execute_source_with_args(
    source: &str,
    filename: &str,
    config: &RuntimeConfig,
    script_args: &[String],
) -> ExitCode {
    let search_paths = module_search_paths();
    execute_source_entry(
        source,
        config,
        script_args,
        &search_paths,
        ModuleExecutionSpec::main(filename),
    )
}

fn execute_source_entry(
    source: &str,
    config: &RuntimeConfig,
    script_args: &[String],
    search_paths: &[PathBuf],
    main_module: ModuleExecutionSpec,
) -> ExitCode {
    let optimize = compiler_optimization_level(config.optimize);
    let code = match prism_compiler::compile_source_code(
        source,
        main_module.filename.as_ref(),
        optimize,
    ) {
        Ok(c) => c,
        Err(e) => {
            return error::format_source_compile_error(
                &e,
                Some(source),
                main_module.filename.as_ref(),
            );
        }
    };

    // Phase 2: Execute.
    let mut vm = if config.jit_enabled() {
        prism_vm::VirtualMachine::with_jit()
    } else {
        prism_vm::VirtualMachine::new()
    };
    vm.set_source_optimization(vm_source_optimization_level(config.optimize));
    vm.set_import_verbosity(config.verbose);
    vm.set_execution_step_limit(config.execution_step_limit);
    vm.reset_imports_with_sys_args(script_args.to_vec());
    for search_path in search_paths {
        vm.add_import_search_path(Arc::<str>::from(search_path.to_string_lossy().into_owned()));
    }
    let filename_for_errors = Arc::clone(&main_module.filename);
    let main_module = Arc::new(prism_vm::imports::ModuleObject::with_metadata(
        Arc::clone(&main_module.module_name),
        None,
        Some(Arc::clone(&main_module.filename)),
        Some(Arc::clone(&main_module.package_name)),
    ));

    let execution_result = vm.execute_in_module_runtime(code, main_module);
    let shutdown_errors = vm.run_shutdown_hooks();

    let exit_code = match execution_result {
        Ok(_) => ExitCode::from(error::EXIT_SUCCESS),
        Err(e) => error::format_runtime_error(&e, Some(source), filename_for_errors.as_ref()),
    };

    emit_shutdown_errors(&shutdown_errors);
    exit_code
}

fn emit_shutdown_errors(errors: &[prism_vm::RuntimeError]) {
    for err in errors {
        eprintln!("Exception ignored in Prism shutdown:");
        eprint!(
            "{}",
            error::format_runtime_error_string(err, None, "<shutdown>")
        );
    }
}

fn run_module_with_search_paths(
    module: &str,
    config: &RuntimeConfig,
    search_paths: &[PathBuf],
) -> ExitCode {
    run_module_with_search_paths_with_args(module, config, search_paths, &[])
}

fn run_module_with_search_paths_with_args(
    module: &str,
    config: &RuntimeConfig,
    search_paths: &[PathBuf],
    script_args: &[String],
) -> ExitCode {
    match resolve_module_entry_in_search_paths(module, search_paths) {
        Some(entry) => {
            let module_argv = build_module_argv(&entry.path, script_args);
            let source = match std::fs::read_to_string(&entry.path) {
                Ok(source) => source,
                Err(err) => {
                    eprintln!(
                        "prism: can't open file '{}': [Errno {}] {}",
                        entry.path.display(),
                        err.raw_os_error().unwrap_or(0),
                        err,
                    );
                    return ExitCode::from(error::EXIT_USAGE_ERROR);
                }
            };

            execute_source_entry(
                &source,
                config,
                &module_argv,
                search_paths,
                ModuleExecutionSpec {
                    module_name: Arc::from("__main__"),
                    package_name: entry.package_name,
                    filename: Arc::from(entry.path.display().to_string()),
                },
            )
        }
        None => {
            eprintln!("prism: No module named '{}'", module);
            ExitCode::from(error::EXIT_ERROR)
        }
    }
}

fn build_module_argv(module_path: &Path, script_args: &[String]) -> Vec<String> {
    let mut argv = Vec::with_capacity(script_args.len().max(1));
    argv.push(module_path.display().to_string());
    if script_args.len() > 1 {
        argv.extend(script_args[1..].iter().cloned());
    }
    argv
}

#[inline]
pub(crate) fn compiler_optimization_level(
    level: CliOptimizationLevel,
) -> prism_compiler::OptimizationLevel {
    match level {
        CliOptimizationLevel::None => prism_compiler::OptimizationLevel::None,
        CliOptimizationLevel::Basic => prism_compiler::OptimizationLevel::Basic,
        CliOptimizationLevel::Full => prism_compiler::OptimizationLevel::Full,
    }
}

#[inline]
fn vm_source_optimization_level(level: CliOptimizationLevel) -> prism_vm::SourceOptimization {
    match level {
        CliOptimizationLevel::None => prism_vm::SourceOptimization::None,
        CliOptimizationLevel::Basic => prism_vm::SourceOptimization::Basic,
        CliOptimizationLevel::Full => prism_vm::SourceOptimization::Full,
    }
}

fn module_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        push_unique_path(&mut paths, cwd);
    }

    if let Some(pythonpath) = std::env::var_os("PYTHONPATH") {
        for path in std::env::split_paths(&pythonpath) {
            push_unique_path(&mut paths, path);
        }
    }

    if let Some(stdlib_path) = source_stdlib_path() {
        push_unique_path(&mut paths, stdlib_path);
    }

    paths
}

fn script_search_paths(path: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(parent) = path.parent() {
        paths.push(parent.to_path_buf());
    }
    for search_path in module_search_paths() {
        if !paths.iter().any(|existing| existing == &search_path) {
            paths.push(search_path);
        }
    }
    paths
}

fn source_stdlib_path() -> Option<PathBuf> {
    let path = PathBuf::from(prism_stdlib::source_stdlib_path());
    path.is_dir().then_some(path)
}

fn push_unique_path(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|existing| existing == &candidate) {
        paths.push(candidate);
    }
}

fn resolve_module_path_in_search_paths(module: &str, search_paths: &[PathBuf]) -> Option<PathBuf> {
    resolve_module_entry_in_search_paths(module, search_paths).map(|entry| entry.path)
}

fn resolve_module_entry_in_search_paths(
    module: &str,
    search_paths: &[PathBuf],
) -> Option<ResolvedModuleEntry> {
    let parts: Vec<&str> = module.split('.').collect();
    if parts.is_empty()
        || parts
            .iter()
            .any(|segment| !is_valid_module_segment(segment))
    {
        return None;
    }

    for base in search_paths {
        let mut module_base = base.clone();
        for part in &parts {
            module_base.push(part);
        }

        let module_file = module_base.with_extension("py");
        if module_file.is_file() {
            let package_name = parts[..parts.len().saturating_sub(1)].join(".");
            return Some(ResolvedModuleEntry {
                path: module_file,
                package_name: Arc::from(package_name),
            });
        }

        let package_main = module_base.join("__main__.py");
        if package_main.is_file() {
            return Some(ResolvedModuleEntry {
                path: package_main,
                package_name: Arc::from(module),
            });
        }
    }

    None
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
