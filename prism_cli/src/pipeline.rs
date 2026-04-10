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
    // Phase 1: Parse.
    let module = match prism_parser::parse(source) {
        Ok(m) => m,
        Err(e) => {
            return error::format_prism_error(&e, Some(source), main_module.filename.as_ref());
        }
    };

    // Phase 2: Compile.
    let optimize = compiler_optimization_level(config.optimize);
    let code = match prism_compiler::Compiler::compile_module_with_optimization(
        &module,
        main_module.filename.as_ref(),
        optimize,
    ) {
        Ok(c) => c,
        Err(e) => {
            return error::format_compile_error(&e, Some(source), main_module.filename.as_ref());
        }
    };

    // Phase 3: Execute.
    let code = Arc::new(code);
    let mut vm = if config.jit_enabled() {
        prism_vm::VirtualMachine::with_jit()
    } else {
        prism_vm::VirtualMachine::new()
    };
    vm.set_compiler_optimization(optimize);
    vm.import_resolver = prism_vm::import::ImportResolver::with_sys_args(script_args.to_vec());
    for search_path in search_paths {
        vm.import_resolver
            .add_search_path(Arc::from(search_path.to_string_lossy().into_owned()));
    }
    let filename_for_errors = Arc::clone(&main_module.filename);
    let main_module = Arc::new(prism_vm::import::ModuleObject::with_metadata(
        Arc::clone(&main_module.module_name),
        None,
        Some(Arc::clone(&main_module.filename)),
        Some(Arc::clone(&main_module.package_name)),
    ));

    match vm.execute_in_module(code, main_module) {
        Ok(_) => ExitCode::from(error::EXIT_SUCCESS),
        Err(e) => error::format_prism_error(&e, Some(source), filename_for_errors.as_ref()),
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
fn compiler_optimization_level(level: CliOptimizationLevel) -> prism_compiler::OptimizationLevel {
    match level {
        CliOptimizationLevel::None => prism_compiler::OptimizationLevel::None,
        CliOptimizationLevel::Basic => prism_compiler::OptimizationLevel::Basic,
        CliOptimizationLevel::Full => prism_compiler::OptimizationLevel::Full,
    }
}

fn module_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd);
    }

    if let Some(pythonpath) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&pythonpath));
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestTempDir {
        path: PathBuf,
    }

    impl TestTempDir {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time went backwards")
                .as_nanos();

            let mut path = std::env::temp_dir();
            path.push(format!(
                "prism_cli_pipeline_tests_{}_{}_{}",
                std::process::id(),
                nanos,
                unique
            ));

            std::fs::create_dir_all(&path).expect("failed to create temp test dir");
            Self { path }
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_file(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create parent dir");
        }
        std::fs::write(path, content).expect("failed to write test file");
    }

    fn execute_with_default_config(source: &str, filename: &str) -> ExitCode {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        super::execute_source(source, filename, &config)
    }

    fn execute_with_config(source: &str, filename: &str, config: &RuntimeConfig) -> ExitCode {
        super::execute_source(source, filename, config)
    }

    // =========================================================================
    // Source Execution Tests
    // =========================================================================

    #[test]
    fn test_execute_empty_source() {
        let code = execute_source("", "<test>", &RuntimeConfig::from_args(&Default::default()));
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_simple_assignment() {
        let code = execute_source(
            "x = 42",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_simple_assignment_with_jit_disabled() {
        let args = crate::args::PrismArgs {
            x_options: vec!["nojit".to_string()],
            ..Default::default()
        };
        let config = RuntimeConfig::from_args(&args);
        let code = execute_with_config("x = 42", "<test>", &config);
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_multiple_statements() {
        let code = execute_source(
            "x = 1\ny = 2\nz = x + y",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_function_def_and_call() {
        let code = execute_source(
            "def add(a, b):\n    return a + b\nresult = add(1, 2)",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_if_statement() {
        let code = execute_source(
            "x = 10\nif x > 5:\n    y = True\nelse:\n    y = False",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_for_loop() {
        let code = execute_source(
            "total = 0\nfor i in range(10):\n    total = total + i",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_while_loop() {
        let code = execute_source(
            "x = 0\nwhile x < 5:\n    x = x + 1",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_list_operations() {
        let code = execute_source(
            "lst = [1, 2, 3]\nx = len(lst)",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_dict_operations() {
        let code = execute_source(
            "d = {'a': 1, 'b': 2}",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_string_operations() {
        let code = execute_with_default_config("s = 'hello'\nx = len(s)", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_print_call() {
        let code = execute_with_default_config("print('hello, world')", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_nested_function() {
        let code = execute_with_default_config(
            "def outer():\n    def inner():\n        return 42\n    return inner()\nresult = outer()",
            "<test>",
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_class_definition() {
        let code = execute_with_default_config("class Foo:\n    x = 42", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_try_except() {
        let code = execute_with_default_config("try:\n    x = 1\nexcept:\n    x = 0", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_list_comprehension() {
        let code = execute_with_default_config("squares = [x * x for x in range(5)]", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_execute_syntax_error_returns_error_code() {
        let code = execute_with_default_config("def", "<test>");
        assert_ne!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_name_error_returns_error_code() {
        let code = execute_with_default_config("print(undefined_variable)", "<test>");
        assert_ne!(code, ExitCode::from(0));
    }

    // =========================================================================
    // File Execution Tests
    // =========================================================================

    #[test]
    fn test_run_file_nonexistent() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_file(Path::new("/nonexistent/path/test.py"), &config);
        assert_eq!(code, ExitCode::from(error::EXIT_USAGE_ERROR));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_execute_only_comments() {
        let code = execute_with_default_config("# just a comment\n# another comment", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_only_whitespace() {
        let code = execute_with_default_config("   \n\n   \n", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_pass_statement() {
        let code = execute_with_default_config("pass", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_multiline_string() {
        let code = execute_with_default_config("s = '''hello\nworld'''", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_boolean_operations() {
        let code = execute_with_default_config(
            "x = True and False\ny = True or False\nz = not True",
            "<test>",
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_comparison_chain() {
        let code = execute_with_default_config("result = 1 < 2 < 3", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_tuple_unpacking() {
        let code = execute_with_default_config("a, b = 1, 2", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_augmented_assignment() {
        let code = execute_with_default_config("x = 1\nx += 2\nx *= 3", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_global_statement() {
        let code = execute_with_default_config(
            "x = 0\ndef inc():\n    global x\n    x = x + 1\ninc()\ninc()",
            "<test>",
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_lambda() {
        let code = execute_with_default_config("f = lambda x: x * 2\nresult = f(21)", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_assert_passes() {
        let code = execute_with_default_config("assert True", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_assert_fails_returns_error() {
        let code = execute_with_default_config("assert False", "<test>");
        assert_ne!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_assert_stripped_with_optimize_basic() {
        let args = crate::args::PrismArgs {
            optimize: crate::args::OptimizationLevel::Basic,
            ..Default::default()
        };
        let config = RuntimeConfig::from_args(&args);
        let code = execute_with_config("assert False", "<test>", &config);
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_delete_statement() {
        let code = execute_with_default_config("x = 1\ndel x", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_resolve_module_path_finds_module_file() {
        let temp = TestTempDir::new();
        write_file(&temp.path.join("demo.py"), "x = 1\n");

        let resolved =
            resolve_module_path_in_search_paths("demo", std::slice::from_ref(&temp.path));
        assert_eq!(resolved, Some(temp.path.join("demo.py")));
    }

    #[test]
    fn test_resolve_module_path_finds_package_main() {
        let temp = TestTempDir::new();
        write_file(&temp.path.join("pkg").join("__main__.py"), "x = 1\n");

        let resolved = resolve_module_path_in_search_paths("pkg", std::slice::from_ref(&temp.path));
        assert_eq!(resolved, Some(temp.path.join("pkg").join("__main__.py")));
    }

    #[test]
    fn test_resolve_module_path_rejects_invalid_module_names() {
        let temp = TestTempDir::new();
        let paths = vec![temp.path.clone()];

        assert_eq!(resolve_module_path_in_search_paths("", &paths), None);
        assert_eq!(
            resolve_module_path_in_search_paths("pkg..mod", &paths),
            None
        );
        assert_eq!(resolve_module_path_in_search_paths("1bad", &paths), None);
        assert_eq!(
            resolve_module_path_in_search_paths("../escape", &paths),
            None
        );
    }

    #[test]
    fn test_run_module_executes_module_file() {
        let temp = TestTempDir::new();
        write_file(&temp.path.join("mymodule.py"), "x = 123\n");

        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code =
            run_module_with_search_paths("mymodule", &config, std::slice::from_ref(&temp.path));
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_file_supports_sibling_source_imports() {
        let temp = TestTempDir::new();
        let main_path = temp.path.join("main.py");
        write_file(&temp.path.join("helper.py"), "VALUE = 123\n");
        write_file(
            &main_path,
            "import helper\nfrom helper import VALUE\nassert helper.VALUE == 123\nassert VALUE == 123\n",
        );

        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_file_with_args(&main_path, &config, &[main_path.display().to_string()]);
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_file_supports_source_packages_and_submodules() {
        let temp = TestTempDir::new();
        let main_path = temp.path.join("main.py");
        write_file(&temp.path.join("pkg").join("__init__.py"), "VALUE = 5\n");
        write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 7\n");
        write_file(
            &main_path,
            "import pkg\nimport pkg.helper\nfrom pkg import helper\nassert pkg.VALUE == 5\nassert pkg.helper.VALUE == 7\nassert helper.VALUE == 7\n",
        );

        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_file_with_args(&main_path, &config, &[main_path.display().to_string()]);
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_module_with_args_sets_argv0_to_module_path() {
        let temp = TestTempDir::new();
        let module_path = temp.path.join("mymodule.py");
        write_file(
            &module_path,
            "import sys\nassert len(sys.argv) == 3\nassert len(sys.argv[0]) > 8\nassert len(sys.argv[1]) == 3\nassert len(sys.argv[2]) == 3\n",
        );

        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let module_args = vec!["mymodule".to_string(), "one".to_string(), "two".to_string()];
        let code = run_module_with_search_paths_with_args(
            "mymodule",
            &config,
            std::slice::from_ref(&temp.path),
            &module_args,
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_module_supports_relative_imports() {
        let temp = TestTempDir::new();
        write_file(&temp.path.join("pkg").join("__init__.py"), "");
        write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 41\n");
        write_file(
            &temp.path.join("pkg").join("main.py"),
            "from .helper import VALUE\nfrom . import helper\nassert VALUE == 41\nassert helper.VALUE == 41\n",
        );

        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code =
            run_module_with_search_paths("pkg.main", &config, std::slice::from_ref(&temp.path));
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_module_supports_parent_relative_imports() {
        let temp = TestTempDir::new();
        write_file(&temp.path.join("pkg").join("__init__.py"), "");
        write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 99\n");
        write_file(
            &temp.path.join("pkg").join("subpkg").join("__init__.py"),
            "",
        );
        write_file(
            &temp.path.join("pkg").join("subpkg").join("main.py"),
            "from ..helper import VALUE\nassert VALUE == 99\n",
        );

        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_module_with_search_paths(
            "pkg.subpkg.main",
            &config,
            std::slice::from_ref(&temp.path),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_module_package_entrypoint_sets_package_context() {
        let temp = TestTempDir::new();
        write_file(&temp.path.join("pkg").join("__init__.py"), "");
        write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 7\n");
        write_file(
            &temp.path.join("pkg").join("__main__.py"),
            "from .helper import VALUE\nimport sys\nassert VALUE == 7\nassert len(sys.argv) == 1\nassert len(sys.argv[0]) > 0\n",
        );

        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_module_with_search_paths("pkg", &config, std::slice::from_ref(&temp.path));
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_module_without_args_sets_single_argv0_path() {
        let temp = TestTempDir::new();
        let module_path = temp.path.join("solo.py");
        write_file(
            &module_path,
            "import sys\nassert len(sys.argv) == 1\nassert len(sys.argv[0]) > 0\n",
        );

        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_module_with_search_paths_with_args(
            "solo",
            &config,
            std::slice::from_ref(&temp.path),
            &[],
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_module_missing_returns_error() {
        let temp = TestTempDir::new();
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_module_with_search_paths(
            "missing_module",
            &config,
            std::slice::from_ref(&temp.path),
        );
        assert_eq!(code, ExitCode::from(error::EXIT_ERROR));
    }

    #[test]
    fn test_run_string_simple() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_string("x = 42", &config);
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_string_defaults_sys_argv0_to_dash_c() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_string("import sys\nassert len(sys.argv[0]) == 2\n", &config);
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_string_supports_dotted_import_binding_top_level_module() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_string(
            "import os.path\nfrom os import path\nassert os.path is not None\nassert path is not None\n",
            &config,
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_build_module_argv_rewrites_first_element_to_resolved_path() {
        let path = Path::new("/tmp/pkg/module.py");
        let script_args = vec![
            "pkg.module".to_string(),
            "alpha".to_string(),
            "beta".to_string(),
        ];
        let argv = build_module_argv(path, &script_args);
        assert_eq!(argv[0], path.display().to_string());
        assert_eq!(argv[1], "alpha");
        assert_eq!(argv[2], "beta");
    }

    #[test]
    fn test_build_module_argv_without_explicit_args_uses_path_only() {
        let path = Path::new("/tmp/pkg/module.py");
        let argv = build_module_argv(path, &[]);
        assert_eq!(argv.len(), 1);
        assert_eq!(argv[0], path.display().to_string());
    }

    #[test]
    fn test_run_string_error() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_string("def", &config);
        assert_ne!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_string_with_args_populates_sys_argv() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let script_args = vec!["prog.py".to_string(), "one".to_string(), "two".to_string()];
        let code = run_string_with_args(
            "import sys\nassert len(sys.argv) == 3\n",
            &config,
            &script_args,
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_string_with_args_supports_from_import() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let script_args = vec!["prog.py".to_string(), "one".to_string(), "two".to_string()];
        let code = run_string_with_args(
            "from sys import argv\nassert len(argv) == 3\n",
            &config,
            &script_args,
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_string_with_args_supports_import_star() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let script_args = vec!["prog.py".to_string(), "one".to_string(), "two".to_string()];
        let code = run_string_with_args(
            "from sys import *\nassert len(argv) == 3\n",
            &config,
            &script_args,
        );
        assert_eq!(code, ExitCode::from(0));
    }
}
