//! CPython compatibility test harness.
//!
//! This module provides a dedicated runner for executing Prism against the
//! CPython regression test corpus with CPython-style discovery rules and a
//! stable machine-readable report format.

use serde::Serialize;
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_TIMEOUT_SECS: u64 = 60;
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(25);
const MAX_WORKDIR_COMPONENT_LEN: usize = 80;
const SPLIT_TEST_DIRS: &[&str] = &[
    "test_asyncio",
    "test_concurrent_futures",
    "test_doctests",
    "test_future_stmt",
    "test_gdb",
    "test_inspect",
    "test_multiprocessing_fork",
    "test_multiprocessing_forkserver",
    "test_multiprocessing_spawn",
    "test_pydoc",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerMode {
    Import,
    Suite,
    TestMain,
}

impl RunnerMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "import" => Some(Self::Import),
            "suite" => Some(Self::Suite),
            "test-main" => Some(Self::TestMain),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Import => "import",
            Self::Suite => "suite",
            Self::TestMain => "test-main",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrismCommand {
    pub executable: PathBuf,
    pub prefix_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliArgs {
    pub cpython_root: PathBuf,
    pub prism_executable: Option<PathBuf>,
    pub prism_args: Vec<String>,
    pub lib_dir: Option<PathBuf>,
    pub test_dir: Option<PathBuf>,
    pub tests: Vec<String>,
    pub runner: RunnerMode,
    pub timeout: Option<Duration>,
    pub start: Option<String>,
    pub list_tests: bool,
    pub fail_fast: bool,
    pub verbose: bool,
    pub quiet: bool,
    pub json_report: Option<PathBuf>,
    pub work_dir: Option<PathBuf>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            cpython_root: PathBuf::new(),
            prism_executable: None,
            prism_args: Vec::new(),
            lib_dir: None,
            test_dir: None,
            tests: Vec::new(),
            runner: RunnerMode::Suite,
            timeout: Some(Duration::from_secs(DEFAULT_TIMEOUT_SECS)),
            start: None,
            list_tests: false,
            fail_fast: false,
            verbose: false,
            quiet: false,
            json_report: None,
            work_dir: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliAction {
    Help,
    Version,
    Run(CliArgs),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessConfig {
    pub prism_command: PrismCommand,
    pub lib_dir: PathBuf,
    pub test_dir: PathBuf,
    pub search_root: PathBuf,
    pub work_root: PathBuf,
    pub prefix_test_package: bool,
    pub runner: RunnerMode,
    pub timeout: Option<Duration>,
    pub fail_fast: bool,
    pub verbose: bool,
    pub quiet: bool,
    pub json_report: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveredTest {
    pub name: String,
    pub module: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuitePlan {
    pub tests: Vec<DiscoveredTest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TestStatus {
    Passed,
    Failed,
    TimedOut,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TestResult {
    pub test_name: String,
    pub module: String,
    pub path: PathBuf,
    pub work_dir: PathBuf,
    pub status: TestStatus,
    pub exit_code: Option<i32>,
    pub duration_ms: u128,
    pub stdout: String,
    pub stderr: String,
    pub command: Vec<String>,
    pub failure_summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SuiteReport {
    pub runner: String,
    pub work_root: PathBuf,
    pub selected: usize,
    pub passed: usize,
    pub failed: usize,
    pub timed_out: usize,
    pub duration_ms: u128,
    pub listed_only: bool,
    pub tests: Vec<String>,
    pub results: Vec<TestResult>,
}

impl SuiteReport {
    pub fn exit_code(&self) -> u8 {
        if self.listed_only || (self.failed == 0 && self.timed_out == 0) {
            0
        } else {
            1
        }
    }
}

#[derive(Debug)]
pub enum HarnessError {
    Argument(String),
    InvalidConfiguration(String),
    UnknownTest(String),
    Io {
        context: &'static str,
        source: io::Error,
    },
    Json {
        context: &'static str,
        source: serde_json::Error,
    },
}

impl HarnessError {
    fn io(context: &'static str, source: io::Error) -> Self {
        Self::Io { context, source }
    }

    fn json(context: &'static str, source: serde_json::Error) -> Self {
        Self::Json { context, source }
    }
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Argument(message)
            | Self::InvalidConfiguration(message)
            | Self::UnknownTest(message) => f.write_str(message),
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::Json { context, source } => write!(f, "{context}: {source}"),
        }
    }
}

impl std::error::Error for HarnessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub trait TestExecutor {
    fn run(
        &self,
        test: &DiscoveredTest,
        config: &HarnessConfig,
    ) -> Result<TestResult, HarnessError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SubprocessPrismExecutor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChildOutcome {
    Completed(std::process::ExitStatus),
    TimedOut,
}

impl HarnessConfig {
    pub fn from_cli(args: &CliArgs) -> Result<Self, HarnessError> {
        let cpython_root = canonicalize_existing_dir(&args.cpython_root, "CPython root")?;
        let lib_dir = args
            .lib_dir
            .clone()
            .unwrap_or_else(|| cpython_root.join("Lib"));
        let lib_dir = canonicalize_existing_dir(&lib_dir, "CPython Lib directory")?;

        let test_dir = args
            .test_dir
            .clone()
            .unwrap_or_else(|| lib_dir.join("test"));
        let test_dir = canonicalize_existing_dir(&test_dir, "CPython test directory")?;

        let prefix_test_package = test_dir == lib_dir.join("test");
        let search_root = if prefix_test_package {
            lib_dir.clone()
        } else {
            test_dir.clone()
        };

        let prism_command = PrismCommand {
            executable: args
                .prism_executable
                .clone()
                .unwrap_or_else(default_prism_executable),
            prefix_args: args.prism_args.clone(),
        };

        if !prism_command.executable.is_file() {
            return Err(HarnessError::InvalidConfiguration(format!(
                "Prism executable does not exist: {}",
                prism_command.executable.display()
            )));
        }

        let work_root = args.work_dir.clone().unwrap_or_else(default_work_root);

        Ok(Self {
            prism_command,
            lib_dir,
            test_dir,
            search_root,
            work_root,
            prefix_test_package,
            runner: args.runner,
            timeout: args.timeout,
            fail_fast: args.fail_fast,
            verbose: args.verbose,
            quiet: args.quiet,
            json_report: args.json_report.clone(),
        })
    }
}

impl TestExecutor for SubprocessPrismExecutor {
    fn run(
        &self,
        test: &DiscoveredTest,
        config: &HarnessConfig,
    ) -> Result<TestResult, HarnessError> {
        let work_dir = config.work_root.join(sanitize_path_component(&test.name));
        recreate_directory(&work_dir)?;

        let bootstrap_path = work_dir.join("bootstrap.py");
        let stdout_path = work_dir.join("stdout.txt");
        let stderr_path = work_dir.join("stderr.txt");

        fs::write(
            &bootstrap_path,
            render_bootstrap_script(&test.module, config.runner, config.verbose),
        )
        .map_err(|source| HarnessError::io("failed to write bootstrap script", source))?;

        let stdout_file = File::create(&stdout_path)
            .map_err(|source| HarnessError::io("failed to create stdout log", source))?;
        let stderr_file = File::create(&stderr_path)
            .map_err(|source| HarnessError::io("failed to create stderr log", source))?;

        let mut command = Command::new(&config.prism_command.executable);
        command.args(&config.prism_command.prefix_args);
        command.arg("-B");
        command.arg(&bootstrap_path);
        command.current_dir(&work_dir);
        command.stdin(Stdio::null());
        command.stdout(Stdio::from(stdout_file));
        command.stderr(Stdio::from(stderr_file));
        configure_child_environment(&mut command, &config.search_root);

        let command_line = describe_command(
            &config.prism_command.executable,
            &config.prism_command.prefix_args,
            &["-B".to_string(), bootstrap_path.display().to_string()],
        );

        let start = Instant::now();
        let mut child = command
            .spawn()
            .map_err(|source| HarnessError::io("failed to spawn Prism test process", source))?;
        let wait_outcome = wait_for_child(&mut child, config.timeout)?;
        let duration_ms = start.elapsed().as_millis();

        let stdout = read_optional_text(&stdout_path)?;
        let stderr = read_optional_text(&stderr_path)?;

        let (status, exit_code) = match wait_outcome {
            ChildOutcome::Completed(exit_status) if exit_status.success() => {
                (TestStatus::Passed, exit_status.code())
            }
            ChildOutcome::Completed(exit_status) => (TestStatus::Failed, exit_status.code()),
            ChildOutcome::TimedOut => (TestStatus::TimedOut, None),
        };

        Ok(TestResult {
            test_name: test.name.clone(),
            module: test.module.clone(),
            path: test.path.clone(),
            work_dir,
            status,
            exit_code,
            duration_ms,
            stdout: stdout.clone(),
            stderr: stderr.clone(),
            command: command_line,
            failure_summary: summarize_failure(status, &stdout, &stderr),
        })
    }
}

pub fn parse_cli_action(raw_args: &[String]) -> Result<CliAction, HarnessError> {
    let mut parsed = CliArgs::default();
    let mut index = 0;

    while index < raw_args.len() {
        let arg = &raw_args[index];
        match arg.as_str() {
            "-h" | "--help" => return Ok(CliAction::Help),
            "-V" | "--version" => return Ok(CliAction::Version),
            "--cpython-root" => {
                index += 1;
                parsed.cpython_root = required_path_value(raw_args, index, "--cpython-root")?;
            }
            "--lib-dir" => {
                index += 1;
                parsed.lib_dir = Some(required_path_value(raw_args, index, "--lib-dir")?);
            }
            "--testdir" => {
                index += 1;
                parsed.test_dir = Some(required_path_value(raw_args, index, "--testdir")?);
            }
            "--prism" => {
                index += 1;
                parsed.prism_executable = Some(required_path_value(raw_args, index, "--prism")?);
            }
            "--prism-arg" => {
                index += 1;
                parsed
                    .prism_args
                    .push(required_string_value(raw_args, index, "--prism-arg")?);
            }
            "--runner" => {
                index += 1;
                let value = required_string_value(raw_args, index, "--runner")?;
                parsed.runner = RunnerMode::parse(&value).ok_or_else(|| {
                    HarnessError::Argument(format!(
                        "invalid runner '{value}', expected one of: import, suite, test-main"
                    ))
                })?;
            }
            "--timeout" => {
                index += 1;
                let value = required_string_value(raw_args, index, "--timeout")?;
                parsed.timeout = parse_timeout(&value)?;
            }
            "--start" | "-S" => {
                index += 1;
                parsed.start = Some(required_string_value(raw_args, index, "--start")?);
            }
            "--json-report" => {
                index += 1;
                parsed.json_report = Some(required_path_value(raw_args, index, "--json-report")?);
            }
            "--work-dir" => {
                index += 1;
                parsed.work_dir = Some(required_path_value(raw_args, index, "--work-dir")?);
            }
            "--list-tests" => parsed.list_tests = true,
            "--fail-fast" => parsed.fail_fast = true,
            "-v" | "--verbose" => parsed.verbose = true,
            "-q" | "--quiet" => parsed.quiet = true,
            value if value.starts_with('-') => {
                return Err(HarnessError::Argument(format!("unknown option: {value}")));
            }
            value => parsed.tests.push(value.to_string()),
        }
        index += 1;
    }

    if parsed.cpython_root.as_os_str().is_empty() {
        return Err(HarnessError::Argument(
            "missing required option --cpython-root <path>".to_string(),
        ));
    }

    Ok(CliAction::Run(parsed))
}

pub fn help_text() -> String {
    format!(
        "\
usage: prism-test --cpython-root <path> [options] [test_name ...]

Run Prism against the CPython regression test corpus using CPython-style
test discovery and Prism subprocess isolation.

options:
  --cpython-root <path>  Path to the CPython source checkout
  --lib-dir <path>       Override the stdlib root (defaults to <root>/Lib)
  --testdir <path>       Override the regression test directory
  --prism <path>         Path to the Prism executable (defaults to sibling prism binary)
  --prism-arg <arg>      Extra argument passed to the Prism executable
  --runner <mode>        One of: import, suite, test-main (default: suite)
  --timeout <secs>       Per-test timeout in seconds (0 disables timeouts, default: {DEFAULT_TIMEOUT_SECS})
  --start <test>         Start execution at the named discovered test
  --list-tests           Print the discovered test names without executing them
  --fail-fast            Stop after the first failing or timed-out test
  --json-report <path>   Write a machine-readable JSON report
  --work-dir <path>      Directory for per-test bootstrap scripts and logs
  -v, --verbose          Print captured output for passing tests
  -q, --quiet            Suppress per-test success lines
  -h, --help             Show this help message
  -V, --version          Print Prism test harness version"
    )
}

pub fn version_text() -> String {
    format!(
        "prism-test {} (CPython {}.{}.{})",
        prism_core::VERSION,
        prism_core::PYTHON_VERSION.0,
        prism_core::PYTHON_VERSION.1,
        prism_core::PYTHON_VERSION.2
    )
}

pub fn execute_cli_with_executor<E: TestExecutor>(
    args: &CliArgs,
    executor: &E,
) -> Result<SuiteReport, HarnessError> {
    let config = HarnessConfig::from_cli(args)?;
    let harness = Harness::new(config);
    let plan = harness.plan(&args.tests, args.start.as_deref())?;

    if args.list_tests {
        for test in &plan.tests {
            println!("{}", test.name);
        }
        return Ok(SuiteReport {
            runner: args.runner.as_str().to_string(),
            work_root: harness.config.work_root.clone(),
            selected: plan.tests.len(),
            passed: 0,
            failed: 0,
            timed_out: 0,
            duration_ms: 0,
            listed_only: true,
            tests: plan.tests.into_iter().map(|test| test.name).collect(),
            results: Vec::new(),
        });
    }

    if !harness.config.quiet {
        println!(
            "Running {} CPython tests from {} with runner '{}'",
            plan.tests.len(),
            display_path(&harness.config.test_dir),
            harness.config.runner.as_str()
        );
        println!(
            "Prism executable: {}",
            display_path(&harness.config.prism_command.executable)
        );
        println!("Search root: {}", display_path(&harness.config.search_root));
        println!("Work root: {}", display_path(&harness.config.work_root));
        println!();
    }

    let report = harness.run_plan(plan, executor)?;
    if let Some(path) = &harness.config.json_report {
        write_json_report(path, &report)?;
    }
    print_summary(&report, harness.config.quiet);
    Ok(report)
}

pub struct Harness {
    config: HarnessConfig,
}

impl Harness {
    pub fn new(config: HarnessConfig) -> Self {
        Self { config }
    }

    pub fn plan(
        &self,
        requested: &[String],
        start: Option<&str>,
    ) -> Result<SuitePlan, HarnessError> {
        let discovered = discover_tests(&self.config)?;
        let selected = select_tests(&discovered, requested, start)?;
        Ok(SuitePlan { tests: selected })
    }

    pub fn run_plan<E: TestExecutor>(
        &self,
        plan: SuitePlan,
        executor: &E,
    ) -> Result<SuiteReport, HarnessError> {
        fs::create_dir_all(&self.config.work_root)
            .map_err(|source| HarnessError::io("failed to create suite work directory", source))?;

        let started = Instant::now();
        let selected_names = plan
            .tests
            .iter()
            .map(|test| test.name.clone())
            .collect::<Vec<_>>();

        let mut results = Vec::with_capacity(plan.tests.len());
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut timed_out = 0usize;

        for (index, test) in plan.tests.iter().enumerate() {
            let result = executor.run(test, &self.config)?;
            match result.status {
                TestStatus::Passed => passed += 1,
                TestStatus::Failed => failed += 1,
                TestStatus::TimedOut => timed_out += 1,
            }

            print_test_progress(index + 1, plan.tests.len(), &result, self.config.quiet);
            if self.config.verbose || result.status != TestStatus::Passed {
                print_captured_output(&result, self.config.quiet);
            }

            let should_stop = self.config.fail_fast
                && matches!(result.status, TestStatus::Failed | TestStatus::TimedOut);
            results.push(result);
            if should_stop {
                break;
            }
        }

        Ok(SuiteReport {
            runner: self.config.runner.as_str().to_string(),
            work_root: self.config.work_root.clone(),
            selected: plan.tests.len(),
            passed,
            failed,
            timed_out,
            duration_ms: started.elapsed().as_millis(),
            listed_only: false,
            tests: selected_names,
            results,
        })
    }
}

pub fn default_prism_executable() -> PathBuf {
    let current = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("prism-test"));
    if cfg!(windows) {
        current.with_file_name("prism.exe")
    } else {
        current.with_file_name("prism")
    }
}

fn configure_child_environment(command: &mut Command, search_root: &Path) {
    for (name, _) in std::env::vars_os() {
        if is_python_env_var(&name) {
            command.env_remove(name);
        }
    }
    command.env("PYTHONPATH", search_root);
    command.env("PYTHONDONTWRITEBYTECODE", "1");
    command.env("PYTHONUNBUFFERED", "1");
}

fn is_python_env_var(name: &OsString) -> bool {
    name.to_string_lossy().starts_with("PYTHON")
}

fn discover_tests(config: &HarnessConfig) -> Result<Vec<DiscoveredTest>, HarnessError> {
    let mut tests = Vec::new();
    discover_tests_inner(&config.test_dir, "", config.prefix_test_package, &mut tests)?;
    Ok(tests)
}

fn discover_tests_inner(
    directory: &Path,
    base_mod: &str,
    prefix_test_package: bool,
    discovered: &mut Vec<DiscoveredTest>,
) -> Result<(), HarnessError> {
    let mut entries = fs::read_dir(directory)
        .map_err(|source| HarnessError::io("failed to read test directory", source))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| HarnessError::io("failed to enumerate test directory", source))?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|source| HarnessError::io("failed to inspect test entry", source))?;
        let raw_name = entry.file_name();
        let raw_name = raw_name.to_string_lossy();
        let (stem, extension) = split_name(&raw_name);
        if !stem.starts_with("test_") {
            continue;
        }

        let fullname = if base_mod.is_empty() {
            stem.to_string()
        } else {
            format!("{base_mod}.{stem}")
        };

        if split_test_dir_names().contains(stem) && file_type.is_dir() {
            let nested_base = if base_mod.is_empty() {
                if prefix_test_package {
                    format!("test.{stem}")
                } else {
                    stem.to_string()
                }
            } else {
                fullname
            };
            discover_tests_inner(&path, &nested_base, prefix_test_package, discovered)?;
            continue;
        }

        if extension == Some("py") || (extension.is_none() && file_type.is_dir()) {
            let module = resolve_module_name(&fullname, prefix_test_package);
            discovered.push(DiscoveredTest {
                name: fullname,
                module,
                path,
            });
        }
    }

    Ok(())
}

fn select_tests(
    discovered: &[DiscoveredTest],
    requested: &[String],
    start: Option<&str>,
) -> Result<Vec<DiscoveredTest>, HarnessError> {
    let mut selected = if requested.is_empty() {
        discovered.to_vec()
    } else {
        let mut dedup = BTreeSet::new();
        let mut ordered = Vec::new();
        for request in requested {
            for test in expand_requested_test(discovered, request)? {
                if dedup.insert(test.name.clone()) {
                    ordered.push(test);
                }
            }
        }
        ordered
    };

    if let Some(start_name) = start {
        let start_name = strip_py_suffix(start_name);
        let Some(start_index) = selected.iter().position(|test| {
            test.name == start_name
                || test.module == start_name
                || test.name.starts_with(&(start_name.clone() + "."))
                || test.module.starts_with(&(start_name.clone() + "."))
        }) else {
            return Err(HarnessError::UnknownTest(format!(
                "start test not found in selection: {start_name}"
            )));
        };
        selected.drain(0..start_index);
    }

    Ok(selected)
}

fn expand_requested_test(
    discovered: &[DiscoveredTest],
    requested: &str,
) -> Result<Vec<DiscoveredTest>, HarnessError> {
    let requested = strip_py_suffix(requested);

    let exact = discovered
        .iter()
        .filter(|test| test.name == requested || test.module == requested)
        .cloned()
        .collect::<Vec<_>>();
    if !exact.is_empty() {
        return Ok(exact);
    }

    let prefixed = if requested.starts_with("test.") {
        requested.clone()
    } else {
        format!("test.{requested}")
    };

    let expanded = discovered
        .iter()
        .filter(|test| {
            test.name.starts_with(&(requested.clone() + "."))
                || test.module.starts_with(&(requested.clone() + "."))
                || test.name.starts_with(&(prefixed.clone() + "."))
                || test.module.starts_with(&(prefixed.clone() + "."))
        })
        .cloned()
        .collect::<Vec<_>>();
    if !expanded.is_empty() {
        return Ok(expanded);
    }

    Err(HarnessError::UnknownTest(format!(
        "requested test not found: {requested}"
    )))
}

fn render_bootstrap_script(module_name: &str, runner: RunnerMode, verbose: bool) -> String {
    let verbosity = if verbose { 2 } else { 1 };
    let import_stmt = format!("import {module_name} as module");

    match runner {
        RunnerMode::Import => format!("{import_stmt}\n"),
        RunnerMode::TestMain => format!(
            "{import_stmt}\nif hasattr(module, 'test_main'):\n    module.test_main()\nelse:\n    raise SystemExit(\"test module does not define test_main\")\n"
        ),
        RunnerMode::Suite => format!(
            "{import_stmt}\nif hasattr(module, 'test_main'):\n    module.test_main()\nelse:\n    import unittest\n    suite = unittest.defaultTestLoader.loadTestsFromModule(module)\n    result = unittest.TextTestRunner(verbosity={verbosity}).run(suite)\n    if not result.wasSuccessful():\n        raise SystemExit(1)\n"
        ),
    }
}

fn describe_command(
    executable: &Path,
    prefix_args: &[String],
    command_args: &[String],
) -> Vec<String> {
    let mut command = Vec::with_capacity(1 + prefix_args.len() + command_args.len());
    command.push(executable.display().to_string());
    command.extend(prefix_args.iter().cloned());
    command.extend(command_args.iter().cloned());
    command
}

fn wait_for_child(
    child: &mut std::process::Child,
    timeout: Option<Duration>,
) -> Result<ChildOutcome, HarnessError> {
    match timeout {
        None => child
            .wait()
            .map(ChildOutcome::Completed)
            .map_err(|source| HarnessError::io("failed to wait for Prism test process", source)),
        Some(timeout) => {
            let start = Instant::now();
            loop {
                if let Some(status) = child.try_wait().map_err(|source| {
                    HarnessError::io("failed to poll Prism test process", source)
                })? {
                    return Ok(ChildOutcome::Completed(status));
                }
                if start.elapsed() >= timeout {
                    child.kill().map_err(|source| {
                        HarnessError::io("failed to terminate timed out test", source)
                    })?;
                    let _ = child.wait();
                    return Ok(ChildOutcome::TimedOut);
                }
                thread::sleep(WAIT_POLL_INTERVAL);
            }
        }
    }
}

fn read_optional_text(path: &Path) -> Result<String, HarnessError> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(String::new()),
        Err(source) => Err(HarnessError::io(
            "failed to read captured process output",
            source,
        )),
    }
}

fn summarize_failure(status: TestStatus, stdout: &str, stderr: &str) -> Option<String> {
    if status == TestStatus::Passed {
        return None;
    }

    last_nonempty_line(stderr)
        .or_else(|| last_nonempty_line(stdout))
        .or_else(|| match status {
            TestStatus::TimedOut => Some("test timed out".to_string()),
            _ => None,
        })
}

fn last_nonempty_line(text: &str) -> Option<String> {
    text.lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

fn print_test_progress(index: usize, total: usize, result: &TestResult, quiet: bool) {
    if quiet && result.status == TestStatus::Passed {
        return;
    }

    let label = match result.status {
        TestStatus::Passed => "ok",
        TestStatus::Failed => "FAIL",
        TestStatus::TimedOut => "TIMEOUT",
    };

    println!(
        "[{index:>4}/{total:<4}] {:<40} {:<7} {:>7.2}s",
        result.test_name,
        label,
        result.duration_ms as f64 / 1000.0
    );
}

fn print_captured_output(result: &TestResult, quiet: bool) {
    if quiet && result.status == TestStatus::Passed {
        return;
    }

    let has_output = !result.stdout.trim().is_empty() || !result.stderr.trim().is_empty();
    if !has_output && result.status == TestStatus::Passed {
        return;
    }

    println!("  work dir: {}", display_path(&result.work_dir));
    if !result.stdout.trim().is_empty() {
        println!("  stdout:");
        for line in result.stdout.lines() {
            println!("    {line}");
        }
    }
    if !result.stderr.trim().is_empty() {
        println!("  stderr:");
        for line in result.stderr.lines() {
            println!("    {line}");
        }
    }
}

fn print_summary(report: &SuiteReport, quiet: bool) {
    if !quiet {
        println!();
    }
    println!(
        "Selected: {}  Passed: {}  Failed: {}  Timed out: {}  Duration: {:.2}s",
        report.selected,
        report.passed,
        report.failed,
        report.timed_out,
        report.duration_ms as f64 / 1000.0
    );

    if report.failed > 0 || report.timed_out > 0 {
        println!("Failures:");
        for result in report
            .results
            .iter()
            .filter(|result| result.status != TestStatus::Passed)
        {
            let summary = result
                .failure_summary
                .as_deref()
                .unwrap_or("see captured stderr");
            println!("  {}: {}", result.test_name, summary);
        }
    }
}

fn write_json_report(path: &Path, report: &SuiteReport) -> Result<(), HarnessError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|source| HarnessError::io("failed to create JSON report directory", source))?;
    }

    let json = serde_json::to_string_pretty(report)
        .map_err(|source| HarnessError::json("failed to serialize JSON report", source))?;
    fs::write(path, json).map_err(|source| HarnessError::io("failed to write JSON report", source))
}

fn required_string_value(
    raw_args: &[String],
    index: usize,
    option: &'static str,
) -> Result<String, HarnessError> {
    raw_args
        .get(index)
        .cloned()
        .ok_or_else(|| HarnessError::Argument(format!("missing value for {option}")))
}

fn required_path_value(
    raw_args: &[String],
    index: usize,
    option: &'static str,
) -> Result<PathBuf, HarnessError> {
    required_string_value(raw_args, index, option).map(PathBuf::from)
}

fn parse_timeout(value: &str) -> Result<Option<Duration>, HarnessError> {
    let seconds = value.parse::<u64>().map_err(|_| {
        HarnessError::Argument(format!(
            "invalid timeout '{value}', expected a non-negative integer number of seconds"
        ))
    })?;
    if seconds == 0 {
        Ok(None)
    } else {
        Ok(Some(Duration::from_secs(seconds)))
    }
}

fn canonicalize_existing_dir(path: &Path, label: &'static str) -> Result<PathBuf, HarnessError> {
    let canonical = path
        .canonicalize()
        .map_err(|source| HarnessError::io("failed to resolve configured path", source))?;
    if !canonical.is_dir() {
        return Err(HarnessError::InvalidConfiguration(format!(
            "{label} is not a directory: {}",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn split_name(name: &str) -> (&str, Option<&str>) {
    match name.rsplit_once('.') {
        Some((stem, extension)) => (stem, Some(extension)),
        None => (name, None),
    }
}

fn resolve_module_name(test_name: &str, prefix_test_package: bool) -> String {
    if prefix_test_package && !test_name.starts_with("test.") {
        format!("test.{test_name}")
    } else {
        test_name.to_string()
    }
}

fn strip_py_suffix(name: &str) -> String {
    match name.strip_suffix(".py") {
        Some(stripped) => stripped.to_string(),
        None => name.to_string(),
    }
}

fn split_test_dir_names() -> &'static BTreeSet<&'static str> {
    static INIT: std::sync::OnceLock<BTreeSet<&'static str>> = std::sync::OnceLock::new();
    INIT.get_or_init(|| SPLIT_TEST_DIRS.iter().copied().collect())
}

fn default_work_root() -> PathBuf {
    let mut path = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis();
    path.push(format!(
        "prism-cpython-tests-{}-{}",
        std::process::id(),
        stamp
    ));
    path
}

fn sanitize_path_component(name: &str) -> String {
    let mut sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        sanitized.push_str("test");
    }
    if sanitized.len() > MAX_WORKDIR_COMPONENT_LEN {
        sanitized.truncate(MAX_WORKDIR_COMPONENT_LEN);
    }
    sanitized
}

fn display_path(path: &Path) -> String {
    let rendered = path.display().to_string();
    rendered.replacen("\\\\?\\", "", 1)
}

fn recreate_directory(path: &Path) -> Result<(), HarnessError> {
    if path.exists() {
        fs::remove_dir_all(path).map_err(|source| {
            HarnessError::io("failed to clear per-test work directory", source)
        })?;
    }
    fs::create_dir_all(path)
        .map_err(|source| HarnessError::io("failed to create per-test work directory", source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

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
                "prism_cpython_tests_{}_{}_{}",
                std::process::id(),
                nanos,
                unique
            ));
            fs::create_dir_all(&path).expect("failed to create temp dir");
            Self { path }
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_file(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("failed to create parent dirs");
        }
        fs::write(path, content).expect("failed to write test file");
    }

    fn make_cli_args(root: &Path) -> CliArgs {
        CliArgs {
            cpython_root: root.to_path_buf(),
            prism_executable: Some(root.join("bin").join("prism.exe")),
            work_dir: Some(root.join("work")),
            ..CliArgs::default()
        }
    }

    fn make_config(root: &Path) -> HarnessConfig {
        let prism = root.join("bin").join("prism.exe");
        write_file(&prism, "");
        HarnessConfig::from_cli(&make_cli_args(root)).expect("config should resolve")
    }

    #[derive(Default)]
    struct FakeExecutor {
        plans: BTreeMap<String, (TestStatus, Option<i32>, &'static str, &'static str)>,
        calls: Mutex<Vec<String>>,
    }

    impl FakeExecutor {
        fn with_plan(
            mut self,
            name: &str,
            status: TestStatus,
            exit_code: Option<i32>,
            stdout: &'static str,
            stderr: &'static str,
        ) -> Self {
            self.plans
                .insert(name.to_string(), (status, exit_code, stdout, stderr));
            self
        }
    }

    impl TestExecutor for FakeExecutor {
        fn run(
            &self,
            test: &DiscoveredTest,
            config: &HarnessConfig,
        ) -> Result<TestResult, HarnessError> {
            self.calls
                .lock()
                .expect("call log should lock")
                .push(test.name.clone());
            let (status, exit_code, stdout, stderr) = self
                .plans
                .get(&test.name)
                .cloned()
                .unwrap_or((TestStatus::Passed, Some(0), "", ""));

            Ok(TestResult {
                test_name: test.name.clone(),
                module: test.module.clone(),
                path: test.path.clone(),
                work_dir: config.work_root.join(sanitize_path_component(&test.name)),
                status,
                exit_code,
                duration_ms: 10,
                stdout: stdout.to_string(),
                stderr: stderr.to_string(),
                command: vec!["prism".to_string(), "bootstrap.py".to_string()],
                failure_summary: summarize_failure(status, stdout, stderr),
            })
        }
    }

    #[test]
    fn test_parse_cli_action_help() {
        let action = parse_cli_action(&["--help".to_string()]).expect("parse should succeed");
        assert_eq!(action, CliAction::Help);
    }

    #[test]
    fn test_parse_cli_action_requires_cpython_root() {
        let err = parse_cli_action(&["test_math".to_string()]).expect_err("parse should fail");
        assert!(err.to_string().contains("--cpython-root"));
    }

    #[test]
    fn test_parse_cli_action_parses_runner_and_prism_args() {
        let action = parse_cli_action(&[
            "--cpython-root".to_string(),
            "C:\\cpython".to_string(),
            "--runner".to_string(),
            "import".to_string(),
            "--prism".to_string(),
            "C:\\bin\\prism.exe".to_string(),
            "--prism-arg".to_string(),
            "--x".to_string(),
            "--timeout".to_string(),
            "0".to_string(),
            "test_math".to_string(),
        ])
        .expect("parse should succeed");

        let CliAction::Run(args) = action else {
            panic!("expected run action");
        };
        assert_eq!(args.runner, RunnerMode::Import);
        assert_eq!(
            args.prism_executable,
            Some(PathBuf::from("C:\\bin\\prism.exe"))
        );
        assert_eq!(args.prism_args, vec!["--x"]);
        assert_eq!(args.timeout, None);
        assert_eq!(args.tests, vec!["test_math"]);
    }

    #[test]
    fn test_render_bootstrap_import_mode_is_minimal() {
        let script = render_bootstrap_script("test.test_math", RunnerMode::Import, false);
        assert!(script.contains("import test.test_math as module"));
        assert!(!script.contains("unittest"));
    }

    #[test]
    fn test_render_bootstrap_suite_mode_uses_unittest() {
        let script = render_bootstrap_script("test.test_math", RunnerMode::Suite, true);
        assert!(script.contains("import unittest"));
        assert!(script.contains("verbosity=2"));
        assert!(script.contains("test_main"));
    }

    #[test]
    fn test_discover_tests_finds_top_level_modules() {
        let temp = TestTempDir::new();
        write_file(
            &temp.path.join("Lib").join("test").join("test_alpha.py"),
            "",
        );
        write_file(&temp.path.join("Lib").join("test").join("helper.py"), "");
        write_file(
            &temp
                .path
                .join("Lib")
                .join("test")
                .join("test_package")
                .join("__init__.py"),
            "",
        );

        let config = make_config(&temp.path);
        let discovered = discover_tests(&config).expect("discovery should succeed");
        let names = discovered
            .into_iter()
            .map(|test| test.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["test_alpha", "test_package"]);
    }

    #[test]
    fn test_discover_tests_splits_selected_packages() {
        let temp = TestTempDir::new();
        write_file(
            &temp
                .path
                .join("Lib")
                .join("test")
                .join("test_asyncio")
                .join("test_events.py"),
            "",
        );
        write_file(
            &temp
                .path
                .join("Lib")
                .join("test")
                .join("test_asyncio")
                .join("test_tasks.py"),
            "",
        );

        let config = make_config(&temp.path);
        let discovered = discover_tests(&config).expect("discovery should succeed");
        let names = discovered
            .into_iter()
            .map(|test| test.name)
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                "test.test_asyncio.test_events",
                "test.test_asyncio.test_tasks",
            ]
        );
    }

    #[test]
    fn test_select_tests_expands_split_package_request() {
        let discovered = vec![
            DiscoveredTest {
                name: "test.test_asyncio.test_events".to_string(),
                module: "test.test_asyncio.test_events".to_string(),
                path: PathBuf::from("test_events.py"),
            },
            DiscoveredTest {
                name: "test.test_asyncio.test_tasks".to_string(),
                module: "test.test_asyncio.test_tasks".to_string(),
                path: PathBuf::from("test_tasks.py"),
            },
        ];

        let selected =
            select_tests(&discovered, &[String::from("test_asyncio")], None).expect("selection");
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_tests_applies_start_cutoff() {
        let discovered = vec![
            DiscoveredTest {
                name: "test_alpha".to_string(),
                module: "test.test_alpha".to_string(),
                path: PathBuf::from("a"),
            },
            DiscoveredTest {
                name: "test_beta".to_string(),
                module: "test.test_beta".to_string(),
                path: PathBuf::from("b"),
            },
            DiscoveredTest {
                name: "test_gamma".to_string(),
                module: "test.test_gamma".to_string(),
                path: PathBuf::from("c"),
            },
        ];

        let selected = select_tests(&discovered, &[], Some("test_beta")).expect("selection");
        assert_eq!(
            selected
                .into_iter()
                .map(|test| test.name)
                .collect::<Vec<_>>(),
            vec!["test_beta", "test_gamma"]
        );
    }

    #[test]
    fn test_run_plan_aggregates_results() {
        let temp = TestTempDir::new();
        write_file(
            &temp.path.join("Lib").join("test").join("test_alpha.py"),
            "",
        );
        write_file(&temp.path.join("Lib").join("test").join("test_beta.py"), "");
        let config = make_config(&temp.path);
        let harness = Harness::new(config);
        let plan = harness.plan(&[], None).expect("plan should build");
        let executor = FakeExecutor::default()
            .with_plan("test_alpha", TestStatus::Passed, Some(0), "", "")
            .with_plan(
                "test_beta",
                TestStatus::Failed,
                Some(1),
                "",
                "AttributeError: missing builtin",
            );

        let report = harness
            .run_plan(plan, &executor)
            .expect("run should succeed");

        assert_eq!(report.selected, 2);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert_eq!(report.timed_out, 0);
        assert_eq!(report.exit_code(), 1);
        assert_eq!(
            executor.calls.lock().expect("calls").clone(),
            vec!["test_alpha", "test_beta"]
        );
    }

    #[test]
    fn test_run_plan_respects_fail_fast() {
        let temp = TestTempDir::new();
        let mut cli = make_cli_args(&temp.path);
        cli.fail_fast = true;
        write_file(
            &temp.path.join("Lib").join("test").join("test_alpha.py"),
            "",
        );
        write_file(&temp.path.join("Lib").join("test").join("test_beta.py"), "");
        write_file(&temp.path.join("bin").join("prism.exe"), "");

        let config = HarnessConfig::from_cli(&cli).expect("config should resolve");
        let harness = Harness::new(config);
        let plan = harness.plan(&[], None).expect("plan should build");
        let executor = FakeExecutor::default()
            .with_plan("test_alpha", TestStatus::Failed, Some(1), "", "Traceback")
            .with_plan("test_beta", TestStatus::Passed, Some(0), "", "");

        let report = harness
            .run_plan(plan, &executor)
            .expect("run should succeed");
        assert_eq!(report.results.len(), 1);
        assert_eq!(
            executor.calls.lock().expect("calls").clone(),
            vec!["test_alpha"]
        );
    }

    #[test]
    fn test_execute_cli_with_executor_writes_json_report() {
        let temp = TestTempDir::new();
        let report_path = temp.path.join("reports").join("suite.json");
        write_file(
            &temp.path.join("Lib").join("test").join("test_alpha.py"),
            "",
        );

        let mut cli = make_cli_args(&temp.path);
        cli.json_report = Some(report_path.clone());
        write_file(&temp.path.join("bin").join("prism.exe"), "");
        let executor = FakeExecutor::default();

        let report = execute_cli_with_executor(&cli, &executor).expect("run should succeed");
        assert_eq!(report.exit_code(), 0);

        let json = fs::read_to_string(report_path).expect("report should be written");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("report should be valid JSON");
        assert_eq!(parsed["selected"], 1);
        assert_eq!(parsed["passed"], 1);
    }

    #[test]
    fn test_sanitize_path_component_truncates_and_normalizes() {
        let sanitized = sanitize_path_component("test weird/name with spaces and punctuation!!!");
        assert!(!sanitized.contains(' '));
        assert!(!sanitized.contains('/'));
        assert!(sanitized.len() <= MAX_WORKDIR_COMPONENT_LEN);
    }

    #[test]
    fn test_display_path_strips_windows_extended_prefix() {
        let path = Path::new(r"\\?\C:\Users\James\Desktop\cpython-3.12\Lib\test");
        assert_eq!(
            display_path(path),
            r"C:\Users\James\Desktop\cpython-3.12\Lib\test"
        );
    }
}
