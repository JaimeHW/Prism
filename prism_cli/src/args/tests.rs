use super::*;

/// Helper to parse from a slice of string slices (skipping program name).
fn parse(args: &[&str]) -> Result<PrismArgs, ArgError> {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    parse_args_vec(&args)
}

// =========================================================================
// Execution Mode Tests
// =========================================================================

#[test]
fn test_no_args_starts_repl() {
    let result = parse(&[]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Repl);
    assert!(result.script_args.is_empty());
}

#[test]
fn test_script_file() {
    let result = parse(&["test.py"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Script(PathBuf::from("test.py")));
    assert_eq!(result.script_args, vec!["test.py"]);
}

#[test]
fn test_script_file_with_args() {
    let result = parse(&["test.py", "arg1", "arg2", "--flag"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Script(PathBuf::from("test.py")));
    assert_eq!(
        result.script_args,
        vec!["test.py", "arg1", "arg2", "--flag"]
    );
}

#[test]
fn test_command_mode() {
    let result = parse(&["-c", "print('hello')"]).unwrap();
    assert_eq!(
        result.mode,
        ExecutionMode::Command("print('hello')".to_string())
    );
    assert_eq!(result.script_args, vec!["-c"]);
}

#[test]
fn test_command_mode_with_args() {
    let result = parse(&["-c", "import sys; print(sys.argv)", "foo", "bar"]).unwrap();
    assert_eq!(
        result.mode,
        ExecutionMode::Command("import sys; print(sys.argv)".to_string())
    );
    assert_eq!(result.script_args, vec!["-c", "foo", "bar"]);
}

#[test]
fn test_command_mode_bundled() {
    let result = parse(&["-cprint(1)"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Command("print(1)".to_string()));
}

#[test]
fn test_command_mode_missing_value() {
    let result = parse(&["-c"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ArgError::MissingValue("-c"));
}

#[test]
fn test_module_mode() {
    let result = parse(&["-m", "json.tool"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Module("json.tool".to_string()));
    assert_eq!(result.script_args, vec!["json.tool"]);
}

#[test]
fn test_module_mode_with_args() {
    let result = parse(&["-m", "http.server", "8080"]).unwrap();
    assert_eq!(
        result.mode,
        ExecutionMode::Module("http.server".to_string())
    );
    assert_eq!(result.script_args, vec!["http.server", "8080"]);
}

#[test]
fn test_module_mode_bundled() {
    let result = parse(&["-mjson.tool"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Module("json.tool".to_string()));
}

#[test]
fn test_module_mode_missing_value() {
    let result = parse(&["-m"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ArgError::MissingValue("-m"));
}

#[test]
fn test_stdin_mode() {
    let result = parse(&["-"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Stdin);
    assert_eq!(result.script_args, vec!["-"]);
}

#[test]
fn test_stdin_mode_with_args() {
    let result = parse(&["-", "arg1", "arg2"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Stdin);
    assert_eq!(result.script_args, vec!["-", "arg1", "arg2"]);
}

#[test]
fn test_version_short() {
    let result = parse(&["-V"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::PrintVersion);
}

#[test]
fn test_version_long() {
    let result = parse(&["--version"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::PrintVersion);
}

#[test]
fn test_help_short() {
    let result = parse(&["-h"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::PrintHelp);
}

#[test]
fn test_help_long() {
    let result = parse(&["--help"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::PrintHelp);
}

// =========================================================================
// Boolean Flag Tests
// =========================================================================

#[test]
fn test_inspect_flag() {
    let result = parse(&["-i"]).unwrap();
    assert!(result.inspect);
    assert_eq!(result.mode, ExecutionMode::Repl);
}

#[test]
fn test_unbuffered_flag() {
    let result = parse(&["-u", "test.py"]).unwrap();
    assert!(result.unbuffered);
}

#[test]
fn test_verbose_single() {
    let result = parse(&["-v"]).unwrap();
    assert_eq!(result.verbose, 1);
}

#[test]
fn test_verbose_double() {
    let result = parse(&["-v", "-v"]).unwrap();
    assert_eq!(result.verbose, 2);
}

#[test]
fn test_verbose_bundled() {
    let result = parse(&["-vvv"]).unwrap();
    assert_eq!(result.verbose, 3);
}

#[test]
fn test_quiet_flag() {
    let result = parse(&["-q"]).unwrap();
    assert!(result.quiet);
}

#[test]
fn test_dont_write_bytecode() {
    let result = parse(&["-B"]).unwrap();
    assert!(result.dont_write_bytecode);
}

#[test]
fn test_ignore_environment() {
    let result = parse(&["-E"]).unwrap();
    assert!(result.ignore_environment);
}

#[test]
fn test_isolated_mode_implies_environment_and_user_site_isolation() {
    let result = parse(&["-I"]).unwrap();
    assert!(result.isolated);
    assert!(result.ignore_environment);
    assert!(result.no_user_site);
}

#[test]
fn test_no_user_site() {
    let result = parse(&["-s"]).unwrap();
    assert!(result.no_user_site);
}

#[test]
fn test_no_site() {
    let result = parse(&["-S"]).unwrap();
    assert!(result.no_site);
}

#[test]
fn test_debug_flag() {
    let result = parse(&["-d"]).unwrap();
    assert!(result.debug);
}

// =========================================================================
// Optimization Level Tests
// =========================================================================

#[test]
fn test_optimize_default() {
    let result = parse(&[]).unwrap();
    assert_eq!(result.optimize, OptimizationLevel::None);
}

#[test]
fn test_optimize_basic() {
    let result = parse(&["-O"]).unwrap();
    assert_eq!(result.optimize, OptimizationLevel::Basic);
}

#[test]
fn test_optimize_full() {
    let result = parse(&["-OO"]).unwrap();
    assert_eq!(result.optimize, OptimizationLevel::Full);
}

#[test]
fn test_optimize_separate_flags() {
    let result = parse(&["-O", "-O"]).unwrap();
    assert_eq!(result.optimize, OptimizationLevel::Full);
}

#[test]
fn test_optimize_triple_clamps() {
    let result = parse(&["-O", "-O", "-O"]).unwrap();
    assert_eq!(result.optimize, OptimizationLevel::Full);
}

// =========================================================================
// Warning Filter Tests
// =========================================================================

#[test]
fn test_warning_single() {
    let result = parse(&["-W", "error"]).unwrap();
    assert_eq!(result.warnings.len(), 1);
    assert_eq!(result.warnings[0].spec, "error");
}

#[test]
fn test_warning_multiple() {
    let result = parse(&["-W", "error", "-W", "ignore::DeprecationWarning"]).unwrap();
    assert_eq!(result.warnings.len(), 2);
    assert_eq!(result.warnings[0].spec, "error");
    assert_eq!(result.warnings[1].spec, "ignore::DeprecationWarning");
}

#[test]
fn test_warning_bundled() {
    let result = parse(&["-Werror"]).unwrap();
    assert_eq!(result.warnings.len(), 1);
    assert_eq!(result.warnings[0].spec, "error");
}

#[test]
fn test_warning_missing_value() {
    let result = parse(&["-W"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ArgError::MissingValue("-W"));
}

// =========================================================================
// X-Option Tests
// =========================================================================

#[test]
fn test_x_option() {
    let result = parse(&["-X", "utf8"]).unwrap();
    assert_eq!(result.x_options, vec!["utf8"]);
}

#[test]
fn test_x_option_bundled() {
    let result = parse(&["-Xutf8"]).unwrap();
    assert_eq!(result.x_options, vec!["utf8"]);
}

#[test]
fn test_x_option_multiple() {
    let result = parse(&["-X", "utf8", "-X", "dev"]).unwrap();
    assert_eq!(result.x_options, vec!["utf8", "dev"]);
}

#[test]
fn test_x_option_missing_value() {
    let result = parse(&["-X"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ArgError::MissingValue("-X"));
}

// =========================================================================
// Bundled Flag Tests
// =========================================================================

#[test]
fn test_bundled_flags() {
    let result = parse(&["-BEsu", "test.py"]).unwrap();
    assert!(result.dont_write_bytecode);
    assert!(result.ignore_environment);
    assert!(result.no_user_site);
    assert!(result.unbuffered);
    assert_eq!(result.mode, ExecutionMode::Script(PathBuf::from("test.py")));
}

#[test]
fn test_bundled_flags_with_command() {
    let result = parse(&["-Oqc", "print(1)"]).unwrap();
    assert_eq!(result.optimize, OptimizationLevel::Basic);
    assert!(result.quiet);
    assert_eq!(result.mode, ExecutionMode::Command("print(1)".to_string()));
}

#[test]
fn test_bundled_verbose_and_others() {
    let result = parse(&["-vvBEI"]).unwrap();
    assert_eq!(result.verbose, 2);
    assert!(result.dont_write_bytecode);
    assert!(result.ignore_environment);
    assert!(result.isolated);
    assert!(result.no_user_site);
}

// =========================================================================
// Double-Dash Terminator Tests
// =========================================================================

#[test]
fn test_double_dash_then_script() {
    let result = parse(&["--", "script.py", "arg1"]).unwrap();
    assert_eq!(
        result.mode,
        ExecutionMode::Script(PathBuf::from("script.py"))
    );
    assert_eq!(result.script_args, vec!["script.py", "arg1"]);
}

#[test]
fn test_double_dash_prevents_flag_parsing() {
    let result = parse(&["--", "-c", "not_a_command"]).unwrap();
    // After --, `-c` is treated as a script filename.
    assert_eq!(result.mode, ExecutionMode::Script(PathBuf::from("-c")));
    assert_eq!(result.script_args, vec!["-c", "not_a_command"]);
}

#[test]
fn test_flags_before_double_dash() {
    let result = parse(&["-B", "-E", "--", "script.py"]).unwrap();
    assert!(result.dont_write_bytecode);
    assert!(result.ignore_environment);
    assert_eq!(
        result.mode,
        ExecutionMode::Script(PathBuf::from("script.py"))
    );
}

// =========================================================================
// Error Tests
// =========================================================================

#[test]
fn test_unknown_flag() {
    let result = parse(&["-Z"]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ArgError::UnknownFlag(f) => assert_eq!(f, "-Z"),
        other => panic!("Expected UnknownFlag, got {:?}", other),
    }
}

#[test]
fn test_unknown_long_flag() {
    let result = parse(&["--foobar"]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ArgError::UnknownFlag(f) => assert!(f.contains("foobar")),
        other => panic!("Expected UnknownFlag, got {:?}", other),
    }
}

// =========================================================================
// Complex Scenario Tests
// =========================================================================

#[test]
fn test_realistic_debug_session() {
    let result = parse(&["-i", "-v", "debug.py", "input.txt"]).unwrap();
    assert!(result.inspect);
    assert_eq!(result.verbose, 1);
    assert_eq!(
        result.mode,
        ExecutionMode::Script(PathBuf::from("debug.py"))
    );
    assert_eq!(result.script_args, vec!["debug.py", "input.txt"]);
}

#[test]
fn test_realistic_production_run() {
    let result = parse(&["-OO", "-B", "-u", "-E", "server.py", "--port", "8080"]).unwrap();
    assert_eq!(result.optimize, OptimizationLevel::Full);
    assert!(result.dont_write_bytecode);
    assert!(result.unbuffered);
    assert!(result.ignore_environment);
    assert_eq!(
        result.mode,
        ExecutionMode::Script(PathBuf::from("server.py"))
    );
    assert_eq!(result.script_args, vec!["server.py", "--port", "8080"]);
}

#[test]
fn test_realistic_warning_control() {
    let result = parse(&[
        "-W",
        "error::ResourceWarning",
        "-W",
        "default::DeprecationWarning",
        "-c",
        "import warnings",
    ])
    .unwrap();
    assert_eq!(result.warnings.len(), 2);
    assert_eq!(result.warnings[0].spec, "error::ResourceWarning");
    assert_eq!(result.warnings[1].spec, "default::DeprecationWarning");
    assert_eq!(
        result.mode,
        ExecutionMode::Command("import warnings".to_string())
    );
}

#[test]
fn test_script_path_with_dashes() {
    // Script name that looks like flags but isn't because it doesn't start with `-`
    let result = parse(&["my-script.py"]).unwrap();
    assert_eq!(
        result.mode,
        ExecutionMode::Script(PathBuf::from("my-script.py"))
    );
}

#[test]
fn test_script_args_after_script_include_dashes() {
    let result = parse(&["run.py", "-v", "--verbose"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Script(PathBuf::from("run.py")));
    // -v and --verbose should NOT be interpreted as prism flags.
    assert_eq!(result.script_args, vec!["run.py", "-v", "--verbose"]);
    assert_eq!(result.verbose, 0);
}

// =========================================================================
// Version/Help String Tests
// =========================================================================

#[test]
fn test_version_string_format() {
    let vs = version_string();
    assert!(vs.starts_with("Prism"));
    assert!(vs.contains("Python"));
    assert!(vs.contains("compatible"));
}

#[test]
fn test_help_text_contains_flags() {
    let ht = help_text();
    assert!(ht.contains("-c cmd"));
    assert!(ht.contains("-m mod"));
    assert!(ht.contains("-V"));
    assert!(ht.contains("-h"));
    assert!(ht.contains("-i"));
    assert!(ht.contains("-u"));
    assert!(ht.contains("-B"));
    assert!(ht.contains("-E"));
    assert!(ht.contains("-I"));
    assert!(ht.contains("-O"));
    assert!(ht.contains("-W arg"));
    assert!(ht.contains("-X opt"));
}

// =========================================================================
// ArgError Display Tests
// =========================================================================

#[test]
fn test_arg_error_missing_value_display() {
    let err = ArgError::MissingValue("-c");
    assert_eq!(err.to_string(), "Argument expected for the -c option");
}

#[test]
fn test_arg_error_unknown_flag_display() {
    let err = ArgError::UnknownFlag("-Z".to_string());
    assert_eq!(err.to_string(), "Unknown option: -Z");
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_empty_command_string() {
    let result = parse(&["-c", ""]).unwrap();
    assert_eq!(result.mode, ExecutionMode::Command(String::new()));
}

#[test]
fn test_version_ignores_subsequent_args() {
    let result = parse(&["-V", "script.py"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::PrintVersion);
}

#[test]
fn test_help_ignores_subsequent_args() {
    let result = parse(&["-h", "script.py"]).unwrap();
    assert_eq!(result.mode, ExecutionMode::PrintHelp);
}

#[test]
fn test_inspect_with_command() {
    let result = parse(&["-i", "-c", "x = 1"]).unwrap();
    assert!(result.inspect);
    assert_eq!(result.mode, ExecutionMode::Command("x = 1".to_string()));
}

#[test]
fn test_default_args() {
    let d = PrismArgs::default();
    assert_eq!(d.mode, ExecutionMode::Repl);
    assert!(!d.inspect);
    assert!(!d.unbuffered);
    assert_eq!(d.verbose, 0);
    assert!(!d.quiet);
    assert_eq!(d.optimize, OptimizationLevel::None);
    assert!(!d.dont_write_bytecode);
    assert!(!d.ignore_environment);
    assert!(!d.isolated);
    assert!(!d.no_user_site);
    assert!(!d.no_site);
    assert!(d.warnings.is_empty());
    assert!(d.x_options.is_empty());
    assert!(!d.debug);
    assert!(d.script_args.is_empty());
}
