//! CPython-compatible command-line argument parser.
//!
//! Hand-rolled for zero-overhead startup — matches CPython's own parser strategy.
//! Supports all standard Python interpreter flags for drop-in compatibility.

use std::ffi::OsString;
use std::path::PathBuf;

// =============================================================================
// Execution Mode
// =============================================================================

/// What Prism should execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Run a script file: `prism script.py [args...]`
    Script(PathBuf),
    /// Run a command string: `prism -c "print('hello')"`
    Command(String),
    /// Run a module: `prism -m module_name`
    Module(String),
    /// Read from stdin: `echo "print(1)" | prism` or `prism -`
    Stdin,
    /// Interactive REPL: `prism` with no arguments
    Repl,
    /// Print version and exit: `prism -V` or `prism --version`
    PrintVersion,
    /// Print help and exit: `prism -h` or `prism --help`
    PrintHelp,
}

// =============================================================================
// Warning Action
// =============================================================================

/// Warning control action (from `-W` flag).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarningFilter {
    /// Raw filter specification (e.g., "error", "ignore::DeprecationWarning").
    pub spec: String,
}

// =============================================================================
// Optimization Level
// =============================================================================

/// Optimization level matching CPython's `-O` / `-OO` flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationLevel {
    /// No optimization (default).
    None = 0,
    /// `-O`: Remove assert statements and `__debug__`-dependent code.
    Basic = 1,
    /// `-OO`: `-O` plus remove docstrings.
    Full = 2,
}

impl Default for OptimizationLevel {
    #[inline]
    fn default() -> Self {
        Self::None
    }
}

// =============================================================================
// Parsed Arguments
// =============================================================================

/// Complete set of parsed CLI arguments.
///
/// Mirror's CPython's argument structure for full drop-in compatibility.
#[derive(Debug, Clone)]
pub struct PrismArgs {
    /// What to execute.
    pub mode: ExecutionMode,

    /// Arguments to pass to the script as `sys.argv`.
    /// For scripts, `sys.argv[0]` is the script path.
    /// For `-c`, `sys.argv[0]` is `"-c"`.
    /// For `-m`, `sys.argv[0]` is the module name.
    pub script_args: Vec<String>,

    /// `-i`: Enter interactive mode after running script/command.
    pub inspect: bool,

    /// `-u`: Force unbuffered stdout and stderr.
    pub unbuffered: bool,

    /// `-v`: Verbose import tracing.
    pub verbose: u32,

    /// `-q`: Quiet mode (suppress version banner in REPL).
    pub quiet: bool,

    /// `-O` / `-OO`: Optimization level.
    pub optimize: OptimizationLevel,

    /// `-B`: Don't write `.pyc` files (no-op for Prism, accepted for compat).
    pub dont_write_bytecode: bool,

    /// `-E`: Ignore `PYTHON*` environment variables.
    pub ignore_environment: bool,

    /// `-s`: Don't add user site-packages to `sys.path`.
    pub no_user_site: bool,

    /// `-S`: Don't import the `site` module.
    pub no_site: bool,

    /// `-W <arg>`: Warning filters, in order of specification.
    pub warnings: Vec<WarningFilter>,

    /// `-X <option>`: Implementation-specific options.
    pub x_options: Vec<String>,

    /// `-d`: Debug mode (parser debugging).
    pub debug: bool,
}

impl Default for PrismArgs {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Repl,
            script_args: Vec::new(),
            inspect: false,
            unbuffered: false,
            verbose: 0,
            quiet: false,
            optimize: OptimizationLevel::None,
            dont_write_bytecode: false,
            ignore_environment: false,
            no_user_site: false,
            no_site: false,
            warnings: Vec::new(),
            x_options: Vec::new(),
            debug: false,
        }
    }
}

// =============================================================================
// Parse Error
// =============================================================================

/// Error during argument parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgError {
    /// Missing required argument value (e.g., `-c` without command).
    MissingValue(&'static str),
    /// Unknown flag.
    UnknownFlag(String),
}

impl std::fmt::Display for ArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgError::MissingValue(flag) => {
                write!(f, "Argument expected for the {} option", flag)
            }
            ArgError::UnknownFlag(flag) => {
                write!(f, "Unknown option: {}", flag)
            }
        }
    }
}

impl std::error::Error for ArgError {}

// =============================================================================
// Parser Entry Point
// =============================================================================

/// Parse command-line arguments into `PrismArgs`.
///
/// This is the primary entry point. Follows CPython's argument parsing
/// semantics exactly:
///
/// 1. Options are parsed left-to-right until a non-option or `--` is found.
/// 2. After `-c <cmd>` or `-m <module>`, all remaining args go to `sys.argv`.
/// 3. After a script path, all remaining args go to `sys.argv`.
/// 4. `-` means read from stdin.
/// 5. If no mode is specified, enter REPL.
pub fn parse_args<I, S>(args: I) -> Result<PrismArgs, ArgError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let args: Vec<String> = args
        .into_iter()
        .map(|s| s.into().to_string_lossy().into_owned())
        .collect();

    parse_args_vec(&args)
}

/// Parse from a pre-collected `Vec<String>`.
///
/// The first element should be the first argument (NOT the program name).
/// The caller is responsible for skipping `argv[0]` (the program name).
pub fn parse_args_vec(args: &[String]) -> Result<PrismArgs, ArgError> {
    let mut result = PrismArgs::default();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        // `--` terminates option parsing; rest goes to script_args.
        if arg == "--" {
            i += 1;
            // If no mode set yet, remaining args are script path + script args.
            if result.mode == ExecutionMode::Repl && i < args.len() {
                result.mode = ExecutionMode::Script(PathBuf::from(&args[i]));
                result.script_args.push(args[i].clone());
                i += 1;
            }
            // Collect remaining as script_args.
            while i < args.len() {
                result.script_args.push(args[i].clone());
                i += 1;
            }
            break;
        }

        // Non-option: treat as script path.
        if !arg.starts_with('-') || arg == "-" {
            if arg == "-" {
                result.mode = ExecutionMode::Stdin;
                result.script_args.push("-".to_string());
            } else {
                result.mode = ExecutionMode::Script(PathBuf::from(arg));
                result.script_args.push(arg.clone());
            }
            i += 1;
            // Remaining args go to sys.argv.
            while i < args.len() {
                result.script_args.push(args[i].clone());
                i += 1;
            }
            break;
        }

        // Option parsing: handle `-xyz` bundled short options.
        let flag_chars: Vec<char> = arg[1..].chars().collect();
        let mut j = 0;

        while j < flag_chars.len() {
            match flag_chars[j] {
                'V' => {
                    // `-V` or `--version`
                    result.mode = ExecutionMode::PrintVersion;
                    return Ok(result);
                }
                'h' => {
                    result.mode = ExecutionMode::PrintHelp;
                    return Ok(result);
                }
                'c' => {
                    // `-c <cmd>`: next arg (or rest of this arg) is the command.
                    let cmd = if j + 1 < flag_chars.len() {
                        // Bundled: `-cprint(1)` → command is `print(1)`
                        flag_chars[j + 1..].iter().collect::<String>()
                    } else {
                        // Separate: `-c "print(1)"` → next arg is the command
                        i += 1;
                        if i >= args.len() {
                            return Err(ArgError::MissingValue("-c"));
                        }
                        args[i].clone()
                    };
                    result.mode = ExecutionMode::Command(cmd);
                    result.script_args.push("-c".to_string());
                    i += 1;
                    // Remaining args go to sys.argv.
                    while i < args.len() {
                        result.script_args.push(args[i].clone());
                        i += 1;
                    }
                    return Ok(result);
                }
                'm' => {
                    // `-m <module>`: next arg is the module name.
                    let module = if j + 1 < flag_chars.len() {
                        flag_chars[j + 1..].iter().collect::<String>()
                    } else {
                        i += 1;
                        if i >= args.len() {
                            return Err(ArgError::MissingValue("-m"));
                        }
                        args[i].clone()
                    };
                    result.mode = ExecutionMode::Module(module.clone());
                    result.script_args.push(module);
                    i += 1;
                    while i < args.len() {
                        result.script_args.push(args[i].clone());
                        i += 1;
                    }
                    return Ok(result);
                }
                'W' => {
                    // `-W <arg>`: warning filter.
                    let spec = if j + 1 < flag_chars.len() {
                        flag_chars[j + 1..].iter().collect::<String>()
                    } else {
                        i += 1;
                        if i >= args.len() {
                            return Err(ArgError::MissingValue("-W"));
                        }
                        args[i].clone()
                    };
                    result.warnings.push(WarningFilter { spec });
                    // Consumed rest of bundled chars.
                    j = flag_chars.len();
                    continue;
                }
                'X' => {
                    // `-X <option>`: implementation-specific option.
                    let opt = if j + 1 < flag_chars.len() {
                        flag_chars[j + 1..].iter().collect::<String>()
                    } else {
                        i += 1;
                        if i >= args.len() {
                            return Err(ArgError::MissingValue("-X"));
                        }
                        args[i].clone()
                    };
                    result.x_options.push(opt);
                    j = flag_chars.len();
                    continue;
                }
                'i' => result.inspect = true,
                'u' => result.unbuffered = true,
                'v' => result.verbose = result.verbose.saturating_add(1),
                'q' => result.quiet = true,
                'O' => {
                    result.optimize = match result.optimize {
                        OptimizationLevel::None => OptimizationLevel::Basic,
                        _ => OptimizationLevel::Full,
                    };
                }
                'B' => result.dont_write_bytecode = true,
                'E' => result.ignore_environment = true,
                's' => result.no_user_site = true,
                'S' => result.no_site = true,
                'd' => result.debug = true,
                '-' => {
                    // Long option: `--version`, `--help`.
                    let long_opt: String = flag_chars[j..].iter().collect();
                    match long_opt.as_str() {
                        "-version" => {
                            result.mode = ExecutionMode::PrintVersion;
                            return Ok(result);
                        }
                        "-help" => {
                            result.mode = ExecutionMode::PrintHelp;
                            return Ok(result);
                        }
                        _ => {
                            return Err(ArgError::UnknownFlag(format!("-{}", long_opt)));
                        }
                    }
                }
                other => {
                    return Err(ArgError::UnknownFlag(format!("-{}", other)));
                }
            }
            j += 1;
        }

        i += 1;
    }

    Ok(result)
}

// =============================================================================
// Version / Help Text
// =============================================================================

/// Build version string matching CPython's format.
///
/// Output: `Prism <version> (Python <py_version> compatible)`
#[inline]
pub fn version_string() -> String {
    format!(
        "Prism {} (Python {}.{}.{} compatible)",
        prism_core::VERSION,
        prism_core::PYTHON_VERSION.0,
        prism_core::PYTHON_VERSION.1,
        prism_core::PYTHON_VERSION.2,
    )
}

/// Build help text matching CPython's format.
pub fn help_text() -> String {
    format!(
        r#"usage: prism [option] ... [-c cmd | -m mod | file | -] [arg] ...
Options (and corresponding environment variables):
-B     : don't write .pyc files on import (PYTHONDONTWRITEBYTECODE=x)
-c cmd : program passed in as string (terminates option list)
-d     : turn on parser debugging output
-E     : ignore PYTHON* environment variables
-h     : print this help message and exit (also --help)
-i     : inspect interactively after running script (PYTHONINSPECT=x)
-m mod : run library module as a script (terminates option list)
-O     : remove assert and __debug__-dependent statements; add .opt-1 before
         .pyc extension; also PYTHONOPTIMIZE=x
-OO    : do -O changes and also discard docstrings; add .opt-2 before
         .pyc extension
-q     : don't print version and copyright messages on interactive startup
-s     : don't add user site-packages directory to sys.path
-S     : don't imply 'import site' on initialization
-u     : force the stdout and stderr streams to be unbuffered;
         this option has no effect on stdin (PYTHONUNBUFFERED=x)
-v     : verbose (trace import statements) (PYTHONVERBOSE=x)
-V     : print the Prism version number and exit (also --version)
-W arg : warning control; arg is action:message:category:module:lineno
-X opt : set implementation-specific option
file   : program read from script file
-      : program read from stdin (default; interactive mode if a tty)
arg ...: arguments passed to program in sys.argv[1:]

Prism {} — Hyper-performant Python runtime
Python {}.{}.{} compatible"#,
        prism_core::VERSION,
        prism_core::PYTHON_VERSION.0,
        prism_core::PYTHON_VERSION.1,
        prism_core::PYTHON_VERSION.2,
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
        let result = parse(&["-vvBE"]).unwrap();
        assert_eq!(result.verbose, 2);
        assert!(result.dont_write_bytecode);
        assert!(result.ignore_environment);
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
    fn test_optimization_level_ord() {
        assert!(OptimizationLevel::None < OptimizationLevel::Basic);
        assert!(OptimizationLevel::Basic < OptimizationLevel::Full);
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
        assert!(!d.no_user_site);
        assert!(!d.no_site);
        assert!(d.warnings.is_empty());
        assert!(d.x_options.is_empty());
        assert!(!d.debug);
        assert!(d.script_args.is_empty());
    }

    #[test]
    fn test_warning_filter_clone() {
        let w = WarningFilter {
            spec: "error".to_string(),
        };
        let w2 = w.clone();
        assert_eq!(w, w2);
    }

    #[test]
    fn test_optimization_level_default() {
        let o: OptimizationLevel = Default::default();
        assert_eq!(o, OptimizationLevel::None);
    }
}
