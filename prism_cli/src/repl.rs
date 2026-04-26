//! Interactive REPL (Read-Eval-Print Loop) matching CPython's behavior.
//!
//! Provides `>>>` and `...` prompts, multiline statement support,
//! expression result display, and error handling without exit.

use crate::config::RuntimeConfig;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;
use std::sync::Arc;

// =============================================================================
// REPL Entry Point
// =============================================================================

/// Start the interactive REPL.
///
/// Behavior matches CPython:
/// - `>>>` primary prompt
/// - `...` continuation prompt for multiline constructs
/// - Displays expression results (non-None)
/// - Errors are printed but don't exit the REPL
/// - `exit()` or `quit()` to exit
/// - Ctrl+D (EOF) exits cleanly
pub fn run_repl(config: &RuntimeConfig) -> ExitCode {
    // Print banner unless `-q` was specified.
    if !config.quiet {
        println!(
            "Prism {} (Python {}.{}.{} compatible)",
            prism_core::VERSION,
            prism_core::PYTHON_VERSION.0,
            prism_core::PYTHON_VERSION.1,
            prism_core::PYTHON_VERSION.2,
        );
        println!("Type \"help\", \"copyright\", \"credits\" or \"license\" for more information.");
    }

    // Create a persistent VM for the REPL session.
    let mut vm = if config.jit_enabled() {
        prism_vm::VirtualMachine::with_jit()
    } else {
        prism_vm::VirtualMachine::new()
    };

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line_buf = String::new();

    loop {
        // Print prompt.
        print!(">>> ");
        if io::stdout().flush().is_err() {
            break;
        }

        // Read first line.
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => {
                // EOF (Ctrl+D).
                println!();
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }

        let trimmed = line_buf.trim();

        // Handle exit commands.
        if trimmed == "exit()" || trimmed == "quit()" {
            break;
        }

        // Handle empty lines.
        if trimmed.is_empty() {
            continue;
        }

        // Collect multiline input for compound statements.
        let mut source = line_buf.clone();
        if needs_continuation(trimmed) {
            loop {
                print!("... ");
                if io::stdout().flush().is_err() {
                    break;
                }
                line_buf.clear();
                match reader.read_line(&mut line_buf) {
                    Ok(0) => break, // EOF
                    Ok(_) => {}
                    Err(_) => break,
                }
                // Empty line terminates multiline input.
                if line_buf.trim().is_empty() {
                    break;
                }
                source.push_str(&line_buf);
            }
        }

        // Execute the input.
        execute_repl_input(&source, &mut vm, config);
    }

    ExitCode::from(crate::error::EXIT_SUCCESS)
}

// =============================================================================
// REPL Execution
// =============================================================================

/// Execute a single REPL input, displaying results or errors.
fn execute_repl_input(source: &str, vm: &mut prism_vm::VirtualMachine, config: &RuntimeConfig) {
    let optimize = crate::pipeline::compiler_optimization_level(config.optimize);
    let code = match prism_compiler::compile_source_code(source, "<stdin>", optimize) {
        Ok(c) => c,
        Err(e) => {
            eprint!(
                "{}",
                crate::error::format_source_compile_error_string(&e, Some(source), "<stdin>")
            );
            return;
        }
    };

    // Execute.
    let main_module = repl_main_module(vm);
    match vm.execute_in_module_runtime(code, main_module) {
        Ok(result) => {
            // Display non-None results (matching CPython REPL behavior).
            if !result.is_none() {
                println!("{}", format_value(&result));
            }
        }
        Err(e) => {
            eprint!(
                "{}",
                crate::error::format_runtime_error_string(&e, Some(source), "<stdin>")
            );
        }
    }
}

fn repl_main_module(vm: &mut prism_vm::VirtualMachine) -> Arc<prism_vm::imports::ModuleObject> {
    if let Some(module) = vm.cached_module("__main__") {
        return module;
    }

    let module = Arc::new(prism_vm::imports::ModuleObject::with_metadata(
        Arc::from("__main__"),
        None,
        Some(Arc::from("<stdin>")),
        Some(Arc::from("")),
    ));
    vm.bind_module(Arc::clone(&module));
    module
}

/// Check if a line needs continuation (starts a compound statement).
///
/// Compound statements that require `...` continuation:
/// - `def`, `class`, `if`, `elif`, `else`, `for`, `while`, `try`, `except`,
///   `finally`, `with`, `match`, `case`
/// - Trailing `:` at end of line
/// - Trailing `\` (explicit line continuation)
/// - Unclosed brackets/parens (simplified: just check for trailing `:`)
#[inline]
fn needs_continuation(line: &str) -> bool {
    let trimmed = line.trim();

    // Explicit line continuation.
    if trimmed.ends_with('\\') {
        return true;
    }

    // Compound statement keywords ending with `:`.
    if trimmed.ends_with(':') {
        let first_word = trimmed.split_whitespace().next().unwrap_or("");
        // Strip trailing `:` from the first word so `"else:"` matches `"else"`.
        let first_word = first_word.strip_suffix(':').unwrap_or(first_word);
        matches!(
            first_word,
            "def"
                | "class"
                | "if"
                | "elif"
                | "else"
                | "for"
                | "while"
                | "try"
                | "except"
                | "finally"
                | "with"
                | "match"
                | "case"
                | "async"
        )
    } else {
        false
    }
}

/// Format a Value for REPL display.
///
/// Matches CPython's `repr()` semantics.
fn format_value(value: &prism_core::Value) -> String {
    if let Some(i) = value.as_int() {
        i.to_string()
    } else if let Some(f) = value.as_float() {
        // Match CPython's float repr.
        if f.fract() == 0.0 && f.is_finite() {
            format!("{:.1}", f)
        } else {
            format!("{}", f)
        }
    } else if let Some(b) = value.as_bool() {
        if b { "True" } else { "False" }.to_string()
    } else if value.is_none() {
        // None is not displayed in REPL (handled by caller).
        "None".to_string()
    } else if value.is_string() {
        // Strings in REPL are shown with quotes (repr).
        format!("'{}'", value)
    } else {
        format!("{}", value)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
