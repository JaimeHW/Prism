//! I/O builtins (print, input, open).

use super::BuiltinError;
use prism_core::Value;
use std::io::{self, Write};

/// Builtin print function.
pub fn builtin_print(args: &[Value]) -> Result<Value, BuiltinError> {
    let mut stdout = io::stdout().lock();
    let mut first = true;

    for arg in args {
        if !first {
            write!(stdout, " ").ok();
        }
        first = false;
        write!(stdout, "{}", format_value(*arg)).ok();
    }

    writeln!(stdout).ok();
    stdout.flush().ok();
    Ok(Value::none())
}

#[inline]
fn format_value(value: Value) -> String {
    if value.is_none() {
        "None".to_string()
    } else if let Some(b) = value.as_bool() {
        if b { "True" } else { "False" }.to_string()
    } else if let Some(i) = value.as_int() {
        i.to_string()
    } else if let Some(f) = value.as_float() {
        if f == f.trunc() && f.abs() < 1e16 {
            format!("{}.0", f as i64)
        } else {
            format!("{}", f)
        }
    } else {
        "<object>".to_string()
    }
}

/// Builtin input function.
pub fn builtin_input(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "input expected at most 1 argument, got {}",
            args.len()
        )));
    }
    if !args.is_empty() {
        print!("{}", format_value(args[0]));
        io::stdout().flush().ok();
    }
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok();
    Ok(Value::none()) // TODO: Return string Value
}

/// Builtin open function placeholder.
pub fn builtin_open(args: &[Value]) -> Result<Value, BuiltinError> {
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "open() not yet implemented".to_string(),
    ))
}
