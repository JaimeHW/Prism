//! I/O builtins (print, input, open).

use super::BuiltinError;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::string::StringObject;
use std::io::{self, BufRead, Write};

/// Builtin print function.
pub fn builtin_print(args: &[Value]) -> Result<Value, BuiltinError> {
    let mut stdout = io::stdout().lock();
    write_print(args, &mut stdout);
    stdout.flush().ok();
    Ok(Value::none())
}

#[inline]
fn format_value(value: Value) -> String {
    if let Some(s) = value_to_string(value) {
        return s;
    }

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
    } else if let Ok(repr) = super::functions::builtin_repr(&[value]) {
        value_to_string(repr).unwrap_or_else(|| "<object>".to_string())
    } else {
        "<object>".to_string()
    }
}

fn write_print<W: Write>(args: &[Value], out: &mut W) {
    let mut first = true;
    for arg in args {
        if !first {
            write!(out, " ").ok();
        }
        first = false;
        write!(out, "{}", format_value(*arg)).ok();
    }
    writeln!(out).ok();
}

/// Builtin input function.
pub fn builtin_input(args: &[Value]) -> Result<Value, BuiltinError> {
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    read_input(args, &mut stdin, &mut stdout)
}

fn read_input<R: BufRead, W: Write>(
    args: &[Value],
    input: &mut R,
    output: &mut W,
) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "input expected at most 1 argument, got {}",
            args.len()
        )));
    }
    if !args.is_empty() {
        write!(output, "{}", format_value(args[0])).ok();
        output.flush().ok();
    }

    let mut line = String::new();
    let bytes = input.read_line(&mut line).unwrap_or(0);
    if bytes == 0 {
        return Err(BuiltinError::ValueError(
            "EOF when reading a line".to_string(),
        ));
    }

    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }

    Ok(Value::string(intern(&line)))
}

/// Builtin open function placeholder.
pub fn builtin_open(args: &[Value]) -> Result<Value, BuiltinError> {
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "open() not yet implemented".to_string(),
    ))
}

#[inline]
fn value_to_string(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        let interned = interned_by_ptr(ptr as *const u8)?;
        return Some(interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    let string_obj = unsafe { &*(ptr as *const StringObject) };
    Some(string_obj.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::{intern, interned_by_ptr};
    use std::io::Cursor;

    fn value_to_rust_string(value: Value) -> String {
        let ptr = value
            .as_string_object_ptr()
            .expect("input()/print() tests expect tagged interned strings");
        let interned = interned_by_ptr(ptr as *const u8).expect("interned pointer should resolve");
        interned.as_str().to_string()
    }

    #[test]
    fn test_print_formats_tagged_strings_and_primitives() {
        let mut out = Vec::new();
        write_print(
            &[
                Value::string(intern("hello")),
                Value::int(3).unwrap(),
                Value::bool(true),
            ],
            &mut out,
        );
        assert_eq!(String::from_utf8(out).unwrap(), "hello 3 True\n");
    }

    #[test]
    fn test_input_strips_newline_and_returns_string() {
        let mut input = Cursor::new(b"alpha\n".to_vec());
        let mut output = Vec::new();

        let value = read_input(&[], &mut input, &mut output).unwrap();
        assert_eq!(value_to_rust_string(value), "alpha");
    }

    #[test]
    fn test_input_strips_crlf() {
        let mut input = Cursor::new(b"beta\r\n".to_vec());
        let mut output = Vec::new();

        let value = read_input(&[], &mut input, &mut output).unwrap();
        assert_eq!(value_to_rust_string(value), "beta");
    }

    #[test]
    fn test_input_emits_prompt() {
        let mut input = Cursor::new(b"value\n".to_vec());
        let mut output = Vec::new();

        let value = read_input(
            &[Value::string(intern("prompt> "))],
            &mut input,
            &mut output,
        )
        .unwrap();
        assert_eq!(value_to_rust_string(value), "value");
        assert_eq!(String::from_utf8(output).unwrap(), "prompt> ");
    }

    #[test]
    fn test_input_eof_error() {
        let mut input = Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();

        let err = read_input(&[], &mut input, &mut output).unwrap_err();
        assert!(matches!(err, BuiltinError::ValueError(_)));
        assert!(err.to_string().contains("EOF when reading a line"));
    }
}
