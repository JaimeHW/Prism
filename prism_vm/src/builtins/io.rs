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
    } else if let Ok(text) = super::types::builtin_str(&[value]) {
        value_to_string(text).unwrap_or_else(|| "<object>".to_string())
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

/// Builtin open function.
pub fn builtin_open(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_open_with_keywords(args, &[])
}

/// VM-aware builtin open function with keyword argument support.
pub fn builtin_open_vm_kw(
    _vm: &mut crate::VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    builtin_open_with_keywords(args, keywords)
}

fn builtin_open_with_keywords(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() > 8 {
        return Err(BuiltinError::TypeError(format!(
            "open() takes at most 8 arguments ({} given)",
            args.len()
        )));
    }

    let mut file = args.first().copied();
    let mut mode = args.get(1).copied();
    let mut buffering = args.get(2).copied();
    let mut encoding = args.get(3).copied();
    let mut errors = args.get(4).copied();
    let mut newline = args.get(5).copied();
    let mut closefd = args.get(6).copied();
    let mut opener = args.get(7).copied();

    for &(name, value) in keywords {
        match name {
            "file" => assign_open_keyword(&mut file, value, 0, args.len(), "file")?,
            "mode" => assign_open_keyword(&mut mode, value, 1, args.len(), "mode")?,
            "buffering" => assign_open_keyword(&mut buffering, value, 2, args.len(), "buffering")?,
            "encoding" => assign_open_keyword(&mut encoding, value, 3, args.len(), "encoding")?,
            "errors" => assign_open_keyword(&mut errors, value, 4, args.len(), "errors")?,
            "newline" => assign_open_keyword(&mut newline, value, 5, args.len(), "newline")?,
            "closefd" => assign_open_keyword(&mut closefd, value, 6, args.len(), "closefd")?,
            "opener" => assign_open_keyword(&mut opener, value, 7, args.len(), "opener")?,
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "open() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let file = file.ok_or_else(|| {
        BuiltinError::TypeError("open() missing required argument 'file' (pos 1)".to_string())
    })?;
    let file_path = value_to_string(file).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "open() argument 'file' must be str, not {}",
            file.type_name()
        ))
    })?;

    let mode = match mode {
        Some(value) => value_to_string(value).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "open() argument 'mode' must be str, not {}",
                value.type_name()
            ))
        })?,
        None => "r".to_string(),
    };

    if let Some(value) = buffering {
        let _ = extract_int_like(value, "buffering")?;
    }

    let encoding = match encoding {
        Some(value) if value.is_none() => None,
        Some(value) => Some(value_to_string(value).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "open() argument 'encoding' must be str or None, not {}",
                value.type_name()
            ))
        })?),
        None => None,
    };

    if let Some(value) = errors {
        if !value.is_none() {
            let _ = value_to_string(value).ok_or_else(|| {
                BuiltinError::TypeError(format!(
                    "open() argument 'errors' must be str or None, not {}",
                    value.type_name()
                ))
            })?;
        }
    }

    if let Some(value) = newline {
        if !value.is_none() {
            let _ = value_to_string(value).ok_or_else(|| {
                BuiltinError::TypeError(format!(
                    "open() argument 'newline' must be str or None, not {}",
                    value.type_name()
                ))
            })?;
        }
    }

    if let Some(value) = closefd {
        if !extract_bool_like(value, "closefd")? {
            return Err(BuiltinError::NotImplemented(
                "open() with closefd=False is not implemented yet".to_string(),
            ));
        }
    }

    if let Some(value) = opener {
        if !value.is_none() {
            return Err(BuiltinError::NotImplemented(
                "open() with opener is not implemented yet".to_string(),
            ));
        }
    }

    crate::stdlib::io::open_file_stream_object(&file_path, &mode, encoding.as_deref())
}

fn assign_open_keyword(
    slot: &mut Option<Value>,
    value: Value,
    positional_index: usize,
    positional_len: usize,
    name: &str,
) -> Result<(), BuiltinError> {
    if positional_len > positional_index {
        return Err(BuiltinError::TypeError(format!(
            "open() got multiple values for argument '{}'",
            name
        )));
    }
    if slot.replace(value).is_some() {
        return Err(BuiltinError::TypeError(format!(
            "open() got multiple values for argument '{}'",
            name
        )));
    }
    Ok(())
}

fn extract_int_like(value: Value, name: &str) -> Result<i64, BuiltinError> {
    if let Some(number) = value.as_int() {
        return Ok(number);
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(i64::from(boolean));
    }
    Err(BuiltinError::TypeError(format!(
        "open() argument '{}' must be an integer",
        name
    )))
}

fn extract_bool_like(value: Value, name: &str) -> Result<bool, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(boolean);
    }
    if let Some(number) = value.as_int() {
        return Ok(number != 0);
    }
    Err(BuiltinError::TypeError(format!(
        "open() argument '{}' must be bool-like",
        name
    )))
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
    use prism_runtime::types::string::StringObject;
    use std::io::Cursor;

    fn value_to_rust_string(value: Value) -> String {
        if let Some(ptr) = value.as_string_object_ptr() {
            let interned =
                interned_by_ptr(ptr as *const u8).expect("interned pointer should resolve");
            return interned.as_str().to_string();
        }

        let ptr = value
            .as_object_ptr()
            .expect("print()/input() tests expect string values");
        let string = unsafe { &*(ptr as *const StringObject) };
        string.as_str().to_string()
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
    fn test_print_uses_exception_display_text() {
        let exc = crate::builtins::get_exception_type("ValueError")
            .expect("ValueError should exist")
            .construct(&[Value::string(intern("boom"))]);
        let mut out = Vec::new();

        write_print(&[exc], &mut out);

        assert_eq!(String::from_utf8(out).unwrap(), "boom\n");
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
