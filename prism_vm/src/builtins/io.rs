//! I/O builtins (print, input, open).

use super::BuiltinError;
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::get_attribute_value;
use crate::truthiness::try_is_truthy;
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

/// VM-aware builtin print function with keyword argument support.
pub fn builtin_print_vm_kw(
    vm: &mut crate::VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let options = parse_print_options(vm, keywords)?;
    let rendered = format_print_output(args, &options.sep, &options.end);

    if let Some(file) = options.file {
        write_print_to_python_file(vm, file, &rendered)?;
        if options.flush {
            flush_python_file(vm, file)?;
        }
        return Ok(Value::none());
    }

    let mut stdout = io::stdout().lock();
    stdout.write_all(rendered.as_bytes()).ok();
    if options.flush {
        stdout.flush().ok();
    }
    Ok(Value::none())
}

struct PrintOptions {
    sep: String,
    end: String,
    file: Option<Value>,
    flush: bool,
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
    out.write_all(format_print_output(args, " ", "\n").as_bytes())
        .ok();
}

fn format_print_output(args: &[Value], sep: &str, end: &str) -> String {
    let mut out = String::new();
    for (index, arg) in args.iter().enumerate() {
        if index > 0 {
            out.push_str(sep);
        }
        out.push_str(&format_value(*arg));
    }
    out.push_str(end);
    out
}

fn parse_print_options(
    vm: &mut crate::VirtualMachine,
    keywords: &[(&str, Value)],
) -> Result<PrintOptions, BuiltinError> {
    let mut options = PrintOptions {
        sep: " ".to_string(),
        end: "\n".to_string(),
        file: None,
        flush: false,
    };

    for &(name, value) in keywords {
        match name {
            "sep" => options.sep = parse_print_text_keyword(value, "sep", " ")?,
            "end" => options.end = parse_print_text_keyword(value, "end", "\n")?,
            "file" => {
                options.file = if value.is_none() { None } else { Some(value) };
            }
            "flush" => options.flush = try_is_truthy(vm, value).map_err(BuiltinError::Raised)?,
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "print() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    Ok(options)
}

fn parse_print_text_keyword(
    value: Value,
    name: &'static str,
    default: &'static str,
) -> Result<String, BuiltinError> {
    if value.is_none() {
        return Ok(default.to_string());
    }

    value_to_string(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{name} must be None or a string, not {}",
            value.type_name()
        ))
    })
}

fn write_print_to_python_file(
    vm: &mut crate::VirtualMachine,
    file: Value,
    rendered: &str,
) -> Result<(), BuiltinError> {
    let write = get_attribute_value(vm, file, &intern("write")).map_err(BuiltinError::Raised)?;
    let rendered_value = vm
        .allocator()
        .alloc_value(StringObject::from_string(rendered.to_string()))
        .ok_or_else(|| {
            BuiltinError::TypeError("out of memory allocating print output".to_string())
        })?;
    invoke_callable_value(vm, write, &[rendered_value]).map_err(BuiltinError::Raised)?;
    Ok(())
}

fn flush_python_file(vm: &mut crate::VirtualMachine, file: Value) -> Result<(), BuiltinError> {
    let flush = get_attribute_value(vm, file, &intern("flush")).map_err(BuiltinError::Raised)?;
    invoke_callable_value(vm, flush, &[]).map_err(BuiltinError::Raised)?;
    Ok(())
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
    use crate::ops::calls::invoke_callable_value;
    use crate::ops::objects::get_attribute_value;
    use prism_core::intern::{intern, interned_by_ptr};
    use prism_runtime::types::string::StringObject;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

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
    fn test_format_print_output_supports_custom_sep_and_end() {
        let rendered = format_print_output(
            &[Value::string(intern("alpha")), Value::int(7).unwrap()],
            " :: ",
            "!",
        );
        assert_eq!(rendered, "alpha :: 7!");
    }

    #[test]
    fn test_print_vm_kw_rejects_non_string_sep() {
        let mut vm = crate::VirtualMachine::new();
        let err = builtin_print_vm_kw(
            &mut vm,
            &[Value::string(intern("alpha"))],
            &[("sep", Value::int(1).unwrap())],
        )
        .expect_err("non-string sep should fail");
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("sep must be None or a string"));
    }

    #[test]
    fn test_print_vm_kw_writes_to_file_argument() {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);

        let mut vm = crate::VirtualMachine::new();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "prism_print_builtin_{}_{}_{}.txt",
            std::process::id(),
            nanos,
            unique
        ));
        let path_string = path.to_string_lossy().to_string();

        let file = crate::stdlib::io::open_file_stream_object(&path_string, "w", None)
            .expect("file stream should open");
        builtin_print_vm_kw(
            &mut vm,
            &[Value::string(intern("alpha")), Value::int(7).unwrap()],
            &[
                ("sep", Value::string(intern("-"))),
                ("end", Value::string(intern("!"))),
                ("file", file),
                ("flush", Value::bool(true)),
            ],
        )
        .expect("print should write to file");

        let close =
            get_attribute_value(&mut vm, file, &intern("close")).expect("file.close should exist");
        invoke_callable_value(&mut vm, close, &[]).expect("file.close should succeed");

        let contents = std::fs::read_to_string(&path).expect("printed file should be readable");
        assert_eq!(contents, "alpha-7!");
        let _ = std::fs::remove_file(&path);
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
