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
        let interned = interned_by_ptr(ptr as *const u8).expect("interned pointer should resolve");
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
