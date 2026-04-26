use super::*;

#[test]
fn test_string_module_exposes_formatter_helpers() {
    let module = StringModule::new();
    assert!(module.get_attr("formatter_parser").is_ok());
    assert!(module.get_attr("formatter_field_name_split").is_ok());
}

#[test]
fn test_formatter_parser_handles_literals_and_fields() {
    let value = formatter_parser_builtin(&[Value::string(intern("x={value!r:>4}"))])
        .expect("formatter_parser should succeed");
    let ptr = value.as_object_ptr().expect("result should be a list");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 2);
}

#[test]
fn test_formatter_parser_handles_escaped_braces() {
    let value = formatter_parser_builtin(&[Value::string(intern("{{name}}"))])
        .expect("formatter_parser should succeed");
    let ptr = value.as_object_ptr().expect("result should be a list");
    let list = unsafe { &*(ptr as *const ListObject) };
    let first_ptr = list
        .get(0)
        .expect("entry should exist")
        .as_object_ptr()
        .expect("entry should be tuple");
    let tuple = unsafe { &*(first_ptr as *const TupleObject) };
    assert_eq!(
        interned_by_ptr(tuple.get(0).unwrap().as_string_object_ptr().unwrap() as *const u8)
            .unwrap()
            .as_str(),
        "{name}"
    );
}

#[test]
fn test_formatter_field_name_split_parses_attrs_and_indexes() {
    let value = formatter_field_name_split_builtin(&[Value::string(intern("user.name[0]"))])
        .expect("split should succeed");
    let ptr = value.as_object_ptr().expect("result should be tuple");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(
        interned_by_ptr(tuple.get(0).unwrap().as_string_object_ptr().unwrap() as *const u8)
            .unwrap()
            .as_str(),
        "user"
    );
}

#[test]
fn test_builtin_str_format_method_supports_positional_fields() {
    let mut vm = VirtualMachine::new();
    let value = builtin_str_format_method(
        &mut vm,
        &[
            Value::string(intern("{}_{}_tmp")),
            Value::string(intern("@test")),
            Value::int(42).unwrap(),
        ],
        &[],
    )
    .expect("str.format should succeed");
    assert_eq!(
        value_to_string(value, "formatted value").expect("result should be a string"),
        "@test_42_tmp"
    );
}

#[test]
fn test_builtin_str_format_method_supports_keyword_fields() {
    let mut vm = VirtualMachine::new();
    let value = builtin_str_format_method(
        &mut vm,
        &[Value::string(intern("/proc/{pid}/statm"))],
        &[("pid", Value::int(123).unwrap())],
    )
    .expect("str.format should succeed");
    assert_eq!(
        value_to_string(value, "formatted value").expect("result should be a string"),
        "/proc/123/statm"
    );
}

#[test]
fn test_builtin_str_format_method_supports_numeric_specs() {
    let mut vm = VirtualMachine::new();
    let value = builtin_str_format_method(
        &mut vm,
        &[
            Value::string(intern("{0} (0x{0:08X})")),
            Value::int(255).unwrap(),
        ],
        &[],
    )
    .expect("str.format should succeed");
    assert_eq!(
        value_to_string(value, "formatted value").expect("result should be a string"),
        "255 (0x000000FF)"
    );
}
