//! Native `_string` module bootstrap surface.
//!
//! CPython's `string.py` depends on `_string.formatter_parser()` and
//! `_string.formatter_field_name_split()`. Prism only needs a compact, native
//! implementation of that helper surface for the current regression targets.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, builtin_ascii, builtin_format, builtin_repr, builtin_str,
};
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::get_attribute_value;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

static FORMATTER_PARSER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_string.formatter_parser"),
        formatter_parser_builtin,
    )
});
static FORMATTER_FIELD_NAME_SPLIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_string.formatter_field_name_split"),
        formatter_field_name_split_builtin,
    )
});

/// Native `_string` module descriptor.
#[derive(Debug, Clone)]
pub struct StringModule {
    attrs: Vec<Arc<str>>,
}

impl StringModule {
    /// Create a new `_string` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("formatter_parser"),
                Arc::from("formatter_field_name_split"),
            ],
        }
    }
}

impl Default for StringModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for StringModule {
    fn name(&self) -> &str {
        "_string"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "formatter_parser" => Ok(builtin_value(&FORMATTER_PARSER_FUNCTION)),
            "formatter_field_name_split" => Ok(builtin_value(&FORMATTER_FIELD_NAME_SPLIT_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_string' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FormatPart {
    literal: String,
    field_name: Option<String>,
    format_spec: Option<String>,
    conversion: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FieldStep {
    Attribute(String),
    Item(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldNumberingMode {
    Automatic,
    Manual,
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn value_to_string(value: Value, context: &str) -> Result<String, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a str")))?;
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a str")))?;
        return Ok(interned.as_str().to_string());
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a str")))?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError(format!("{context} must be a str")));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(string.as_str().to_string())
}

fn tuple_value(parts: Vec<Value>) -> Value {
    leak_object_value(TupleObject::from_vec(parts))
}

fn list_value(values: Vec<Value>) -> Value {
    leak_object_value(ListObject::from_iter(values))
}

fn parse_format_string(input: &str) -> Result<Vec<FormatPart>, BuiltinError> {
    let mut result = Vec::new();
    let chars = input.chars().collect::<Vec<_>>();
    let mut literal = String::new();
    let mut index = 0;

    while index < chars.len() {
        match chars[index] {
            '{' => {
                if index + 1 < chars.len() && chars[index + 1] == '{' {
                    literal.push('{');
                    index += 2;
                    continue;
                }

                let start = index + 1;
                let mut depth = 1;
                let mut cursor = start;
                while cursor < chars.len() {
                    match chars[cursor] {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    cursor += 1;
                }

                if cursor >= chars.len() {
                    return Err(BuiltinError::ValueError(
                        "Single '{' encountered in format string".to_string(),
                    ));
                }

                let field = chars[start..cursor].iter().collect::<String>();
                let (field_name, format_spec, conversion) = split_field_expression(&field)?;
                result.push(FormatPart {
                    literal: std::mem::take(&mut literal),
                    field_name: Some(field_name),
                    format_spec,
                    conversion,
                });
                index = cursor + 1;
            }
            '}' => {
                if index + 1 < chars.len() && chars[index + 1] == '}' {
                    literal.push('}');
                    index += 2;
                    continue;
                }
                return Err(BuiltinError::ValueError(
                    "Single '}' encountered in format string".to_string(),
                ));
            }
            ch => {
                literal.push(ch);
                index += 1;
            }
        }
    }

    result.push(FormatPart {
        literal,
        field_name: None,
        format_spec: None,
        conversion: None,
    });
    Ok(result)
}

fn split_field_expression(
    field: &str,
) -> Result<(String, Option<String>, Option<String>), BuiltinError> {
    let mut conversion_index = None;
    let mut format_index = None;
    let mut bracket_depth = 0_i32;

    for (index, ch) in field.char_indices() {
        match ch {
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            '!' if bracket_depth == 0 && conversion_index.is_none() => {
                conversion_index = Some(index)
            }
            ':' if bracket_depth == 0 && format_index.is_none() => {
                format_index = Some(index);
                break;
            }
            _ => {}
        }
    }

    let field_end = conversion_index.or(format_index).unwrap_or(field.len());
    let field_name = field[..field_end].to_string();
    let conversion = conversion_index.map(|start| {
        let end = format_index.unwrap_or(field.len());
        field[start + 1..end].to_string()
    });
    let format_spec = format_index.map(|start| field[start + 1..].to_string());
    Ok((field_name, format_spec, conversion))
}

fn parse_field_path(input: &str) -> (String, Vec<FieldStep>) {
    let mut first = String::new();
    let mut index = 0;
    let chars = input.chars().collect::<Vec<_>>();

    while index < chars.len() {
        match chars[index] {
            '.' | '[' => break,
            ch => first.push(ch),
        }
        index += 1;
    }

    let mut rest = Vec::new();
    while index < chars.len() {
        match chars[index] {
            '.' => {
                index += 1;
                let start = index;
                while index < chars.len() && chars[index] != '.' && chars[index] != '[' {
                    index += 1;
                }
                let name = chars[start..index].iter().collect::<String>();
                rest.push(FieldStep::Attribute(name));
            }
            '[' => {
                index += 1;
                let start = index;
                while index < chars.len() && chars[index] != ']' {
                    index += 1;
                }
                let item = chars[start..index].iter().collect::<String>();
                rest.push(FieldStep::Item(item));
                if index < chars.len() && chars[index] == ']' {
                    index += 1;
                }
            }
            _ => index += 1,
        }
    }

    (first, rest)
}

fn split_field_name(input: &str) -> (Value, Vec<Value>) {
    let (first, steps) = parse_field_path(input);
    let first_value = if !first.is_empty() && first.chars().all(|ch| ch.is_ascii_digit()) {
        Value::int(first.parse::<i64>().unwrap_or_default()).unwrap()
    } else {
        Value::string(intern(&first))
    };

    let rest = steps
        .into_iter()
        .map(|step| match step {
            FieldStep::Attribute(name) => {
                tuple_value(vec![Value::bool(true), Value::string(intern(&name))])
            }
            FieldStep::Item(item) => {
                let item_value = if item.chars().all(|ch| ch.is_ascii_digit()) && !item.is_empty() {
                    Value::int(item.parse::<i64>().unwrap_or_default()).unwrap()
                } else {
                    Value::string(intern(&item))
                };
                tuple_value(vec![Value::bool(false), item_value])
            }
        })
        .collect();

    (first_value, rest)
}

fn formatter_parser_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "formatter_parser() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let input = value_to_string(args[0], "format_string")?;
    let parts = parse_format_string(&input)?;
    let mut values = Vec::with_capacity(parts.len());
    for part in parts {
        values.push(tuple_value(vec![
            Value::string(intern(&part.literal)),
            part.field_name
                .map(|value| Value::string(intern(&value)))
                .unwrap_or_else(Value::none),
            part.format_spec
                .map(|value| Value::string(intern(&value)))
                .unwrap_or_else(Value::none),
            part.conversion
                .map(|value| Value::string(intern(&value)))
                .unwrap_or_else(Value::none),
        ]));
    }
    Ok(list_value(values))
}

fn formatter_field_name_split_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "formatter_field_name_split() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let input = value_to_string(args[0], "field_name")?;
    let (first, rest) = split_field_name(&input);
    Ok(tuple_value(vec![first, list_value(rest)]))
}

#[inline]
fn runtime_error_to_builtin_error(err: RuntimeError) -> BuiltinError {
    let display = err.to_string();
    match err.kind {
        RuntimeErrorKind::TypeError { message } => BuiltinError::TypeError(message.to_string()),
        RuntimeErrorKind::UnsupportedOperandTypes { op, left, right } => BuiltinError::TypeError(
            format!("unsupported operand type(s) for {op}: '{left}' and '{right}'"),
        ),
        RuntimeErrorKind::NotCallable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not callable", type_name))
        }
        RuntimeErrorKind::NotIterable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not iterable", type_name))
        }
        RuntimeErrorKind::NotSubscriptable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not subscriptable", type_name))
        }
        RuntimeErrorKind::AttributeError { type_name, attr } => BuiltinError::AttributeError(
            format!("'{}' object has no attribute '{}'", type_name, attr),
        ),
        RuntimeErrorKind::IndexError { index, length } => BuiltinError::IndexError(format!(
            "index {} out of range for sequence of length {}",
            index, length
        )),
        RuntimeErrorKind::KeyError { key } => BuiltinError::KeyError(key.to_string()),
        RuntimeErrorKind::ValueError { message } => BuiltinError::ValueError(message.to_string()),
        RuntimeErrorKind::OverflowError { message } => {
            BuiltinError::OverflowError(message.to_string())
        }
        _ => BuiltinError::TypeError(display),
    }
}

#[inline]
fn owned_string_value(text: String) -> Value {
    leak_object_value(StringObject::from_string(text))
}

#[inline]
fn lookup_keyword(keywords: &[(&str, Value)], name: &str) -> Option<Value> {
    keywords
        .iter()
        .find_map(|(keyword, value)| (*keyword == name).then_some(*value))
}

fn resolve_field_root(
    root: &str,
    positional: &[Value],
    keywords: &[(&str, Value)],
    auto_index: &mut usize,
    numbering_mode: &mut Option<FieldNumberingMode>,
) -> Result<Value, BuiltinError> {
    if root.is_empty() {
        if matches!(numbering_mode, Some(FieldNumberingMode::Manual)) {
            return Err(BuiltinError::ValueError(
                "cannot switch from manual field specification to automatic field numbering"
                    .to_string(),
            ));
        }
        *numbering_mode = Some(FieldNumberingMode::Automatic);
        let index = *auto_index;
        *auto_index += 1;
        return positional.get(index).copied().ok_or_else(|| {
            BuiltinError::IndexError(format!(
                "Replacement index {index} out of range for positional args tuple"
            ))
        });
    }

    if matches!(numbering_mode, Some(FieldNumberingMode::Automatic)) {
        return Err(BuiltinError::ValueError(
            "cannot switch from automatic field numbering to manual field specification"
                .to_string(),
        ));
    }
    *numbering_mode = Some(FieldNumberingMode::Manual);

    if root.chars().all(|ch| ch.is_ascii_digit()) {
        let index = root
            .parse::<usize>()
            .map_err(|_| BuiltinError::ValueError(format!("invalid field index {root}")))?;
        return positional.get(index).copied().ok_or_else(|| {
            BuiltinError::IndexError(format!(
                "Replacement index {index} out of range for positional args tuple"
            ))
        });
    }

    lookup_keyword(keywords, root).ok_or_else(|| BuiltinError::KeyError(root.to_string()))
}

fn resolve_item_step(
    vm: &mut VirtualMachine,
    value: Value,
    item: &str,
) -> Result<Value, BuiltinError> {
    let key = if !item.is_empty() && item.chars().all(|ch| ch.is_ascii_digit()) {
        Value::int(item.parse::<i64>().unwrap_or_default()).unwrap()
    } else {
        Value::string(intern(item))
    };
    let getitem = get_attribute_value(vm, value, &intern("__getitem__"))
        .map_err(runtime_error_to_builtin_error)?;
    invoke_callable_value(vm, getitem, &[key]).map_err(runtime_error_to_builtin_error)
}

fn resolve_format_field(
    vm: &mut VirtualMachine,
    field_name: &str,
    positional: &[Value],
    keywords: &[(&str, Value)],
    auto_index: &mut usize,
    numbering_mode: &mut Option<FieldNumberingMode>,
) -> Result<Value, BuiltinError> {
    let (root, steps) = parse_field_path(field_name);
    let mut current = resolve_field_root(&root, positional, keywords, auto_index, numbering_mode)?;

    for step in steps {
        current = match step {
            FieldStep::Attribute(name) => get_attribute_value(vm, current, &intern(&name))
                .map_err(runtime_error_to_builtin_error)?,
            FieldStep::Item(item) => resolve_item_step(vm, current, &item)?,
        };
    }

    Ok(current)
}

fn apply_conversion(value: Value, conversion: Option<&str>) -> Result<Value, BuiltinError> {
    match conversion {
        None | Some("") => Ok(value),
        Some("r") => builtin_repr(&[value]),
        Some("s") => builtin_str(&[value]),
        Some("a") => builtin_ascii(&[value]),
        Some(other) => Err(BuiltinError::ValueError(format!(
            "Unknown conversion specifier {other}"
        ))),
    }
}

fn render_format_spec(
    vm: &mut VirtualMachine,
    format_spec: Option<&str>,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<String, BuiltinError> {
    let Some(format_spec) = format_spec else {
        return Ok(String::new());
    };
    if !format_spec.contains('{') && !format_spec.contains('}') {
        return Ok(format_spec.to_string());
    }

    format_template(vm, format_spec, positional, keywords)
}

fn format_field_value(
    vm: &mut VirtualMachine,
    value: Value,
    conversion: Option<&str>,
    format_spec: Option<&str>,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<String, BuiltinError> {
    let converted = apply_conversion(value, conversion)?;
    let resolved_spec = render_format_spec(vm, format_spec, positional, keywords)?;
    let formatted = if resolved_spec.is_empty() {
        builtin_format(&[converted])?
    } else {
        builtin_format(&[converted, owned_string_value(resolved_spec)])?
    };
    value_to_string(formatted, "formatted value")
}

fn format_template(
    vm: &mut VirtualMachine,
    template: &str,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<String, BuiltinError> {
    let parts = parse_format_string(template)?;
    let mut rendered = String::new();
    let mut auto_index = 0_usize;
    let mut numbering_mode = None;

    for part in parts {
        rendered.push_str(&part.literal);
        let Some(field_name) = part.field_name.as_deref() else {
            continue;
        };
        let value = resolve_format_field(
            vm,
            field_name,
            positional,
            keywords,
            &mut auto_index,
            &mut numbering_mode,
        )?;
        let fragment = format_field_value(
            vm,
            value,
            part.conversion.as_deref(),
            part.format_spec.as_deref(),
            positional,
            keywords,
        )?;
        rendered.push_str(&fragment);
    }

    Ok(rendered)
}

pub(crate) fn builtin_str_format_method(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "unbound str.format() call requires a string receiver".to_string(),
        ));
    }

    let template = value_to_string(args[0], "str.format() receiver")?;
    let rendered = format_template(vm, &template, &args[1..], keywords)?;
    Ok(owned_string_value(rendered))
}

#[cfg(test)]
mod tests {
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
}
