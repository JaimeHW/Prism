//! Native `fnmatch` module.
//!
//! Prism's CPython regression harness imports `unittest.loader`, which in turn
//! imports `fnmatch`. The upstream `fnmatch.py` currently trips Prism's parser,
//! so this module provides a native implementation of the core shell-pattern
//! matching API with CPython-compatible semantics for `*`, `?`, and bracket
//! character classes.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, get_iterator_mut, value_to_iterator};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use regex::Regex;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock, Mutex};

static FILTER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("fnmatch.filter"), builtin_filter));
static FNMATCH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("fnmatch.fnmatch"), builtin_fnmatch));
static FNMATCHCASE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("fnmatch.fnmatchcase"), builtin_fnmatchcase)
});
static TRANSLATE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("fnmatch.translate"), builtin_translate));
static REGEX_CACHE: LazyLock<Mutex<FxHashMap<String, Regex>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

/// Native `fnmatch` module descriptor.
#[derive(Debug, Clone)]
pub struct FnmatchModule {
    attrs: Vec<Arc<str>>,
}

impl FnmatchModule {
    /// Create a new `fnmatch` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("filter"),
                Arc::from("fnmatch"),
                Arc::from("fnmatchcase"),
                Arc::from("translate"),
            ],
        }
    }
}

impl Default for FnmatchModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for FnmatchModule {
    fn name(&self) -> &str {
        "fnmatch"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "filter" => Ok(builtin_value(&FILTER_FUNCTION)),
            "fnmatch" => Ok(builtin_value(&FNMATCH_FUNCTION)),
            "fnmatchcase" => Ok(builtin_value(&FNMATCHCASE_FUNCTION)),
            "translate" => Ok(builtin_value(&TRANSLATE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'fnmatch' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn leak_object_value<T>(object: T) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(object)) as *const ())
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

fn normalize_name(value: &str) -> String {
    if cfg!(windows) {
        value.replace('/', "\\").to_ascii_lowercase()
    } else {
        value.to_string()
    }
}

fn translate_pattern(pattern: &str) -> String {
    let chars = pattern.chars().collect::<Vec<_>>();
    let mut result = String::new();
    let mut index = 0;

    while index < chars.len() {
        match chars[index] {
            '*' => {
                while index + 1 < chars.len() && chars[index + 1] == '*' {
                    index += 1;
                }
                result.push_str(".*");
            }
            '?' => result.push('.'),
            '[' => {
                let start = index + 1;
                let mut cursor = start;
                let mut negate = false;
                if cursor < chars.len() && chars[cursor] == '!' {
                    negate = true;
                    cursor += 1;
                }
                if cursor < chars.len() && chars[cursor] == ']' {
                    cursor += 1;
                }
                while cursor < chars.len() && chars[cursor] != ']' {
                    cursor += 1;
                }
                if cursor >= chars.len() {
                    result.push_str(r"\[");
                } else {
                    let mut class = String::new();
                    for ch in &chars[start + usize::from(negate)..cursor] {
                        match ch {
                            '\\' | '[' | ']' | '&' | '~' | '|' => {
                                class.push('\\');
                                class.push(*ch);
                            }
                            _ => class.push(*ch),
                        }
                    }
                    if negate {
                        result.push_str("[^");
                        result.push_str(&class);
                        result.push(']');
                    } else {
                        result.push('[');
                        result.push_str(&class);
                        result.push(']');
                    }
                    index = cursor;
                }
            }
            other => result.push_str(&regex::escape(&other.to_string())),
        }
        index += 1;
    }

    format!(r"(?s:{result})\z")
}

fn matches_pattern(name: &str, pattern: &str) -> Result<bool, BuiltinError> {
    let regex_text = translate_pattern(pattern);
    let mut cache = REGEX_CACHE
        .lock()
        .expect("fnmatch regex cache should not be poisoned");
    let regex = cache.entry(regex_text.clone()).or_insert_with(|| {
        Regex::new(&regex_text).expect("generated fnmatch regex should be valid")
    });
    Ok(regex.is_match(name))
}

fn collect_iterable_values(value: Value) -> Result<Vec<Value>, BuiltinError> {
    if let Some(iterator) = get_iterator_mut(&value) {
        return Ok(iterator.collect_remaining());
    }

    let mut iterator = value_to_iterator(&value).map_err(BuiltinError::from)?;
    Ok(iterator.collect_remaining())
}

fn builtin_fnmatch(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "fnmatch() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let name = normalize_name(&value_to_string(args[0], "name")?);
    let pattern = normalize_name(&value_to_string(args[1], "pattern")?);
    Ok(Value::bool(matches_pattern(&name, &pattern)?))
}

fn builtin_fnmatchcase(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "fnmatchcase() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let name = value_to_string(args[0], "name")?;
    let pattern = value_to_string(args[1], "pattern")?;
    Ok(Value::bool(matches_pattern(&name, &pattern)?))
}

fn builtin_filter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "filter() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let names = collect_iterable_values(args[0])?;
    let pattern = normalize_name(&value_to_string(args[1], "pattern")?);
    let mut result = ListObject::with_capacity(names.len());
    for value in names {
        let name = value_to_string(value, "name")?;
        if matches_pattern(&normalize_name(&name), &pattern)? {
            result.push(Value::string(intern(&name)));
        }
    }
    Ok(leak_object_value(result))
}

fn builtin_translate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "translate() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let pattern = value_to_string(args[0], "pattern")?;
    Ok(Value::string(intern(&translate_pattern(&pattern))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnmatch_module_exposes_core_api() {
        let module = FnmatchModule::new();
        assert!(module.get_attr("fnmatch").is_ok());
        assert!(module.get_attr("fnmatchcase").is_ok());
        assert!(module.get_attr("filter").is_ok());
        assert!(module.get_attr("translate").is_ok());
    }

    #[test]
    fn test_fnmatchcase_matches_shell_patterns() {
        let result = builtin_fnmatchcase(&[
            Value::string(intern("test_alpha.py")),
            Value::string(intern("test_*.py")),
        ])
        .expect("fnmatchcase should succeed");
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_fnmatch_applies_windows_normalization() {
        let result = builtin_fnmatch(&[
            Value::string(intern("A/Path.TXT")),
            Value::string(intern("a\\*.txt")),
        ])
        .expect("fnmatch should succeed");
        assert_eq!(result.as_bool(), Some(cfg!(windows)));
    }

    #[test]
    fn test_translate_returns_regex_wrapper() {
        let result =
            builtin_translate(&[Value::string(intern("file?.py"))]).expect("translate should work");
        assert_eq!(
            interned_by_ptr(result.as_string_object_ptr().unwrap() as *const u8)
                .unwrap()
                .as_str(),
            r"(?s:file.\.py)\z"
        );
    }

    #[test]
    fn test_filter_returns_matching_names() {
        let names = ListObject::from_slice(&[
            Value::string(intern("alpha.py")),
            Value::string(intern("beta.txt")),
            Value::string(intern("gamma.py")),
        ]);
        let value = builtin_filter(&[leak_object_value(names), Value::string(intern("*.py"))])
            .expect("filter should succeed");
        let ptr = value.as_object_ptr().expect("filter should return a list");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert_eq!(list.len(), 2);
    }
}
