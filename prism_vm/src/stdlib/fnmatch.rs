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
use prism_runtime::types::bytes::BytesObject;
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
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum MatchText {
    Str(String),
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TranslateToken {
    Star,
    Text(String),
}

fn value_to_match_text(value: Value, context: &str) -> Result<MatchText, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a str")))?;
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a str")))?;
        return Ok(MatchText::Str(interned.as_str().to_string()));
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be str or bytes")))?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::STR => {
            let string = unsafe { &*(ptr as *const StringObject) };
            Ok(MatchText::Str(string.as_str().to_string()))
        }
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            Ok(MatchText::Bytes(bytes.as_bytes().to_vec()))
        }
        _ => Err(BuiltinError::TypeError(format!(
            "{context} must be str or bytes"
        ))),
    }
}

fn match_text_to_string(value: &MatchText) -> String {
    match value {
        MatchText::Str(text) => text.clone(),
        MatchText::Bytes(bytes) => bytes.iter().map(|byte| char::from(*byte)).collect(),
    }
}

fn normalize_match_text(value: &MatchText) -> String {
    match value {
        MatchText::Str(text) => {
            if cfg!(windows) {
                text.replace('/', "\\").to_ascii_lowercase()
            } else {
                text.clone()
            }
        }
        MatchText::Bytes(bytes) => {
            let normalized = if cfg!(windows) {
                bytes
                    .iter()
                    .map(|byte| match *byte {
                        b'/' => b'\\',
                        value => value.to_ascii_lowercase(),
                    })
                    .collect::<Vec<_>>()
            } else {
                bytes.clone()
            };
            normalized
                .iter()
                .map(|byte| char::from(*byte))
                .collect::<String>()
        }
    }
}

fn ensure_compatible_types(name: &MatchText, pattern: &MatchText) -> Result<(), BuiltinError> {
    match (name, pattern) {
        (MatchText::Str(_), MatchText::Str(_)) | (MatchText::Bytes(_), MatchText::Bytes(_)) => {
            Ok(())
        }
        _ => Err(BuiltinError::TypeError(
            "cannot mix bytes and nonbytes patterns".to_string(),
        )),
    }
}

fn translate_pattern_public(pattern: &str) -> String {
    translate_pattern(pattern, r"\Z", true)
}

fn translate_pattern_regex(pattern: &str) -> String {
    translate_pattern(pattern, r"\z", false)
}

fn translate_pattern(pattern: &str, anchor: &str, atomic_stars: bool) -> String {
    let chars = pattern.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < chars.len() {
        match chars[index] {
            '*' => {
                while index + 1 < chars.len() && chars[index + 1] == '*' {
                    index += 1;
                }
                tokens.push(TranslateToken::Star);
            }
            '?' => tokens.push(TranslateToken::Text(".".to_string())),
            '[' => {
                let (mut translated, next_index) = translate_character_class(&chars, index);
                if !atomic_stars && translated == "(?!)" {
                    translated = r"\b\B".to_string();
                }
                tokens.push(TranslateToken::Text(translated));
                index = next_index;
                continue;
            }
            other => tokens.push(TranslateToken::Text(regex::escape(&other.to_string()))),
        }
        index += 1;
    }

    let result = if atomic_stars {
        assemble_public_pattern(&tokens)
    } else {
        assemble_regex_pattern(&tokens)
    };
    format!("(?s:{result}){anchor}")
}

fn matches_pattern(name: &str, pattern: &str) -> Result<bool, BuiltinError> {
    let regex_text = translate_pattern_regex(pattern);
    let mut cache = REGEX_CACHE
        .lock()
        .expect("fnmatch regex cache should not be poisoned");
    let regex = cache.entry(regex_text.clone()).or_insert_with(|| {
        Regex::new(&regex_text).expect("generated fnmatch regex should be valid")
    });
    Ok(regex.find(name).is_some_and(|matched| matched.start() == 0))
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

    let name = value_to_match_text(args[0], "name")?;
    let pattern = value_to_match_text(args[1], "pattern")?;
    ensure_compatible_types(&name, &pattern)?;
    Ok(Value::bool(matches_pattern(
        &normalize_match_text(&name),
        &normalize_match_text(&pattern),
    )?))
}

fn builtin_fnmatchcase(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "fnmatchcase() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let name = value_to_match_text(args[0], "name")?;
    let pattern = value_to_match_text(args[1], "pattern")?;
    ensure_compatible_types(&name, &pattern)?;
    Ok(Value::bool(matches_pattern(
        &match_text_to_string(&name),
        &match_text_to_string(&pattern),
    )?))
}

fn builtin_filter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "filter() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let names = collect_iterable_values(args[0])?;
    let pattern = value_to_match_text(args[1], "pattern")?;
    let normalized_pattern = normalize_match_text(&pattern);
    let mut result = ListObject::with_capacity(names.len());
    for value in names {
        let name = value_to_match_text(value, "name")?;
        ensure_compatible_types(&name, &pattern)?;
        if matches_pattern(&normalize_match_text(&name), &normalized_pattern)? {
            result.push(value);
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

    let pattern = match_text_to_string(&value_to_match_text(args[0], "pattern")?);
    Ok(Value::string(intern(&translate_pattern_public(&pattern))))
}

fn translate_character_class(chars: &[char], start: usize) -> (String, usize) {
    let mut cursor = start + 1;
    if cursor < chars.len() && chars[cursor] == '!' {
        cursor += 1;
    }
    if cursor < chars.len() && chars[cursor] == ']' {
        cursor += 1;
    }
    while cursor < chars.len() && chars[cursor] != ']' {
        cursor += 1;
    }
    if cursor >= chars.len() {
        return (r"\[".to_string(), start + 1);
    }
    (build_character_class(chars, start + 1, cursor), cursor + 1)
}

fn build_character_class(chars: &[char], start: usize, end: usize) -> String {
    let raw = chars[start..end].iter().collect::<String>();
    let mut stuff = if !raw.contains('-') {
        raw.replace('\\', r"\\")
    } else {
        let mut chunks = Vec::new();
        let mut chunk_start = start;
        let mut search = if chars.get(start) == Some(&'!') {
            start + 2
        } else {
            start + 1
        };

        while let Some(relative) = chars[search..end].iter().position(|&ch| ch == '-') {
            let hyphen = search + relative;
            chunks.push(chars[chunk_start..hyphen].iter().collect::<String>());
            chunk_start = hyphen + 1;
            search = hyphen + 3;
            if search >= end {
                break;
            }
        }

        let tail = chars[chunk_start..end].iter().collect::<String>();
        if !tail.is_empty() {
            chunks.push(tail);
        } else if let Some(last) = chunks.last_mut() {
            last.push('-');
        }

        for index in (1..chunks.len()).rev() {
            let previous = chunks[index - 1].chars().collect::<Vec<_>>();
            let current = chunks[index].chars().collect::<Vec<_>>();
            let (Some(previous_last), Some(_current_first)) =
                (previous.last().copied(), current.first().copied())
            else {
                continue;
            };
            if previous_last > current[0] {
                let mut merged = previous[..previous.len().saturating_sub(1)]
                    .iter()
                    .collect::<String>();
                merged.extend(current.iter().skip(1));
                chunks[index - 1] = merged;
                chunks.remove(index);
            }
        }

        chunks
            .into_iter()
            .map(|chunk| chunk.replace('\\', r"\\").replace('-', r"\-"))
            .collect::<Vec<_>>()
            .join("-")
    };

    let mut escaped = String::with_capacity(stuff.len());
    for ch in stuff.chars() {
        if matches!(ch, '&' | '~' | '|') {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    stuff = escaped;

    if stuff.is_empty() {
        "(?!)".to_string()
    } else if stuff == "!" {
        ".".to_string()
    } else {
        if let Some(rest) = stuff.strip_prefix('!') {
            stuff = format!("^{rest}");
        } else if stuff.starts_with('^') || stuff.starts_with('[') {
            stuff.insert(0, '\\');
        }
        format!("[{stuff}]")
    }
}

fn assemble_regex_pattern(tokens: &[TranslateToken]) -> String {
    let mut result = String::new();
    for token in tokens {
        match token {
            TranslateToken::Star => result.push_str(".*"),
            TranslateToken::Text(text) => result.push_str(text),
        }
    }
    result
}

fn assemble_public_pattern(tokens: &[TranslateToken]) -> String {
    let mut result = String::new();
    let mut index = 0usize;

    while index < tokens.len() && !matches!(tokens[index], TranslateToken::Star) {
        if let TranslateToken::Text(text) = &tokens[index] {
            result.push_str(text);
        }
        index += 1;
    }

    while index < tokens.len() {
        debug_assert!(matches!(tokens[index], TranslateToken::Star));
        index += 1;
        if index == tokens.len() {
            result.push_str(".*");
            break;
        }

        let mut fixed = String::new();
        while index < tokens.len() && !matches!(tokens[index], TranslateToken::Star) {
            if let TranslateToken::Text(text) = &tokens[index] {
                fixed.push_str(text);
            }
            index += 1;
        }

        if index == tokens.len() {
            result.push_str(".*");
            result.push_str(&fixed);
        } else {
            result.push_str("(?>.*?");
            result.push_str(&fixed);
            result.push(')');
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_value(value: &[u8]) -> Value {
        leak_object_value(BytesObject::from_slice(value))
    }

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
            r"(?s:file.\.py)\Z"
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

    #[test]
    fn test_fnmatch_requires_match_from_start_of_name() {
        let result = builtin_fnmatch(&[
            Value::string(intern("\nfoo")),
            Value::string(intern("foo*")),
        ])
        .expect("fnmatch should succeed");
        assert_eq!(result.as_bool(), Some(false));
    }

    #[test]
    fn test_fnmatchcase_handles_descending_ranges_without_panicking() {
        let matched = builtin_fnmatchcase(&[
            Value::string(intern("axb")),
            Value::string(intern("a[z-^]b")),
        ])
        .expect("fnmatchcase should succeed");
        assert_eq!(matched.as_bool(), Some(false));
    }

    #[test]
    fn test_translate_matches_cpython_public_formatting() {
        let translate = builtin_translate(&[Value::string(intern("**a*a****a"))])
            .expect("translate should work");
        assert_eq!(
            interned_by_ptr(translate.as_string_object_ptr().unwrap() as *const u8)
                .unwrap()
                .as_str(),
            r"(?s:(?>.*?a)(?>.*?a).*a)\Z"
        );
    }

    #[test]
    fn test_fnmatch_supports_bytes_and_rejects_mixed_types() {
        let matched = builtin_fnmatch(&[bytes_value(b"test\xff"), bytes_value(b"te*\xff")])
            .expect("bytes fnmatch should succeed");
        assert_eq!(matched.as_bool(), Some(true));

        let err = builtin_fnmatch(&[Value::string(intern("test")), bytes_value(b"*")])
            .expect_err("mixed string and bytes patterns should fail");
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_filter_preserves_bytes_entries() {
        let names = ListObject::from_slice(&[
            bytes_value(b"alpha.py"),
            bytes_value(b"beta.txt"),
            bytes_value(b"gamma.py"),
        ]);
        let value = builtin_filter(&[leak_object_value(names), bytes_value(b"*.py")])
            .expect("filter should succeed");
        let ptr = value.as_object_ptr().expect("filter should return a list");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert_eq!(list.len(), 2);

        let first_ptr = list.as_slice()[0]
            .as_object_ptr()
            .expect("bytes entry should be an object");
        let first = unsafe { &*(first_ptr as *const BytesObject) };
        assert_eq!(first.as_bytes(), b"alpha.py");
    }
}
