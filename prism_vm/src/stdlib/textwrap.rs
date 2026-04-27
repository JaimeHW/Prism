//! Native subset of CPython's `textwrap` module.
//!
//! The regression suite imports `textwrap.dedent` very early from many tests.
//! Keeping this tiny, allocation-conscious implementation native avoids pulling
//! in a large Python bootstrap chain before Prism's source stdlib is complete.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use std::sync::{Arc, LazyLock};

static DEDENT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("textwrap.dedent"), dedent));

/// Native `textwrap` module descriptor.
#[derive(Debug, Clone)]
pub struct TextwrapModule {
    attrs: Vec<Arc<str>>,
}

impl TextwrapModule {
    /// Create a new `textwrap` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("dedent")],
        }
    }
}

impl Default for TextwrapModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TextwrapModule {
    fn name(&self) -> &str {
        "textwrap"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "dedent" => Ok(builtin_value(&DEDENT_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'textwrap' has no attribute '{}'",
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

fn dedent(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "dedent() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let text = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("dedent() argument must be str".to_string()))?;
    Ok(string_value(dedent_text(text.as_str())))
}

fn dedent_text(text: &str) -> String {
    let mut margin: Option<&str> = None;

    for line in text.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        if content
            .bytes()
            .all(|byte| matches!(byte, b' ' | b'\t' | b'\r'))
        {
            continue;
        }

        let indent_len = content
            .bytes()
            .take_while(|byte| matches!(*byte, b' ' | b'\t'))
            .count();
        let indent = &content[..indent_len];
        margin = Some(match margin {
            None => indent,
            Some(current) if indent.starts_with(current) => current,
            Some(current) if current.starts_with(indent) => indent,
            Some(current) => common_whitespace_prefix(current, indent),
        });

        if margin == Some("") {
            break;
        }
    }

    let margin = margin.unwrap_or("");
    if margin.is_empty() {
        return normalize_blank_lines(text);
    }

    let mut output = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (content, newline) = line
            .strip_suffix('\n')
            .map(|content| (content, "\n"))
            .unwrap_or((line, ""));
        if content
            .bytes()
            .all(|byte| matches!(byte, b' ' | b'\t' | b'\r'))
        {
            output.push_str(newline);
        } else if let Some(rest) = content.strip_prefix(margin) {
            output.push_str(rest);
            output.push_str(newline);
        } else {
            output.push_str(content);
            output.push_str(newline);
        }
    }
    output
}

fn normalize_blank_lines(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (content, newline) = line
            .strip_suffix('\n')
            .map(|content| (content, "\n"))
            .unwrap_or((line, ""));
        if content
            .bytes()
            .all(|byte| matches!(byte, b' ' | b'\t' | b'\r'))
        {
            output.push_str(newline);
        } else {
            output.push_str(content);
            output.push_str(newline);
        }
    }
    output
}

fn common_whitespace_prefix<'a>(left: &'a str, right: &str) -> &'a str {
    let bytes = left.as_bytes();
    let right_bytes = right.as_bytes();
    let mut end = 0;
    while end < bytes.len() && end < right_bytes.len() && bytes[end] == right_bytes[end] {
        end += 1;
    }
    &left[..end]
}

#[inline]
fn string_value(text: String) -> Value {
    if text.len() <= 64 {
        Value::string(intern(&text))
    } else {
        crate::alloc_managed_value(StringObject::from_string(text))
    }
}
