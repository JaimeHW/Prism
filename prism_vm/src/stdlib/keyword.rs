//! Native `keyword` module.
//!
//! CPython exposes keyword metadata as a small pure-Python module. Prism keeps
//! the lookup path native so parser-facing tooling and compatibility tests get
//! immutable, branch-light checks without importing extra source.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use std::sync::{Arc, LazyLock};

const KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

const SOFT_KEYWORDS: &[&str] = &["_", "case", "match", "type"];

static ISKEYWORD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("keyword.iskeyword"), iskeyword));
static ISSOFTKEYWORD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("keyword.issoftkeyword"), issoftkeyword));

/// Native `keyword` module descriptor.
#[derive(Debug, Clone)]
pub struct KeywordModule {
    attrs: Vec<Arc<str>>,
    kwlist: Value,
    softkwlist: Value,
    all: Value,
}

impl KeywordModule {
    /// Create a new `keyword` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__all__"),
                Arc::from("iskeyword"),
                Arc::from("issoftkeyword"),
                Arc::from("kwlist"),
                Arc::from("softkwlist"),
            ],
            kwlist: string_list_value(KEYWORDS),
            softkwlist: string_list_value(SOFT_KEYWORDS),
            all: string_list_value(&["iskeyword", "issoftkeyword", "kwlist", "softkwlist"]),
        }
    }
}

impl Default for KeywordModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for KeywordModule {
    fn name(&self) -> &str {
        "keyword"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "iskeyword" => Ok(builtin_value(&ISKEYWORD_FUNCTION)),
            "issoftkeyword" => Ok(builtin_value(&ISSOFTKEYWORD_FUNCTION)),
            "kwlist" => Ok(self.kwlist),
            "softkwlist" => Ok(self.softkwlist),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'keyword' has no attribute '{}'",
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

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}

fn iskeyword(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_one_arg("iskeyword", args)?;
    Ok(Value::bool(matches_keyword(args[0], KEYWORDS)))
}

fn issoftkeyword(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_one_arg("issoftkeyword", args)?;
    Ok(Value::bool(matches_keyword(args[0], SOFT_KEYWORDS)))
}

#[inline]
fn exact_one_arg(function: &str, args: &[Value]) -> Result<(), BuiltinError> {
    if args.len() == 1 {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{function}() takes exactly one argument ({} given)",
            args.len()
        )))
    }
}

#[inline]
fn matches_keyword(value: Value, haystack: &[&str]) -> bool {
    value_as_string_ref(value)
        .is_some_and(|string| keyword_index(string.as_str(), haystack).is_some())
}

#[inline]
fn keyword_index(needle: &str, haystack: &[&str]) -> Option<usize> {
    haystack.binary_search(&needle).ok()
}
