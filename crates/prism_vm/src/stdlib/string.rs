//! Native public `string` module constants.
//!
//! The high-use constants are exposed natively so tests and bootstrap code can
//! avoid importing CPython's source `string.py` for simple table lookups.

use super::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::sync::Arc;

const ASCII_LOWERCASE: &str = "abcdefghijklmnopqrstuvwxyz";
const ASCII_UPPERCASE: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ASCII_LETTERS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const DIGITS: &str = "0123456789";
const HEXDIGITS: &str = "0123456789abcdefABCDEF";
const OCTDIGITS: &str = "01234567";
const PUNCTUATION: &str = r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##;
const WHITESPACE: &str = " \t\n\r\x0b\x0c";
const PRINTABLE: &str = concat!(
    "0123456789",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##,
    " \t\n\r\x0b\x0c"
);

const EXPORTS: &[&str] = &[
    "ascii_lowercase",
    "ascii_uppercase",
    "ascii_letters",
    "digits",
    "hexdigits",
    "octdigits",
    "punctuation",
    "whitespace",
    "printable",
];

/// Native `string` module descriptor.
#[derive(Debug, Clone)]
pub struct StringPublicModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl StringPublicModule {
    /// Create a native public `string` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS
                .iter()
                .copied()
                .chain(["__all__"])
                .map(Arc::from)
                .collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for StringPublicModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for StringPublicModule {
    fn name(&self) -> &str {
        "string"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "ascii_lowercase" => Ok(string_value(ASCII_LOWERCASE)),
            "ascii_uppercase" => Ok(string_value(ASCII_UPPERCASE)),
            "ascii_letters" => Ok(string_value(ASCII_LETTERS)),
            "digits" => Ok(string_value(DIGITS)),
            "hexdigits" => Ok(string_value(HEXDIGITS)),
            "octdigits" => Ok(string_value(OCTDIGITS)),
            "punctuation" => Ok(string_value(PUNCTUATION)),
            "whitespace" => Ok(string_value(WHITESPACE)),
            "printable" => Ok(string_value(PRINTABLE)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'string' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[inline]
fn string_value(value: &str) -> Value {
    Value::string(intern(value))
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}
