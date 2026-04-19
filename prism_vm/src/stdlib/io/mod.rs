//! Python `io` module implementation.
//!
//! High-performance I/O primitives providing:
//! - `StringIO` - In-memory text stream
//! - `BytesIO` - In-memory binary stream
//! - `FileMode` - Mode string parser
//! - Constants - `SEEK_SET`, `SEEK_CUR`, `SEEK_END`, `DEFAULT_BUFFER_SIZE`

pub mod bytes_io;
pub mod open_fn;
pub mod string_io;

#[cfg(test)]
mod tests;

pub use bytes_io::BytesIO;
pub use open_fn::{DEFAULT_BUFFER_SIZE, FileMode, SEEK_CUR, SEEK_END, SEEK_SET};
pub use string_io::{IoError, StringIO};

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use std::sync::{Arc, LazyLock};

static TEXT_IO_WRAPPER_PLACEHOLDER: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("io.TextIOWrapper"),
        builtin_text_io_wrapper_placeholder,
    )
});

/// The `io` module implementation.
#[derive(Debug, Clone)]
pub struct IoModule {
    module_name: Arc<str>,
    /// Cached attribute names for fast lookup.
    attrs: Vec<Arc<str>>,
}

impl IoModule {
    /// Create a new io module instance.
    pub fn new() -> Self {
        Self::with_name("io")
    }

    /// Create an io-compatible module with a specific import name.
    pub fn with_name(module_name: &'static str) -> Self {
        let attrs = vec![
            Arc::from("StringIO"),
            Arc::from("BytesIO"),
            Arc::from("open"),
            Arc::from("FileIO"),
            Arc::from("BufferedReader"),
            Arc::from("BufferedWriter"),
            Arc::from("BufferedRandom"),
            Arc::from("TextIOWrapper"),
            Arc::from("DEFAULT_BUFFER_SIZE"),
            Arc::from("SEEK_SET"),
            Arc::from("SEEK_CUR"),
            Arc::from("SEEK_END"),
            Arc::from("IOBase"),
            Arc::from("RawIOBase"),
            Arc::from("BufferedIOBase"),
            Arc::from("TextIOBase"),
            Arc::from("UnsupportedOperation"),
        ];

        Self {
            module_name: Arc::from(module_name),
            attrs,
        }
    }
}

impl Default for IoModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for IoModule {
    fn name(&self) -> &str {
        self.module_name.as_ref()
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "DEFAULT_BUFFER_SIZE" => Ok(Value::int(DEFAULT_BUFFER_SIZE as i64).unwrap()),
            "SEEK_SET" => Ok(Value::int(SEEK_SET as i64).unwrap()),
            "SEEK_CUR" => Ok(Value::int(SEEK_CUR as i64).unwrap()),
            "SEEK_END" => Ok(Value::int(SEEK_END as i64).unwrap()),
            "TextIOWrapper" => Ok(Value::object_ptr(
                &*TEXT_IO_WRAPPER_PLACEHOLDER as *const BuiltinFunctionObject as *const (),
            )),
            "StringIO" | "BytesIO" | "open" | "FileIO" | "BufferedReader" | "BufferedWriter"
            | "BufferedRandom" | "IOBase" | "RawIOBase" | "BufferedIOBase" | "TextIOBase" => {
                Err(ModuleError::AttributeError(format!(
                    "{}.{} is not yet callable as an object",
                    self.module_name, name
                )))
            }
            "UnsupportedOperation" => Err(ModuleError::AttributeError(format!(
                "{}.UnsupportedOperation is not yet accessible as a type",
                self.module_name
            ))),
            _ => Err(ModuleError::AttributeError(format!(
                "module '{}' has no attribute '{}'",
                self.module_name, name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn builtin_text_io_wrapper_placeholder(args: &[Value]) -> Result<Value, BuiltinError> {
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "io.TextIOWrapper() is not implemented yet in Prism".to_string(),
    ))
}
