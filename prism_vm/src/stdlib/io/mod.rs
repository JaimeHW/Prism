//! Python `io` module implementation.
//!
//! High-performance I/O primitives providing:
//! - `StringIO` - In-memory text stream
//! - `BytesIO` - In-memory binary stream
//! - `FileMode` - Mode string parser
//! - Constants - `SEEK_SET`, `SEEK_CUR`, `SEEK_END`, `DEFAULT_BUFFER_SIZE`

pub mod bytes_io;
pub mod open_fn;
mod python_streams;
pub mod string_io;

#[cfg(test)]
mod tests;

pub use bytes_io::BytesIO;
pub use open_fn::{DEFAULT_BUFFER_SIZE, FileMode, SEEK_CUR, SEEK_END, SEEK_SET};
pub use string_io::{IoError, StringIO};

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use std::sync::{Arc, LazyLock};

pub(crate) use python_streams::{
    new_stderr_stream_object, new_stdin_stream_object, new_stdout_stream_object,
    open_file_stream_object,
};

static TEXT_IO_WRAPPER_PLACEHOLDER: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("io.TextIOWrapper"),
        builtin_text_io_wrapper_placeholder,
    )
});
static IO_BASE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_io_root_class("IOBase"));
static RAW_IO_BASE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_io_subclass("RawIOBase", &IO_BASE_CLASS));
static BUFFERED_IO_BASE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_io_subclass("BufferedIOBase", &IO_BASE_CLASS));
static TEXT_IO_BASE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_io_subclass("TextIOBase", &IO_BASE_CLASS));

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
            "StringIO" => Ok(python_streams::string_io_constructor_value()),
            "BytesIO" => Ok(python_streams::bytes_io_constructor_value()),
            "IOBase" => Ok(io_class_value(&IO_BASE_CLASS)),
            "RawIOBase" => Ok(io_class_value(&RAW_IO_BASE_CLASS)),
            "BufferedIOBase" => Ok(io_class_value(&BUFFERED_IO_BASE_CLASS)),
            "TextIOBase" => Ok(io_class_value(&TEXT_IO_BASE_CLASS)),
            "open" | "FileIO" | "BufferedReader" | "BufferedWriter" | "BufferedRandom" => {
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

fn io_class_value(class: &LazyLock<Arc<PyClassObject>>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

fn build_io_root_class(name: &'static str) -> Arc<PyClassObject> {
    let class = Arc::new(PyClassObject::new_simple(intern(name)));
    finalize_io_class(name, class)
}

fn build_io_subclass(
    name: &'static str,
    base: &LazyLock<Arc<PyClassObject>>,
) -> Arc<PyClassObject> {
    let parent = &**base;
    let class = Arc::new(
        PyClassObject::new(intern(name), &[parent.class_id()], |class_id| {
            (class_id == parent.class_id()).then(|| parent.mro().to_vec().into())
        })
        .expect("stdlib io class hierarchy should have a valid MRO"),
    );
    finalize_io_class(name, class)
}

fn finalize_io_class(name: &'static str, class: Arc<PyClassObject>) -> Arc<PyClassObject> {
    class.set_attr(intern("__module__"), Value::string(intern("io")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);

    class
}
