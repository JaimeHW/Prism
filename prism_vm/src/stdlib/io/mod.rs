//! Python `io` module implementation.
//!
//! High-performance I/O primitives providing:
//! - `StringIO` - In-memory text stream
//! - `BytesIO` - In-memory binary stream
//! - `FileMode` - Mode string parser
//! - Constants - `SEEK_SET`, `SEEK_CUR`, `SEEK_END`, `DEFAULT_BUFFER_SIZE`

pub mod bytes_io;
mod newline_decoder;
pub mod open_fn;
mod python_streams;
pub mod string_io;


pub use bytes_io::BytesIO;
pub use open_fn::{DEFAULT_BUFFER_SIZE, FileMode, SEEK_CUR, SEEK_END, SEEK_SET};
pub use string_io::{IoError, StringIO};

use super::{Module, ModuleError, ModuleResult};
use crate::stdlib::io::newline_decoder::incremental_newline_decoder_constructor_value;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use std::sync::{Arc, LazyLock};

pub(crate) use python_streams::{
    new_stderr_stream_object, new_stdin_stream_object, new_stdout_stream_object,
    open_file_stream_object,
};
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
            Arc::from("open_code"),
            Arc::from("FileIO"),
            Arc::from("BufferedReader"),
            Arc::from("BufferedWriter"),
            Arc::from("BufferedRandom"),
            Arc::from("TextIOWrapper"),
            Arc::from("IncrementalNewlineDecoder"),
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
            "open_code" => Ok(python_streams::open_code_function_value()),
            "open" | "FileIO" | "BufferedReader" | "BufferedWriter" | "BufferedRandom" => {
                Ok(python_streams::open_function_value())
            }
            "TextIOWrapper" => Ok(python_streams::text_io_wrapper_class_value()),
            "IncrementalNewlineDecoder" => Ok(incremental_newline_decoder_constructor_value()),
            "StringIO" => Ok(python_streams::string_io_class_value()),
            "BytesIO" => Ok(python_streams::bytes_io_class_value()),
            "IOBase" => Ok(io_class_value(&IO_BASE_CLASS)),
            "RawIOBase" => Ok(io_class_value(&RAW_IO_BASE_CLASS)),
            "BufferedIOBase" => Ok(io_class_value(&BUFFERED_IO_BASE_CLASS)),
            "TextIOBase" => Ok(io_class_value(&TEXT_IO_BASE_CLASS)),
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

fn io_class_value(class: &LazyLock<Arc<PyClassObject>>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

fn build_io_root_class(name: &'static str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);
    let class = Arc::new(class);
    finalize_io_class(name, class)
}

fn build_io_subclass(
    name: &'static str,
    base: &LazyLock<Arc<PyClassObject>>,
) -> Arc<PyClassObject> {
    build_io_subclass_with_flags(name, base, ClassFlags::empty())
}

pub(super) fn build_io_subclass_with_flags(
    name: &'static str,
    base: &LazyLock<Arc<PyClassObject>>,
    extra_flags: ClassFlags,
) -> Arc<PyClassObject> {
    let parent = &**base;
    let mut class = PyClassObject::new(intern(name), &[parent.class_id()], |class_id| {
        (class_id == parent.class_id()).then(|| parent.mro().to_vec().into())
    })
    .expect("stdlib io class hierarchy should have a valid MRO");
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE | extra_flags);
    let class = Arc::new(class);
    finalize_io_class(name, class)
}

pub(super) fn finalize_io_class(
    name: &'static str,
    class: Arc<PyClassObject>,
) -> Arc<PyClassObject> {
    class.set_attr(intern("__module__"), Value::string(intern("io")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);

    class
}
