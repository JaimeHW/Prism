use super::{BytesIO, DEFAULT_BUFFER_SIZE, IoError, SEEK_CUR, StringIO};
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class,
    builtin_type_object_type_id,
};
use crate::ops::objects::extract_type_id;
use crate::stdlib::sys::standard_streams;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::io::mode::FileMode as HostFileMode;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, LazyLock};

const STREAM_KIND_ATTR: &str = "_prism_stream_kind";
const STRING_BUFFER_ATTR: &str = "_prism_string_buffer";
const BYTES_BUFFER_ATTR: &str = "_prism_bytes_buffer";
const FILE_PATH_ATTR: &str = "_prism_file_path";
const FILE_FD_ATTR: &str = "_prism_file_fd";
const FILE_MODE_ATTR: &str = "_prism_file_mode";
const CLOSEFD_ATTR: &str = "_prism_closefd";
const POSITION_ATTR: &str = "_prism_stream_position";
const CLOSED_ATTR: &str = "closed";
const ENCODING_ATTR: &str = "encoding";
const STRING_IO_DOC: &str = "Text I/O implementation using an in-memory buffer.";
const BYTES_IO_DOC: &str = "Buffered I/O implementation using an in-memory bytes buffer.";

static STRING_IO_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_string_io_class);
static BYTES_IO_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_bytes_io_class);
static TEXT_IO_WRAPPER_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(build_text_io_wrapper_class);
static STRING_IO_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("io.StringIO.__new__"), string_io_new)
});
static STRING_IO_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("io.StringIO.__init__"), string_io_init)
});
static BYTES_IO_NEW_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("io.BytesIO.__new__"), bytes_io_new));
static BYTES_IO_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("io.BytesIO.__init__"), bytes_io_init)
});
static TEXT_IO_WRAPPER_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("io.TextIOWrapper.__new__"), text_io_wrapper_new)
});
static TEXT_IO_WRAPPER_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("io.TextIOWrapper.__init__"), text_io_wrapper_init)
});
static STREAM_WRITE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.write"), stream_write));
static STREAM_FLUSH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.flush"), stream_flush));
static STREAM_READ_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.read"), stream_read));
static STREAM_READABLE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.readable"), stream_readable));
static STREAM_WRITABLE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.writable"), stream_writable));
static STREAM_SEEKABLE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.seekable"), stream_seekable));
static STREAM_CLOSE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.close"), stream_close));
static STREAM_READLINE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.readline"), stream_readline));
static STREAM_READLINES_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("io._stream.readlines"), stream_readlines)
});
static STREAM_ENTER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.__enter__"), stream_enter));
static STREAM_EXIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.__exit__"), stream_exit));
static STRING_IO_GETVALUE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("io.StringIO.getvalue"), string_io_getvalue)
});
static STRING_IO_READ_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.StringIO.read"), string_io_read));
static STRING_IO_SEEK_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.StringIO.seek"), string_io_seek));
static STRING_IO_TELL_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.StringIO.tell"), string_io_tell));
static STRING_IO_TRUNCATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("io.StringIO.truncate"), string_io_truncate)
});
static BYTES_IO_GETVALUE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("io.BytesIO.getvalue"), bytes_io_getvalue)
});
static BYTES_IO_GETBUFFER_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("io.BytesIO.getbuffer"), bytes_io_getbuffer)
});
static BYTES_IO_READ_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.BytesIO.read"), bytes_io_read));
static BYTES_IO_SEEK_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.BytesIO.seek"), bytes_io_seek));
static BYTES_IO_TELL_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.BytesIO.tell"), bytes_io_tell));
static BYTES_IO_TRUNCATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("io.BytesIO.truncate"), bytes_io_truncate)
});
static FILE_STREAM_READ_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.File.read"), file_stream_read));
static FILE_STREAM_SEEK_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.File.seek"), file_stream_seek));
static FILE_STREAM_TELL_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.File.tell"), file_stream_tell));
static FILE_STREAM_TRUNCATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("io.File.truncate"), file_stream_truncate)
});
static FILE_STREAM_FILENO_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.File.fileno"), file_stream_fileno));
static OPEN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("io.open"), io_open));
static OPEN_CODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.open_code"), io_open_code));

#[cfg(windows)]
#[link(name = "ucrt")]
unsafe extern "C" {
    #[link_name = "_read"]
    fn crt_read(fd: i32, buffer: *mut c_void, count: u32) -> i32;
    #[link_name = "_write"]
    fn crt_write(fd: i32, buffer: *const c_void, count: u32) -> i32;
    #[link_name = "_close"]
    fn crt_close(fd: i32) -> i32;
    #[link_name = "_lseeki64"]
    fn crt_lseeki64(fd: i32, offset: i64, origin: i32) -> i64;
    #[link_name = "_chsize_s"]
    fn crt_chsize_s(fd: i32, size: i64) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamKind {
    StringIo,
    BytesIo,
    TextFile,
    BinaryFile,
    StdIn,
    StdOut,
    StdErr,
    StdInBuffer,
    StdOutBuffer,
    StdErrBuffer,
}

impl StreamKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::StringIo => "stringio",
            Self::BytesIo => "bytesio",
            Self::TextFile => "textfile",
            Self::BinaryFile => "binaryfile",
            Self::StdIn => "stdin",
            Self::StdOut => "stdout",
            Self::StdErr => "stderr",
            Self::StdInBuffer => "stdin.buffer",
            Self::StdOutBuffer => "stdout.buffer",
            Self::StdErrBuffer => "stderr.buffer",
        }
    }

    fn from_receiver(receiver: Value) -> Result<Self, BuiltinError> {
        let shaped = shaped_object_ref(receiver)?;
        let kind_value = shaped
            .get_property(STREAM_KIND_ATTR)
            .ok_or_else(|| BuiltinError::AttributeError("missing stream kind".to_string()))?;
        let kind = string_from_value(kind_value, "stream kind")?;
        match kind.as_str() {
            "stringio" => Ok(Self::StringIo),
            "bytesio" => Ok(Self::BytesIo),
            "textfile" => Ok(Self::TextFile),
            "binaryfile" => Ok(Self::BinaryFile),
            "stdin" => Ok(Self::StdIn),
            "stdout" => Ok(Self::StdOut),
            "stderr" => Ok(Self::StdErr),
            "stdin.buffer" => Ok(Self::StdInBuffer),
            "stdout.buffer" => Ok(Self::StdOutBuffer),
            "stderr.buffer" => Ok(Self::StdErrBuffer),
            _ => Err(BuiltinError::TypeError(format!(
                "unsupported stream backend '{}'",
                kind
            ))),
        }
    }
}

pub(super) fn string_io_class_value() -> Value {
    class_value(string_io_class())
}

pub(super) fn bytes_io_class_value() -> Value {
    class_value(bytes_io_class())
}

pub(super) fn text_io_wrapper_class_value() -> Value {
    class_value(text_io_wrapper_class())
}

pub(super) fn open_function_value() -> Value {
    builtin_value(&OPEN_FUNCTION)
}

pub(super) fn open_code_function_value() -> Value {
    builtin_value(&OPEN_CODE_FUNCTION)
}

fn string_io_class() -> &'static Arc<PyClassObject> {
    &STRING_IO_CLASS
}

fn bytes_io_class() -> &'static Arc<PyClassObject> {
    &BYTES_IO_CLASS
}

fn text_io_wrapper_class() -> &'static Arc<PyClassObject> {
    &TEXT_IO_WRAPPER_CLASS
}

fn build_string_io_class() -> Arc<PyClassObject> {
    let class = super::build_io_subclass_with_flags(
        "StringIO",
        &super::TEXT_IO_BASE_CLASS,
        ClassFlags::HAS_NEW | ClassFlags::HAS_INIT,
    );
    class.set_attr(intern("__doc__"), string_value(STRING_IO_DOC));
    class.set_attr(intern("__new__"), builtin_value(&STRING_IO_NEW_METHOD));
    class.set_attr(intern("__init__"), builtin_value(&STRING_IO_INIT_METHOD));
    set_common_stream_methods(class.as_ref());
    class.set_attr(
        intern("getvalue"),
        builtin_value(&STRING_IO_GETVALUE_METHOD),
    );
    class.set_attr(intern("read"), builtin_value(&STRING_IO_READ_METHOD));
    class.set_attr(intern("seek"), builtin_value(&STRING_IO_SEEK_METHOD));
    class.set_attr(intern("tell"), builtin_value(&STRING_IO_TELL_METHOD));
    class.set_attr(
        intern("truncate"),
        builtin_value(&STRING_IO_TRUNCATE_METHOD),
    );
    class
}

fn build_bytes_io_class() -> Arc<PyClassObject> {
    let class = super::build_io_subclass_with_flags(
        "BytesIO",
        &super::BUFFERED_IO_BASE_CLASS,
        ClassFlags::HAS_NEW | ClassFlags::HAS_INIT,
    );
    class.set_attr(intern("__doc__"), string_value(BYTES_IO_DOC));
    class.set_attr(intern("__new__"), builtin_value(&BYTES_IO_NEW_METHOD));
    class.set_attr(intern("__init__"), builtin_value(&BYTES_IO_INIT_METHOD));
    set_common_stream_methods(class.as_ref());
    class.set_attr(intern("getvalue"), builtin_value(&BYTES_IO_GETVALUE_METHOD));
    class.set_attr(
        intern("getbuffer"),
        builtin_value(&BYTES_IO_GETBUFFER_METHOD),
    );
    class.set_attr(intern("read"), builtin_value(&BYTES_IO_READ_METHOD));
    class.set_attr(intern("seek"), builtin_value(&BYTES_IO_SEEK_METHOD));
    class.set_attr(intern("tell"), builtin_value(&BYTES_IO_TELL_METHOD));
    class.set_attr(intern("truncate"), builtin_value(&BYTES_IO_TRUNCATE_METHOD));
    class
}

fn build_text_io_wrapper_class() -> Arc<PyClassObject> {
    let class = super::build_io_subclass_with_flags(
        "TextIOWrapper",
        &super::TEXT_IO_BASE_CLASS,
        ClassFlags::HAS_NEW | ClassFlags::HAS_INIT,
    );
    class.set_attr(
        intern("__doc__"),
        string_value("Character and line based wrapper over a buffered binary stream."),
    );
    class.set_attr(
        intern("__new__"),
        builtin_value(&TEXT_IO_WRAPPER_NEW_METHOD),
    );
    class.set_attr(
        intern("__init__"),
        builtin_value(&TEXT_IO_WRAPPER_INIT_METHOD),
    );
    set_common_stream_methods(class.as_ref());
    set_file_stream_methods(class.as_ref());
    class
}

fn set_common_stream_methods(class: &PyClassObject) {
    class.set_attr(intern("write"), builtin_value(&STREAM_WRITE_METHOD));
    class.set_attr(intern("flush"), builtin_value(&STREAM_FLUSH_METHOD));
    class.set_attr(intern("readable"), builtin_value(&STREAM_READABLE_METHOD));
    class.set_attr(intern("writable"), builtin_value(&STREAM_WRITABLE_METHOD));
    class.set_attr(intern("seekable"), builtin_value(&STREAM_SEEKABLE_METHOD));
    class.set_attr(intern("close"), builtin_value(&STREAM_CLOSE_METHOD));
    class.set_attr(intern("readline"), builtin_value(&STREAM_READLINE_METHOD));
    class.set_attr(intern("readlines"), builtin_value(&STREAM_READLINES_METHOD));
    class.set_attr(intern("__enter__"), builtin_value(&STREAM_ENTER_METHOD));
    class.set_attr(intern("__exit__"), builtin_value(&STREAM_EXIT_METHOD));
}

fn set_file_stream_methods(class: &PyClassObject) {
    class.set_attr(intern("read"), builtin_value(&FILE_STREAM_READ_METHOD));
    class.set_attr(intern("seek"), builtin_value(&FILE_STREAM_SEEK_METHOD));
    class.set_attr(intern("tell"), builtin_value(&FILE_STREAM_TELL_METHOD));
    class.set_attr(
        intern("truncate"),
        builtin_value(&FILE_STREAM_TRUNCATE_METHOD),
    );
    class.set_attr(intern("fileno"), builtin_value(&FILE_STREAM_FILENO_METHOD));
}

pub(crate) fn new_stdin_stream_object() -> Value {
    new_standard_stream_object(StreamKind::StdIn)
}

pub(crate) fn new_stdout_stream_object() -> Value {
    new_standard_stream_object(StreamKind::StdOut)
}

pub(crate) fn new_stderr_stream_object() -> Value {
    new_standard_stream_object(StreamKind::StdErr)
}

pub(crate) fn open_file_stream_object(
    path: &str,
    mode: &str,
    encoding: Option<&str>,
) -> Result<Value, BuiltinError> {
    let parsed_mode = parse_file_mode(mode)?;

    if parsed_mode.binary {
        if encoding.is_some() {
            return Err(BuiltinError::ValueError(
                "binary mode doesn't take an encoding argument".to_string(),
            ));
        }
    } else if let Some(name) = encoding {
        validate_encoding(name)?;
    }

    parsed_mode
        .to_open_options()
        .open(path)
        .map_err(host_stream_error)?;

    let initial_position = if parsed_mode.append {
        std::fs::metadata(path).map_err(host_stream_error)?.len()
    } else {
        0
    };

    Ok(new_file_stream_object(
        path,
        mode,
        if parsed_mode.binary {
            None
        } else {
            Some(encoding.unwrap_or("utf-8"))
        },
        parsed_mode.binary,
        initial_position,
    ))
}

pub(crate) fn open_fd_stream_object(
    fd: i32,
    mode: &str,
    encoding: Option<&str>,
    closefd: bool,
) -> Result<Value, BuiltinError> {
    if fd < 0 {
        return Err(BuiltinError::ValueError(
            "negative file descriptor".to_string(),
        ));
    }

    let parsed_mode = parse_file_mode(mode)?;
    if parsed_mode.binary {
        if encoding.is_some() {
            return Err(BuiltinError::ValueError(
                "binary mode doesn't take an encoding argument".to_string(),
            ));
        }
    } else if let Some(name) = encoding {
        validate_encoding(name)?;
    }

    Ok(new_fd_stream_object(
        fd,
        mode,
        if parsed_mode.binary {
            None
        } else {
            Some(encoding.unwrap_or("utf-8"))
        },
        parsed_mode.binary,
        closefd,
    ))
}

#[derive(Debug)]
struct IoOpenArgs {
    file: Value,
    mode: String,
    buffering: i64,
    encoding: Option<String>,
    errors: Option<String>,
    newline: Option<String>,
    closefd: bool,
    opener: Option<Value>,
}

fn io_open(positional: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let args = parse_io_open_args(positional, keywords)?;
    if args.opener.is_some() {
        return Err(BuiltinError::NotImplemented(
            "io.open() opener callback is not implemented yet".to_string(),
        ));
    }
    if args.buffering == 0 && !parse_file_mode(&args.mode)?.binary {
        return Err(BuiltinError::ValueError(
            "can't have unbuffered text I/O".to_string(),
        ));
    }
    if let Some(errors) = args.errors.as_deref() {
        validate_text_error_policy(errors)?;
    }
    if let Some(newline) = args.newline.as_deref() {
        validate_newline_policy(newline)?;
    }

    if let Some(path) = string_value_to_owned(args.file) {
        if !args.closefd {
            return Err(BuiltinError::ValueError(
                "Cannot use closefd=False with file name".to_string(),
            ));
        }
        return open_file_stream_object(&path, &args.mode, args.encoding.as_deref());
    }

    let fd = i32::try_from(int_like_value(args.file, "open() file descriptor")?).map_err(|_| {
        BuiltinError::OverflowError("open() file descriptor is out of range".to_string())
    })?;
    open_fd_stream_object(fd, &args.mode, args.encoding.as_deref(), args.closefd)
}

fn io_open_code(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "open_code() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let path = string_from_value(args[0], "open_code() path")?;
    open_file_stream_object(&path, "rb", None)
}

fn parse_io_open_args(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<IoOpenArgs, BuiltinError> {
    if positional.len() > 8 {
        return Err(BuiltinError::TypeError(format!(
            "open() takes at most 8 arguments ({} given)",
            positional.len()
        )));
    }

    let mut slots: [Option<Value>; 8] = [None; 8];
    for (index, &value) in positional.iter().enumerate() {
        slots[index] = Some(value);
    }

    for &(name, value) in keywords {
        let index = match name {
            "file" => 0,
            "mode" => 1,
            "buffering" => 2,
            "encoding" => 3,
            "errors" => 4,
            "newline" => 5,
            "closefd" => 6,
            "opener" => 7,
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "'{name}' is an invalid keyword argument for open()"
                )));
            }
        };
        if slots[index].replace(value).is_some() {
            return Err(BuiltinError::TypeError(format!(
                "open() got multiple values for argument '{name}'"
            )));
        }
    }

    let file = slots[0]
        .ok_or_else(|| BuiltinError::TypeError("open() missing required argument 'file'".into()))?;
    let mode = match slots[1] {
        Some(value) => string_from_value(value, "open() mode")?,
        None => "r".to_string(),
    };
    let buffering = match slots[2] {
        Some(value) => int_like_value(value, "open() buffering")?,
        None => -1,
    };
    let encoding = optional_string(slots[3], "open() encoding")?;
    let errors = optional_string(slots[4], "open() errors")?;
    let newline = optional_string(slots[5], "open() newline")?;
    let closefd = match slots[6] {
        Some(value) => truthy_closefd(value)?,
        None => true,
    };
    let opener = slots[7].filter(|value| !value.is_none());

    Ok(IoOpenArgs {
        file,
        mode,
        buffering,
        encoding,
        errors,
        newline,
        closefd,
        opener,
    })
}

fn string_io_new(positional: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let Some(&class_value) = positional.first() else {
        return Err(BuiltinError::TypeError(
            "StringIO.__new__() missing type argument".to_string(),
        ));
    };
    let class = heap_type_from_value(class_value, "StringIO.__new__")?;
    let _ = parse_string_io_init_args(&positional[1..], keywords)?;

    let instance = new_heap_instance(class);
    initialize_string_io_state(instance, "")?;
    Ok(instance)
}

fn string_io_init(positional: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let Some(&receiver) = positional.first() else {
        return Err(BuiltinError::TypeError(
            "StringIO.__init__() missing self".to_string(),
        ));
    };
    let initial = parse_string_io_init_args(&positional[1..], keywords)?;
    initialize_string_io_state(receiver, &initial)?;
    Ok(Value::none())
}

fn bytes_io_new(positional: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let Some(&class_value) = positional.first() else {
        return Err(BuiltinError::TypeError(
            "BytesIO.__new__() missing type argument".to_string(),
        ));
    };
    let class = heap_type_from_value(class_value, "BytesIO.__new__")?;
    let _ = parse_bytes_io_init_args(&positional[1..], keywords)?;

    let instance = new_heap_instance(class);
    initialize_bytes_io_state(instance, &[])?;
    Ok(instance)
}

fn bytes_io_init(positional: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let Some(&receiver) = positional.first() else {
        return Err(BuiltinError::TypeError(
            "BytesIO.__init__() missing self".to_string(),
        ));
    };
    let initial = parse_bytes_io_init_args(&positional[1..], keywords)?;
    initialize_bytes_io_state(receiver, &initial)?;
    Ok(Value::none())
}

fn text_io_wrapper_new(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let Some(&class_value) = positional.first() else {
        return Err(BuiltinError::TypeError(
            "TextIOWrapper.__new__() missing type argument".to_string(),
        ));
    };
    let class = heap_type_from_value(class_value, "TextIOWrapper.__new__")?;
    let init_args = parse_text_io_wrapper_init_args(&positional[1..], keywords)?;

    let instance = new_heap_instance(class);
    initialize_text_io_wrapper_state(instance, &init_args)?;
    Ok(instance)
}

fn text_io_wrapper_init(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let Some(&receiver) = positional.first() else {
        return Err(BuiltinError::TypeError(
            "TextIOWrapper.__init__() missing self".to_string(),
        ));
    };
    let init_args = parse_text_io_wrapper_init_args(&positional[1..], keywords)?;
    initialize_text_io_wrapper_state(receiver, &init_args)?;
    Ok(Value::none())
}

fn stream_write(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "write() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    match StreamKind::from_receiver(receiver)? {
        StreamKind::StringIo => {
            let mut stream = load_string_io(receiver)?;
            let data = string_from_value(args[1], "write() argument")?;
            let written = stream.write(&data).map_err(map_io_error)?;
            store_string_io(receiver, stream)?;
            Ok(Value::int(written as i64).unwrap())
        }
        StreamKind::BytesIo => {
            let mut stream = load_bytes_io(receiver)?;
            let data = bytes_from_value(args[1], "write() argument")?;
            let written = stream.write(&data).map_err(map_io_error)?;
            store_bytes_io(receiver, stream)?;
            Ok(Value::int(written as i64).unwrap())
        }
        StreamKind::TextFile => {
            let data = string_from_value(args[1], "write() argument")?;
            let written = write_text_file(receiver, &data)?;
            Ok(Value::int(written as i64).unwrap())
        }
        StreamKind::BinaryFile => {
            let data = bytes_from_value(args[1], "write() argument")?;
            let written = write_binary_file(receiver, &data)?;
            Ok(Value::int(written as i64).unwrap())
        }
        StreamKind::StdOut => {
            let data = string_from_value(args[1], "write() argument")?;
            standard_streams()
                .write_stdout(&data)
                .map_err(host_stream_error)?;
            Ok(Value::int(data.len() as i64).unwrap())
        }
        StreamKind::StdErr => {
            let data = string_from_value(args[1], "write() argument")?;
            standard_streams()
                .write_stderr(&data)
                .map_err(host_stream_error)?;
            Ok(Value::int(data.len() as i64).unwrap())
        }
        StreamKind::StdOutBuffer => {
            let data = bytes_from_value(args[1], "write() argument")?;
            standard_streams()
                .write_stdout_bytes(&data)
                .map_err(host_stream_error)?;
            Ok(Value::int(data.len() as i64).unwrap())
        }
        StreamKind::StdErrBuffer => {
            let data = bytes_from_value(args[1], "write() argument")?;
            standard_streams()
                .write_stderr_bytes(&data)
                .map_err(host_stream_error)?;
            Ok(Value::int(data.len() as i64).unwrap())
        }
        StreamKind::StdIn | StreamKind::StdInBuffer => {
            Err(BuiltinError::OSError("stdin is not writable".to_string()))
        }
    }
}

fn stream_flush(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "flush() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    match StreamKind::from_receiver(args[0])? {
        StreamKind::StdOut | StreamKind::StdOutBuffer => standard_streams()
            .flush_stdout()
            .map_err(host_stream_error)?,
        StreamKind::StdErr | StreamKind::StdErrBuffer => standard_streams()
            .flush_stderr()
            .map_err(host_stream_error)?,
        StreamKind::StringIo
        | StreamKind::BytesIo
        | StreamKind::TextFile
        | StreamKind::BinaryFile
        | StreamKind::StdIn
        | StreamKind::StdInBuffer => {}
    }

    Ok(Value::none())
}

fn stream_read(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "read() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let count = optional_count(args.get(1).copied(), "read")?;
    match StreamKind::from_receiver(receiver)? {
        StreamKind::StdIn => {
            let bytes = standard_streams()
                .read_bytes(count)
                .map_err(host_stream_error)?;
            Ok(string_value(&decode_text_bytes(&bytes, "utf-8")?))
        }
        StreamKind::StdInBuffer => {
            let bytes = standard_streams()
                .read_bytes(count)
                .map_err(host_stream_error)?;
            Ok(bytes_value(&bytes))
        }
        other => Err(BuiltinError::OSError(format!(
            "{} is not readable",
            other.as_str()
        ))),
    }
}

fn stream_readable(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "readable")?;
    let readable = match StreamKind::from_receiver(receiver)? {
        StreamKind::StringIo
        | StreamKind::BytesIo
        | StreamKind::StdIn
        | StreamKind::StdInBuffer => true,
        StreamKind::TextFile | StreamKind::BinaryFile => file_mode(receiver)?.read,
        StreamKind::StdOut
        | StreamKind::StdErr
        | StreamKind::StdOutBuffer
        | StreamKind::StdErrBuffer => false,
    };
    Ok(Value::bool(readable && !is_closed(receiver)?))
}

fn stream_writable(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "writable")?;
    let writable = match StreamKind::from_receiver(receiver)? {
        StreamKind::StringIo
        | StreamKind::BytesIo
        | StreamKind::StdOut
        | StreamKind::StdErr
        | StreamKind::StdOutBuffer
        | StreamKind::StdErrBuffer => true,
        StreamKind::TextFile | StreamKind::BinaryFile => file_mode(receiver)?.write,
        StreamKind::StdIn | StreamKind::StdInBuffer => false,
    };
    Ok(Value::bool(writable && !is_closed(receiver)?))
}

fn stream_seekable(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "seekable")?;
    let seekable = match StreamKind::from_receiver(receiver)? {
        StreamKind::StringIo | StreamKind::BytesIo => true,
        StreamKind::TextFile | StreamKind::BinaryFile => match file_descriptor(receiver)? {
            Some(fd) => fd_seek_current(fd).is_ok(),
            None => true,
        },
        StreamKind::StdIn
        | StreamKind::StdOut
        | StreamKind::StdErr
        | StreamKind::StdInBuffer
        | StreamKind::StdOutBuffer
        | StreamKind::StdErrBuffer => false,
    };
    Ok(Value::bool(seekable && !is_closed(receiver)?))
}

fn stream_close(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "close")?;
    if !is_closed(receiver)? {
        if matches!(
            StreamKind::from_receiver(receiver)?,
            StreamKind::TextFile | StreamKind::BinaryFile
        ) && file_closefd(receiver)?
        {
            if let Some(fd) = file_descriptor(receiver)? {
                close_fd(fd).map_err(host_stream_error)?;
            }
        }
    }
    set_closed(receiver, true)?;
    Ok(Value::none())
}

fn stream_readline(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "readline")?;
    match StreamKind::from_receiver(receiver)? {
        StreamKind::StringIo => {
            let mut stream = load_string_io(receiver)?;
            let line = stream.readline().map_err(map_io_error)?.to_string();
            store_string_io(receiver, stream)?;
            Ok(string_value(&line))
        }
        StreamKind::BytesIo => {
            let mut stream = load_bytes_io(receiver)?;
            let line = stream.readline().map_err(map_io_error)?.to_vec();
            store_bytes_io(receiver, stream)?;
            Ok(bytes_value(&line))
        }
        StreamKind::TextFile => Ok(string_value(&read_text_file_line(receiver)?)),
        StreamKind::BinaryFile => Ok(bytes_value(&read_binary_file_line(receiver)?)),
        StreamKind::StdIn => {
            let line = standard_streams().read_line().map_err(host_stream_error)?;
            Ok(string_value(&line))
        }
        StreamKind::StdInBuffer => {
            let line = standard_streams()
                .read_line_bytes()
                .map_err(host_stream_error)?;
            Ok(bytes_value(&line))
        }
        StreamKind::StdOut
        | StreamKind::StdErr
        | StreamKind::StdOutBuffer
        | StreamKind::StdErrBuffer => {
            Err(BuiltinError::OSError("stream is not readable".to_string()))
        }
    }
}

fn stream_readlines(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "readlines() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let hint = optional_count(args.get(1).copied(), "hint")?;
    let lines = match StreamKind::from_receiver(receiver)? {
        StreamKind::StringIo => {
            let mut stream = load_string_io(receiver)?;
            let mut lines = Vec::new();
            let mut total = 0usize;
            loop {
                let line = stream.readline().map_err(map_io_error)?.to_string();
                if line.is_empty() {
                    break;
                }
                total = total.saturating_add(line.len());
                lines.push(string_value(&line));
                if let Some(limit) = hint {
                    if total >= limit {
                        break;
                    }
                }
            }
            store_string_io(receiver, stream)?;
            lines
        }
        StreamKind::BytesIo => {
            let mut stream = load_bytes_io(receiver)?;
            let mut lines = Vec::new();
            let mut total = 0usize;
            loop {
                let line = stream.readline().map_err(map_io_error)?.to_vec();
                if line.is_empty() {
                    break;
                }
                total = total.saturating_add(line.len());
                lines.push(bytes_value(&line));
                if let Some(limit) = hint {
                    if total >= limit {
                        break;
                    }
                }
            }
            store_bytes_io(receiver, stream)?;
            lines
        }
        StreamKind::TextFile => read_text_file_lines(receiver, hint)?
            .into_iter()
            .map(|line| string_value(&line))
            .collect(),
        StreamKind::BinaryFile => read_binary_file_lines(receiver, hint)?
            .into_iter()
            .map(|line| bytes_value(&line))
            .collect(),
        StreamKind::StdIn => {
            let mut lines = Vec::new();
            let mut total = 0usize;
            loop {
                let line = standard_streams().read_line().map_err(host_stream_error)?;
                if line.is_empty() {
                    break;
                }
                total = total.saturating_add(line.len());
                lines.push(string_value(&line));
                if let Some(limit) = hint {
                    if total >= limit {
                        break;
                    }
                }
            }
            lines
        }
        StreamKind::StdInBuffer => {
            let mut lines = Vec::new();
            let mut total = 0usize;
            loop {
                let line = standard_streams()
                    .read_line_bytes()
                    .map_err(host_stream_error)?;
                if line.is_empty() {
                    break;
                }
                total = total.saturating_add(line.len());
                lines.push(bytes_value(&line));
                if let Some(limit) = hint {
                    if total >= limit {
                        break;
                    }
                }
            }
            lines
        }
        StreamKind::StdOut
        | StreamKind::StdErr
        | StreamKind::StdOutBuffer
        | StreamKind::StdErrBuffer => {
            return Err(BuiltinError::OSError("stream is not readable".to_string()));
        }
    };

    Ok(list_value(lines))
}

fn string_io_getvalue(args: &[Value]) -> Result<Value, BuiltinError> {
    let stream = load_string_io(expect_no_method_args(args, "getvalue")?)?;
    Ok(string_value(stream.getvalue()))
}

fn string_io_read(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "read() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let mut stream = load_string_io(receiver)?;
    let count = optional_count(args.get(1).copied(), "read")?;
    let value = stream.read(count).map_err(map_io_error)?.to_string();
    store_string_io(receiver, stream)?;
    Ok(string_value(&value))
}

fn string_io_seek(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "seek() takes 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let mut stream = load_string_io(receiver)?;
    let offset = int_like_value(args[1], "seek offset")?;
    let whence = args
        .get(2)
        .copied()
        .map(|value| int_like_value(value, "whence"))
        .transpose()?
        .unwrap_or(0);
    if whence < 0 {
        return Err(BuiltinError::ValueError(
            "whence must be non-negative".to_string(),
        ));
    }
    let position = stream.seek(offset, whence as u32).map_err(map_io_error)?;
    store_string_io(receiver, stream)?;
    Ok(Value::int(position as i64).unwrap())
}

fn string_io_tell(args: &[Value]) -> Result<Value, BuiltinError> {
    let stream = load_string_io(expect_no_method_args(args, "tell")?)?;
    Ok(Value::int(stream.tell().map_err(map_io_error)? as i64).unwrap())
}

fn string_io_truncate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "truncate() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let mut stream = load_string_io(receiver)?;
    let size = optional_size(args.get(1).copied(), "truncate")?;
    let result = stream.truncate(size).map_err(map_io_error)?;
    store_string_io(receiver, stream)?;
    Ok(Value::int(result as i64).unwrap())
}

fn bytes_io_getvalue(args: &[Value]) -> Result<Value, BuiltinError> {
    let stream = load_bytes_io(expect_no_method_args(args, "getvalue")?)?;
    Ok(bytes_value(stream.getvalue()))
}

fn bytes_io_getbuffer(args: &[Value]) -> Result<Value, BuiltinError> {
    let stream = load_bytes_io(expect_no_method_args(args, "getbuffer")?)?;
    Ok(bytes_value(stream.getbuffer()))
}

fn bytes_io_read(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "read() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let mut stream = load_bytes_io(receiver)?;
    let count = optional_count(args.get(1).copied(), "read")?;
    let value = stream.read(count).map_err(map_io_error)?.to_vec();
    store_bytes_io(receiver, stream)?;
    Ok(bytes_value(&value))
}

fn bytes_io_seek(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "seek() takes 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let mut stream = load_bytes_io(receiver)?;
    let offset = int_like_value(args[1], "seek offset")?;
    let whence = args
        .get(2)
        .copied()
        .map(|value| int_like_value(value, "whence"))
        .transpose()?
        .unwrap_or(0);
    if whence < 0 {
        return Err(BuiltinError::ValueError(
            "whence must be non-negative".to_string(),
        ));
    }
    let position = stream.seek(offset, whence as u32).map_err(map_io_error)?;
    store_bytes_io(receiver, stream)?;
    Ok(Value::int(position as i64).unwrap())
}

fn bytes_io_tell(args: &[Value]) -> Result<Value, BuiltinError> {
    let stream = load_bytes_io(expect_no_method_args(args, "tell")?)?;
    Ok(Value::int(stream.tell().map_err(map_io_error)? as i64).unwrap())
}

fn bytes_io_truncate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "truncate() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let mut stream = load_bytes_io(receiver)?;
    let size = optional_size(args.get(1).copied(), "truncate")?;
    let result = stream.truncate(size).map_err(map_io_error)?;
    store_bytes_io(receiver, stream)?;
    Ok(Value::int(result as i64).unwrap())
}

fn file_stream_read(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "read() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let count = optional_count(args.get(1).copied(), "read")?;
    match StreamKind::from_receiver(receiver)? {
        StreamKind::TextFile => Ok(string_value(&read_text_file(receiver, count)?)),
        StreamKind::BinaryFile => Ok(bytes_value(&read_binary_file(receiver, count)?)),
        other => Err(BuiltinError::TypeError(format!(
            "read() is not supported for {}",
            other.as_str()
        ))),
    }
}

fn file_stream_seek(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "seek() takes 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let offset = int_like_value(args[1], "seek offset")?;
    let whence = args
        .get(2)
        .copied()
        .map(|value| int_like_value(value, "whence"))
        .transpose()?
        .unwrap_or(0);
    if whence < 0 {
        return Err(BuiltinError::ValueError(
            "whence must be non-negative".to_string(),
        ));
    }

    match StreamKind::from_receiver(receiver)? {
        StreamKind::TextFile | StreamKind::BinaryFile => {
            let position = seek_file(receiver, offset, whence as u32)?;
            Ok(Value::int(position as i64).unwrap())
        }
        other => Err(BuiltinError::TypeError(format!(
            "seek() is not supported for {}",
            other.as_str()
        ))),
    }
}

fn file_stream_tell(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "tell")?;
    match StreamKind::from_receiver(receiver)? {
        StreamKind::TextFile | StreamKind::BinaryFile => {
            Ok(Value::int(stream_position(receiver)? as i64).unwrap())
        }
        other => Err(BuiltinError::TypeError(format!(
            "tell() is not supported for {}",
            other.as_str()
        ))),
    }
}

fn file_stream_truncate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "truncate() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let size = optional_size(args.get(1).copied(), "truncate")?;
    match StreamKind::from_receiver(receiver)? {
        StreamKind::TextFile | StreamKind::BinaryFile => {
            let size = truncate_file(receiver, size)?;
            Ok(Value::int(size as i64).unwrap())
        }
        other => Err(BuiltinError::TypeError(format!(
            "truncate() is not supported for {}",
            other.as_str()
        ))),
    }
}

fn file_stream_fileno(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "fileno")?;
    ensure_open(receiver)?;
    let Some(fd) = file_descriptor(receiver)? else {
        return Err(BuiltinError::OSError(
            "fileno() is not available for path-backed Prism streams".to_string(),
        ));
    };
    Ok(Value::int(i64::from(fd)).unwrap())
}

fn stream_enter(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_method_args(args, "__enter__")
}

fn stream_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "__exit__() takes 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    set_closed(args[0], true)?;
    Ok(Value::none())
}

fn new_heap_instance(class: &PyClassObject) -> Value {
    let instance = allocate_heap_instance_for_class(class);
    crate::alloc_managed_value(instance)
}

fn initialize_string_io_state(receiver: Value, initial: &str) -> Result<(), BuiltinError> {
    let shaped = shaped_object_mut(receiver)?;
    shaped.set_property(
        intern(STREAM_KIND_ATTR),
        string_value(StreamKind::StringIo.as_str()),
        shape_registry(),
    );
    shaped.set_property(
        intern(STRING_BUFFER_ATTR),
        string_value(initial),
        shape_registry(),
    );
    shaped.set_property(
        intern(POSITION_ATTR),
        Value::int(0).unwrap(),
        shape_registry(),
    );
    shaped.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
    Ok(())
}

fn initialize_bytes_io_state(receiver: Value, initial: &[u8]) -> Result<(), BuiltinError> {
    let shaped = shaped_object_mut(receiver)?;
    shaped.set_property(
        intern(STREAM_KIND_ATTR),
        string_value(StreamKind::BytesIo.as_str()),
        shape_registry(),
    );
    shaped.set_property(
        intern(BYTES_BUFFER_ATTR),
        bytes_value(initial),
        shape_registry(),
    );
    shaped.set_property(
        intern(POSITION_ATTR),
        Value::int(0).unwrap(),
        shape_registry(),
    );
    shaped.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
    Ok(())
}

fn initialize_text_io_wrapper_state(
    receiver: Value,
    init_args: &TextIoWrapperInitArgs,
) -> Result<(), BuiltinError> {
    let buffer = init_args.buffer;
    let buffer_kind = StreamKind::from_receiver(buffer)?;
    let target_kind = match buffer_kind {
        StreamKind::BinaryFile | StreamKind::TextFile => StreamKind::TextFile,
        StreamKind::StdIn | StreamKind::StdInBuffer => StreamKind::StdIn,
        StreamKind::StdOut | StreamKind::StdOutBuffer => StreamKind::StdOut,
        StreamKind::StdErr | StreamKind::StdErrBuffer => StreamKind::StdErr,
        _ => {
            return Err(BuiltinError::TypeError(
                "TextIOWrapper() argument 'buffer' must be a file-backed or standard stream"
                    .to_string(),
            ));
        }
    };

    let source = shaped_object_ref(buffer)?;
    let encoding = match init_args.encoding.as_deref() {
        Some(name) => name.to_string(),
        None => source
            .get_property(ENCODING_ATTR)
            .map(|value| string_from_value(value, ENCODING_ATTR))
            .transpose()?
            .unwrap_or_else(|| "utf-8".to_string()),
    };
    validate_encoding(&encoding)?;

    let receiver_shaped = shaped_object_mut(receiver)?;
    receiver_shaped.set_property(
        intern(STREAM_KIND_ATTR),
        string_value(target_kind.as_str()),
        shape_registry(),
    );
    receiver_shaped.set_property(
        intern(ENCODING_ATTR),
        string_value(&encoding),
        shape_registry(),
    );

    if let Some(path) = source.get_property(FILE_PATH_ATTR) {
        receiver_shaped.set_property(intern(FILE_PATH_ATTR), path, shape_registry());
    }
    if let Some(fd) = source.get_property(FILE_FD_ATTR) {
        receiver_shaped.set_property(intern(FILE_FD_ATTR), fd, shape_registry());
    }
    if let Some(mode) = source.get_property(FILE_MODE_ATTR) {
        receiver_shaped.set_property(intern(FILE_MODE_ATTR), mode, shape_registry());
    }
    if let Some(closefd) = source.get_property(CLOSEFD_ATTR) {
        receiver_shaped.set_property(intern(CLOSEFD_ATTR), closefd, shape_registry());
    }
    if let Some(position) = source.get_property(POSITION_ATTR) {
        receiver_shaped.set_property(intern(POSITION_ATTR), position, shape_registry());
    } else {
        receiver_shaped.set_property(
            intern(POSITION_ATTR),
            Value::int(0).unwrap(),
            shape_registry(),
        );
    }
    if let Some(closed) = source.get_property(CLOSED_ATTR) {
        receiver_shaped.set_property(intern(CLOSED_ATTR), closed, shape_registry());
    } else {
        receiver_shaped.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
    }

    Ok(())
}

fn new_file_stream_object(
    path: &str,
    mode: &str,
    encoding: Option<&str>,
    binary: bool,
    initial_position: u64,
) -> Value {
    let mut object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
    object.set_property(
        intern(STREAM_KIND_ATTR),
        Value::string(intern(if binary { "binaryfile" } else { "textfile" })),
        shape_registry(),
    );
    object.set_property(intern(FILE_PATH_ATTR), string_value(path), shape_registry());
    object.set_property(intern(FILE_MODE_ATTR), string_value(mode), shape_registry());
    object.set_property(
        intern(POSITION_ATTR),
        Value::int(initial_position as i64).unwrap(),
        shape_registry(),
    );
    object.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
    if let Some(encoding) = encoding {
        object.set_property(
            intern(ENCODING_ATTR),
            string_value(encoding),
            shape_registry(),
        );
    }
    install_file_stream_methods(&mut object);
    finalize_stream_object(object)
}

fn new_fd_stream_object(
    fd: i32,
    mode: &str,
    encoding: Option<&str>,
    binary: bool,
    closefd: bool,
) -> Value {
    let mut object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
    object.set_property(
        intern(STREAM_KIND_ATTR),
        Value::string(intern(if binary { "binaryfile" } else { "textfile" })),
        shape_registry(),
    );
    object.set_property(
        intern(FILE_FD_ATTR),
        Value::int(i64::from(fd)).unwrap(),
        shape_registry(),
    );
    object.set_property(intern(FILE_MODE_ATTR), string_value(mode), shape_registry());
    object.set_property(
        intern(POSITION_ATTR),
        Value::int(0).unwrap(),
        shape_registry(),
    );
    object.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
    object.set_property(intern(CLOSEFD_ATTR), Value::bool(closefd), shape_registry());
    if let Some(encoding) = encoding {
        object.set_property(
            intern(ENCODING_ATTR),
            string_value(encoding),
            shape_registry(),
        );
    }
    install_file_stream_methods(&mut object);
    finalize_stream_object(object)
}

fn install_file_stream_methods(object: &mut ShapedObject) {
    object.set_property(
        intern("write"),
        bound_builtin_value(&STREAM_WRITE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("flush"),
        bound_builtin_value(&STREAM_FLUSH_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("readable"),
        bound_builtin_value(&STREAM_READABLE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("writable"),
        bound_builtin_value(&STREAM_WRITABLE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("seekable"),
        bound_builtin_value(&STREAM_SEEKABLE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("close"),
        bound_builtin_value(&STREAM_CLOSE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("readline"),
        bound_builtin_value(&STREAM_READLINE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("readlines"),
        bound_builtin_value(&STREAM_READLINES_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("read"),
        bound_builtin_value(&FILE_STREAM_READ_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("seek"),
        bound_builtin_value(&FILE_STREAM_SEEK_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("tell"),
        bound_builtin_value(&FILE_STREAM_TELL_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("truncate"),
        bound_builtin_value(&FILE_STREAM_TRUNCATE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("fileno"),
        bound_builtin_value(&FILE_STREAM_FILENO_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("__enter__"),
        bound_builtin_value(&STREAM_ENTER_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("__exit__"),
        bound_builtin_value(&STREAM_EXIT_METHOD, Value::none()),
        shape_registry(),
    );
}

fn new_standard_stream_object(kind: StreamKind) -> Value {
    new_standard_stream_object_inner(kind, true)
}

fn new_standard_stream_buffer_object(kind: StreamKind) -> Value {
    new_standard_stream_object_inner(kind, false)
}

fn new_standard_stream_object_inner(kind: StreamKind, attach_buffer: bool) -> Value {
    let mut object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
    object.set_property(
        intern(STREAM_KIND_ATTR),
        Value::string(intern(kind.as_str())),
        shape_registry(),
    );
    object.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
    if matches!(
        kind,
        StreamKind::StdIn | StreamKind::StdOut | StreamKind::StdErr
    ) {
        object.set_property(
            intern(ENCODING_ATTR),
            Value::string(intern("utf-8")),
            shape_registry(),
        );
    }
    if attach_buffer && let Some(buffer_kind) = standard_stream_buffer_kind(kind) {
        object.set_property(
            intern("buffer"),
            new_standard_stream_buffer_object(buffer_kind),
            shape_registry(),
        );
    }
    object.set_property(
        intern("flush"),
        bound_builtin_value(&STREAM_FLUSH_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("readable"),
        bound_builtin_value(&STREAM_READABLE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("writable"),
        bound_builtin_value(&STREAM_WRITABLE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("seekable"),
        bound_builtin_value(&STREAM_SEEKABLE_METHOD, Value::none()),
        shape_registry(),
    );

    match kind {
        StreamKind::StdIn => {
            object.set_property(
                intern("read"),
                bound_builtin_value(&STREAM_READ_METHOD, Value::none()),
                shape_registry(),
            );
            object.set_property(
                intern("readline"),
                bound_builtin_value(&STREAM_READLINE_METHOD, Value::none()),
                shape_registry(),
            );
            object.set_property(
                intern("readlines"),
                bound_builtin_value(&STREAM_READLINES_METHOD, Value::none()),
                shape_registry(),
            );
        }
        StreamKind::StdOut | StreamKind::StdErr => {
            object.set_property(
                intern("write"),
                bound_builtin_value(&STREAM_WRITE_METHOD, Value::none()),
                shape_registry(),
            );
        }
        StreamKind::StdInBuffer => {
            object.set_property(
                intern("read"),
                bound_builtin_value(&STREAM_READ_METHOD, Value::none()),
                shape_registry(),
            );
            object.set_property(
                intern("readline"),
                bound_builtin_value(&STREAM_READLINE_METHOD, Value::none()),
                shape_registry(),
            );
            object.set_property(
                intern("readlines"),
                bound_builtin_value(&STREAM_READLINES_METHOD, Value::none()),
                shape_registry(),
            );
        }
        StreamKind::StdOutBuffer | StreamKind::StdErrBuffer => {
            object.set_property(
                intern("write"),
                bound_builtin_value(&STREAM_WRITE_METHOD, Value::none()),
                shape_registry(),
            );
        }
        StreamKind::StringIo | StreamKind::BytesIo => {}
        StreamKind::TextFile | StreamKind::BinaryFile => {
            unreachable!("file-backed streams must be created via new_file_stream_object")
        }
    }

    finalize_stream_object(object)
}

#[inline]
fn standard_stream_buffer_kind(kind: StreamKind) -> Option<StreamKind> {
    match kind {
        StreamKind::StdIn => Some(StreamKind::StdInBuffer),
        StreamKind::StdOut => Some(StreamKind::StdOutBuffer),
        StreamKind::StdErr => Some(StreamKind::StdErrBuffer),
        _ => None,
    }
}

fn finalize_stream_object(mut object: ShapedObject) -> Value {
    let value = crate::alloc_managed_value(object);
    let shaped = unsafe { &mut *(value.as_object_ptr().unwrap() as *mut ShapedObject) };
    for method_name in shaped.property_names() {
        if !matches!(
            method_name.as_str(),
            "write"
                | "flush"
                | "readable"
                | "writable"
                | "seekable"
                | "close"
                | "readline"
                | "readlines"
                | "getvalue"
                | "getbuffer"
                | "read"
                | "seek"
                | "tell"
                | "truncate"
                | "fileno"
                | "__enter__"
                | "__exit__"
        ) {
            continue;
        }
        let Some(method_value) = shaped.get_property_interned(&method_name) else {
            continue;
        };
        let method_ptr = method_value
            .as_object_ptr()
            .expect("stream helper methods should be builtin functions");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        shaped.set_property(
            method_name,
            crate::alloc_managed_value(builtin.bind(value)),
            shape_registry(),
        );
    }
    value
}

fn load_string_io(receiver: Value) -> Result<StringIO, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    let buffer = string_from_value(
        shaped
            .get_property(STRING_BUFFER_ATTR)
            .ok_or_else(|| BuiltinError::AttributeError("missing StringIO buffer".to_string()))?,
        STRING_BUFFER_ATTR,
    )?;
    let position = usize_value(
        shaped
            .get_property(POSITION_ATTR)
            .ok_or_else(|| BuiltinError::AttributeError("missing StringIO position".to_string()))?,
        POSITION_ATTR,
    )?;
    let closed = bool_value(
        shaped.get_property(CLOSED_ATTR).ok_or_else(|| {
            BuiltinError::AttributeError("missing StringIO closed state".to_string())
        })?,
        CLOSED_ATTR,
    )?;
    Ok(StringIO::from_parts(buffer, position, closed))
}

fn store_string_io(receiver: Value, stream: StringIO) -> Result<(), BuiltinError> {
    let (buffer, position, closed) = stream.into_parts();
    let shaped = shaped_object_mut(receiver)?;
    shaped.set_property(
        intern(STRING_BUFFER_ATTR),
        string_value(&buffer),
        shape_registry(),
    );
    shaped.set_property(
        intern(POSITION_ATTR),
        Value::int(position as i64).unwrap(),
        shape_registry(),
    );
    shaped.set_property(intern(CLOSED_ATTR), Value::bool(closed), shape_registry());
    Ok(())
}

fn load_bytes_io(receiver: Value) -> Result<BytesIO, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    let buffer = bytes_from_value(
        shaped
            .get_property(BYTES_BUFFER_ATTR)
            .ok_or_else(|| BuiltinError::AttributeError("missing BytesIO buffer".to_string()))?,
        BYTES_BUFFER_ATTR,
    )?;
    let position = usize_value(
        shaped
            .get_property(POSITION_ATTR)
            .ok_or_else(|| BuiltinError::AttributeError("missing BytesIO position".to_string()))?,
        POSITION_ATTR,
    )?;
    let closed = bool_value(
        shaped.get_property(CLOSED_ATTR).ok_or_else(|| {
            BuiltinError::AttributeError("missing BytesIO closed state".to_string())
        })?,
        CLOSED_ATTR,
    )?;
    Ok(BytesIO::from_parts(buffer, position, closed))
}

fn store_bytes_io(receiver: Value, stream: BytesIO) -> Result<(), BuiltinError> {
    let (buffer, position, closed) = stream.into_parts();
    let shaped = shaped_object_mut(receiver)?;
    shaped.set_property(
        intern(BYTES_BUFFER_ATTR),
        bytes_value(&buffer),
        shape_registry(),
    );
    shaped.set_property(
        intern(POSITION_ATTR),
        Value::int(position as i64).unwrap(),
        shape_registry(),
    );
    shaped.set_property(intern(CLOSED_ATTR), Value::bool(closed), shape_registry());
    Ok(())
}

fn parse_file_mode(mode: &str) -> Result<HostFileMode, BuiltinError> {
    HostFileMode::parse(mode).map_err(|err| BuiltinError::ValueError(err.to_string()))
}

fn validate_encoding(name: &str) -> Result<(), BuiltinError> {
    match name.to_ascii_lowercase().as_str() {
        "utf-8" | "utf8" | "locale" | "ascii" | "latin-1" | "latin1" | "iso-8859-1"
        | "iso8859-1" => Ok(()),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown text encoding '{}'",
            name
        ))),
    }
}

fn validate_text_error_policy(name: &str) -> Result<(), BuiltinError> {
    match name {
        "strict" | "ignore" | "replace" | "surrogateescape" | "backslashreplace" => Ok(()),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown error handler name '{}'",
            name
        ))),
    }
}

fn validate_newline_policy(newline: &str) -> Result<(), BuiltinError> {
    match newline {
        "" | "\n" | "\r" | "\r\n" => Ok(()),
        _ => Err(BuiltinError::ValueError(
            "illegal newline value".to_string(),
        )),
    }
}

fn file_mode(receiver: Value) -> Result<HostFileMode, BuiltinError> {
    parse_file_mode(&file_mode_string(receiver)?)
}

fn file_mode_string(receiver: Value) -> Result<String, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    let value = shaped
        .get_property(FILE_MODE_ATTR)
        .ok_or_else(|| BuiltinError::AttributeError("missing file mode".to_string()))?;
    string_from_value(value, FILE_MODE_ATTR)
}

fn file_path(receiver: Value) -> Result<String, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    let value = shaped
        .get_property(FILE_PATH_ATTR)
        .ok_or_else(|| BuiltinError::AttributeError("missing file path".to_string()))?;
    string_from_value(value, FILE_PATH_ATTR)
}

fn file_descriptor(receiver: Value) -> Result<Option<i32>, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    let Some(value) = shaped.get_property(FILE_FD_ATTR) else {
        return Ok(None);
    };
    let fd = int_like_value(value, FILE_FD_ATTR)?;
    i32::try_from(fd)
        .map(Some)
        .map_err(|_| BuiltinError::OverflowError("stored file descriptor is out of range".into()))
}

fn file_closefd(receiver: Value) -> Result<bool, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    match shaped.get_property(CLOSEFD_ATTR) {
        Some(value) => bool_value(value, CLOSEFD_ATTR),
        None => Ok(false),
    }
}

fn file_encoding(receiver: Value) -> Result<String, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    let value = shaped
        .get_property(ENCODING_ATTR)
        .ok_or_else(|| BuiltinError::AttributeError("missing text encoding".to_string()))?;
    string_from_value(value, ENCODING_ATTR)
}

fn stream_position(receiver: Value) -> Result<u64, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    let value = shaped
        .get_property(POSITION_ATTR)
        .ok_or_else(|| BuiltinError::AttributeError("missing stream position".to_string()))?;
    Ok(usize_value(value, POSITION_ATTR)? as u64)
}

fn set_stream_position(receiver: Value, position: u64) -> Result<(), BuiltinError> {
    let shaped = shaped_object_mut(receiver)?;
    shaped.set_property(
        intern(POSITION_ATTR),
        Value::int(position as i64).unwrap(),
        shape_registry(),
    );
    Ok(())
}

fn ensure_open(receiver: Value) -> Result<(), BuiltinError> {
    if is_closed(receiver)? {
        return Err(BuiltinError::ValueError(
            "I/O operation on closed file".to_string(),
        ));
    }
    Ok(())
}

fn reopen_file(receiver: Value) -> Result<std::fs::File, BuiltinError> {
    ensure_open(receiver)?;

    let mode = file_mode(receiver)?;
    let path = file_path(receiver)?;
    let mut options = OpenOptions::new();
    options
        .read(mode.read)
        .write(mode.write)
        .append(mode.append);
    options.open(path).map_err(host_stream_error)
}

fn seek_file(receiver: Value, offset: i64, whence: u32) -> Result<u64, BuiltinError> {
    ensure_open(receiver)?;

    if let Some(fd) = file_descriptor(receiver)? {
        let position = fd_seek(fd, offset, whence).map_err(host_stream_error)?;
        set_stream_position(receiver, position)?;
        return Ok(position);
    }

    let current = stream_position(receiver)? as i64;
    let end = std::fs::metadata(file_path(receiver)?)
        .map_err(host_stream_error)?
        .len() as i64;
    let target = match whence {
        0 => offset,
        1 => current.saturating_add(offset),
        2 => end.saturating_add(offset),
        other => {
            return Err(BuiltinError::ValueError(format!(
                "invalid whence ({other}, should be 0, 1 or 2)"
            )));
        }
    };
    if target < 0 {
        return Err(BuiltinError::ValueError(
            "negative seek position".to_string(),
        ));
    }

    let target = target as u64;
    set_stream_position(receiver, target)?;
    Ok(target)
}

fn truncate_file(receiver: Value, size: Option<usize>) -> Result<usize, BuiltinError> {
    ensure_open(receiver)?;

    let mode = file_mode(receiver)?;
    if !mode.write {
        return Err(BuiltinError::OSError("stream is not writable".to_string()));
    }

    let target = size.unwrap_or(stream_position(receiver)? as usize);
    if let Some(fd) = file_descriptor(receiver)? {
        fd_truncate(fd, target as u64).map_err(host_stream_error)?;
        if stream_position(receiver)? > target as u64 {
            set_stream_position(receiver, target as u64)?;
        }
        return Ok(target);
    }

    let mut file = reopen_file(receiver)?;
    file.set_len(target as u64).map_err(host_stream_error)?;
    if stream_position(receiver)? > target as u64 {
        set_stream_position(receiver, target as u64)?;
    }
    Ok(target)
}

fn read_text_file(receiver: Value, count: Option<usize>) -> Result<String, BuiltinError> {
    let bytes = read_file_bytes(receiver, count)?;
    decode_text_bytes(&bytes, &file_encoding(receiver)?)
}

fn read_binary_file(receiver: Value, count: Option<usize>) -> Result<Vec<u8>, BuiltinError> {
    read_file_bytes(receiver, count)
}

fn read_file_bytes(receiver: Value, count: Option<usize>) -> Result<Vec<u8>, BuiltinError> {
    ensure_open(receiver)?;

    let mode = file_mode(receiver)?;
    if !mode.read {
        return Err(BuiltinError::OSError("stream is not readable".to_string()));
    }

    if let Some(fd) = file_descriptor(receiver)? {
        return read_fd_bytes(receiver, fd, count);
    }

    let position = stream_position(receiver)?;
    let mut file = reopen_file(receiver)?;
    file.seek(SeekFrom::Start(position))
        .map_err(host_stream_error)?;

    let mut buffer = Vec::new();
    match count {
        Some(limit) => {
            buffer.resize(limit, 0);
            let read = file.read(&mut buffer).map_err(host_stream_error)?;
            buffer.truncate(read);
        }
        None => {
            file.read_to_end(&mut buffer).map_err(host_stream_error)?;
        }
    }

    let position = file.stream_position().map_err(host_stream_error)?;
    set_stream_position(receiver, position)?;
    Ok(buffer)
}

fn read_text_file_line(receiver: Value) -> Result<String, BuiltinError> {
    ensure_open(receiver)?;

    let mode = file_mode(receiver)?;
    if !mode.read {
        return Err(BuiltinError::OSError("stream is not readable".to_string()));
    }

    if let Some(fd) = file_descriptor(receiver)? {
        let line = read_fd_line_bytes(receiver, fd)?;
        return decode_text_bytes(&line, &file_encoding(receiver)?);
    }

    let position = stream_position(receiver)?;
    let file = reopen_file(receiver)?;
    let mut reader = BufReader::new(file);
    reader
        .seek(SeekFrom::Start(position))
        .map_err(host_stream_error)?;

    let mut line = Vec::new();
    reader
        .read_until(b'\n', &mut line)
        .map_err(host_stream_error)?;
    let position = reader.stream_position().map_err(host_stream_error)?;
    set_stream_position(receiver, position)?;
    decode_text_bytes(&line, &file_encoding(receiver)?)
}

fn read_binary_file_line(receiver: Value) -> Result<Vec<u8>, BuiltinError> {
    ensure_open(receiver)?;

    let mode = file_mode(receiver)?;
    if !mode.read {
        return Err(BuiltinError::OSError("stream is not readable".to_string()));
    }

    if let Some(fd) = file_descriptor(receiver)? {
        return read_fd_line_bytes(receiver, fd);
    }

    let position = stream_position(receiver)?;
    let file = reopen_file(receiver)?;
    let mut reader = BufReader::new(file);
    reader
        .seek(SeekFrom::Start(position))
        .map_err(host_stream_error)?;

    let mut line = Vec::new();
    reader
        .read_until(b'\n', &mut line)
        .map_err(host_stream_error)?;
    let position = reader.stream_position().map_err(host_stream_error)?;
    set_stream_position(receiver, position)?;
    Ok(line)
}

fn read_text_file_lines(receiver: Value, hint: Option<usize>) -> Result<Vec<String>, BuiltinError> {
    ensure_open(receiver)?;

    let mode = file_mode(receiver)?;
    if !mode.read {
        return Err(BuiltinError::OSError("stream is not readable".to_string()));
    }

    if let Some(fd) = file_descriptor(receiver)? {
        let encoding = file_encoding(receiver)?;
        let binary_lines = read_fd_lines(receiver, fd, hint)?;
        return binary_lines
            .iter()
            .map(|line| decode_text_bytes(line, &encoding))
            .collect();
    }

    let position = stream_position(receiver)?;
    let file = reopen_file(receiver)?;
    let mut reader = BufReader::new(file);
    reader
        .seek(SeekFrom::Start(position))
        .map_err(host_stream_error)?;

    let encoding = file_encoding(receiver)?;
    let mut lines = Vec::new();
    let mut total = 0usize;
    loop {
        let mut line = Vec::new();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .map_err(host_stream_error)?;
        if bytes_read == 0 {
            break;
        }
        total = total.saturating_add(line.len());
        lines.push(decode_text_bytes(&line, &encoding)?);
        if let Some(limit) = hint {
            if total >= limit {
                break;
            }
        }
    }

    let position = reader.stream_position().map_err(host_stream_error)?;
    set_stream_position(receiver, position)?;
    Ok(lines)
}

fn read_binary_file_lines(
    receiver: Value,
    hint: Option<usize>,
) -> Result<Vec<Vec<u8>>, BuiltinError> {
    ensure_open(receiver)?;

    let mode = file_mode(receiver)?;
    if !mode.read {
        return Err(BuiltinError::OSError("stream is not readable".to_string()));
    }

    if let Some(fd) = file_descriptor(receiver)? {
        return read_fd_lines(receiver, fd, hint);
    }

    let position = stream_position(receiver)?;
    let file = reopen_file(receiver)?;
    let mut reader = BufReader::new(file);
    reader
        .seek(SeekFrom::Start(position))
        .map_err(host_stream_error)?;

    let mut lines = Vec::new();
    let mut total = 0usize;
    loop {
        let mut line = Vec::new();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .map_err(host_stream_error)?;
        if bytes_read == 0 {
            break;
        }
        total = total.saturating_add(line.len());
        lines.push(line);
        if let Some(limit) = hint {
            if total >= limit {
                break;
            }
        }
    }

    let position = reader.stream_position().map_err(host_stream_error)?;
    set_stream_position(receiver, position)?;
    Ok(lines)
}

fn write_text_file(receiver: Value, data: &str) -> Result<usize, BuiltinError> {
    let encoding = file_encoding(receiver)?;
    let bytes = encode_text_bytes(data, &encoding)?;
    write_file_bytes(receiver, &bytes)?;
    Ok(data.chars().count())
}

fn write_binary_file(receiver: Value, data: &[u8]) -> Result<usize, BuiltinError> {
    write_file_bytes(receiver, data)?;
    Ok(data.len())
}

fn write_file_bytes(receiver: Value, data: &[u8]) -> Result<(), BuiltinError> {
    ensure_open(receiver)?;

    let mode = file_mode(receiver)?;
    if !mode.write {
        return Err(BuiltinError::OSError("stream is not writable".to_string()));
    }

    if let Some(fd) = file_descriptor(receiver)? {
        write_fd_bytes(receiver, fd, data)?;
        return Ok(());
    }

    let mut file = reopen_file(receiver)?;
    if mode.append {
        file.seek(SeekFrom::End(0)).map_err(host_stream_error)?;
    } else {
        file.seek(SeekFrom::Start(stream_position(receiver)?))
            .map_err(host_stream_error)?;
    }
    file.write_all(data).map_err(host_stream_error)?;
    let position = if mode.append {
        file.metadata().map_err(host_stream_error)?.len()
    } else {
        file.stream_position().map_err(host_stream_error)?
    };
    set_stream_position(receiver, position)?;
    Ok(())
}

fn read_fd_bytes(receiver: Value, fd: i32, count: Option<usize>) -> Result<Vec<u8>, BuiltinError> {
    let mut output = Vec::new();
    match count {
        Some(limit) => {
            output.resize(limit, 0);
            let read = read_fd(fd, &mut output).map_err(host_stream_error)?;
            output.truncate(read);
        }
        None => {
            let mut chunk = [0u8; DEFAULT_BUFFER_SIZE];
            loop {
                let read = read_fd(fd, &mut chunk).map_err(host_stream_error)?;
                if read == 0 {
                    break;
                }
                output.extend_from_slice(&chunk[..read]);
            }
        }
    }

    let position = stream_position(receiver)?.saturating_add(output.len() as u64);
    set_stream_position(receiver, position)?;
    Ok(output)
}

fn read_fd_line_bytes(receiver: Value, fd: i32) -> Result<Vec<u8>, BuiltinError> {
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let read = read_fd(fd, &mut byte).map_err(host_stream_error)?;
        if read == 0 {
            break;
        }
        line.push(byte[0]);
        if byte[0] == b'\n' {
            break;
        }
    }

    let position = stream_position(receiver)?.saturating_add(line.len() as u64);
    set_stream_position(receiver, position)?;
    Ok(line)
}

fn read_fd_lines(
    receiver: Value,
    fd: i32,
    hint: Option<usize>,
) -> Result<Vec<Vec<u8>>, BuiltinError> {
    let mut lines = Vec::new();
    let mut total = 0usize;
    loop {
        let line = read_fd_line_bytes(receiver, fd)?;
        if line.is_empty() {
            break;
        }
        total = total.saturating_add(line.len());
        lines.push(line);
        if let Some(limit) = hint {
            if total >= limit {
                break;
            }
        }
    }
    Ok(lines)
}

fn write_fd_bytes(receiver: Value, fd: i32, data: &[u8]) -> Result<(), BuiltinError> {
    let mut written_total = 0usize;
    while written_total < data.len() {
        let written = write_fd(fd, &data[written_total..]).map_err(host_stream_error)?;
        if written == 0 {
            return Err(BuiltinError::OSError(
                "write() returned zero bytes".to_string(),
            ));
        }
        written_total = written_total.saturating_add(written);
    }
    let position = stream_position(receiver)?.saturating_add(written_total as u64);
    set_stream_position(receiver, position)?;
    Ok(())
}

fn decode_text_bytes(bytes: &[u8], encoding: &str) -> Result<String, BuiltinError> {
    match encoding.to_ascii_lowercase().as_str() {
        "utf-8" | "utf8" | "locale" => String::from_utf8(bytes.to_vec())
            .map_err(|err| BuiltinError::ValueError(err.to_string())),
        "ascii" => {
            if let Some(index) = bytes.iter().position(|byte| *byte > 0x7f) {
                return Err(BuiltinError::ValueError(format!(
                    "ascii decode error at byte {}",
                    index
                )));
            }
            Ok(bytes.iter().map(|byte| char::from(*byte)).collect())
        }
        "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" => {
            Ok(bytes.iter().map(|byte| char::from(*byte)).collect())
        }
        _ => Err(BuiltinError::ValueError(format!(
            "unknown text encoding '{}'",
            encoding
        ))),
    }
}

fn encode_text_bytes(data: &str, encoding: &str) -> Result<Vec<u8>, BuiltinError> {
    match encoding.to_ascii_lowercase().as_str() {
        "utf-8" | "utf8" | "locale" => Ok(data.as_bytes().to_vec()),
        "ascii" => data
            .chars()
            .map(|ch| {
                u8::try_from(ch as u32).map_err(|_| {
                    BuiltinError::ValueError("character cannot be encoded in ASCII".to_string())
                })
            })
            .collect(),
        "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" => data
            .chars()
            .map(|ch| {
                u8::try_from(ch as u32).map_err(|_| {
                    BuiltinError::ValueError("character cannot be encoded in latin-1".to_string())
                })
            })
            .collect(),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown text encoding '{}'",
            encoding
        ))),
    }
}

fn expect_no_method_args(args: &[Value], name: &str) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{name}() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(args[0])
}

fn int_like_value(value: Value, context: &str) -> Result<i64, BuiltinError> {
    if let Some(int) = value.as_int() {
        return Ok(int);
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(i64::from(boolean));
    }
    Err(BuiltinError::TypeError(format!(
        "{context} must be an integer, not {}",
        value.type_name()
    )))
}

fn optional_count(value: Option<Value>, name: &str) -> Result<Option<usize>, BuiltinError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let count = int_like_value(value, name)?;
    if count < 0 {
        return Ok(None);
    }
    Ok(Some(count as usize))
}

fn optional_size(value: Option<Value>, name: &str) -> Result<Option<usize>, BuiltinError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let size = int_like_value(value, name)?;
    if size < 0 {
        return Err(BuiltinError::ValueError(format!(
            "{name} size cannot be negative"
        )));
    }
    Ok(Some(size as usize))
}

struct TextIoWrapperInitArgs {
    buffer: Value,
    encoding: Option<String>,
}

fn parse_text_io_wrapper_init_args(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<TextIoWrapperInitArgs, BuiltinError> {
    if positional.len() > 6 {
        return Err(BuiltinError::TypeError(format!(
            "TextIOWrapper() takes at most 6 arguments ({} given)",
            positional.len()
        )));
    }

    let mut buffer = positional.first().copied();
    let mut buffer_seen = buffer.is_some();
    let mut encoding = positional
        .get(1)
        .copied()
        .map(|value| parse_optional_encoding(value, "encoding"))
        .transpose()?
        .flatten();
    let mut encoding_seen = positional.get(1).is_some();
    let mut errors_seen = positional.get(2).is_some();
    let mut newline_seen = positional.get(3).is_some();
    let mut line_buffering_seen = positional.get(4).is_some();
    let mut write_through_seen = positional.get(5).is_some();

    if let Some(&errors) = positional.get(2) {
        validate_optional_text_option(errors, "errors")?;
    }
    if let Some(&newline) = positional.get(3) {
        validate_string_io_newline(newline)?;
    }
    if let Some(&line_buffering) = positional.get(4) {
        let _ = bool_value(line_buffering, "line_buffering")?;
    }
    if let Some(&write_through) = positional.get(5) {
        let _ = bool_value(write_through, "write_through")?;
    }

    for &(name, value) in keywords {
        match name {
            "buffer" => {
                if buffer_seen {
                    return Err(BuiltinError::TypeError(
                        "TextIOWrapper() got multiple values for argument 'buffer'".to_string(),
                    ));
                }
                buffer = Some(value);
                buffer_seen = true;
            }
            "encoding" => {
                if encoding_seen {
                    return Err(BuiltinError::TypeError(
                        "TextIOWrapper() got multiple values for argument 'encoding'".to_string(),
                    ));
                }
                encoding = parse_optional_encoding(value, "encoding")?;
                encoding_seen = true;
            }
            "errors" => {
                if errors_seen {
                    return Err(BuiltinError::TypeError(
                        "TextIOWrapper() got multiple values for argument 'errors'".to_string(),
                    ));
                }
                validate_optional_text_option(value, "errors")?;
                errors_seen = true;
            }
            "newline" => {
                if newline_seen {
                    return Err(BuiltinError::TypeError(
                        "TextIOWrapper() got multiple values for argument 'newline'".to_string(),
                    ));
                }
                validate_string_io_newline(value)?;
                newline_seen = true;
            }
            "line_buffering" => {
                if line_buffering_seen {
                    return Err(BuiltinError::TypeError(
                        "TextIOWrapper() got multiple values for argument 'line_buffering'"
                            .to_string(),
                    ));
                }
                let _ = bool_value(value, "line_buffering")?;
                line_buffering_seen = true;
            }
            "write_through" => {
                if write_through_seen {
                    return Err(BuiltinError::TypeError(
                        "TextIOWrapper() got multiple values for argument 'write_through'"
                            .to_string(),
                    ));
                }
                let _ = bool_value(value, "write_through")?;
                write_through_seen = true;
            }
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "TextIOWrapper() got an unexpected keyword argument '{name}'"
                )));
            }
        }
    }

    let buffer = buffer.ok_or_else(|| {
        BuiltinError::TypeError("TextIOWrapper() missing required argument 'buffer'".to_string())
    })?;

    Ok(TextIoWrapperInitArgs { buffer, encoding })
}

fn parse_string_io_init_args(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<String, BuiltinError> {
    if positional.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "StringIO() takes at most 2 arguments ({} given)",
            positional.len()
        )));
    }

    let mut initial = positional
        .first()
        .copied()
        .map(string_io_initial_value)
        .transpose()?
        .unwrap_or_default();
    let mut initial_seen = !positional.is_empty();
    let mut newline_seen = positional.get(1).is_some();

    if let Some(&newline) = positional.get(1) {
        validate_string_io_newline(newline)?;
    }

    for &(name, value) in keywords {
        match name {
            "initial_value" => {
                if initial_seen {
                    return Err(BuiltinError::TypeError(
                        "StringIO() got multiple values for argument 'initial_value'".to_string(),
                    ));
                }
                initial = string_io_initial_value(value)?;
                initial_seen = true;
            }
            "newline" => {
                if newline_seen {
                    return Err(BuiltinError::TypeError(
                        "StringIO() got multiple values for argument 'newline'".to_string(),
                    ));
                }
                validate_string_io_newline(value)?;
                newline_seen = true;
            }
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "StringIO() got an unexpected keyword argument '{name}'"
                )));
            }
        }
    }

    Ok(initial)
}

fn string_io_initial_value(value: Value) -> Result<String, BuiltinError> {
    if value.is_none() {
        Ok(String::new())
    } else {
        string_from_value(value, "initial_value")
    }
}

fn validate_string_io_newline(value: Value) -> Result<(), BuiltinError> {
    if value.is_none() {
        return Ok(());
    }

    let newline = string_from_value(value, "newline")?;
    if matches!(newline.as_str(), "" | "\n" | "\r" | "\r\n") {
        Ok(())
    } else {
        Err(BuiltinError::ValueError(
            "illegal newline value".to_string(),
        ))
    }
}

fn parse_optional_encoding(value: Value, context: &str) -> Result<Option<String>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }
    let encoding = string_from_value(value, context)?;
    validate_encoding(&encoding)?;
    Ok(Some(encoding))
}

fn validate_optional_text_option(value: Value, context: &str) -> Result<(), BuiltinError> {
    if value.is_none() {
        return Ok(());
    }
    let _ = string_from_value(value, context)?;
    Ok(())
}

fn parse_bytes_io_init_args(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Vec<u8>, BuiltinError> {
    if positional.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "BytesIO() takes at most 1 argument ({} given)",
            positional.len()
        )));
    }

    let mut initial = positional
        .first()
        .copied()
        .map(bytes_io_initial_value)
        .transpose()?
        .unwrap_or_default();
    let mut initial_seen = !positional.is_empty();

    for &(name, value) in keywords {
        match name {
            "initial_bytes" => {
                if initial_seen {
                    return Err(BuiltinError::TypeError(
                        "BytesIO() got multiple values for argument 'initial_bytes'".to_string(),
                    ));
                }
                initial = bytes_io_initial_value(value)?;
                initial_seen = true;
            }
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "BytesIO() got an unexpected keyword argument '{name}'"
                )));
            }
        }
    }

    Ok(initial)
}

fn bytes_io_initial_value(value: Value) -> Result<Vec<u8>, BuiltinError> {
    if value.is_none() {
        Ok(Vec::new())
    } else {
        bytes_from_value(value, "initial_bytes")
    }
}

fn heap_type_from_value(
    value: Value,
    context: &str,
) -> Result<&'static PyClassObject, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context}() argument 1 must be a type")))?;

    if extract_type_id(ptr) != TypeId::TYPE || builtin_type_object_type_id(ptr).is_some() {
        return Err(BuiltinError::TypeError(format!(
            "{context}() argument 1 must be a type"
        )));
    }

    Ok(unsafe { &*(ptr as *const PyClassObject) })
}

fn string_from_value(value: Value, context: &str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|value| value.as_str().to_string())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!("{context} must be str, not {}", value.type_name()))
        })
}

fn optional_string(value: Option<Value>, context: &str) -> Result<Option<String>, BuiltinError> {
    match value {
        Some(value) if !value.is_none() => string_from_value(value, context).map(Some),
        _ => Ok(None),
    }
}

fn string_value_to_owned(value: Value) -> Option<String> {
    value_as_string_ref(value).map(|value| value.as_str().to_string())
}

fn truthy_closefd(value: Value) -> Result<bool, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(boolean);
    }
    int_like_value(value, "open() closefd").map(|value| value != 0)
}

fn bytes_from_value(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        ))
    })?;
    let type_id = crate::ops::objects::extract_type_id(ptr);
    if type_id != prism_runtime::object::type_obj::TypeId::BYTES
        && type_id != prism_runtime::object::type_obj::TypeId::BYTEARRAY
    {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        )));
    }
    Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec())
}

fn bool_value(value: Value, context: &str) -> Result<bool, BuiltinError> {
    value.as_bool().ok_or_else(|| {
        BuiltinError::TypeError(format!("{context} must be bool, not {}", value.type_name()))
    })
}

fn usize_value(value: Value, context: &str) -> Result<usize, BuiltinError> {
    let value = int_like_value(value, context)?;
    if value < 0 {
        return Err(BuiltinError::ValueError(format!(
            "{context} cannot be negative"
        )));
    }
    Ok(value as usize)
}

fn string_value(value: &str) -> Value {
    if value.is_empty() {
        Value::string(intern(""))
    } else {
        crate::alloc_managed_value(StringObject::new(value))
    }
}

fn bytes_value(value: &[u8]) -> Value {
    crate::alloc_managed_value(BytesObject::from_slice(value))
}

fn list_value(items: Vec<Value>) -> Value {
    crate::alloc_managed_value(ListObject::from_iter(items))
}

fn is_closed(receiver: Value) -> Result<bool, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    bool_value(
        shaped
            .get_property(CLOSED_ATTR)
            .ok_or_else(|| BuiltinError::AttributeError("missing closed attribute".to_string()))?,
        CLOSED_ATTR,
    )
}

fn set_closed(receiver: Value, closed: bool) -> Result<(), BuiltinError> {
    let shaped = shaped_object_mut(receiver)?;
    shaped.set_property(intern(CLOSED_ATTR), Value::bool(closed), shape_registry());
    Ok(())
}

fn shaped_object_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "expected object-backed stream receiver, got {}",
            value.type_name()
        ))
    })?;
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn shaped_object_mut(value: Value) -> Result<&'static mut ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "expected object-backed stream receiver, got {}",
            value.type_name()
        ))
    })?;
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn class_value(class: &'static Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

fn bound_builtin_value(function: &'static BuiltinFunctionObject, receiver: Value) -> Value {
    let bound = Box::new(function.bind(receiver));
    Value::object_ptr(Box::into_raw(bound) as *const ())
}

fn map_io_error(error: IoError) -> BuiltinError {
    match error {
        IoError::ValueError(message) => BuiltinError::ValueError(message),
        IoError::UnsupportedOperation(message) => BuiltinError::OSError(message),
        IoError::OsError(message) => BuiltinError::OSError(message),
    }
}

fn read_fd(fd: i32, buffer: &mut [u8]) -> Result<usize, std::io::Error> {
    if buffer.is_empty() {
        return Ok(0);
    }

    #[cfg(windows)]
    {
        let count = buffer.len().min(i32::MAX as usize) as u32;
        let read = unsafe { crt_read(fd, buffer.as_mut_ptr().cast::<c_void>(), count) };
        if read < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(read as usize);
    }

    #[cfg(unix)]
    {
        let count = buffer.len().min(isize::MAX as usize);
        let read = unsafe { libc::read(fd, buffer.as_mut_ptr().cast::<c_void>(), count) };
        if read < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(read as usize);
    }

    #[cfg(not(any(windows, unix)))]
    {
        let _ = (fd, buffer);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "file descriptor I/O is not supported on this platform",
        ))
    }
}

fn write_fd(fd: i32, buffer: &[u8]) -> Result<usize, std::io::Error> {
    if buffer.is_empty() {
        return Ok(0);
    }

    #[cfg(windows)]
    {
        let count = buffer.len().min(i32::MAX as usize) as u32;
        let written = unsafe { crt_write(fd, buffer.as_ptr().cast::<c_void>(), count) };
        if written < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(written as usize);
    }

    #[cfg(unix)]
    {
        let count = buffer.len().min(isize::MAX as usize);
        let written = unsafe { libc::write(fd, buffer.as_ptr().cast::<c_void>(), count) };
        if written < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(written as usize);
    }

    #[cfg(not(any(windows, unix)))]
    {
        let _ = (fd, buffer);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "file descriptor I/O is not supported on this platform",
        ))
    }
}

fn close_fd(fd: i32) -> Result<(), std::io::Error> {
    #[cfg(windows)]
    {
        if unsafe { crt_close(fd) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(());
    }

    #[cfg(unix)]
    {
        if unsafe { libc::close(fd) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(());
    }

    #[cfg(not(any(windows, unix)))]
    {
        let _ = fd;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "file descriptor I/O is not supported on this platform",
        ))
    }
}

fn fd_seek(fd: i32, offset: i64, whence: u32) -> Result<u64, std::io::Error> {
    let origin = match whence {
        0..=2 => whence as i32,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid whence ({whence}, should be 0, 1 or 2)"),
            ));
        }
    };

    #[cfg(windows)]
    {
        let position = unsafe { crt_lseeki64(fd, offset, origin) };
        if position < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(position as u64);
    }

    #[cfg(unix)]
    {
        let position = unsafe { libc::lseek(fd, offset as libc::off_t, origin) };
        if position < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(position as u64);
    }

    #[cfg(not(any(windows, unix)))]
    {
        let _ = (fd, offset, origin);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "file descriptor seeking is not supported on this platform",
        ))
    }
}

fn fd_seek_current(fd: i32) -> Result<u64, std::io::Error> {
    fd_seek(fd, 0, SEEK_CUR)
}

fn fd_truncate(fd: i32, size: u64) -> Result<(), std::io::Error> {
    #[cfg(windows)]
    {
        let size = i64::try_from(size).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "size out of range")
        })?;
        if unsafe { crt_chsize_s(fd, size) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(());
    }

    #[cfg(unix)]
    {
        let size = libc::off_t::try_from(size).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "size out of range")
        })?;
        if unsafe { libc::ftruncate(fd, size) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(());
    }

    #[cfg(not(any(windows, unix)))]
    {
        let _ = (fd, size);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "file descriptor truncation is not supported on this platform",
        ))
    }
}

fn host_stream_error(error: std::io::Error) -> BuiltinError {
    BuiltinError::OSError(error.to_string())
}
