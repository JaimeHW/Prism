use super::{BytesIO, IoError, StringIO};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::stdlib::sys::standard_streams;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::io::mode::FileMode as HostFileMode;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, LazyLock};

const STREAM_KIND_ATTR: &str = "_prism_stream_kind";
const STRING_BUFFER_ATTR: &str = "_prism_string_buffer";
const BYTES_BUFFER_ATTR: &str = "_prism_bytes_buffer";
const FILE_PATH_ATTR: &str = "_prism_file_path";
const FILE_MODE_ATTR: &str = "_prism_file_mode";
const POSITION_ATTR: &str = "_prism_stream_position";
const CLOSED_ATTR: &str = "closed";
const ENCODING_ATTR: &str = "encoding";

static STRING_IO_CONSTRUCTOR: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.StringIO"), string_io_new));
static BYTES_IO_CONSTRUCTOR: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io.BytesIO"), bytes_io_new));
static STREAM_WRITE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.write"), stream_write));
static STREAM_FLUSH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("io._stream.flush"), stream_flush));
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamKind {
    StringIo,
    BytesIo,
    TextFile,
    BinaryFile,
    StdIn,
    StdOut,
    StdErr,
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
            _ => Err(BuiltinError::TypeError(format!(
                "unsupported stream backend '{}'",
                kind
            ))),
        }
    }
}

pub(super) fn string_io_constructor_value() -> Value {
    builtin_value(&STRING_IO_CONSTRUCTOR)
}

pub(super) fn bytes_io_constructor_value() -> Value {
    builtin_value(&BYTES_IO_CONSTRUCTOR)
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

fn string_io_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "StringIO() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let initial = match args.first().copied() {
        None => String::new(),
        Some(value) if value.is_none() => String::new(),
        Some(value) => string_from_value(value, "initial_value")?,
    };

    Ok(new_string_io_object(&initial))
}

fn bytes_io_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "BytesIO() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let initial = match args.first().copied() {
        None => Vec::new(),
        Some(value) if value.is_none() => Vec::new(),
        Some(value) => bytes_from_value(value, "initial_bytes")?,
    };

    Ok(new_bytes_io_object(&initial))
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
        StreamKind::StdIn => Err(BuiltinError::OSError("stdin is not writable".to_string())),
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
        StreamKind::StdOut => standard_streams()
            .flush_stdout()
            .map_err(host_stream_error)?,
        StreamKind::StdErr => standard_streams()
            .flush_stderr()
            .map_err(host_stream_error)?,
        StreamKind::StringIo
        | StreamKind::BytesIo
        | StreamKind::TextFile
        | StreamKind::BinaryFile
        | StreamKind::StdIn => {}
    }

    Ok(Value::none())
}

fn stream_readable(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "readable")?;
    let readable = match StreamKind::from_receiver(receiver)? {
        StreamKind::StringIo | StreamKind::BytesIo | StreamKind::StdIn => true,
        StreamKind::TextFile | StreamKind::BinaryFile => file_mode(receiver)?.read,
        StreamKind::StdOut | StreamKind::StdErr => false,
    };
    Ok(Value::bool(readable && !is_closed(receiver)?))
}

fn stream_writable(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "writable")?;
    let writable = match StreamKind::from_receiver(receiver)? {
        StreamKind::StringIo | StreamKind::BytesIo | StreamKind::StdOut | StreamKind::StdErr => {
            true
        }
        StreamKind::TextFile | StreamKind::BinaryFile => file_mode(receiver)?.write,
        StreamKind::StdIn => false,
    };
    Ok(Value::bool(writable && !is_closed(receiver)?))
}

fn stream_seekable(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "seekable")?;
    let seekable = matches!(
        StreamKind::from_receiver(receiver)?,
        StreamKind::StringIo | StreamKind::BytesIo | StreamKind::TextFile | StreamKind::BinaryFile
    );
    Ok(Value::bool(seekable && !is_closed(receiver)?))
}

fn stream_close(args: &[Value]) -> Result<Value, BuiltinError> {
    let receiver = expect_no_method_args(args, "close")?;
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
        StreamKind::StdOut | StreamKind::StdErr => {
            Err(BuiltinError::OSError("stream is not readable".to_string()))
        }
    }
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

fn new_string_io_object(initial: &str) -> Value {
    let mut object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
    object.set_property(
        intern(STREAM_KIND_ATTR),
        Value::string(intern("stringio")),
        shape_registry(),
    );
    object.set_property(
        intern(STRING_BUFFER_ATTR),
        string_value(initial),
        shape_registry(),
    );
    object.set_property(
        intern(POSITION_ATTR),
        Value::int(0).unwrap(),
        shape_registry(),
    );
    object.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
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
        intern("getvalue"),
        bound_builtin_value(&STRING_IO_GETVALUE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("read"),
        bound_builtin_value(&STRING_IO_READ_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("seek"),
        bound_builtin_value(&STRING_IO_SEEK_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("tell"),
        bound_builtin_value(&STRING_IO_TELL_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("truncate"),
        bound_builtin_value(&STRING_IO_TRUNCATE_METHOD, Value::none()),
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
    finalize_stream_object(object)
}

fn new_bytes_io_object(initial: &[u8]) -> Value {
    let mut object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
    object.set_property(
        intern(STREAM_KIND_ATTR),
        Value::string(intern("bytesio")),
        shape_registry(),
    );
    object.set_property(
        intern(BYTES_BUFFER_ATTR),
        bytes_value(initial),
        shape_registry(),
    );
    object.set_property(
        intern(POSITION_ATTR),
        Value::int(0).unwrap(),
        shape_registry(),
    );
    object.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
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
        intern("getvalue"),
        bound_builtin_value(&BYTES_IO_GETVALUE_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("getbuffer"),
        bound_builtin_value(&BYTES_IO_GETBUFFER_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("read"),
        bound_builtin_value(&BYTES_IO_READ_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("seek"),
        bound_builtin_value(&BYTES_IO_SEEK_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("tell"),
        bound_builtin_value(&BYTES_IO_TELL_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("truncate"),
        bound_builtin_value(&BYTES_IO_TRUNCATE_METHOD, Value::none()),
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
    finalize_stream_object(object)
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
        intern("__enter__"),
        bound_builtin_value(&STREAM_ENTER_METHOD, Value::none()),
        shape_registry(),
    );
    object.set_property(
        intern("__exit__"),
        bound_builtin_value(&STREAM_EXIT_METHOD, Value::none()),
        shape_registry(),
    );
    finalize_stream_object(object)
}

fn new_standard_stream_object(kind: StreamKind) -> Value {
    let mut object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
    object.set_property(
        intern(STREAM_KIND_ATTR),
        Value::string(intern(kind.as_str())),
        shape_registry(),
    );
    object.set_property(intern(CLOSED_ATTR), Value::bool(false), shape_registry());
    object.set_property(
        intern(ENCODING_ATTR),
        Value::string(intern("utf-8")),
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

    match kind {
        StreamKind::StdIn => {
            object.set_property(
                intern("readline"),
                bound_builtin_value(&STREAM_READLINE_METHOD, Value::none()),
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
        StreamKind::StringIo | StreamKind::BytesIo => {}
        StreamKind::TextFile | StreamKind::BinaryFile => {
            unreachable!("file-backed streams must be created via new_file_stream_object")
        }
    }

    finalize_stream_object(object)
}

fn finalize_stream_object(mut object: ShapedObject) -> Value {
    let value = Value::object_ptr(Box::into_raw(Box::new(object)) as *const ());
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
                | "getvalue"
                | "getbuffer"
                | "read"
                | "seek"
                | "tell"
                | "truncate"
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
        let bound = Box::new(builtin.bind(value));
        shaped.set_property(
            method_name,
            Value::object_ptr(Box::into_raw(bound) as *const ()),
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
        "utf-8" | "utf8" | "ascii" | "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" => Ok(()),
        _ => Err(BuiltinError::ValueError(format!(
            "unknown text encoding '{}'",
            name
        ))),
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

fn decode_text_bytes(bytes: &[u8], encoding: &str) -> Result<String, BuiltinError> {
    match encoding.to_ascii_lowercase().as_str() {
        "utf-8" | "utf8" => String::from_utf8(bytes.to_vec())
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
        "utf-8" | "utf8" => Ok(data.as_bytes().to_vec()),
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

fn string_from_value(value: Value, context: &str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|value| value.as_str().to_string())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!("{context} must be str, not {}", value.type_name()))
        })
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
        Value::object_ptr(Box::into_raw(Box::new(StringObject::new(value))) as *const ())
    }
}

fn bytes_value(value: &[u8]) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(BytesObject::from_slice(value))) as *const ())
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

fn host_stream_error(error: std::io::Error) -> BuiltinError {
    BuiltinError::OSError(error.to_string())
}
