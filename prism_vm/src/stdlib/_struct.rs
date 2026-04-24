//! Native `_struct` bootstrap module.
//!
//! CPython's `struct.py` and `pickle.py` expect a compact, C-accelerated
//! compatibility layer underneath the pure-Python API. Prism only needs a
//! focused subset today, but the parser and layout engine are structured so
//! more format codes can be added without rewriting the public surface.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, builtin_iter};
use crate::truthiness::is_truthy;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const MODULE_DOC: &str = "Native bootstrap implementation of the _struct module.";

static CALCSIZE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_struct.calcsize"), calcsize_builtin));
static PACK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_struct.pack"), pack_builtin));
static PACK_INTO_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_struct.pack_into"), pack_into_builtin));
static UNPACK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_struct.unpack"), unpack_builtin));
static UNPACK_FROM_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_struct.unpack_from"), unpack_from_builtin)
});
static ITER_UNPACK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_struct.iter_unpack"), iter_unpack_builtin)
});
static CLEARCACHE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_struct._clearcache"), clearcache_builtin)
});
static STRUCT_CONSTRUCTOR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_struct.Struct"), struct_constructor));

static STRUCT_PACK_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_struct.Struct.pack"), struct_pack));
static STRUCT_PACK_INTO_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_struct.Struct.pack_into"), struct_pack_into)
});
static STRUCT_UNPACK_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_struct.Struct.unpack"), struct_unpack));
static STRUCT_UNPACK_FROM_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_struct.Struct.unpack_from"), struct_unpack_from)
});
static STRUCT_ITER_UNPACK_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_struct.Struct.iter_unpack"), struct_iter_unpack)
});

/// Native `_struct` module descriptor.
#[derive(Debug, Clone)]
pub struct StructModule {
    attrs: Vec<Arc<str>>,
}

impl StructModule {
    /// Create a new `_struct` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__doc__"),
                Arc::from("_clearcache"),
                Arc::from("Struct"),
                Arc::from("calcsize"),
                Arc::from("error"),
                Arc::from("iter_unpack"),
                Arc::from("pack"),
                Arc::from("pack_into"),
                Arc::from("unpack"),
                Arc::from("unpack_from"),
            ],
        }
    }
}

impl Default for StructModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for StructModule {
    fn name(&self) -> &str {
        "_struct"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "_clearcache" => Ok(builtin_value(&CLEARCACHE_FUNCTION)),
            "Struct" => Ok(builtin_value(&STRUCT_CONSTRUCTOR_FUNCTION)),
            "calcsize" => Ok(builtin_value(&CALCSIZE_FUNCTION)),
            "error" => Ok(value_error_type_value()),
            "iter_unpack" => Ok(builtin_value(&ITER_UNPACK_FUNCTION)),
            "pack" => Ok(builtin_value(&PACK_FUNCTION)),
            "pack_into" => Ok(builtin_value(&PACK_INTO_FUNCTION)),
            "unpack" => Ok(builtin_value(&UNPACK_FUNCTION)),
            "unpack_from" => Ok(builtin_value(&UNPACK_FROM_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_struct' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Native,
    Standard { endian: Endian },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FormatCode {
    Pad,
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Pointer,
    Float32,
    Float64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Field {
    code: FormatCode,
    offset: usize,
    size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FormatSpec {
    layout: Layout,
    fields: Vec<Field>,
    size: usize,
    value_count: usize,
}

impl Layout {
    #[inline]
    fn endianness(self) -> Endian {
        match self {
            Self::Native => native_endian(),
            Self::Standard { endian } => endian,
        }
    }

    #[inline]
    fn uses_native_alignment(self) -> bool {
        matches!(self, Self::Native)
    }
}

impl FormatCode {
    fn from_char(ch: char) -> Option<Self> {
        Some(match ch {
            'x' => Self::Pad,
            '?' => Self::Bool,
            'b' => Self::Int8,
            'B' => Self::UInt8,
            'h' => Self::Int16,
            'H' => Self::UInt16,
            'i' | 'l' => Self::Int32,
            'I' | 'L' => Self::UInt32,
            'q' => Self::Int64,
            'Q' => Self::UInt64,
            'P' => Self::Pointer,
            'f' => Self::Float32,
            'd' => Self::Float64,
            _ => return None,
        })
    }

    #[inline]
    fn size(self, layout: Layout) -> usize {
        match layout {
            Layout::Native => match self {
                Self::Pad => 1,
                Self::Bool => std::mem::size_of::<bool>(),
                Self::Int8 => std::mem::size_of::<i8>(),
                Self::UInt8 => std::mem::size_of::<u8>(),
                Self::Int16 => std::mem::size_of::<i16>(),
                Self::UInt16 => std::mem::size_of::<u16>(),
                Self::Int32 => std::mem::size_of::<i32>(),
                Self::UInt32 => std::mem::size_of::<u32>(),
                Self::Int64 => std::mem::size_of::<i64>(),
                Self::UInt64 => std::mem::size_of::<u64>(),
                Self::Pointer => std::mem::size_of::<usize>(),
                Self::Float32 => std::mem::size_of::<f32>(),
                Self::Float64 => std::mem::size_of::<f64>(),
            },
            Layout::Standard { .. } => match self {
                Self::Pad | Self::Bool | Self::Int8 | Self::UInt8 => 1,
                Self::Int16 | Self::UInt16 => 2,
                Self::Int32 | Self::UInt32 | Self::Float32 => 4,
                Self::Int64 | Self::UInt64 | Self::Float64 => 8,
                Self::Pointer => std::mem::size_of::<usize>(),
            },
        }
    }

    #[inline]
    fn alignment(self, layout: Layout) -> usize {
        if !layout.uses_native_alignment() {
            return 1;
        }

        match self {
            Self::Pad => 1,
            Self::Bool => std::mem::align_of::<bool>(),
            Self::Int8 => std::mem::align_of::<i8>(),
            Self::UInt8 => std::mem::align_of::<u8>(),
            Self::Int16 => std::mem::align_of::<i16>(),
            Self::UInt16 => std::mem::align_of::<u16>(),
            Self::Int32 => std::mem::align_of::<i32>(),
            Self::UInt32 => std::mem::align_of::<u32>(),
            Self::Int64 => std::mem::align_of::<i64>(),
            Self::UInt64 => std::mem::align_of::<u64>(),
            Self::Pointer => std::mem::align_of::<usize>(),
            Self::Float32 => std::mem::align_of::<f32>(),
            Self::Float64 => std::mem::align_of::<f64>(),
        }
    }

    fn pack_into_slice(
        self,
        value: Value,
        layout: Layout,
        out: &mut [u8],
    ) -> Result<(), BuiltinError> {
        match self {
            Self::Pad => {
                out.fill(0);
                Ok(())
            }
            Self::Bool => {
                out[0] = u8::from(is_truthy(value));
                Ok(())
            }
            Self::Int8 => {
                out[0] = integer_argument(value, "pack expected integer")?
                    .to_i8()
                    .ok_or_else(out_of_range_error)?
                    .to_ne_bytes()[0];
                Ok(())
            }
            Self::UInt8 => {
                out[0] = integer_argument(value, "pack expected integer")?
                    .to_u8()
                    .ok_or_else(out_of_range_error)?;
                Ok(())
            }
            Self::Int16 => write_i16(
                out,
                layout.endianness(),
                integer_argument(value, "pack expected integer")?
                    .to_i16()
                    .ok_or_else(out_of_range_error)?,
            ),
            Self::UInt16 => write_u16(
                out,
                layout.endianness(),
                integer_argument(value, "pack expected integer")?
                    .to_u16()
                    .ok_or_else(out_of_range_error)?,
            ),
            Self::Int32 => write_i32(
                out,
                layout.endianness(),
                integer_argument(value, "pack expected integer")?
                    .to_i32()
                    .ok_or_else(out_of_range_error)?,
            ),
            Self::UInt32 => write_u32(
                out,
                layout.endianness(),
                integer_argument(value, "pack expected integer")?
                    .to_u32()
                    .ok_or_else(out_of_range_error)?,
            ),
            Self::Int64 => write_i64(
                out,
                layout.endianness(),
                integer_argument(value, "pack expected integer")?
                    .to_i64()
                    .ok_or_else(out_of_range_error)?,
            ),
            Self::UInt64 => write_u64(
                out,
                layout.endianness(),
                integer_argument(value, "pack expected integer")?
                    .to_u64()
                    .ok_or_else(out_of_range_error)?,
            ),
            Self::Pointer => write_usize(
                out,
                integer_argument(value, "pack expected integer")?
                    .to_usize()
                    .ok_or_else(out_of_range_error)?,
            ),
            Self::Float32 => write_f32(out, layout.endianness(), float_argument(value)? as f32),
            Self::Float64 => write_f64(out, layout.endianness(), float_argument(value)?),
        }
    }

    fn unpack_from_slice(self, layout: Layout, data: &[u8]) -> Result<Value, BuiltinError> {
        Ok(match self {
            Self::Pad => Value::none(),
            Self::Bool => Value::bool(data[0] != 0),
            Self::Int8 => Value::int(i64::from(i8::from_ne_bytes([data[0]]))).unwrap(),
            Self::UInt8 => Value::int(i64::from(data[0])).unwrap(),
            Self::Int16 => Value::int(i64::from(read_i16(data, layout.endianness()))).unwrap(),
            Self::UInt16 => Value::int(i64::from(read_u16(data, layout.endianness()))).unwrap(),
            Self::Int32 => Value::int(i64::from(read_i32(data, layout.endianness()))).unwrap(),
            Self::UInt32 => Value::int(i64::from(read_u32(data, layout.endianness()))).unwrap(),
            Self::Int64 => Value::int(read_i64(data, layout.endianness())).unwrap_or_else(|| {
                prism_runtime::types::int::bigint_to_value(BigInt::from(read_i64(
                    data,
                    layout.endianness(),
                )))
            }),
            Self::UInt64 => {
                let value = read_u64(data, layout.endianness());
                if value <= i64::MAX as u64 {
                    if let Some(inline) = Value::int(value as i64) {
                        inline
                    } else {
                        prism_runtime::types::int::bigint_to_value(BigInt::from(value))
                    }
                } else {
                    prism_runtime::types::int::bigint_to_value(BigInt::from(value))
                }
            }
            Self::Pointer => {
                let value = read_usize(data);
                if value <= i64::MAX as usize {
                    Value::int(value as i64).unwrap_or_else(|| {
                        prism_runtime::types::int::bigint_to_value(BigInt::from(value))
                    })
                } else {
                    prism_runtime::types::int::bigint_to_value(BigInt::from(value))
                }
            }
            Self::Float32 => Value::float(f64::from(read_f32(data, layout.endianness()))),
            Self::Float64 => Value::float(read_f64(data, layout.endianness())),
        })
    }
}

#[inline]
fn native_endian() -> Endian {
    if cfg!(target_endian = "little") {
        Endian::Little
    } else {
        Endian::Big
    }
}

#[inline]
fn align_up(offset: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return offset;
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn bound_builtin_value(function: &'static BuiltinFunctionObject, receiver: Value) -> Value {
    let bound = Box::new(function.bind(receiver));
    Value::object_ptr(Box::into_raw(bound) as *const ())
}

#[inline]
fn leak_object_value<T>(object: T) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(object)) as *const ())
}

#[inline]
fn bytes_value(data: &[u8]) -> Value {
    leak_object_value(BytesObject::from_slice(data))
}

#[inline]
fn tuple_value(items: Vec<Value>) -> Value {
    leak_object_value(TupleObject::from_vec(items))
}

#[inline]
fn value_error_type_value() -> Value {
    Value::object_ptr((&*crate::builtins::VALUE_ERROR) as *const _ as *const ())
}

fn calcsize_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "calcsize() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let spec = parse_format_spec(value_to_rust_string(
        args[0],
        "calcsize() format must be a str",
    )?)?;
    Ok(Value::int(spec.size as i64).expect("struct size should fit in i64"))
}

fn pack_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "pack() missing required argument 'format'".to_string(),
        ));
    }

    let format = value_to_rust_string(args[0], "pack() format must be a str")?;
    let buffer = pack_with_format(&format, &args[1..])?;
    Ok(bytes_value(&buffer))
}

fn unpack_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "unpack() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let format = value_to_rust_string(args[0], "unpack() format must be a str")?;
    unpack_tuple_value(
        &format,
        &value_to_bytes_like(args[1], "unpack() buffer must be bytes-like")?,
        0,
    )
}

fn pack_into_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 3 {
        return Err(BuiltinError::TypeError(
            "pack_into() requires a format, buffer, offset, and values".to_string(),
        ));
    }

    let format = value_to_rust_string(args[0], "pack_into() format must be a str")?;
    let buffer_ptr =
        writable_bytearray_ptr(args[1], "pack_into() requires a writable bytearray buffer")?;
    let offset = offset_argument(
        args[2],
        buffer_len(buffer_ptr),
        "pack_into() offset out of range",
    )?;
    let encoded = pack_with_format(&format, &args[3..])?;
    if offset + encoded.len() > buffer_len(buffer_ptr) {
        return Err(BuiltinError::ValueError(
            "pack_into() requires a buffer of at least the requested size".to_string(),
        ));
    }

    let buffer = unsafe { &mut *(buffer_ptr as *mut BytesObject) };
    for (index, byte) in encoded.into_iter().enumerate() {
        let ok = buffer.set((offset + index) as i64, byte);
        debug_assert!(ok, "validated bytearray offset should always be writable");
    }
    Ok(Value::none())
}

fn unpack_from_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "unpack_from() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let format = value_to_rust_string(args[0], "unpack_from() format must be a str")?;
    let buffer = value_to_bytes_like(args[1], "unpack_from() buffer must be bytes-like")?;
    let offset = if let Some(offset_value) = args.get(2) {
        offset_argument(
            *offset_value,
            buffer.len(),
            "unpack_from() offset out of range",
        )?
    } else {
        0
    };
    unpack_tuple_value(&format, &buffer, offset)
}

fn iter_unpack_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "iter_unpack() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let format = value_to_rust_string(args[0], "iter_unpack() format must be a str")?;
    let spec = parse_format_spec(format.clone())?;
    if spec.size == 0 {
        return Err(BuiltinError::ValueError(
            "iter_unpack() requires a non-empty format".to_string(),
        ));
    }

    let buffer = value_to_bytes_like(args[1], "iter_unpack() buffer must be bytes-like")?;
    if buffer.len() % spec.size != 0 {
        return Err(BuiltinError::ValueError(
            "iter_unpack() buffer size must be a multiple of the format size".to_string(),
        ));
    }

    let mut tuples = Vec::with_capacity(buffer.len() / spec.size);
    for offset in (0..buffer.len()).step_by(spec.size) {
        tuples.push(unpack_from_bytes(&spec, &buffer, offset)?);
    }

    let list_value = leak_object_value(prism_runtime::types::list::ListObject::from_iter(tuples));
    builtin_iter(&[list_value])
}

fn clearcache_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_clearcache() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(Value::none())
}

fn struct_constructor(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "Struct() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let format = value_to_rust_string(args[0], "Struct() format must be a str")?;
    let spec = parse_format_spec(format.clone())?;

    let object = Box::new(ShapedObject::with_empty_shape(
        shape_registry().empty_shape(),
    ));
    let ptr = Box::into_raw(object);
    let receiver = Value::object_ptr(ptr as *const ());
    let shaped = unsafe { &mut *ptr };
    shaped.set_property(
        intern("format"),
        Value::string(intern(&format)),
        shape_registry(),
    );
    shaped.set_property(
        intern("size"),
        Value::int(spec.size as i64).expect("struct size should fit in i64"),
        shape_registry(),
    );
    shaped.set_property(
        intern("pack"),
        bound_builtin_value(&STRUCT_PACK_METHOD, receiver),
        shape_registry(),
    );
    shaped.set_property(
        intern("pack_into"),
        bound_builtin_value(&STRUCT_PACK_INTO_METHOD, receiver),
        shape_registry(),
    );
    shaped.set_property(
        intern("unpack"),
        bound_builtin_value(&STRUCT_UNPACK_METHOD, receiver),
        shape_registry(),
    );
    shaped.set_property(
        intern("unpack_from"),
        bound_builtin_value(&STRUCT_UNPACK_FROM_METHOD, receiver),
        shape_registry(),
    );
    shaped.set_property(
        intern("iter_unpack"),
        bound_builtin_value(&STRUCT_ITER_UNPACK_METHOD, receiver),
        shape_registry(),
    );
    Ok(receiver)
}

fn struct_pack(args: &[Value]) -> Result<Value, BuiltinError> {
    let (format, values) = split_struct_receiver_args(args, "pack")?;
    let buffer = pack_with_format(&format, values)?;
    Ok(bytes_value(&buffer))
}

fn struct_pack_into(args: &[Value]) -> Result<Value, BuiltinError> {
    let (format, values) = split_struct_receiver_args(args, "pack_into")?;
    let mut forwarded = Vec::with_capacity(values.len() + 1);
    forwarded.push(Value::string(intern(&format)));
    forwarded.extend_from_slice(values);
    pack_into_builtin(&forwarded)
}

fn struct_unpack(args: &[Value]) -> Result<Value, BuiltinError> {
    let (format, values) = split_struct_receiver_args(args, "unpack")?;
    let mut forwarded = Vec::with_capacity(values.len() + 1);
    forwarded.push(Value::string(intern(&format)));
    forwarded.extend_from_slice(values);
    unpack_builtin(&forwarded)
}

fn struct_unpack_from(args: &[Value]) -> Result<Value, BuiltinError> {
    let (format, values) = split_struct_receiver_args(args, "unpack_from")?;
    let mut forwarded = Vec::with_capacity(values.len() + 1);
    forwarded.push(Value::string(intern(&format)));
    forwarded.extend_from_slice(values);
    unpack_from_builtin(&forwarded)
}

fn struct_iter_unpack(args: &[Value]) -> Result<Value, BuiltinError> {
    let (format, values) = split_struct_receiver_args(args, "iter_unpack")?;
    let mut forwarded = Vec::with_capacity(values.len() + 1);
    forwarded.push(Value::string(intern(&format)));
    forwarded.extend_from_slice(values);
    iter_unpack_builtin(&forwarded)
}

fn split_struct_receiver_args<'a>(
    args: &'a [Value],
    method_name: &str,
) -> Result<(String, &'a [Value]), BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "Struct.{method_name}() missing bound receiver"
        )));
    }
    Ok((struct_receiver_format(args[0])?, &args[1..]))
}

fn struct_receiver_format(receiver: Value) -> Result<String, BuiltinError> {
    let ptr = receiver.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("Struct method receiver must be an object".to_string())
    })?;
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    let format = shaped.get_property("format").ok_or_else(|| {
        BuiltinError::TypeError("Struct receiver is missing its format".to_string())
    })?;
    value_to_rust_string(format, "Struct.format must be a str")
}

fn pack_with_format(format: &str, values: &[Value]) -> Result<Vec<u8>, BuiltinError> {
    let spec = parse_format_spec(format.to_string())?;
    if values.len() != spec.value_count {
        return Err(BuiltinError::TypeError(format!(
            "pack expected {} item(s) for packing (got {})",
            spec.value_count,
            values.len()
        )));
    }

    let mut buffer = vec![0_u8; spec.size];
    let mut value_index = 0;
    for field in &spec.fields {
        if field.code == FormatCode::Pad {
            continue;
        }
        let slice = &mut buffer[field.offset..field.offset + field.size];
        field
            .code
            .pack_into_slice(values[value_index], spec.layout, slice)?;
        value_index += 1;
    }
    Ok(buffer)
}

fn unpack_tuple_value(format: &str, buffer: &[u8], offset: usize) -> Result<Value, BuiltinError> {
    let spec = parse_format_spec(format.to_string())?;
    Ok(unpack_from_bytes(&spec, buffer, offset)?)
}

fn unpack_from_bytes(
    spec: &FormatSpec,
    buffer: &[u8],
    offset: usize,
) -> Result<Value, BuiltinError> {
    let end = offset.checked_add(spec.size).ok_or_else(|| {
        BuiltinError::ValueError("unpack requires a buffer of the requested size".to_string())
    })?;
    if end > buffer.len() {
        return Err(BuiltinError::ValueError(
            "unpack requires a buffer of the requested size".to_string(),
        ));
    }

    let mut items = Vec::with_capacity(spec.value_count);
    for field in &spec.fields {
        if field.code == FormatCode::Pad {
            continue;
        }
        let start = offset + field.offset;
        let end = start + field.size;
        items.push(
            field
                .code
                .unpack_from_slice(spec.layout, &buffer[start..end])?,
        );
    }
    Ok(tuple_value(items))
}

fn parse_format_spec(format: String) -> Result<FormatSpec, BuiltinError> {
    let mut chars = format.chars().peekable();
    let layout = match chars.peek().copied() {
        Some('@') => {
            chars.next();
            Layout::Native
        }
        Some('=') => {
            chars.next();
            Layout::Standard {
                endian: native_endian(),
            }
        }
        Some('<') => {
            chars.next();
            Layout::Standard {
                endian: Endian::Little,
            }
        }
        Some('>') | Some('!') => {
            chars.next();
            Layout::Standard {
                endian: Endian::Big,
            }
        }
        _ => Layout::Native,
    };

    let mut fields = Vec::new();
    let mut offset = 0_usize;
    let mut value_count = 0_usize;
    let mut count_text = String::new();

    while let Some(ch) = chars.next() {
        if ch.is_ascii_whitespace() {
            continue;
        }

        if ch.is_ascii_digit() {
            count_text.push(ch);
            while let Some(next) = chars.peek().copied() {
                if next.is_ascii_digit() {
                    count_text.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            let Some(code_char) = chars.next() else {
                return Err(BuiltinError::ValueError(
                    "repeat count given without format specifier".to_string(),
                ));
            };
            let count = count_text.parse::<usize>().map_err(|_| {
                BuiltinError::ValueError("struct repeat count is too large".to_string())
            })?;
            count_text.clear();
            push_fields(
                &mut fields,
                &mut offset,
                &mut value_count,
                layout,
                code_char,
                count,
            )?;
            continue;
        }

        push_fields(&mut fields, &mut offset, &mut value_count, layout, ch, 1)?;
    }

    Ok(FormatSpec {
        layout,
        fields,
        size: offset,
        value_count,
    })
}

fn push_fields(
    fields: &mut Vec<Field>,
    offset: &mut usize,
    value_count: &mut usize,
    layout: Layout,
    code_char: char,
    repeat: usize,
) -> Result<(), BuiltinError> {
    let code = FormatCode::from_char(code_char).ok_or_else(|| {
        BuiltinError::ValueError(format!("bad char in struct format: '{}'", code_char))
    })?;
    if matches!(code, FormatCode::Pointer) && !matches!(layout, Layout::Native) {
        return Err(BuiltinError::ValueError(format!(
            "bad char in struct format: '{}'",
            code_char
        )));
    }
    let size = code.size(layout);
    let alignment = code.alignment(layout);

    for _ in 0..repeat {
        *offset = align_up(*offset, alignment);
        fields.push(Field {
            code,
            offset: *offset,
            size,
        });
        *offset += size;
        if code != FormatCode::Pad {
            *value_count += 1;
        }
    }
    Ok(())
}

fn value_to_rust_string(value: Value, context: &str) -> Result<String, BuiltinError> {
    if let Some(string) = value_as_string_ref(value) {
        return Ok(string.as_str().to_string());
    }

    Err(BuiltinError::TypeError(format!(
        "{context}, not {}",
        value.type_name()
    )))
}

fn integer_argument(value: Value, context: &str) -> Result<BigInt, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(BigInt::from(if flag { 1_i64 } else { 0_i64 }));
    }

    value_to_bigint(value).ok_or_else(|| BuiltinError::TypeError(context.to_string()))
}

fn float_argument(value: Value) -> Result<f64, BuiltinError> {
    if let Some(float) = value.as_float() {
        return Ok(float);
    }
    if let Some(flag) = value.as_bool() {
        return Ok(if flag { 1.0 } else { 0.0 });
    }
    if let Some(integer) = value_to_bigint(value) {
        return integer
            .to_f64()
            .ok_or_else(|| BuiltinError::OverflowError("float too large to pack".to_string()));
    }
    Err(BuiltinError::TypeError(
        "required argument is not a float".to_string(),
    ))
}

fn value_to_bytes_like(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context}, not {}",
            value.type_name()
        )));
    };

    match extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec()),
        _ => Err(BuiltinError::TypeError(format!(
            "{context}, not {}",
            value.type_name()
        ))),
    }
}

fn writable_bytearray_ptr(value: Value, context: &str) -> Result<*const (), BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(context.to_string()))?;
    if extract_type_id(ptr) != TypeId::BYTEARRAY {
        return Err(BuiltinError::TypeError(context.to_string()));
    }
    Ok(ptr)
}

#[inline]
fn extract_type_id(ptr: *const ()) -> TypeId {
    crate::ops::objects::extract_type_id(ptr)
}

#[inline]
fn buffer_len(ptr: *const ()) -> usize {
    unsafe { &*(ptr as *const BytesObject) }.len()
}

fn offset_argument(value: Value, buffer_len: usize, context: &str) -> Result<usize, BuiltinError> {
    let raw = if let Some(flag) = value.as_bool() {
        if flag { 1_i64 } else { 0_i64 }
    } else {
        integer_argument(value, "offset must be an integer")?
            .to_i64()
            .ok_or_else(|| BuiltinError::OverflowError("offset is too large".to_string()))?
    };

    let adjusted = if raw < 0 {
        raw.checked_add(buffer_len as i64)
            .ok_or_else(|| BuiltinError::ValueError(context.to_string()))?
    } else {
        raw
    };

    if adjusted < 0 {
        return Err(BuiltinError::ValueError(context.to_string()));
    }

    Ok(adjusted as usize)
}

#[inline]
fn out_of_range_error() -> BuiltinError {
    BuiltinError::ValueError("argument out of range".to_string())
}

#[inline]
fn write_i16(out: &mut [u8], endian: Endian, value: i16) -> Result<(), BuiltinError> {
    let bytes = match endian {
        Endian::Little => value.to_le_bytes(),
        Endian::Big => value.to_be_bytes(),
    };
    out.copy_from_slice(&bytes);
    Ok(())
}

#[inline]
fn write_u16(out: &mut [u8], endian: Endian, value: u16) -> Result<(), BuiltinError> {
    let bytes = match endian {
        Endian::Little => value.to_le_bytes(),
        Endian::Big => value.to_be_bytes(),
    };
    out.copy_from_slice(&bytes);
    Ok(())
}

#[inline]
fn write_i32(out: &mut [u8], endian: Endian, value: i32) -> Result<(), BuiltinError> {
    let bytes = match endian {
        Endian::Little => value.to_le_bytes(),
        Endian::Big => value.to_be_bytes(),
    };
    out.copy_from_slice(&bytes);
    Ok(())
}

#[inline]
fn write_u32(out: &mut [u8], endian: Endian, value: u32) -> Result<(), BuiltinError> {
    let bytes = match endian {
        Endian::Little => value.to_le_bytes(),
        Endian::Big => value.to_be_bytes(),
    };
    out.copy_from_slice(&bytes);
    Ok(())
}

#[inline]
fn write_i64(out: &mut [u8], endian: Endian, value: i64) -> Result<(), BuiltinError> {
    let bytes = match endian {
        Endian::Little => value.to_le_bytes(),
        Endian::Big => value.to_be_bytes(),
    };
    out.copy_from_slice(&bytes);
    Ok(())
}

#[inline]
fn write_u64(out: &mut [u8], endian: Endian, value: u64) -> Result<(), BuiltinError> {
    let bytes = match endian {
        Endian::Little => value.to_le_bytes(),
        Endian::Big => value.to_be_bytes(),
    };
    out.copy_from_slice(&bytes);
    Ok(())
}

#[inline]
fn write_usize(out: &mut [u8], value: usize) -> Result<(), BuiltinError> {
    let bytes = value.to_ne_bytes();
    out.copy_from_slice(&bytes);
    Ok(())
}

#[inline]
fn write_f32(out: &mut [u8], endian: Endian, value: f32) -> Result<(), BuiltinError> {
    write_u32(out, endian, value.to_bits())
}

#[inline]
fn write_f64(out: &mut [u8], endian: Endian, value: f64) -> Result<(), BuiltinError> {
    write_u64(out, endian, value.to_bits())
}

#[inline]
fn read_i16(data: &[u8], endian: Endian) -> i16 {
    let mut bytes = [0_u8; 2];
    bytes.copy_from_slice(data);
    match endian {
        Endian::Little => i16::from_le_bytes(bytes),
        Endian::Big => i16::from_be_bytes(bytes),
    }
}

#[inline]
fn read_u16(data: &[u8], endian: Endian) -> u16 {
    let mut bytes = [0_u8; 2];
    bytes.copy_from_slice(data);
    match endian {
        Endian::Little => u16::from_le_bytes(bytes),
        Endian::Big => u16::from_be_bytes(bytes),
    }
}

#[inline]
fn read_i32(data: &[u8], endian: Endian) -> i32 {
    let mut bytes = [0_u8; 4];
    bytes.copy_from_slice(data);
    match endian {
        Endian::Little => i32::from_le_bytes(bytes),
        Endian::Big => i32::from_be_bytes(bytes),
    }
}

#[inline]
fn read_u32(data: &[u8], endian: Endian) -> u32 {
    let mut bytes = [0_u8; 4];
    bytes.copy_from_slice(data);
    match endian {
        Endian::Little => u32::from_le_bytes(bytes),
        Endian::Big => u32::from_be_bytes(bytes),
    }
}

#[inline]
fn read_i64(data: &[u8], endian: Endian) -> i64 {
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(data);
    match endian {
        Endian::Little => i64::from_le_bytes(bytes),
        Endian::Big => i64::from_be_bytes(bytes),
    }
}

#[inline]
fn read_u64(data: &[u8], endian: Endian) -> u64 {
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(data);
    match endian {
        Endian::Little => u64::from_le_bytes(bytes),
        Endian::Big => u64::from_be_bytes(bytes),
    }
}

#[inline]
fn read_usize(data: &[u8]) -> usize {
    let mut bytes = [0_u8; std::mem::size_of::<usize>()];
    bytes.copy_from_slice(data);
    usize::from_ne_bytes(bytes)
}

#[inline]
fn read_f32(data: &[u8], endian: Endian) -> f32 {
    f32::from_bits(read_u32(data, endian))
}

#[inline]
fn read_f64(data: &[u8], endian: Endian) -> f64 {
    f64::from_bits(read_u64(data, endian))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::interned_by_ptr;

    fn bytes_to_vec(value: Value) -> Vec<u8> {
        let ptr = value.as_object_ptr().expect("bytes-like object");
        unsafe { &*(ptr as *const BytesObject) }.to_vec()
    }

    fn tuple_items(value: Value) -> Vec<Value> {
        let ptr = value.as_object_ptr().expect("tuple object");
        unsafe { &*(ptr as *const TupleObject) }.as_slice().to_vec()
    }

    #[test]
    fn test_calcsize_supports_pickle_formats() {
        for (format, expected) in [
            ("<B", 1_i64),
            ("<H", 2),
            ("<I", 4),
            ("<Q", 8),
            ("<i", 4),
            (">d", 8),
            ("P", std::mem::size_of::<usize>() as i64),
        ] {
            let result = calcsize_builtin(&[Value::string(intern(format))]).expect("calcsize");
            assert_eq!(result.as_int(), Some(expected));
        }
    }

    #[test]
    fn test_pointer_format_is_native_only() {
        let value = Value::int(0x1234).unwrap();
        let packed = pack_builtin(&[Value::string(intern("P")), value]).expect("pack pointer");
        assert_eq!(bytes_to_vec(packed).len(), std::mem::size_of::<usize>());

        let unpacked = unpack_builtin(&[Value::string(intern("P")), packed]).expect("unpack");
        assert_eq!(tuple_items(unpacked), vec![value]);

        let err = calcsize_builtin(&[Value::string(intern("<P"))])
            .expect_err("standard pointer format should be rejected");
        assert!(err.to_string().contains("bad char in struct format"));
    }

    #[test]
    fn test_pack_and_unpack_cover_pickle_formats() {
        let cases = [
            ("<B", Value::int(0x7f).unwrap(), vec![0x7f]),
            ("<H", Value::int(0x1234).unwrap(), vec![0x34, 0x12]),
            (
                "<I",
                Value::int(0x1234_5678).unwrap(),
                vec![0x78, 0x56, 0x34, 0x12],
            ),
            (
                "<Q",
                prism_runtime::types::int::bigint_to_value(BigInt::from(0x0102_0304_0506_0708_u64)),
                vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
            ),
            ("<i", Value::int(-2).unwrap(), vec![0xfe, 0xff, 0xff, 0xff]),
        ];

        for (format, input, expected) in cases {
            let packed = pack_builtin(&[Value::string(intern(format)), input]).expect("pack");
            assert_eq!(bytes_to_vec(packed), expected);

            let unpacked =
                unpack_builtin(&[Value::string(intern(format)), packed]).expect("unpack");
            let items = tuple_items(unpacked);
            assert_eq!(items.len(), 1);
            assert_eq!(
                prism_runtime::types::int::value_to_bigint(items[0]),
                prism_runtime::types::int::value_to_bigint(input)
            );
        }

        let packed = pack_builtin(&[Value::string(intern(">d")), Value::float(1.5)]).expect("pack");
        let unpacked = unpack_builtin(&[Value::string(intern(">d")), packed]).expect("unpack");
        let items = tuple_items(unpacked);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_float(), Some(1.5));
    }

    #[test]
    fn test_pack_into_unpack_from_and_iter_unpack_work_with_offsets() {
        let target = leak_object_value(BytesObject::repeat_with_type(0, 10, TypeId::BYTEARRAY));
        pack_into_builtin(&[
            Value::string(intern("<I")),
            target,
            Value::int(2).unwrap(),
            Value::int(0x1122_3344).unwrap(),
        ])
        .expect("pack_into");

        assert_eq!(
            bytes_to_vec(target),
            vec![0, 0, 0x44, 0x33, 0x22, 0x11, 0, 0, 0, 0]
        );

        let unpacked =
            unpack_from_builtin(&[Value::string(intern("<I")), target, Value::int(2).unwrap()])
                .expect("unpack_from");
        assert_eq!(
            tuple_items(unpacked),
            vec![Value::int(0x1122_3344).unwrap()]
        );

        let iter = iter_unpack_builtin(&[
            Value::string(intern("<H")),
            bytes_value(&[1, 0, 2, 0, 3, 0, 4, 0]),
        ])
        .expect("iter_unpack");
        let first = crate::builtins::builtin_next(&[iter]).expect("first tuple");
        let second = crate::builtins::builtin_next(&[iter]).expect("second tuple");
        assert_eq!(tuple_items(first), vec![Value::int(1).unwrap()]);
        assert_eq!(tuple_items(second), vec![Value::int(2).unwrap()]);
    }

    #[test]
    fn test_struct_constructor_binds_format_and_methods() {
        let receiver = struct_constructor(&[Value::string(intern("<I"))]).expect("Struct()");
        let ptr = receiver.as_object_ptr().expect("struct helper object");
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        let format = shaped.get_property("format").expect("format");
        assert_eq!(
            interned_by_ptr(format.as_string_object_ptr().unwrap() as *const u8)
                .expect("interned format")
                .as_str(),
            "<I"
        );
        assert_eq!(shaped.get_property("size").unwrap().as_int(), Some(4));

        let pack_value = shaped.get_property("pack").expect("pack method");
        let pack_ptr = pack_value.as_object_ptr().expect("pack builtin object");
        let pack_builtin = unsafe { &*(pack_ptr as *const BuiltinFunctionObject) };
        let packed = pack_builtin
            .call(&[Value::int(7).unwrap()])
            .expect("bound pack");
        assert_eq!(bytes_to_vec(packed), vec![7, 0, 0, 0]);
    }
}
