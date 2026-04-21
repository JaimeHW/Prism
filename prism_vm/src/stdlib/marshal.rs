//! Native `marshal` module bootstrap surface.
//!
//! Prism uses this native module to cover the scalar marshal semantics that
//! CPython's regression suite relies on during early compatibility work. The
//! codec is intentionally structured so more marshal tags can be added without
//! rewriting the public module surface.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::{bigint_to_value, is_int_value, value_to_bigint};
use prism_runtime::types::string::value_as_string_ref;
use std::sync::{Arc, LazyLock};

const MARSHAL_VERSION: i64 = 4;

const TYPE_NONE: u8 = b'N';
const TYPE_FALSE: u8 = b'F';
const TYPE_TRUE: u8 = b'T';
const TYPE_INT: u8 = b'i';
const TYPE_LONG: u8 = b'l';
const TYPE_STRING: u8 = b's';
const TYPE_UNICODE: u8 = b'u';

static DUMPS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("marshal.dumps"), marshal_dumps));
static LOADS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("marshal.loads"), marshal_loads));

/// Native `marshal` module descriptor.
#[derive(Debug, Clone)]
pub struct MarshalModule {
    attrs: Vec<Arc<str>>,
}

impl MarshalModule {
    /// Create a new `marshal` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("dumps"), Arc::from("loads"), Arc::from("version")],
        }
    }
}

impl Default for MarshalModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for MarshalModule {
    fn name(&self) -> &str {
        "marshal"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "dumps" => Ok(builtin_value(&DUMPS_FUNCTION)),
            "loads" => Ok(builtin_value(&LOADS_FUNCTION)),
            "version" => Ok(Value::int(MARSHAL_VERSION).expect("marshal version should fit")),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'marshal' has no attribute '{}'",
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
fn bytes_value(value: &[u8]) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(BytesObject::from_slice(value))) as *const ())
}

fn marshal_dumps(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "dumps() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    if args.len() == 2 {
        value_to_version(args[1])?;
    }

    let mut buffer = Vec::new();
    encode_marshaled_value(args[0], &mut buffer)?;
    Ok(bytes_value(&buffer))
}

fn marshal_loads(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "loads() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let bytes = value_to_bytes_like(args[0], "loads() argument")?;
    decode_marshaled_value(&bytes)
}

fn value_to_version(value: Value) -> Result<i32, BuiltinError> {
    let bigint = value_to_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "marshal version must be int, not {}",
            value.type_name()
        ))
    })?;

    bigint.to_i32().ok_or_else(|| {
        BuiltinError::OverflowError("Python int too large to convert to C int".to_string())
    })
}

fn value_to_bytes_like(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        )));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec()),
        _ => Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        ))),
    }
}

fn encode_marshaled_value(value: Value, out: &mut Vec<u8>) -> Result<(), BuiltinError> {
    if value.is_none() {
        out.push(TYPE_NONE);
        return Ok(());
    }

    if let Some(boolean) = value.as_bool() {
        out.push(if boolean { TYPE_TRUE } else { TYPE_FALSE });
        return Ok(());
    }

    if is_int_value(value) {
        let bigint = value_to_bigint(value)
            .ok_or_else(|| BuiltinError::ValueError("unmarshallable object".to_string()))?;
        encode_int(&bigint, out);
        return Ok(());
    }

    if let Some(string) = value_as_string_ref(value) {
        let text = string.as_str();
        out.push(TYPE_UNICODE);
        write_len(text.len(), out)?;
        out.extend_from_slice(text.as_bytes());
        return Ok(());
    }

    if let Some(bytes) = value.as_object_ptr().and_then(byte_sequence_ref_from_ptr) {
        out.push(TYPE_STRING);
        write_len(bytes.len(), out)?;
        out.extend_from_slice(bytes.as_bytes());
        return Ok(());
    }

    Err(BuiltinError::ValueError(
        "unmarshallable object".to_string(),
    ))
}

fn byte_sequence_ref_from_ptr(ptr: *const ()) -> Option<&'static BytesObject> {
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Some(unsafe { &*(ptr as *const BytesObject) }),
        _ => None,
    }
}

fn write_len(len: usize, out: &mut Vec<u8>) -> Result<(), BuiltinError> {
    let len = i32::try_from(len)
        .map_err(|_| BuiltinError::ValueError("unmarshallable object".to_string()))?;
    out.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn encode_int(value: &BigInt, out: &mut Vec<u8>) {
    if let Some(n) = value.to_i32() {
        out.push(TYPE_INT);
        out.extend_from_slice(&n.to_le_bytes());
        return;
    }

    out.push(TYPE_LONG);
    encode_long_digits(value, out);
}

fn encode_long_digits(value: &BigInt, out: &mut Vec<u8>) {
    const BASE_BITS: u32 = 15;
    const BASE: i32 = 1 << BASE_BITS;
    const BASE_MASK: i64 = (1 << BASE_BITS) - 1;

    let (sign, bytes) = value.to_bytes_le();
    let mut magnitude = BigInt::from_bytes_le(Sign::Plus, &bytes);
    let mut digits = Vec::new();
    while magnitude != BigInt::from(0_u8) {
        let digit = (&magnitude & BigInt::from(BASE_MASK))
            .to_i32()
            .expect("masked marshal digit should fit in i32");
        digits.push(u16::try_from(digit).expect("marshal digit should fit in u16"));
        magnitude >>= BASE_BITS;
    }

    let count = if sign == Sign::Minus {
        -(digits.len() as i32)
    } else {
        digits.len() as i32
    };
    out.extend_from_slice(&count.to_le_bytes());
    for digit in digits {
        let digit = i32::from(digit % u16::try_from(BASE).expect("base should fit u16"));
        out.extend_from_slice(&(digit as u16).to_le_bytes());
    }
}

fn decode_marshaled_value(bytes: &[u8]) -> Result<Value, BuiltinError> {
    let Some((&tag, payload)) = bytes.split_first() else {
        return Err(BuiltinError::ValueError(
            "EOF read where object expected".to_string(),
        ));
    };

    match tag {
        TYPE_NONE => Ok(Value::none()),
        TYPE_FALSE => Ok(Value::bool(false)),
        TYPE_TRUE => Ok(Value::bool(true)),
        TYPE_INT => {
            let n = read_i32(payload)?;
            Value::int(i64::from(n)).ok_or_else(|| {
                BuiltinError::ValueError("bad marshal data (integer overflow)".to_string())
            })
        }
        TYPE_LONG => decode_long_value(payload),
        TYPE_STRING => {
            let (data, _) = read_payload_bytes(payload)?;
            Ok(bytes_value(data))
        }
        TYPE_UNICODE => {
            let (data, _) = read_payload_bytes(payload)?;
            let text = std::str::from_utf8(data).map_err(|_| {
                BuiltinError::ValueError("bad marshal data (invalid utf-8)".to_string())
            })?;
            Ok(Value::object_ptr(Box::into_raw(Box::new(
                prism_runtime::types::string::StringObject::new(text),
            )) as *const ()))
        }
        _ => Err(BuiltinError::ValueError(
            "bad marshal data (unknown type code)".to_string(),
        )),
    }
}

fn read_i32(bytes: &[u8]) -> Result<i32, BuiltinError> {
    let raw = bytes
        .get(..4)
        .ok_or_else(|| BuiltinError::ValueError("EOF read where not expected".to_string()))?;
    let mut buf = [0_u8; 4];
    buf.copy_from_slice(raw);
    Ok(i32::from_le_bytes(buf))
}

fn read_payload_bytes(bytes: &[u8]) -> Result<(&[u8], &[u8]), BuiltinError> {
    let len = read_i32(bytes)?;
    if len < 0 {
        return Err(BuiltinError::ValueError(
            "bad marshal data (bytes object size out of range)".to_string(),
        ));
    }
    let len = usize::try_from(len).expect("non-negative i32 should fit usize");
    let start = 4;
    let end = start + len;
    if end > bytes.len() {
        return Err(BuiltinError::ValueError(
            "marshal data too short".to_string(),
        ));
    }
    Ok((&bytes[start..end], &bytes[end..]))
}

fn decode_long_value(bytes: &[u8]) -> Result<Value, BuiltinError> {
    let digit_count = read_i32(bytes)?;
    let negative = digit_count < 0;
    let digit_count = digit_count.unsigned_abs() as usize;
    let raw_digits = bytes
        .get(4..4 + digit_count * 2)
        .ok_or_else(|| BuiltinError::ValueError("marshal data too short".to_string()))?;

    let mut value = BigInt::from(0_u8);
    for (index, chunk) in raw_digits.chunks_exact(2).enumerate() {
        let digit = u16::from_le_bytes([chunk[0], chunk[1]]);
        if digit >= (1 << 15) {
            return Err(BuiltinError::ValueError(
                "bad marshal data (digit out of range in long)".to_string(),
            ));
        }
        let term = BigInt::from(digit) << (index * 15);
        value += term;
    }

    if negative {
        value = -value;
    }

    Ok(bigint_to_value(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::intern;

    fn bytes_to_vec(value: Value) -> Vec<u8> {
        let ptr = value.as_object_ptr().expect("bytes should be heap-backed");
        let bytes = unsafe { &*(ptr as *const BytesObject) };
        bytes.to_vec()
    }

    #[test]
    fn test_marshal_module_exposes_version_and_callables() {
        let module = MarshalModule::new();

        assert_eq!(
            module.get_attr("version").unwrap().as_int(),
            Some(MARSHAL_VERSION)
        );
        assert!(module.get_attr("dumps").is_ok());
        assert!(module.get_attr("loads").is_ok());
        assert_eq!(
            module.dir(),
            vec![Arc::from("dumps"), Arc::from("loads"), Arc::from("version")]
        );
    }

    #[test]
    fn test_marshal_dumps_and_loads_bool_values() {
        let true_bytes =
            marshal_dumps(&[Value::bool(true)]).expect("marshal.dumps(True) should succeed");
        let false_bytes =
            marshal_dumps(&[Value::bool(false)]).expect("marshal.dumps(False) should succeed");

        assert_eq!(bytes_to_vec(true_bytes), vec![TYPE_TRUE]);
        assert_eq!(bytes_to_vec(false_bytes), vec![TYPE_FALSE]);
        assert_eq!(marshal_loads(&[true_bytes]).unwrap().as_bool(), Some(true));
        assert_eq!(
            marshal_loads(&[false_bytes]).unwrap().as_bool(),
            Some(false)
        );
    }

    #[test]
    fn test_marshal_round_trips_small_and_large_ints() {
        let small =
            marshal_dumps(&[Value::int(123_456).unwrap()]).expect("small ints should marshal");
        assert_eq!(marshal_loads(&[small]).unwrap().as_int(), Some(123_456));

        let big = bigint_to_value(BigInt::from(1_u8) << 80_u32);
        let round_tripped =
            marshal_loads(&[marshal_dumps(&[big]).expect("big ints should marshal")])
                .expect("big ints should unmarshal");
        assert_eq!(value_to_bigint(round_tripped), value_to_bigint(big));
    }

    #[test]
    fn test_marshal_round_trips_string_and_bytes() {
        let text = Value::string(intern("prism"));
        let text_round_trip =
            marshal_loads(&[marshal_dumps(&[text]).expect("strings should marshal")])
                .expect("strings should unmarshal");
        assert_eq!(
            value_as_string_ref(text_round_trip)
                .expect("string round-trip should return a string")
                .as_str(),
            "prism"
        );

        let bytes = bytes_value(b"abc");
        let bytes_round_trip =
            marshal_loads(&[marshal_dumps(&[bytes]).expect("bytes should marshal")])
                .expect("bytes should unmarshal");
        assert_eq!(bytes_to_vec(bytes_round_trip), b"abc");
    }

    #[test]
    fn test_marshal_loads_ignores_trailing_bytes() {
        let result = decode_marshaled_value(&[TYPE_TRUE, 0x99, 0x98])
            .expect("trailing bytes should be ignored");
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_marshal_rejects_unsupported_values() {
        let err = marshal_dumps(&[bytes_value(b"abc"), Value::none(), Value::none()])
            .expect_err("extra marshal.dumps args should fail");
        assert!(
            err.to_string()
                .contains("takes from 1 to 2 positional arguments")
        );

        let object = crate::builtins::builtin_object(&[]).expect("object() should succeed");
        let err = marshal_dumps(&[object]).expect_err("unsupported values should fail");
        assert!(err.to_string().contains("unmarshallable object"));
    }
}
