//! Native `binascii` bootstrap module.
//!
//! CPython's pure-Python stdlib imports `binascii` very early through
//! `base64`, `email`, and `quopri`. This module provides the binary/ASCII
//! codecs those import paths need while keeping the public API aligned with
//! CPython's C extension surface.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::truthiness::is_truthy;
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::memoryview::value_as_memoryview_ref;
use prism_runtime::types::string::value_as_string_ref;
use std::sync::{Arc, LazyLock};

const MODULE_DOC: &str = "Conversion between binary data and ASCII";
const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";
const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";
const QP_MAX_LINE_SIZE: usize = 76;

static A2B_UU_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("binascii.a2b_uu"), a2b_uu_builtin));
static B2A_UU_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("binascii.b2a_uu"), b2a_uu_builtin));
static A2B_BASE64_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("binascii.a2b_base64"), a2b_base64_builtin)
});
static B2A_BASE64_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("binascii.b2a_base64"), b2a_base64_builtin)
});
static A2B_HEX_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("binascii.a2b_hex"), a2b_hex_builtin));
static B2A_HEX_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("binascii.b2a_hex"), b2a_hex_builtin));
static HEXLIFY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("binascii.hexlify"), b2a_hex_builtin));
static UNHEXLIFY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("binascii.unhexlify"), a2b_hex_builtin));
static A2B_QP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("binascii.a2b_qp"), a2b_qp_builtin));
static B2A_QP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("binascii.b2a_qp"), b2a_qp_builtin));
static CRC32_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("binascii.crc32"), crc32_builtin));
static CRC_HQX_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("binascii.crc_hqx"), crc_hqx_builtin));

/// Native `binascii` module descriptor.
#[derive(Debug, Clone)]
pub struct BinasciiModule {
    attrs: Vec<Arc<str>>,
}

impl BinasciiModule {
    /// Create a new `binascii` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__doc__"),
                Arc::from("Error"),
                Arc::from("Incomplete"),
                Arc::from("a2b_base64"),
                Arc::from("a2b_hex"),
                Arc::from("a2b_qp"),
                Arc::from("a2b_uu"),
                Arc::from("b2a_base64"),
                Arc::from("b2a_hex"),
                Arc::from("b2a_qp"),
                Arc::from("b2a_uu"),
                Arc::from("crc32"),
                Arc::from("crc_hqx"),
                Arc::from("hexlify"),
                Arc::from("unhexlify"),
            ],
        }
    }
}

impl Default for BinasciiModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for BinasciiModule {
    fn name(&self) -> &str {
        "binascii"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "Error" => Ok(value_error_type_value()),
            "Incomplete" => Ok(exception_type_value()),
            "a2b_base64" => Ok(builtin_value(&A2B_BASE64_FUNCTION)),
            "a2b_hex" => Ok(builtin_value(&A2B_HEX_FUNCTION)),
            "a2b_qp" => Ok(builtin_value(&A2B_QP_FUNCTION)),
            "a2b_uu" => Ok(builtin_value(&A2B_UU_FUNCTION)),
            "b2a_base64" => Ok(builtin_value(&B2A_BASE64_FUNCTION)),
            "b2a_hex" => Ok(builtin_value(&B2A_HEX_FUNCTION)),
            "b2a_qp" => Ok(builtin_value(&B2A_QP_FUNCTION)),
            "b2a_uu" => Ok(builtin_value(&B2A_UU_FUNCTION)),
            "crc32" => Ok(builtin_value(&CRC32_FUNCTION)),
            "crc_hqx" => Ok(builtin_value(&CRC_HQX_FUNCTION)),
            "hexlify" => Ok(builtin_value(&HEXLIFY_FUNCTION)),
            "unhexlify" => Ok(builtin_value(&UNHEXLIFY_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'binascii' has no attribute '{}'",
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
fn value_error_type_value() -> Value {
    Value::object_ptr((&*crate::builtins::VALUE_ERROR) as *const _ as *const ())
}

#[inline]
fn exception_type_value() -> Value {
    Value::object_ptr((&*crate::builtins::EXCEPTION) as *const _ as *const ())
}

#[inline]
fn bytes_value(data: Vec<u8>) -> Value {
    crate::alloc_managed_value(BytesObject::from_vec(data))
}

fn a2b_uu_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "a2b_uu() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let data = ascii_data_arg(args[0], "a2b_uu() argument")?;
    a2b_uu_bytes(&data).map(bytes_value)
}

fn b2a_uu_builtin(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (data_arg, backtick) =
        parse_single_data_with_kwonly_bool("b2a_uu", args, keywords, "backtick", false)?;
    let data = bytes_like_arg(data_arg, "b2a_uu() argument")?;
    b2a_uu_bytes(&data, backtick).map(bytes_value)
}

fn a2b_base64_builtin(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (data_arg, strict_mode) =
        parse_single_data_with_kwonly_bool("a2b_base64", args, keywords, "strict_mode", false)?;
    let data = ascii_data_arg(data_arg, "a2b_base64() argument")?;
    a2b_base64_bytes(&data, strict_mode).map(bytes_value)
}

fn b2a_base64_builtin(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (data_arg, newline) =
        parse_single_data_with_kwonly_bool("b2a_base64", args, keywords, "newline", true)?;
    let data = bytes_like_arg(data_arg, "b2a_base64() argument")?;
    Ok(bytes_value(b2a_base64_bytes(&data, newline)))
}

fn a2b_hex_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "a2b_hex() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let data = ascii_data_arg(args[0], "a2b_hex() argument")?;
    a2b_hex_bytes(&data).map(bytes_value)
}

fn b2a_hex_builtin(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (data_arg, sep, bytes_per_sep) = parse_b2a_hex_args(args, keywords)?;
    let data = bytes_like_arg(data_arg, "b2a_hex() argument")?;
    Ok(bytes_value(b2a_hex_bytes(&data, sep, bytes_per_sep)))
}

fn a2b_qp_builtin(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (data_arg, header) = parse_a2b_qp_args(args, keywords)?;
    let data = ascii_data_arg(data_arg, "a2b_qp() argument")?;
    Ok(bytes_value(a2b_qp_bytes(&data, header)))
}

fn b2a_qp_builtin(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (data_arg, quotetabs, istext, header) = parse_b2a_qp_args(args, keywords)?;
    let data = bytes_like_arg(data_arg, "b2a_qp() argument")?;
    Ok(bytes_value(b2a_qp_bytes(&data, quotetabs, istext, header)))
}

fn crc32_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "crc32() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let data = bytes_like_arg(args[0], "crc32() argument")?;
    let crc = if args.len() == 2 {
        value_to_masked_u32(args[1], "crc32() second argument")?
    } else {
        0
    };
    Ok(bigint_to_value(BigInt::from(crc32_update(&data, crc))))
}

fn crc_hqx_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "crc_hqx() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let data = bytes_like_arg(args[0], "crc_hqx() argument")?;
    let crc = value_to_masked_u32(args[1], "crc_hqx() second argument")?;
    Ok(bigint_to_value(BigInt::from(crc_hqx_update(&data, crc))))
}

fn parse_single_data_with_kwonly_bool(
    function_name: &str,
    args: &[Value],
    keywords: &[(&str, Value)],
    keyword_name: &str,
    default: bool,
) -> Result<(Value, bool), BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes exactly one positional argument ({} given)",
            function_name,
            args.len()
        )));
    }
    reject_unknown_keywords(function_name, keywords, &[keyword_name])?;
    let flag = keyword_value(function_name, keywords, keyword_name)?
        .map(is_truthy)
        .unwrap_or(default);
    Ok((args[0], flag))
}

fn parse_b2a_hex_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<(Value, Option<u8>, i64), BuiltinError> {
    if !(1..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "b2a_hex() takes from 1 to 3 positional arguments but {} were given",
            args.len()
        )));
    }
    reject_unknown_keywords("b2a_hex", keywords, &["sep", "bytes_per_sep"])?;

    let mut sep = args.get(1).copied();
    let mut bytes_per_sep = args.get(2).copied();
    if let Some(value) = keyword_value("b2a_hex", keywords, "sep")? {
        if sep.is_some() {
            return Err(multiple_values_error("b2a_hex", "sep"));
        }
        sep = Some(value);
    }
    if let Some(value) = keyword_value("b2a_hex", keywords, "bytes_per_sep")? {
        if bytes_per_sep.is_some() {
            return Err(multiple_values_error("b2a_hex", "bytes_per_sep"));
        }
        bytes_per_sep = Some(value);
    }

    let sep = match sep {
        Some(value) => separator_byte(value)?,
        None => None,
    };
    let bytes_per_sep = match bytes_per_sep {
        Some(value) => value_to_i64(value, "bytes_per_sep")?,
        None => 1,
    };

    Ok((args[0], sep, bytes_per_sep))
}

fn parse_a2b_qp_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<(Value, bool), BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "a2b_qp() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }
    reject_unknown_keywords("a2b_qp", keywords, &["data", "header"])?;

    let mut data = args.first().copied();
    let mut header = args.get(1).copied().map(is_truthy).unwrap_or(false);
    if let Some(value) = keyword_value("a2b_qp", keywords, "data")? {
        if data.is_some() {
            return Err(multiple_values_error("a2b_qp", "data"));
        }
        data = Some(value);
    }
    if let Some(value) = keyword_value("a2b_qp", keywords, "header")? {
        if args.len() >= 2 {
            return Err(multiple_values_error("a2b_qp", "header"));
        }
        header = is_truthy(value);
    }

    let Some(data) = data else {
        return Err(BuiltinError::TypeError(
            "a2b_qp() missing required argument 'data'".to_string(),
        ));
    };

    Ok((data, header))
}

fn parse_b2a_qp_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<(Value, bool, bool, bool), BuiltinError> {
    if args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "b2a_qp() takes from 1 to 4 positional arguments but {} were given",
            args.len()
        )));
    }
    reject_unknown_keywords(
        "b2a_qp",
        keywords,
        &["data", "quotetabs", "istext", "header"],
    )?;

    let mut data = args.first().copied();
    let mut quotetabs = args.get(1).copied().map(is_truthy).unwrap_or(false);
    let mut istext = args.get(2).copied().map(is_truthy).unwrap_or(true);
    let mut header = args.get(3).copied().map(is_truthy).unwrap_or(false);

    if let Some(value) = keyword_value("b2a_qp", keywords, "data")? {
        if data.is_some() {
            return Err(multiple_values_error("b2a_qp", "data"));
        }
        data = Some(value);
    }
    if let Some(value) = keyword_value("b2a_qp", keywords, "quotetabs")? {
        if args.len() >= 2 {
            return Err(multiple_values_error("b2a_qp", "quotetabs"));
        }
        quotetabs = is_truthy(value);
    }
    if let Some(value) = keyword_value("b2a_qp", keywords, "istext")? {
        if args.len() >= 3 {
            return Err(multiple_values_error("b2a_qp", "istext"));
        }
        istext = is_truthy(value);
    }
    if let Some(value) = keyword_value("b2a_qp", keywords, "header")? {
        if args.len() >= 4 {
            return Err(multiple_values_error("b2a_qp", "header"));
        }
        header = is_truthy(value);
    }

    let Some(data) = data else {
        return Err(BuiltinError::TypeError(
            "b2a_qp() missing required argument 'data'".to_string(),
        ));
    };

    Ok((data, quotetabs, istext, header))
}

fn keyword_value(
    function_name: &str,
    keywords: &[(&str, Value)],
    target: &str,
) -> Result<Option<Value>, BuiltinError> {
    let mut found = None;
    for (name, value) in keywords {
        if *name == target {
            if found.is_some() {
                return Err(BuiltinError::TypeError(format!(
                    "{}() got multiple values for keyword argument '{}'",
                    function_name, target
                )));
            }
            found = Some(*value);
        }
    }
    Ok(found)
}

fn reject_unknown_keywords(
    function_name: &str,
    keywords: &[(&str, Value)],
    allowed: &[&str],
) -> Result<(), BuiltinError> {
    for (name, _) in keywords {
        if !allowed.iter().any(|allowed_name| allowed_name == name) {
            return Err(BuiltinError::TypeError(format!(
                "{}() got an unexpected keyword argument '{}'",
                function_name, name
            )));
        }
    }
    Ok(())
}

fn multiple_values_error(function_name: &str, argument_name: &str) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "{}() got multiple values for argument '{}'",
        function_name, argument_name
    ))
}

fn bytes_like_arg(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be a bytes-like object, not {}",
            value.type_name()
        )));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec()),
        TypeId::MEMORYVIEW => {
            let view = value_as_memoryview_ref(value)
                .ok_or_else(|| BuiltinError::TypeError("invalid memoryview object".to_string()))?;
            if view.released() {
                return Err(BuiltinError::ValueError(
                    "operation forbidden on released memoryview object".to_string(),
                ));
            }
            Ok(view.to_vec())
        }
        _ => {
            if let Some(bytes) = super::array::value_as_array_bytes(value)? {
                return Ok(bytes);
            }
            Err(BuiltinError::TypeError(format!(
                "{context} must be a bytes-like object, not {}",
                value.type_name()
            )))
        }
    }
}

fn ascii_data_arg(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    if let Some(text) = value_as_string_ref(value) {
        let text = text.as_str();
        if !text.is_ascii() {
            return Err(BuiltinError::ValueError(
                "string argument should contain only ASCII characters".to_string(),
            ));
        }
        return Ok(text.as_bytes().to_vec());
    }

    bytes_like_arg(value, context)
}

fn separator_byte(value: Value) -> Result<Option<u8>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }

    if let Some(text) = value_as_string_ref(value) {
        let text = text.as_str();
        if text.len() != 1 || !text.is_ascii() {
            return Err(BuiltinError::ValueError(
                "sep must be length 1 and ASCII".to_string(),
            ));
        }
        return Ok(Some(text.as_bytes()[0]));
    }

    let bytes = bytes_like_arg(value, "sep")?;
    if bytes.len() != 1 {
        return Err(BuiltinError::ValueError("sep must be length 1".to_string()));
    }
    Ok(Some(bytes[0]))
}

fn value_to_i64(value: Value, context: &str) -> Result<i64, BuiltinError> {
    numeric_bigint(value, context)?
        .to_i64()
        .ok_or_else(|| BuiltinError::OverflowError("Python int too large to convert".to_string()))
}

fn value_to_masked_u32(value: Value, context: &str) -> Result<u32, BuiltinError> {
    let value = numeric_bigint(value, context)?;
    Ok(mask_bigint_to_u32(value))
}

fn numeric_bigint(value: Value, context: &str) -> Result<BigInt, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(BigInt::from(flag as u8));
    }
    value_to_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{context} must be an integer, not {}",
            value.type_name()
        ))
    })
}

fn mask_bigint_to_u32(value: BigInt) -> u32 {
    let modulus = BigInt::from(1_u64) << 32_u32;
    let mut normalized = value % &modulus;
    if normalized.sign() == Sign::Minus {
        normalized += &modulus;
    }
    normalized
        .to_u32()
        .expect("32-bit masked integer must fit u32")
}

fn a2b_uu_bytes(data: &[u8]) -> Result<Vec<u8>, BuiltinError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let expected_len = data[0].wrapping_sub(b' ') & 0x3f;
    let expected_len = expected_len as usize;
    let mut out = Vec::with_capacity(expected_len);
    let mut index = 1;
    let mut leftbits = 0_u32;
    let mut leftchar = 0_u32;

    while out.len() < expected_len {
        let byte = if index < data.len() {
            let byte = data[index];
            index += 1;
            byte
        } else {
            0
        };
        let value = if byte == b'\n' || byte == b'\r' || byte == 0 {
            0
        } else {
            if !(b' '..=b'`').contains(&byte) {
                return Err(BuiltinError::ValueError("Illegal char".to_string()));
            }
            byte.wrapping_sub(b' ') & 0x3f
        };

        leftchar = (leftchar << 6) | u32::from(value);
        leftbits += 6;
        if leftbits >= 8 {
            leftbits -= 8;
            out.push(((leftchar >> leftbits) & 0xff) as u8);
            leftchar &= (1_u32 << leftbits) - 1;
        }
    }

    while index < data.len() {
        let byte = data[index];
        index += 1;
        if byte != b' ' && byte != b'`' && byte != b'\n' && byte != b'\r' {
            return Err(BuiltinError::ValueError("Trailing garbage".to_string()));
        }
    }

    Ok(out)
}

fn b2a_uu_bytes(data: &[u8], backtick: bool) -> Result<Vec<u8>, BuiltinError> {
    if data.len() > 45 {
        return Err(BuiltinError::ValueError(
            "At most 45 bytes at once".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(2 + data.len().div_ceil(3) * 4);
    if backtick && data.is_empty() {
        out.push(b'`');
    } else {
        out.push(b' ' + data.len() as u8);
    }

    let mut index = 0;
    let mut leftbits = 0_u32;
    let mut leftchar = 0_u32;
    let mut remaining = data.len();
    while remaining > 0 || leftbits != 0 {
        if remaining > 0 {
            leftchar = (leftchar << 8) | u32::from(data[index]);
            index += 1;
            remaining -= 1;
        } else {
            leftchar <<= 8;
        }
        leftbits += 8;

        while leftbits >= 6 {
            let value = ((leftchar >> (leftbits - 6)) & 0x3f) as u8;
            leftbits -= 6;
            if backtick && value == 0 {
                out.push(b'`');
            } else {
                out.push(value + b' ');
            }
        }
    }

    out.push(b'\n');
    Ok(out)
}

fn a2b_base64_bytes(data: &[u8], strict_mode: bool) -> Result<Vec<u8>, BuiltinError> {
    if strict_mode && data.first() == Some(&b'=') {
        return Err(BuiltinError::ValueError(
            "Leading padding not allowed".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(data.len().div_ceil(4) * 3);
    let mut quad_pos = 0_u8;
    let mut leftchar = 0_u8;
    let mut pads = 0_u8;
    let mut padding_started = false;
    let mut data_chars = 0_usize;

    for (index, byte) in data.iter().copied().enumerate() {
        if byte == b'=' {
            padding_started = true;
            if strict_mode && quad_pos == 0 {
                return Err(BuiltinError::ValueError(
                    "Excess padding not allowed".to_string(),
                ));
            }
            if quad_pos >= 2 {
                pads = pads.saturating_add(1);
                if quad_pos + pads >= 4 {
                    if strict_mode && index + 1 < data.len() {
                        return Err(BuiltinError::ValueError(
                            "Excess data after padding".to_string(),
                        ));
                    }
                    return Ok(out);
                }
            }
            continue;
        }

        let Some(value) = base64_value(byte) else {
            if strict_mode {
                return Err(BuiltinError::ValueError(
                    "Only base64 data is allowed".to_string(),
                ));
            }
            continue;
        };

        if strict_mode && padding_started {
            return Err(BuiltinError::ValueError(
                "Discontinuous padding not allowed".to_string(),
            ));
        }
        pads = 0;
        data_chars += 1;

        match quad_pos {
            0 => {
                quad_pos = 1;
                leftchar = value;
            }
            1 => {
                quad_pos = 2;
                out.push((leftchar << 2) | (value >> 4));
                leftchar = value & 0x0f;
            }
            2 => {
                quad_pos = 3;
                out.push((leftchar << 4) | (value >> 2));
                leftchar = value & 0x03;
            }
            _ => {
                quad_pos = 0;
                out.push((leftchar << 6) | value);
                leftchar = 0;
            }
        }
    }

    if quad_pos == 0 {
        return Ok(out);
    }

    if quad_pos == 1 {
        Err(BuiltinError::ValueError(format!(
            "Invalid base64-encoded string: number of data characters ({data_chars}) cannot be 1 more than a multiple of 4"
        )))
    } else {
        Err(BuiltinError::ValueError("Incorrect padding".to_string()))
    }
}

fn b2a_base64_bytes(data: &[u8], newline: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len().div_ceil(3) * 4 + usize::from(newline));
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);

        out.push(BASE64_TABLE[(b0 >> 2) as usize]);
        out.push(BASE64_TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize]);
        if chunk.len() >= 2 {
            out.push(BASE64_TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize]);
        } else {
            out.push(b'=');
        }
        if chunk.len() == 3 {
            out.push(BASE64_TABLE[(b2 & 0x3f) as usize]);
        } else {
            out.push(b'=');
        }
    }
    if newline {
        out.push(b'\n');
    }
    out
}

#[inline]
fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn a2b_hex_bytes(data: &[u8]) -> Result<Vec<u8>, BuiltinError> {
    if data.len() % 2 != 0 {
        return Err(BuiltinError::ValueError("Odd-length string".to_string()));
    }

    let mut out = Vec::with_capacity(data.len() / 2);
    for pair in data.chunks_exact(2) {
        let Some(high) = hex_value(pair[0]) else {
            return Err(BuiltinError::ValueError(
                "Non-hexadecimal digit found".to_string(),
            ));
        };
        let Some(low) = hex_value(pair[1]) else {
            return Err(BuiltinError::ValueError(
                "Non-hexadecimal digit found".to_string(),
            ));
        };
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn b2a_hex_bytes(data: &[u8], sep: Option<u8>, bytes_per_sep: i64) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let group = bytes_per_sep.unsigned_abs() as usize;
    let use_sep = sep.is_some() && group > 0;
    let mut out = Vec::with_capacity(data.len() * 2 + data.len().saturating_sub(1));
    for (index, byte) in data.iter().copied().enumerate() {
        if use_sep && index > 0 {
            let insert = if bytes_per_sep > 0 {
                (data.len() - index).is_multiple_of(group)
            } else {
                index.is_multiple_of(group)
            };
            if insert {
                out.push(sep.expect("separator checked above"));
            }
        }
        out.push(HEX_LOWER[(byte >> 4) as usize]);
        out.push(HEX_LOWER[(byte & 0x0f) as usize]);
    }
    out
}

#[inline]
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn a2b_qp_bytes(data: &[u8], header: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut index = 0;
    while index < data.len() {
        match data[index] {
            b'=' => {
                index += 1;
                if index >= data.len() {
                    break;
                }

                if data[index] == b'\n' || data[index] == b'\r' {
                    if data[index] != b'\n' {
                        while index < data.len() && data[index] != b'\n' {
                            index += 1;
                        }
                    }
                    if index < data.len() {
                        index += 1;
                    }
                } else if data[index] == b'=' {
                    out.push(b'=');
                    index += 1;
                } else if index + 1 < data.len() {
                    match (hex_value(data[index]), hex_value(data[index + 1])) {
                        (Some(high), Some(low)) => {
                            out.push((high << 4) | low);
                            index += 2;
                        }
                        _ => out.push(b'='),
                    }
                } else {
                    out.push(b'=');
                }
            }
            b'_' if header => {
                out.push(b' ');
                index += 1;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    out
}

fn b2a_qp_bytes(data: &[u8], quotetabs: bool, istext: bool, header: bool) -> Vec<u8> {
    let crlf = data
        .iter()
        .position(|byte| *byte == b'\n')
        .is_some_and(|index| index > 0 && data[index - 1] == b'\r');
    let mut out = Vec::with_capacity(data.len());
    let mut index = 0;
    let mut line_len = 0_usize;

    while index < data.len() {
        if qp_needs_quote(data, index, line_len, quotetabs, istext, header) {
            if line_len + 3 >= QP_MAX_LINE_SIZE {
                push_qp_soft_break(&mut out, crlf);
                line_len = 0;
            }
            push_qp_escape(&mut out, data[index]);
            index += 1;
            line_len += 3;
            continue;
        }

        if istext
            && (data[index] == b'\n'
                || (index + 1 < data.len() && data[index] == b'\r' && data[index + 1] == b'\n'))
        {
            line_len = 0;
            if out
                .last()
                .is_some_and(|byte| *byte == b' ' || *byte == b'\t')
            {
                let ch = out.pop().expect("last byte checked above");
                push_qp_escape(&mut out, ch);
            }
            if crlf {
                out.push(b'\r');
            }
            out.push(b'\n');
            if data[index] == b'\r' {
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }

        if index + 1 != data.len() && data[index + 1] != b'\n' && line_len + 1 >= QP_MAX_LINE_SIZE {
            push_qp_soft_break(&mut out, crlf);
            line_len = 0;
        }

        line_len += 1;
        if header && data[index] == b' ' {
            out.push(b'_');
        } else {
            out.push(data[index]);
        }
        index += 1;
    }

    out
}

fn qp_needs_quote(
    data: &[u8],
    index: usize,
    line_len: usize,
    quotetabs: bool,
    istext: bool,
    header: bool,
) -> bool {
    let byte = data[index];
    byte > 126
        || byte == b'='
        || (header && byte == b'_')
        || (byte == b'.'
            && line_len == 0
            && (index + 1 == data.len()
                || data[index + 1] == b'\n'
                || data[index + 1] == b'\r'
                || data[index + 1] == 0))
        || (!istext && (byte == b'\r' || byte == b'\n'))
        || ((byte == b'\t' || byte == b' ') && index + 1 == data.len())
        || (byte < 33
            && byte != b'\r'
            && byte != b'\n'
            && (quotetabs || (byte != b'\t' && byte != b' ')))
}

#[inline]
fn push_qp_soft_break(out: &mut Vec<u8>, crlf: bool) {
    out.push(b'=');
    if crlf {
        out.push(b'\r');
    }
    out.push(b'\n');
}

#[inline]
fn push_qp_escape(out: &mut Vec<u8>, byte: u8) {
    out.push(b'=');
    out.push(HEX_UPPER[(byte >> 4) as usize]);
    out.push(HEX_UPPER[(byte & 0x0f) as usize]);
}

fn crc32_update(data: &[u8], crc: u32) -> u32 {
    let mut crc = crc ^ 0xffff_ffff;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            if crc & 1 == 0 {
                crc >>= 1;
            } else {
                crc = (crc >> 1) ^ 0xedb8_8320;
            }
        }
    }
    crc ^ 0xffff_ffff
}

fn crc_hqx_update(data: &[u8], crc: u32) -> u32 {
    let mut crc = crc & 0xffff;
    for byte in data {
        crc ^= u32::from(*byte) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = ((crc << 1) ^ 0x1021) & 0xffff;
            } else {
                crc = (crc << 1) & 0xffff;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests;
