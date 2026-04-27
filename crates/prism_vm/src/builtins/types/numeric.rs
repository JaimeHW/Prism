use super::*;

pub fn builtin_int(args: &[Value]) -> Result<Value, BuiltinError> {
    let Some((arg, explicit_base)) = parse_builtin_int_args(args)? else {
        return Ok(Value::int(0).expect("zero should be representable"));
    };

    if let Some(value) = builtin_int_native(arg, explicit_base)? {
        return Ok(value);
    }

    if let Some(buffer_arg) = int_buffer_argument(arg) {
        return parse_int_text_argument(&buffer_arg, 10);
    }

    Err(builtin_int_unsupported_argument(arg))
}

pub fn builtin_int_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let Some((arg, explicit_base)) = parse_builtin_int_args(args)? else {
        return Ok(Value::int(0).expect("zero should be representable"));
    };

    if let Some(value) = builtin_int_native(arg, explicit_base)? {
        return Ok(value);
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__int__")? {
        return int_protocol_result(result, "__int__");
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__index__")? {
        return int_protocol_result(result, "__index__");
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__trunc__")? {
        return int_trunc_protocol_result(vm, result);
    }

    if let Some(buffer_arg) = int_buffer_argument(arg) {
        return parse_int_text_argument(&buffer_arg, 10);
    }

    Err(builtin_int_unsupported_argument(arg))
}

pub(super) fn builtin_int_kw(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let args = collect_builtin_int_keyword_args(positional, keywords)?;
    builtin_int(&args)
}

pub(super) fn builtin_int_kw_vm(
    vm: &mut VirtualMachine,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let args = collect_builtin_int_keyword_args(positional, keywords)?;
    builtin_int_vm(vm, &args)
}

fn collect_builtin_int_keyword_args(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Vec<Value>, BuiltinError> {
    if positional.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "int() takes at most 2 arguments ({} given)",
            positional.len() + keywords.len()
        )));
    }

    let mut base = None;
    for &(name, value) in keywords {
        match name {
            "base" => {
                if positional.len() >= 2 {
                    return Err(BuiltinError::TypeError(format!(
                        "int() takes at most 2 arguments ({} given)",
                        positional.len() + 1
                    )));
                }
                if base.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "int() got multiple values for keyword argument 'base'".to_string(),
                    ));
                }
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "int() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    match (positional, base) {
        ([], None) => Ok(Vec::new()),
        ([], Some(_)) => Err(BuiltinError::TypeError(
            "int() missing string argument".to_string(),
        )),
        ([arg], None) => Ok(vec![*arg]),
        ([arg], Some(base)) => Ok(vec![*arg, base]),
        ([arg, base], None) => Ok(vec![*arg, *base]),
        _ => unreachable!("positional argument count was validated"),
    }
}

#[inline]
fn parse_builtin_int_args(args: &[Value]) -> Result<Option<(Value, Option<u32>)>, BuiltinError> {
    if args.is_empty() {
        return Ok(None);
    }
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "int() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }

    let explicit_base = if args.len() == 2 {
        Some(parse_int_base_argument(args[1])?)
    } else {
        None
    };
    Ok(Some((args[0], explicit_base)))
}

#[inline]
fn builtin_int_native(
    arg: Value,
    explicit_base: Option<u32>,
) -> Result<Option<Value>, BuiltinError> {
    if let Some(text_arg) = int_text_argument(arg) {
        return parse_int_text_argument(&text_arg, explicit_base.unwrap_or(10)).map(Some);
    }

    if explicit_base.is_some() {
        return Err(BuiltinError::TypeError(
            "int() can't convert non-string with explicit base".to_string(),
        ));
    }

    if arg.as_int().is_some() || value_as_heap_int(arg).is_some() {
        return Ok(Some(arg));
    }
    if let Some(integer) = value_to_bigint(arg) {
        return Ok(Some(bigint_to_value(integer)));
    }
    if let Some(f) = arg.as_float() {
        return float_to_int_value(f).map(Some);
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Some(
            Value::int(if b { 1 } else { 0 }).expect("bool integer should be representable"),
        ));
    }

    Ok(None)
}

fn float_to_int_value(value: f64) -> Result<Value, BuiltinError> {
    if value.is_nan() {
        return Err(BuiltinError::ValueError(
            "cannot convert float NaN to integer".to_string(),
        ));
    }
    if value.is_infinite() {
        return Err(BuiltinError::OverflowError(
            "cannot convert float infinity to integer".to_string(),
        ));
    }

    const F64_EXPONENT_BIAS: i32 = 1023;
    const F64_FRACTION_BITS_U64: u32 = 52;
    const F64_FRACTION_MASK: u64 = (1_u64 << F64_FRACTION_BITS_U64) - 1;
    const F64_EXPONENT_MASK: u64 = 0x7ff;

    let bits = value.to_bits();
    let exponent_bits = ((bits >> F64_FRACTION_BITS_U64) & F64_EXPONENT_MASK) as i32;
    if exponent_bits == 0 {
        return Ok(Value::int(0).expect("zero should be representable"));
    }

    let exponent = exponent_bits - F64_EXPONENT_BIAS;
    if exponent < 0 {
        return Ok(Value::int(0).expect("zero should be representable"));
    }

    let significand = (1_u64 << F64_FRACTION_BITS_U64) | (bits & F64_FRACTION_MASK);
    let magnitude = if exponent >= F64_FRACTION_BITS_U64 as i32 {
        BigInt::from(significand) << ((exponent - F64_FRACTION_BITS_U64 as i32) as usize)
    } else {
        BigInt::from(significand >> (F64_FRACTION_BITS_U64 as i32 - exponent))
    };
    let integer = if value.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    };
    Ok(bigint_to_value(integer))
}

#[inline]
fn builtin_int_unsupported_argument(arg: Value) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "int() argument must be a string, a bytes-like object or a real number, not '{}'",
        get_type_name(arg)
    ))
}

#[inline]
fn int_protocol_result(result: Value, method_name: &'static str) -> Result<Value, BuiltinError> {
    if let Some(boolean) = result.as_bool() {
        return Ok(Value::int(i64::from(boolean)).expect("bool integer should be representable"));
    }

    let integer = value_to_bigint(result).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{method_name} returned non-int (type {})",
            result.type_name()
        ))
    })?;
    Ok(bigint_to_value(integer))
}

fn int_trunc_protocol_result(
    vm: &mut VirtualMachine,
    result: Value,
) -> Result<Value, BuiltinError> {
    if let Some(boolean) = result.as_bool() {
        return Ok(Value::int(i64::from(boolean)).expect("bool integer should be representable"));
    }
    if let Some(integer) = value_to_bigint(result) {
        return Ok(bigint_to_value(integer));
    }

    if let Some(indexed) = invoke_zero_arg_special_method(vm, result, "__index__")? {
        return int_protocol_result(indexed, "__index__");
    }

    Err(BuiltinError::TypeError(format!(
        "__trunc__ returned non-Integral (type {})",
        result.type_name()
    )))
}

enum IntTextArgument {
    Str(String),
    Bytes(Vec<u8>),
}

impl IntTextArgument {
    #[inline]
    fn raw_bytes(&self) -> &[u8] {
        match self {
            Self::Str(text) => text.as_bytes(),
            Self::Bytes(bytes) => bytes,
        }
    }

    fn invalid_literal(&self, base: u32) -> BuiltinError {
        match self {
            Self::Str(text) => BuiltinError::ValueError(format!(
                "invalid literal for int() with base {base}: {:?}",
                text
            )),
            Self::Bytes(bytes) => {
                let text = String::from_utf8_lossy(bytes);
                BuiltinError::ValueError(format!(
                    "invalid literal for int() with base {base}: b{:?}",
                    text
                ))
            }
        }
    }
}

#[inline]
fn int_text_argument(value: Value) -> Option<IntTextArgument> {
    if let Some(text) = value_to_owned_string(value) {
        return Some(IntTextArgument::Str(text));
    }

    let ptr = value.as_object_ptr()?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            Some(IntTextArgument::Bytes(bytes.as_bytes().to_vec()))
        }
        _ => None,
    }
}

#[inline]
fn int_buffer_argument(value: Value) -> Option<IntTextArgument> {
    value_as_memoryview_ref(value).map(|view| IntTextArgument::Bytes(view.as_bytes().to_vec()))
}

#[inline]
fn parse_int_base_argument(value: Value) -> Result<u32, BuiltinError> {
    let base = if let Some(integer) = prism_runtime::types::int::value_to_i64(value) {
        integer
    } else if let Some(boolean) = value.as_bool() {
        if boolean { 1 } else { 0 }
    } else {
        return Err(BuiltinError::TypeError(format!(
            "'{}' object cannot be interpreted as an integer",
            value.type_name()
        )));
    };

    if base == 0 || (2..=36).contains(&base) {
        Ok(base as u32)
    } else {
        Err(BuiltinError::ValueError(
            "int() base must be >= 2 and <= 36, or 0".to_string(),
        ))
    }
}

fn parse_int_text_argument(argument: &IntTextArgument, base: u32) -> Result<Value, BuiltinError> {
    let trimmed = trim_ascii_whitespace(argument.raw_bytes());
    if trimmed.is_empty() {
        return Err(argument.invalid_literal(base));
    }

    let (negative, digits) = match trimmed[0] {
        b'+' => (false, &trimmed[1..]),
        b'-' => (true, &trimmed[1..]),
        _ => (false, trimmed),
    };
    if digits.is_empty() {
        return Err(argument.invalid_literal(base));
    }

    let (resolved_base, digits, allow_leading_underscore) = resolve_int_parse_base(base, digits);
    let normalized = normalize_int_digits(digits, resolved_base, allow_leading_underscore)
        .ok_or_else(|| argument.invalid_literal(resolved_base))?;

    let mut value = BigInt::parse_bytes(&normalized, resolved_base)
        .ok_or_else(|| argument.invalid_literal(resolved_base))?;
    if negative {
        value = -value;
    }
    Ok(bigint_to_value(value))
}

#[inline]
fn trim_ascii_whitespace(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|byte| !is_python_ascii_whitespace(*byte))
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|byte| !is_python_ascii_whitespace(*byte))
        .map(|index| index + 1)
        .unwrap_or(start);
    &bytes[start..end]
}

#[inline]
fn is_python_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

#[inline]
fn resolve_int_parse_base(base: u32, digits: &[u8]) -> (u32, &[u8], bool) {
    if let Some((prefixed_base, prefix_len)) = int_base_prefix(digits) {
        if base == 0 {
            return (prefixed_base, &digits[prefix_len..], true);
        }
        if prefixed_base == base {
            return (base, &digits[prefix_len..], true);
        }
    }

    if base == 0 {
        (10, digits, false)
    } else {
        (base, digits, false)
    }
}

#[inline]
fn int_base_prefix(bytes: &[u8]) -> Option<(u32, usize)> {
    if bytes.len() < 2 || bytes[0] != b'0' {
        return None;
    }

    match bytes[1] {
        b'b' | b'B' => Some((2, 2)),
        b'o' | b'O' => Some((8, 2)),
        b'x' | b'X' => Some((16, 2)),
        _ => None,
    }
}

fn normalize_int_digits(
    digits: &[u8],
    base: u32,
    allow_leading_underscore: bool,
) -> Option<Vec<u8>> {
    let mut normalized = Vec::with_capacity(digits.len());
    let mut saw_digit = false;
    let mut previous_was_underscore = false;
    let mut leading_underscore_allowed = allow_leading_underscore;

    for &byte in digits {
        if byte == b'_' {
            if previous_was_underscore || (!saw_digit && !leading_underscore_allowed) {
                return None;
            }
            previous_was_underscore = true;
            leading_underscore_allowed = false;
            continue;
        }

        let digit = ascii_digit_value(byte)?;
        if digit >= base {
            return None;
        }

        normalized.push(byte.to_ascii_lowercase());
        saw_digit = true;
        previous_was_underscore = false;
        leading_underscore_allowed = false;
    }

    if !saw_digit || previous_was_underscore {
        return None;
    }

    Some(normalized)
}

#[inline]
fn ascii_digit_value(byte: u8) -> Option<u32> {
    match byte {
        b'0'..=b'9' => Some((byte - b'0') as u32),
        b'a'..=b'z' => Some((byte - b'a') as u32 + 10),
        b'A'..=b'Z' => Some((byte - b'A') as u32 + 10),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ByteOrder {
    Big,
    Little,
}

pub(crate) fn builtin_int_from_bytes(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    builtin_int_from_bytes_impl(None, args, keywords)
}

pub(crate) fn builtin_int_from_bytes_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    builtin_int_from_bytes_impl(Some(vm), args, keywords)
}

fn builtin_int_from_bytes_impl(
    vm: Option<&mut VirtualMachine>,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "from_bytes() descriptor requires a type receiver".to_string(),
        ));
    }

    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "from_bytes() takes at most 2 positional arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver_type = int_from_bytes_receiver_type(args[0])?;
    let mut bytes_arg = args.get(1).copied();
    let mut byteorder_arg = args.get(2).copied();
    let mut signed_arg: Option<Value> = None;

    for &(name, value) in keywords {
        match name {
            "bytes" => assign_from_bytes_keyword(&mut bytes_arg, value, 1, args.len(), "bytes")?,
            "byteorder" => {
                assign_from_bytes_keyword(&mut byteorder_arg, value, 2, args.len(), "byteorder")?
            }
            "signed" => {
                if signed_arg.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "from_bytes() got multiple values for argument 'signed'".to_string(),
                    ));
                }
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "from_bytes() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let bytes_arg = bytes_arg.ok_or_else(|| {
        BuiltinError::TypeError(
            "from_bytes() missing required argument 'bytes' (pos 1)".to_string(),
        )
    })?;
    let byteorder = match byteorder_arg {
        Some(value) => parse_from_bytes_byteorder(value)?,
        None => ByteOrder::Big,
    };
    let signed = signed_arg
        .map(crate::truthiness::is_truthy)
        .unwrap_or(false);

    let bytes = match vm {
        Some(vm) => value_to_byte_sequence_with_vm(vm, bytes_arg, "from_bytes() argument 1")?,
        None => value_to_byte_sequence(bytes_arg, "from_bytes() argument 1")?,
    };
    let value = decode_bigint_from_bytes(&bytes, byteorder, signed);

    match receiver_type {
        TypeId::BOOL => Ok(Value::bool(!value.is_zero())),
        TypeId::INT => Ok(bigint_to_value(value)),
        other => unreachable!("unexpected int.from_bytes receiver type: {other:?}"),
    }
}

pub(crate) fn builtin_int_to_bytes(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "to_bytes() descriptor requires an int receiver".to_string(),
        ));
    }

    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "to_bytes() takes at most 2 positional arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let value = int_to_bytes_receiver_value(args[0])?;
    let mut length_arg = args.get(1).copied();
    let mut byteorder_arg = args.get(2).copied();
    let mut signed_arg: Option<Value> = None;

    for &(name, value) in keywords {
        match name {
            "length" => assign_to_bytes_keyword(&mut length_arg, value, 1, args.len(), "length")?,
            "byteorder" => {
                assign_to_bytes_keyword(&mut byteorder_arg, value, 2, args.len(), "byteorder")?
            }
            "signed" => {
                if signed_arg.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "to_bytes() got multiple values for argument 'signed'".to_string(),
                    ));
                }
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "to_bytes() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let length = match length_arg {
        Some(value) => parse_to_bytes_length(value)?,
        None => 1,
    };
    let byteorder = match byteorder_arg {
        Some(value) => parse_to_bytes_byteorder(value)?,
        None => ByteOrder::Big,
    };
    let signed = signed_arg
        .map(crate::truthiness::is_truthy)
        .unwrap_or(false);

    let bytes = encode_bigint_to_bytes(&value, length, byteorder, signed)?;
    Ok(crate::alloc_managed_value(BytesObject::from_vec(bytes)))
}

fn int_from_bytes_receiver_type(receiver: Value) -> Result<TypeId, BuiltinError> {
    let class_type = class_value_to_type_id(receiver).ok_or_else(|| {
        BuiltinError::TypeError("from_bytes() descriptor requires a type receiver".to_string())
    })?;

    match class_type {
        TypeId::INT | TypeId::BOOL => Ok(class_type),
        _ if class_value_is_subtype(receiver, TypeId::INT) => Err(BuiltinError::NotImplemented(
            "from_bytes() for int subclasses is not implemented yet".to_string(),
        )),
        _ => Err(BuiltinError::TypeError(
            "from_bytes() requires the built-in int or bool type".to_string(),
        )),
    }
}

fn int_to_bytes_receiver_value(receiver: Value) -> Result<BigInt, BuiltinError> {
    if let Some(boolean) = receiver.as_bool() {
        return Ok(BigInt::from(u8::from(boolean)));
    }

    value_to_bigint(receiver).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor 'int.to_bytes' requires an 'int' object but received '{}'",
            receiver.type_name()
        ))
    })
}

fn assign_from_bytes_keyword(
    slot: &mut Option<Value>,
    value: Value,
    positional_index: usize,
    positional_len: usize,
    name: &str,
) -> Result<(), BuiltinError> {
    if positional_len > positional_index {
        return Err(BuiltinError::TypeError(format!(
            "from_bytes() got multiple values for argument '{}'",
            name
        )));
    }

    if slot.replace(value).is_some() {
        return Err(BuiltinError::TypeError(format!(
            "from_bytes() got multiple values for argument '{}'",
            name
        )));
    }

    Ok(())
}

fn assign_to_bytes_keyword(
    slot: &mut Option<Value>,
    value: Value,
    positional_index: usize,
    positional_len: usize,
    name: &str,
) -> Result<(), BuiltinError> {
    if positional_len > positional_index {
        return Err(BuiltinError::TypeError(format!(
            "to_bytes() got multiple values for argument '{}'",
            name
        )));
    }

    if slot.replace(value).is_some() {
        return Err(BuiltinError::TypeError(format!(
            "to_bytes() got multiple values for argument '{}'",
            name
        )));
    }

    Ok(())
}

fn parse_from_bytes_byteorder(value: Value) -> Result<ByteOrder, BuiltinError> {
    parse_byteorder_arg(value, "from_bytes")
}

fn parse_to_bytes_byteorder(value: Value) -> Result<ByteOrder, BuiltinError> {
    parse_byteorder_arg(value, "to_bytes")
}

fn parse_byteorder_arg(value: Value, fn_name: &str) -> Result<ByteOrder, BuiltinError> {
    let Some(byteorder) = value_to_owned_string(value) else {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() argument 'byteorder' must be str, not {}",
            value.type_name()
        )));
    };

    match byteorder.as_str() {
        "big" => Ok(ByteOrder::Big),
        "little" => Ok(ByteOrder::Little),
        _ => Err(BuiltinError::ValueError(
            "byteorder must be either 'little' or 'big'".to_string(),
        )),
    }
}

fn parse_to_bytes_length(value: Value) -> Result<usize, BuiltinError> {
    let Some(length) = value_to_bigint(value).or_else(|| {
        value
            .as_bool()
            .map(|boolean| BigInt::from(u8::from(boolean)))
    }) else {
        return Err(BuiltinError::TypeError(format!(
            "'{}' object cannot be interpreted as an integer",
            value.type_name()
        )));
    };

    if length.sign() == Sign::Minus {
        return Err(BuiltinError::ValueError(
            "length argument must be non-negative".to_string(),
        ));
    }

    usize::try_from(&length)
        .map_err(|_| BuiltinError::OverflowError("int too big to convert".to_string()))
}

fn value_to_byte_sequence(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    if let Some(ptr) = value.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::BYTES | TypeId::BYTEARRAY => {
                return Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec());
            }
            _ => {}
        }
    }

    let values = iter_values(value).map_err(|_| {
        BuiltinError::TypeError(format!(
            "{context} must be a bytes-like object or iterable of integers"
        ))
    })?;

    let mut bytes = Vec::with_capacity(values.len());
    for item in values {
        bytes.push(value_to_single_byte(item, context)?);
    }
    Ok(bytes)
}

fn value_to_byte_sequence_with_vm(
    vm: &mut VirtualMachine,
    value: Value,
    context: &str,
) -> Result<Vec<u8>, BuiltinError> {
    if let Some(ptr) = value.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::BYTES | TypeId::BYTEARRAY => {
                return Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec());
            }
            _ => {}
        }
    }

    let values = iter_values_with_vm(vm, value)?;
    let mut bytes = Vec::with_capacity(values.len());
    for item in values {
        bytes.push(value_to_single_byte(item, context)?);
    }
    Ok(bytes)
}

fn value_to_single_byte(value: Value, context: &str) -> Result<u8, BuiltinError> {
    if let Some(number) = value_to_bigint(value) {
        if (BigInt::from(0_u8)..=BigInt::from(u8::MAX)).contains(&number) {
            return Ok(number
                .try_into()
                .expect("validated byte-sized bigint should convert to u8"));
        }
    } else if let Some(boolean) = value.as_bool() {
        return Ok(u8::from(boolean));
    }

    Err(BuiltinError::ValueError(format!(
        "{context} must yield integers in range(0, 256)"
    )))
}

fn decode_bigint_from_bytes(bytes: &[u8], byteorder: ByteOrder, signed: bool) -> BigInt {
    match (byteorder, signed) {
        (ByteOrder::Big, true) => BigInt::from_signed_bytes_be(bytes),
        (ByteOrder::Little, true) => BigInt::from_signed_bytes_le(bytes),
        (ByteOrder::Big, false) => BigInt::from_bytes_be(Sign::Plus, bytes),
        (ByteOrder::Little, false) => BigInt::from_bytes_le(Sign::Plus, bytes),
    }
}

fn encode_bigint_to_bytes(
    value: &BigInt,
    length: usize,
    byteorder: ByteOrder,
    signed: bool,
) -> Result<Vec<u8>, BuiltinError> {
    if value.is_zero() {
        return Ok(vec![0; length]);
    }

    if !signed && value.sign() == Sign::Minus {
        return Err(BuiltinError::OverflowError(
            "can't convert negative int to unsigned".to_string(),
        ));
    }

    let mut encoded = if signed {
        match byteorder {
            ByteOrder::Big => value.to_signed_bytes_be(),
            ByteOrder::Little => value.to_signed_bytes_le(),
        }
    } else {
        match byteorder {
            ByteOrder::Big => value.to_bytes_be().1,
            ByteOrder::Little => value.to_bytes_le().1,
        }
    };

    if encoded.len() > length {
        return Err(BuiltinError::OverflowError(
            "int too big to convert".to_string(),
        ));
    }

    let padding_len = length - encoded.len();
    if padding_len == 0 {
        return Ok(encoded);
    }

    let pad = if signed && value.sign() == Sign::Minus {
        0xFF
    } else {
        0x00
    };

    match byteorder {
        ByteOrder::Big => {
            let mut padded = vec![pad; padding_len];
            padded.extend_from_slice(&encoded);
            Ok(padded)
        }
        ByteOrder::Little => {
            encoded.resize(length, pad);
            Ok(encoded)
        }
    }
}

enum FloatTextArgument {
    Str(String),
    Bytes(Vec<u8>),
}

impl FloatTextArgument {
    #[inline]
    fn ascii_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Str(text) if text.is_ascii() => Some(text.as_bytes()),
            Self::Str(_) => None,
            Self::Bytes(bytes) => Some(bytes),
        }
    }

    fn invalid_literal(&self) -> BuiltinError {
        match self {
            Self::Str(text) => BuiltinError::ValueError(format!(
                "could not convert string to float: {}",
                crate::builtins::quote_python_string(text)
            )),
            Self::Bytes(bytes) => BuiltinError::ValueError(format!(
                "could not convert string to float: {}",
                crate::builtins::quote_python_bytes(bytes)
            )),
        }
    }
}

fn normalize_unicode_float_text(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            normalized.push(ch);
        } else if let Some(digit) = prism_core::python_unicode::python_decimal_digit(ch) {
            normalized.push((b'0' + digit) as char);
        } else if ch.is_whitespace() {
            normalized.push(' ');
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

#[inline]
fn float_text_argument(value: Value) -> Result<Option<FloatTextArgument>, BuiltinError> {
    if let Some(text) = value_to_owned_string(value) {
        return Ok(Some(FloatTextArgument::Str(text)));
    }

    if let Some(view) = value_as_memoryview_ref(value) {
        if view.released() {
            return Err(BuiltinError::ValueError(
                "operation forbidden on released memoryview object".to_string(),
            ));
        }
        return Ok(Some(FloatTextArgument::Bytes(view.as_bytes().to_vec())));
    }

    if let Some(bytes) = value_as_bytes_ref(value) {
        return Ok(Some(FloatTextArgument::Bytes(bytes.as_bytes().to_vec())));
    }

    if let Some(bytes) = crate::stdlib::array::value_as_array_bytes(value)? {
        return Ok(Some(FloatTextArgument::Bytes(bytes)));
    }

    Ok(None)
}

#[inline]
fn normalize_float_special_value(bytes: &[u8]) -> Option<f64> {
    let (negative, rest) = match bytes.first().copied() {
        Some(b'+') => (false, &bytes[1..]),
        Some(b'-') => (true, &bytes[1..]),
        _ => (false, bytes),
    };

    if rest.eq_ignore_ascii_case(b"inf") || rest.eq_ignore_ascii_case(b"infinity") {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }

    if rest.eq_ignore_ascii_case(b"nan") {
        return Some(f64::from_bits(if negative {
            0xfff8_0000_0000_0000
        } else {
            0x7ff8_0000_0000_0000
        }));
    }

    None
}

fn normalize_float_literal(bytes: &[u8]) -> Option<String> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Prev {
        Start,
        Sign,
        Digit,
        Underscore,
        Dot,
        Exp,
    }

    let mut normalized = String::with_capacity(bytes.len());
    let mut prev = Prev::Start;
    let mut saw_mantissa_digit = false;
    let mut saw_exponent_digit = false;
    let mut saw_dot = false;
    let mut in_exponent = false;

    for (index, &byte) in bytes.iter().enumerate() {
        match byte {
            b'0'..=b'9' => {
                normalized.push(byte as char);
                if in_exponent {
                    saw_exponent_digit = true;
                } else {
                    saw_mantissa_digit = true;
                }
                prev = Prev::Digit;
            }
            b'_' => {
                if prev != Prev::Digit {
                    return None;
                }
                match bytes.get(index + 1).copied() {
                    Some(b'0'..=b'9') => prev = Prev::Underscore,
                    _ => return None,
                }
            }
            b'.' => {
                if in_exponent || saw_dot {
                    return None;
                }
                normalized.push('.');
                saw_dot = true;
                prev = Prev::Dot;
            }
            b'e' | b'E' => {
                if in_exponent || !saw_mantissa_digit {
                    return None;
                }
                normalized.push('e');
                in_exponent = true;
                saw_exponent_digit = false;
                prev = Prev::Exp;
            }
            b'+' | b'-' => {
                if index == 0 || prev == Prev::Exp {
                    normalized.push(byte as char);
                    prev = Prev::Sign;
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }

    if !saw_mantissa_digit
        || matches!(
            prev,
            Prev::Start | Prev::Sign | Prev::Underscore | Prev::Exp
        )
    {
        return None;
    }
    if in_exponent && !saw_exponent_digit {
        return None;
    }

    Some(normalized)
}

fn parse_float_text_argument(argument: &FloatTextArgument) -> Result<Value, BuiltinError> {
    let unicode_normalized;
    let raw_bytes = match argument.ascii_bytes() {
        Some(bytes) => bytes,
        None => {
            let FloatTextArgument::Str(text) = argument else {
                unreachable!("bytes-backed float text is always ASCII-normalized directly")
            };
            unicode_normalized = normalize_unicode_float_text(text);
            unicode_normalized.as_bytes()
        }
    };

    let trimmed = trim_ascii_whitespace(raw_bytes);
    if trimmed.is_empty() {
        return Err(argument.invalid_literal());
    }

    if let Some(value) = normalize_float_special_value(trimmed) {
        return Ok(Value::float(value));
    }

    let normalized = normalize_float_literal(trimmed).ok_or_else(|| argument.invalid_literal())?;
    normalized
        .parse::<f64>()
        .map(Value::float)
        .map_err(|_| argument.invalid_literal())
}

#[inline]
fn builtin_float_unsupported_argument(arg: Value) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "float() argument must be a string or a real number, not '{}'",
        get_type_name(arg)
    ))
}

#[inline]
fn builtin_float_base(arg: Value) -> Result<Option<Value>, BuiltinError> {
    if let Some(text_arg) = float_text_argument(arg)? {
        return parse_float_text_argument(&text_arg).map(Some);
    }

    if let Some(f) = prism_runtime::types::float::value_to_f64(arg) {
        return Ok(Some(Value::float(f)));
    }
    if let Some(i) = arg.as_int() {
        return Ok(Some(Value::float(i as f64)));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Some(Value::float(if b { 1.0 } else { 0.0 })));
    }

    Ok(None)
}

#[inline]
fn float_from_index_value(value: Value) -> Result<Value, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(Value::float(if boolean { 1.0 } else { 0.0 }));
    }
    if let Some(integer) = value.as_int() {
        return Ok(Value::float(integer as f64));
    }

    let bigint = value_to_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "__index__ returned non-int (type {})",
            value.type_name()
        ))
    })?;
    let float = bigint
        .to_f64()
        .filter(|float| float.is_finite())
        .ok_or_else(|| {
            BuiltinError::OverflowError("int too large to convert to float".to_string())
        })?;
    Ok(Value::float(float))
}

fn runtime_type_name(value: Value) -> String {
    if let Some(ptr) = value.as_object_ptr() {
        let type_id = crate::ops::objects::extract_type_id(ptr);
        if type_id.raw() >= TypeId::FIRST_USER_TYPE
            && let Some(class) = global_class(ClassId(type_id.raw()))
        {
            return class.name().as_str().to_string();
        }
    }

    get_type_name(value).to_string()
}

fn float_subclass_return_warning(receiver: Value, result: Value) -> String {
    format!(
        "{}.__float__ returned non-float (type {}).  The ability to return an instance of a strict subclass of float is deprecated, and may be removed in a future version of Python.",
        runtime_type_name(receiver),
        runtime_type_name(result)
    )
}

/// Builtin float constructor.
pub fn builtin_float(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::float(0.0));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "float() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    builtin_float_base(args[0])?.ok_or_else(|| builtin_float_unsupported_argument(args[0]))
}

pub fn builtin_float_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::float(0.0));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "float() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let arg = args[0];
    if let Some(value) = builtin_float_exact_base(arg)? {
        return Ok(value);
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__float__")? {
        if let Some(float) = prism_runtime::types::float::value_to_f64(result) {
            if result.as_float().is_none() {
                crate::stdlib::_warnings::emit_deprecation_warning(
                    vm,
                    &float_subclass_return_warning(arg, result),
                )
                .map_err(BuiltinError::Raised)?;
            }
            return Ok(Value::float(float));
        }
        return Err(BuiltinError::TypeError(format!(
            "__float__ returned non-float (type {})",
            result.type_name()
        )));
    }

    if let Some(text_arg) = float_text_argument(arg)? {
        return parse_float_text_argument(&text_arg);
    }

    if let Some(float) = prism_runtime::types::float::value_to_f64(arg) {
        return Ok(Value::float(float));
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__index__")? {
        return float_from_index_value(result);
    }

    Err(builtin_float_unsupported_argument(arg))
}

#[inline]
fn builtin_float_exact_base(arg: Value) -> Result<Option<Value>, BuiltinError> {
    if let Some(f) = arg.as_float() {
        return Ok(Some(Value::float(f)));
    }
    if let Some(i) = arg.as_int() {
        return Ok(Some(Value::float(i as f64)));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Some(Value::float(if b { 1.0 } else { 0.0 })));
    }
    if is_exact_float_text_argument(arg)
        && let Some(text_arg) = float_text_argument(arg)?
    {
        return parse_float_text_argument(&text_arg).map(Some);
    }

    Ok(None)
}

#[inline]
fn is_exact_float_text_argument(value: Value) -> bool {
    if value.is_string() {
        return true;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    matches!(
        crate::ops::objects::extract_type_id(ptr),
        TypeId::BYTES | TypeId::BYTEARRAY | TypeId::MEMORYVIEW
    )
}

const F64_FRACTION_BITS: u32 = 52;
const F64_MIN_NORMAL_EXP: i64 = -1022;
const F64_MIN_SUBNORMAL_EXP: i64 = -1074;
const F64_MAX_NORMAL_EXP: i64 = 1023;
const HEX_FLOAT_EXP_LIMIT: i64 = 1_000_000;

struct HexFloatParts {
    negative: bool,
    mantissa: BigInt,
    binary_exp: i64,
}

/// Builtin implementation backing `float.fromhex`.
pub(crate) fn builtin_float_fromhex_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "float.fromhex() takes exactly one argument ({given} given)"
        )));
    }

    if !class_value_is_subtype(args[0], TypeId::FLOAT) {
        return Err(BuiltinError::TypeError(
            "descriptor 'fromhex' for 'float' objects doesn't apply to a non-float type"
                .to_string(),
        ));
    }

    let Some(text) = value_to_owned_string(args[1]) else {
        return Err(BuiltinError::TypeError(format!(
            "float.fromhex() argument must be str, not {}",
            args[1].type_name()
        )));
    };
    let float = parse_float_fromhex_text(&text)?;

    if class_value_to_type_id(args[0]) == Some(TypeId::FLOAT) {
        return Ok(Value::float(float));
    }

    invoke_callable_value(vm, args[0], &[Value::float(float)])
        .map_err(runtime_error_to_builtin_error)
}

fn parse_float_fromhex_text(text: &str) -> Result<f64, BuiltinError> {
    let trimmed = trim_ascii_whitespace(text.as_bytes());
    if trimmed.is_empty() {
        return Err(invalid_hex_float_error(text));
    }

    if let Some(value) = normalize_float_special_value(trimmed) {
        return Ok(value);
    }

    let parts = parse_hex_float_parts(trimmed).ok_or_else(|| invalid_hex_float_error(text))?;
    hex_float_parts_to_f64(parts)
}

fn parse_hex_float_parts(bytes: &[u8]) -> Option<HexFloatParts> {
    let (negative, mut rest) = match bytes.first().copied()? {
        b'+' => (false, &bytes[1..]),
        b'-' => (true, &bytes[1..]),
        _ => (false, bytes),
    };

    if rest.len() >= 2 && rest[0] == b'0' && matches!(rest[1], b'x' | b'X') {
        rest = &rest[2..];
    }
    if rest.is_empty() {
        return None;
    }

    let mut mantissa = BigInt::zero();
    let mut saw_digit = false;
    let mut saw_dot = false;
    let mut fraction_digits = 0_i64;
    let mut index = 0_usize;

    while index < rest.len() {
        let byte = rest[index];
        if matches!(byte, b'p' | b'P') {
            break;
        }
        if byte == b'.' {
            if saw_dot {
                return None;
            }
            saw_dot = true;
            index += 1;
            continue;
        }

        let digit = hex_digit_value(byte)?;
        mantissa = (mantissa << 4_usize) + BigInt::from(digit);
        saw_digit = true;
        if saw_dot {
            fraction_digits = fraction_digits.saturating_add(1).min(HEX_FLOAT_EXP_LIMIT);
        }
        index += 1;
    }

    if !saw_digit {
        return None;
    }

    let mut exponent = 0_i64;
    if index < rest.len() {
        if !matches!(rest[index], b'p' | b'P') {
            return None;
        }
        index += 1;
        let exponent_negative = match rest.get(index).copied() {
            Some(b'+') => {
                index += 1;
                false
            }
            Some(b'-') => {
                index += 1;
                true
            }
            _ => false,
        };
        let exponent_start = index;
        while index < rest.len() && rest[index].is_ascii_digit() {
            index += 1;
        }
        if exponent_start == index || index != rest.len() {
            return None;
        }
        exponent =
            parse_saturating_decimal_exponent(&rest[exponent_start..index], exponent_negative)?;
    }

    let fractional_bits = fraction_digits.saturating_mul(4).min(HEX_FLOAT_EXP_LIMIT);
    let binary_exp = exponent
        .saturating_sub(fractional_bits)
        .clamp(-HEX_FLOAT_EXP_LIMIT, HEX_FLOAT_EXP_LIMIT);

    Some(HexFloatParts {
        negative,
        mantissa,
        binary_exp,
    })
}

fn parse_saturating_decimal_exponent(bytes: &[u8], negative: bool) -> Option<i64> {
    let mut value = 0_i64;
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value
            .saturating_mul(10)
            .saturating_add(i64::from(byte - b'0'));
        if value > HEX_FLOAT_EXP_LIMIT {
            return Some(if negative {
                -HEX_FLOAT_EXP_LIMIT
            } else {
                HEX_FLOAT_EXP_LIMIT
            });
        }
    }
    Some(if negative { -value } else { value })
}

#[inline]
fn hex_digit_value(byte: u8) -> Option<u32> {
    let digit = ascii_digit_value(byte)?;
    (digit < 16).then_some(digit)
}

fn hex_float_parts_to_f64(parts: HexFloatParts) -> Result<f64, BuiltinError> {
    if parts.mantissa.is_zero() {
        let zero = if parts.negative { -0.0 } else { 0.0 };
        return Ok(zero);
    }

    let mantissa_bits = i64::try_from(parts.mantissa.bits()).unwrap_or(i64::MAX);
    let floor_log2 = mantissa_bits
        .saturating_sub(1)
        .saturating_add(parts.binary_exp);

    if floor_log2 > F64_MAX_NORMAL_EXP + 1 {
        return Err(hex_float_overflow_error());
    }
    if floor_log2 < F64_MIN_SUBNORMAL_EXP - 2 {
        let zero = if parts.negative { -0.0 } else { 0.0 };
        return Ok(zero);
    }

    let sign_bit = if parts.negative { 1_u64 << 63 } else { 0 };
    if floor_log2 < F64_MIN_NORMAL_EXP {
        return hex_float_subnormal_to_f64(&parts.mantissa, parts.binary_exp, sign_bit);
    }

    hex_float_normal_to_f64(&parts.mantissa, parts.binary_exp, floor_log2, sign_bit)
}

fn hex_float_subnormal_to_f64(
    mantissa: &BigInt,
    binary_exp: i64,
    sign_bit: u64,
) -> Result<f64, BuiltinError> {
    let units = round_scaled_to_even(mantissa, binary_exp - F64_MIN_SUBNORMAL_EXP);
    let normal_threshold = BigInt::from(1_u64) << F64_FRACTION_BITS as usize;
    if units.is_zero() {
        return Ok(f64::from_bits(sign_bit));
    }
    if units < normal_threshold {
        let fraction = units
            .to_u64()
            .expect("subnormal significand should fit in f64 fraction bits");
        return Ok(f64::from_bits(sign_bit | fraction));
    }
    if units == normal_threshold {
        return Ok(f64::from_bits(sign_bit | (1_u64 << F64_FRACTION_BITS)));
    }

    hex_float_normal_to_f64(mantissa, binary_exp, F64_MIN_NORMAL_EXP, sign_bit)
}

fn hex_float_normal_to_f64(
    mantissa: &BigInt,
    binary_exp: i64,
    mut exponent: i64,
    sign_bit: u64,
) -> Result<f64, BuiltinError> {
    let unit_exp = exponent - i64::from(F64_FRACTION_BITS);
    let mut significand = round_scaled_to_even(mantissa, binary_exp - unit_exp);
    let hidden_bit = BigInt::from(1_u64) << F64_FRACTION_BITS as usize;
    let carry_bit = BigInt::from(1_u64) << (F64_FRACTION_BITS as usize + 1);

    if significand == carry_bit {
        significand = hidden_bit.clone();
        exponent += 1;
    }
    if exponent > F64_MAX_NORMAL_EXP {
        return Err(hex_float_overflow_error());
    }

    let fraction = (significand - hidden_bit)
        .to_u64()
        .expect("normal significand should fit in f64 fraction bits");
    let exponent_bits = ((exponent + 1023) as u64) << F64_FRACTION_BITS;
    Ok(f64::from_bits(sign_bit | exponent_bits | fraction))
}

fn round_scaled_to_even(mantissa: &BigInt, shift: i64) -> BigInt {
    if shift >= 0 {
        return mantissa << usize::try_from(shift).unwrap_or(usize::MAX);
    }

    let right_shift = usize::try_from(-shift).unwrap_or(usize::MAX);
    let quotient = mantissa >> right_shift;
    let remainder = mantissa - (&quotient << right_shift);
    if remainder.is_zero() {
        return quotient;
    }

    let halfway = BigInt::from(1_u8) << (right_shift - 1);
    let round_up = match remainder.cmp(&halfway) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => (&quotient & BigInt::from(1_u8)) != BigInt::zero(),
    };

    if round_up {
        quotient + BigInt::from(1_u8)
    } else {
        quotient
    }
}

fn invalid_hex_float_error(text: &str) -> BuiltinError {
    BuiltinError::ValueError(format!(
        "invalid hexadecimal floating-point string: {:?}",
        text
    ))
}

#[inline]
fn hex_float_overflow_error() -> BuiltinError {
    BuiltinError::OverflowError("hexadecimal value too large to represent as a float".to_string())
}

#[inline]
pub(crate) fn native_float_format_description() -> &'static str {
    if cfg!(target_endian = "little") {
        "IEEE, little-endian"
    } else {
        "IEEE, big-endian"
    }
}

/// Builtin implementation backing `float.__getformat__`.
pub(crate) fn builtin_float_getformat(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "float.__getformat__() takes exactly 1 argument ({} given)",
            given
        )));
    }

    let receiver = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(
            "descriptor '__getformat__' for 'float' objects doesn't apply to a non-type object"
                .to_string(),
        )
    })?;
    if crate::ops::objects::extract_type_id(receiver) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "descriptor '__getformat__' for 'float' objects doesn't apply to a non-type object"
                .to_string(),
        ));
    }

    let Some(kind) = value_to_owned_string(args[1]) else {
        return Err(BuiltinError::TypeError(format!(
            "float.__getformat__() argument 1 must be str, not {}",
            args[1].type_name()
        )));
    };

    match kind.as_str() {
        "double" | "float" => Ok(Value::string(intern(native_float_format_description()))),
        _ => Err(BuiltinError::ValueError(
            "__getformat__() argument 1 must be 'double' or 'float'".to_string(),
        )),
    }
}
