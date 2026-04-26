//! Old-style Python string (`%`) formatting.
//!
//! This module implements the runtime semantics for `"..." % value`, including
//! positional tuples, mapping lookups, escaped percent signs, and the common
//! conversion types used across the CPython standard library.

use super::BuiltinError;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::{BytesObject, value_as_bytes_ref};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use prism_runtime::types::tuple::TupleObject;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PercentSpec {
    mapping_key: Option<String>,
    alternate: bool,
    zero_pad: bool,
    left_adjust: bool,
    sign_plus: bool,
    sign_space: bool,
    width_from_arg: bool,
    width: Option<usize>,
    precision_from_arg: bool,
    precision: Option<usize>,
    conversion: char,
}

impl PercentSpec {
    fn parse(template: &str, start: usize) -> Result<(Self, usize), BuiltinError> {
        let bytes = template.as_bytes();
        let mut index = start;
        let mut spec = Self {
            mapping_key: None,
            alternate: false,
            zero_pad: false,
            left_adjust: false,
            sign_plus: false,
            sign_space: false,
            width_from_arg: false,
            width: None,
            precision_from_arg: false,
            precision: None,
            conversion: '\0',
        };

        if index >= bytes.len() {
            return Err(BuiltinError::ValueError("incomplete format".to_string()));
        }

        if bytes[index] == b'(' {
            index += 1;
            let key_start = index;
            while index < bytes.len() && bytes[index] != b')' {
                index += 1;
            }
            if index >= bytes.len() {
                return Err(BuiltinError::ValueError(
                    "incomplete format key".to_string(),
                ));
            }
            spec.mapping_key = Some(template[key_start..index].to_string());
            index += 1;
        }

        while index < bytes.len() {
            match bytes[index] {
                b'#' => spec.alternate = true,
                b'0' => spec.zero_pad = true,
                b'-' => spec.left_adjust = true,
                b'+' => spec.sign_plus = true,
                b' ' => spec.sign_space = true,
                _ => break,
            }
            index += 1;
        }

        if index < bytes.len() && bytes[index] == b'*' {
            spec.width_from_arg = true;
            index += 1;
        } else {
            let width_start = index;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
            if index > width_start {
                spec.width = Some(
                    template[width_start..index]
                        .parse()
                        .map_err(|_| BuiltinError::ValueError("invalid width".to_string()))?,
                );
            }
        }

        if index < bytes.len() && bytes[index] == b'.' {
            index += 1;
            if index < bytes.len() && bytes[index] == b'*' {
                spec.precision_from_arg = true;
                index += 1;
            } else {
                let precision_start = index;
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
                let precision_digits = &template[precision_start..index];
                spec.precision = Some(if precision_digits.is_empty() {
                    0
                } else {
                    precision_digits
                        .parse()
                        .map_err(|_| BuiltinError::ValueError("invalid precision".to_string()))?
                });
            }
        }

        while index < bytes.len() && matches!(bytes[index], b'h' | b'l' | b'L') {
            index += 1;
        }

        if index >= bytes.len() {
            return Err(BuiltinError::ValueError("incomplete format".to_string()));
        }

        let conversion = bytes[index] as char;
        if !matches!(
            conversion,
            's' | 'r'
                | 'a'
                | 'c'
                | 'd'
                | 'i'
                | 'u'
                | 'o'
                | 'x'
                | 'X'
                | 'e'
                | 'E'
                | 'f'
                | 'F'
                | 'g'
                | 'G'
        ) {
            return Err(BuiltinError::ValueError(format!(
                "unsupported format character '{}' (0x{:02x})",
                conversion, bytes[index]
            )));
        }
        spec.conversion = conversion;
        Ok((spec, index + 1))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BytesPercentSpec {
    mapping_key: Option<String>,
    alternate: bool,
    zero_pad: bool,
    left_adjust: bool,
    sign_plus: bool,
    sign_space: bool,
    width_from_arg: bool,
    width: Option<usize>,
    precision_from_arg: bool,
    precision: Option<usize>,
    conversion: u8,
}

impl BytesPercentSpec {
    fn parse(template: &[u8], start: usize) -> Result<(Self, usize), BuiltinError> {
        let mut index = start;
        let mut spec = Self {
            mapping_key: None,
            alternate: false,
            zero_pad: false,
            left_adjust: false,
            sign_plus: false,
            sign_space: false,
            width_from_arg: false,
            width: None,
            precision_from_arg: false,
            precision: None,
            conversion: b'\0',
        };

        if index >= template.len() {
            return Err(BuiltinError::ValueError("incomplete format".to_string()));
        }

        if template[index] == b'(' {
            index += 1;
            let key_start = index;
            while index < template.len() && template[index] != b')' {
                index += 1;
            }
            if index >= template.len() {
                return Err(BuiltinError::ValueError(
                    "incomplete format key".to_string(),
                ));
            }
            let key = std::str::from_utf8(&template[key_start..index])
                .map_err(|_| {
                    BuiltinError::ValueError(
                        "bytes format keys must be valid ASCII or UTF-8".to_string(),
                    )
                })?
                .to_string();
            spec.mapping_key = Some(key);
            index += 1;
        }

        while index < template.len() {
            match template[index] {
                b'#' => spec.alternate = true,
                b'0' => spec.zero_pad = true,
                b'-' => spec.left_adjust = true,
                b'+' => spec.sign_plus = true,
                b' ' => spec.sign_space = true,
                _ => break,
            }
            index += 1;
        }

        if index < template.len() && template[index] == b'*' {
            spec.width_from_arg = true;
            index += 1;
        } else {
            let width_start = index;
            while index < template.len() && template[index].is_ascii_digit() {
                index += 1;
            }
            if index > width_start {
                spec.width = Some(parse_ascii_usize(&template[width_start..index], "width")?);
            }
        }

        if index < template.len() && template[index] == b'.' {
            index += 1;
            if index < template.len() && template[index] == b'*' {
                spec.precision_from_arg = true;
                index += 1;
            } else {
                let precision_start = index;
                while index < template.len() && template[index].is_ascii_digit() {
                    index += 1;
                }
                spec.precision = Some(if index == precision_start {
                    0
                } else {
                    parse_ascii_usize(&template[precision_start..index], "precision")?
                });
            }
        }

        while index < template.len() && matches!(template[index], b'h' | b'l' | b'L') {
            index += 1;
        }

        if index >= template.len() {
            return Err(BuiltinError::ValueError("incomplete format".to_string()));
        }

        let conversion = template[index];
        if !matches!(
            conversion,
            b'b' | b's'
                | b'r'
                | b'a'
                | b'c'
                | b'd'
                | b'i'
                | b'u'
                | b'o'
                | b'x'
                | b'X'
                | b'e'
                | b'E'
                | b'f'
                | b'F'
                | b'g'
                | b'G'
        ) {
            return Err(BuiltinError::ValueError(format!(
                "unsupported format character '{}' (0x{:02x})",
                conversion as char, conversion
            )));
        }
        spec.conversion = conversion;
        Ok((spec, index + 1))
    }

    fn as_text_spec(&self) -> PercentSpec {
        PercentSpec {
            mapping_key: None,
            alternate: self.alternate,
            zero_pad: self.zero_pad,
            left_adjust: self.left_adjust,
            sign_plus: self.sign_plus,
            sign_space: self.sign_space,
            width_from_arg: false,
            width: self.width,
            precision_from_arg: false,
            precision: self.precision,
            conversion: self.conversion as char,
        }
    }
}

fn parse_ascii_usize(bytes: &[u8], name: &'static str) -> Result<usize, BuiltinError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| BuiltinError::ValueError(format!("invalid {name}")))?;
    text.parse()
        .map_err(|_| BuiltinError::ValueError(format!("invalid {name}")))
}

struct PositionalArgs<'a> {
    tuple: Option<&'a [Value]>,
    index: usize,
    single: Option<Value>,
    single_consumed: bool,
}

impl<'a> PositionalArgs<'a> {
    fn new(arguments: Value) -> Self {
        if let Some(ptr) = arguments.as_object_ptr() {
            if crate::ops::objects::extract_type_id(ptr) == TypeId::TUPLE {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                return Self {
                    tuple: Some(tuple.as_slice()),
                    index: 0,
                    single: None,
                    single_consumed: false,
                };
            }
        }

        Self {
            tuple: None,
            index: 0,
            single: Some(arguments),
            single_consumed: false,
        }
    }

    fn next(&mut self) -> Result<Value, BuiltinError> {
        if let Some(tuple) = self.tuple {
            let value = tuple.get(self.index).copied().ok_or_else(|| {
                BuiltinError::TypeError("not enough arguments for format string".to_string())
            })?;
            self.index += 1;
            return Ok(value);
        }

        if self.single_consumed {
            return Err(BuiltinError::TypeError(
                "not enough arguments for format string".to_string(),
            ));
        }

        self.single_consumed = true;
        Ok(self.single.unwrap_or(Value::none()))
    }

    fn finish(&self) -> Result<(), BuiltinError> {
        if let Some(tuple) = self.tuple {
            if self.index < tuple.len() {
                return Err(BuiltinError::TypeError(
                    "not all arguments converted during string formatting".to_string(),
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn percent_format_string(
    template: &str,
    arguments: Value,
) -> Result<Value, BuiltinError> {
    let mut rendered = String::with_capacity(template.len());
    let mut positional = PositionalArgs::new(arguments);
    let bytes = template.as_bytes();
    let mut literal_start = 0usize;
    let mut index = 0usize;

    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }

        rendered.push_str(&template[literal_start..index]);
        index += 1;
        if index >= bytes.len() {
            return Err(BuiltinError::ValueError("incomplete format".to_string()));
        }

        if bytes[index] == b'%' {
            rendered.push('%');
            index += 1;
            literal_start = index;
            continue;
        }

        let (mut spec, next_index) = PercentSpec::parse(template, index)?;
        index = next_index;

        if spec.width_from_arg {
            let width = positional_int_arg(&mut positional)?;
            if width < 0 {
                spec.left_adjust = true;
                spec.width = Some(
                    usize::try_from(width.unsigned_abs())
                        .map_err(|_| BuiltinError::OverflowError("width too large".to_string()))?,
                );
            } else {
                spec.width = Some(width as usize);
            }
        }

        if spec.precision_from_arg {
            let precision = positional_int_arg(&mut positional)?;
            spec.precision = usize::try_from(precision).ok();
        }

        let value = if let Some(key) = spec.mapping_key.as_deref() {
            mapping_argument(arguments, key)?
        } else {
            positional.next()?
        };

        rendered.push_str(&format_argument(value, &spec)?);
        literal_start = index;
    }

    rendered.push_str(&template[literal_start..]);
    positional.finish()?;
    Ok(string_to_value(rendered))
}

pub(crate) fn percent_format_bytes(
    template: &BytesObject,
    arguments: Value,
) -> Result<Value, BuiltinError> {
    let mut rendered = Vec::with_capacity(template.len());
    let mut positional = PositionalArgs::new(arguments);
    let bytes = template.as_bytes();
    let mut literal_start = 0usize;
    let mut index = 0usize;

    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }

        rendered.extend_from_slice(&bytes[literal_start..index]);
        index += 1;
        if index >= bytes.len() {
            return Err(BuiltinError::ValueError("incomplete format".to_string()));
        }

        if bytes[index] == b'%' {
            rendered.push(b'%');
            index += 1;
            literal_start = index;
            continue;
        }

        let (mut spec, next_index) = BytesPercentSpec::parse(bytes, index)?;
        index = next_index;

        if spec.width_from_arg {
            let width = positional_int_arg(&mut positional)?;
            if width < 0 {
                spec.left_adjust = true;
                spec.width = Some(
                    usize::try_from(width.unsigned_abs())
                        .map_err(|_| BuiltinError::OverflowError("width too large".to_string()))?,
                );
            } else {
                spec.width = Some(width as usize);
            }
        }

        if spec.precision_from_arg {
            let precision = positional_int_arg(&mut positional)?;
            spec.precision = usize::try_from(precision).ok();
        }

        let value = if let Some(key) = spec.mapping_key.as_deref() {
            mapping_argument(arguments, key)?
        } else {
            positional.next()?
        };

        rendered.extend_from_slice(&format_bytes_argument(value, &spec)?);
        literal_start = index;
    }

    rendered.extend_from_slice(&bytes[literal_start..]);
    positional.finish()?;
    Ok(bytes_to_value(rendered, template.header.type_id))
}

fn mapping_argument(arguments: Value, key: &str) -> Result<Value, BuiltinError> {
    let Some(ptr) = arguments.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "format requires a mapping".to_string(),
        ));
    };

    if crate::ops::objects::extract_type_id(ptr) != TypeId::DICT {
        return Err(BuiltinError::TypeError(
            "format requires a mapping".to_string(),
        ));
    }

    let dict = unsafe { &*(ptr as *const DictObject) };
    dict.get(Value::string(intern(key)))
        .ok_or_else(|| BuiltinError::KeyError(key.to_string()))
}

fn positional_int_arg(positional: &mut PositionalArgs<'_>) -> Result<i64, BuiltinError> {
    let value = positional.next()?;
    if let Some(integer) = value.as_int() {
        return Ok(integer);
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(if boolean { 1 } else { 0 });
    }
    Err(BuiltinError::TypeError("* wants int".to_string()))
}

fn format_argument(value: Value, spec: &PercentSpec) -> Result<String, BuiltinError> {
    match spec.conversion {
        's' => Ok(apply_string_width(
            string_precision(render_str(value)?, spec.precision),
            spec,
        )),
        'r' => Ok(apply_string_width(
            string_precision(render_repr(value)?, spec.precision),
            spec,
        )),
        'a' => Ok(apply_string_width(
            string_precision(render_ascii(value)?, spec.precision),
            spec,
        )),
        'c' => format_char(value, spec),
        'd' | 'i' | 'u' | 'o' | 'x' | 'X' => format_integer(value, spec),
        'e' | 'E' | 'f' | 'F' | 'g' | 'G' => format_float(value, spec),
        _ => Err(BuiltinError::ValueError("unsupported format".to_string())),
    }
}

fn format_bytes_argument(value: Value, spec: &BytesPercentSpec) -> Result<Vec<u8>, BuiltinError> {
    match spec.conversion {
        b'b' | b's' => Ok(apply_bytes_width(
            bytes_precision(raw_bytes_argument(value, spec.conversion)?, spec.precision),
            spec,
        )),
        b'r' => Ok(apply_bytes_width(
            bytes_precision(render_repr(value)?.into_bytes(), spec.precision),
            spec,
        )),
        b'a' => Ok(apply_bytes_width(
            bytes_precision(render_ascii(value)?.into_bytes(), spec.precision),
            spec,
        )),
        b'c' => format_byte_char(value, spec),
        b'd' | b'i' | b'u' | b'o' | b'x' | b'X' => {
            Ok(format_integer(value, &spec.as_text_spec())?.into_bytes())
        }
        b'e' | b'E' | b'f' | b'F' | b'g' | b'G' => {
            Ok(format_float(value, &spec.as_text_spec())?.into_bytes())
        }
        _ => Err(BuiltinError::ValueError("unsupported format".to_string())),
    }
}

fn raw_bytes_argument(value: Value, conversion: u8) -> Result<Vec<u8>, BuiltinError> {
    value_as_bytes_ref(value)
        .map(BytesObject::to_vec)
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "%{} requires a bytes-like object, or an object that implements __bytes__, not {}",
                conversion as char,
                type_name_of(value)
            ))
        })
}

fn render_str(value: Value) -> Result<String, BuiltinError> {
    string_value(super::types::builtin_str(&[value])?)
}

fn render_repr(value: Value) -> Result<String, BuiltinError> {
    string_value(super::functions::builtin_repr(&[value])?)
}

fn render_ascii(value: Value) -> Result<String, BuiltinError> {
    string_value(super::functions::builtin_ascii(&[value])?)
}

fn string_value(value: Value) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|string| string.as_str().to_string())
        .ok_or_else(|| BuiltinError::TypeError("expected string result".to_string()))
}

fn format_char(value: Value, spec: &PercentSpec) -> Result<String, BuiltinError> {
    let rendered = if let Some(integer) = value.as_int() {
        code_point_to_string(integer)?
    } else if let Some(boolean) = value.as_bool() {
        code_point_to_string(if boolean { 1 } else { 0 })?
    } else if let Some(string) = value_as_string_ref(value) {
        let mut chars = string.as_str().chars();
        let Some(ch) = chars.next() else {
            return Err(BuiltinError::TypeError(
                "%c requires an int or a single character".to_string(),
            ));
        };
        if chars.next().is_some() {
            return Err(BuiltinError::TypeError(
                "%c requires an int or a single character".to_string(),
            ));
        }
        ch.to_string()
    } else {
        return Err(BuiltinError::TypeError(format!(
            "%c requires int or char, not {}",
            type_name_of(value)
        )));
    };

    Ok(apply_string_width(rendered, spec))
}

fn format_byte_char(value: Value, spec: &BytesPercentSpec) -> Result<Vec<u8>, BuiltinError> {
    let rendered = if let Some(integer) = value.as_int() {
        byte_code_point(integer)?
    } else if let Some(boolean) = value.as_bool() {
        byte_code_point(if boolean { 1 } else { 0 })?
    } else if let Some(bytes) = value_as_bytes_ref(value) {
        let data = bytes.as_bytes();
        if data.len() != 1 {
            return Err(BuiltinError::TypeError(
                "%c requires an integer in range(256) or a single byte".to_string(),
            ));
        }
        data[0]
    } else {
        return Err(BuiltinError::TypeError(format!(
            "%c requires an integer in range(256) or a single byte, not {}",
            type_name_of(value)
        )));
    };

    Ok(apply_bytes_width(vec![rendered], spec))
}

fn byte_code_point(code_point: i64) -> Result<u8, BuiltinError> {
    u8::try_from(code_point)
        .map_err(|_| BuiltinError::OverflowError("%c arg not in range(256)".to_string()))
}

fn code_point_to_string(code_point: i64) -> Result<String, BuiltinError> {
    if !(0..=0x10FFFF).contains(&code_point) {
        return Err(BuiltinError::OverflowError(
            "%c arg not in range(0x110000)".to_string(),
        ));
    }

    let Some(ch) = char::from_u32(code_point as u32) else {
        return Err(BuiltinError::OverflowError(
            "%c arg not in range(0x110000)".to_string(),
        ));
    };

    Ok(ch.to_string())
}

fn format_integer(value: Value, spec: &PercentSpec) -> Result<String, BuiltinError> {
    let integer = int_argument(value, spec.conversion)?;
    let negative = integer < 0;
    let magnitude = if negative {
        -(integer as i128)
    } else {
        integer as i128
    };

    let mut digits = match spec.conversion {
        'd' | 'i' | 'u' => magnitude.to_string(),
        'o' => format!("{:o}", magnitude as u128),
        'x' => format!("{:x}", magnitude as u128),
        'X' => format!("{:X}", magnitude as u128),
        _ => unreachable!(),
    };

    if let Some(precision) = spec.precision {
        if digits.len() < precision {
            digits = format!("{}{digits}", "0".repeat(precision - digits.len()));
        }
    }

    let prefix = if spec.alternate {
        match spec.conversion {
            'o' => "0o",
            'x' => "0x",
            'X' => "0X",
            _ => "",
        }
    } else {
        ""
    };

    Ok(apply_numeric_width(digits, negative, prefix, spec))
}

fn int_argument(value: Value, conversion: char) -> Result<i64, BuiltinError> {
    if let Some(integer) = value.as_int() {
        return Ok(integer);
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(if boolean { 1 } else { 0 });
    }
    Err(BuiltinError::TypeError(format!(
        "%{} format: an integer is required, not {}",
        conversion,
        type_name_of(value)
    )))
}

fn format_float(value: Value, spec: &PercentSpec) -> Result<String, BuiltinError> {
    let float = float_argument(value, spec.conversion)?;
    let negative = float.is_sign_negative();
    let magnitude = float.abs();
    let precision = spec.precision.unwrap_or(6);

    let body = match spec.conversion {
        'e' => format!("{magnitude:.precision$e}"),
        'E' => format!("{magnitude:.precision$E}"),
        'f' | 'F' => format!("{magnitude:.precision$}"),
        'g' | 'G' => format!("{magnitude:.precision$}"),
        _ => unreachable!(),
    };

    Ok(apply_numeric_width(body, negative, "", spec))
}

fn float_argument(value: Value, conversion: char) -> Result<f64, BuiltinError> {
    if let Some(float) = value.as_float() {
        return Ok(float);
    }
    if let Some(integer) = value.as_int() {
        return Ok(integer as f64);
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(if boolean { 1.0 } else { 0.0 });
    }
    Err(BuiltinError::TypeError(format!(
        "%{} format: a real number is required, not {}",
        conversion,
        type_name_of(value)
    )))
}

fn string_precision(text: String, precision: Option<usize>) -> String {
    match precision {
        Some(limit) => text.chars().take(limit).collect(),
        None => text,
    }
}

fn bytes_precision(mut bytes: Vec<u8>, precision: Option<usize>) -> Vec<u8> {
    if let Some(limit) = precision {
        bytes.truncate(limit);
    }
    bytes
}

fn apply_string_width(text: String, spec: &PercentSpec) -> String {
    let width = spec.width.unwrap_or(0);
    let char_len = text.chars().count();
    if width <= char_len {
        return text;
    }

    let padding = " ".repeat(width - char_len);
    if spec.left_adjust {
        format!("{text}{padding}")
    } else {
        format!("{padding}{text}")
    }
}

fn apply_bytes_width(mut bytes: Vec<u8>, spec: &BytesPercentSpec) -> Vec<u8> {
    let width = spec.width.unwrap_or(0);
    if width <= bytes.len() {
        return bytes;
    }

    let padding_len = width - bytes.len();
    if spec.left_adjust {
        bytes.extend(std::iter::repeat_n(b' ', padding_len));
        return bytes;
    }

    let mut padded = Vec::with_capacity(width);
    padded.extend(std::iter::repeat_n(b' ', padding_len));
    padded.extend_from_slice(&bytes);
    padded
}

fn apply_numeric_width(digits: String, negative: bool, prefix: &str, spec: &PercentSpec) -> String {
    let sign = if negative {
        "-"
    } else if spec.sign_plus {
        "+"
    } else if spec.sign_space {
        " "
    } else {
        ""
    };

    let width = spec.width.unwrap_or(0);
    let min_width = sign.len() + prefix.len() + digits.len();
    if width <= min_width {
        return format!("{sign}{prefix}{digits}");
    }

    let padding_len = width - min_width;
    if spec.left_adjust {
        return format!("{sign}{prefix}{digits}{}", " ".repeat(padding_len));
    }

    if spec.zero_pad && spec.precision.is_none() {
        return format!("{sign}{prefix}{}{digits}", "0".repeat(padding_len));
    }

    format!("{}{sign}{prefix}{digits}", " ".repeat(padding_len))
}

fn string_to_value(text: String) -> Value {
    if text.is_empty() {
        return Value::string(intern(""));
    }

    crate::alloc_managed_value(StringObject::from_string(text))
}

fn bytes_to_value(data: Vec<u8>, type_id: TypeId) -> Value {
    crate::alloc_managed_value(BytesObject::from_vec_with_type(data, type_id))
}

fn type_name_of(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if value.is_string() {
        "str"
    } else if let Some(ptr) = value.as_object_ptr() {
        crate::ops::objects::extract_type_id(ptr).name()
    } else {
        "object"
    }
}

#[cfg(test)]
mod tests {
    use super::{percent_format_bytes, percent_format_string};
    use crate::builtins::BuiltinError;
    use prism_core::Value;
    use prism_core::intern::intern;
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::types::bytes::{BytesObject, value_as_bytes_ref};
    use prism_runtime::types::dict::DictObject;
    use prism_runtime::types::string::value_as_string_ref;
    use prism_runtime::types::tuple::TupleObject;

    fn render(format: &str, arguments: Value) -> String {
        let value = percent_format_string(format, arguments).expect("format should succeed");
        value_as_string_ref(value)
            .expect("result should be a string")
            .as_str()
            .to_string()
    }

    fn render_bytes(format: &[u8], arguments: Value) -> Vec<u8> {
        let template = BytesObject::from_slice(format);
        let value = percent_format_bytes(&template, arguments).expect("format should succeed");
        let rendered = value_as_bytes_ref(value)
            .expect("result should be bytes")
            .as_bytes()
            .to_vec();
        unsafe {
            drop(Box::from_raw(
                value.as_object_ptr().expect("bytes result should be boxed") as *mut BytesObject,
            ));
        }
        rendered
    }

    fn boxed_value<T>(object: T) -> (Value, *mut T) {
        let ptr = Box::into_raw(Box::new(object));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
    }

    #[test]
    fn test_percent_format_single_string_argument() {
        assert_eq!(
            render("hello %s", Value::string(intern("world"))),
            "hello world"
        );
    }

    #[test]
    fn test_percent_format_tuple_arguments() {
        let tuple =
            TupleObject::from_slice(&[Value::string(intern("value")), Value::int(7).unwrap()]);
        let (value, ptr) = boxed_value(tuple);
        assert_eq!(render("%s = %d", value), "value = 7");
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_percent_format_mapping_argument() {
        let mut dict = DictObject::new();
        dict.set(
            Value::string(intern("prog")),
            Value::string(intern("prism")),
        );
        let (value, ptr) = boxed_value(dict);
        assert_eq!(render("%(prog)s", value), "prism");
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_percent_format_repr_and_percent_escape() {
        assert_eq!(render("%r %%", Value::string(intern("x"))), "'x' %");
    }

    #[test]
    fn test_percent_format_string_precision_and_width() {
        assert_eq!(render("%-6.3s", Value::string(intern("python"))), "pyt   ");
    }

    #[test]
    fn test_percent_format_integer_flags() {
        assert_eq!(render("%#06x", Value::int(31).unwrap()), "0x001f");
    }

    #[test]
    fn test_percent_format_char_from_integer() {
        assert_eq!(render("%c", Value::int(65).unwrap()), "A");
    }

    #[test]
    fn test_percent_format_rejects_missing_arguments() {
        let err = percent_format_string("%s %s", Value::string(intern("only")))
            .expect_err("format should fail");
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("not enough arguments"));
    }

    #[test]
    fn test_percent_format_rejects_extra_tuple_arguments() {
        let tuple = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (value, ptr) = boxed_value(tuple);
        let err = percent_format_string("%d", value).expect_err("format should fail");
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("not all arguments converted"));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_percent_format_bytes_inserts_raw_byte_arguments() {
        let (argument, argument_ptr) = boxed_value(BytesObject::from_slice(b"GLIBC_2.9"));
        assert_eq!(
            render_bytes(b"[xxx%sxxx]", argument),
            b"[xxxGLIBC_2.9xxx]".to_vec()
        );
        unsafe { drop_boxed(argument_ptr) };
    }

    #[test]
    fn test_percent_format_bytes_honors_tuple_width_precision_and_numeric_specs() {
        let (argument, argument_ptr) = boxed_value(BytesObject::from_slice(b"abcdef"));
        let tuple = TupleObject::from_slice(&[argument, Value::int(31).unwrap()]);
        let (tuple_value, tuple_ptr) = boxed_value(tuple);
        assert_eq!(
            render_bytes(b"%-5.3s:%#06x", tuple_value),
            b"abc  :0x001f".to_vec()
        );
        unsafe {
            drop_boxed(tuple_ptr);
            drop_boxed(argument_ptr);
        }
    }

    #[test]
    fn test_percent_format_bytearray_preserves_receiver_type() {
        let template = BytesObject::bytearray_from_slice(b"%c:%b");
        let (argument, argument_ptr) = boxed_value(BytesObject::from_slice(b"z"));
        let tuple = TupleObject::from_slice(&[Value::int(65).unwrap(), argument]);
        let (tuple_value, tuple_ptr) = boxed_value(tuple);
        let result =
            percent_format_bytes(&template, tuple_value).expect("bytearray format should succeed");
        assert_eq!(value_as_bytes_ref(result).unwrap().as_bytes(), b"A:z");
        assert_eq!(
            value_as_bytes_ref(result).unwrap().header.type_id,
            TypeId::BYTEARRAY
        );

        unsafe {
            drop(Box::from_raw(
                result.as_object_ptr().expect("result should be boxed") as *mut BytesObject,
            ));
            drop_boxed(tuple_ptr);
            drop_boxed(argument_ptr);
        }
    }
}
