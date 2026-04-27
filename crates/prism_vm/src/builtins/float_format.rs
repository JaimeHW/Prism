//! Python-compatible float formatting shared by `format()` and `%`.

use super::BuiltinError;

const DEFAULT_PRECISION: usize = 6;
const MAX_FLOAT_FORMAT_PRECISION: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum FloatFormatSign {
    #[default]
    MinusOnly,
    Plus,
    Space,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FloatFormatSpec {
    pub alternate: bool,
    pub zero_pad: bool,
    pub left_adjust: bool,
    pub sign: FloatFormatSign,
    pub width: Option<usize>,
    pub precision: Option<usize>,
    pub ty: Option<char>,
}

pub(crate) fn python_float_str(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    let mut buffer = ryu::Buffer::new();
    let mut rendered = buffer.format(value).to_string();
    if rendered.contains('e') || rendered.contains('E') {
        return normalize_exponent(rendered, false);
    }
    if let Some(exponential) = fixed_short_float_to_exponential(&rendered) {
        return exponential;
    }
    if value.fract() == 0.0 && !rendered.contains('.') {
        rendered.push_str(".0");
    }
    rendered
}

pub(crate) fn format_python_float(
    value: f64,
    spec: &FloatFormatSpec,
) -> Result<String, BuiltinError> {
    let ty = spec.ty;
    if let Some(ty) = ty {
        if matches!(ty, 's' | 'b' | 'c' | 'd' | 'o' | 'x' | 'X') {
            return Err(BuiltinError::ValueError(format!(
                "unknown format code '{ty}' for object of type 'float'"
            )));
        }
        if !matches!(ty, 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | '%') {
            return Err(BuiltinError::ValueError(format!(
                "unknown format code '{ty}' for object of type 'float'"
            )));
        }
    }

    if let Some(precision) = spec.precision {
        if precision > MAX_FLOAT_FORMAT_PRECISION {
            return Err(BuiltinError::OverflowError(
                "formatted float is too long".to_string(),
            ));
        }
    }

    let upper = matches!(ty, Some('E' | 'F' | 'G'));
    let negative = value.is_sign_negative() && !value.is_nan();
    let magnitude = value.abs();

    let body = if let Some(special) = special_float_body(value, upper) {
        special
    } else {
        match ty {
            Some('e' | 'E') => format_exponential(magnitude, spec.precision, spec.alternate, upper),
            Some('f' | 'F') => format_fixed(magnitude, spec.precision, spec.alternate),
            Some('g' | 'G') => {
                format_general(magnitude, spec.precision, spec.alternate, upper, false)
            }
            Some('%') => format_percent(magnitude, spec.precision, spec.alternate),
            None if spec.precision.is_some() => {
                format_general(magnitude, spec.precision, spec.alternate, false, true)
            }
            None => python_float_str(magnitude),
            Some(_) => unreachable!("validated float format type"),
        }
    };

    Ok(apply_float_width(body, negative, spec))
}

fn special_float_body(value: f64, upper: bool) -> Option<String> {
    if value.is_nan() {
        return Some(if upper { "NAN" } else { "nan" }.to_string());
    }
    if value.is_infinite() {
        return Some(if upper { "INF" } else { "inf" }.to_string());
    }
    None
}

fn format_fixed(magnitude: f64, precision: Option<usize>, alternate: bool) -> String {
    let precision = precision.unwrap_or(DEFAULT_PRECISION);
    let mut body = format!("{magnitude:.precision$}");
    if alternate && precision == 0 {
        body.push('.');
    }
    body
}

fn format_percent(magnitude: f64, precision: Option<usize>, alternate: bool) -> String {
    let mut body = format_fixed(magnitude * 100.0, precision, alternate);
    body.push('%');
    body
}

fn format_exponential(
    magnitude: f64,
    precision: Option<usize>,
    alternate: bool,
    upper: bool,
) -> String {
    let precision = precision.unwrap_or(DEFAULT_PRECISION);
    let mut body = format!("{magnitude:.precision$e}");
    if alternate && precision == 0 {
        insert_decimal_before_exponent(&mut body);
    }
    normalize_exponent(body, upper)
}

fn format_general(
    magnitude: f64,
    precision: Option<usize>,
    alternate: bool,
    upper: bool,
    default_type: bool,
) -> String {
    let precision = precision.unwrap_or(DEFAULT_PRECISION).max(1);
    let exponent = decimal_exponent(magnitude);
    let exponent_cutoff = if default_type {
        precision.saturating_sub(1) as i32
    } else {
        precision as i32
    };

    if exponent < -4 || exponent >= exponent_cutoff {
        let mut body = format!("{magnitude:.prec$e}", prec = precision - 1);
        if alternate {
            if precision == 1 {
                insert_decimal_before_exponent(&mut body);
            }
        } else {
            strip_exponential_trailing_zeros(&mut body);
        }
        normalize_exponent(body, upper)
    } else {
        let fractional_digits = (precision as i32 - exponent - 1).max(0) as usize;
        let mut body = format!("{magnitude:.fractional_digits$}");
        if alternate {
            if !body.contains('.') {
                body.push('.');
            }
        } else {
            strip_fixed_trailing_zeros(&mut body);
        }
        body
    }
}

fn decimal_exponent(value: f64) -> i32 {
    if value == 0.0 {
        return 0;
    }

    let mut exponent = value.log10().floor() as i32;
    while value < 10.0_f64.powi(exponent) {
        exponent -= 1;
    }
    while value >= 10.0_f64.powi(exponent + 1) {
        exponent += 1;
    }
    exponent
}

fn normalize_exponent(mut body: String, upper: bool) -> String {
    let marker_index = body
        .find('e')
        .or_else(|| body.find('E'))
        .expect("exponential formatter must emit an exponent");
    let exponent_text = &body[marker_index + 1..];
    let exponent = exponent_text.parse::<i32>().unwrap_or(0);
    body.truncate(marker_index);

    let marker = if upper { 'E' } else { 'e' };
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{body}{marker}{sign}{:02}", exponent.unsigned_abs())
}

fn fixed_short_float_to_exponential(text: &str) -> Option<String> {
    let (sign, unsigned) = text
        .strip_prefix('-')
        .map_or(("", text), |rest| ("-", rest));
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));

    let (exponent, mut digits) =
        if let Some(first_non_zero) = whole.bytes().position(|byte| byte != b'0') {
            (
                (whole.len() - first_non_zero - 1) as i32,
                format!("{}{}", &whole[first_non_zero..], fraction),
            )
        } else {
            let first_non_zero = fraction.bytes().position(|byte| byte != b'0')?;
            (
                -((first_non_zero as i32) + 1),
                fraction[first_non_zero..].to_string(),
            )
        };

    if !(-4..16).contains(&exponent) {
        while digits.ends_with('0') {
            digits.pop();
        }
        let mut body = String::with_capacity(sign.len() + digits.len() + 6);
        body.push_str(sign);
        body.push(digits.as_bytes()[0] as char);
        if digits.len() > 1 {
            body.push('.');
            body.push_str(&digits[1..]);
        }
        body.push('e');
        body.push(if exponent < 0 { '-' } else { '+' });
        body.push_str(&format!("{:02}", exponent.unsigned_abs()));
        return Some(body);
    }

    None
}

fn insert_decimal_before_exponent(body: &mut String) {
    let marker_index = body
        .find('e')
        .or_else(|| body.find('E'))
        .unwrap_or(body.len());
    if !body[..marker_index].contains('.') {
        body.insert(marker_index, '.');
    }
}

fn strip_exponential_trailing_zeros(body: &mut String) {
    let marker_index = body
        .find('e')
        .or_else(|| body.find('E'))
        .expect("exponential formatter must emit an exponent");
    let exponent = body[marker_index..].to_string();
    body.truncate(marker_index);
    strip_fixed_trailing_zeros(body);
    body.push_str(&exponent);
}

fn strip_fixed_trailing_zeros(body: &mut String) {
    if !body.contains('.') {
        return;
    }
    while body.ends_with('0') {
        body.pop();
    }
    if body.ends_with('.') {
        body.pop();
    }
}

fn apply_float_width(body: String, negative: bool, spec: &FloatFormatSpec) -> String {
    let sign = if negative {
        "-"
    } else {
        match spec.sign {
            FloatFormatSign::MinusOnly => "",
            FloatFormatSign::Plus => "+",
            FloatFormatSign::Space => " ",
        }
    };

    let width = spec.width.unwrap_or(0);
    let core_len = sign.len() + body.len();
    if width <= core_len {
        return format!("{sign}{body}");
    }

    let padding_len = width - core_len;
    if spec.left_adjust {
        return format!("{sign}{body}{}", " ".repeat(padding_len));
    }
    if spec.zero_pad {
        return format!("{sign}{}{body}", "0".repeat(padding_len));
    }
    format!("{}{sign}{body}", " ".repeat(padding_len))
}
