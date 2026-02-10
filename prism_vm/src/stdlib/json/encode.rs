//! JSON encoding (serialization) implementation.
//!
//! High-performance Value to JSON string conversion.

use prism_core::Value;
use std::sync::Arc;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during JSON encoding.
#[derive(Debug, Clone)]
pub struct JsonEncodeError {
    /// Error message.
    pub msg: Arc<str>,
}

impl JsonEncodeError {
    fn new(msg: impl Into<Arc<str>>) -> Self {
        Self { msg: msg.into() }
    }
}

impl std::fmt::Display for JsonEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSONEncodeError: {}", self.msg)
    }
}

impl std::error::Error for JsonEncodeError {}

// =============================================================================
// Encoder
// =============================================================================

/// Serialize a Prism Value to a JSON string.
///
/// # Examples
///
/// ```ignore
/// let s = dumps(Value::int_unchecked(42))?;
/// assert_eq!(s, "42");
/// ```
pub fn dumps(value: Value) -> Result<String, JsonEncodeError> {
    let mut encoder = Encoder::new();
    encoder.encode_value(value)?;
    Ok(encoder.output)
}

/// Serialize a value to JSON with custom indentation.
pub fn dumps_pretty(value: Value, indent: usize) -> Result<String, JsonEncodeError> {
    let mut encoder = Encoder::with_indent(indent);
    encoder.encode_value(value)?;
    Ok(encoder.output)
}

struct Encoder {
    output: String,
    indent: Option<usize>,
    depth: usize,
}

impl Encoder {
    fn new() -> Self {
        Self {
            output: String::with_capacity(256),
            indent: None,
            depth: 0,
        }
    }

    fn with_indent(spaces: usize) -> Self {
        Self {
            output: String::with_capacity(256),
            indent: Some(spaces),
            depth: 0,
        }
    }

    fn encode_value(&mut self, value: Value) -> Result<(), JsonEncodeError> {
        if value.is_none() {
            self.output.push_str("null");
        } else if let Some(b) = value.as_bool() {
            self.output.push_str(if b { "true" } else { "false" });
        } else if let Some(i) = value.as_int() {
            self.output.push_str(&i.to_string());
        } else if let Some(f) = value.as_float() {
            if f.is_nan() {
                return Err(JsonEncodeError::new("NaN is not JSON serializable"));
            }
            if f.is_infinite() {
                return Err(JsonEncodeError::new("Infinity is not JSON serializable"));
            }
            self.output.push_str(&f.to_string());
        } else if value.is_string() {
            self.encode_string(value)?;
        } else {
            // For unsupported types, encode as null
            // TODO: Add list/dict support
            self.output.push_str("null");
        }
        Ok(())
    }

    fn encode_string(&mut self, value: Value) -> Result<(), JsonEncodeError> {
        self.output.push('"');

        // Get string representation
        // Since we can't easily extract the string, use is_string check
        // For now, output a placeholder
        // TODO: Add proper string extraction when Value interface is extended
        self.output.push_str("string");

        self.output.push('"');
        Ok(())
    }

    #[allow(dead_code)]
    fn write_indent(&mut self) {
        if let Some(spaces) = self.indent {
            self.output.push('\n');
            for _ in 0..(self.depth * spaces) {
                self.output.push(' ');
            }
        }
    }
}

/// Escape a string for JSON output.
#[allow(dead_code)]
fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');

    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c < ' ' => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }

    result.push('"');
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod encode_tests {
    use super::*;

    #[test]
    fn test_encode_null() {
        let s = dumps(Value::none()).unwrap();
        assert_eq!(s, "null");
    }

    #[test]
    fn test_encode_true() {
        let s = dumps(Value::bool(true)).unwrap();
        assert_eq!(s, "true");
    }

    #[test]
    fn test_encode_false() {
        let s = dumps(Value::bool(false)).unwrap();
        assert_eq!(s, "false");
    }

    #[test]
    fn test_encode_integer() {
        let s = dumps(Value::int_unchecked(42)).unwrap();
        assert_eq!(s, "42");
    }

    #[test]
    fn test_encode_negative() {
        let s = dumps(Value::int_unchecked(-123)).unwrap();
        assert_eq!(s, "-123");
    }

    #[test]
    fn test_encode_float() {
        let s = dumps(Value::float(3.14)).unwrap();
        assert!(s.starts_with("3.14"));
    }

    #[test]
    fn test_encode_nan_error() {
        let err = dumps(Value::float(f64::NAN)).unwrap_err();
        assert!(err.msg.contains("NaN"));
    }

    #[test]
    fn test_encode_infinity_error() {
        let err = dumps(Value::float(f64::INFINITY)).unwrap_err();
        assert!(err.msg.contains("Infinity"));
    }

    #[test]
    fn test_escape_string() {
        let escaped = escape_string("hello\nworld");
        assert_eq!(escaped, r#""hello\nworld""#);
    }

    #[test]
    fn test_escape_quotes() {
        let escaped = escape_string(r#"say "hello""#);
        assert_eq!(escaped, r#""say \"hello\"""#);
    }

    #[test]
    fn test_escape_backslash() {
        let escaped = escape_string(r"path\to\file");
        assert_eq!(escaped, r#""path\\to\\file""#);
    }

    #[test]
    fn test_escape_control_char() {
        let escaped = escape_string("\x00");
        assert_eq!(escaped, r#""\u0000""#);
    }
}
