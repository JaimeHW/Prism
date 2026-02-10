//! JSON decoding (parsing) implementation.
//!
//! High-performance JSON to Value conversion with zero-copy string parsing.

use prism_core::Value;
use prism_core::intern::intern;
use std::sync::Arc;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during JSON decoding.
#[derive(Debug, Clone)]
pub struct JsonDecodeError {
    /// Error message.
    pub msg: Arc<str>,
    /// Position in input where error occurred.
    pub pos: usize,
    /// Line number (1-indexed).
    pub line: usize,
    /// Column number (1-indexed).
    pub col: usize,
}

impl JsonDecodeError {
    fn new(msg: impl Into<Arc<str>>, input: &str, pos: usize) -> Self {
        let (line, col) = Self::compute_line_col(input, pos);
        Self {
            msg: msg.into(),
            pos,
            line,
            col,
        }
    }

    fn compute_line_col(input: &str, pos: usize) -> (usize, usize) {
        let prefix = &input[..pos.min(input.len())];
        let line = prefix.matches('\n').count() + 1;
        let col = prefix.rfind('\n').map(|i| pos - i).unwrap_or(pos + 1);
        (line, col)
    }
}

impl std::fmt::Display for JsonDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "JSONDecodeError: {} (line {}, column {})",
            self.msg, self.line, self.col
        )
    }
}

impl std::error::Error for JsonDecodeError {}

// =============================================================================
// Decoder
// =============================================================================

/// Parse a JSON string into a Prism Value.
///
/// # Examples
///
/// ```ignore
/// let val = loads(r#"{"key": [1, 2, 3]}"#)?;
/// ```
pub fn loads(input: &str) -> Result<Value, JsonDecodeError> {
    let mut decoder = Decoder::new(input);
    let value = decoder.parse_value()?;
    decoder.skip_whitespace();
    if decoder.pos < decoder.input.len() {
        return Err(JsonDecodeError::new(
            "Extra data after JSON value",
            input,
            decoder.pos,
        ));
    }
    Ok(value)
}

struct Decoder<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    #[inline]
    fn advance(&mut self) {
        self.pos += 1;
    }

    #[inline]
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            match c {
                b' ' | b'\t' | b'\n' | b'\r' => self.advance(),
                _ => break,
            }
        }
    }

    fn parse_value(&mut self) -> Result<Value, JsonDecodeError> {
        self.skip_whitespace();

        match self.peek() {
            Some(b'"') => self.parse_string(),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') => self.parse_true(),
            Some(b'f') => self.parse_false(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(JsonDecodeError::new(
                format!("Unexpected character: '{}'", c as char),
                self.input,
                self.pos,
            )),
            None => Err(JsonDecodeError::new(
                "Unexpected end of input",
                self.input,
                self.pos,
            )),
        }
    }

    fn parse_string(&mut self) -> Result<Value, JsonDecodeError> {
        self.advance(); // Skip opening quote

        let start = self.pos;
        let mut has_escapes = false;

        loop {
            match self.peek() {
                Some(b'"') => {
                    let s = if has_escapes {
                        self.unescape_string(&self.input[start..self.pos])?
                    } else {
                        self.input[start..self.pos].to_string()
                    };
                    self.advance(); // Skip closing quote
                    return Ok(Value::string(intern(&s)));
                }
                Some(b'\\') => {
                    has_escapes = true;
                    self.advance();
                    if self.peek().is_none() {
                        return Err(JsonDecodeError::new(
                            "Unterminated string escape",
                            self.input,
                            self.pos,
                        ));
                    }
                    self.advance();
                }
                Some(c) if c < 0x20 => {
                    return Err(JsonDecodeError::new(
                        "Control character in string",
                        self.input,
                        self.pos,
                    ));
                }
                Some(_) => self.advance(),
                None => {
                    return Err(JsonDecodeError::new(
                        "Unterminated string",
                        self.input,
                        start - 1,
                    ));
                }
            }
        }
    }

    fn unescape_string(&self, s: &str) -> Result<String, JsonDecodeError> {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some('/') => result.push('/'),
                    Some('b') => result.push('\x08'),
                    Some('f') => result.push('\x0C'),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if hex.len() != 4 {
                            return Err(JsonDecodeError::new(
                                "Invalid unicode escape",
                                self.input,
                                self.pos,
                            ));
                        }
                        let code = u16::from_str_radix(&hex, 16).map_err(|_| {
                            JsonDecodeError::new("Invalid unicode escape", self.input, self.pos)
                        })?;
                        if let Some(ch) = char::from_u32(code as u32) {
                            result.push(ch);
                        } else {
                            // Handle surrogate pairs
                            result.push(char::REPLACEMENT_CHARACTER);
                        }
                    }
                    _ => {
                        return Err(JsonDecodeError::new(
                            "Invalid escape sequence",
                            self.input,
                            self.pos,
                        ));
                    }
                }
            } else {
                result.push(c);
            }
        }

        Ok(result)
    }

    fn parse_number(&mut self) -> Result<Value, JsonDecodeError> {
        let start = self.pos;

        // Optional minus
        if self.peek() == Some(b'-') {
            self.advance();
        }

        // Integer part
        match self.peek() {
            Some(b'0') => self.advance(),
            Some(c) if c.is_ascii_digit() => {
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            _ => {
                return Err(JsonDecodeError::new("Invalid number", self.input, self.pos));
            }
        }

        let mut is_float = false;

        // Fractional part
        if self.peek() == Some(b'.') {
            is_float = true;
            self.advance();
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(JsonDecodeError::new(
                    "Invalid number: expected digit after decimal point",
                    self.input,
                    self.pos,
                ));
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Exponent part
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.advance();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.advance();
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(JsonDecodeError::new(
                    "Invalid number: expected digit in exponent",
                    self.input,
                    self.pos,
                ));
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        let num_str = &self.input[start..self.pos];

        if is_float {
            let f: f64 = num_str
                .parse()
                .map_err(|_| JsonDecodeError::new("Invalid float", self.input, start))?;
            Ok(Value::float(f))
        } else {
            // Try i64 first, fall back to float for large numbers
            match num_str.parse::<i64>() {
                Ok(i) => Ok(Value::int_unchecked(i)),
                Err(_) => {
                    let f: f64 = num_str
                        .parse()
                        .map_err(|_| JsonDecodeError::new("Invalid integer", self.input, start))?;
                    Ok(Value::float(f))
                }
            }
        }
    }

    fn parse_object(&mut self) -> Result<Value, JsonDecodeError> {
        self.advance(); // Skip '{'
        self.skip_whitespace();

        // For now, return None as we don't have dict Value support yet
        // TODO: Implement proper dict support
        if self.peek() == Some(b'}') {
            self.advance();
            return Ok(Value::none());
        }

        // Parse key-value pairs (consume them but return None for now)
        loop {
            self.skip_whitespace();

            // Parse key
            if self.peek() != Some(b'"') {
                return Err(JsonDecodeError::new(
                    "Expected string key",
                    self.input,
                    self.pos,
                ));
            }
            let _key = self.parse_string()?;

            self.skip_whitespace();

            // Expect colon
            if self.peek() != Some(b':') {
                return Err(JsonDecodeError::new(
                    "Expected ':' after object key",
                    self.input,
                    self.pos,
                ));
            }
            self.advance();

            // Parse value
            let _value = self.parse_value()?;

            self.skip_whitespace();

            match self.peek() {
                Some(b',') => {
                    self.advance();
                    self.skip_whitespace();
                }
                Some(b'}') => {
                    self.advance();
                    return Ok(Value::none()); // TODO: Return actual dict
                }
                _ => {
                    return Err(JsonDecodeError::new(
                        "Expected ',' or '}' in object",
                        self.input,
                        self.pos,
                    ));
                }
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, JsonDecodeError> {
        self.advance(); // Skip '['
        self.skip_whitespace();

        // For now, return None as we don't have list Value support yet
        // TODO: Implement proper list support
        if self.peek() == Some(b']') {
            self.advance();
            return Ok(Value::none());
        }

        loop {
            let _value = self.parse_value()?;

            self.skip_whitespace();

            match self.peek() {
                Some(b',') => {
                    self.advance();
                    self.skip_whitespace();
                }
                Some(b']') => {
                    self.advance();
                    return Ok(Value::none()); // TODO: Return actual list
                }
                _ => {
                    return Err(JsonDecodeError::new(
                        "Expected ',' or ']' in array",
                        self.input,
                        self.pos,
                    ));
                }
            }
        }
    }

    fn parse_true(&mut self) -> Result<Value, JsonDecodeError> {
        if self.input[self.pos..].starts_with("true") {
            self.pos += 4;
            Ok(Value::bool(true))
        } else {
            Err(JsonDecodeError::new(
                "Expected 'true'",
                self.input,
                self.pos,
            ))
        }
    }

    fn parse_false(&mut self) -> Result<Value, JsonDecodeError> {
        if self.input[self.pos..].starts_with("false") {
            self.pos += 5;
            Ok(Value::bool(false))
        } else {
            Err(JsonDecodeError::new(
                "Expected 'false'",
                self.input,
                self.pos,
            ))
        }
    }

    fn parse_null(&mut self) -> Result<Value, JsonDecodeError> {
        if self.input[self.pos..].starts_with("null") {
            self.pos += 4;
            Ok(Value::none())
        } else {
            Err(JsonDecodeError::new(
                "Expected 'null'",
                self.input,
                self.pos,
            ))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod decode_tests {
    use super::*;

    #[test]
    fn test_decode_null() {
        let val = loads("null").unwrap();
        assert!(val.is_none());
    }

    #[test]
    fn test_decode_true() {
        let val = loads("true").unwrap();
        assert_eq!(val.as_bool(), Some(true));
    }

    #[test]
    fn test_decode_false() {
        let val = loads("false").unwrap();
        assert_eq!(val.as_bool(), Some(false));
    }

    #[test]
    fn test_decode_integer() {
        let val = loads("42").unwrap();
        assert_eq!(val.as_int(), Some(42));
    }

    #[test]
    fn test_decode_negative_integer() {
        let val = loads("-123").unwrap();
        assert_eq!(val.as_int(), Some(-123));
    }

    #[test]
    fn test_decode_float() {
        let val = loads("3.14").unwrap();
        let f = val.as_float().unwrap();
        assert!((f - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_decode_float_exponent() {
        let val = loads("1.5e10").unwrap();
        let f = val.as_float().unwrap();
        assert!((f - 1.5e10).abs() < 1e5);
    }

    #[test]
    fn test_decode_string() {
        let val = loads(r#""hello""#).unwrap();
        assert!(val.is_string());
    }

    #[test]
    fn test_decode_string_escapes() {
        let val = loads(r#""hello\nworld""#).unwrap();
        assert!(val.is_string());
    }

    #[test]
    fn test_decode_empty_object() {
        let val = loads("{}").unwrap();
        // Currently returns None until dict support
        assert!(val.is_none());
    }

    #[test]
    fn test_decode_empty_array() {
        let val = loads("[]").unwrap();
        // Currently returns None until list support
        assert!(val.is_none());
    }

    #[test]
    fn test_decode_whitespace() {
        let val = loads("  42  ").unwrap();
        assert_eq!(val.as_int(), Some(42));
    }

    #[test]
    fn test_decode_error_extra_data() {
        let err = loads("42 extra").unwrap_err();
        assert!(err.msg.contains("Extra data"));
    }

    #[test]
    fn test_decode_error_unterminated_string() {
        let err = loads(r#""unterminated"#).unwrap_err();
        assert!(err.msg.contains("Unterminated"));
    }

    #[test]
    fn test_decode_error_invalid_escape() {
        let err = loads(r#""\x""#).unwrap_err();
        assert!(err.msg.contains("Invalid escape"));
    }

    #[test]
    fn test_decode_error_unexpected_char() {
        let err = loads("undefined").unwrap_err();
        assert!(err.msg.contains("Unexpected"));
    }
}
