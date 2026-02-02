//! Path join operations with stack-allocated buffers.

use super::SEP;
use std::path::Path;

/// Maximum path length for stack allocation.
const MAX_STACK_PATH: usize = 4096;

/// Join two path components.
#[inline]
pub fn join<P: AsRef<Path>, Q: AsRef<Path>>(base: P, part: Q) -> String {
    let base = base.as_ref();
    let part = part.as_ref();
    base.join(part).to_string_lossy().into_owned()
}

/// Join multiple path components.
pub fn join_many<P: AsRef<Path>>(parts: &[P]) -> String {
    if parts.is_empty() {
        return String::new();
    }
    let mut result = parts[0].as_ref().to_path_buf();
    for part in &parts[1..] {
        result.push(part);
    }
    result.to_string_lossy().into_owned()
}

/// Stack-allocated path buffer for zero-heap joins (small paths).
#[derive(Debug)]
pub struct StackPath {
    buf: [u8; MAX_STACK_PATH],
    len: usize,
}

impl StackPath {
    /// Create empty path.
    #[inline]
    pub fn new() -> Self {
        Self {
            buf: [0; MAX_STACK_PATH],
            len: 0,
        }
    }

    /// Create from string, returns None if too long.
    pub fn from_str(s: &str) -> Option<Self> {
        if s.len() > MAX_STACK_PATH {
            return None;
        }
        let mut sp = Self::new();
        sp.buf[..s.len()].copy_from_slice(s.as_bytes());
        sp.len = s.len();
        Some(sp)
    }

    /// Append a path component.
    pub fn push(&mut self, part: &str) -> bool {
        let need = if self.len == 0 {
            part.len()
        } else {
            1 + part.len()
        };
        if self.len + need > MAX_STACK_PATH {
            return false;
        }
        if self.len > 0 {
            self.buf[self.len] = SEP as u8;
            self.len += 1;
        }
        self.buf[self.len..self.len + part.len()].copy_from_slice(part.as_bytes());
        self.len += part.len();
        true
    }

    /// Get as string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }

    /// Get length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for StackPath {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_basic() {
        let p = join("a", "b");
        assert!(p.contains("a") && p.contains("b"));
    }

    #[test]
    fn test_join_empty_base() {
        let p = join("", "b");
        assert_eq!(p, "b");
    }

    #[test]
    fn test_join_empty_part() {
        let p = join("a", "");
        assert!(p.contains("a"));
    }

    #[test]
    fn test_join_many_empty() {
        let parts: Vec<&str> = vec![];
        assert_eq!(join_many(&parts), "");
    }

    #[test]
    fn test_join_many_single() {
        let parts = vec!["a"];
        assert_eq!(join_many(&parts), "a");
    }

    #[test]
    fn test_join_many_multiple() {
        let parts = vec!["a", "b", "c"];
        let p = join_many(&parts);
        assert!(p.contains("a") && p.contains("b") && p.contains("c"));
    }

    #[test]
    fn test_stack_path_new() {
        let sp = StackPath::new();
        assert!(sp.is_empty());
        assert_eq!(sp.len(), 0);
    }

    #[test]
    fn test_stack_path_from_str() {
        let sp = StackPath::from_str("hello").unwrap();
        assert_eq!(sp.as_str(), "hello");
        assert_eq!(sp.len(), 5);
    }

    #[test]
    fn test_stack_path_push() {
        let mut sp = StackPath::from_str("a").unwrap();
        assert!(sp.push("b"));
        let s = sp.as_str();
        assert!(s.contains("a") && s.contains("b"));
    }

    #[test]
    fn test_stack_path_too_long() {
        let long_str = "x".repeat(MAX_STACK_PATH + 1);
        assert!(StackPath::from_str(&long_str).is_none());
    }
}
