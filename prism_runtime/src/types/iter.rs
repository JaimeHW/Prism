//! Python iterator protocol implementations.
//!
//! Provides a unified iterator type that can wrap different iterable objects
//! and implements the Python iteration protocol.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::list::ListObject;
use crate::types::range::RangeIterator;
use crate::types::string::StringObject;
use crate::types::tuple::TupleObject;
use prism_core::Value;
use std::fmt;
use std::sync::Arc;

// =============================================================================
// IteratorObject
// =============================================================================

/// Python iterator object.
///
/// Wraps different iterable types and provides a unified iteration interface.
/// Each iterator kind is optimized for its specific type.
///
/// # Performance
///
/// - Range iteration: O(1) per step, no memory allocation
/// - List/Tuple iteration: O(1) per step, shared reference
/// - String char iteration: O(n) overall due to UTF-8 decoding
#[repr(C)]
pub struct IteratorObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Iterator implementation.
    kind: IterKind,
    /// Whether the iterator is exhausted.
    exhausted: bool,
}

/// Internal iterator state.
enum IterKind {
    /// Iterator over a range (most efficient).
    Range(RangeIterator),

    /// Iterator over a list.
    List { list: Arc<ListObject>, index: usize },

    /// Iterator over a tuple.
    Tuple {
        tuple: Arc<TupleObject>,
        index: usize,
    },

    /// Iterator over string characters.
    StringChars {
        string: Arc<StringObject>,
        /// Byte offset into UTF-8 string.
        byte_offset: usize,
    },

    /// Iterator over a generic sequence of values.
    /// Used as fallback for custom iterables.
    Values { values: Vec<Value>, index: usize },

    /// Empty iterator.
    Empty,
}

impl IteratorObject {
    /// Create an empty iterator.
    #[inline]
    pub fn empty() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Empty,
            exhausted: true,
        }
    }

    /// Create an iterator over a range.
    #[inline]
    pub fn from_range(iter: RangeIterator) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Range(iter),
            exhausted: false,
        }
    }

    /// Create an iterator over a list.
    #[inline]
    pub fn from_list(list: Arc<ListObject>) -> Self {
        let exhausted = list.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::List { list, index: 0 },
            exhausted,
        }
    }

    /// Create an iterator over a tuple.
    #[inline]
    pub fn from_tuple(tuple: Arc<TupleObject>) -> Self {
        let exhausted = tuple.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Tuple { tuple, index: 0 },
            exhausted,
        }
    }

    /// Create an iterator over string characters.
    #[inline]
    pub fn from_string_chars(string: Arc<StringObject>) -> Self {
        let exhausted = string.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::StringChars {
                string,
                byte_offset: 0,
            },
            exhausted,
        }
    }

    /// Create an iterator over a vector of values.
    #[inline]
    pub fn from_values(values: Vec<Value>) -> Self {
        let exhausted = values.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Values { values, index: 0 },
            exhausted,
        }
    }

    /// Check if the iterator is exhausted.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Get the next value from the iterator.
    ///
    /// Returns `Some(value)` if there are more elements, `None` if exhausted.
    pub fn next(&mut self) -> Option<Value> {
        if self.exhausted {
            return None;
        }

        match &mut self.kind {
            IterKind::Range(iter) => match iter.next() {
                Some(v) => Value::int(v), // Value::int returns Option<Value>
                None => {
                    self.exhausted = true;
                    None
                }
            },

            IterKind::List { list, index } => {
                if *index < list.len() {
                    let value = list.get(*index as i64);
                    *index += 1;
                    value
                } else {
                    self.exhausted = true;
                    None
                }
            }

            IterKind::Tuple { tuple, index } => {
                if *index < tuple.len() {
                    let value = tuple.get(*index as i64);
                    *index += 1;
                    value
                } else {
                    self.exhausted = true;
                    None
                }
            }

            IterKind::StringChars {
                string,
                byte_offset,
            } => {
                let s = string.as_str();
                if *byte_offset >= s.len() {
                    self.exhausted = true;
                    return None;
                }

                // Get the next char and its byte length
                let remaining = &s[*byte_offset..];
                if let Some(c) = remaining.chars().next() {
                    *byte_offset += c.len_utf8();
                    // Return the character as a single-char string
                    // Note: For now, return as interned string via string method
                    let interned = prism_core::intern::intern(&c.to_string());
                    Some(Value::string(interned))
                } else {
                    self.exhausted = true;
                    None
                }
            }

            IterKind::Values { values, index } => {
                if *index < values.len() {
                    let value = values[*index];
                    *index += 1;
                    Some(value)
                } else {
                    self.exhausted = true;
                    None
                }
            }

            IterKind::Empty => None,
        }
    }

    /// Get the remaining length hint, if known.
    pub fn size_hint(&self) -> Option<usize> {
        if self.exhausted {
            return Some(0);
        }

        match &self.kind {
            IterKind::Range(iter) => Some(iter.len()),
            IterKind::List { list, index } => Some(list.len().saturating_sub(*index)),
            IterKind::Tuple { tuple, index } => Some(tuple.len().saturating_sub(*index)),
            IterKind::StringChars {
                string,
                byte_offset,
            } => {
                // We can't know exactly without scanning, so return None
                // Could count remaining chars but that's O(n)
                let remaining_bytes = string.len().saturating_sub(*byte_offset);
                if remaining_bytes == 0 {
                    Some(0)
                } else {
                    None // Unknown without counting
                }
            }
            IterKind::Values { values, index } => Some(values.len().saturating_sub(*index)),
            IterKind::Empty => Some(0),
        }
    }

    /// Collect remaining elements into a vector.
    pub fn collect_remaining(&mut self) -> Vec<Value> {
        let mut result = Vec::new();
        while let Some(v) = self.next() {
            result.push(v);
        }
        result
    }
}

impl fmt::Debug for IteratorObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind_name = match &self.kind {
            IterKind::Range(_) => "range_iterator",
            IterKind::List { .. } => "list_iterator",
            IterKind::Tuple { .. } => "tuple_iterator",
            IterKind::StringChars { .. } => "str_iterator",
            IterKind::Values { .. } => "iterator",
            IterKind::Empty => "empty_iterator",
        };
        write!(f, "<{}>", kind_name)
    }
}

impl PyObject for IteratorObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::range::RangeObject;

    #[test]
    fn test_empty_iterator() {
        let mut iter = IteratorObject::empty();
        assert!(iter.is_exhausted());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_range_iterator() {
        let range = RangeObject::from_stop(5);
        let mut iter = IteratorObject::from_range(range.iter());

        let mut values = Vec::new();
        while let Some(v) = iter.next() {
            values.push(v.as_int().unwrap());
        }
        assert_eq!(values, vec![0, 1, 2, 3, 4]);
        assert!(iter.is_exhausted());
    }

    #[test]
    fn test_list_iterator() {
        let list = Arc::new(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));
        let mut iter = IteratorObject::from_list(list);

        let mut values = Vec::new();
        while let Some(v) = iter.next() {
            values.push(v.as_int().unwrap());
        }
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_tuple_iterator() {
        let tuple = Arc::new(TupleObject::from_slice(&[
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
        ]));
        let mut iter = IteratorObject::from_tuple(tuple);

        let mut values = Vec::new();
        while let Some(v) = iter.next() {
            values.push(v.as_int().unwrap());
        }
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn test_values_iterator() {
        let values = vec![Value::int(100).unwrap(), Value::int(200).unwrap()];
        let mut iter = IteratorObject::from_values(values);

        assert_eq!(iter.size_hint(), Some(2));
        assert_eq!(iter.next().unwrap().as_int().unwrap(), 100);
        assert_eq!(iter.size_hint(), Some(1));
        assert_eq!(iter.next().unwrap().as_int().unwrap(), 200);
        assert_eq!(iter.size_hint(), Some(0));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_collect_remaining() {
        let range = RangeObject::new(0, 5, 1);
        let mut iter = IteratorObject::from_range(range.iter());

        // Consume first two
        iter.next();
        iter.next();

        // Collect remaining
        let remaining = iter.collect_remaining();
        assert_eq!(remaining.len(), 3);
        assert_eq!(remaining[0].as_int().unwrap(), 2);
        assert_eq!(remaining[1].as_int().unwrap(), 3);
        assert_eq!(remaining[2].as_int().unwrap(), 4);
    }

    #[test]
    fn test_iterator_debug() {
        let iter = IteratorObject::empty();
        let debug = format!("{:?}", iter);
        assert!(debug.contains("empty_iterator"));
    }
}
