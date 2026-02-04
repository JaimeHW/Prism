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
// Tuple creation helpers for composite iterators
// =============================================================================

/// Create a (index, value) tuple for enumerate.
///
/// # Performance
/// Uses Box::leak for now, should integrate with GC in production.
#[inline]
fn create_tuple_pair(index: i64, value: Value) -> Value {
    let tuple = TupleObject::from_slice(&[Value::int(index).unwrap_or(Value::none()), value]);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    Value::object_ptr(ptr)
}

/// Create a tuple from a vector of values.
#[inline]
fn create_tuple_from_values(values: Vec<Value>) -> Value {
    let tuple = TupleObject::from_slice(&values);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    Value::object_ptr(ptr)
}

/// Create a (key, value) tuple for dict items.
#[inline]
fn create_tuple_pair_values(key: Value, value: Value) -> Value {
    let tuple = TupleObject::from_slice(&[key, value]);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    Value::object_ptr(ptr)
}

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

    // =========================================================================
    // Composite iterators (Phase 3.4 Extensions)
    // =========================================================================
    /// Enumerate iterator - yields (index, value) tuples.
    ///
    /// # Performance
    /// - O(1) per iteration step
    /// - Single allocation for boxed inner iterator
    Enumerate {
        inner: Box<IteratorObject>,
        index: i64,
    },

    /// Zip iterator - parallel iteration over multiple iterables.
    ///
    /// # Performance
    /// - O(k) per step where k = number of iterators
    /// - Stops on shortest iterator (Python semantics)
    Zip { iterators: Vec<IteratorObject> },

    /// Map iterator - applies function to each element.
    ///
    /// # Note
    /// Map requires a callback mechanism. For now, stores function Value
    /// that will be called by VM when iterating.
    Map {
        func: Value,
        inner: Box<IteratorObject>,
    },

    /// Filter iterator - yields elements where predicate is truthy.
    ///
    /// # Note
    /// Filter requires callback for predicate. When func is None,
    /// acts as identity filter (filters falsy values).
    Filter {
        func: Option<Value>,
        inner: Box<IteratorObject>,
    },

    /// Reversed iterator - iterates in reverse order.
    ///
    /// # Performance
    /// - Materializes sequence on creation: O(n) space
    /// - O(1) per iteration step
    Reversed {
        values: Vec<Value>,
        /// Index counting back from end (starts at values.len() - 1)
        reverse_index: usize,
    },

    /// Dict keys iterator.
    ///
    /// # Performance
    /// - O(1) per step with index-based access
    DictKeys { keys: Vec<Value>, index: usize },

    /// Dict values iterator.
    DictValues { values: Vec<Value>, index: usize },

    /// Dict items iterator - yields (key, value) tuples.
    DictItems {
        items: Vec<(Value, Value)>,
        index: usize,
    },

    /// Set iterator.
    SetIter { values: Vec<Value>, index: usize },
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

    // =========================================================================
    // Composite iterator constructors (Phase 3.4)
    // =========================================================================

    /// Create an enumerate iterator.
    ///
    /// # Arguments
    /// * `inner` - The iterator to enumerate
    /// * `start` - Starting index (default 0)
    ///
    /// # Performance
    /// - O(1) construction
    /// - O(1) per iteration step
    #[inline]
    pub fn enumerate(inner: IteratorObject, start: i64) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Enumerate {
                inner: Box::new(inner),
                index: start,
            },
            exhausted: false,
        }
    }

    /// Create a zip iterator over multiple iterators.
    ///
    /// # Performance
    /// - O(k) construction where k = number of iterators
    /// - O(k) per iteration step
    /// - Terminates when any iterator is exhausted
    #[inline]
    pub fn zip(iterators: Vec<IteratorObject>) -> Self {
        let exhausted = iterators.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Zip { iterators },
            exhausted,
        }
    }

    /// Create a map iterator.
    ///
    /// # Note
    /// The function must be called externally when iterating.
    /// This iterator stores the function and yields elements that
    /// need to be passed through the function by the caller.
    #[inline]
    pub fn map(func: Value, inner: IteratorObject) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Map {
                func,
                inner: Box::new(inner),
            },
            exhausted: false,
        }
    }

    /// Create a filter iterator.
    ///
    /// # Arguments
    /// * `func` - Predicate function, or None for identity filter (filters falsy)
    /// * `inner` - Iterator to filter
    ///
    /// # Note
    /// For now, when `func` is Some, the predicate must be evaluated externally.
    /// When `func` is None, performs identity filtering on truthiness.
    #[inline]
    pub fn filter(func: Option<Value>, inner: IteratorObject) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Filter {
                func,
                inner: Box::new(inner),
            },
            exhausted: false,
        }
    }

    /// Create a reversed iterator from a sequence.
    ///
    /// # Performance
    /// - O(n) construction (materializes the sequence)
    /// - O(1) per iteration step
    #[inline]
    pub fn reversed(values: Vec<Value>) -> Self {
        let len = values.len();
        let exhausted = values.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Reversed {
                values,
                reverse_index: len,
            },
            exhausted,
        }
    }

    /// Create a dict keys iterator.
    #[inline]
    pub fn dict_keys(keys: Vec<Value>) -> Self {
        let exhausted = keys.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::DictKeys { keys, index: 0 },
            exhausted,
        }
    }

    /// Create a dict values iterator.
    #[inline]
    pub fn dict_values(values: Vec<Value>) -> Self {
        let exhausted = values.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::DictValues { values, index: 0 },
            exhausted,
        }
    }

    /// Create a dict items iterator.
    #[inline]
    pub fn dict_items(items: Vec<(Value, Value)>) -> Self {
        let exhausted = items.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::DictItems { items, index: 0 },
            exhausted,
        }
    }

    /// Create a set iterator.
    #[inline]
    pub fn set_iter(values: Vec<Value>) -> Self {
        let exhausted = values.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::SetIter { values, index: 0 },
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

            // =================================================================
            // Composite iterator implementations
            // =================================================================
            IterKind::Enumerate { inner, index } => {
                match inner.next() {
                    Some(value) => {
                        let idx = *index;
                        *index += 1;
                        // Return (index, value) as a 2-tuple
                        Some(create_tuple_pair(idx, value))
                    }
                    None => {
                        self.exhausted = true;
                        None
                    }
                }
            }

            IterKind::Zip { iterators } => {
                if iterators.is_empty() {
                    self.exhausted = true;
                    return None;
                }

                // Collect one element from each iterator
                let mut values = Vec::with_capacity(iterators.len());
                for iter in iterators.iter_mut() {
                    match iter.next() {
                        Some(v) => values.push(v),
                        None => {
                            // Any exhausted iterator ends zip
                            self.exhausted = true;
                            return None;
                        }
                    }
                }

                Some(create_tuple_from_values(values))
            }

            IterKind::Map { func: _, inner } => {
                // Map iterator: returns raw value, caller must apply function
                // This is a "lazy" map that requires VM integration for the call
                match inner.next() {
                    Some(value) => Some(value),
                    None => {
                        self.exhausted = true;
                        None
                    }
                }
            }

            IterKind::Filter { func, inner } => {
                // Identity filter when func is None: skip falsy values
                if func.is_none() {
                    loop {
                        match inner.next() {
                            Some(value) => {
                                if value.is_truthy() {
                                    return Some(value);
                                }
                                // Skip falsy, continue loop
                            }
                            None => {
                                self.exhausted = true;
                                return None;
                            }
                        }
                    }
                } else {
                    // With predicate function: caller must evaluate
                    // For now, just return next value (VM must handle predicate)
                    match inner.next() {
                        Some(value) => Some(value),
                        None => {
                            self.exhausted = true;
                            None
                        }
                    }
                }
            }

            IterKind::Reversed {
                values,
                reverse_index,
            } => {
                if *reverse_index == 0 {
                    self.exhausted = true;
                    return None;
                }
                *reverse_index -= 1;
                Some(values[*reverse_index])
            }

            IterKind::DictKeys { keys, index } => {
                if *index < keys.len() {
                    let key = keys[*index];
                    *index += 1;
                    Some(key)
                } else {
                    self.exhausted = true;
                    None
                }
            }

            IterKind::DictValues { values, index } => {
                if *index < values.len() {
                    let value = values[*index];
                    *index += 1;
                    Some(value)
                } else {
                    self.exhausted = true;
                    None
                }
            }

            IterKind::DictItems { items, index } => {
                if *index < items.len() {
                    let (key, value) = items[*index];
                    *index += 1;
                    Some(create_tuple_pair_values(key, value))
                } else {
                    self.exhausted = true;
                    None
                }
            }

            IterKind::SetIter { values, index } => {
                if *index < values.len() {
                    let value = values[*index];
                    *index += 1;
                    Some(value)
                } else {
                    self.exhausted = true;
                    None
                }
            }
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

            // Composite iterators
            IterKind::Enumerate { inner, .. } => inner.size_hint(),
            IterKind::Zip { iterators } => {
                // Minimum of all iterator size hints
                iterators.iter().filter_map(|i| i.size_hint()).min()
            }
            IterKind::Map { inner, .. } => inner.size_hint(),
            IterKind::Filter { .. } => None, // Cannot know without evaluating predicate
            IterKind::Reversed {
                values,
                reverse_index,
            } => Some(*reverse_index.min(&values.len())),
            IterKind::DictKeys { keys, index } => Some(keys.len().saturating_sub(*index)),
            IterKind::DictValues { values, index } => Some(values.len().saturating_sub(*index)),
            IterKind::DictItems { items, index } => Some(items.len().saturating_sub(*index)),
            IterKind::SetIter { values, index } => Some(values.len().saturating_sub(*index)),
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
            // Composite iterators
            IterKind::Enumerate { .. } => "enumerate",
            IterKind::Zip { .. } => "zip",
            IterKind::Map { .. } => "map",
            IterKind::Filter { .. } => "filter",
            IterKind::Reversed { .. } => "reversed",
            IterKind::DictKeys { .. } => "dict_keys",
            IterKind::DictValues { .. } => "dict_values",
            IterKind::DictItems { .. } => "dict_items",
            IterKind::SetIter { .. } => "set_iterator",
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

    // =========================================================================
    // Composite iterator tests (Phase 3.4)
    // =========================================================================

    #[test]
    fn test_enumerate_basic() {
        let list = Arc::new(ListObject::from_slice(&[
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
        ]));
        let inner = IteratorObject::from_list(list);
        let mut enumerate = IteratorObject::enumerate(inner, 0);

        assert_eq!(format!("{:?}", enumerate), "<enumerate>");

        // First: (0, 10)
        let pair1 = enumerate.next().unwrap();
        assert!(!pair1.is_none());

        // Second: (1, 20)
        let pair2 = enumerate.next().unwrap();
        assert!(!pair2.is_none());

        // Third: (2, 30)
        let pair3 = enumerate.next().unwrap();
        assert!(!pair3.is_none());

        // Exhausted
        assert!(enumerate.next().is_none());
        assert!(enumerate.is_exhausted());
    }

    #[test]
    fn test_enumerate_with_start() {
        let values = vec![Value::int(100).unwrap(), Value::int(200).unwrap()];
        let inner = IteratorObject::from_values(values);
        let mut enumerate = IteratorObject::enumerate(inner, 5);

        // First: (5, 100)
        let pair1 = enumerate.next().unwrap();
        assert!(!pair1.is_none());

        // Second: (6, 200)
        let pair2 = enumerate.next().unwrap();
        assert!(!pair2.is_none());

        assert!(enumerate.next().is_none());
    }

    #[test]
    fn test_enumerate_empty() {
        let inner = IteratorObject::empty();
        let mut enumerate = IteratorObject::enumerate(inner, 0);
        assert!(enumerate.next().is_none());
    }

    #[test]
    fn test_zip_two_iterators() {
        let list1 = Arc::new(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));
        let list2 = Arc::new(ListObject::from_slice(&[
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
        ]));
        let iter1 = IteratorObject::from_list(list1);
        let iter2 = IteratorObject::from_list(list2);
        let mut zip_iter = IteratorObject::zip(vec![iter1, iter2]);

        assert_eq!(format!("{:?}", zip_iter), "<zip>");

        // Should yield 3 tuples
        let t1 = zip_iter.next();
        assert!(t1.is_some());
        let t2 = zip_iter.next();
        assert!(t2.is_some());
        let t3 = zip_iter.next();
        assert!(t3.is_some());

        assert!(zip_iter.next().is_none());
        assert!(zip_iter.is_exhausted());
    }

    #[test]
    fn test_zip_unequal_lengths() {
        // Short iterator
        let list1 = Arc::new(ListObject::from_slice(&[Value::int(1).unwrap()]));
        // Long iterator
        let list2 = Arc::new(ListObject::from_slice(&[
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
        ]));
        let iter1 = IteratorObject::from_list(list1);
        let iter2 = IteratorObject::from_list(list2);
        let mut zip_iter = IteratorObject::zip(vec![iter1, iter2]);

        // Only 1 element because first iterator has only 1
        assert!(zip_iter.next().is_some());
        assert!(zip_iter.next().is_none());
    }

    #[test]
    fn test_zip_empty() {
        let mut zip_iter = IteratorObject::zip(vec![]);
        assert!(zip_iter.next().is_none());
        assert!(zip_iter.is_exhausted());
    }

    #[test]
    fn test_zip_size_hint() {
        let list1 = Arc::new(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]));
        let list2 = Arc::new(ListObject::from_slice(&[
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
        ]));
        let iter1 = IteratorObject::from_list(list1);
        let iter2 = IteratorObject::from_list(list2);
        let zip_iter = IteratorObject::zip(vec![iter1, iter2]);

        // Should be minimum of the two
        assert_eq!(zip_iter.size_hint(), Some(2));
    }

    #[test]
    fn test_reversed_basic() {
        let values = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];
        let mut reversed = IteratorObject::reversed(values);

        assert_eq!(format!("{:?}", reversed), "<reversed>");
        assert_eq!(reversed.size_hint(), Some(3));

        assert_eq!(reversed.next().unwrap().as_int().unwrap(), 3);
        assert_eq!(reversed.next().unwrap().as_int().unwrap(), 2);
        assert_eq!(reversed.next().unwrap().as_int().unwrap(), 1);
        assert!(reversed.next().is_none());
        assert!(reversed.is_exhausted());
    }

    #[test]
    fn test_reversed_empty() {
        let mut reversed = IteratorObject::reversed(vec![]);
        assert!(reversed.is_exhausted());
        assert!(reversed.next().is_none());
        assert_eq!(reversed.size_hint(), Some(0));
    }

    #[test]
    fn test_reversed_single() {
        let values = vec![Value::int(42).unwrap()];
        let mut reversed = IteratorObject::reversed(values);

        assert_eq!(reversed.next().unwrap().as_int().unwrap(), 42);
        assert!(reversed.next().is_none());
    }

    #[test]
    fn test_dict_keys_iterator() {
        let keys = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];
        let mut iter = IteratorObject::dict_keys(keys);

        assert_eq!(format!("{:?}", iter), "<dict_keys>");
        assert_eq!(iter.size_hint(), Some(3));

        assert_eq!(iter.next().unwrap().as_int().unwrap(), 1);
        assert_eq!(iter.next().unwrap().as_int().unwrap(), 2);
        assert_eq!(iter.next().unwrap().as_int().unwrap(), 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dict_values_iterator() {
        let values = vec![Value::int(100).unwrap(), Value::int(200).unwrap()];
        let mut iter = IteratorObject::dict_values(values);

        assert_eq!(format!("{:?}", iter), "<dict_values>");

        assert_eq!(iter.next().unwrap().as_int().unwrap(), 100);
        assert_eq!(iter.next().unwrap().as_int().unwrap(), 200);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dict_items_iterator() {
        let items = vec![
            (Value::int(1).unwrap(), Value::int(100).unwrap()),
            (Value::int(2).unwrap(), Value::int(200).unwrap()),
        ];
        let mut iter = IteratorObject::dict_items(items);

        assert_eq!(format!("{:?}", iter), "<dict_items>");
        assert_eq!(iter.size_hint(), Some(2));

        // Returns tuples
        let item1 = iter.next();
        assert!(item1.is_some());
        let item2 = iter.next();
        assert!(item2.is_some());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_set_iterator() {
        let values = vec![
            Value::int(5).unwrap(),
            Value::int(10).unwrap(),
            Value::int(15).unwrap(),
        ];
        let mut iter = IteratorObject::set_iter(values);

        assert_eq!(format!("{:?}", iter), "<set_iterator>");

        assert_eq!(iter.next().unwrap().as_int().unwrap(), 5);
        assert_eq!(iter.next().unwrap().as_int().unwrap(), 10);
        assert_eq!(iter.next().unwrap().as_int().unwrap(), 15);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_filter_identity() {
        // Identity filter: filters out falsy values
        let values = vec![
            Value::int(0).unwrap(), // falsy
            Value::int(1).unwrap(), // truthy
            Value::int(0).unwrap(), // falsy
            Value::int(2).unwrap(), // truthy
            Value::none(),          // falsy
            Value::int(3).unwrap(), // truthy
        ];
        let inner = IteratorObject::from_values(values);
        let mut filter = IteratorObject::filter(None, inner);

        assert_eq!(format!("{:?}", filter), "<filter>");

        // Should only yield truthy values: 1, 2, 3
        assert_eq!(filter.next().unwrap().as_int().unwrap(), 1);
        assert_eq!(filter.next().unwrap().as_int().unwrap(), 2);
        assert_eq!(filter.next().unwrap().as_int().unwrap(), 3);
        assert!(filter.next().is_none());
    }

    #[test]
    fn test_filter_all_falsy() {
        let values = vec![
            Value::int(0).unwrap(),
            Value::none(),
            Value::int(0).unwrap(),
        ];
        let inner = IteratorObject::from_values(values);
        let mut filter = IteratorObject::filter(None, inner);

        // All falsy, should yield nothing
        assert!(filter.next().is_none());
    }

    #[test]
    fn test_filter_all_truthy() {
        let values = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];
        let inner = IteratorObject::from_values(values);
        let mut filter = IteratorObject::filter(None, inner);

        // All truthy
        assert_eq!(filter.next().unwrap().as_int().unwrap(), 1);
        assert_eq!(filter.next().unwrap().as_int().unwrap(), 2);
        assert_eq!(filter.next().unwrap().as_int().unwrap(), 3);
        assert!(filter.next().is_none());
    }

    #[test]
    fn test_map_basic() {
        // Map iterator stores func but returns raw values for VM to process
        let values = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
        let inner = IteratorObject::from_values(values);
        let map_iter = IteratorObject::map(Value::none(), inner);

        assert_eq!(format!("{:?}", map_iter), "<map>");
    }

    #[test]
    fn test_enumerate_size_hint() {
        let values = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];
        let inner = IteratorObject::from_values(values);
        let enumerate = IteratorObject::enumerate(inner, 0);

        assert_eq!(enumerate.size_hint(), Some(3));
    }

    #[test]
    fn test_composite_iterator_debug_formats() {
        // Verify all debug formats are correct
        let empty_vals: Vec<Value> = vec![];
        let single_val = vec![Value::int(1).unwrap()];

        assert!(
            format!(
                "{:?}",
                IteratorObject::enumerate(IteratorObject::empty(), 0)
            )
            .contains("enumerate")
        );
        assert!(format!("{:?}", IteratorObject::zip(vec![])).contains("zip"));
        assert!(
            format!(
                "{:?}",
                IteratorObject::map(Value::none(), IteratorObject::empty())
            )
            .contains("map")
        );
        assert!(
            format!(
                "{:?}",
                IteratorObject::filter(None, IteratorObject::empty())
            )
            .contains("filter")
        );
        assert!(format!("{:?}", IteratorObject::reversed(empty_vals.clone())).contains("reversed"));
        assert!(
            format!("{:?}", IteratorObject::dict_keys(empty_vals.clone())).contains("dict_keys")
        );
        assert!(
            format!("{:?}", IteratorObject::dict_values(empty_vals.clone()))
                .contains("dict_values")
        );
        assert!(format!("{:?}", IteratorObject::dict_items(vec![])).contains("dict_items"));
        assert!(format!("{:?}", IteratorObject::set_iter(single_val)).contains("set_iterator"));
    }
}
