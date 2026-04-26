//! Python iterator protocol implementations.
//!
//! Provides a unified iterator type that can wrap different iterable objects
//! and implements the Python iteration protocol.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::bytes::BytesObject;
use crate::types::list::{ListObject, value_as_list_ref};
use crate::types::range::RangeIterator;
use crate::types::string::StringObject;
use crate::types::tuple::TupleObject;
use prism_core::Value;
use prism_core::intern::{InternedString, interned_by_ptr};
use std::fmt;

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

    /// Infinite arithmetic progression, used by `itertools.count()`.
    Count {
        current: CountState,
        step: CountStep,
    },

    /// Repeats one value either forever or for a bounded number of iterations.
    Repeat {
        value: Value,
        remaining: Option<usize>,
    },

    /// Iterator over a list.
    List { list: Value, index: usize },

    /// Iterator over a tuple.
    Tuple { tuple: Value, index: usize },

    /// Iterator over string characters.
    StringChars {
        string: Value,
        /// Byte offset into UTF-8 string.
        byte_offset: usize,
    },

    /// Iterator over bytes / bytearray.
    Bytes { bytes: Value, index: usize },

    /// Iterator over a generic sequence of values.
    /// Used as fallback for custom iterables.
    Values { values: Vec<Value>, index: usize },

    /// Proxy over an existing iterator value.
    ///
    /// This preserves Python's iterator identity semantics for composite
    /// iterator constructors such as `enumerate(iterable)` when the iterable is
    /// already an iterator value. Native Prism iterator objects can be driven
    /// directly; generators and protocol-based iterators are advanced through
    /// the VM-aware `next_with` path.
    SharedIterator { iterator: Value },

    /// Empty iterator.
    Empty,

    // =========================================================================
    // Composite iterators (Phase 3.4 Extensions)
    // =========================================================================
    /// Chain iterator - yields all values from each source iterator in order.
    ///
    /// # Performance
    /// - O(1) per yielded value
    /// - O(k) construction for k source iterables
    /// - Lazy: advances each source only when needed
    Chain {
        iterators: Vec<IteratorObject>,
        current: usize,
    },

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

    /// Islice iterator - yields a stepped slice of another iterator.
    ///
    /// # Performance
    /// - O(start) initial skip by consumption
    /// - O(step) amortized per yielded element
    ISlice {
        inner: Box<IteratorObject>,
        next_yield: usize,
        stop: Option<usize>,
        step: usize,
        pos: usize,
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

enum StringValueRef<'a> {
    Heap(&'a StringObject),
    Interned(InternedString),
}

#[derive(Clone, Copy)]
enum CountState {
    Int(i64),
    Float(f64),
}

#[derive(Clone, Copy)]
enum CountStep {
    Int(i64),
    Float(f64),
}

impl StringValueRef<'_> {
    #[inline(always)]
    fn as_str(&self) -> &str {
        match self {
            Self::Heap(string) => string.as_str(),
            Self::Interned(interned) => interned.as_str(),
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.as_str().is_empty()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.as_str().len()
    }
}

#[inline(always)]
fn object_ref<T>(value: Value, expected: TypeId) -> &'static T {
    let ptr = value
        .as_object_ptr()
        .expect("iterator backing value must be an object");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    debug_assert_eq!(header.type_id, expected);
    unsafe { &*(ptr as *const T) }
}

#[inline(always)]
fn list_from_value(value: Value) -> &'static ListObject {
    value_as_list_ref(value).expect("iterator backing value must provide list storage")
}

#[inline(always)]
fn tuple_from_value(value: Value) -> &'static TupleObject {
    object_ref(value, TypeId::TUPLE)
}

#[inline(always)]
fn string_from_value(value: Value) -> StringValueRef<'static> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .expect("string iterator backing value must provide a string payload")
            as *const u8;
        let interned = interned_by_ptr(ptr)
            .expect("string iterator backing value must resolve through the interner");
        return StringValueRef::Interned(interned);
    }

    StringValueRef::Heap(object_ref(value, TypeId::STR))
}

#[inline(always)]
fn bytes_from_value(value: Value) -> &'static BytesObject {
    let ptr = value
        .as_object_ptr()
        .expect("bytes iterator backing value must be an object");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    debug_assert!(matches!(header.type_id, TypeId::BYTES | TypeId::BYTEARRAY));
    unsafe { &*(ptr as *const BytesObject) }
}

#[inline(always)]
fn iterator_from_value(value: Value) -> &'static IteratorObject {
    object_ref(value, TypeId::ITERATOR)
}

#[inline(always)]
fn native_iterator_from_value(value: Value) -> Option<&'static IteratorObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::ITERATOR {
        return None;
    }

    Some(unsafe { &*(ptr as *const IteratorObject) })
}

#[inline(always)]
fn iterator_from_value_mut(value: Value) -> &'static mut IteratorObject {
    let ptr = value
        .as_object_ptr()
        .expect("iterator proxy backing value must be an object");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    debug_assert_eq!(header.type_id, TypeId::ITERATOR);
    unsafe { &mut *(ptr as *mut IteratorObject) }
}

#[inline(always)]
fn native_iterator_from_value_mut(value: Value) -> Option<&'static mut IteratorObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::ITERATOR {
        return None;
    }

    Some(unsafe { &mut *(ptr as *mut IteratorObject) })
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

    /// Create an infinite count iterator.
    #[inline]
    pub fn count(start: Value, step: Value) -> Option<Self> {
        let (current, step) = match (start.as_int(), step.as_int()) {
            (Some(start), Some(step)) => (CountState::Int(start), CountStep::Int(step)),
            _ => {
                let start = start
                    .as_float()
                    .or_else(|| start.as_int().map(|v| v as f64))?;
                let step = step
                    .as_float()
                    .or_else(|| step.as_int().map(|v| v as f64))?;
                (CountState::Float(start), CountStep::Float(step))
            }
        };

        Some(Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Count { current, step },
            exhausted: false,
        })
    }

    /// Create a repeat iterator.
    #[inline]
    pub fn repeat(value: Value, remaining: Option<usize>) -> Self {
        let exhausted = matches!(remaining, Some(0));
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Repeat { value, remaining },
            exhausted,
        }
    }

    /// Create an iterator over a list.
    #[inline]
    pub fn from_list(list: Value) -> Self {
        let list_ref = list_from_value(list);
        let exhausted = list_ref.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::List { list, index: 0 },
            exhausted,
        }
    }

    /// Create an iterator over a tuple.
    #[inline]
    pub fn from_tuple(tuple: Value) -> Self {
        let tuple_ref = tuple_from_value(tuple);
        let exhausted = tuple_ref.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Tuple { tuple, index: 0 },
            exhausted,
        }
    }

    /// Create an iterator over string characters.
    #[inline]
    pub fn from_string_chars(string: Value) -> Self {
        let string_ref = string_from_value(string);
        let exhausted = string_ref.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::StringChars {
                string,
                byte_offset: 0,
            },
            exhausted,
        }
    }

    /// Create an iterator over bytes or bytearray.
    #[inline]
    pub fn from_bytes(bytes: Value) -> Self {
        let bytes_ref = bytes_from_value(bytes);
        let exhausted = bytes_ref.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Bytes { bytes, index: 0 },
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

    /// Create a proxy iterator that forwards to an existing iterator object.
    #[inline]
    pub fn from_existing_iterator(iterator: Value) -> Self {
        let exhausted = native_iterator_from_value(iterator)
            .map(IteratorObject::is_exhausted)
            .unwrap_or(false);
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::SharedIterator { iterator },
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

    /// Create an islice iterator over another iterator.
    #[inline]
    pub fn islice(inner: IteratorObject, start: usize, stop: Option<usize>, step: usize) -> Self {
        let exhausted =
            inner.is_exhausted() || matches!(stop, Some(stop_index) if start >= stop_index);
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::ISlice {
                inner: Box::new(inner),
                next_yield: start,
                stop,
                step: step.max(1),
                pos: 0,
            },
            exhausted,
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

    /// Create a chain iterator over a sequence of iterators.
    #[inline]
    pub fn chain(iterators: Vec<IteratorObject>) -> Self {
        let exhausted = iterators.is_empty();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::Chain {
                iterators,
                current: 0,
            },
            exhausted,
        }
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
                Some(v) => Some(v),
                None => {
                    self.exhausted = true;
                    None
                }
            },

            IterKind::Count { current, step } => match (*current, *step) {
                (CountState::Int(value), CountStep::Int(delta)) => {
                    *current = CountState::Int(value.wrapping_add(delta));
                    Some(Value::int_unchecked(value))
                }
                (CountState::Int(value), CountStep::Float(delta)) => {
                    let value = value as f64;
                    *current = CountState::Float(value + delta);
                    Some(Value::float(value))
                }
                (CountState::Float(value), CountStep::Int(delta)) => {
                    *current = CountState::Float(value + delta as f64);
                    Some(Value::float(value))
                }
                (CountState::Float(value), CountStep::Float(delta)) => {
                    *current = CountState::Float(value + delta);
                    Some(Value::float(value))
                }
            },

            IterKind::Repeat { value, remaining } => match remaining {
                Some(0) => {
                    self.exhausted = true;
                    None
                }
                Some(count) => {
                    *count -= 1;
                    Some(*value)
                }
                None => Some(*value),
            },

            IterKind::List { list, index } => {
                let list = list_from_value(*list);
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
                let tuple = tuple_from_value(*tuple);
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
                let string = string_from_value(*string);
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

            IterKind::Bytes { bytes, index } => {
                let bytes = bytes_from_value(*bytes);
                if *index < bytes.len() {
                    let value = Value::int(bytes.as_bytes()[*index] as i64);
                    *index += 1;
                    value
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

            IterKind::SharedIterator { iterator } => {
                match native_iterator_from_value_mut(*iterator) {
                    Some(iter) => {
                        let value = iter.next();
                        self.exhausted = iter.is_exhausted();
                        value
                    }
                    None => {
                        self.exhausted = true;
                        None
                    }
                }
            }

            IterKind::Empty => None,

            // =================================================================
            // Composite iterator implementations
            // =================================================================
            IterKind::Chain { iterators, current } => {
                while *current < iterators.len() {
                    if let Some(value) = iterators[*current].next() {
                        return Some(value);
                    }
                    *current += 1;
                }

                self.exhausted = true;
                None
            }

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

            IterKind::ISlice {
                inner,
                next_yield,
                stop,
                step,
                pos,
            } => {
                if stop.is_some_and(|stop_index| *next_yield >= stop_index) {
                    self.exhausted = true;
                    return None;
                }

                while *pos < *next_yield {
                    if inner.next().is_none() {
                        self.exhausted = true;
                        return None;
                    }
                    *pos += 1;
                }

                match inner.next() {
                    Some(value) => {
                        *pos += 1;
                        *next_yield = next_yield.saturating_add(*step);
                        Some(value)
                    }
                    None => {
                        self.exhausted = true;
                        None
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
            IterKind::Range(iter) => iter.remaining_len(),
            IterKind::Count { .. } => None,
            IterKind::Repeat { remaining, .. } => *remaining,
            IterKind::List { list, index } => {
                Some(list_from_value(*list).len().saturating_sub(*index))
            }
            IterKind::Tuple { tuple, index } => {
                Some(tuple_from_value(*tuple).len().saturating_sub(*index))
            }
            IterKind::StringChars {
                string,
                byte_offset,
            } => {
                // We can't know exactly without scanning, so return None
                // Could count remaining chars but that's O(n)
                let remaining_bytes = string_from_value(*string)
                    .len()
                    .saturating_sub(*byte_offset);
                if remaining_bytes == 0 {
                    Some(0)
                } else {
                    None // Unknown without counting
                }
            }
            IterKind::Bytes { bytes, index } => {
                Some(bytes_from_value(*bytes).len().saturating_sub(*index))
            }
            IterKind::Values { values, index } => Some(values.len().saturating_sub(*index)),
            IterKind::SharedIterator { iterator } => {
                native_iterator_from_value(*iterator).and_then(IteratorObject::size_hint)
            }
            IterKind::Empty => Some(0),

            // Composite iterators
            IterKind::Chain { iterators, current } => {
                let mut total = 0usize;
                for iterator in iterators.iter().skip(*current) {
                    total = total.checked_add(iterator.size_hint()?)?;
                }
                Some(total)
            }
            IterKind::Enumerate { inner, .. } => inner.size_hint(),
            IterKind::Zip { iterators } => {
                // Minimum of all iterator size hints
                iterators.iter().filter_map(|i| i.size_hint()).min()
            }
            IterKind::Map { inner, .. } => inner.size_hint(),
            IterKind::Filter { .. } => None, // Cannot know without evaluating predicate
            IterKind::ISlice {
                next_yield,
                stop,
                step,
                ..
            } => match stop {
                Some(stop_index) if *next_yield >= *stop_index => Some(0),
                Some(stop_index) => Some((stop_index - next_yield + step - 1) / step),
                None => None,
            },
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

    /// Advance the iterator using VM-provided callable and truthiness hooks.
    ///
    /// This is required for iterators such as `map` and `filter` whose
    /// semantics depend on evaluating Python callables while iterating.
    pub fn next_with<E, F, T, N>(
        &mut self,
        invoke: &mut F,
        is_truthy: &mut T,
        advance_iterator: &mut N,
    ) -> Result<Option<Value>, E>
    where
        F: FnMut(Value, &[Value]) -> Result<Value, E>,
        T: FnMut(Value) -> Result<bool, E>,
        N: FnMut(Value) -> Result<Option<Value>, E>,
    {
        if self.exhausted {
            return Ok(None);
        }

        match &mut self.kind {
            IterKind::SharedIterator { iterator } => {
                let next = if let Some(iter) = native_iterator_from_value_mut(*iterator) {
                    let next = iter.next_with(invoke, is_truthy, advance_iterator)?;
                    self.exhausted = iter.is_exhausted();
                    next
                } else {
                    let next = advance_iterator(*iterator)?;
                    self.exhausted = next.is_none();
                    next
                };
                return Ok(next);
            }
            IterKind::Map { func, inner } => {
                return match inner.next_with(invoke, is_truthy, advance_iterator)? {
                    Some(value) => invoke(*func, &[value]).map(Some),
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                };
            }
            IterKind::Enumerate { inner, index } => {
                return match inner.next_with(invoke, is_truthy, advance_iterator)? {
                    Some(value) => {
                        let current_index = *index;
                        *index += 1;
                        Ok(Some(create_tuple_pair(current_index, value)))
                    }
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                };
            }
            IterKind::Filter { func, inner } => loop {
                match inner.next_with(invoke, is_truthy, advance_iterator)? {
                    Some(value) => {
                        let keep = if let Some(predicate) = *func {
                            let predicate_result = invoke(predicate, &[value])?;
                            is_truthy(predicate_result)?
                        } else {
                            is_truthy(value)?
                        };

                        if keep {
                            return Ok(Some(value));
                        }
                    }
                    None => {
                        self.exhausted = true;
                        return Ok(None);
                    }
                }
            },
            IterKind::Chain { iterators, current } => {
                while *current < iterators.len() {
                    match iterators[*current].next_with(invoke, is_truthy, advance_iterator)? {
                        Some(value) => return Ok(Some(value)),
                        None => *current += 1,
                    }
                }

                self.exhausted = true;
                return Ok(None);
            }
            IterKind::ISlice {
                inner,
                next_yield,
                stop,
                step,
                pos,
            } => {
                if stop.is_some_and(|stop_index| *next_yield >= stop_index) {
                    self.exhausted = true;
                    return Ok(None);
                }

                while *pos < *next_yield {
                    if inner
                        .next_with(invoke, is_truthy, advance_iterator)?
                        .is_none()
                    {
                        self.exhausted = true;
                        return Ok(None);
                    }
                    *pos += 1;
                }

                return match inner.next_with(invoke, is_truthy, advance_iterator)? {
                    Some(value) => {
                        *pos += 1;
                        *next_yield = next_yield.saturating_add(*step);
                        Ok(Some(value))
                    }
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                };
            }
            _ => {}
        }

        Ok(self.next())
    }
}

impl fmt::Debug for IteratorObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind_name = match &self.kind {
            IterKind::Range(_) => "range_iterator",
            IterKind::Count { .. } => "count",
            IterKind::Repeat { .. } => "repeat",
            IterKind::List { .. } => "list_iterator",
            IterKind::Tuple { .. } => "tuple_iterator",
            IterKind::StringChars { .. } => "str_iterator",
            IterKind::Bytes { .. } => "bytes_iterator",
            IterKind::Values { .. } => "iterator",
            IterKind::SharedIterator { .. } => "iterator",
            IterKind::Empty => "empty_iterator",
            // Composite iterators
            IterKind::Chain { .. } => "chain",
            IterKind::Enumerate { .. } => "enumerate",
            IterKind::Zip { .. } => "zip",
            IterKind::Map { .. } => "map",
            IterKind::Filter { .. } => "filter",
            IterKind::ISlice { .. } => "islice",
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
mod tests;
