//! Python iterator protocol implementations.
//!
//! Provides a unified iterator type that can wrap different iterable objects
//! and implements the Python iteration protocol.

use crate::object::type_obj::TypeId;
use crate::object::views::GenericAliasObject;
use crate::object::{ObjectHeader, PyObject};
use crate::types::bytes::BytesObject;
use crate::types::list::{ListObject, value_as_list_ref};
use crate::types::range::RangeIterator;
use crate::types::string::StringObject;
use crate::types::tuple::TupleObject;
use prism_core::Value;
use prism_core::intern::{InternedString, interned_by_ptr};
use std::fmt;

/// Host-provided fast length reader for guarded snapshot iterators.
///
/// The iterator type lives in `prism_runtime`, while some owner objects, such
/// as `collections.deque`, are implemented by `prism_vm`. A function pointer
/// keeps the runtime independent while still allowing O(1) mutation checks.
pub type IteratorLenGuard = fn(Value) -> Option<usize>;

/// Error raised while advancing a native iterator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IteratorAdvanceError {
    message: &'static str,
}

/// Reducer state for Python iterator pickling.
///
/// The VM owns object allocation and builtin lookup, so the runtime exposes a
/// compact description instead of manufacturing Python tuples directly.
#[derive(Debug, Clone)]
pub enum IteratorReduction {
    /// Reconstruct as `iter()` with no arguments, producing an empty iterator.
    Empty,
    /// Reconstruct as `iter(empty_iterable)` for exhausted iterator types whose
    /// CPython reducers preserve an empty source container.
    EmptyIterable(IteratorEmptyIterable),
    /// Reconstruct as `iter(iterable)` and optionally restore an index with
    /// `__setstate__`.
    Iterable { iterable: Value, state: Option<i64> },
    /// Reconstruct as `reversed(iterable)` and optionally restore a reverse
    /// iterator index with `__setstate__`.
    ReversedIterable { iterable: Value, state: Option<i64> },
    /// Reconstruct as `iter(callable, sentinel)`.
    CallSentinel { callable: Value, sentinel: Value },
    /// Reconstruct from a snapshot list of remaining values.
    RemainingValues(Vec<Value>),
    /// This iterator needs VM-side calls to snapshot safely.
    RequiresVm(&'static str),
}

/// Empty iterable payloads used by CPython-compatible iterator reducers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IteratorEmptyIterable {
    /// Empty tuple `()`.
    Tuple,
    /// Empty list `[]`.
    List,
    /// Empty string `""`.
    String,
}

impl IteratorAdvanceError {
    #[inline]
    fn mutated(message: &'static str) -> Self {
        Self { message }
    }

    #[inline]
    pub fn message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for IteratorAdvanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for IteratorAdvanceError {}

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

    /// Snapshot iterator guarded by the source container's current length.
    ///
    /// CPython's dict, set, and deque iterators raise `RuntimeError` once a
    /// size-changing mutation is observed. Capturing a compact snapshot keeps
    /// iteration cache-local while the O(1) guard preserves that contract.
    GuardedValues {
        owner: Value,
        expected_len: usize,
        len_guard: IteratorLenGuard,
        mutation_message: &'static str,
        values: Vec<Value>,
        index: usize,
    },

    /// Proxy over an existing iterator value.
    ///
    /// This preserves Python's iterator identity semantics for composite
    /// iterator constructors such as `enumerate(iterable)` when the iterable is
    /// already an iterator value. Native Prism iterator objects can be driven
    /// directly; generators and protocol-based iterators are advanced through
    /// the VM-aware `next_with` path.
    SharedIterator { iterator: Value },

    /// Lazy iterator over an object that supports the legacy sequence protocol.
    ///
    /// This mirrors CPython's `PySeqIterObject`: each advance calls the cached
    /// `__getitem__` callable with the current non-negative index and treats
    /// `IndexError` as exhaustion. Keeping this lazy is essential for unbounded
    /// sequences such as objects whose `__getitem__` returns the index.
    SequenceGetItem {
        callable: Value,
        self_arg: Option<Value>,
        index: i64,
    },

    /// Iterator returned by `iter(callable, sentinel)`.
    ///
    /// Each step invokes the zero-argument callable and stops when the result
    /// compares equal to the sentinel. The VM provides the call/equality hook
    /// so Python exceptions and rich comparison semantics are preserved.
    CallSentinel { callable: Value, sentinel: Value },

    /// One-shot iterator over a `types.GenericAlias`.
    ///
    /// CPython exposes this for starred typing unpacking: `iter(tuple[int])`
    /// yields a starred alias once, then reduces as `iter(())` after exhaustion.
    GenericAlias { alias: Value, yielded: bool },

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

    /// Reverse iterator over a live list.
    ReversedList { list: Value, reverse_index: usize },

    /// Reverse snapshot iterator guarded by source container length.
    GuardedReversedValues {
        owner: Value,
        expected_len: usize,
        len_guard: IteratorLenGuard,
        mutation_message: &'static str,
        values: Vec<Value>,
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
fn generic_alias_from_value(value: Value) -> &'static GenericAliasObject {
    object_ref(value, TypeId::GENERIC_ALIAS)
}

#[inline]
fn create_starred_generic_alias(alias: Value) -> Value {
    let alias = generic_alias_from_value(alias);
    let object = GenericAliasObject::new_with_starred(alias.origin(), alias.args().to_vec(), true);
    let ptr = Box::leak(Box::new(object)) as *mut GenericAliasObject as *const ();
    Value::object_ptr(ptr)
}

#[inline]
fn byte_offset_for_char_index(text: &str, char_index: usize) -> usize {
    if char_index == 0 {
        return 0;
    }

    text.char_indices()
        .nth(char_index)
        .map_or(text.len(), |(byte_offset, _)| byte_offset)
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
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::List { list, index: 0 },
            exhausted: false,
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

    /// Create a snapshot iterator that checks the backing collection length.
    #[inline]
    pub fn guarded_values(
        owner: Value,
        values: Vec<Value>,
        len_guard: IteratorLenGuard,
        mutation_message: &'static str,
    ) -> Self {
        let exhausted = values.is_empty();
        let expected_len = len_guard(owner).unwrap_or(values.len());
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::GuardedValues {
                owner,
                expected_len,
                len_guard,
                mutation_message,
                values,
                index: 0,
            },
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

    /// Create a lazy iterator from a bound `__getitem__` target.
    #[inline]
    pub fn from_sequence_getitem(callable: Value, self_arg: Option<Value>) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::SequenceGetItem {
                callable,
                self_arg,
                index: 0,
            },
            exhausted: false,
        }
    }

    /// Create an iterator for `iter(callable, sentinel)`.
    #[inline]
    pub fn from_call_sentinel(callable: Value, sentinel: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::CallSentinel { callable, sentinel },
            exhausted: false,
        }
    }

    /// Create the one-shot iterator returned by `types.GenericAlias.__iter__`.
    #[inline]
    pub fn from_generic_alias(alias: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::GenericAlias {
                alias,
                yielded: false,
            },
            exhausted: false,
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

    /// Create a reverse iterator over a live list.
    #[inline]
    pub fn reversed_list(list: Value) -> Self {
        let len = list_from_value(list).len();
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::ReversedList {
                list,
                reverse_index: len,
            },
            exhausted: len == 0,
        }
    }

    /// Create a reverse snapshot iterator guarded by backing collection length.
    #[inline]
    pub fn guarded_reversed_values(
        owner: Value,
        values: Vec<Value>,
        len_guard: IteratorLenGuard,
        mutation_message: &'static str,
    ) -> Self {
        let len = values.len();
        let expected_len = len_guard(owner).unwrap_or(len);
        Self {
            header: ObjectHeader::new(TypeId::ITERATOR),
            kind: IterKind::GuardedReversedValues {
                owner,
                expected_len,
                len_guard,
                mutation_message,
                values,
                reverse_index: len,
            },
            exhausted: len == 0,
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

    /// Return the builtin constructor name used by this iterator's reducer.
    #[inline]
    pub fn reduction_builtin_name(&self) -> &'static str {
        match self.kind {
            IterKind::ReversedList { .. } => "reversed",
            _ => "iter",
        }
    }

    /// Restore an iterator position from a Python `__setstate__` index.
    ///
    /// Negative states clamp to zero, matching CPython iterator behavior. For
    /// finite native iterators, states beyond the available length exhaust the
    /// iterator. For lazy sequence iterators the next `__getitem__` decides
    /// whether the index is in range.
    pub fn set_state(&mut self, state: i64) {
        let raw_state = state;
        let state = raw_state.max(0);
        let usize_state = usize::try_from(state).unwrap_or(usize::MAX);
        self.exhausted = false;

        match &mut self.kind {
            IterKind::List { list, index } => {
                let len = list_from_value(*list).len();
                *index = usize_state.min(len);
            }
            IterKind::Tuple { tuple, index } => {
                let len = tuple_from_value(*tuple).len();
                *index = usize_state.min(len);
                self.exhausted = *index >= len;
            }
            IterKind::StringChars {
                string,
                byte_offset,
            } => {
                let string = string_from_value(*string);
                let text = string.as_str();
                *byte_offset = byte_offset_for_char_index(text, usize_state);
                self.exhausted = *byte_offset >= text.len();
            }
            IterKind::Bytes { bytes, index } => {
                let len = bytes_from_value(*bytes).len();
                *index = usize_state.min(len);
                self.exhausted = *index >= len;
            }
            IterKind::Values { values, index }
            | IterKind::DictKeys {
                keys: values,
                index,
            }
            | IterKind::DictValues { values, index }
            | IterKind::SetIter { values, index } => {
                *index = usize_state.min(values.len());
                self.exhausted = *index >= values.len();
            }
            IterKind::GuardedValues { values, index, .. } => {
                *index = usize_state.min(values.len());
                self.exhausted = *index >= values.len();
            }
            IterKind::DictItems { items, index } => {
                *index = usize_state.min(items.len());
                self.exhausted = *index >= items.len();
            }
            IterKind::SharedIterator { iterator } => {
                if let Some(iter) = native_iterator_from_value_mut(*iterator) {
                    iter.set_state(state);
                    self.exhausted = iter.is_exhausted();
                }
            }
            IterKind::SequenceGetItem { index, .. } => {
                *index = state;
            }
            IterKind::CallSentinel { .. } => {}
            IterKind::GenericAlias { yielded, .. } => {
                *yielded = state > 0;
                self.exhausted = *yielded;
            }
            IterKind::ReversedList {
                list,
                reverse_index,
            } => {
                let len = list_from_value(*list).len();
                if raw_state < 0 || len == 0 {
                    *reverse_index = 0;
                    self.exhausted = true;
                } else {
                    let index = usize::try_from(raw_state)
                        .unwrap_or(usize::MAX)
                        .min(len - 1);
                    *reverse_index = index + 1;
                }
            }
            IterKind::Empty => {
                self.exhausted = true;
            }
            _ => {}
        }
    }

    /// Return reducer state for `iterator.__reduce__`.
    ///
    /// This mirrors CPython's approach: live sequence iterators keep the
    /// original iterable plus an index, callable-sentinel iterators preserve
    /// their callable/sentinel pair, and snapshot-style iterators fall back to
    /// a compact list of remaining values without mutating the iterator.
    pub fn reduction_state(&self) -> Result<IteratorReduction, IteratorAdvanceError> {
        match &self.kind {
            IterKind::Range(iter) => {
                if self.exhausted {
                    return Ok(IteratorReduction::RemainingValues(Vec::new()));
                }
                Ok(IteratorReduction::RemainingValues(
                    iter.clone().collect::<Vec<_>>(),
                ))
            }
            IterKind::Count { .. }
            | IterKind::Repeat {
                remaining: None, ..
            } => Ok(IteratorReduction::RequiresVm(
                "infinite iterator cannot be snapshotted",
            )),
            IterKind::Repeat {
                value,
                remaining: Some(count),
            } => Ok(IteratorReduction::RemainingValues(vec![*value; *count])),
            IterKind::List { list, index } => {
                if self.exhausted {
                    return Ok(IteratorReduction::EmptyIterable(
                        IteratorEmptyIterable::List,
                    ));
                }
                Ok(IteratorReduction::Iterable {
                    iterable: *list,
                    state: Some(i64::try_from(*index).unwrap_or(i64::MAX)),
                })
            }
            IterKind::Tuple { tuple, index } => {
                if self.exhausted {
                    return Ok(IteratorReduction::EmptyIterable(
                        IteratorEmptyIterable::Tuple,
                    ));
                }
                Ok(IteratorReduction::Iterable {
                    iterable: *tuple,
                    state: Some(i64::try_from(*index).unwrap_or(i64::MAX)),
                })
            }
            IterKind::StringChars {
                string,
                byte_offset,
            } => {
                if self.exhausted {
                    return Ok(IteratorReduction::EmptyIterable(
                        IteratorEmptyIterable::String,
                    ));
                }
                let string_ref = string_from_value(*string);
                let text = string_ref.as_str();
                let char_index = text[..(*byte_offset).min(text.len())].chars().count();
                Ok(IteratorReduction::Iterable {
                    iterable: *string,
                    state: Some(i64::try_from(char_index).unwrap_or(i64::MAX)),
                })
            }
            IterKind::Bytes { bytes, index } => {
                if self.exhausted {
                    return Ok(IteratorReduction::EmptyIterable(
                        IteratorEmptyIterable::Tuple,
                    ));
                }
                Ok(IteratorReduction::Iterable {
                    iterable: *bytes,
                    state: Some(i64::try_from(*index).unwrap_or(i64::MAX)),
                })
            }
            IterKind::Values { values, index } => Ok(IteratorReduction::RemainingValues(
                values.iter().skip(*index).copied().collect(),
            )),
            IterKind::GuardedValues {
                owner,
                expected_len,
                len_guard,
                mutation_message,
                values,
                index,
            } => {
                if len_guard(*owner) != Some(*expected_len) {
                    return Err(IteratorAdvanceError::mutated(mutation_message));
                }
                Ok(IteratorReduction::RemainingValues(
                    values.iter().skip(*index).copied().collect(),
                ))
            }
            IterKind::SharedIterator { iterator } => match native_iterator_from_value(*iterator) {
                Some(iter) => iter.reduction_state(),
                None => Ok(IteratorReduction::RequiresVm(
                    "protocol iterator cannot be snapshotted without VM context",
                )),
            },
            IterKind::SequenceGetItem {
                self_arg, index, ..
            } => match self_arg {
                _ if self.exhausted => Ok(IteratorReduction::EmptyIterable(
                    IteratorEmptyIterable::Tuple,
                )),
                Some(iterable) => Ok(IteratorReduction::Iterable {
                    iterable: *iterable,
                    state: Some(*index),
                }),
                None => Ok(IteratorReduction::RequiresVm(
                    "unbound sequence iterator cannot be reduced",
                )),
            },
            IterKind::CallSentinel { callable, sentinel } => {
                if self.exhausted {
                    return Ok(IteratorReduction::EmptyIterable(
                        IteratorEmptyIterable::Tuple,
                    ));
                }
                Ok(IteratorReduction::CallSentinel {
                    callable: *callable,
                    sentinel: *sentinel,
                })
            }
            IterKind::GenericAlias { alias, yielded } => {
                if self.exhausted || *yielded {
                    return Ok(IteratorReduction::EmptyIterable(
                        IteratorEmptyIterable::Tuple,
                    ));
                }
                Ok(IteratorReduction::Iterable {
                    iterable: *alias,
                    state: None,
                })
            }
            IterKind::Empty => Ok(IteratorReduction::EmptyIterable(
                IteratorEmptyIterable::Tuple,
            )),
            IterKind::Chain { .. }
            | IterKind::Enumerate { .. }
            | IterKind::Zip { .. }
            | IterKind::Map { .. }
            | IterKind::Filter { .. }
            | IterKind::ISlice { .. } => Ok(IteratorReduction::RequiresVm(
                "composite iterator requires VM context to reduce",
            )),
            IterKind::Reversed {
                values,
                reverse_index,
            } => Ok(IteratorReduction::RemainingValues(
                values.iter().take(*reverse_index).rev().copied().collect(),
            )),
            IterKind::ReversedList {
                list,
                reverse_index,
            } => {
                if self.exhausted {
                    return Ok(IteratorReduction::EmptyIterable(
                        IteratorEmptyIterable::List,
                    ));
                }
                Ok(IteratorReduction::ReversedIterable {
                    iterable: *list,
                    state: Some(i64::try_from(*reverse_index).unwrap_or(i64::MAX) - 1),
                })
            }
            IterKind::GuardedReversedValues {
                owner,
                expected_len,
                len_guard,
                mutation_message,
                values,
                reverse_index,
            } => {
                if len_guard(*owner) != Some(*expected_len) {
                    return Err(IteratorAdvanceError::mutated(mutation_message));
                }
                Ok(IteratorReduction::RemainingValues(
                    values.iter().take(*reverse_index).rev().copied().collect(),
                ))
            }
            IterKind::DictKeys { keys, index } => Ok(IteratorReduction::RemainingValues(
                keys.iter().skip(*index).copied().collect(),
            )),
            IterKind::DictValues { values, index } => Ok(IteratorReduction::RemainingValues(
                values.iter().skip(*index).copied().collect(),
            )),
            IterKind::DictItems { items, index } => Ok(IteratorReduction::RemainingValues(
                items
                    .iter()
                    .skip(*index)
                    .map(|&(key, value)| create_tuple_pair_values(key, value))
                    .collect(),
            )),
            IterKind::SetIter { values, index } => Ok(IteratorReduction::RemainingValues(
                values.iter().skip(*index).copied().collect(),
            )),
        }
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

    /// Get the next value from the iterator, preserving iterator errors.
    ///
    /// Returns `Some(value)` if there are more elements, `None` if exhausted.
    pub fn next_checked(&mut self) -> Result<Option<Value>, IteratorAdvanceError> {
        if self.exhausted {
            return Ok(None);
        }

        let next = match &mut self.kind {
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
                    return Ok(None);
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

            IterKind::GuardedValues {
                owner,
                expected_len,
                len_guard,
                mutation_message,
                values,
                index,
            } => {
                if len_guard(*owner) != Some(*expected_len) {
                    self.exhausted = true;
                    return Err(IteratorAdvanceError::mutated(mutation_message));
                }
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
                        let value = iter.next_checked()?;
                        self.exhausted = iter.is_exhausted();
                        value
                    }
                    None => {
                        self.exhausted = true;
                        None
                    }
                }
            }

            IterKind::SequenceGetItem { .. } => {
                return Err(IteratorAdvanceError::mutated(
                    "sequence iterator requires VM context",
                ));
            }

            IterKind::CallSentinel { .. } => {
                return Err(IteratorAdvanceError::mutated(
                    "callable iterator requires VM context",
                ));
            }

            IterKind::GenericAlias { alias, yielded } => {
                if *yielded {
                    self.exhausted = true;
                    None
                } else {
                    *yielded = true;
                    Some(create_starred_generic_alias(*alias))
                }
            }

            IterKind::Empty => None,

            // =================================================================
            // Composite iterator implementations
            // =================================================================
            IterKind::Chain { iterators, current } => {
                while *current < iterators.len() {
                    if let Some(value) = iterators[*current].next_checked()? {
                        return Ok(Some(value));
                    }
                    *current += 1;
                }

                self.exhausted = true;
                None
            }

            IterKind::Enumerate { inner, index } => {
                match inner.next_checked()? {
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
                    return Ok(None);
                }

                // Collect one element from each iterator
                let mut values = Vec::with_capacity(iterators.len());
                for iter in iterators.iter_mut() {
                    match iter.next_checked()? {
                        Some(v) => values.push(v),
                        None => {
                            // Any exhausted iterator ends zip
                            self.exhausted = true;
                            return Ok(None);
                        }
                    }
                }

                Some(create_tuple_from_values(values))
            }

            IterKind::Map { func: _, inner } => {
                // Map iterator: returns raw value, caller must apply function
                // This is a "lazy" map that requires VM integration for the call
                match inner.next_checked()? {
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
                        match inner.next_checked()? {
                            Some(value) => {
                                if value.is_truthy() {
                                    return Ok(Some(value));
                                }
                                // Skip falsy, continue loop
                            }
                            None => {
                                self.exhausted = true;
                                return Ok(None);
                            }
                        }
                    }
                } else {
                    // With predicate function: caller must evaluate
                    // For now, just return next value (VM must handle predicate)
                    match inner.next_checked()? {
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
                    return Ok(None);
                }

                while *pos < *next_yield {
                    if inner.next_checked()?.is_none() {
                        self.exhausted = true;
                        return Ok(None);
                    }
                    *pos += 1;
                }

                match inner.next_checked()? {
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
                    return Ok(None);
                }
                *reverse_index -= 1;
                Some(values[*reverse_index])
            }

            IterKind::ReversedList {
                list,
                reverse_index,
            } => {
                let list = list_from_value(*list);
                if *reverse_index == 0 || list.len() < *reverse_index {
                    self.exhausted = true;
                    None
                } else {
                    *reverse_index -= 1;
                    list.get(*reverse_index as i64)
                }
            }

            IterKind::GuardedReversedValues {
                owner,
                expected_len,
                len_guard,
                mutation_message,
                values,
                reverse_index,
            } => {
                if len_guard(*owner) != Some(*expected_len) {
                    self.exhausted = true;
                    return Err(IteratorAdvanceError::mutated(mutation_message));
                }
                if *reverse_index == 0 {
                    self.exhausted = true;
                    None
                } else {
                    *reverse_index -= 1;
                    Some(values[*reverse_index])
                }
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
        };

        Ok(next)
    }

    /// Get the next value from the iterator.
    ///
    /// This legacy infallible entry point is kept for static runtime helpers
    /// that cannot surface Python exceptions. VM-facing iteration should use
    /// `next_checked` or `next_with`.
    #[inline]
    pub fn next(&mut self) -> Option<Value> {
        self.next_checked().ok().flatten()
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
            IterKind::GuardedValues {
                owner,
                expected_len,
                len_guard,
                values,
                index,
                ..
            } => {
                if len_guard(*owner) == Some(*expected_len) {
                    Some(values.len().saturating_sub(*index))
                } else {
                    Some(0)
                }
            }
            IterKind::SharedIterator { iterator } => {
                native_iterator_from_value(*iterator).and_then(IteratorObject::size_hint)
            }
            IterKind::SequenceGetItem { .. } | IterKind::CallSentinel { .. } => None,
            IterKind::GenericAlias { yielded, .. } => Some(usize::from(!*yielded)),
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
            IterKind::ReversedList {
                list,
                reverse_index,
            } => {
                if list_from_value(*list).len() < *reverse_index {
                    Some(0)
                } else {
                    Some(*reverse_index)
                }
            }
            IterKind::GuardedReversedValues {
                owner,
                expected_len,
                len_guard,
                values,
                reverse_index,
                ..
            } => {
                if len_guard(*owner) == Some(*expected_len) {
                    Some(*reverse_index.min(&values.len()))
                } else {
                    Some(0)
                }
            }
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
    pub fn next_with<E, F, T, N, S, C>(
        &mut self,
        invoke: &mut F,
        is_truthy: &mut T,
        advance_iterator: &mut N,
        advance_sequence: &mut S,
        advance_call_sentinel: &mut C,
    ) -> Result<Option<Value>, E>
    where
        E: From<IteratorAdvanceError>,
        F: FnMut(Value, &[Value]) -> Result<Value, E>,
        T: FnMut(Value) -> Result<bool, E>,
        N: FnMut(Value) -> Result<Option<Value>, E>,
        S: FnMut(Value, Option<Value>, i64) -> Result<Option<Value>, E>,
        C: FnMut(Value, Value) -> Result<Option<Value>, E>,
    {
        if self.exhausted {
            return Ok(None);
        }

        match &mut self.kind {
            IterKind::SharedIterator { iterator } => {
                let next = if let Some(iter) = native_iterator_from_value_mut(*iterator) {
                    let next = iter.next_with(
                        invoke,
                        is_truthy,
                        advance_iterator,
                        advance_sequence,
                        advance_call_sentinel,
                    )?;
                    self.exhausted = iter.is_exhausted();
                    next
                } else {
                    let next = advance_iterator(*iterator)?;
                    self.exhausted = next.is_none();
                    next
                };
                return Ok(next);
            }
            IterKind::SequenceGetItem {
                callable,
                self_arg,
                index,
            } => {
                return match advance_sequence(*callable, *self_arg, *index)? {
                    Some(value) => {
                        if let Some(next_index) = index.checked_add(1) {
                            *index = next_index;
                        } else {
                            self.exhausted = true;
                        }
                        Ok(Some(value))
                    }
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                };
            }
            IterKind::CallSentinel { callable, sentinel } => {
                return match advance_call_sentinel(*callable, *sentinel)? {
                    Some(value) if !self.exhausted => Ok(Some(value)),
                    Some(_) | None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                };
            }
            IterKind::GenericAlias { alias, yielded } => {
                return if *yielded {
                    self.exhausted = true;
                    Ok(None)
                } else {
                    *yielded = true;
                    Ok(Some(create_starred_generic_alias(*alias)))
                };
            }
            IterKind::Map { func, inner } => {
                return match inner.next_with(
                    invoke,
                    is_truthy,
                    advance_iterator,
                    advance_sequence,
                    advance_call_sentinel,
                )? {
                    Some(value) => invoke(*func, &[value]).map(Some),
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                };
            }
            IterKind::Enumerate { inner, index } => {
                return match inner.next_with(
                    invoke,
                    is_truthy,
                    advance_iterator,
                    advance_sequence,
                    advance_call_sentinel,
                )? {
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
            IterKind::Zip { iterators } => {
                if iterators.is_empty() {
                    self.exhausted = true;
                    return Ok(None);
                }

                let mut values = Vec::with_capacity(iterators.len());
                for iter in iterators.iter_mut() {
                    match iter.next_with(
                        invoke,
                        is_truthy,
                        advance_iterator,
                        advance_sequence,
                        advance_call_sentinel,
                    )? {
                        Some(value) => values.push(value),
                        None => {
                            self.exhausted = true;
                            return Ok(None);
                        }
                    }
                }

                return Ok(Some(create_tuple_from_values(values)));
            }
            IterKind::Filter { func, inner } => loop {
                match inner.next_with(
                    invoke,
                    is_truthy,
                    advance_iterator,
                    advance_sequence,
                    advance_call_sentinel,
                )? {
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
                    match iterators[*current].next_with(
                        invoke,
                        is_truthy,
                        advance_iterator,
                        advance_sequence,
                        advance_call_sentinel,
                    )? {
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
                        .next_with(
                            invoke,
                            is_truthy,
                            advance_iterator,
                            advance_sequence,
                            advance_call_sentinel,
                        )?
                        .is_none()
                    {
                        self.exhausted = true;
                        return Ok(None);
                    }
                    *pos += 1;
                }

                return match inner.next_with(
                    invoke,
                    is_truthy,
                    advance_iterator,
                    advance_sequence,
                    advance_call_sentinel,
                )? {
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

        self.next_checked().map_err(E::from)
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
            IterKind::GuardedValues { .. } => "iterator",
            IterKind::SharedIterator { .. } => "iterator",
            IterKind::SequenceGetItem { .. } => "iterator",
            IterKind::CallSentinel { .. } => "callable_iterator",
            IterKind::GenericAlias { .. } => "generic_alias_iterator",
            IterKind::Empty => "empty_iterator",
            // Composite iterators
            IterKind::Chain { .. } => "chain",
            IterKind::Enumerate { .. } => "enumerate",
            IterKind::Zip { .. } => "zip",
            IterKind::Map { .. } => "map",
            IterKind::Filter { .. } => "filter",
            IterKind::ISlice { .. } => "islice",
            IterKind::Reversed { .. } => "reversed",
            IterKind::ReversedList { .. } => "list_reverseiterator",
            IterKind::GuardedReversedValues { .. } => "reversed",
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
