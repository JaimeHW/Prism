//! Cell object for Python closure semantics.
//!
//! A `Cell` represents a single captured variable in a closure. Python closures
//! capture variables by reference, allowing inner functions to see updates from
//! outer scopes (and vice versa).
//!
//! # Performance
//!
//! - Single pointer indirection (Cell → Value)
//! - Cache-line optimized at 64 bytes
//! - Lock-free reads via atomic operations
//! - Thread-safe writes with relaxed ordering (Python GIL semantics assumed)
//!
//! # Python Semantics
//!
//! ```python
//! def outer():
//!     x = 10
//!     def inner():
//!         nonlocal x
//!         x = x + 1
//!         return x
//!     return inner
//!
//! f = outer()
//! f()  # Returns 11, x is shared via cell
//! f()  # Returns 12
//! ```
//!
//! # Implementation Notes
//!
//! Unlike CPython's `PyCellObject`, we use atomic value storage for thread
//! safety in concurrent scenarios. This adds minimal overhead while enabling
//! safe multi-threaded closure use.

use crate::object::ObjectHeader;
use crate::object::type_obj::TypeId;
use prism_core::Value;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Constants
// =============================================================================

/// Sentinel value indicating an unbound cell (deleted or not yet assigned).
const UNBOUND_SENTINEL: u64 = 0xFFFF_FFFF_FFFF_FFFE;

// =============================================================================
// Cell Object
// =============================================================================

/// A cell object holding a reference to a captured variable.
///
/// Cells enable Python's closure semantics where inner functions can read
/// and write variables from enclosing scopes.
///
/// # Memory Layout
///
/// The cell is 64-byte aligned to fit in a single cache line:
/// - ObjectHeader: 16 bytes
/// - value: 8 bytes (atomic u64 for NaN-boxed Value)
/// - padding: 40 bytes (reserved for future expansion)
///
/// # Thread Safety
///
/// All operations are thread-safe using atomic operations. Under Python's GIL
/// semantics, this provides sequential consistency. In free-threaded scenarios,
/// users must ensure proper synchronization.
#[repr(C, align(64))]
pub struct Cell {
    /// Standard object header for GC and type identification.
    header: ObjectHeader,

    /// The contained value as an atomic u64 (NaN-boxed).
    ///
    /// Uses `AtomicU64` for thread-safe access.
    /// - `UNBOUND_SENTINEL` indicates unbound/deleted state
    /// - Otherwise contains a valid NaN-boxed Value
    value: AtomicU64,

    /// Padding to fill cache line.
    _padding: [u8; 40],
}

// =============================================================================
// Cell Implementation
// =============================================================================

impl Cell {
    /// Create a new cell containing the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The initial value to store in the cell
    ///
    /// # Returns
    ///
    /// A new cell containing the value.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cell = Cell::new(Value::int(42).unwrap());
    /// assert_eq!(cell.get().as_int(), Some(42));
    /// ```
    #[inline]
    pub fn new(value: Value) -> Self {
        Cell {
            header: ObjectHeader::new(TypeId::CELL),
            value: AtomicU64::new(value.to_bits()),
            _padding: [0; 40],
        }
    }

    /// Create a new unbound cell.
    ///
    /// Unbound cells represent variables that have been deleted or not yet
    /// assigned. Accessing an unbound cell raises `UnboundLocalError`.
    ///
    /// # Returns
    ///
    /// A new cell in the unbound state.
    #[inline]
    pub fn unbound() -> Self {
        Cell {
            header: ObjectHeader::new(TypeId::CELL),
            value: AtomicU64::new(UNBOUND_SENTINEL),
            _padding: [0; 40],
        }
    }

    /// Get the value from the cell.
    ///
    /// # Returns
    ///
    /// The value if the cell is bound, or `None` if unbound.
    ///
    /// # Thread Safety
    ///
    /// This operation uses `Acquire` ordering to ensure visibility of
    /// any writes made before the corresponding `set()` call.
    #[inline]
    pub fn get(&self) -> Option<Value> {
        let bits = self.value.load(Ordering::Acquire);
        if bits == UNBOUND_SENTINEL {
            None
        } else {
            Some(Value::from_bits(bits))
        }
    }

    /// Get the value from the cell, returning `Value::none()` if unbound.
    ///
    /// This is a convenience method for cases where unbound should be
    /// treated as None (Python semantics for some operations).
    ///
    /// # Returns
    ///
    /// The value if bound, or `Value::none()` if unbound.
    #[inline]
    pub fn get_or_none(&self) -> Value {
        let bits = self.value.load(Ordering::Acquire);
        if bits == UNBOUND_SENTINEL {
            Value::none()
        } else {
            Value::from_bits(bits)
        }
    }

    /// Get the raw value, panicking if unbound.
    ///
    /// # Panics
    ///
    /// Panics if the cell is unbound.
    ///
    /// # Returns
    ///
    /// The contained value.
    #[inline]
    pub fn get_unchecked(&self) -> Value {
        let bits = self.value.load(Ordering::Acquire);
        debug_assert!(bits != UNBOUND_SENTINEL, "Cell is unbound");
        Value::from_bits(bits)
    }

    /// Set the value in the cell.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to store
    ///
    /// # Thread Safety
    ///
    /// This operation uses `Release` ordering to ensure that all writes
    /// before this call are visible to subsequent `get()` calls.
    #[inline]
    pub fn set(&self, value: Value) {
        self.value.store(value.to_bits(), Ordering::Release);
    }

    /// Clear the cell, making it unbound.
    ///
    /// After this call, `get()` will return `None` and `is_empty()` will
    /// return `true`.
    #[inline]
    pub fn clear(&self) {
        self.value.store(UNBOUND_SENTINEL, Ordering::Release);
    }

    /// Check if the cell is unbound (empty).
    ///
    /// # Returns
    ///
    /// `true` if the cell has no value, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.value.load(Ordering::Acquire) == UNBOUND_SENTINEL
    }

    /// Check if the cell is bound (has a value).
    ///
    /// # Returns
    ///
    /// `true` if the cell has a value, `false` otherwise.
    #[inline]
    pub fn is_bound(&self) -> bool {
        self.value.load(Ordering::Acquire) != UNBOUND_SENTINEL
    }

    /// Swap the value in the cell, returning the old value.
    ///
    /// # Arguments
    ///
    /// * `value` - The new value to store
    ///
    /// # Returns
    ///
    /// The previous value, or `None` if the cell was unbound.
    #[inline]
    pub fn swap(&self, value: Value) -> Option<Value> {
        let old_bits = self.value.swap(value.to_bits(), Ordering::AcqRel);
        if old_bits == UNBOUND_SENTINEL {
            None
        } else {
            Some(Value::from_bits(old_bits))
        }
    }

    /// Get the object header.
    #[inline]
    pub fn header(&self) -> &ObjectHeader {
        &self.header
    }

    /// Get the object header mutably.
    #[inline]
    pub fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(value) => write!(f, "Cell({:?})", value),
            None => write!(f, "Cell(<unbound>)"),
        }
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(value) => write!(f, "<cell: {:?}>", value),
            None => write!(f, "<cell: empty>"),
        }
    }
}

impl Clone for Cell {
    /// Clone the cell, creating a new cell with the same value.
    ///
    /// Note: This creates a *copy* of the cell, not a shared reference.
    /// For shared closure semantics, use `Arc<Cell>`.
    fn clone(&self) -> Self {
        let bits = self.value.load(Ordering::Acquire);
        Cell {
            header: ObjectHeader::new(TypeId::CELL),
            value: AtomicU64::new(bits),
            _padding: [0; 40],
        }
    }
}

impl Default for Cell {
    /// Create an unbound cell.
    fn default() -> Self {
        Self::unbound()
    }
}

// Safety: Cell uses atomic operations for all value access
unsafe impl Send for Cell {}
unsafe impl Sync for Cell {}
