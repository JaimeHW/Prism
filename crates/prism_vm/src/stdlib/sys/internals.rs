//! Internal functions for debugging and introspection.
//!
//! Provides low-level access to interpreter internals.

use prism_core::Value;
use prism_core::intern::interned_by_ptr;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::collections::HashSet;
use std::sync::RwLock;

// =============================================================================
// Reference Counting (Stub for GC-based runtime)
// =============================================================================

/// Get the reference count of an object.
///
/// In a GC-based runtime, this returns 1 for most objects.
/// The value is primarily for debugging.
#[inline]
pub fn getrefcount(_value: &Value) -> usize {
    // In a tracing GC, ref count isn't meaningful
    // Return 1 to indicate "at least one reference"
    1
}

// =============================================================================
// Object Size
// =============================================================================

/// Get the size of an object in bytes.
///
/// Returns the approximate memory usage of the value.
#[inline]
pub fn getsizeof(value: &Value) -> usize {
    if value.is_string() {
        let payload = value
            .as_string_object_ptr()
            .and_then(|ptr| interned_by_ptr(ptr as *const u8))
            .map(|text| text.as_str().len())
            .unwrap_or(0);
        return std::mem::size_of::<Value>() + payload;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return std::mem::size_of::<Value>();
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            std::mem::size_of::<BytesObject>() + bytes.capacity().max(bytes.len())
        }
        TypeId::LIST => {
            let list = unsafe { &*(ptr as *const ListObject) };
            std::mem::size_of::<ListObject>() + list.len() * std::mem::size_of::<Value>()
        }
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            std::mem::size_of::<TupleObject>() + tuple.len() * std::mem::size_of::<Value>()
        }
        TypeId::DICT => {
            let dict = unsafe { &*(ptr as *const DictObject) };
            std::mem::size_of::<DictObject>()
                + dict.len() * (std::mem::size_of::<Value>() * 2 + std::mem::size_of::<u64>())
        }
        TypeId::STR => {
            let string = unsafe { &*(ptr as *const StringObject) };
            std::mem::size_of::<StringObject>() + string.len()
        }
        _ => std::mem::size_of::<ObjectHeader>(),
    }
}

/// Get the size of an object in bytes with default.
#[inline]
pub fn getsizeof_default(value: &Value, default: usize) -> usize {
    if value.is_none() {
        default
    } else {
        getsizeof(value)
    }
}

// =============================================================================
// String Interning
// =============================================================================

/// Global string intern pool.
static INTERN_POOL: RwLock<Option<HashSet<String>>> = RwLock::new(None);

/// Initialize the intern pool if needed.
fn init_intern_pool() {
    let mut pool = INTERN_POOL.write().unwrap();
    if pool.is_none() {
        *pool = Some(HashSet::new());
    }
}

/// Intern a string.
///
/// Returns the interned string. Identical strings share the same allocation.
pub fn intern(s: String) -> String {
    init_intern_pool();

    let pool = INTERN_POOL.read().unwrap();
    if let Some(ref set) = *pool {
        if let Some(existing) = set.get(&s) {
            return existing.clone();
        }
    }
    drop(pool);

    // Not found, need to insert
    let mut pool = INTERN_POOL.write().unwrap();
    if let Some(ref mut set) = *pool {
        set.insert(s.clone());
    }
    s
}

/// Check if a string is interned.
pub fn is_interned(s: &str) -> bool {
    let pool = INTERN_POOL.read().unwrap();
    if let Some(ref set) = *pool {
        set.contains(s)
    } else {
        false
    }
}

/// Get the number of interned strings.
pub fn intern_count() -> usize {
    let pool = INTERN_POOL.read().unwrap();
    if let Some(ref set) = *pool {
        set.len()
    } else {
        0
    }
}

// =============================================================================
// Audit Hooks
// =============================================================================

/// Audit event for security monitoring.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Event name.
    pub name: String,
    /// Event arguments (simplified).
    pub args: Vec<String>,
}

impl AuditEvent {
    /// Create a new audit event.
    #[inline]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: Vec::new(),
        }
    }

    /// Create with arguments.
    #[inline]
    pub fn with_args(name: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }
}

// =============================================================================
// Frame Introspection
// =============================================================================

/// Get the current call stack depth.
///
/// This is a simplified implementation that tracks recursion depth.
#[derive(Debug, Default)]
pub struct CallDepth {
    depth: usize,
}

impl CallDepth {
    /// Create with zero depth.
    #[inline]
    pub fn new() -> Self {
        Self { depth: 0 }
    }

    /// Get current depth.
    #[inline]
    pub fn get(&self) -> usize {
        self.depth
    }

    /// Increment depth (entering function).
    #[inline]
    pub fn enter(&mut self) {
        self.depth += 1;
    }

    /// Decrement depth (leaving function).
    #[inline]
    pub fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Reset to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.depth = 0;
    }
}
