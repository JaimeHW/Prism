//! GC handles for preventing collection of specific objects.
//!
//! Handles are used when Rust code needs to hold references to
//! GC-managed objects across potential collection points.

use std::marker::PhantomData;
use std::ptr::NonNull;

/// Raw handle to a GC-managed object.
#[derive(Clone, Copy)]
pub struct RawHandle {
    /// Pointer to the object.
    pub ptr: *const (),
}

impl RawHandle {
    /// Create a new raw handle.
    pub fn new(ptr: *const ()) -> Self {
        Self { ptr }
    }

    /// Create a null handle.
    pub fn null() -> Self {
        Self {
            ptr: std::ptr::null(),
        }
    }

    /// Check if the handle is null.
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
}

// Safety: Handles are just pointers, safe to send.
unsafe impl Send for RawHandle {}
unsafe impl Sync for RawHandle {}

/// A typed handle to a GC-managed object.
///
/// GcHandle keeps an object alive as long as the handle exists.
/// When the handle is dropped, the object may be collected
/// (if no other references exist).
///
/// # Example
///
/// ```ignore
/// let handle = gc.make_handle(my_object);
/// // Object is guaranteed alive while handle exists
/// process(&handle);
/// // handle is dropped, object may be collected
/// ```
pub struct GcHandle<T> {
    /// The raw handle.
    raw: RawHandle,
    /// Type marker.
    _marker: PhantomData<*const T>,
}

impl<T> GcHandle<T> {
    /// Create a new handle from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid T that is managed by the GC.
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            raw: RawHandle::new(ptr as *const ()),
            _marker: PhantomData,
        }
    }

    /// Create a handle from a NonNull pointer.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid T that is managed by the GC.
    pub unsafe fn from_non_null(ptr: NonNull<T>) -> Self {
        Self::from_raw(ptr.as_ptr())
    }

    /// Get the raw handle.
    pub fn raw(&self) -> RawHandle {
        self.raw
    }

    /// Get a reference to the object.
    ///
    /// # Safety
    ///
    /// The object must still be alive (not collected).
    pub unsafe fn get(&self) -> &T {
        &*(self.raw.ptr as *const T)
    }

    /// Get a mutable reference to the object.
    ///
    /// # Safety
    ///
    /// The object must still be alive, and no other references
    /// to it may exist.
    pub unsafe fn get_mut(&mut self) -> &mut T {
        &mut *(self.raw.ptr as *mut T)
    }

    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.raw.ptr as *const T
    }
}

// Handles are not automatically Send/Sync for type safety.
// The GC-managed object's thread safety determines this.

/// A scope that tracks handles and ensures they're cleaned up.
///
/// HandleScope is used to create temporary handles that are
/// automatically unregistered when the scope is dropped.
///
/// # Example
///
/// ```ignore
/// {
///     let scope = HandleScope::new(&gc);
///     let handle = scope.create_handle(value);
///     // Use handle...
/// } // All handles in scope are unregistered
/// ```
pub struct HandleScope {
    /// Handles created in this scope.
    handles: Vec<RawHandle>,
}

impl HandleScope {
    /// Create a new handle scope.
    pub fn new() -> Self {
        Self {
            handles: Vec::new(),
        }
    }

    /// Create a typed handle in this scope.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid GC-managed object.
    pub unsafe fn create_handle<T>(&mut self, ptr: *const T) -> GcHandle<T> {
        let raw = RawHandle::new(ptr as *const ());
        self.handles.push(raw);
        GcHandle::from_raw(ptr)
    }

    /// Create a handle from a Value.
    pub fn create_value_handle(&mut self, value: prism_core::Value) -> Option<RawHandle> {
        if let Some(ptr) = value.as_object_ptr() {
            let raw = RawHandle::new(ptr);
            self.handles.push(raw);
            Some(raw)
        } else {
            None
        }
    }

    /// Get all handles in this scope.
    pub fn handles(&self) -> &[RawHandle] {
        &self.handles
    }

    /// Number of handles in this scope.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Check if scope is empty.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Clear all handles in this scope.
    pub fn clear(&mut self) {
        self.handles.clear();
    }
}

impl Default for HandleScope {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_handle() {
        let value = 42i64;
        let handle = RawHandle::new(&value as *const _ as *const ());
        assert!(!handle.is_null());
    }

    #[test]
    fn test_null_handle() {
        let handle = RawHandle::null();
        assert!(handle.is_null());
    }

    #[test]
    fn test_handle_scope() {
        let mut scope = HandleScope::new();
        assert!(scope.is_empty());

        let value = 42i64;
        unsafe {
            let _handle: GcHandle<i64> = scope.create_handle(&value);
        }

        assert_eq!(scope.len(), 1);
    }

    #[test]
    fn test_handle_scope_clear() {
        let mut scope = HandleScope::new();

        let value = 42i64;
        unsafe {
            let _h1: GcHandle<i64> = scope.create_handle(&value);
            let _h2: GcHandle<i64> = scope.create_handle(&value);
        }
        assert_eq!(scope.len(), 2);

        scope.clear();
        assert!(scope.is_empty());
    }
}
