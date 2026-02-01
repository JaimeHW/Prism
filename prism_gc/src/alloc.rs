//! GC-managed reference type.
//!
//! GcRef<T> is a smart pointer to a GC-managed object.
//! It provides safe access to the object while ensuring
//! proper GC integration.

use crate::trace::Trace;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// A reference to a GC-managed object.
///
/// GcRef provides safe access to objects allocated through the GC.
/// The object will remain alive as long as it's reachable from
/// roots or other live objects.
///
/// # Safety
///
/// GcRef does NOT prevent the object from being collected.
/// To keep an object alive across GC points, use GcHandle instead.
///
/// # Example
///
/// ```ignore
/// let obj: GcRef<MyObject> = gc.alloc(MyObject::new());
/// obj.method(); // Access the object
/// ```
pub struct GcRef<T: Trace> {
    /// Pointer to the GC-managed object.
    ptr: NonNull<T>,
    /// Marker for the type.
    _marker: PhantomData<T>,
}

impl<T: Trace> GcRef<T> {
    /// Create a new GcRef from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid, GC-managed object of type T.
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ptr),
            _marker: PhantomData,
        }
    }

    /// Create a GcRef from NonNull.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid, GC-managed object of type T.
    pub unsafe fn from_non_null(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get the NonNull pointer.
    pub fn as_non_null(&self) -> NonNull<T> {
        self.ptr
    }

    /// Get the pointer as void pointer (for GC operations).
    pub fn as_void_ptr(&self) -> *const () {
        self.ptr.as_ptr() as *const ()
    }
}

impl<T: Trace> Clone for GcRef<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T: Trace> Copy for GcRef<T> {}

impl<T: Trace> Deref for GcRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: Trace> DerefMut for GcRef<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: Trace + std::fmt::Debug> std::fmt::Debug for GcRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcRef({:?})", self.deref())
    }
}

impl<T: Trace> PartialEq for GcRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T: Trace> Eq for GcRef<T> {}

impl<T: Trace> std::hash::Hash for GcRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

// Safety: GcRef is just a pointer. Thread safety depends on T.
// By default, we don't implement Send/Sync - the GC must manage this.

/// Trait for types that support GC tracing through GcRef.
unsafe impl<T: Trace> Trace for GcRef<T> {
    fn trace(&self, tracer: &mut dyn crate::trace::Tracer) {
        tracer.trace_ptr(self.as_void_ptr());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple traceable type for testing
    struct TestObject {
        value: i32,
    }

    unsafe impl Trace for TestObject {
        fn trace(&self, _tracer: &mut dyn crate::trace::Tracer) {}
    }

    #[test]
    fn test_gc_ref_creation() {
        let mut obj = TestObject { value: 42 };
        let gc_ref = unsafe { GcRef::from_raw(&mut obj) };

        assert_eq!(gc_ref.value, 42);
    }

    #[test]
    fn test_gc_ref_clone() {
        let mut obj = TestObject { value: 42 };
        let gc_ref = unsafe { GcRef::from_raw(&mut obj) };
        let cloned = gc_ref.clone();

        assert_eq!(gc_ref.as_ptr(), cloned.as_ptr());
    }

    #[test]
    fn test_gc_ref_deref_mut() {
        let mut obj = TestObject { value: 42 };
        let mut gc_ref = unsafe { GcRef::from_raw(&mut obj) };

        gc_ref.value = 100;
        assert_eq!(gc_ref.value, 100);
    }
}
