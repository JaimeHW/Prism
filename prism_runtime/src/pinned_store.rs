//! Stable fallback storage for runtime-owned heap objects.
//!
//! Most hot-path allocations should go through the VM's managed heap. This
//! store exists for standalone helpers and tests that need to materialize
//! stable object pointers without a live VM allocator.

use std::pin::Pin;
use std::sync::Mutex;

/// Process-lifetime pinned object storage.
///
/// Objects inserted here keep a stable address for the lifetime of the
/// process. This is intentionally a fallback path; VM-executed code should
/// prefer the managed heap so objects can participate in GC.
pub struct PinnedObjectStore<T> {
    objects: Mutex<Vec<Pin<Box<T>>>>,
}

impl<T> Default for PinnedObjectStore<T> {
    fn default() -> Self {
        Self {
            objects: Mutex::new(Vec::new()),
        }
    }
}

impl<T> PinnedObjectStore<T> {
    /// Pin an object and return a stable raw pointer to it.
    pub fn alloc(&self, value: T) -> *const T {
        let pinned = Box::into_pin(Box::new(value));
        let ptr = (&*pinned) as *const T;
        self.objects
            .lock()
            .expect("pinned object store mutex poisoned")
            .push(pinned);
        ptr
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.objects
            .lock()
            .expect("pinned object store mutex poisoned")
            .len()
    }
}
