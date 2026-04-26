//! Large Object Space for objects that exceed the large object threshold.
//!
//! Large objects are allocated directly from the system allocator
//! and managed individually. They bypass the nursery to avoid
//! expensive copying during minor GC.

use parking_lot::Mutex;
use std::alloc::Layout;
use std::collections::HashMap;
use std::ptr::NonNull;

/// Metadata for a large object.
struct LargeObject {
    /// Pointer to the allocated memory.
    ptr: NonNull<u8>,
    /// Exact allocation layout required by the system allocator.
    layout: Layout,
    /// GC mark flag.
    marked: bool,
}

impl LargeObject {
    #[inline]
    fn size(&self) -> usize {
        self.layout.size()
    }
}

/// Large Object Space for objects > threshold.
pub struct LargeObjectSpace {
    /// Map from pointer to object metadata.
    objects: Mutex<HashMap<usize, LargeObject>>,
    /// Total bytes allocated.
    allocated: std::sync::atomic::AtomicUsize,
}

impl LargeObjectSpace {
    /// Create a new large object space.
    pub fn new() -> Self {
        Self {
            objects: Mutex::new(HashMap::new()),
            allocated: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Allocate a large object.
    pub fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(size, 8).ok()?;
        self.alloc_layout(layout)
    }

    /// Allocate a large object with an explicit layout.
    pub fn alloc_layout(&self, layout: Layout) -> Option<NonNull<u8>> {
        let layout = normalize_layout(layout)?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            return None;
        }

        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        let addr = ptr.as_ptr() as usize;

        // Track the allocation
        let mut objects = self.objects.lock();
        objects.insert(
            addr,
            LargeObject {
                ptr,
                layout,
                marked: false,
            },
        );

        self.allocated
            .fetch_add(layout.size(), std::sync::atomic::Ordering::Relaxed);

        Some(ptr)
    }

    /// Check if a pointer is a large object.
    pub fn contains(&self, ptr: *const ()) -> bool {
        let objects = self.objects.lock();
        let addr = ptr as usize;
        objects.values().any(|obj| {
            addr >= obj.ptr.as_ptr() as usize && addr < obj.ptr.as_ptr() as usize + obj.size()
        })
    }

    /// Get the size of a large object.
    pub fn size_of(&self, ptr: *const ()) -> Option<usize> {
        let objects = self.objects.lock();
        objects.get(&(ptr as usize)).map(LargeObject::size)
    }

    /// Mark a large object as live.
    ///
    /// Returns `true` when `ptr` points inside a managed large object.
    pub fn mark(&self, ptr: *const ()) -> bool {
        let mut objects = self.objects.lock();
        let addr = ptr as usize;
        for obj in objects.values_mut() {
            let start = obj.ptr.as_ptr() as usize;
            let end = start + obj.size();
            if addr >= start && addr < end {
                obj.marked = true;
                return true;
            }
        }
        false
    }

    /// Clear all marks (before GC).
    pub fn clear_marks(&self) {
        let mut objects = self.objects.lock();
        for obj in objects.values_mut() {
            obj.marked = false;
        }
    }

    /// Sweep unmarked large objects.
    ///
    /// Returns (bytes_freed, objects_freed).
    pub fn sweep(&self) -> (usize, usize) {
        let mut objects = self.objects.lock();
        let mut bytes_freed = 0;
        let mut objects_freed = 0;

        // Collect unmarked objects to free
        let to_free: Vec<usize> = objects
            .iter()
            .filter(|(_, obj)| !obj.marked)
            .map(|(addr, _)| *addr)
            .collect();

        // Free unmarked objects
        for addr in to_free {
            if let Some(obj) = objects.remove(&addr) {
                bytes_freed += obj.size();
                objects_freed += 1;
                unsafe {
                    std::alloc::dealloc(obj.ptr.as_ptr(), obj.layout);
                }
            }
        }

        self.allocated
            .fetch_sub(bytes_freed, std::sync::atomic::Ordering::Relaxed);

        (bytes_freed, objects_freed)
    }

    /// Get total bytes allocated.
    pub fn usage(&self) -> usize {
        self.allocated.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get number of large objects.
    pub fn count(&self) -> usize {
        self.objects.lock().len()
    }

    /// Iterate over all large objects.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(*const (), usize),
    {
        let objects = self.objects.lock();
        for (addr, obj) in objects.iter() {
            f(*addr as *const (), obj.size());
        }
    }
}

fn normalize_layout(layout: Layout) -> Option<Layout> {
    Layout::from_size_align(layout.size().max(8), layout.align().max(8)).ok()
}

impl Default for LargeObjectSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LargeObjectSpace {
    fn drop(&mut self) {
        // Free all remaining large objects
        let objects = self.objects.get_mut();
        for (_, obj) in objects.drain() {
            unsafe {
                std::alloc::dealloc(obj.ptr.as_ptr(), obj.layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_object_alloc() {
        let los = LargeObjectSpace::new();

        let ptr = los.alloc(16 * 1024).expect("Alloc failed");
        assert_eq!(los.usage(), 16 * 1024);
        assert_eq!(los.count(), 1);
        assert!(los.contains(ptr.as_ptr() as *const ()));
    }

    #[test]
    fn test_large_object_sweep() {
        let los = LargeObjectSpace::new();

        let ptr1 = los.alloc(1024).expect("Alloc 1 failed");
        let ptr2 = los.alloc(2048).expect("Alloc 2 failed");

        // Mark only ptr1
        los.mark(ptr1.as_ptr() as *const ());

        // Sweep should free ptr2
        let (bytes_freed, objects_freed) = los.sweep();

        assert_eq!(bytes_freed, 2048);
        assert_eq!(objects_freed, 1);
        assert_eq!(los.count(), 1);
        assert!(los.contains(ptr1.as_ptr() as *const ()));
        assert!(!los.contains(ptr2.as_ptr() as *const ()));
    }

    #[test]
    fn test_large_object_mark_accepts_interior_pointer() {
        let los = LargeObjectSpace::new();

        let ptr = los.alloc(1024).expect("Alloc failed");
        let interior = unsafe { ptr.as_ptr().add(32) } as *const ();

        assert!(los.mark(interior));
        let (bytes_freed, objects_freed) = los.sweep();

        assert_eq!(bytes_freed, 0);
        assert_eq!(objects_freed, 0);
        assert_eq!(los.count(), 1);
        assert!(los.contains(ptr.as_ptr() as *const ()));
    }

    #[test]
    fn test_clear_and_sweep_all() {
        let los = LargeObjectSpace::new();

        let ptr1 = los.alloc(1024).expect("Alloc 1 failed");
        let ptr2 = los.alloc(2048).expect("Alloc 2 failed");

        // Mark both (collect pointers first to avoid deadlock)
        los.mark(ptr1.as_ptr() as *const ());
        los.mark(ptr2.as_ptr() as *const ());

        // Sweep should free nothing
        let (bytes_freed, _) = los.sweep();
        assert_eq!(bytes_freed, 0);
        assert_eq!(los.count(), 2);

        // Clear marks and sweep all
        los.clear_marks();
        let (bytes_freed, objects_freed) = los.sweep();
        assert_eq!(bytes_freed, 1024 + 2048);
        assert_eq!(objects_freed, 2);
        assert_eq!(los.count(), 0);
    }

    #[test]
    fn test_large_object_layout_preserves_alignment() {
        let los = LargeObjectSpace::new();
        let layout = Layout::from_size_align(17 * 1024, 64).expect("valid layout");

        let ptr = los.alloc_layout(layout).expect("Alloc failed");

        assert_eq!(ptr.as_ptr() as usize % 64, 0);
        assert_eq!(los.size_of(ptr.as_ptr() as *const ()), Some(17 * 1024));
        assert_eq!(
            los.objects
                .lock()
                .get(&(ptr.as_ptr() as usize))
                .map(|obj| obj.layout.align()),
            Some(64)
        );
    }

    #[test]
    fn test_large_object_sweep_uses_original_layout() {
        let los = LargeObjectSpace::new();
        let layout = Layout::from_size_align(17 * 1024, 64).expect("valid layout");
        let ptr = los.alloc_layout(layout).expect("Alloc failed");

        let (bytes_freed, objects_freed) = los.sweep();

        assert_eq!(bytes_freed, 17 * 1024);
        assert_eq!(objects_freed, 1);
        assert_eq!(los.count(), 0);
        assert!(!los.contains(ptr.as_ptr() as *const ()));
    }

    #[test]
    fn test_large_object_drop_uses_original_layout() {
        let layout = Layout::from_size_align(17 * 1024, 64).expect("valid layout");
        let ptr_addr = {
            let los = LargeObjectSpace::new();
            let ptr = los.alloc_layout(layout).expect("Alloc failed");
            assert_eq!(ptr.as_ptr() as usize % 64, 0);
            ptr.as_ptr() as usize
        };

        assert_ne!(ptr_addr, 0);
    }
}
