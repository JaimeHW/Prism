//! Nursery (young generation) with bump-pointer allocation.
//!
//! The nursery uses a semi-space design for copying collection:
//! - FromSpace: Active allocation space
//! - ToSpace: Copy destination during minor GC
//!
//! Allocation is O(1) using bump-pointer:
//! ```text
//! alloc_ptr += size;
//! return alloc_ptr - size;
//! ```

use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Nursery (young generation) with semi-space copying collection.
pub struct Nursery {
    /// FromSpace: current allocation space.
    from_space: Space,
    /// ToSpace: copy destination during GC.
    to_space: Space,
    /// Size of each semi-space in bytes.
    size: usize,
}

/// A semi-space in the nursery.
struct Space {
    /// Start of the space.
    start: *mut u8,
    /// End of the space (start + size).
    end: *mut u8,
    /// Current allocation pointer (grows upward).
    alloc_ptr: AtomicPtr<u8>,
    /// Size of the space.
    size: usize,
}

impl Space {
    /// Allocate a new space with the given size.
    fn new(size: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(size, 8).expect("Invalid nursery layout");

        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            panic!("Failed to allocate nursery space of {} bytes", size);
        }

        let end = unsafe { ptr.add(size) };

        Self {
            start: ptr,
            end,
            alloc_ptr: AtomicPtr::new(ptr),
            size,
        }
    }

    /// Try to allocate `size` bytes.
    #[inline]
    fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        loop {
            let current = self.alloc_ptr.load(Ordering::Relaxed);
            let new_ptr = unsafe { current.add(size) };

            if new_ptr > self.end {
                return None; // Space exhausted
            }

            // CAS to claim the space
            if self
                .alloc_ptr
                .compare_exchange_weak(current, new_ptr, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return NonNull::new(current);
            }
        }
    }

    /// Check if a pointer is within this space.
    #[inline]
    fn contains(&self, ptr: *const ()) -> bool {
        let addr = ptr as usize;
        let start = self.start as usize;
        let end = self.end as usize;
        addr >= start && addr < end
    }

    /// Get bytes allocated in this space.
    #[inline]
    fn allocated(&self) -> usize {
        let current = self.alloc_ptr.load(Ordering::Relaxed);
        (current as usize).saturating_sub(self.start as usize)
    }

    /// Get remaining free bytes.
    #[inline]
    fn free(&self) -> usize {
        self.size.saturating_sub(self.allocated())
    }

    /// Reset the space for reuse.
    fn reset(&self) {
        self.alloc_ptr.store(self.start, Ordering::Release);

        // Zero memory for safety (helps debugging, prevents info leaks)
        #[cfg(debug_assertions)]
        unsafe {
            std::ptr::write_bytes(self.start, 0, self.size);
        }
    }

    /// Get the start pointer.
    #[inline]
    fn start(&self) -> *mut u8 {
        self.start
    }

    /// Get the allocation pointer.
    #[inline]
    fn alloc_ptr(&self) -> *mut u8 {
        self.alloc_ptr.load(Ordering::Relaxed)
    }
}

impl Drop for Space {
    fn drop(&mut self) {
        if !self.start.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.size, 8).expect("Invalid layout");
            unsafe {
                std::alloc::dealloc(self.start, layout);
            }
        }
    }
}

// Safety: Space can be sent between threads (has atomic alloc_ptr).
unsafe impl Send for Space {}
unsafe impl Sync for Space {}

impl Nursery {
    /// Create a new nursery with the given semi-space size.
    pub fn new(size: usize) -> Self {
        Self {
            from_space: Space::new(size),
            to_space: Space::new(size),
            size,
        }
    }

    /// Allocate memory in the from-space.
    ///
    /// Returns None if the nursery is full (should trigger minor GC).
    #[inline]
    pub fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        self.from_space.alloc(size)
    }

    /// Check if a pointer is in the nursery.
    #[inline]
    pub fn contains(&self, ptr: *const ()) -> bool {
        self.from_space.contains(ptr) || self.to_space.contains(ptr)
    }

    /// Check if a pointer is in the from-space.
    #[inline]
    pub fn in_from_space(&self, ptr: *const ()) -> bool {
        self.from_space.contains(ptr)
    }

    /// Check if a pointer is in the to-space.
    #[inline]
    pub fn in_to_space(&self, ptr: *const ()) -> bool {
        self.to_space.contains(ptr)
    }

    /// Get bytes allocated in from-space.
    #[inline]
    pub fn allocated(&self) -> usize {
        self.from_space.allocated()
    }

    /// Get remaining free bytes.
    #[inline]
    pub fn free(&self) -> usize {
        self.from_space.free()
    }

    /// Get the size of each semi-space.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get from-space start.
    #[inline]
    pub fn from_start(&self) -> *mut u8 {
        self.from_space.start()
    }

    /// Get from-space allocation pointer.
    #[inline]
    pub fn from_alloc_ptr(&self) -> *mut u8 {
        self.from_space.alloc_ptr()
    }

    /// Get to-space start.
    #[inline]
    pub fn to_start(&self) -> *mut u8 {
        self.to_space.start()
    }

    /// Get to-space allocation pointer.
    #[inline]
    pub fn to_alloc_ptr(&self) -> *mut u8 {
        self.to_space.alloc_ptr()
    }

    /// Allocate in to-space (used during copying collection).
    pub fn alloc_to_space(&self, size: usize) -> Option<NonNull<u8>> {
        self.to_space.alloc(size)
    }

    /// Swap from-space and to-space after copying collection.
    ///
    /// The to-space (containing survivors) becomes the new from-space.
    /// The old from-space is reset and becomes the new to-space.
    pub fn swap_spaces(&mut self) {
        // Reset the old from-space (it will become to-space)
        self.from_space.reset();

        // Swap the pointers
        std::mem::swap(&mut self.from_space, &mut self.to_space);
    }

    /// Reset both spaces (for testing or after OOM).
    pub fn reset(&mut self) {
        self.from_space.reset();
        self.to_space.reset();
    }

    /// Check if nursery is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.from_space.free() == 0
    }

    /// Get usage ratio (0.0 to 1.0).
    #[inline]
    pub fn usage_ratio(&self) -> f64 {
        self.allocated() as f64 / self.size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nursery_creation() {
        let nursery = Nursery::new(1024 * 1024); // 1MB
        assert_eq!(nursery.size(), 1024 * 1024);
        assert_eq!(nursery.allocated(), 0);
    }

    #[test]
    fn test_nursery_allocation() {
        let nursery = Nursery::new(1024);

        let ptr1 = nursery.alloc(64).expect("Alloc 1 failed");
        assert_eq!(nursery.allocated(), 64);

        let ptr2 = nursery.alloc(64).expect("Alloc 2 failed");
        assert_eq!(nursery.allocated(), 128);

        // Pointers should be consecutive
        assert_eq!(ptr2.as_ptr() as usize - ptr1.as_ptr() as usize, 64);
    }

    #[test]
    fn test_nursery_exhaustion() {
        let nursery = Nursery::new(128);

        // Fill the nursery
        let _ = nursery.alloc(64);
        let _ = nursery.alloc(64);

        // Should fail - nursery is full
        assert!(nursery.alloc(1).is_none());
        assert!(nursery.is_full());
    }

    #[test]
    fn test_nursery_contains() {
        let nursery = Nursery::new(1024);
        let ptr = nursery.alloc(64).expect("Alloc failed");

        assert!(nursery.contains(ptr.as_ptr() as *const ()));
        assert!(nursery.in_from_space(ptr.as_ptr() as *const ()));
        assert!(!nursery.in_to_space(ptr.as_ptr() as *const ()));
    }

    #[test]
    fn test_nursery_swap() {
        let mut nursery = Nursery::new(1024);

        // Allocate in from-space
        let ptr1 = nursery.alloc(64).expect("Alloc failed");
        assert!(nursery.in_from_space(ptr1.as_ptr() as *const ()));

        // Allocate in to-space (simulating copy)
        let ptr2 = nursery.alloc_to_space(64).expect("ToSpace alloc failed");
        assert!(nursery.in_to_space(ptr2.as_ptr() as *const ()));

        // Swap spaces
        nursery.swap_spaces();

        // ptr2 should now be in from-space
        assert!(nursery.in_from_space(ptr2.as_ptr() as *const ()));
    }
}
