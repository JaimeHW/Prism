//! GC-Managed Arrays with Inline Data Storage
//!
//! GcArray provides contiguous, GC-managed storage for elements.
//! Unlike Vec<T> which allocates data on the system heap, GcArray
//! stores data inline with the object header, enabling:
//!
//! - Better cache locality (header + data are contiguous)
//! - GC compaction support (the GC can move the entire array)
//! - Proper tracing of contained references
//!
//! # Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  GcHeader  │  GcArray<T>  │  data[0]  data[1]  ...  data[N] │
//! │  (8 bytes) │   (header)   │        (inline elements)        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! // Create via GC heap allocation
//! let array: GcRef<GcArray<Value>> = heap.alloc_array(capacity);
//!
//! // Access elements
//! array.push(value);
//! let v = array.get(0);
//! ```

use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;

// =============================================================================
// GcArray Header
// =============================================================================

/// A GC-managed array with inline element storage.
///
/// The elements are stored directly after the GcArrayHeader in memory,
/// providing contiguous allocation and excellent cache locality.
#[repr(C)]
pub struct GcArray<T> {
    /// Capacity of the array in elements.
    capacity: u32,
    /// Current length (number of used elements).
    len: u32,
    /// Marker for element type.
    _marker: PhantomData<T>,
    // Elements follow directly after this header in memory.
    // We use a flexible array member pattern.
}

impl<T> GcArray<T> {
    /// Calculate the layout for a GcArray with the given capacity.
    ///
    /// Returns (total_size, data_offset) where:
    /// - total_size: Total bytes needed for header + elements
    /// - data_offset: Offset from GcArray start to first element
    pub fn layout_for_capacity(capacity: usize) -> (Layout, usize) {
        let header_layout = Layout::new::<GcArray<T>>();
        let element_layout = Layout::new::<T>();

        // Calculate aligned offset for elements
        let data_offset = header_layout.size();
        let aligned_offset =
            (data_offset + element_layout.align() - 1) & !(element_layout.align() - 1);

        // Total size: header + padding + elements
        let elements_size = capacity.saturating_mul(element_layout.size());
        let total_size = aligned_offset.saturating_add(elements_size);

        // Create the combined layout with proper alignment
        let layout = Layout::from_size_align(
            total_size,
            header_layout.align().max(element_layout.align()),
        )
        .expect("Invalid GcArray layout");

        (layout, aligned_offset)
    }

    /// Get the total allocation size in bytes for a given capacity.
    pub fn allocation_size(capacity: usize) -> usize {
        Self::layout_for_capacity(capacity).0.size()
    }

    /// Initialize a GcArray header in the given memory location.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid memory region for a GcArray with the given capacity.
    /// - The memory must be properly aligned.
    /// - The caller must ensure the memory is zeroed or properly initialized.
    pub unsafe fn init_at(ptr: NonNull<Self>, capacity: usize) {
        let array = ptr.as_ptr();
        unsafe {
            (*array).capacity = capacity.min(u32::MAX as usize) as u32;
            (*array).len = 0;
        }
    }

    /// Get the capacity of this array.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Get the current length (number of elements).
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Check if the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if the array is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len as usize >= self.capacity as usize
    }

    /// Get a pointer to the element data.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid only as long as the GcArray is alive.
    #[inline]
    pub fn data_ptr(&self) -> *const T {
        let (_, data_offset) = Self::layout_for_capacity(0);
        unsafe { (self as *const Self as *const u8).add(data_offset) as *const T }
    }

    /// Get a mutable pointer to the element data.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid only as long as the GcArray is alive.
    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut T {
        let (_, data_offset) = Self::layout_for_capacity(0);
        unsafe { (self as *mut Self as *mut u8).add(data_offset) as *mut T }
    }

    /// Get an element by index.
    ///
    /// Returns None if the index is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len as usize {
            unsafe { Some(&*self.data_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Get a mutable reference to an element by index.
    ///
    /// Returns None if the index is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len as usize {
            unsafe { Some(&mut *self.data_ptr_mut().add(index)) }
        } else {
            None
        }
    }

    /// Get an element by index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { &*self.data_ptr().add(index) }
    }

    /// Get a mutable reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        unsafe { &mut *self.data_ptr_mut().add(index) }
    }

    /// Set an element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if index >= len.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        assert!(index < self.len as usize, "Index out of bounds");
        unsafe {
            std::ptr::write(self.data_ptr_mut().add(index), value);
        }
    }

    /// Push an element to the end of the array.
    ///
    /// Returns true if successful, false if the array is full.
    #[inline]
    pub fn push(&mut self, value: T) -> bool {
        let len = self.len as usize;
        if len >= self.capacity as usize {
            return false; // Full
        }

        unsafe {
            std::ptr::write(self.data_ptr_mut().add(len), value);
        }
        self.len += 1;
        true
    }

    /// Pop an element from the end of the array.
    ///
    /// Returns None if the array is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        unsafe { Some(std::ptr::read(self.data_ptr().add(self.len as usize))) }
    }

    /// Clear all elements from the array.
    pub fn clear(&mut self) {
        // Drop all elements
        for i in 0..self.len as usize {
            unsafe {
                std::ptr::drop_in_place(self.data_ptr_mut().add(i));
            }
        }
        self.len = 0;
    }

    /// Get an iterator over the elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len as usize).map(move |i| unsafe { &*self.data_ptr().add(i) })
    }

    /// Get a mutable iterator over the elements.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        let len = self.len as usize;
        let data = self.data_ptr_mut();
        (0..len).map(move |i| unsafe { &mut *data.add(i) })
    }

    /// Get a slice over all elements.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.len as usize) }
    }

    /// Get a mutable slice over all elements.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr_mut(), self.len as usize) }
    }
}

impl<T: Clone> GcArray<T> {
    /// Fill the array with a cloned value up to capacity.
    pub fn fill(&mut self, value: T) {
        for i in self.len as usize..self.capacity as usize {
            unsafe {
                std::ptr::write(self.data_ptr_mut().add(i), value.clone());
            }
        }
        self.len = self.capacity;
    }

    /// Extend from a slice, returning number of elements copied.
    pub fn extend_from_slice(&mut self, slice: &[T]) -> usize {
        let available = self.capacity as usize - self.len as usize;
        let to_copy = slice.len().min(available);

        for (i, item) in slice.iter().take(to_copy).enumerate() {
            unsafe {
                std::ptr::write(self.data_ptr_mut().add(self.len as usize + i), item.clone());
            }
        }
        self.len += to_copy as u32;
        to_copy
    }
}

impl<T: Copy> GcArray<T> {
    /// Copy elements from another slice (faster for Copy types).
    pub fn copy_from_slice(&mut self, slice: &[T]) -> usize {
        let available = self.capacity as usize - self.len as usize;
        let to_copy = slice.len().min(available);

        if to_copy > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    slice.as_ptr(),
                    self.data_ptr_mut().add(self.len as usize),
                    to_copy,
                );
            }
            self.len += to_copy as u32;
        }
        to_copy
    }
}

// =============================================================================
// Drop Implementation
// =============================================================================

impl<T> Drop for GcArray<T> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len as usize {
            unsafe {
                std::ptr::drop_in_place(self.data_ptr_mut().add(i));
            }
        }
    }
}
