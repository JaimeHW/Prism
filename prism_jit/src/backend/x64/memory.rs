//! Executable memory management for JIT code.
//!
//! This module provides:
//! - Platform-specific executable memory allocation (VirtualAlloc/mmap)
//! - Write-then-execute (W^X) security model support
//! - Code cache management with LRU eviction
//!
//! # Safety
//! All memory management is inherently unsafe. This module encapsulates
//! the unsafety behind safe APIs where possible.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

// =============================================================================
// Platform-specific imports
// =============================================================================

#[cfg(windows)]
mod platform {
    use std::ptr;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
        VirtualFree, VirtualProtect,
    };

    pub const PAGE_SIZE: usize = 4096;

    /// Allocate memory with read-write permissions.
    pub unsafe fn alloc_rw(size: usize) -> *mut u8 {
        unsafe {
            VirtualAlloc(ptr::null(), size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as *mut u8
        }
    }

    /// Free allocated memory.
    pub unsafe fn free(ptr: *mut u8, _size: usize) {
        unsafe {
            VirtualFree(ptr as *mut _, 0, MEM_RELEASE);
        }
    }

    /// Make memory executable (and read-only).
    pub unsafe fn make_executable(ptr: *mut u8, size: usize) -> bool {
        let mut old_protect = 0;
        unsafe { VirtualProtect(ptr as *mut _, size, PAGE_EXECUTE_READ, &mut old_protect) != 0 }
    }

    /// Make memory writable (remove execute permission).
    pub unsafe fn make_writable(ptr: *mut u8, size: usize) -> bool {
        let mut old_protect = 0;
        unsafe { VirtualProtect(ptr as *mut _, size, PAGE_READWRITE, &mut old_protect) != 0 }
    }
}

#[cfg(unix)]
mod platform {
    use std::ptr;

    pub const PAGE_SIZE: usize = 4096;

    /// Allocate memory with read-write permissions.
    pub unsafe fn alloc_rw(size: usize) -> *mut u8 {
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            ptr::null_mut()
        } else {
            ptr as *mut u8
        }
    }

    /// Free allocated memory.
    pub unsafe fn free(ptr: *mut u8, size: usize) {
        unsafe {
            libc::munmap(ptr as *mut _, size);
        }
    }

    /// Make memory executable (and read-only).
    pub unsafe fn make_executable(ptr: *mut u8, size: usize) -> bool {
        unsafe { libc::mprotect(ptr as *mut _, size, libc::PROT_READ | libc::PROT_EXEC) == 0 }
    }

    /// Make memory writable (remove execute permission).
    pub unsafe fn make_writable(ptr: *mut u8, size: usize) -> bool {
        unsafe { libc::mprotect(ptr as *mut _, size, libc::PROT_READ | libc::PROT_WRITE) == 0 }
    }
}

pub use platform::PAGE_SIZE;

// =============================================================================
// Executable Buffer
// =============================================================================

/// A buffer of executable memory for JIT-compiled code.
///
/// The buffer follows a W^X (Write XOR Execute) model:
/// 1. Initially writable for code emission
/// 2. Made executable (and non-writable) before execution
/// 3. Can be made writable again for patching
///
/// # Memory Layout
/// ```text
/// +------------------+
/// |    Code region   | <- Executable after finalization
/// +------------------+
/// |   Constant pool  | <- Read-only data (floats, addresses)
/// +------------------+
/// ```
pub struct ExecutableBuffer {
    /// Pointer to the allocated memory.
    ptr: NonNull<u8>,
    /// Total allocated size (page-aligned).
    capacity: usize,
    /// Current write position.
    len: usize,
    /// Whether the buffer is currently executable.
    is_executable: bool,
}

impl std::fmt::Debug for ExecutableBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutableBuffer")
            .field("ptr", &self.ptr.as_ptr())
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("is_executable", &self.is_executable)
            .finish()
    }
}

impl ExecutableBuffer {
    /// Minimum allocation size (one page).
    pub const MIN_SIZE: usize = PAGE_SIZE;

    /// Create a new executable buffer with at least `min_capacity` bytes.
    ///
    /// The actual capacity will be rounded up to the nearest page boundary.
    pub fn new(min_capacity: usize) -> Option<Self> {
        let capacity = Self::align_to_page(min_capacity.max(Self::MIN_SIZE));

        let ptr = unsafe { platform::alloc_rw(capacity) };
        let ptr = NonNull::new(ptr)?;

        Some(ExecutableBuffer {
            ptr,
            capacity,
            len: 0,
            is_executable: false,
        })
    }

    /// Create a buffer with default capacity (64 KB).
    pub fn with_default_capacity() -> Option<Self> {
        Self::new(64 * 1024)
    }

    /// Get the current length (bytes written).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get available space.
    #[inline]
    pub fn available(&self) -> usize {
        self.capacity - self.len
    }

    /// Check if the buffer is currently executable.
    #[inline]
    pub fn is_executable(&self) -> bool {
        self.is_executable
    }

    /// Get a pointer to the start of the buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get the current write position pointer.
    #[inline]
    pub fn current_ptr(&self) -> *const u8 {
        unsafe { self.ptr.as_ptr().add(self.len) }
    }

    /// Get a slice of the written bytes.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Write a single byte.
    ///
    /// # Panics
    /// Panics if the buffer is executable or if there's no space.
    #[inline]
    pub fn emit_u8(&mut self, byte: u8) {
        assert!(!self.is_executable, "Cannot write to executable buffer");
        assert!(self.len < self.capacity, "Buffer overflow");

        unsafe {
            self.ptr.as_ptr().add(self.len).write(byte);
        }
        self.len += 1;
    }

    /// Write a slice of bytes.
    ///
    /// # Panics
    /// Panics if the buffer is executable or if there's not enough space.
    #[inline]
    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        assert!(!self.is_executable, "Cannot write to executable buffer");
        assert!(self.len + bytes.len() <= self.capacity, "Buffer overflow");

        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.ptr.as_ptr().add(self.len),
                bytes.len(),
            );
        }
        self.len += bytes.len();
    }

    /// Write a little-endian u16.
    #[inline]
    pub fn emit_u16(&mut self, val: u16) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Write a little-endian u32.
    #[inline]
    pub fn emit_u32(&mut self, val: u32) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Write a little-endian u64.
    #[inline]
    pub fn emit_u64(&mut self, val: u64) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Write a little-endian i8.
    #[inline]
    pub fn emit_i8(&mut self, val: i8) {
        self.emit_u8(val as u8);
    }

    /// Write a little-endian i32.
    #[inline]
    pub fn emit_i32(&mut self, val: i32) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Reserve space and return the offset for later patching.
    #[inline]
    pub fn reserve(&mut self, count: usize) -> usize {
        let offset = self.len;
        assert!(self.len + count <= self.capacity, "Buffer overflow");
        self.len += count;
        offset
    }

    /// Patch bytes at a specific offset.
    ///
    /// # Safety
    /// The buffer must be writable (not executable).
    pub fn patch_bytes(&mut self, offset: usize, bytes: &[u8]) {
        assert!(!self.is_executable, "Cannot patch executable buffer");
        assert!(offset + bytes.len() <= self.len, "Patch out of bounds");

        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.ptr.as_ptr().add(offset),
                bytes.len(),
            );
        }
    }

    /// Patch a u32 at a specific offset.
    #[inline]
    pub fn patch_u32(&mut self, offset: usize, val: u32) {
        self.patch_bytes(offset, &val.to_le_bytes());
    }

    /// Patch an i32 at a specific offset.
    #[inline]
    pub fn patch_i32(&mut self, offset: usize, val: i32) {
        self.patch_bytes(offset, &val.to_le_bytes());
    }

    /// Make the buffer executable (and non-writable).
    ///
    /// Returns `true` on success.
    pub fn make_executable(&mut self) -> bool {
        if self.is_executable {
            return true;
        }

        let success = unsafe { platform::make_executable(self.ptr.as_ptr(), self.capacity) };
        if success {
            self.is_executable = true;
        }
        success
    }

    /// Make the buffer writable again (for patching).
    ///
    /// Returns `true` on success.
    pub fn make_writable(&mut self) -> bool {
        if !self.is_executable {
            return true;
        }

        let success = unsafe { platform::make_writable(self.ptr.as_ptr(), self.capacity) };
        if success {
            self.is_executable = false;
        }
        success
    }

    /// Get a function pointer to the start of the buffer.
    ///
    /// # Safety
    /// - The buffer must be executable.
    /// - The code must be valid for the signature `F`.
    #[inline]
    pub unsafe fn as_fn<F>(&self) -> F
    where
        F: Copy,
    {
        debug_assert!(self.is_executable, "Buffer must be executable");
        debug_assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*const ()>(),
            "F must be a function pointer"
        );
        unsafe { std::mem::transmute_copy(&self.ptr.as_ptr()) }
    }

    /// Get a function pointer to a specific offset.
    ///
    /// # Safety
    /// - The buffer must be executable.
    /// - The code at `offset` must be valid for the signature `F`.
    #[inline]
    pub unsafe fn as_fn_at<F>(&self, offset: usize) -> F
    where
        F: Copy,
    {
        debug_assert!(self.is_executable, "Buffer must be executable");
        debug_assert!(offset < self.len, "Offset out of bounds");
        let ptr = unsafe { self.ptr.as_ptr().add(offset) };
        unsafe { std::mem::transmute_copy(&ptr) }
    }

    /// Align a size up to the nearest page boundary.
    #[inline]
    const fn align_to_page(size: usize) -> usize {
        (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
    }

    /// Clear the buffer for reuse.
    pub fn clear(&mut self) {
        if self.is_executable {
            self.make_writable();
        }
        self.len = 0;
    }
}

impl Drop for ExecutableBuffer {
    fn drop(&mut self) {
        unsafe {
            platform::free(self.ptr.as_ptr(), self.capacity);
        }
    }
}

// ExecutableBuffer is Send + Sync because we manage synchronization externally
unsafe impl Send for ExecutableBuffer {}
unsafe impl Sync for ExecutableBuffer {}

// =============================================================================
// Code Cache
// =============================================================================

/// Statistics for the code cache.
#[derive(Debug, Default)]
pub struct CodeCacheStats {
    /// Total bytes allocated.
    pub bytes_allocated: AtomicUsize,
    /// Number of compilation units.
    pub compilation_count: AtomicUsize,
    /// Number of evictions.
    pub eviction_count: AtomicUsize,
}

impl CodeCacheStats {
    /// Create new empty stats.
    pub const fn new() -> Self {
        CodeCacheStats {
            bytes_allocated: AtomicUsize::new(0),
            compilation_count: AtomicUsize::new(0),
            eviction_count: AtomicUsize::new(0),
        }
    }

    /// Record a new allocation.
    pub fn record_allocation(&self, bytes: usize) {
        self.bytes_allocated.fetch_add(bytes, Ordering::Relaxed);
        self.compilation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an eviction.
    pub fn record_eviction(&self, bytes: usize) {
        self.bytes_allocated.fetch_sub(bytes, Ordering::Relaxed);
        self.eviction_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total allocated bytes.
    pub fn total_bytes(&self) -> usize {
        self.bytes_allocated.load(Ordering::Relaxed)
    }

    /// Get compilation count.
    pub fn compilations(&self) -> usize {
        self.compilation_count.load(Ordering::Relaxed)
    }

    /// Get eviction count.
    pub fn evictions(&self) -> usize {
        self.eviction_count.load(Ordering::Relaxed)
    }
}

/// A compiled code entry in the cache.
pub struct CompiledCode {
    /// The executable buffer containing the code.
    buffer: ExecutableBuffer,
    /// Code identifier (e.g., function pointer or hash).
    id: u64,
    /// Size of the actual code (may be less than buffer capacity).
    code_size: usize,
}

impl CompiledCode {
    /// Create a new compiled code entry.
    pub fn new(buffer: ExecutableBuffer, id: u64) -> Self {
        let code_size = buffer.len();
        CompiledCode {
            buffer,
            id,
            code_size,
        }
    }

    /// Get the code identifier.
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the code size.
    #[inline]
    pub fn code_size(&self) -> usize {
        self.code_size
    }

    /// Get the underlying buffer.
    #[inline]
    pub fn buffer(&self) -> &ExecutableBuffer {
        &self.buffer
    }

    /// Get a function pointer to the code.
    ///
    /// # Safety
    /// The code must be valid for the signature `F`.
    #[inline]
    pub unsafe fn as_fn<F: Copy>(&self) -> F {
        unsafe { self.buffer.as_fn() }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
