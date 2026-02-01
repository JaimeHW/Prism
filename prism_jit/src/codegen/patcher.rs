//! Code Patching Infrastructure.
//!
//! Provides thread-safe runtime code modification for zero-cost OSR and
//! dispatch table patching. Uses memory protection toggling to ensure
//! safe modification of executable pages.
//!
//! # Architecture
//!
//! The patching system operates in three phases:
//! 1. **Protection Change**: Mark page writable
//! 2. **Atomic Patch**: Write new bytes with memory fence
//! 3. **Protection Restore**: Mark page executable-only
//!
//! This ensures we never have a page that is both writable AND executable,
//! maintaining W^X (write XOR execute) invariant for security.

use std::io;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use libc::{PROT_EXEC, PROT_READ, PROT_WRITE, mprotect};

#[cfg(windows)]
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualProtect};

// =============================================================================
// Constants
// =============================================================================

/// Size of a near jump instruction (jmp rel32) on x86-64.
pub const JMP_REL32_SIZE: usize = 5;

/// Opcode for near jump (jmp rel32).
pub const JMP_REL32_OPCODE: u8 = 0xE9;

/// Size of a 5-byte nop sequence (0x0F 0x1F 0x44 0x00 0x00).
pub const NOP5_SIZE: usize = 5;

/// 5-byte nop sequence bytes.
pub const NOP5_BYTES: [u8; 5] = [0x0F, 0x1F, 0x44, 0x00, 0x00];

/// Size of a test instruction for safepoint polling.
#[allow(dead_code)]
pub const TEST_MEM_SIZE: usize = 7;

// =============================================================================
// Patch Types
// =============================================================================

/// Types of patches that can be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchType {
    /// Replace 5-byte nop with jmp rel32.
    NopToJump,
    /// Replace jmp rel32 with 5-byte nop.
    JumpToNop,
    /// Replace dispatch table entry (8-byte pointer).
    DispatchEntry,
    /// Replace guard with unconditional jump.
    GuardBailout,
}

/// A patch descriptor containing all information needed to apply a patch.
#[derive(Debug, Clone)]
pub struct PatchDescriptor {
    /// Type of patch.
    pub patch_type: PatchType,
    /// Address to patch.
    pub address: *mut u8,
    /// Target address for jumps (relative offset computed automatically).
    pub target: Option<*const u8>,
    /// Original bytes (for rollback).
    pub original: [u8; 8],
    /// Original byte count.
    pub original_len: u8,
}

impl PatchDescriptor {
    /// Create a nop-to-jump patch.
    ///
    /// # Safety
    /// Address must point to a valid 5-byte nop sequence.
    #[inline]
    pub unsafe fn nop_to_jump(nop_address: *mut u8, target: *const u8) -> Self {
        let mut original = [0u8; 8];
        // SAFETY: Caller guarantees nop_address points to valid 5-byte sequence
        unsafe {
            ptr::copy_nonoverlapping(nop_address, original.as_mut_ptr(), 5);
        }

        Self {
            patch_type: PatchType::NopToJump,
            address: nop_address,
            target: Some(target),
            original,
            original_len: 5,
        }
    }

    /// Create a jump-to-nop patch (for rollback).
    ///
    /// # Safety
    /// Address must point to a valid 5-byte jump instruction.
    #[inline]
    pub unsafe fn jump_to_nop(jump_address: *mut u8) -> Self {
        let mut original = [0u8; 8];
        // SAFETY: Caller guarantees jump_address points to valid 5-byte jump
        unsafe {
            ptr::copy_nonoverlapping(jump_address, original.as_mut_ptr(), 5);
        }

        Self {
            patch_type: PatchType::JumpToNop,
            address: jump_address,
            target: None,
            original,
            original_len: 5,
        }
    }

    /// Create a dispatch entry patch.
    ///
    /// # Safety
    /// Address must point to a valid 8-byte function pointer.
    #[inline]
    pub unsafe fn dispatch_entry(entry_address: *mut u8, new_handler: *const u8) -> Self {
        let mut original = [0u8; 8];
        // SAFETY: Caller guarantees entry_address points to valid 8-byte pointer
        unsafe {
            ptr::copy_nonoverlapping(entry_address, original.as_mut_ptr(), 8);
        }

        Self {
            patch_type: PatchType::DispatchEntry,
            address: entry_address,
            target: Some(new_handler),
            original,
            original_len: 8,
        }
    }
}

// SAFETY: PatchDescriptor contains raw pointers but is only used for
// describing patches within a single thread's context during patching.
unsafe impl Send for PatchDescriptor {}
unsafe impl Sync for PatchDescriptor {}

// =============================================================================
// Patcher
// =============================================================================

/// Thread-safe code patcher with W^X protection.
///
/// Maintains the invariant that pages are never simultaneously writable
/// and executable. Uses atomic operations for patch counting statistics.
#[derive(Debug)]
pub struct Patcher {
    /// Total patches applied.
    patches_applied: AtomicU64,
    /// Total patches rolled back.
    patches_rolled_back: AtomicU64,
    /// Page size for protection operations.
    page_size: usize,
}

impl Default for Patcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Patcher {
    /// Create a new patcher.
    #[inline]
    pub fn new() -> Self {
        Self {
            patches_applied: AtomicU64::new(0),
            patches_rolled_back: AtomicU64::new(0),
            page_size: Self::get_page_size(),
        }
    }

    /// Get the system page size.
    #[cfg(unix)]
    fn get_page_size() -> usize {
        // SAFETY: sysconf is safe to call with _SC_PAGESIZE
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    #[cfg(windows)]
    fn get_page_size() -> usize {
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
        // SAFETY: SYSTEM_INFO can be zero-initialized and GetSystemInfo always succeeds
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        unsafe { GetSystemInfo(&mut info) };
        info.dwPageSize as usize
    }

    /// Apply a patch atomically with W^X protection.
    ///
    /// # Safety
    /// The patch descriptor must describe a valid patch location.
    #[inline]
    pub unsafe fn apply(&self, patch: &PatchDescriptor) -> io::Result<()> {
        // Compute page-aligned address
        let page_start = self.page_align(patch.address as usize);

        // Phase 1: Make writable
        // SAFETY: Caller guarantees patch.address is in valid memory
        unsafe { self.make_writable(page_start)? };

        // Phase 2: Apply patch
        // SAFETY: Page is now writable and patch descriptor is valid
        let result = unsafe { self.write_patch(patch) };

        // Phase 3: Restore executable (even if write failed)
        // SAFETY: Page is in valid memory
        let restore_result = unsafe { self.make_executable(page_start) };

        // Memory fence to ensure visibility
        std::sync::atomic::fence(Ordering::SeqCst);

        // Handle errors
        result?;
        restore_result?;

        self.patches_applied.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Apply multiple patches atomically.
    ///
    /// All patches in the same page are grouped for efficiency.
    ///
    /// # Safety
    /// All patch descriptors must describe valid patch locations.
    pub unsafe fn apply_batch(&self, patches: &[PatchDescriptor]) -> io::Result<()> {
        if patches.is_empty() {
            return Ok(());
        }

        // Group patches by page
        let mut pages: Vec<usize> = patches
            .iter()
            .map(|p| self.page_align(p.address as usize))
            .collect();
        pages.sort_unstable();
        pages.dedup();

        // Make all pages writable
        for &page in &pages {
            // SAFETY: Caller guarantees all patch addresses are valid
            unsafe { self.make_writable(page)? };
        }

        // Apply all patches
        let mut write_error = None;
        for patch in patches {
            // SAFETY: Pages are now writable and patches are valid
            if let Err(e) = unsafe { self.write_patch(patch) } {
                write_error = Some(e);
                break;
            }
        }

        // Restore all pages to executable
        for &page in &pages {
            // SAFETY: Pages are in valid memory
            let _ = unsafe { self.make_executable(page) };
        }

        // Memory fence
        std::sync::atomic::fence(Ordering::SeqCst);

        if let Some(e) = write_error {
            return Err(e);
        }

        self.patches_applied
            .fetch_add(patches.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Rollback a patch using the original bytes.
    ///
    /// # Safety
    /// The patch descriptor must have valid original bytes.
    #[inline]
    pub unsafe fn rollback(&self, patch: &PatchDescriptor) -> io::Result<()> {
        let page_start = self.page_align(patch.address as usize);

        // SAFETY: Caller guarantees patch.address is in valid memory
        unsafe { self.make_writable(page_start)? };

        // Write original bytes back
        // SAFETY: Page is writable and original bytes are valid
        unsafe {
            ptr::copy_nonoverlapping(
                patch.original.as_ptr(),
                patch.address,
                patch.original_len as usize,
            );
        }

        // SAFETY: Page is in valid memory
        unsafe { self.make_executable(page_start)? };

        std::sync::atomic::fence(Ordering::SeqCst);

        self.patches_rolled_back.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Write patch bytes to the target address.
    ///
    /// # Safety
    /// The page containing patch.address must be writable.
    unsafe fn write_patch(&self, patch: &PatchDescriptor) -> io::Result<()> {
        match patch.patch_type {
            PatchType::NopToJump | PatchType::GuardBailout => {
                let target = patch.target.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "Jump target required")
                })?;

                // Compute relative offset
                // SAFETY: Pointer arithmetic within JMP_REL32_SIZE bounds
                let next_ip = unsafe { patch.address.add(JMP_REL32_SIZE) } as isize;
                let target_addr = target as isize;
                let rel_offset = target_addr - next_ip;

                // Check offset fits in i32
                if rel_offset > i32::MAX as isize || rel_offset < i32::MIN as isize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Jump offset out of range",
                    ));
                }

                // Write jmp rel32
                // SAFETY: patch.address is valid and page is writable
                unsafe {
                    *patch.address = JMP_REL32_OPCODE;
                    let offset_ptr = patch.address.add(1) as *mut i32;
                    ptr::write_unaligned(offset_ptr, rel_offset as i32);
                }
            }

            PatchType::JumpToNop => {
                // Write 5-byte nop
                // SAFETY: patch.address is valid and page is writable
                unsafe {
                    ptr::copy_nonoverlapping(NOP5_BYTES.as_ptr(), patch.address, NOP5_SIZE);
                }
            }

            PatchType::DispatchEntry => {
                let target = patch.target.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "Handler address required")
                })?;

                // Write 8-byte pointer atomically
                // SAFETY: patch.address is valid, aligned, and page is writable
                unsafe {
                    let ptr = patch.address as *mut u64;
                    ptr::write_volatile(ptr, target as u64);
                }
            }
        }

        Ok(())
    }

    /// Page-align an address.
    #[inline]
    fn page_align(&self, addr: usize) -> usize {
        addr & !(self.page_size - 1)
    }

    /// Make a page writable.
    ///
    /// # Safety
    /// Page must be mapped in the process address space.
    #[cfg(unix)]
    unsafe fn make_writable(&self, page: usize) -> io::Result<()> {
        // SAFETY: Caller guarantees page is mapped
        let result = unsafe {
            mprotect(
                page as *mut libc::c_void,
                self.page_size,
                PROT_READ | PROT_WRITE,
            )
        };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    #[cfg(windows)]
    unsafe fn make_writable(&self, page: usize) -> io::Result<()> {
        let mut old_protect: u32 = 0;
        // SAFETY: Caller guarantees page is mapped
        let result = unsafe {
            VirtualProtect(
                page as *mut std::ffi::c_void,
                self.page_size,
                PAGE_READWRITE,
                &mut old_protect,
            )
        };
        if result == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Make a page executable.
    ///
    /// # Safety
    /// Page must be mapped in the process address space.
    #[cfg(unix)]
    unsafe fn make_executable(&self, page: usize) -> io::Result<()> {
        // SAFETY: Caller guarantees page is mapped
        let result = unsafe {
            mprotect(
                page as *mut libc::c_void,
                self.page_size,
                PROT_READ | PROT_EXEC,
            )
        };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    #[cfg(windows)]
    unsafe fn make_executable(&self, page: usize) -> io::Result<()> {
        let mut old_protect: u32 = 0;
        // SAFETY: Caller guarantees page is mapped
        let result = unsafe {
            VirtualProtect(
                page as *mut std::ffi::c_void,
                self.page_size,
                PAGE_EXECUTE_READ,
                &mut old_protect,
            )
        };
        if result == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Get patch statistics.
    #[inline]
    pub fn stats(&self) -> PatchStats {
        PatchStats {
            patches_applied: self.patches_applied.load(Ordering::Relaxed),
            patches_rolled_back: self.patches_rolled_back.load(Ordering::Relaxed),
        }
    }
}

/// Patch statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct PatchStats {
    /// Total patches applied.
    pub patches_applied: u64,
    /// Total patches rolled back.
    pub patches_rolled_back: u64,
}

// =============================================================================
// Jump Encoding Utilities
// =============================================================================

/// Encode a relative near jump instruction.
///
/// Returns the 5-byte encoding of `jmp rel32`.
#[inline]
pub fn encode_jmp_rel32(from: *const u8, to: *const u8) -> Option<[u8; 5]> {
    // SAFETY: Pointer arithmetic is safe for offset calculation
    let next_ip = unsafe { from.add(5) } as isize;
    let target = to as isize;
    let offset = target - next_ip;

    if offset > i32::MAX as isize || offset < i32::MIN as isize {
        return None;
    }

    let mut bytes = [0u8; 5];
    bytes[0] = JMP_REL32_OPCODE;
    let offset_bytes = (offset as i32).to_le_bytes();
    bytes[1..5].copy_from_slice(&offset_bytes);

    Some(bytes)
}

/// Verify that bytes at an address match expected bytes.
///
/// # Safety
/// Address must be valid for reading `expected.len()` bytes.
#[inline]
pub unsafe fn verify_bytes(address: *const u8, expected: &[u8]) -> bool {
    for (i, &byte) in expected.iter().enumerate() {
        // SAFETY: Caller guarantees address is valid for expected.len() bytes
        if unsafe { *address.add(i) } != byte {
            return false;
        }
    }
    true
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_jmp_rel32() {
        // Test forward jump
        let from = 0x1000 as *const u8;
        let to = 0x1100 as *const u8;
        let bytes = encode_jmp_rel32(from, to).expect("Should encode");
        assert_eq!(bytes[0], JMP_REL32_OPCODE);

        // Offset should be 0x100 - 5 = 0xFB
        let offset = i32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
        assert_eq!(offset, 0xFB);
    }

    #[test]
    fn test_encode_jmp_rel32_backward() {
        // Test backward jump
        let from = 0x1100 as *const u8;
        let to = 0x1000 as *const u8;
        let bytes = encode_jmp_rel32(from, to).expect("Should encode");
        assert_eq!(bytes[0], JMP_REL32_OPCODE);

        let offset = i32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
        assert_eq!(offset, -0x105); // -0x100 - 5
    }

    #[test]
    fn test_patcher_creation() {
        let patcher = Patcher::new();
        let stats = patcher.stats();
        assert_eq!(stats.patches_applied, 0);
        assert_eq!(stats.patches_rolled_back, 0);
    }

    #[test]
    fn test_nop5_bytes() {
        // Verify NOP5 is correct encoding for 5-byte NOP
        assert_eq!(NOP5_BYTES[0], 0x0F);
        assert_eq!(NOP5_BYTES[1], 0x1F);
        assert_eq!(NOP5_SIZE, 5);
    }

    #[test]
    fn test_page_alignment() {
        let patcher = Patcher::new();
        let page_size = patcher.page_size;

        assert_eq!(patcher.page_align(0x1000), 0x1000 & !(page_size - 1));
        assert_eq!(patcher.page_align(0x1001), 0x1000 & !(page_size - 1));
    }
}
