//! Deoptimization Trampoline.
//!
//! Provides a page of trampolines indexed by deopt_id. When a guard fails,
//! it hijacks the return address to point to the appropriate trampoline entry.
//! The trampoline then captures state and dispatches to the recovery handler.
//!
//! # Memory Layout
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │ Trampoline Page (4KB)                                                  │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │ Entry 0:  [save regs] [load deopt_id=0] [jmp handler]  (16 bytes)      │
//! │ Entry 1:  [save regs] [load deopt_id=1] [jmp handler]  (16 bytes)      │
//! │ Entry 2:  [save regs] [load deopt_id=2] [jmp handler]  (16 bytes)      │
//! │ ...                                                                    │
//! │ Entry 255: [save regs] [load deopt_id=255] [jmp handler] (16 bytes)    │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │ Shared handler code                                                    │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```

use std::sync::atomic::{AtomicU32, Ordering};

// =============================================================================
// Constants
// =============================================================================

/// Size of each trampoline entry in bytes.
pub const TRAMPOLINE_ENTRY_SIZE: usize = 32;

/// Maximum number of trampoline entries per page.
/// With 32-byte entries, we can fit 128 entries in 4KB.
pub const MAX_TRAMPOLINE_ENTRIES: usize = 128;

/// Page size for trampoline allocation.
pub const TRAMPOLINE_PAGE_SIZE: usize = 4096;

// =============================================================================
// Trampoline Entry
// =============================================================================

/// Metadata for a single trampoline entry.
#[derive(Debug, Clone)]
pub struct DeoptTrampolineEntry {
    /// Deopt ID (index in trampoline table).
    pub deopt_id: u32,
    /// Bytecode offset to resume at.
    pub bc_offset: u32,
    /// Code object ID.
    pub code_id: u64,
    /// Reason for this deopt point.
    pub reason: super::DeoptReason,
    /// Whether this entry is active.
    pub active: bool,
}

impl Default for DeoptTrampolineEntry {
    fn default() -> Self {
        Self {
            deopt_id: 0,
            bc_offset: 0,
            code_id: 0,
            reason: super::DeoptReason::Explicit,
            active: false,
        }
    }
}

// =============================================================================
// Trampoline
// =============================================================================

/// Deoptimization trampoline page.
///
/// Manages a page of trampoline stubs for deoptimization. Each stub
/// loads its deopt_id and jumps to the shared handler.
#[derive(Debug)]
pub struct DeoptTrampoline {
    /// Trampoline code buffer (executable memory).
    /// In a real implementation, this would be mmap'd executable memory.
    code: Vec<u8>,
    /// Entry metadata.
    entries: Vec<DeoptTrampolineEntry>,
    /// Next available entry index.
    next_entry: AtomicU32,
    /// Address of the shared handler.
    handler_address: *const u8,
}

// SAFETY: DeoptTrampoline manages its own memory and synchronization.
unsafe impl Send for DeoptTrampoline {}
unsafe impl Sync for DeoptTrampoline {}

impl Default for DeoptTrampoline {
    fn default() -> Self {
        Self::new(std::ptr::null())
    }
}

impl DeoptTrampoline {
    /// Create a new trampoline page.
    ///
    /// # Arguments
    /// * `handler_address` - Address of the shared deopt handler function.
    pub fn new(handler_address: *const u8) -> Self {
        let mut entries = Vec::with_capacity(MAX_TRAMPOLINE_ENTRIES);
        for _ in 0..MAX_TRAMPOLINE_ENTRIES {
            entries.push(DeoptTrampolineEntry::default());
        }

        let mut trampoline = Self {
            code: vec![0u8; TRAMPOLINE_PAGE_SIZE],
            entries,
            next_entry: AtomicU32::new(0),
            handler_address,
        };

        // Initialize trampoline code
        trampoline.init_code();

        trampoline
    }

    /// Initialize trampoline code for all entries.
    fn init_code(&mut self) {
        for i in 0..MAX_TRAMPOLINE_ENTRIES {
            let offset = i * TRAMPOLINE_ENTRY_SIZE;
            self.emit_entry(offset, i as u32);
        }
    }

    /// Emit code for a single trampoline entry.
    fn emit_entry(&mut self, offset: usize, deopt_id: u32) {
        // Entry format (32 bytes) - all entries jump to shared handler at page end:
        // [0]     push rax             ; 0x50 - save scratch register
        // [1]     nop                  ; 0x90 - alignment
        // [2-6]   mov eax, deopt_id    ; 0xB8 + imm32
        // [7-11]  jmp rel32 to handler ; 0xE9 + imm32
        // [12-31] nop padding          ; 0x90 * 20

        let code = &mut self.code[offset..offset + TRAMPOLINE_ENTRY_SIZE];

        // push rax (1 byte)
        code[0] = 0x50;

        // nop for alignment (1 byte)
        code[1] = 0x90;

        // mov eax, imm32 (5 bytes)
        code[2] = 0xB8;
        code[3..7].copy_from_slice(&deopt_id.to_le_bytes());

        // jmp rel32 - calculate offset to shared handler at end of entries
        // Shared handler is at offset MAX_TRAMPOLINE_ENTRIES * TRAMPOLINE_ENTRY_SIZE
        let handler_offset = MAX_TRAMPOLINE_ENTRIES * TRAMPOLINE_ENTRY_SIZE;
        let current_end = offset + 12; // Position after jmp instruction
        let rel_offset = (handler_offset as i32 - current_end as i32);

        code[7] = 0xE9; // jmp rel32
        code[8..12].copy_from_slice(&rel_offset.to_le_bytes());

        // Fill remaining with nops
        for i in 12..TRAMPOLINE_ENTRY_SIZE {
            code[i] = 0x90;
        }
    }

    /// Allocate a new trampoline entry.
    ///
    /// Returns the deopt_id and address of the trampoline.
    pub fn allocate(
        &self,
        bc_offset: u32,
        code_id: u64,
        reason: super::DeoptReason,
    ) -> Option<(u32, *const u8)> {
        let deopt_id = self.next_entry.fetch_add(1, Ordering::Relaxed);

        if deopt_id as usize >= MAX_TRAMPOLINE_ENTRIES {
            // Rollback and fail
            self.next_entry.fetch_sub(1, Ordering::Relaxed);
            return None;
        }

        // Calculate trampoline address
        let entry_offset = deopt_id as usize * TRAMPOLINE_ENTRY_SIZE;
        let trampoline_addr = self.code.as_ptr().wrapping_add(entry_offset);

        // Note: In a real implementation, we'd update entries[deopt_id] here
        // But that requires interior mutability (Mutex or similar)
        let _ = (bc_offset, code_id, reason); // Suppress warnings

        Some((deopt_id, trampoline_addr))
    }

    /// Get the trampoline address for a deopt_id.
    #[inline]
    pub fn get_address(&self, deopt_id: u32) -> Option<*const u8> {
        if deopt_id as usize >= MAX_TRAMPOLINE_ENTRIES {
            return None;
        }

        let entry_offset = deopt_id as usize * TRAMPOLINE_ENTRY_SIZE;
        Some(self.code.as_ptr().wrapping_add(entry_offset))
    }

    /// Get entry metadata.
    #[inline]
    pub fn get_entry(&self, deopt_id: u32) -> Option<&DeoptTrampolineEntry> {
        self.entries.get(deopt_id as usize)
    }

    /// Number of allocated entries.
    #[inline]
    pub fn allocated_count(&self) -> u32 {
        self.next_entry.load(Ordering::Relaxed)
    }

    /// Check if trampoline is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.allocated_count() as usize >= MAX_TRAMPOLINE_ENTRIES
    }

    /// Reset all entries (for testing or when code is invalidated).
    pub fn reset(&self) {
        self.next_entry.store(0, Ordering::Relaxed);
    }

    /// Get the handler address.
    #[inline]
    pub fn handler_address(&self) -> *const u8 {
        self.handler_address
    }
}

// =============================================================================
// Return Address Hijacking
// =============================================================================

/// Utilities for return address manipulation.
pub mod hijack {
    /// Read the return address from the stack.
    ///
    /// # Safety
    /// Must be called from a context where rbp points to a valid stack frame.
    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn read_return_address() -> *const u8 {
        let ret_addr: *const u8;
        // SAFETY: Inline assembly to read return address from [rbp + 8]
        unsafe {
            std::arch::asm!(
                "mov {}, [rbp + 8]",
                out(reg) ret_addr,
                options(nostack, preserves_flags)
            );
        }
        ret_addr
    }

    /// Write a new return address to the stack.
    ///
    /// # Safety
    /// Must be called from a context where rbp points to a valid stack frame.
    /// The new address must be valid executable code.
    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn write_return_address(addr: *const u8) {
        // SAFETY: Inline assembly to write return address to [rbp + 8]
        unsafe {
            std::arch::asm!(
                "mov [rbp + 8], {}",
                in(reg) addr,
                options(nostack, preserves_flags)
            );
        }
    }

    /// Non-x86_64 stubs
    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn read_return_address() -> *const u8 {
        std::ptr::null()
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn write_return_address(_addr: *const u8) {}
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trampoline_creation() {
        let handler = 0x12345678 as *const u8;
        let trampoline = DeoptTrampoline::new(handler);

        assert_eq!(trampoline.allocated_count(), 0);
        assert!(!trampoline.is_full());
        assert_eq!(trampoline.handler_address(), handler);
    }

    #[test]
    fn test_trampoline_allocate() {
        let trampoline = DeoptTrampoline::new(std::ptr::null());

        let result = trampoline.allocate(100, 1, super::super::DeoptReason::TypeGuard);
        assert!(result.is_some());

        let (deopt_id, addr) = result.unwrap();
        assert_eq!(deopt_id, 0);
        assert!(!addr.is_null());

        assert_eq!(trampoline.allocated_count(), 1);
    }

    #[test]
    fn test_trampoline_get_address() {
        let trampoline = DeoptTrampoline::new(std::ptr::null());

        let addr = trampoline.get_address(0);
        assert!(addr.is_some());

        let addr = trampoline.get_address(MAX_TRAMPOLINE_ENTRIES as u32);
        assert!(addr.is_none());
    }

    #[test]
    fn test_trampoline_reset() {
        let trampoline = DeoptTrampoline::new(std::ptr::null());

        trampoline.allocate(100, 1, super::super::DeoptReason::TypeGuard);
        trampoline.allocate(200, 2, super::super::DeoptReason::Overflow);

        assert_eq!(trampoline.allocated_count(), 2);

        trampoline.reset();
        assert_eq!(trampoline.allocated_count(), 0);
    }

    #[test]
    fn test_trampoline_entry_default() {
        let entry = DeoptTrampolineEntry::default();
        assert_eq!(entry.deopt_id, 0);
        assert!(!entry.active);
    }

    #[test]
    fn test_constants() {
        assert_eq!(TRAMPOLINE_ENTRY_SIZE, 32);
        assert_eq!(MAX_TRAMPOLINE_ENTRIES, 128);
        assert!(MAX_TRAMPOLINE_ENTRIES * TRAMPOLINE_ENTRY_SIZE <= TRAMPOLINE_PAGE_SIZE);
    }
}
