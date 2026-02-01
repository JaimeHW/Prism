//! OSR-specific code patching.
//!
//! Manages patchpoints at loop headers for zero-cost On-Stack Replacement.
//! When a loop becomes hot, the nop slide at its header is patched to
//! jump directly to compiled JIT code.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Interpreter Code                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ loop_header:                                                    │
//! │   [5-byte nop] ──────────┐                                      │
//! │   ...loop body...        │  (cold: nop executes, continues)     │
//! │                          │                                      │
//! │                          ▼  (hot: patched to jmp)               │
//! │                    ┌─────────────────┐                          │
//! │                    │ OSR Entry Stub  │                          │
//! │                    │ - Save caller   │                          │
//! │                    │ - Setup JIT     │                          │
//! │                    │   frame         │                          │
//! │                    │ - Jump to JIT   │                          │
//! │                    │   loop body     │                          │
//! │                    └─────────────────┘                          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use super::patcher::{NOP5_BYTES, PatchDescriptor, Patcher};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

// =============================================================================
// Patchpoint
// =============================================================================

/// A patchpoint represents a location in compiled code that can be patched.
#[derive(Debug, Clone)]
pub struct Patchpoint {
    /// Unique identifier for this patchpoint.
    pub id: u32,
    /// Code ID this patchpoint belongs to.
    pub code_id: u64,
    /// Bytecode offset of the loop header.
    pub bc_offset: u32,
    /// Address of the patchable nop sequence.
    pub nop_address: *mut u8,
    /// Target address when patch is activated (OSR entry stub).
    pub target_address: *const u8,
    /// Whether this patchpoint is currently active (patched to jump).
    pub is_active: bool,
}

// SAFETY: Patchpoint is only used for bookkeeping within the JIT.
unsafe impl Send for Patchpoint {}
unsafe impl Sync for Patchpoint {}

/// State of a patchpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PatchpointState {
    /// Patchpoint contains nop, loop runs in interpreter.
    Inactive,
    /// Patchpoint contains jump to OSR entry.
    Active,
    /// Patchpoint has been invalidated (code discarded).
    Invalidated,
}

// =============================================================================
// Patchpoint Registry
// =============================================================================

/// Registry for tracking all patchpoints across compiled code.
///
/// Thread-safe for concurrent access from JIT compiler and runtime.
#[derive(Debug)]
pub struct PatchpointRegistry {
    /// Next patchpoint ID.
    next_id: AtomicU32,
    /// Patchpoints indexed by (code_id, bc_offset).
    by_location: RwLock<HashMap<(u64, u32), Arc<Patchpoint>>>,
    /// Patchpoints indexed by ID for fast lookup.
    by_id: RwLock<HashMap<u32, Arc<Patchpoint>>>,
    /// The code patcher.
    patcher: Patcher,
}

impl Default for PatchpointRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PatchpointRegistry {
    /// Create a new patchpoint registry.
    #[inline]
    pub fn new() -> Self {
        Self {
            next_id: AtomicU32::new(1),
            by_location: RwLock::new(HashMap::with_capacity(64)),
            by_id: RwLock::new(HashMap::with_capacity(64)),
            patcher: Patcher::new(),
        }
    }

    /// Register a new patchpoint at a loop header.
    ///
    /// # Arguments
    /// * `code_id` - The code object ID
    /// * `bc_offset` - Bytecode offset of the loop header
    /// * `nop_address` - Address of the 5-byte nop sequence
    ///
    /// # Returns
    /// The patchpoint ID.
    pub fn register(&self, code_id: u64, bc_offset: u32, nop_address: *mut u8) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let patchpoint = Arc::new(Patchpoint {
            id,
            code_id,
            bc_offset,
            nop_address,
            target_address: std::ptr::null(),
            is_active: false,
        });

        {
            let mut by_location = self.by_location.write().unwrap();
            by_location.insert((code_id, bc_offset), Arc::clone(&patchpoint));
        }

        {
            let mut by_id = self.by_id.write().unwrap();
            by_id.insert(id, patchpoint);
        }

        id
    }

    /// Activate a patchpoint, patching the nop to jump to OSR entry.
    ///
    /// # Safety
    /// The target address must be valid executable code.
    pub unsafe fn activate(
        &self,
        code_id: u64,
        bc_offset: u32,
        target: *const u8,
    ) -> Result<(), OsrPatchError> {
        let patchpoint = {
            let by_location = self.by_location.read().unwrap();
            by_location
                .get(&(code_id, bc_offset))
                .cloned()
                .ok_or(OsrPatchError::PatchpointNotFound)?
        };

        // Verify the nop is still there
        // SAFETY: patchpoint.nop_address was validated at registration
        if !unsafe { super::patcher::verify_bytes(patchpoint.nop_address, &NOP5_BYTES) } {
            return Err(OsrPatchError::InvalidNopSequence);
        }

        // Create and apply the patch
        // SAFETY: nop_address is valid and target is provided by caller
        let patch = unsafe { PatchDescriptor::nop_to_jump(patchpoint.nop_address, target) };
        // SAFETY: Patch descriptor is valid
        unsafe {
            self.patcher
                .apply(&patch)
                .map_err(OsrPatchError::PatchFailed)?;
        }

        Ok(())
    }

    /// Deactivate a patchpoint, restoring the nop.
    ///
    /// # Safety
    /// The patchpoint must have been previously activated.
    pub unsafe fn deactivate(&self, code_id: u64, bc_offset: u32) -> Result<(), OsrPatchError> {
        let patchpoint = {
            let by_location = self.by_location.read().unwrap();
            by_location
                .get(&(code_id, bc_offset))
                .cloned()
                .ok_or(OsrPatchError::PatchpointNotFound)?
        };

        // Create patch to restore nop
        // SAFETY: nop_address is valid from registration
        let patch = unsafe { PatchDescriptor::jump_to_nop(patchpoint.nop_address) };
        // SAFETY: Patch descriptor is valid
        unsafe {
            self.patcher
                .rollback(&patch)
                .map_err(OsrPatchError::PatchFailed)?;
        }

        Ok(())
    }

    /// Lookup a patchpoint by location.
    #[inline]
    pub fn lookup(&self, code_id: u64, bc_offset: u32) -> Option<Arc<Patchpoint>> {
        let by_location = self.by_location.read().unwrap();
        by_location.get(&(code_id, bc_offset)).cloned()
    }

    /// Lookup a patchpoint by ID.
    #[inline]
    pub fn lookup_by_id(&self, id: u32) -> Option<Arc<Patchpoint>> {
        let by_id = self.by_id.read().unwrap();
        by_id.get(&id).cloned()
    }

    /// Remove all patchpoints for a code object (on invalidation).
    pub fn invalidate_code(&self, code_id: u64) -> usize {
        let mut removed = 0;

        // Find and remove from by_location
        let to_remove: Vec<(u64, u32)> = {
            let by_location = self.by_location.read().unwrap();
            by_location
                .iter()
                .filter(|((cid, _), _)| *cid == code_id)
                .map(|(k, _)| *k)
                .collect()
        };

        {
            let mut by_location = self.by_location.write().unwrap();
            let mut by_id = self.by_id.write().unwrap();

            for key in to_remove {
                if let Some(pp) = by_location.remove(&key) {
                    by_id.remove(&pp.id);
                    removed += 1;
                }
            }
        }

        removed
    }

    /// Get patcher statistics.
    #[inline]
    pub fn stats(&self) -> OsrPatchStats {
        let by_location = self.by_location.read().unwrap();
        let patch_stats = self.patcher.stats();

        OsrPatchStats {
            total_patchpoints: by_location.len(),
            patches_applied: patch_stats.patches_applied,
            patches_rolled_back: patch_stats.patches_rolled_back,
        }
    }
}

/// OSR patching statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct OsrPatchStats {
    /// Total registered patchpoints.
    pub total_patchpoints: usize,
    /// Total patches applied.
    pub patches_applied: u64,
    /// Total patches rolled back.
    pub patches_rolled_back: u64,
}

/// Errors that can occur during OSR patching.
#[derive(Debug)]
pub enum OsrPatchError {
    /// Patchpoint not found in registry.
    PatchpointNotFound,
    /// The nop sequence at the patchpoint is invalid.
    InvalidNopSequence,
    /// Failed to apply the patch.
    PatchFailed(std::io::Error),
}

impl std::fmt::Display for OsrPatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PatchpointNotFound => write!(f, "Patchpoint not found"),
            Self::InvalidNopSequence => write!(f, "Invalid NOP sequence at patchpoint"),
            Self::PatchFailed(e) => write!(f, "Patch failed: {}", e),
        }
    }
}

impl std::error::Error for OsrPatchError {}

// =============================================================================
// OSR Entry Stub Builder
// =============================================================================

/// Builds OSR entry stubs that bridge from interpreter to JIT code.
#[derive(Debug, Default)]
pub struct OsrEntryStubBuilder {
    /// Buffer for stub code.
    buffer: Vec<u8>,
}

impl OsrEntryStubBuilder {
    /// Create a new stub builder.
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(64),
        }
    }

    /// Build an OSR entry stub.
    ///
    /// The stub:
    /// 1. Saves callee-saved registers
    /// 2. Sets up JIT frame from interpreter state
    /// 3. Jumps to the JIT loop body
    ///
    /// # Arguments
    /// * `jit_target` - Address of compiled loop body
    /// * `frame_size` - Size of JIT stack frame
    /// * `register_count` - Number of registers to materialize
    pub fn build(
        &mut self,
        jit_target: *const u8,
        frame_size: u32,
        register_count: u8,
    ) -> OsrEntryStub {
        self.buffer.clear();

        // Prologue: save RBP, establish frame
        self.emit_push_rbp();
        self.emit_mov_rbp_rsp();

        // Allocate stack space
        if frame_size > 0 {
            self.emit_sub_rsp_imm32(frame_size);
        }

        // The actual register materialization would go here.
        // For now, we just emit a jump to the JIT target.
        // In a full implementation, we'd load values from the
        // interpreter frame (passed in RDI) into JIT locations.

        // Jump to JIT code
        self.emit_jmp_abs(jit_target);

        OsrEntryStub {
            code: self.buffer.clone(),
            entry_offset: 0,
            frame_size,
            register_count,
        }
    }

    // x86-64 instruction emitters

    #[inline]
    fn emit_push_rbp(&mut self) {
        self.buffer.push(0x55); // push rbp
    }

    #[inline]
    fn emit_mov_rbp_rsp(&mut self) {
        self.buffer.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
    }

    #[inline]
    fn emit_sub_rsp_imm32(&mut self, imm: u32) {
        // sub rsp, imm32
        self.buffer.extend_from_slice(&[0x48, 0x81, 0xEC]);
        self.buffer.extend_from_slice(&imm.to_le_bytes());
    }

    #[inline]
    fn emit_jmp_abs(&mut self, target: *const u8) {
        // mov rax, target
        self.buffer.extend_from_slice(&[0x48, 0xB8]);
        self.buffer
            .extend_from_slice(&(target as u64).to_le_bytes());
        // jmp rax
        self.buffer.extend_from_slice(&[0xFF, 0xE0]);
    }
}

/// An OSR entry stub ready to be installed.
#[derive(Debug, Clone)]
pub struct OsrEntryStub {
    /// The machine code.
    pub code: Vec<u8>,
    /// Offset to the entry point within code.
    pub entry_offset: usize,
    /// Stack frame size.
    pub frame_size: u32,
    /// Number of registers to materialize.
    pub register_count: u8,
}

impl OsrEntryStub {
    /// Get the entry point address (once code is placed in executable memory).
    #[inline]
    pub fn entry_address(&self, base: *const u8) -> *const u8 {
        // SAFETY: Adding entry_offset to base is within bounds
        unsafe { base.add(self.entry_offset) }
    }

    /// Get the size of the stub.
    #[inline]
    pub fn size(&self) -> usize {
        self.code.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patchpoint_registry_creation() {
        let registry = PatchpointRegistry::new();
        let stats = registry.stats();
        assert_eq!(stats.total_patchpoints, 0);
    }

    #[test]
    fn test_patchpoint_registration() {
        let registry = PatchpointRegistry::new();

        let code_id = 1;
        let bc_offset = 10;
        let nop_addr = 0x1000 as *mut u8;

        let id = registry.register(code_id, bc_offset, nop_addr);
        assert!(id > 0);

        let pp = registry.lookup(code_id, bc_offset);
        assert!(pp.is_some());
        assert_eq!(pp.unwrap().bc_offset, bc_offset);
    }

    #[test]
    fn test_patchpoint_lookup_by_id() {
        let registry = PatchpointRegistry::new();

        let id = registry.register(1, 20, 0x2000 as *mut u8);
        let pp = registry.lookup_by_id(id);

        assert!(pp.is_some());
        assert_eq!(pp.unwrap().id, id);
    }

    #[test]
    fn test_invalidate_code() {
        let registry = PatchpointRegistry::new();

        registry.register(1, 10, 0x1000 as *mut u8);
        registry.register(1, 20, 0x1100 as *mut u8);
        registry.register(2, 10, 0x2000 as *mut u8);

        let removed = registry.invalidate_code(1);
        assert_eq!(removed, 2);

        assert!(registry.lookup(1, 10).is_none());
        assert!(registry.lookup(1, 20).is_none());
        assert!(registry.lookup(2, 10).is_some());
    }

    #[test]
    fn test_osr_entry_stub_builder() {
        let mut builder = OsrEntryStubBuilder::new();
        let target = 0x12345678 as *const u8;

        let stub = builder.build(target, 64, 8);

        assert!(!stub.code.is_empty());
        assert!(stub.size() > 0);
    }

    #[test]
    fn test_osr_patch_error_display() {
        let err = OsrPatchError::PatchpointNotFound;
        assert_eq!(format!("{}", err), "Patchpoint not found");

        let err = OsrPatchError::InvalidNopSequence;
        assert_eq!(format!("{}", err), "Invalid NOP sequence at patchpoint");
    }
}
