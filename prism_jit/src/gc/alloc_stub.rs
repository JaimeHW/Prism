//! JIT Allocation Fast-Path
//!
//! Provides inline machine code sequences for zero-overhead object allocation
//! in JIT-compiled code.
//!
//! # Architecture
//!
//! ```text
//! Fast Path (inline, ~10 instructions):
//! ┌─────────────────────────────────────────────────────────────┐
//! │  1. Load TLAB ptr/end from thread-local                    │
//! │  2. Compute new_ptr = ptr + size                            │
//! │  3. If new_ptr > end: jump to slow_path                     │
//! │  4. Store new_ptr to TLAB ptr                               │
//! │  5. Return ptr (allocation succeeded)                       │
//! └─────────────────────────────────────────────────────────────┘
//!
//! Slow Path (out-of-line call):
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Call gc_alloc_slow(size) → refill TLAB, retry              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let stub = AllocStub::new(&tlab_offsets);
//! let code = stub.emit_fast_path(&mut asm, size_reg, result_reg);
//! ```

use crate::backend::x64::Assembler;
use crate::backend::x64::registers::Gpr;

// =============================================================================
// Thread-Local TLAB Offsets
// =============================================================================

/// Offsets for TLAB fields in thread-local storage.
///
/// These are architecture-specific and must match the layout of the
/// thread context structure used by the runtime.
#[derive(Debug, Clone, Copy)]
pub struct TlabOffsets {
    /// Offset of `tlab.ptr` from thread context base.
    pub ptr_offset: i32,
    /// Offset of `tlab.end` from thread context base.
    pub end_offset: i32,
}

impl Default for TlabOffsets {
    fn default() -> Self {
        // Default layout assumes Tlab is at offset 0 in thread context.
        // ptr is first field (offset 0), end is second field (offset 8).
        Self {
            ptr_offset: 0,
            end_offset: 8,
        }
    }
}

// =============================================================================
// Allocation Fast-Path Emitter
// =============================================================================

/// Emitter for inline TLAB allocation sequences.
pub struct AllocStub {
    /// TLAB field offsets.
    offsets: TlabOffsets,
    /// Register holding thread context base (e.g., R14).
    thread_reg: Gpr,
}

impl AllocStub {
    /// Create a new allocation stub with default offsets.
    pub fn new() -> Self {
        Self {
            offsets: TlabOffsets::default(),
            thread_reg: Gpr::R14, // Convention: R14 holds thread context
        }
    }

    /// Create with custom offsets and thread register.
    pub fn with_config(offsets: TlabOffsets, thread_reg: Gpr) -> Self {
        Self {
            offsets,
            thread_reg,
        }
    }

    /// Get the thread context register.
    pub fn thread_reg(&self) -> Gpr {
        self.thread_reg
    }

    /// Emit the fast-path allocation sequence.
    ///
    /// # Arguments
    ///
    /// * `asm` - Assembler for emitting code
    /// * `size_bytes` - Compile-time known allocation size (must be 8-byte aligned)
    /// * `result_reg` - Register to receive the allocated pointer
    /// * `scratch_reg` - Scratch register for intermediate calculations
    ///
    /// # Returns
    ///
    /// Offset of the slow-path jump target that needs to be patched.
    pub fn emit_fast_path_const_size(
        &self,
        asm: &mut Assembler,
        size_bytes: usize,
        result_reg: Gpr,
        scratch_reg: Gpr,
    ) -> u32 {
        // Ensure size is 8-byte aligned
        let aligned_size = (size_bytes + 7) & !7;

        // 1. Load current TLAB ptr into result_reg
        // mov result_reg, [thread_reg + ptr_offset]
        self.emit_load_tlab_ptr(asm, result_reg);

        // 2. Compute new_ptr = ptr + size
        // lea scratch_reg, [result_reg + aligned_size]
        self.emit_lea_add_imm(asm, scratch_reg, result_reg, aligned_size as i32);

        // 3. Load TLAB end
        // mov temp, [thread_reg + end_offset]
        // We reuse the comparison inline

        // 4. Compare new_ptr with end
        // cmp scratch_reg, [thread_reg + end_offset]
        self.emit_cmp_tlab_end(asm, scratch_reg);

        // 5. Jump to slow path if new_ptr > end
        // ja slow_path
        let slow_path_offset = asm.offset() as u32;
        asm.emit_bytes(&[0x0F, 0x87]); // JA rel32
        asm.emit_bytes(&[0x00, 0x00, 0x00, 0x00]); // Placeholder for offset

        // 6. Store new_ptr to TLAB ptr (fast path succeeds)
        // mov [thread_reg + ptr_offset], scratch_reg
        self.emit_store_tlab_ptr(asm, scratch_reg);

        // result_reg now contains the allocated pointer
        slow_path_offset
    }

    /// Emit fast-path with dynamic size in a register.
    ///
    /// # Arguments
    ///
    /// * `asm` - Assembler for emitting code
    /// * `size_reg` - Register containing the allocation size
    /// * `result_reg` - Register to receive the allocated pointer
    ///
    /// # Returns
    ///
    /// Offset of the slow-path jump target.
    pub fn emit_fast_path_dynamic_size(
        &self,
        asm: &mut Assembler,
        size_reg: Gpr,
        result_reg: Gpr,
    ) -> u32 {
        // 1. Align size to 8 bytes
        // add size_reg, 7
        // and size_reg, ~7
        self.emit_align_size(asm, size_reg);

        // 2. Load current TLAB ptr into result_reg
        self.emit_load_tlab_ptr(asm, result_reg);

        // 3. Compute new_ptr = ptr + size
        // add size_reg, result_reg (size_reg now holds new_ptr)
        asm.add_rr(size_reg, result_reg);

        // 4. Compare new_ptr with end
        self.emit_cmp_tlab_end(asm, size_reg);

        // 5. Jump to slow path if new_ptr > end
        let slow_path_offset = asm.offset() as u32;
        asm.emit_bytes(&[0x0F, 0x87]); // JA rel32
        asm.emit_bytes(&[0x00, 0x00, 0x00, 0x00]); // Placeholder

        // 6. Store new_ptr to TLAB ptr
        self.emit_store_tlab_ptr(asm, size_reg);

        slow_path_offset
    }

    /// Emit the slow-path call stub.
    ///
    /// This is called when the TLAB is exhausted and needs to be refilled.
    pub fn emit_slow_path(
        &self,
        asm: &mut Assembler,
        size_reg: Gpr,
        result_reg: Gpr,
        slow_path_fn: usize,
    ) {
        // Save caller-saved registers if needed
        // (Caller should handle this based on register allocation)

        // Set up argument: size in RDI (System V) or RCX (Windows)
        #[cfg(windows)]
        let arg_reg = Gpr::Rcx;
        #[cfg(not(windows))]
        let arg_reg = Gpr::Rdi;

        if size_reg != arg_reg {
            asm.mov_rr(arg_reg, size_reg);
        }

        // Call gc_alloc_slow
        // mov rax, slow_path_fn
        // call rax
        asm.mov_ri64(Gpr::Rax, slow_path_fn as i64);
        asm.emit_bytes(&[0xFF, 0xD0]); // CALL RAX

        // Move result from RAX to result_reg
        if result_reg != Gpr::Rax {
            asm.mov_rr(result_reg, Gpr::Rax);
        }
    }

    /// Patch the slow-path jump offset.
    pub fn patch_slow_path_offset(code: &mut [u8], jump_offset: u32, target_offset: u32) {
        let relative = target_offset.wrapping_sub(jump_offset + 6) as i32;
        let bytes = relative.to_le_bytes();

        // Patch the 4-byte relative offset after the JA opcode (2 bytes)
        let patch_pos = (jump_offset + 2) as usize;
        code[patch_pos..patch_pos + 4].copy_from_slice(&bytes);
    }

    // =========================================================================
    // Internal emission helpers
    // =========================================================================

    fn emit_load_tlab_ptr(&self, asm: &mut Assembler, dst: Gpr) {
        // mov dst, [thread_reg + ptr_offset]
        self.emit_load_from_thread(asm, dst, self.offsets.ptr_offset);
    }

    fn emit_store_tlab_ptr(&self, asm: &mut Assembler, src: Gpr) {
        // mov [thread_reg + ptr_offset], src
        self.emit_store_to_thread(asm, self.offsets.ptr_offset, src);
    }

    fn emit_cmp_tlab_end(&self, asm: &mut Assembler, reg: Gpr) {
        // cmp reg, [thread_reg + end_offset]
        self.emit_cmp_with_thread(asm, reg, self.offsets.end_offset);
    }

    fn emit_load_from_thread(&self, asm: &mut Assembler, dst: Gpr, offset: i32) {
        // REX.W mov dst, [thread_reg + offset]
        let rex = 0x48 | (dst.is_extended() as u8) << 2 | (self.thread_reg.is_extended() as u8);
        let modrm = 0x80 | (dst.encoding() & 7) << 3 | (self.thread_reg.encoding() & 7);

        asm.emit_bytes(&[rex, 0x8B, modrm]);
        asm.emit_bytes(&offset.to_le_bytes());
    }

    fn emit_store_to_thread(&self, asm: &mut Assembler, offset: i32, src: Gpr) {
        // REX.W mov [thread_reg + offset], src
        let rex = 0x48 | (src.is_extended() as u8) << 2 | (self.thread_reg.is_extended() as u8);
        let modrm = 0x80 | (src.encoding() & 7) << 3 | (self.thread_reg.encoding() & 7);

        asm.emit_bytes(&[rex, 0x89, modrm]);
        asm.emit_bytes(&offset.to_le_bytes());
    }

    fn emit_cmp_with_thread(&self, asm: &mut Assembler, reg: Gpr, offset: i32) {
        // REX.W cmp reg, [thread_reg + offset]
        let rex = 0x48 | (reg.is_extended() as u8) << 2 | (self.thread_reg.is_extended() as u8);
        let modrm = 0x80 | (reg.encoding() & 7) << 3 | (self.thread_reg.encoding() & 7);

        asm.emit_bytes(&[rex, 0x3B, modrm]);
        asm.emit_bytes(&offset.to_le_bytes());
    }

    fn emit_lea_add_imm(&self, asm: &mut Assembler, dst: Gpr, base: Gpr, imm: i32) {
        // REX.W lea dst, [base + imm]
        let rex = 0x48 | (dst.is_extended() as u8) << 2 | (base.is_extended() as u8);
        let modrm = 0x80 | (dst.encoding() & 7) << 3 | (base.encoding() & 7);

        asm.emit_bytes(&[rex, 0x8D, modrm]);
        asm.emit_bytes(&imm.to_le_bytes());
    }

    fn emit_align_size(&self, asm: &mut Assembler, reg: Gpr) {
        // add reg, 7
        // and reg, ~7
        let rex = 0x48 | (reg.is_extended() as u8);

        // add reg, 7
        asm.emit_bytes(&[rex, 0x83, 0xC0 | (reg.encoding() & 7), 0x07]);

        // and reg, ~7 (-8 in signed form)
        asm.emit_bytes(&[rex, 0x83, 0xE0 | (reg.encoding() & 7), 0xF8]);
    }
}

impl Default for AllocStub {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Runtime Slow-Path Function
// =============================================================================

/// Signature for the slow-path allocation function.
///
/// This is called when the TLAB is exhausted and needs to be refilled.
pub type AllocSlowFn = unsafe extern "C" fn(size: usize) -> *mut u8;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tlab_offsets_default() {
        let offsets = TlabOffsets::default();
        assert_eq!(offsets.ptr_offset, 0);
        assert_eq!(offsets.end_offset, 8);
    }

    #[test]
    fn test_alloc_stub_new() {
        let stub = AllocStub::new();
        assert_eq!(stub.thread_reg(), Gpr::R14);
    }

    #[test]
    fn test_alloc_stub_custom_config() {
        let offsets = TlabOffsets {
            ptr_offset: 16,
            end_offset: 24,
        };
        let stub = AllocStub::with_config(offsets, Gpr::R13);
        assert_eq!(stub.thread_reg(), Gpr::R13);
        assert_eq!(stub.offsets.ptr_offset, 16);
    }

    #[test]
    fn test_emit_fast_path_const_size() {
        let stub = AllocStub::new();
        let mut asm = Assembler::new();

        let slow_path_offset = stub.emit_fast_path_const_size(
            &mut asm,
            64, // 64 bytes
            Gpr::Rax,
            Gpr::Rcx,
        );

        // Should have emitted some code
        assert!(asm.offset() > 0);
        // Should have returned an offset for patching
        assert!(slow_path_offset > 0);
    }

    #[test]
    fn test_emit_fast_path_dynamic_size() {
        let stub = AllocStub::new();
        let mut asm = Assembler::new();

        let slow_path_offset = stub.emit_fast_path_dynamic_size(
            &mut asm,
            Gpr::Rcx, // size in RCX
            Gpr::Rax, // result in RAX
        );

        assert!(asm.offset() > 0);
        assert!(slow_path_offset > 0);
    }

    #[test]
    fn test_size_alignment() {
        // Check our alignment logic
        let sizes = [1, 7, 8, 9, 15, 16, 17, 63, 64, 65];
        let expected = [8, 8, 8, 16, 16, 16, 24, 64, 64, 72];

        for (size, aligned) in sizes.iter().zip(expected.iter()) {
            let result = (*size + 7) & !7;
            assert_eq!(result, *aligned, "Alignment failed for size {}", size);
        }
    }

    #[test]
    fn test_emit_slow_path() {
        let stub = AllocStub::new();
        let mut asm = Assembler::new();

        // Mock slow path function address
        let slow_path_fn = 0x12345678usize;

        stub.emit_slow_path(&mut asm, Gpr::Rcx, Gpr::Rax, slow_path_fn);

        assert!(asm.offset() > 0);
    }

    #[test]
    fn test_patch_slow_path_offset() {
        // Test the patching logic
        let mut code = vec![0u8; 16];
        // Set up bytes like: [JA opcode bytes] [placeholder]
        code[0] = 0x0F;
        code[1] = 0x87;
        // Placeholder for offset at bytes 2-5

        // Patch to jump to offset 100 from jump at offset 0
        AllocStub::patch_slow_path_offset(&mut code, 0, 100);

        // Relative = 100 - (0 + 6) = 94
        let expected = 94i32.to_le_bytes();
        assert_eq!(&code[2..6], &expected);
    }
}
