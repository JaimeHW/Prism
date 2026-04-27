//! Vector Spill and Reload Code Emission
//!
//! This module provides high-performance, width-aware spill and reload code emission
//! for all vector register widths (XMM, YMM, ZMM). It bridges the register allocator
//! output (which produces `SpillSlot` allocations) with the backend instruction encoders.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//! │   RegAllocator  │ ──▶ │  VectorSpiller  │ ──▶ │    Assembler    │
//! │  (AllocationMap)│     │  (this module)  │     │  (machine code) │
//! └─────────────────┘     └─────────────────┘     └─────────────────┘
//! ```
//!
//! # Design Principles
//!
//! 1. **Width-Aware Instruction Selection**: Automatically selects optimal move
//!    instruction based on register width (movaps for XMM, vmovaps for YMM, vmovaps
//!    with EVEX for ZMM).
//!
//! 2. **Alignment-Optimized**: Uses aligned moves when stack slots are properly
//!    aligned, falling back to unaligned moves when necessary. Aligned moves have
//!    lower latency on most microarchitectures.
//!
//! 3. **Move Elimination Awareness**: Detects and elides self-moves (same source
//!    and destination registers).
//!
//! 4. **Zero Idiom Detection**: Provides utilities for efficient register zeroing
//!    using VPXOR which is recognized by modern CPUs as a zero-idiom.
//!
//! # Performance Considerations
//!
//! - Aligned stores (VMOVAPS/VMOVAPD) are preferred over unaligned (VMOVUPS/VMOVUPD)
//!   when alignment is guaranteed
//! - For integer data, VMOVDQA64/32 are preferred over floating-point moves
//! - Register-to-register moves can be elided by the processor's move elimination unit
//!
//! # Usage
//!
//! ```ignore
//! let mut spiller = VectorSpiller::new();
//!
//! // Emit a spill from YMM0 to stack
//! spiller.emit_spill(&mut asm, PReg::Ymm(Ymm::Ymm0), slot, DataKind::Float64)?;
//!
//! // Emit a reload from stack to YMM0
//! spiller.emit_reload(&mut asm, PReg::Ymm(Ymm::Ymm0), slot, DataKind::Float64)?;
//! ```

use crate::backend::x64::assembler::Assembler;
use crate::backend::x64::evex::{
    self, encode_vmovapd_zmm_mr, encode_vmovapd_zmm_rm, encode_vmovapd_zmm_rr,
    encode_vmovdqa64_zmm_rr, encode_vmovupd_zmm_mr, encode_vmovupd_zmm_rm, encode_vmovupd_zmm_rr,
};
use crate::backend::x64::registers::{Gpr, MemOperand};
use crate::backend::x64::simd::{
    self, Ymm, Zmm, encode_vmovapd_mr, encode_vmovapd_rm, encode_vmovapd_xmm_rr,
    encode_vmovapd_ymm_rr, encode_vmovdqa_mr, encode_vmovdqa_rm, encode_vmovdqa_xmm_rr,
    encode_vmovdqa_ymm_rr, encode_vmovupd_mr, encode_vmovupd_rm, encode_vmovupd_ymm_rr,
};
use crate::regalloc::spill::{SpillSlot, SpillWidth};
use crate::regalloc::{PReg, RegClass};
use std::fmt;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during spill/reload code emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpillError {
    /// Register width mismatch between PReg and SpillSlot.
    WidthMismatch { reg_width: u16, slot_width: u32 },
    /// Invalid register type for vector operation.
    InvalidRegister(PReg),
    /// Stack offset out of range for displacement encoding.
    OffsetOutOfRange(i32),
    /// Attempt to spill from a GPR using vector operations.
    NotVectorRegister,
}

impl fmt::Display for SpillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpillError::WidthMismatch {
                reg_width,
                slot_width,
            } => {
                write!(
                    f,
                    "Register width {} bits doesn't match slot width {} bytes",
                    reg_width, slot_width
                )
            }
            SpillError::InvalidRegister(r) => {
                write!(f, "Invalid register for spill/reload: {}", r)
            }
            SpillError::OffsetOutOfRange(off) => {
                write!(f, "Stack offset {} out of range for displacement", off)
            }
            SpillError::NotVectorRegister => {
                write!(f, "Expected vector register (XMM/YMM/ZMM), got GPR")
            }
        }
    }
}

impl std::error::Error for SpillError {}

// =============================================================================
// Data Kind
// =============================================================================

/// The data type hint for optimal instruction selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataKind {
    /// Packed double-precision floating-point (pd suffix).
    Float64,
    /// Packed single-precision floating-point (ps suffix).
    Float32,
    /// 64-bit packed integers (q suffix for AVX-512).
    Int64,
    /// 32-bit packed integers (d suffix for AVX-512).
    Int32,
    /// Unknown/generic data - uses pd variants.
    Generic,
}

impl DataKind {
    /// Get the width of a single element in bits.
    #[inline]
    pub const fn element_width(self) -> u32 {
        match self {
            DataKind::Float64 | DataKind::Int64 => 64,
            DataKind::Float32 | DataKind::Int32 => 32,
            DataKind::Generic => 64,
        }
    }

    /// Check if this is a floating-point data kind.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, DataKind::Float64 | DataKind::Float32)
    }

    /// Check if this is an integer data kind.
    #[inline]
    pub const fn is_integer(self) -> bool {
        matches!(self, DataKind::Int64 | DataKind::Int32)
    }
}

impl Default for DataKind {
    fn default() -> Self {
        DataKind::Generic
    }
}

// =============================================================================
// Frame Base
// =============================================================================

/// The base register for stack frame addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameBase {
    /// Address relative to RBP (frame pointer).
    Rbp,
    /// Address relative to RSP (stack pointer).
    Rsp,
}

impl Default for FrameBase {
    fn default() -> Self {
        FrameBase::Rbp
    }
}

impl FrameBase {
    /// Convert to the corresponding GPR.
    #[inline]
    pub fn as_gpr(self) -> Gpr {
        match self {
            FrameBase::Rbp => Gpr::Rbp,
            FrameBase::Rsp => Gpr::Rsp,
        }
    }
}

// =============================================================================
// Spill Statistics
// =============================================================================

/// Statistics about spill/reload operations for profiling.
#[derive(Debug, Clone, Default)]
pub struct SpillStats {
    /// Total spills emitted.
    pub total_spills: u64,
    /// Total reloads emitted.
    pub total_reloads: u64,
    /// XMM (128-bit) spills.
    pub xmm_spills: u64,
    /// YMM (256-bit) spills.
    pub ymm_spills: u64,
    /// ZMM (512-bit) spills.
    pub zmm_spills: u64,
    /// GPR (64-bit) spills.
    pub gpr_spills: u64,
    /// Aligned memory operations used.
    pub aligned_ops: u64,
    /// Unaligned memory operations used.
    pub unaligned_ops: u64,
    /// Register-to-register moves emitted.
    pub reg_moves: u64,
    /// Self-moves elided.
    pub elided_moves: u64,
}

impl SpillStats {
    /// Create new empty statistics.
    #[inline]
    pub const fn new() -> Self {
        SpillStats {
            total_spills: 0,
            total_reloads: 0,
            xmm_spills: 0,
            ymm_spills: 0,
            zmm_spills: 0,
            gpr_spills: 0,
            aligned_ops: 0,
            unaligned_ops: 0,
            reg_moves: 0,
            elided_moves: 0,
        }
    }

    /// Reset all statistics.
    #[inline]
    pub fn reset(&mut self) {
        *self = SpillStats::new();
    }

    /// Get total memory operations (spills + reloads).
    #[inline]
    pub fn total_memory_ops(&self) -> u64 {
        self.total_spills + self.total_reloads
    }
}

// =============================================================================
// Vector Spiller
// =============================================================================

/// High-level vector spill/reload code emitter.
///
/// This is the primary interface for emitting spill and reload code for
/// vector registers. It handles width detection, alignment optimization,
/// and instruction selection transparently.
#[derive(Debug, Clone)]
pub struct VectorSpiller {
    /// Base register for frame addressing.
    frame_base: FrameBase,
    /// Prefer aligned instructions when slot is aligned (default: true).
    prefer_aligned: bool,
    /// Prefer integer move instructions for generic data (default: false).
    prefer_integer_moves: bool,
    /// Enable move elimination for self-moves (default: true).
    elide_self_moves: bool,
    /// Statistics tracking.
    stats: SpillStats,
}

impl Default for VectorSpiller {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorSpiller {
    /// Create a new VectorSpiller with default settings.
    #[inline]
    pub fn new() -> Self {
        VectorSpiller {
            frame_base: FrameBase::Rbp,
            prefer_aligned: true,
            prefer_integer_moves: false,
            elide_self_moves: true,
            stats: SpillStats::new(),
        }
    }

    /// Create with custom frame base.
    #[inline]
    pub fn with_frame_base(mut self, base: FrameBase) -> Self {
        self.frame_base = base;
        self
    }

    /// Disable aligned instruction preference.
    #[inline]
    pub fn with_unaligned_only(mut self) -> Self {
        self.prefer_aligned = false;
        self
    }

    /// Prefer integer move instructions (VMOVDQA) over floating-point (VMOVAPS).
    #[inline]
    pub fn with_integer_moves(mut self) -> Self {
        self.prefer_integer_moves = true;
        self
    }

    /// Disable self-move elision.
    #[inline]
    pub fn without_move_elision(mut self) -> Self {
        self.elide_self_moves = false;
        self
    }

    /// Get the current statistics.
    #[inline]
    pub fn stats(&self) -> &SpillStats {
        &self.stats
    }

    /// Reset statistics.
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    // =========================================================================
    // Core Spill/Reload API
    // =========================================================================

    /// Emit a spill operation: store register to stack.
    ///
    /// # Arguments
    /// * `code` - Buffer to emit machine code into
    /// * `reg` - The physical register to spill
    /// * `slot` - The stack slot to spill to
    /// * `kind` - Data type hint for instruction selection
    ///
    /// # Returns
    /// Number of bytes emitted, or an error.
    pub fn emit_spill(
        &mut self,
        code: &mut Vec<u8>,
        reg: PReg,
        slot: SpillSlot,
        kind: DataKind,
    ) -> Result<usize, SpillError> {
        let start_len = code.len();

        // Validate width match
        self.validate_width(reg, slot)?;

        // Create memory operand
        let mem = self.slot_to_mem(slot);

        // Check alignment for instruction selection
        let use_aligned = self.prefer_aligned && self.is_slot_aligned(&slot);

        // Emit based on register type
        match reg {
            PReg::Xmm(x) => {
                let ymm = Ymm::from_xmm(x);
                self.emit_xmm_store(code, ymm, &mem, kind, use_aligned);
                self.stats.xmm_spills += 1;
            }
            PReg::Ymm(y) => {
                self.emit_ymm_store(code, y, &mem, kind, use_aligned);
                self.stats.ymm_spills += 1;
            }
            PReg::Zmm(z) => {
                self.emit_zmm_store(code, z, &mem, kind, use_aligned);
                self.stats.zmm_spills += 1;
            }
            PReg::Gpr(_) => {
                return Err(SpillError::NotVectorRegister);
            }
        }

        // Update stats
        self.stats.total_spills += 1;
        if use_aligned {
            self.stats.aligned_ops += 1;
        } else {
            self.stats.unaligned_ops += 1;
        }

        Ok(code.len() - start_len)
    }

    /// Emit a reload operation: load register from stack.
    pub fn emit_reload(
        &mut self,
        code: &mut Vec<u8>,
        reg: PReg,
        slot: SpillSlot,
        kind: DataKind,
    ) -> Result<usize, SpillError> {
        let start_len = code.len();

        // Validate width match
        self.validate_width(reg, slot)?;

        // Create memory operand
        let mem = self.slot_to_mem(slot);

        // Check alignment
        let use_aligned = self.prefer_aligned && self.is_slot_aligned(&slot);

        // Emit based on register type
        match reg {
            PReg::Xmm(x) => {
                let ymm = Ymm::from_xmm(x);
                self.emit_xmm_load(code, ymm, &mem, kind, use_aligned);
            }
            PReg::Ymm(y) => {
                self.emit_ymm_load(code, y, &mem, kind, use_aligned);
            }
            PReg::Zmm(z) => {
                self.emit_zmm_load(code, z, &mem, kind, use_aligned);
            }
            PReg::Gpr(_) => {
                return Err(SpillError::NotVectorRegister);
            }
        }

        // Update stats
        self.stats.total_reloads += 1;
        if use_aligned {
            self.stats.aligned_ops += 1;
        } else {
            self.stats.unaligned_ops += 1;
        }

        Ok(code.len() - start_len)
    }

    /// Emit a register-to-register move.
    ///
    /// Automatically selects the optimal move instruction based on
    /// register width and data kind. Self-moves are elided if enabled.
    pub fn emit_move(
        &mut self,
        code: &mut Vec<u8>,
        dst: PReg,
        src: PReg,
        kind: DataKind,
    ) -> Result<usize, SpillError> {
        // Check for self-move
        if self.elide_self_moves && dst == src {
            self.stats.elided_moves += 1;
            return Ok(0);
        }

        // Width must match
        if dst.width() != src.width() {
            return Err(SpillError::WidthMismatch {
                reg_width: dst.width(),
                slot_width: (src.width() / 8) as u32,
            });
        }

        let start_len = code.len();

        match (dst, src) {
            (PReg::Xmm(dx), PReg::Xmm(sx)) => {
                let dy = Ymm::from_xmm(dx);
                let sy = Ymm::from_xmm(sx);
                let enc = if self.prefer_integer_moves || kind.is_integer() {
                    encode_vmovdqa_xmm_rr(dy, sy)
                } else {
                    encode_vmovapd_xmm_rr(dy, sy)
                };
                code.extend_from_slice(enc.as_slice());
            }
            (PReg::Ymm(dy), PReg::Ymm(sy)) => {
                let enc = if self.prefer_integer_moves || kind.is_integer() {
                    encode_vmovdqa_ymm_rr(dy, sy)
                } else {
                    encode_vmovapd_ymm_rr(dy, sy)
                };
                code.extend_from_slice(enc.as_slice());
            }
            (PReg::Zmm(dz), PReg::Zmm(sz)) => {
                let enc = if self.prefer_integer_moves || kind.is_integer() {
                    evex::encode_vmovdqa64_zmm_rr(dz, sz)
                } else {
                    encode_vmovapd_zmm_rr(dz, sz)
                };
                code.extend_from_slice(enc.as_slice());
            }
            _ => {
                return Err(SpillError::InvalidRegister(dst));
            }
        }

        self.stats.reg_moves += 1;

        Ok(code.len() - start_len)
    }

    // =========================================================================
    // Internal: Memory Addressing
    // =========================================================================

    /// Convert a spill slot to a memory operand.
    #[inline]
    fn slot_to_mem(&self, slot: SpillSlot) -> MemOperand {
        MemOperand::base_disp(self.frame_base.as_gpr(), slot.offset())
    }

    /// Check if a slot is properly aligned for its width.
    #[inline]
    fn is_slot_aligned(&self, slot: &SpillSlot) -> bool {
        let alignment = slot.alignment() as i32;
        let offset = slot.offset();
        // For RBP-relative addressing, offset is negative
        // Alignment check: offset must be divisible by alignment
        (offset.unsigned_abs() as i32) % alignment == 0
    }

    /// Validate that register width matches slot width.
    fn validate_width(&self, reg: PReg, slot: SpillSlot) -> Result<(), SpillError> {
        let reg_bytes = reg.width() / 8;
        let slot_bytes = slot.size();
        if reg_bytes as u32 != slot_bytes {
            return Err(SpillError::WidthMismatch {
                reg_width: reg.width(),
                slot_width: slot_bytes,
            });
        }
        Ok(())
    }

    // =========================================================================
    // Internal: XMM (128-bit) Operations
    // =========================================================================

    fn emit_xmm_store(
        &self,
        code: &mut Vec<u8>,
        reg: Ymm,
        mem: &MemOperand,
        kind: DataKind,
        aligned: bool,
    ) {
        // XMM uses VEX L=0 encoding through the Ymm-typed functions
        let enc = if self.prefer_integer_moves || kind.is_integer() {
            if aligned {
                // VMOVDQA xmm, [mem] uses opcode 7F for store
                simd::encode_vmovdqa_mr(mem, reg)
            } else {
                simd::encode_vmovupd_mr(mem, reg) // Fallback to unaligned float
            }
        } else if aligned {
            simd::encode_vmovapd_mr(mem, reg)
        } else {
            simd::encode_vmovupd_mr(mem, reg)
        };
        code.extend_from_slice(enc.as_slice());
    }

    fn emit_xmm_load(
        &self,
        code: &mut Vec<u8>,
        reg: Ymm,
        mem: &MemOperand,
        kind: DataKind,
        aligned: bool,
    ) {
        let enc = if self.prefer_integer_moves || kind.is_integer() {
            if aligned {
                simd::encode_vmovdqa_rm(reg, mem)
            } else {
                simd::encode_vmovupd_rm(reg, mem) // Fallback
            }
        } else if aligned {
            simd::encode_vmovapd_rm(reg, mem)
        } else {
            simd::encode_vmovupd_rm(reg, mem)
        };
        code.extend_from_slice(enc.as_slice());
    }

    // =========================================================================
    // Internal: YMM (256-bit) Operations
    // =========================================================================

    fn emit_ymm_store(
        &self,
        code: &mut Vec<u8>,
        reg: Ymm,
        mem: &MemOperand,
        kind: DataKind,
        aligned: bool,
    ) {
        let enc = if self.prefer_integer_moves || kind.is_integer() {
            if aligned {
                encode_vmovdqa_mr(mem, reg)
            } else {
                simd::encode_vmovdqu_mr(mem, reg)
            }
        } else if aligned {
            encode_vmovapd_mr(mem, reg)
        } else {
            encode_vmovupd_mr(mem, reg)
        };
        code.extend_from_slice(enc.as_slice());
    }

    fn emit_ymm_load(
        &self,
        code: &mut Vec<u8>,
        reg: Ymm,
        mem: &MemOperand,
        kind: DataKind,
        aligned: bool,
    ) {
        let enc = if self.prefer_integer_moves || kind.is_integer() {
            if aligned {
                encode_vmovdqa_rm(reg, mem)
            } else {
                simd::encode_vmovdqu_rm(reg, mem)
            }
        } else if aligned {
            encode_vmovapd_rm(reg, mem)
        } else {
            encode_vmovupd_rm(reg, mem)
        };
        code.extend_from_slice(enc.as_slice());
    }

    // =========================================================================
    // Internal: ZMM (512-bit) Operations
    // =========================================================================

    fn emit_zmm_store(
        &self,
        code: &mut Vec<u8>,
        reg: Zmm,
        mem: &MemOperand,
        _kind: DataKind,
        aligned: bool,
    ) {
        // For ZMM, we use EVEX encoding
        // Note: VMOVDQA64 for stores requires the store opcode variant
        let enc = if aligned {
            encode_vmovapd_zmm_mr(mem, reg)
        } else {
            encode_vmovupd_zmm_mr(mem, reg)
        };
        code.extend_from_slice(enc.as_slice());
    }

    fn emit_zmm_load(
        &self,
        code: &mut Vec<u8>,
        reg: Zmm,
        mem: &MemOperand,
        _kind: DataKind,
        aligned: bool,
    ) {
        let enc = if aligned {
            encode_vmovapd_zmm_rm(reg, mem)
        } else {
            encode_vmovupd_zmm_rm(reg, mem)
        };
        code.extend_from_slice(enc.as_slice());
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get the required spill slot width for a register class.
#[inline]
pub const fn spill_width_for_class(class: RegClass) -> SpillWidth {
    SpillWidth::from_reg_class(class)
}

/// Get the optimal alignment for a vector width.
#[inline]
pub const fn alignment_for_width(width: SpillWidth) -> u32 {
    width.alignment()
}

/// Check if an instruction encoding would benefit from aligned access.
///
/// Returns true if the memory address is aligned to the access width.
#[inline]
pub fn is_aligned_access(offset: i32, width: SpillWidth) -> bool {
    let alignment = width.alignment() as i32;
    (offset.abs()) % alignment == 0
}
