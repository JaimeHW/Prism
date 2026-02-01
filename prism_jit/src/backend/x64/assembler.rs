//! High-level x64 assembler with label support.
//!
//! This module provides a high-level interface for emitting x64 machine code,
//! building on top of the low-level encoder. Features include:
//!
//! - Fluent API for instruction emission
//! - Label creation and resolution
//! - Forward reference patching
//! - Constant pool management
//! - Relocation tracking for GC
//!
//! # Example
//! ```ignore
//! let mut asm = Assembler::new();
//! let loop_start = asm.create_label();
//! asm.bind_label(loop_start);
//! asm.add_rr(Gpr::Rax, Gpr::Rcx);
//! asm.sub_ri(Gpr::Rcx, 1);
//! asm.jnz(loop_start);
//! asm.ret();
//! let code = asm.finalize();
//! ```

use super::encoder::*;
use super::memory::ExecutableBuffer;
use super::registers::{CallingConvention, Gpr, MemOperand, Xmm};

use std::collections::HashMap;

// =============================================================================
// Label Management
// =============================================================================

/// A label representing a position in the code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(u32);

impl Label {
    /// Create a new label with the given ID.
    const fn new(id: u32) -> Self {
        Label(id)
    }

    /// Get the label's ID.
    pub const fn id(self) -> u32 {
        self.0
    }
}

/// Represents an unresolved reference to a label.
#[derive(Debug, Clone, Copy)]
struct LabelRef {
    /// Offset in the code buffer where the reference is.
    offset: usize,
    /// Size of the reference (1 for rel8, 4 for rel32).
    size: u8,
    /// Target label.
    label: Label,
}

// =============================================================================
// Relocation Types
// =============================================================================

/// Types of relocations that may need to be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationType {
    /// Absolute 64-bit pointer (for constant pool entries).
    Abs64,
    /// RIP-relative 32-bit offset.
    Rel32,
}

/// A relocation entry for GC or linking.
#[derive(Debug, Clone, Copy)]
pub struct Relocation {
    /// Offset in the code buffer.
    pub offset: usize,
    /// Type of relocation.
    pub kind: RelocationType,
    /// Index into the constant pool (for Abs64) or other metadata.
    pub index: u32,
}

// =============================================================================
// Constant Pool
// =============================================================================

/// A constant pool entry.
#[derive(Debug, Clone, Copy)]
pub enum ConstantPoolEntry {
    /// A 64-bit integer constant.
    Int64(i64),
    /// A 64-bit floating-point constant.
    Float64(f64),
    /// An absolute address (e.g., function pointer).
    Address(u64),
}

impl ConstantPoolEntry {
    /// Get the raw bytes of this constant.
    fn to_bytes(self) -> [u8; 8] {
        match self {
            ConstantPoolEntry::Int64(v) => v.to_le_bytes(),
            ConstantPoolEntry::Float64(v) => v.to_le_bytes(),
            ConstantPoolEntry::Address(v) => v.to_le_bytes(),
        }
    }
}

/// Manages constants that need to be emitted with the code.
#[derive(Debug, Default)]
pub struct ConstantPool {
    entries: Vec<ConstantPoolEntry>,
    /// Map from constant bytes to index (for deduplication).
    dedup_map: HashMap<[u8; 8], u32>,
}

impl ConstantPool {
    /// Create a new empty constant pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constant and return its index.
    pub fn add(&mut self, entry: ConstantPoolEntry) -> u32 {
        let bytes = entry.to_bytes();

        // Check for duplicate
        if let Some(&idx) = self.dedup_map.get(&bytes) {
            return idx;
        }

        let idx = self.entries.len() as u32;
        self.entries.push(entry);
        self.dedup_map.insert(bytes, idx);
        idx
    }

    /// Add a 64-bit float constant.
    pub fn add_f64(&mut self, val: f64) -> u32 {
        self.add(ConstantPoolEntry::Float64(val))
    }

    /// Add a 64-bit integer constant.
    pub fn add_i64(&mut self, val: i64) -> u32 {
        self.add(ConstantPoolEntry::Int64(val))
    }

    /// Add an address constant.
    pub fn add_address(&mut self, addr: u64) -> u32 {
        self.add(ConstantPoolEntry::Address(addr))
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by index.
    pub fn get(&self, idx: u32) -> Option<ConstantPoolEntry> {
        self.entries.get(idx as usize).copied()
    }

    /// Total size in bytes (8 bytes per entry).
    pub fn size(&self) -> usize {
        self.entries.len() * 8
    }

    /// Emit the constant pool to bytes.
    pub fn emit(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size());
        for entry in &self.entries {
            bytes.extend_from_slice(&entry.to_bytes());
        }
        bytes
    }
}

// =============================================================================
// Main Assembler
// =============================================================================

/// High-level x64 assembler with label support.
pub struct Assembler {
    /// Code bytes being assembled.
    code: Vec<u8>,
    /// Next label ID.
    next_label: u32,
    /// Map from label to bound offset.
    label_offsets: HashMap<Label, usize>,
    /// Unresolved label references.
    label_refs: Vec<LabelRef>,
    /// Constant pool.
    constants: ConstantPool,
    /// Relocations for GC.
    relocations: Vec<Relocation>,
    /// Target calling convention.
    calling_convention: CallingConvention,
}

impl Assembler {
    /// Create a new assembler with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(4096)
    }

    /// Create a new assembler with specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Assembler {
            code: Vec::with_capacity(capacity),
            next_label: 0,
            label_offsets: HashMap::new(),
            label_refs: Vec::new(),
            constants: ConstantPool::new(),
            relocations: Vec::new(),
            calling_convention: CallingConvention::host(),
        }
    }

    /// Get the current code offset.
    #[inline]
    pub fn offset(&self) -> usize {
        self.code.len()
    }

    /// Get the current code as a slice.
    pub fn code(&self) -> &[u8] {
        &self.code
    }

    /// Get the calling convention.
    pub fn calling_convention(&self) -> CallingConvention {
        self.calling_convention
    }

    /// Get the constant pool.
    pub fn constants(&self) -> &ConstantPool {
        &self.constants
    }

    /// Get mutable access to the constant pool.
    pub fn constants_mut(&mut self) -> &mut ConstantPool {
        &mut self.constants
    }

    /// Get the relocations.
    pub fn relocations(&self) -> &[Relocation] {
        &self.relocations
    }

    // =========================================================================
    // Label Management
    // =========================================================================

    /// Create a new unbound label.
    pub fn create_label(&mut self) -> Label {
        let label = Label::new(self.next_label);
        self.next_label += 1;
        label
    }

    /// Bind a label to the current offset.
    pub fn bind_label(&mut self, label: Label) {
        let offset = self.offset();
        self.label_offsets.insert(label, offset);
    }

    /// Check if a label is bound.
    pub fn is_label_bound(&self, label: Label) -> bool {
        self.label_offsets.contains_key(&label)
    }

    /// Get the offset of a bound label.
    pub fn label_offset(&self, label: Label) -> Option<usize> {
        self.label_offsets.get(&label).copied()
    }

    // =========================================================================
    // Low-Level Emission
    // =========================================================================

    /// Emit raw bytes.
    #[inline]
    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    /// Emit a single byte.
    #[inline]
    pub fn emit_u8(&mut self, byte: u8) {
        self.code.push(byte);
    }

    /// Emit a 32-bit value (little-endian).
    #[inline]
    pub fn emit_u32(&mut self, val: u32) {
        self.code.extend_from_slice(&val.to_le_bytes());
    }

    /// Emit a 64-bit value (little-endian).
    #[inline]
    pub fn emit_u64(&mut self, val: u64) {
        self.code.extend_from_slice(&val.to_le_bytes());
    }

    /// Emit an encoded instruction.
    #[inline]
    pub fn emit_encoded(&mut self, enc: &EncodedInst) {
        self.emit_bytes(enc.as_slice());
    }

    /// Patch bytes at a specific offset.
    #[inline]
    pub fn patch_bytes(&mut self, offset: usize, bytes: &[u8]) {
        self.code[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// Patch a 32-bit value at a specific offset.
    #[inline]
    pub fn patch_i32(&mut self, offset: usize, val: i32) {
        self.patch_bytes(offset, &val.to_le_bytes());
    }

    // =========================================================================
    // Arithmetic Instructions
    // =========================================================================

    /// ADD r64, r64
    #[inline]
    pub fn add_rr(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_add_rr(dst, src));
    }

    /// ADD r64, imm32
    #[inline]
    pub fn add_ri(&mut self, dst: Gpr, imm: i32) {
        self.emit_encoded(&encode_add_ri32(dst, imm));
    }

    /// ADD r64, [mem]
    #[inline]
    pub fn add_rm(&mut self, dst: Gpr, mem: &MemOperand) {
        self.emit_encoded(&encode_add_rm(dst, mem));
    }

    /// SUB r64, r64
    #[inline]
    pub fn sub_rr(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_sub_rr(dst, src));
    }

    /// SUB r64, imm32
    #[inline]
    pub fn sub_ri(&mut self, dst: Gpr, imm: i32) {
        self.emit_encoded(&encode_sub_ri32(dst, imm));
    }

    /// IMUL r64, r64
    #[inline]
    pub fn imul_rr(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_imul_rr(dst, src));
    }

    /// IMUL r64, r64, imm32
    #[inline]
    pub fn imul_rri(&mut self, dst: Gpr, src: Gpr, imm: i32) {
        self.emit_encoded(&encode_imul_rri32(dst, src, imm));
    }

    /// IDIV r64 (RDX:RAX / r64 -> RAX=quotient, RDX=remainder)
    #[inline]
    pub fn idiv(&mut self, src: Gpr) {
        self.emit_encoded(&encode_idiv(src));
    }

    /// DIV r64 (unsigned)
    #[inline]
    pub fn div(&mut self, src: Gpr) {
        self.emit_encoded(&encode_div(src));
    }

    /// NEG r64
    #[inline]
    pub fn neg(&mut self, dst: Gpr) {
        self.emit_encoded(&encode_neg(dst));
    }

    /// INC r64
    #[inline]
    pub fn inc(&mut self, dst: Gpr) {
        self.emit_encoded(&encode_inc(dst));
    }

    /// DEC r64
    #[inline]
    pub fn dec(&mut self, dst: Gpr) {
        self.emit_encoded(&encode_dec(dst));
    }

    // =========================================================================
    // Bitwise Instructions
    // =========================================================================

    /// AND r64, r64
    #[inline]
    pub fn and_rr(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_and_rr(dst, src));
    }

    /// AND r64, imm32
    #[inline]
    pub fn and_ri(&mut self, dst: Gpr, imm: i32) {
        self.emit_encoded(&encode_and_ri32(dst, imm));
    }

    /// OR r64, r64
    #[inline]
    pub fn or_rr(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_or_rr(dst, src));
    }

    /// OR r64, imm32
    #[inline]
    pub fn or_ri(&mut self, dst: Gpr, imm: i32) {
        self.emit_encoded(&encode_or_ri32(dst, imm));
    }

    /// XOR r64, r64
    #[inline]
    pub fn xor_rr(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_xor_rr(dst, src));
    }

    /// XOR r64, imm32
    #[inline]
    pub fn xor_ri(&mut self, dst: Gpr, imm: i32) {
        self.emit_encoded(&encode_xor_ri32(dst, imm));
    }

    /// NOT r64
    #[inline]
    pub fn not(&mut self, dst: Gpr) {
        self.emit_encoded(&encode_not(dst));
    }

    /// SHL r64, imm8
    #[inline]
    pub fn shl_ri(&mut self, dst: Gpr, imm: u8) {
        self.emit_encoded(&encode_shl_ri(dst, imm));
    }

    /// SHL r64, cl
    #[inline]
    pub fn shl_cl(&mut self, dst: Gpr) {
        self.emit_encoded(&encode_shl_cl(dst));
    }

    /// SHR r64, imm8
    #[inline]
    pub fn shr_ri(&mut self, dst: Gpr, imm: u8) {
        self.emit_encoded(&encode_shr_ri(dst, imm));
    }

    /// SHR r64, cl
    #[inline]
    pub fn shr_cl(&mut self, dst: Gpr) {
        self.emit_encoded(&encode_shr_cl(dst));
    }

    /// SAR r64, imm8 (arithmetic shift right)
    #[inline]
    pub fn sar_ri(&mut self, dst: Gpr, imm: u8) {
        self.emit_encoded(&encode_sar_ri(dst, imm));
    }

    /// SAR r64, cl
    #[inline]
    pub fn sar_cl(&mut self, dst: Gpr) {
        self.emit_encoded(&encode_sar_cl(dst));
    }

    // =========================================================================
    // Comparison Instructions
    // =========================================================================

    /// CMP r64, r64
    #[inline]
    pub fn cmp_rr(&mut self, left: Gpr, right: Gpr) {
        self.emit_encoded(&encode_cmp_rr(left, right));
    }

    /// CMP r64, imm32
    #[inline]
    pub fn cmp_ri(&mut self, left: Gpr, imm: i32) {
        self.emit_encoded(&encode_cmp_ri32(left, imm));
    }

    /// CMP r64, [mem]
    #[inline]
    pub fn cmp_rm(&mut self, left: Gpr, mem: &MemOperand) {
        self.emit_encoded(&encode_cmp_rm(left, mem));
    }

    /// TEST r64, r64
    #[inline]
    pub fn test_rr(&mut self, left: Gpr, right: Gpr) {
        self.emit_encoded(&encode_test_rr(left, right));
    }

    /// TEST r64, imm32
    #[inline]
    pub fn test_ri(&mut self, dst: Gpr, imm: i32) {
        self.emit_encoded(&encode_test_ri32(dst, imm));
    }

    // =========================================================================
    // Data Movement Instructions
    // =========================================================================

    /// MOV r64, r64
    #[inline]
    pub fn mov_rr(&mut self, dst: Gpr, src: Gpr) {
        if dst != src {
            self.emit_encoded(&encode_mov_rr(dst, src));
        }
    }

    /// MOV r32, r32 (zero-extends)
    #[inline]
    pub fn mov_rr32(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_mov_rr32(dst, src));
    }

    /// MOV r64, imm64
    #[inline]
    pub fn mov_ri64(&mut self, dst: Gpr, imm: i64) {
        // Optimize small constants
        if imm == 0 {
            // xor rXX, rXX (3-4 bytes vs 10 for movabs)
            self.emit_encoded(&encode_xor_rr(dst, dst));
        } else if imm >= 0 && imm <= u32::MAX as i64 {
            // Use 32-bit move (5-6 bytes, zero-extends)
            self.emit_encoded(&encode_mov_ri32(dst, imm as u32));
        } else if imm >= i32::MIN as i64 && imm <= i32::MAX as i64 {
            // Use sign-extended 32-bit immediate where possible
            // MOV r64, imm32 (sign-extended) - 7 bytes
            // Actually there's no such encoding, use movabs
            self.emit_encoded(&encode_mov_ri64(dst, imm));
        } else {
            self.emit_encoded(&encode_mov_ri64(dst, imm));
        }
    }

    /// MOV r32, imm32 (zero-extends to 64-bit)
    #[inline]
    pub fn mov_ri32(&mut self, dst: Gpr, imm: u32) {
        if imm == 0 {
            self.emit_encoded(&encode_xor_rr(dst, dst));
        } else {
            self.emit_encoded(&encode_mov_ri32(dst, imm));
        }
    }

    /// MOV r64, [mem]
    #[inline]
    pub fn mov_rm(&mut self, dst: Gpr, mem: &MemOperand) {
        self.emit_encoded(&encode_mov_rm(dst, mem));
    }

    /// MOV [mem], r64
    #[inline]
    pub fn mov_mr(&mut self, mem: &MemOperand, src: Gpr) {
        self.emit_encoded(&encode_mov_mr(mem, src));
    }

    /// MOV r32, [mem] (32-bit, zero-extends)
    #[inline]
    pub fn mov_rm32(&mut self, dst: Gpr, mem: &MemOperand) {
        self.emit_encoded(&encode_mov_rm32(dst, mem));
    }

    /// MOV [mem], r32
    #[inline]
    pub fn mov_mr32(&mut self, mem: &MemOperand, src: Gpr) {
        self.emit_encoded(&encode_mov_mr32(mem, src));
    }

    /// LEA r64, [mem]
    #[inline]
    pub fn lea(&mut self, dst: Gpr, mem: &MemOperand) {
        self.emit_encoded(&encode_lea(dst, mem));
    }

    /// MOVZX r64, r/m8
    #[inline]
    pub fn movzx_rb(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_movzx_rb(dst, src));
    }

    /// MOVZX r64, r/m16
    #[inline]
    pub fn movzx_rw(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_movzx_rw(dst, src));
    }

    /// MOVSX r64, r/m8
    #[inline]
    pub fn movsx_rb(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_movsx_rb(dst, src));
    }

    /// MOVSX r64, r/m16
    #[inline]
    pub fn movsx_rw(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_movsx_rw(dst, src));
    }

    /// MOVSXD r64, r/m32
    #[inline]
    pub fn movsxd(&mut self, dst: Gpr, src: Gpr) {
        self.emit_encoded(&encode_movsxd(dst, src));
    }

    /// CDQ - Sign-extend EAX into EDX:EAX
    #[inline]
    pub fn cdq(&mut self) {
        self.emit_encoded(&encode_cdq());
    }

    /// CQO - Sign-extend RAX into RDX:RAX
    #[inline]
    pub fn cqo(&mut self) {
        self.emit_encoded(&encode_cqo());
    }

    // =========================================================================
    // Stack Instructions
    // =========================================================================

    /// PUSH r64
    #[inline]
    pub fn push(&mut self, src: Gpr) {
        self.emit_encoded(&encode_push(src));
    }

    /// POP r64
    #[inline]
    pub fn pop(&mut self, dst: Gpr) {
        self.emit_encoded(&encode_pop(dst));
    }

    // =========================================================================
    // Control Flow Instructions
    // =========================================================================

    /// JMP to label (uses rel8 if possible, rel32 otherwise).
    #[inline]
    pub fn jmp(&mut self, target: Label) {
        if let Some(target_offset) = self.label_offsets.get(&target).copied() {
            // Backward jump - we know the distance
            let here = self.offset() + 2; // Size of jmp rel8
            let rel = target_offset as isize - here as isize;
            if rel >= -128 && rel <= 127 {
                self.emit_encoded(&encode_jmp_rel8(rel as i8));
            } else {
                // Need rel32
                let here = self.offset() + 5;
                let rel = target_offset as isize - here as isize;
                self.emit_encoded(&encode_jmp_rel32(rel as i32));
            }
        } else {
            // Forward jump - use rel32 and patch later
            let ref_offset = self.offset() + 1; // Offset of the rel32
            self.emit_encoded(&encode_jmp_rel32(0));
            self.label_refs.push(LabelRef {
                offset: ref_offset,
                size: 4,
                label: target,
            });
        }
    }

    /// JMP r64 (indirect)
    #[inline]
    pub fn jmp_r(&mut self, target: Gpr) {
        self.emit_encoded(&encode_jmp_r(target));
    }

    /// Conditional jump to label.
    #[inline]
    pub fn jcc(&mut self, cond: Condition, target: Label) {
        if let Some(target_offset) = self.label_offsets.get(&target).copied() {
            // Backward jump
            let here = self.offset() + 2; // Size of jcc rel8
            let rel = target_offset as isize - here as isize;
            if rel >= -128 && rel <= 127 {
                self.emit_encoded(&encode_jcc_rel8(cond, rel as i8));
            } else {
                let here = self.offset() + 6; // Size of jcc rel32
                let rel = target_offset as isize - here as isize;
                self.emit_encoded(&encode_jcc_rel32(cond, rel as i32));
            }
        } else {
            // Forward jump - use rel32 and patch later
            let ref_offset = self.offset() + 2; // Offset of the rel32 (after 0F XX)
            self.emit_encoded(&encode_jcc_rel32(cond, 0));
            self.label_refs.push(LabelRef {
                offset: ref_offset,
                size: 4,
                label: target,
            });
        }
    }

    /// JE/JZ - Jump if equal/zero
    #[inline]
    pub fn je(&mut self, target: Label) {
        self.jcc(Condition::Equal, target);
    }

    /// JZ - Jump if zero (alias for je)
    #[inline]
    pub fn jz(&mut self, target: Label) {
        self.je(target);
    }

    /// JNE/JNZ - Jump if not equal/not zero
    #[inline]
    pub fn jne(&mut self, target: Label) {
        self.jcc(Condition::NotEqual, target);
    }

    /// JL - Jump if less (signed)

    #[inline]
    pub fn jl(&mut self, target: Label) {
        self.jcc(Condition::Less, target);
    }

    /// JLE - Jump if less or equal (signed)
    #[inline]
    pub fn jle(&mut self, target: Label) {
        self.jcc(Condition::LessEqual, target);
    }

    /// JG - Jump if greater (signed)
    #[inline]
    pub fn jg(&mut self, target: Label) {
        self.jcc(Condition::Greater, target);
    }

    /// JGE - Jump if greater or equal (signed)
    #[inline]
    pub fn jge(&mut self, target: Label) {
        self.jcc(Condition::GreaterEqual, target);
    }

    /// JB - Jump if below (unsigned)
    #[inline]
    pub fn jb(&mut self, target: Label) {
        self.jcc(Condition::Below, target);
    }

    /// JBE - Jump if below or equal (unsigned)
    #[inline]
    pub fn jbe(&mut self, target: Label) {
        self.jcc(Condition::BelowEqual, target);
    }

    /// JA - Jump if above (unsigned)
    #[inline]
    pub fn ja(&mut self, target: Label) {
        self.jcc(Condition::Above, target);
    }

    /// JAE - Jump if above or equal (unsigned)
    #[inline]
    pub fn jae(&mut self, target: Label) {
        self.jcc(Condition::AboveEqual, target);
    }

    /// JO - Jump if overflow
    #[inline]
    pub fn jo(&mut self, target: Label) {
        self.jcc(Condition::Overflow, target);
    }

    /// JNO - Jump if no overflow
    #[inline]
    pub fn jno(&mut self, target: Label) {
        self.jcc(Condition::NoOverflow, target);
    }

    /// JS - Jump if sign (negative)
    #[inline]
    pub fn js(&mut self, target: Label) {
        self.jcc(Condition::Sign, target);
    }

    /// JNS - Jump if no sign (non-negative)
    #[inline]
    pub fn jns(&mut self, target: Label) {
        self.jcc(Condition::NoSign, target);
    }

    /// SETcc r8
    #[inline]
    pub fn setcc(&mut self, cond: Condition, dst: Gpr) {
        self.emit_encoded(&encode_setcc(cond, dst));
    }

    /// CALL rel32 (to a label)
    pub fn call(&mut self, target: Label) {
        if let Some(target_offset) = self.label_offsets.get(&target).copied() {
            let here = self.offset() + 5;
            let rel = target_offset as isize - here as isize;
            self.emit_encoded(&encode_call_rel32(rel as i32));
        } else {
            let ref_offset = self.offset() + 1;
            self.emit_encoded(&encode_call_rel32(0));
            self.label_refs.push(LabelRef {
                offset: ref_offset,
                size: 4,
                label: target,
            });
        }
    }

    /// CALL r64 (indirect)
    #[inline]
    pub fn call_r(&mut self, target: Gpr) {
        self.emit_encoded(&encode_call_r(target));
    }

    /// CALL absolute address (loads address into scratch, then calls)
    pub fn call_abs(&mut self, addr: u64, scratch: Gpr) {
        self.mov_ri64(scratch, addr as i64);
        self.call_r(scratch);
    }

    /// RET
    #[inline]
    pub fn ret(&mut self) {
        self.emit_encoded(&encode_ret());
    }

    /// RET imm16
    #[inline]
    pub fn ret_imm(&mut self, imm: u16) {
        self.emit_encoded(&encode_ret_imm(imm));
    }

    /// NOP
    #[inline]
    pub fn nop(&mut self) {
        self.emit_encoded(&encode_nop());
    }

    /// INT3 (breakpoint)
    #[inline]
    pub fn int3(&mut self) {
        self.emit_encoded(&encode_int3());
    }

    /// UD2 (undefined instruction)
    #[inline]
    pub fn ud2(&mut self) {
        self.emit_encoded(&encode_ud2());
    }

    // =========================================================================
    // SSE Floating Point Instructions
    // =========================================================================

    /// MOVSD xmm, xmm
    #[inline]
    pub fn movsd_rr(&mut self, dst: Xmm, src: Xmm) {
        if dst != src {
            self.emit_encoded(&encode_movsd_rr(dst, src));
        }
    }

    /// MOVSD xmm, [mem]
    #[inline]
    pub fn movsd_rm(&mut self, dst: Xmm, mem: &MemOperand) {
        self.emit_encoded(&encode_movsd_rm(dst, mem));
    }

    /// MOVSD [mem], xmm
    #[inline]
    pub fn movsd_mr(&mut self, mem: &MemOperand, src: Xmm) {
        self.emit_encoded(&encode_movsd_mr(mem, src));
    }

    /// ADDSD xmm, xmm
    #[inline]
    pub fn addsd(&mut self, dst: Xmm, src: Xmm) {
        self.emit_encoded(&encode_addsd(dst, src));
    }

    /// SUBSD xmm, xmm
    #[inline]
    pub fn subsd(&mut self, dst: Xmm, src: Xmm) {
        self.emit_encoded(&encode_subsd(dst, src));
    }

    /// MULSD xmm, xmm
    #[inline]
    pub fn mulsd(&mut self, dst: Xmm, src: Xmm) {
        self.emit_encoded(&encode_mulsd(dst, src));
    }

    /// DIVSD xmm, xmm
    #[inline]
    pub fn divsd(&mut self, dst: Xmm, src: Xmm) {
        self.emit_encoded(&encode_divsd(dst, src));
    }

    /// SQRTSD xmm, xmm
    #[inline]
    pub fn sqrtsd(&mut self, dst: Xmm, src: Xmm) {
        self.emit_encoded(&encode_sqrtsd(dst, src));
    }

    /// UCOMISD xmm, xmm
    #[inline]
    pub fn ucomisd(&mut self, left: Xmm, right: Xmm) {
        self.emit_encoded(&encode_ucomisd(left, right));
    }

    /// CVTSI2SD xmm, r64
    #[inline]
    pub fn cvtsi2sd(&mut self, dst: Xmm, src: Gpr) {
        self.emit_encoded(&encode_cvtsi2sd(dst, src));
    }

    /// CVTTSD2SI r64, xmm
    #[inline]
    pub fn cvttsd2si(&mut self, dst: Gpr, src: Xmm) {
        self.emit_encoded(&encode_cvttsd2si(dst, src));
    }

    /// XORPD xmm, xmm (for zeroing XMM registers)
    #[inline]
    pub fn xorpd(&mut self, dst: Xmm, src: Xmm) {
        self.emit_encoded(&encode_xorpd(dst, src));
    }

    /// Zero an XMM register using XORPD
    #[inline]
    pub fn zero_xmm(&mut self, dst: Xmm) {
        self.xorpd(dst, dst);
    }

    // =========================================================================
    // Prologue/Epilogue Helpers
    // =========================================================================

    /// Emit a standard function prologue.
    pub fn emit_prologue(&mut self, frame_size: i32, saved_regs: &[Gpr]) {
        // Push callee-saved registers
        for &reg in saved_regs {
            self.push(reg);
        }

        // Set up frame pointer (optional)
        self.push(Gpr::Rbp);
        self.mov_rr(Gpr::Rbp, Gpr::Rsp);

        // Allocate stack space
        if frame_size > 0 {
            self.sub_ri(Gpr::Rsp, frame_size);
        }
    }

    /// Emit a standard function epilogue.
    pub fn emit_epilogue(&mut self, frame_size: i32, saved_regs: &[Gpr]) {
        // Deallocate stack space
        if frame_size > 0 {
            self.add_ri(Gpr::Rsp, frame_size);
        }

        // Restore frame pointer
        self.pop(Gpr::Rbp);

        // Pop callee-saved registers (in reverse order)
        for &reg in saved_regs.iter().rev() {
            self.pop(reg);
        }

        self.ret();
    }

    // =========================================================================
    // Finalization
    // =========================================================================

    /// Resolve all label references.
    fn resolve_labels(&mut self) -> Result<(), String> {
        for ref_ in &self.label_refs {
            let target_offset = self
                .label_offsets
                .get(&ref_.label)
                .ok_or_else(|| format!("Unbound label: {:?}", ref_.label))?;

            // Calculate relative offset from end of instruction
            let here = ref_.offset + ref_.size as usize;
            let rel = *target_offset as isize - here as isize;

            match ref_.size {
                1 => {
                    if rel < -128 || rel > 127 {
                        return Err(format!("Label reference out of range for rel8: {}", rel));
                    }
                    self.code[ref_.offset] = rel as i8 as u8;
                }
                4 => {
                    let bytes = (rel as i32).to_le_bytes();
                    self.code[ref_.offset..ref_.offset + 4].copy_from_slice(&bytes);
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    /// Finalize the assembly and return the code bytes.
    pub fn finalize(mut self) -> Result<Vec<u8>, String> {
        self.resolve_labels()?;

        // Append constant pool if needed
        if !self.constants.is_empty() {
            // Align to 8 bytes
            while self.code.len() % 8 != 0 {
                self.code.push(0x00);
            }
            self.code.extend(self.constants.emit());
        }

        Ok(self.code)
    }

    /// Finalize and create an executable buffer.
    pub fn finalize_executable(self) -> Result<ExecutableBuffer, String> {
        let code = self.finalize()?;
        let mut buf = ExecutableBuffer::new(code.len())
            .ok_or_else(|| "Failed to allocate executable memory".to_string())?;

        buf.emit_bytes(&code);
        if !buf.make_executable() {
            return Err("Failed to make memory executable".to_string());
        }

        Ok(buf)
    }

    /// Get the code bytes without consuming the assembler.
    pub fn code_bytes(&self) -> &[u8] {
        &self.code
    }
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assembler_basic() {
        let mut asm = Assembler::new();
        asm.mov_ri64(Gpr::Rax, 42);
        asm.ret();

        let code = asm.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_assembler_labels_backward() {
        let mut asm = Assembler::new();
        let loop_label = asm.create_label();

        asm.bind_label(loop_label);
        asm.add_ri(Gpr::Rax, 1);
        asm.cmp_ri(Gpr::Rax, 10);
        asm.jl(loop_label);
        asm.ret();

        let code = asm.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_assembler_labels_forward() {
        let mut asm = Assembler::new();
        let skip = asm.create_label();

        asm.cmp_ri(Gpr::Rax, 0);
        asm.je(skip);
        asm.mov_ri64(Gpr::Rax, 1);
        asm.bind_label(skip);
        asm.ret();

        let code = asm.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_assembler_mov_optimization() {
        let mut asm = Assembler::new();

        // mov rax, 0 should become xor rax, rax
        asm.mov_ri64(Gpr::Rax, 0);
        let code_zero = asm.code().len();

        // mov rax, 1 should use 32-bit move
        asm.mov_ri64(Gpr::Rax, 1);
        let code_one = asm.code().len() - code_zero;

        // mov rax, imm64 for large values
        asm.mov_ri64(Gpr::Rax, 0x123456789ABCDEF0u64 as i64);
        let code_large = asm.code().len() - code_zero - code_one;

        // xor rax, rax is 3 bytes
        assert!(code_zero <= 4);
        // mov eax, 1 is 5 bytes
        assert!(code_one <= 6);
        // movabs is 10 bytes
        assert_eq!(code_large, 10);
    }

    #[test]
    fn test_constant_pool() {
        let mut pool = ConstantPool::new();

        let idx1 = pool.add_f64(3.14);
        let idx2 = pool.add_f64(3.14); // Should be deduplicated
        let idx3 = pool.add_i64(42);

        assert_eq!(idx1, idx2);
        assert_ne!(idx1, idx3);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn test_execute_simple() {
        let mut asm = Assembler::new();
        asm.mov_ri64(Gpr::Rax, 42);
        asm.ret();

        let buf = asm.finalize_executable().unwrap();

        type Fn = unsafe extern "C" fn() -> i64;
        let f: Fn = unsafe { buf.as_fn() };
        let result = unsafe { f() };

        assert_eq!(result, 42);
    }

    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn test_execute_loop() {
        let mut asm = Assembler::new();
        let loop_label = asm.create_label();
        let exit = asm.create_label();

        // int sum = 0;
        asm.xor_rr(Gpr::Rax, Gpr::Rax);
        // int i = 10;
        asm.mov_ri64(Gpr::Rcx, 10);

        // loop:
        asm.bind_label(loop_label);
        // if (i == 0) goto exit;
        asm.test_rr(Gpr::Rcx, Gpr::Rcx);
        asm.je(exit);
        // sum += i;
        asm.add_rr(Gpr::Rax, Gpr::Rcx);
        // i--;
        asm.dec(Gpr::Rcx);
        // goto loop;
        asm.jmp(loop_label);

        // exit:
        asm.bind_label(exit);
        asm.ret();

        let buf = asm.finalize_executable().unwrap();

        type Fn = unsafe extern "C" fn() -> i64;
        let f: Fn = unsafe { buf.as_fn() };
        let result = unsafe { f() };

        // sum = 10 + 9 + 8 + ... + 1 = 55
        assert_eq!(result, 55);
    }

    #[test]
    fn test_prologue_epilogue() {
        let mut asm = Assembler::new();
        asm.emit_prologue(32, &[Gpr::Rbx, Gpr::R12]);
        asm.mov_ri64(Gpr::Rax, 0);
        asm.emit_epilogue(32, &[Gpr::Rbx, Gpr::R12]);

        let code = asm.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_sse_operations() {
        let mut asm = Assembler::new();
        asm.zero_xmm(Xmm::Xmm0);
        asm.cvtsi2sd(Xmm::Xmm1, Gpr::Rax);
        asm.addsd(Xmm::Xmm0, Xmm::Xmm1);
        asm.mulsd(Xmm::Xmm0, Xmm::Xmm1);
        asm.cvttsd2si(Gpr::Rax, Xmm::Xmm0);
        asm.ret();

        let code = asm.finalize().unwrap();
        assert!(!code.is_empty());
    }
}
