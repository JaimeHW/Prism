//! Tuple-specialized JIT templates for high-performance tuple operations.
//!
//! Provides type-specialized native code generation for:
//! - **Tuple indexing** (`tuple[i]`) — fully inline with negative index normalization
//! - **Tuple length** (`len(tuple)`) — fully inline, no runtime call
//! - **Tuple containment** (`x in tuple`) — guard + deopt for linear scan
//! - **Tuple concatenation** (`tuple + tuple`) — guard + deopt for allocation
//! - **Tuple repetition** (`tuple * n`) — guard + deopt for allocation
//!
//! # TupleObject Memory Layout
//!
//! `TupleObject` is `#[repr(C)]`:
//! ```text
//! Offset  Size  Field
//! ──────  ────  ─────────────────────
//!   0      4    type_id  (TypeId::TUPLE = 7)
//!   4      4    gc_flags (AtomicU32)
//!   8      8    hash     (u64)
//!  16      8    Box<[Value]>.ptr   (data pointer)
//!  24      8    Box<[Value]>.len   (element count)
//! ```
//!
//! Unlike `ListObject` (which uses `Vec<Value>` with ptr/len/cap),
//! `TupleObject` uses `Box<[Value]>` with only ptr/len — no capacity field,
//! since tuples are immutable and never grow.
//!
//! # Optimization Strategy
//!
//! 1. **Fully inlined** (`TupleIndex`, `TupleLen`): These are simple field
//!    loads and arithmetic. The JIT can emit all logic inline:
//!    - Index: type guard, extract index, normalize negative, bounds check, load
//!    - Len: type guard, load len, NaN-box as int
//!
//! 2. **Guard + deopt** (`TupleContains`, `TupleConcat`, `TupleRepeat`):
//!    The inline type guard eliminates interpreter-level type dispatch.
//!    The actual operation (linear scan, allocation) deopts to the interpreter.

use super::specialize_common::{
    emit_int_check_and_extract, emit_typed_object_check_and_extract, type_ids,
};
use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Gpr;
use crate::backend::x64::registers::{MemOperand, Scale};

// =============================================================================
// Tuple Layout Constants
// =============================================================================

/// Memory layout constants for `TupleObject`.
///
/// These offsets correspond to the `#[repr(C)]` layout of `TupleObject`:
/// - `ObjectHeader` (16 bytes) at offset 0
/// - `Box<[Value]>` (ptr + len, no cap) starting at offset 16
pub mod tuple_layout {
    /// Offset of `Box<[Value]>.ptr` within TupleObject.
    ///
    /// This is `ObjectHeader::SIZE` = 16 bytes.
    pub const ITEMS_PTR_OFFSET: i32 = 16;

    /// Offset of `Box<[Value]>.len` within TupleObject.
    ///
    /// Immediately follows the data pointer: 16 + 8 = 24.
    pub const ITEMS_LEN_OFFSET: i32 = 24;

    /// Size of a single `Value` (u64 = 8 bytes).
    pub const VALUE_SIZE: i32 = 8;
}

// =============================================================================
// Helper: Emit tuple type check and extract
// =============================================================================

/// Emit code to verify a value is a tuple and extract the object pointer.
///
/// Performs:
/// 1. Check OBJECT_TAG in NaN-box
/// 2. Extract 48-bit pointer payload
/// 3. Load `type_id` from ObjectHeader and verify it equals `TypeId::TUPLE`
///
/// After this function, `dst` contains the raw `TupleObject` pointer.
#[inline]
fn emit_tuple_check_and_extract(
    ctx: &mut TemplateContext,
    src: Gpr,
    dst: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    emit_typed_object_check_and_extract(ctx, src, dst, scratch, type_ids::TUPLE, deopt_idx);
}

// =============================================================================
// Helper: Emit integer NaN-box for length results
// =============================================================================

/// Emit code to NaN-box a non-negative integer value (e.g. tuple length).
///
/// Applies `PAYLOAD_MASK` then ORs with `(QNAN_BITS | INT_TAG)`.
/// The value register is modified in-place. `scratch` is clobbered.
#[inline]
fn emit_int_box(ctx: &mut TemplateContext, value: Gpr, scratch: Gpr) {
    ctx.asm.mov_ri64(scratch, value_tags::PAYLOAD_MASK as i64);
    ctx.asm.and_rr(value, scratch);
    let tag = (value_tags::QNAN_BITS | value_tags::INT_TAG) as i64;
    ctx.asm.mov_ri64(scratch, tag);
    ctx.asm.or_rr(value, scratch);
}

// =============================================================================
// Tuple Index Template
// =============================================================================

/// Template for tuple indexing (`tuple[i]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check tuple: OBJECT_TAG + TypeId::TUPLE
/// 2. Type-check index: INT_TAG, extract signed 48-bit integer
/// 3. Load tuple length
/// 4. Normalize negative index: `if index < 0 { index += len }`
/// 5. Bounds check: unsigned `index >= len` → deopt (IndexError)
/// 6. Load items pointer, load `items[index * 8]`
/// 7. Store result to destination register
///
/// This is fully inlined — no runtime call needed. The negative index
/// normalization uses the same branchless pattern as `ListIndexTemplate`:
/// test-and-branch with `JGE` to skip the `ADD` for positive indices.
///
/// # Estimated Code Size
///
/// ~200 bytes: tuple guard (~32), int guard (~20), len load (~7),
/// negative normalization (~12), bounds check (~8), items load (~14),
/// indexed load (~4), store (~7), overhead (~40+)
pub struct TupleIndexTemplate {
    pub dst_reg: u8,
    pub tuple_reg: u8,
    pub index_reg: u8,
    pub deopt_idx: usize,
}

impl TupleIndexTemplate {
    /// Create a new tuple index template.
    #[inline]
    pub fn new(dst_reg: u8, tuple_reg: u8, index_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            tuple_reg,
            index_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleIndexTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load tuple value from frame
        let tuple_slot = ctx.frame.register_slot(self.tuple_reg as u16);
        ctx.asm.mov_rm(acc, &tuple_slot);

        // Type-check: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = TupleObject pointer

        // Load index value from frame
        let index_slot = ctx.frame.register_slot(self.index_reg as u16);
        ctx.asm.mov_rm(scratch1, &index_slot);

        // Type-check index: INT_TAG
        emit_int_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);
        // scratch1 = sign-extended index value

        // Load length: scratch2 = tuple->items.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: tuple_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(scratch2, &len_mem);

        // Handle negative index: if index < 0, index += len
        ctx.asm.cmp_ri(scratch1, 0);
        let positive_label = ctx.asm.create_label();
        ctx.asm.jge(positive_label);
        // Negative path: scratch1 += scratch2 (index += len)
        ctx.asm.add_rr(scratch1, scratch2);
        ctx.asm.bind_label(positive_label);

        // Bounds check: unsigned index >= len → deopt (IndexError)
        // After normalization, if still negative, unsigned cmp catches it
        ctx.asm.cmp_rr(scratch1, scratch2);
        ctx.asm.jae(ctx.deopt_label(self.deopt_idx));

        // Load items pointer: acc = tuple->items.ptr
        let items_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: tuple_layout::ITEMS_PTR_OFFSET,
        };
        ctx.asm.mov_rm(acc, &items_mem);

        // Load value: acc = items_ptr[index * 8]
        let item_mem = MemOperand {
            base: Some(acc),
            index: Some(scratch1),
            scale: Scale::X8,
            disp: 0,
        };
        ctx.asm.mov_rm(acc, &item_mem);

        // Store result to destination
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        200
    }
}

// =============================================================================
// Tuple Length Template
// =============================================================================

/// Template for tuple length (`len(tuple)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check tuple: OBJECT_TAG + TypeId::TUPLE
/// 2. Load `items.len` field (offset 24)
/// 3. NaN-box as integer: `(QNAN_BITS | INT_TAG) | (len & PAYLOAD_MASK)`
/// 4. Store result
///
/// Fully inlined — trivial field load + integer boxing.
///
/// # Estimated Code Size
///
/// ~120 bytes: tuple guard (~32), len load (~7), int boxing (~24),
/// store (~7), overhead (~50)
pub struct TupleLenTemplate {
    pub dst_reg: u8,
    pub tuple_reg: u8,
    pub deopt_idx: usize,
}

impl TupleLenTemplate {
    /// Create a new tuple length template.
    #[inline]
    pub fn new(dst_reg: u8, tuple_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            tuple_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleLenTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load tuple value from frame
        let tuple_slot = ctx.frame.register_slot(self.tuple_reg as u16);
        ctx.asm.mov_rm(acc, &tuple_slot);

        // Type-check: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = TupleObject pointer

        // Load length: acc = tuple->items.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: tuple_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(acc, &len_mem);

        // NaN-box as integer
        emit_int_box(ctx, acc, scratch1);

        // Store result to destination
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Tuple Contains Template
// =============================================================================

/// Template for tuple containment check (`x in tuple`).
///
/// # Code Generation Strategy
///
/// 1. Type-check tuple: OBJECT_TAG + TypeId::TUPLE
/// 2. Deopt to interpreter for linear scan
///
/// The inline type guard eliminates interpreter-level polymorphic dispatch.
/// The actual containment check requires a loop over items with value
/// comparison — too complex for Tier 1 inline code.
///
/// # Estimated Code Size
///
/// ~80 bytes: tuple guard (~32), value load (~7), jmp (~8), overhead (~33)
pub struct TupleContainsTemplate {
    pub tuple_reg: u8,
    pub value_reg: u8,
    pub dst_reg: u8,
    pub deopt_idx: usize,
}

impl TupleContainsTemplate {
    /// Create a new tuple contains template.
    #[inline]
    pub fn new(tuple_reg: u8, value_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            tuple_reg,
            value_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleContainsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load tuple value from frame
        let tuple_slot = ctx.frame.register_slot(self.tuple_reg as u16);
        ctx.asm.mov_rm(acc, &tuple_slot);

        // Type-check: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt to interpreter for linear scan
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80
    }
}

// =============================================================================
// Tuple Concat Template
// =============================================================================

/// Template for tuple concatenation (`tuple + tuple`).
///
/// # Code Generation Strategy
///
/// 1. Type-check LHS: OBJECT_TAG + TypeId::TUPLE
/// 2. Type-check RHS: OBJECT_TAG + TypeId::TUPLE
/// 3. Deopt to interpreter for allocation + copy
///
/// Both operands must be tuples. The double type guard eliminates both
/// type-dispatch checks on re-entry to the interpreter.
///
/// # Estimated Code Size
///
/// ~120 bytes: 2× tuple guard (~64), jmp (~8), loads (~14), overhead (~34)
pub struct TupleConcatTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl TupleConcatTemplate {
    /// Create a new tuple concat template.
    #[inline]
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleConcatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load LHS tuple
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);

        // Type-check LHS: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load RHS tuple
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(scratch1, &rhs_slot);

        // Type-check RHS: OBJECT_TAG + TypeId::TUPLE
        // Use acc as scratch since we've consumed the LHS pointer already
        emit_tuple_check_and_extract(ctx, scratch1, scratch1, acc, self.deopt_idx);

        // Deopt to interpreter for allocation
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Tuple Repeat Template
// =============================================================================

/// Template for tuple repetition (`tuple * n`).
///
/// # Code Generation Strategy
///
/// 1. Type-check tuple: OBJECT_TAG + TypeId::TUPLE
/// 2. Type-check count: INT_TAG, extract signed integer
/// 3. Deopt to interpreter for allocation + repeated copy
///
/// The double type guard eliminates both receiver-type and count-type
/// dispatch in the interpreter on re-entry.
///
/// # Estimated Code Size
///
/// ~120 bytes: tuple guard (~32), int guard (~20), jmp (~8),
/// loads (~14), overhead (~46)
pub struct TupleRepeatTemplate {
    pub dst_reg: u8,
    pub tuple_reg: u8,
    pub count_reg: u8,
    pub deopt_idx: usize,
}

impl TupleRepeatTemplate {
    /// Create a new tuple repeat template.
    #[inline]
    pub fn new(dst_reg: u8, tuple_reg: u8, count_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            tuple_reg,
            count_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleRepeatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load tuple value from frame
        let tuple_slot = ctx.frame.register_slot(self.tuple_reg as u16);
        ctx.asm.mov_rm(acc, &tuple_slot);

        // Type-check: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load count value from frame
        let count_slot = ctx.frame.register_slot(self.count_reg as u16);
        ctx.asm.mov_rm(scratch1, &count_slot);

        // Type-check count: INT_TAG
        emit_int_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt to interpreter for allocation
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}
