//! List-specialized JIT templates for high-performance list operations.
//!
//! Provides type-specialized native code generation for:
//! - **List indexing** (`list[i]`) — inline bounds check + direct load
//! - **List store** (`list[i] = v`) — inline bounds check + direct store
//! - **List append** (`list.append(v)`) — fast path when capacity suffices
//! - **List concatenation** (`list + list`) — type-checked, deopt for work
//! - **List repetition** (`list * int`) — type-checked, deopt for work
//!
//! # Memory Model
//!
//! `ListObject` is `#[repr(C)]`:
//! ```text
//! Offset  Size  Field
//! ──────  ────  ─────────────────────
//!   0      4    ObjectHeader.type_id  (TypeId, u32)
//!   4      4    ObjectHeader.gc_flags (AtomicU32)
//!   8      8    ObjectHeader.hash     (u64)
//!  16      8    Vec<Value>.ptr        (pointer to items)
//!  24      8    Vec<Value>.len        (usize)
//!  32      8    Vec<Value>.cap        (usize)
//!  40      8    mutation_version      (u64)
//! ```
//!
//! NaN-boxing: Lists are object-tagged via `OBJECT_TAG`. The payload is a
//! 48-bit pointer to a heap-allocated `ListObject`.
//!
//! # Performance Strategy
//!
//! - Inline type guard: extract OBJECT_TAG, deref pointer, check `type_id == LIST`
//! - Inline bounds check for indexing/store: handles negative indices
//! - Fast-path append: check `len < cap`, store at `items[len]`, increment `len`
//! - Deopt to interpreter for concat/repeat (Tier 2 will inline these)

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Gpr;
use crate::backend::x64::registers::{MemOperand, Scale};

// =============================================================================
// Layout Constants
// =============================================================================

/// `ListObject` field offsets, derived from `#[repr(C)]` layout.
///
/// ```text
/// ObjectHeader: 16 bytes (type_id u32 + gc_flags u32 + hash u64)
/// Vec<Value>: 24 bytes (ptr usize + len usize + cap usize)
/// mutation_version: 8 bytes
/// ```
pub(super) mod list_layout {
    /// Offset of `type_id` (u32) within ObjectHeader.
    pub const TYPE_ID_OFFSET: i32 = 0;

    /// Offset of Vec<Value>.ptr within ListObject.
    pub const ITEMS_PTR_OFFSET: i32 = 16;

    /// Offset of Vec<Value>.len within ListObject.
    pub const ITEMS_LEN_OFFSET: i32 = 24;

    /// Offset of Vec<Value>.cap within ListObject.
    pub const ITEMS_CAP_OFFSET: i32 = 32;

    /// Offset of the structural mutation version within ListObject.
    pub const MUTATION_VERSION_OFFSET: i32 = 40;

    /// Size of a single Value (u64 = 8 bytes).
    pub const VALUE_SIZE: i32 = 8;

    /// TypeId::LIST raw value.
    pub const LIST_TYPE_ID: u32 = 6;
}

#[inline]
pub(super) fn emit_bump_list_mutation_version(
    ctx: &mut TemplateContext,
    list_ptr: Gpr,
    scratch: Gpr,
) {
    let version_mem = MemOperand {
        base: Some(list_ptr),
        index: None,
        scale: Scale::X1,
        disp: list_layout::MUTATION_VERSION_OFFSET,
    };
    ctx.asm.mov_rm(scratch, &version_mem);
    ctx.asm.inc(scratch);
    ctx.asm.mov_mr(&version_mem, scratch);
}

// =============================================================================
// Helper: Object Tag Check
// =============================================================================

/// Compute the upper-16-bit tag check value for object pointers.
///
/// This is `((QNAN_BITS | OBJECT_TAG) >> 48) as u16`.
#[inline]
pub(super) const fn object_tag_check() -> u16 {
    ((value_tags::QNAN_BITS | value_tags::OBJECT_TAG) >> 48) as u16
}

/// Emit code to verify a value is an object and extract the pointer payload.
///
/// Performs:
/// 1. Copy value to scratch
/// 2. Shift right by 48 to isolate tag
/// 3. Compare with object_tag_check
/// 4. Jump to deopt on mismatch
/// 5. Mask out tag bits to extract payload pointer
///
/// After this function, `dst` contains the raw object pointer (48-bit).
pub(super) fn emit_object_check_and_extract(
    ctx: &mut TemplateContext,
    src: Gpr,
    dst: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    // Copy value for tag extraction
    ctx.asm.mov_rr(scratch, src);
    ctx.asm.shr_ri(scratch, 48);

    // Compare with object tag
    ctx.asm.cmp_ri(scratch, object_tag_check() as i32);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Extract payload (pointer): dst = src & PAYLOAD_MASK
    if src != dst {
        ctx.asm.mov_rr(dst, src);
    }
    ctx.asm.shl_ri(dst, 16);
    ctx.asm.shr_ri(dst, 16);
}

/// Emit code to verify the object pointer points to a ListObject.
///
/// Loads the `type_id` field from `ObjectHeader` at the given pointer
/// and compares it with `TypeId::LIST` (6). Deopts on mismatch.
///
/// Requires `obj_ptr` to be a valid object pointer (already extracted from
/// NaN-box via `emit_object_check_and_extract`).
pub(super) fn emit_list_type_guard(
    ctx: &mut TemplateContext,
    obj_ptr: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    // Load type_id (u32) from ObjectHeader at offset 0
    let type_id_mem = MemOperand {
        base: Some(obj_ptr),
        index: None,
        scale: Scale::X1,
        disp: list_layout::TYPE_ID_OFFSET,
    };
    ctx.asm.mov_rm32(scratch, &type_id_mem);

    // Compare with LIST type ID
    ctx.asm.cmp_ri(scratch, list_layout::LIST_TYPE_ID as i32);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));
}

/// Emit code to verify a value is a list and extract the object pointer.
///
/// This is the combined operation:
/// 1. Check OBJECT_TAG in NaN-box
/// 2. Extract 48-bit pointer payload
/// 3. Dereference and check TypeId == LIST
///
/// After this, `dst` points to the ListObject on the heap.
pub(super) fn emit_list_check_and_extract(
    ctx: &mut TemplateContext,
    src: Gpr,
    dst: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    emit_object_check_and_extract(ctx, src, dst, scratch, deopt_idx);
    emit_list_type_guard(ctx, dst, scratch, deopt_idx);
}

/// Emit code to verify a value is an integer and extract the payload.
///
/// After this function, `dst` contains the sign-extended 48-bit integer.
pub(super) fn emit_int_check_and_extract_for_list(
    ctx: &mut TemplateContext,
    src: Gpr,
    dst: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    ctx.asm.mov_rr(scratch, src);
    ctx.asm.shr_ri(scratch, 48);

    ctx.asm.cmp_ri(scratch, value_tags::int_tag_check() as i32);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Extract payload with sign extension (48-bit signed)
    if src != dst {
        ctx.asm.mov_rr(dst, src);
    }
    ctx.asm.shl_ri(dst, 16);
    ctx.asm.sar_ri(dst, 16);
}

/// Emit code to box an object pointer back into a NaN-boxed value.
///
/// Constructs: result = QNAN_BITS | OBJECT_TAG | (ptr & PAYLOAD_MASK)
#[allow(dead_code)]
pub(super) fn emit_object_box(ctx: &mut TemplateContext, ptr_reg: Gpr, scratch: Gpr) {
    let tag = (value_tags::QNAN_BITS | value_tags::OBJECT_TAG) as i64;
    ctx.asm.mov_ri64(scratch, tag);
    ctx.asm.or_rr(ptr_reg, scratch);
}

// =============================================================================
// List Index Template
// =============================================================================

/// Template for inline list indexing (`list[i]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list operand (OBJECT_TAG + TypeId::LIST)
/// 2. Type-check index operand (INT_TAG)
/// 3. Extract list pointer and index value
/// 4. Load list length from `Vec.len`
/// 5. Handle negative indices: `if index < 0 { index += len }`
/// 6. Bounds check: `0 <= index < len`
/// 7. Load items pointer from `Vec.ptr`
/// 8. Load `items[index]` (8 bytes per Value)
/// 9. Store result to destination
///
/// # Performance
///
/// Hot path (positive in-bounds index) requires:
/// - 2 tag checks (~20 bytes each)
/// - 1 length load + comparison
/// - 1 indexed memory load
/// Total: ~120 bytes for the fast path
///
/// # Estimated Code Size
///
/// ~200 bytes: 2×tag checks (40), type_id check (16), index normalization (32),
/// bounds check (16), indexed load (24), store (16), overhead (56)
pub struct ListIndexTemplate {
    /// Destination register slot for the loaded value.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the index.
    pub index_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListIndexTemplate {
    /// Create a new list index template.
    pub fn new(dst_reg: u8, list_reg: u8, index_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            index_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListIndexTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load list value from frame
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);

        // Type-check: OBJECT_TAG + TypeId::LIST
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = raw ListObject pointer

        // Load index value from frame
        let index_slot = ctx.frame.register_slot(self.index_reg as u16);
        ctx.asm.mov_rm(scratch1, &index_slot);

        // Type-check index: INT_TAG
        emit_int_check_and_extract_for_list(ctx, scratch1, scratch1, scratch2, self.deopt_idx);
        // scratch1 = sign-extended index value

        // Load length: scratch2 = list->vec.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(scratch2, &len_mem);

        // Handle negative index: if index < 0, index += len
        ctx.asm.cmp_ri(scratch1, 0);
        let positive_label = ctx.asm.create_label();
        ctx.asm.jge(positive_label);
        // Negative path: scratch1 += scratch2 (index += len)
        ctx.asm.add_rr(scratch1, scratch2);
        ctx.asm.bind_label(positive_label);

        // Bounds check: 0 <= index < len
        // After normalization, index must be >= 0 (checked by unsigned cmp below)
        // If still negative after adding len, the unsigned comparison catches it
        ctx.asm.cmp_rr(scratch1, scratch2);
        ctx.asm.jae(ctx.deopt_label(self.deopt_idx)); // unsigned >= means out of bounds

        // Load items pointer: acc preserved as list ptr
        // We need: items_ptr = *(list_ptr + ITEMS_PTR_OFFSET)
        let items_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_PTR_OFFSET,
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
// List Store Template
// =============================================================================

/// Template for inline list store (`list[i] = value`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list (OBJECT_TAG + TypeId::LIST)
/// 2. Type-check index (INT_TAG)
/// 3. Handle negative indices
/// 4. Bounds check
/// 5. Load items pointer
/// 6. Store value at `items[index]`
///
/// # Estimated Code Size
///
/// ~200 bytes: similar to ListIndexTemplate
pub struct ListStoreTemplate {
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the index.
    pub index_reg: u8,
    /// Register slot containing the value to store.
    pub value_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListStoreTemplate {
    /// Create a new list store template.
    pub fn new(list_reg: u8, index_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            list_reg,
            index_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListStoreTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load list value from frame
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);

        // Type-check: OBJECT_TAG + TypeId::LIST
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load index value
        let index_slot = ctx.frame.register_slot(self.index_reg as u16);
        ctx.asm.mov_rm(scratch1, &index_slot);

        // Type-check index: INT_TAG
        emit_int_check_and_extract_for_list(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Load length
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(scratch2, &len_mem);

        // Handle negative index: if index < 0, index += len
        ctx.asm.cmp_ri(scratch1, 0);
        let positive_label = ctx.asm.create_label();
        ctx.asm.jge(positive_label);
        ctx.asm.add_rr(scratch1, scratch2);
        ctx.asm.bind_label(positive_label);

        // Bounds check: unsigned index < len
        ctx.asm.cmp_rr(scratch1, scratch2);
        ctx.asm.jae(ctx.deopt_label(self.deopt_idx));

        ctx.asm.push(acc);

        // Load items pointer
        let items_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_PTR_OFFSET,
        };
        ctx.asm.mov_rm(acc, &items_mem);

        // Load value to store from frame
        let value_slot = ctx.frame.register_slot(self.value_reg as u16);
        ctx.asm.mov_rm(scratch2, &value_slot);

        // Store value: items_ptr[index * 8] = value
        let item_mem = MemOperand {
            base: Some(acc),
            index: Some(scratch1),
            scale: Scale::X8,
            disp: 0,
        };
        ctx.asm.mov_mr(&item_mem, scratch2);

        ctx.asm.pop(acc);
        emit_bump_list_mutation_version(ctx, acc, scratch2);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        200
    }
}

// =============================================================================
// List Append Fast Template
// =============================================================================

/// Template for fast-path list append (`list.append(item)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list (OBJECT_TAG + TypeId::LIST)
/// 2. Load len and cap from Vec header
/// 3. Fast path: `len < cap` → store at `items[len]`, increment len
/// 4. Slow path: deopt to interpreter (triggers Vec reallocation)
///
/// # Performance
///
/// The fast path (capacity available) is:
/// - 1 tag check + type_id check
/// - 1 len/cap comparison
/// - 1 indexed store + len increment
///
/// This avoids the overhead of method dispatch, Vec::push bounds check,
/// and potential reallocation in the common case.
///
/// # Estimated Code Size
///
/// ~180 bytes: tag check (20), type_id check (16), len/cap load (24),
/// capacity check (12), store (24), len update (24), overhead (60)
pub struct ListAppendFastTemplate {
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the item to append.
    pub item_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListAppendFastTemplate {
    /// Create a new list append fast template.
    pub fn new(list_reg: u8, item_reg: u8, deopt_idx: usize) -> Self {
        Self {
            list_reg,
            item_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListAppendFastTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load list value from frame
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);

        // Type-check: OBJECT_TAG + TypeId::LIST
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = ListObject pointer

        // Load length: scratch1 = list->vec.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(scratch1, &len_mem);

        // Load capacity: scratch2 = list->vec.cap
        let cap_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_CAP_OFFSET,
        };
        ctx.asm.mov_rm(scratch2, &cap_mem);

        // Fast path check: len < cap
        ctx.asm.cmp_rr(scratch1, scratch2);
        ctx.asm.jae(ctx.deopt_label(self.deopt_idx)); // len >= cap → need realloc

        // Load items pointer: scratch2 = list->vec.ptr
        let items_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_PTR_OFFSET,
        };
        ctx.asm.mov_rm(scratch2, &items_mem);

        // Load item from frame into accumulator (reuse acc temporarily)
        // Save list pointer first — we need it to update len
        // Strategy: use scratch2 (items ptr), load item into acc-saved via push
        // Actually, let's be smarter: acc still holds list ptr, we load item
        // value from frame into a known location.
        //
        // Register allocation:
        //   acc     = list ptr (need for len update)
        //   scratch1 = len (need as index for store)
        //   scratch2 = items ptr (need for store)
        //
        // We need one more register for the item value. Push/pop acc to free it:
        ctx.asm.push(acc); // save list ptr on stack

        // Load item value from frame
        let item_slot = ctx.frame.register_slot(self.item_reg as u16);
        ctx.asm.mov_rm(acc, &item_slot);

        // Store item at items_ptr[len * 8]
        let item_mem = MemOperand {
            base: Some(scratch2),
            index: Some(scratch1),
            scale: Scale::X8,
            disp: 0,
        };
        ctx.asm.mov_mr(&item_mem, acc);

        // Restore list pointer
        ctx.asm.pop(acc);

        // Increment length: list->vec.len += 1
        ctx.asm.inc(scratch1);
        ctx.asm.mov_mr(&len_mem, scratch1);
        emit_bump_list_mutation_version(ctx, acc, scratch2);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        180
    }
}

// =============================================================================
// List Concat Template
// =============================================================================

/// Template for list concatenation (`list + list`).
///
/// # Code Generation Strategy
///
/// 1. Type-check both operands as lists (OBJECT_TAG + TypeId::LIST)
/// 2. Deopt to interpreter for actual concatenation (requires allocation)
///
/// In Tier 2, this will allocate a new list with combined capacity and memcpy
/// both item arrays. For Tier 1, the inline type guards still eliminate the
/// overhead of generic binary_add dispatch.
///
/// # Estimated Code Size
///
/// ~120 bytes: 2×(tag check + type_id check) + deopt
pub struct ListConcatTemplate {
    /// Destination register slot.
    pub dst_reg: u8,
    /// Register slot for left operand.
    pub lhs_reg: u8,
    /// Register slot for right operand.
    pub rhs_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListConcatTemplate {
    /// Create a new list concatenation template.
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListConcatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Type-check LHS as list
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Type-check RHS as list
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Actual concatenation requires allocation — deopt to interpreter
        // Tier 2 will inline the allocation + memcpy
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// List Repeat Template
// =============================================================================

/// Template for list repetition (`list * int` or `int * list`).
///
/// # Code Generation Strategy
///
/// 1. Type-check: one list + one integer
/// 2. Extract count, guard against negative and excessive values
/// 3. Deopt to interpreter for actual repetition
///
/// # Estimated Code Size
///
/// ~160 bytes: tag checks, type_id check, count guards, deopt
pub struct ListRepeatTemplate {
    /// Destination register slot.
    pub dst_reg: u8,
    /// Register slot for the list operand.
    pub list_reg: u8,
    /// Register slot for the count operand.
    pub count_reg: u8,
    /// Whether list is the first operand (true) or second (false).
    pub list_first: bool,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListRepeatTemplate {
    /// Maximum repeat count before deopting to interpreter.
    /// Prevents inline template from creating enormous lists.
    pub const MAX_REPEAT_INLINE: i64 = 1_000_000;

    /// Create a new list repeat template.
    pub fn new(
        dst_reg: u8,
        list_reg: u8,
        count_reg: u8,
        list_first: bool,
        deopt_idx: usize,
    ) -> Self {
        Self {
            dst_reg,
            list_reg,
            count_reg,
            list_first,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListRepeatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        if self.list_first {
            // list_reg is list, count_reg is int
            let list_slot = ctx.frame.register_slot(self.list_reg as u16);
            ctx.asm.mov_rm(acc, &list_slot);
            emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

            let count_slot = ctx.frame.register_slot(self.count_reg as u16);
            ctx.asm.mov_rm(scratch1, &count_slot);
            emit_int_check_and_extract_for_list(ctx, scratch1, scratch1, scratch2, self.deopt_idx);
        } else {
            // count_reg is int, list_reg is list (reversed: int * list)
            let count_slot = ctx.frame.register_slot(self.count_reg as u16);
            ctx.asm.mov_rm(acc, &count_slot);
            emit_int_check_and_extract_for_list(ctx, acc, scratch1, scratch2, self.deopt_idx);

            let list_slot = ctx.frame.register_slot(self.list_reg as u16);
            ctx.asm.mov_rm(acc, &list_slot);
            emit_list_check_and_extract(ctx, acc, acc, scratch2, self.deopt_idx);
        }

        // Guard: count must be <= MAX_REPEAT_INLINE
        // (negative count → Python returns empty list, handled by interpreter)
        ctx.asm.mov_ri64(scratch2, Self::MAX_REPEAT_INLINE);
        ctx.asm.cmp_rr(scratch1, scratch2);
        ctx.asm.jg(ctx.deopt_label(self.deopt_idx));

        // Guard: count must be >= 0 (negative → deopt to return empty list)
        ctx.asm.cmp_ri(scratch1, 0);
        ctx.asm.jl(ctx.deopt_label(self.deopt_idx));

        // Actual repetition requires allocation — deopt to interpreter
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        160
    }
}
