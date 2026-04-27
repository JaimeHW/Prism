//! Enhanced list operation templates for the Tier 1 JIT.
//!
//! Builds on the core list specialization in `list_specialize.rs` to add:
//! - **Pop last** (`list.pop()`) — fully inline: decrement len, load item
//! - **Length** (`len(list)`) — fully inline: load vec.len, box as int
//! - **Clear** (`list.clear()`) — fully inline: set vec.len = 0
//! - **Contains** (`x in list`) — guard-only: type check, deopt for scan
//! - **Insert** (`list.insert(i, v)`) — guard-only: type checks, deopt for memmove
//! - **Remove** (`list.remove(i)`) — guard-only: type checks, deopt for memmove
//!
//! All templates reuse helpers from `list_specialize.rs` via `pub(super)` visibility.

use super::list_specialize::{
    emit_bump_list_mutation_version, emit_int_check_and_extract_for_list,
    emit_list_check_and_extract, list_layout,
};
use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::registers::{MemOperand, Scale};

// =============================================================================
// ListPopLastTemplate
// =============================================================================

/// Template for inline pop from list end (`list.pop()` with no arguments).
///
/// # Code Generation Strategy
///
/// 1. Type-check list (OBJECT_TAG + TypeId::LIST)
/// 2. Load len from Vec header
/// 3. Guard: len > 0 (empty list → deopt to raise IndexError)
/// 4. Decrement len: `len -= 1`
/// 5. Write new len back to Vec header
/// 6. Load items pointer
/// 7. Load `items[new_len]` (the popped element)
/// 8. Store result to destination
///
/// # Performance
///
/// This is the optimal pop implementation: no memmove required since we pop
/// from the end. O(1) with only a bounds check and indexed load.
///
/// # Estimated Code Size
///
/// ~160 bytes: tag check (20), type_id check (16), len load (12),
/// empty guard (8), len decrement + store (16), items load (12),
/// indexed load (16), result store (8), overhead (52)
pub struct ListPopLastTemplate {
    /// Destination register slot for the popped value.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListPopLastTemplate {
    /// Create a new pop-last template.
    pub fn new(dst_reg: u8, list_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListPopLastTemplate {
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

        // Guard: len > 0 (empty → deopt to raise IndexError)
        ctx.asm.cmp_ri(scratch1, 0);
        ctx.asm.je(ctx.deopt_label(self.deopt_idx));

        // Decrement length: scratch1 = len - 1
        ctx.asm.dec(scratch1);

        // Write new length back
        ctx.asm.mov_mr(&len_mem, scratch1);
        emit_bump_list_mutation_version(ctx, acc, scratch2);

        // Load items pointer: scratch2 = list->vec.ptr
        let items_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_PTR_OFFSET,
        };
        ctx.asm.mov_rm(scratch2, &items_mem);

        // Load popped value: acc = items[new_len]
        let item_mem = MemOperand {
            base: Some(scratch2),
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
        160
    }
}

// =============================================================================
// ListLenTemplate
// =============================================================================

/// Template for inline list length (`len(list)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list (OBJECT_TAG + TypeId::LIST)
/// 2. Load len from Vec header
/// 3. NaN-box the length as INT_TAG
/// 4. Store result to destination
///
/// # Performance
///
/// Trivial operation: one memory load + int boxing. This eliminates the
/// overhead of `__len__` method dispatch entirely.
///
/// # Estimated Code Size
///
/// ~120 bytes: tag check (20), type_id check (16), len load (12),
/// int boxing (32), result store (8), overhead (32)
pub struct ListLenTemplate {
    /// Destination register slot for the length value.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListLenTemplate {
    /// Create a new list length template.
    pub fn new(dst_reg: u8, list_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListLenTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load list value from frame
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);

        // Type-check: OBJECT_TAG + TypeId::LIST
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = ListObject pointer

        // Load length: acc = list->vec.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(acc, &len_mem);

        // NaN-box the length as integer
        // result = (QNAN_BITS | INT_TAG) | (len & PAYLOAD_MASK)
        ctx.asm.mov_ri64(scratch1, value_tags::PAYLOAD_MASK as i64);
        ctx.asm.and_rr(acc, scratch1);
        let tag = (value_tags::QNAN_BITS | value_tags::INT_TAG) as i64;
        ctx.asm.mov_ri64(scratch1, tag);
        ctx.asm.or_rr(acc, scratch1);

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
// ListClearTemplate
// =============================================================================

/// Template for inline list clear (`list.clear()`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list (OBJECT_TAG + TypeId::LIST)
/// 2. Set vec.len = 0
///
/// This does NOT deallocate the backing array (same as Python's `list.clear()`).
/// The capacity is preserved for potential reuse, which is the optimal behavior
/// for lists that are cleared and refilled in loops.
///
/// # Performance
///
/// Single memory store after type guard. O(1). Note that unlike Python's
/// `list.clear()`, we don't need to drop individual elements since Values
/// are Copy types (NaN-boxed u64).
///
/// # Estimated Code Size
///
/// ~100 bytes: tag check (20), type_id check (16), zero store (16),
/// overhead (48)
pub struct ListClearTemplate {
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListClearTemplate {
    /// Create a new list clear template.
    pub fn new(list_reg: u8, deopt_idx: usize) -> Self {
        Self {
            list_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListClearTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load list value from frame
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);

        // Type-check: OBJECT_TAG + TypeId::LIST
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = ListObject pointer

        // Set vec.len = 0 when the list is non-empty.
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(scratch1, &len_mem);
        ctx.asm.cmp_ri(scratch1, 0);
        let done_label = ctx.asm.create_label();
        ctx.asm.je(done_label);
        ctx.asm.xor_rr(scratch1, scratch1); // scratch1 = 0
        ctx.asm.mov_mr(&len_mem, scratch1);
        emit_bump_list_mutation_version(ctx, acc, scratch1);
        ctx.asm.bind_label(done_label);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

// =============================================================================
// ListContainsTemplate
// =============================================================================

/// Template for list containment check (`value in list`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list operand (OBJECT_TAG + TypeId::LIST)
/// 2. Deopt to interpreter for actual linear scan
///
/// The inline type guard eliminates the overhead of generic `__contains__`
/// dispatch. The actual linear scan is deferred to the interpreter because
/// it requires a loop with per-element equality comparison, which would
/// generate too much code for Tier 1.
///
/// Tier 2 can inline the scan loop for homogeneous-type lists.
///
/// # Estimated Code Size
///
/// ~80 bytes: tag check (20), type_id check (16), deopt (8), overhead (36)
pub struct ListContainsTemplate {
    /// Destination register slot for the boolean result.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the value to search for.
    pub value_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListContainsTemplate {
    /// Create a new list contains template.
    pub fn new(dst_reg: u8, list_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListContainsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Type-check list operand
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt to interpreter for actual containment check
        // The type guard alone eliminates generic __contains__ dispatch overhead
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80
    }
}

// =============================================================================
// ListInsertTemplate
// =============================================================================

/// Template for list insertion (`list.insert(index, value)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list operand (OBJECT_TAG + TypeId::LIST)
/// 2. Type-check index operand (INT_TAG)
/// 3. Deopt to interpreter for actual insertion
///
/// Insert requires:
/// - Potential reallocation (if len == cap)
/// - memmove to shift elements right
/// - Index normalization (negative indices, clamping to [0, len])
///
/// These operations are too complex and code-size-expensive for Tier 1 inline
/// emission. The type guards still eliminate method dispatch overhead.
///
/// # Estimated Code Size
///
/// ~120 bytes: 2×tag checks (40), type_id check (16), deopt (8),
/// overhead (56)
pub struct ListInsertTemplate {
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the index.
    pub index_reg: u8,
    /// Register slot containing the value to insert.
    pub value_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListInsertTemplate {
    /// Create a new list insert template.
    pub fn new(list_reg: u8, index_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            list_reg,
            index_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListInsertTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Type-check list operand
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Type-check index operand
        let index_slot = ctx.frame.register_slot(self.index_reg as u16);
        ctx.asm.mov_rm(scratch1, &index_slot);
        emit_int_check_and_extract_for_list(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt to interpreter for memmove + potential realloc
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// ListRemoveTemplate
// =============================================================================

/// Template for list removal by index (`list.pop(index)` / `del list[index]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list operand (OBJECT_TAG + TypeId::LIST)
/// 2. Type-check index operand (INT_TAG)
/// 3. Deopt to interpreter for actual removal
///
/// Remove requires memmove to shift elements left after the removal point,
/// which is too complex for Tier 1 inline emission. The type guards eliminate
/// method dispatch overhead.
///
/// # Estimated Code Size
///
/// ~120 bytes: 2×tag checks (40), type_id check (16), deopt (8),
/// overhead (56)
pub struct ListRemoveTemplate {
    /// Destination register slot for the removed value.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the index.
    pub index_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListRemoveTemplate {
    /// Create a new list remove template.
    pub fn new(dst_reg: u8, list_reg: u8, index_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            index_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListRemoveTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Type-check list operand
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Type-check index operand
        let index_slot = ctx.frame.register_slot(self.index_reg as u16);
        ctx.asm.mov_rm(scratch1, &index_slot);
        emit_int_check_and_extract_for_list(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt to interpreter for memmove
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}
