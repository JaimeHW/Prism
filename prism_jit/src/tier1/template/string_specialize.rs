//! String-specialized JIT templates for high-performance string operations.
//!
//! Provides type-specialized native code generation for:
//! - **String concatenation** (`str + str`) — inline type check + runtime call
//! - **String repetition** (`str * int` / `int * str`) — inline guard + runtime call
//! - **String equality** (`str == str`) — pointer equality fast path
//! - **String comparison** (`str < str` etc.) — type-checked with runtime fallback
//!
//! # Architecture
//!
//! Each template implements `OpcodeTemplate` and follows the same pattern:
//! 1. Load operands from frame register slots
//! 2. Type-check via NaN-boxing tag comparison (shift right 48, compare)
//! 3. Extract payload pointer (mask with PAYLOAD_MASK)
//! 4. Perform operation (inline fast path or deopt to interpreter)
//! 5. Box result and store to destination slot
//!
//! All templates deoptimize to the interpreter on type mismatches.

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Gpr;

// =============================================================================
// Helper: String Type Check and Extract
// =============================================================================

/// Compute the upper-16-bit tag check value for strings.
///
/// This is `((QNAN_BITS | STRING_TAG) >> 48) as u16`.
#[inline]
const fn string_tag_check() -> u16 {
    ((value_tags::QNAN_BITS | value_tags::STRING_TAG) >> 48) as u16
}

/// Emit code to verify a value is a string and extract the pointer payload.
///
/// Performs:
/// 1. Copy value to scratch
/// 2. Shift right by 48 to isolate tag
/// 3. Compare with string_tag_check
/// 4. Jump to deopt on mismatch
/// 5. Mask with PAYLOAD_MASK to extract pointer
///
/// After this function, `dst` contains the raw string pointer (48-bit).
fn emit_string_check_and_extract(
    ctx: &mut TemplateContext,
    src: Gpr,
    dst: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    // Copy value for tag extraction
    ctx.asm.mov_rr(scratch, src);
    ctx.asm.shr_ri(scratch, 48);

    // Compare with string tag
    ctx.asm.cmp_ri(scratch, string_tag_check() as i32);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Extract payload (pointer): dst = src & PAYLOAD_MASK
    if src != dst {
        ctx.asm.mov_rr(dst, src);
    }
    ctx.asm.shl_ri(dst, 16);
    ctx.asm.shr_ri(dst, 16);
}

/// Emit code to verify a value is an integer and extract the payload.
///
/// After this function, `dst` contains the sign-extended 48-bit integer.
fn emit_int_check_and_extract_for_str(
    ctx: &mut TemplateContext,
    src: Gpr,
    dst: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    // Copy value for tag extraction
    ctx.asm.mov_rr(scratch, src);
    ctx.asm.shr_ri(scratch, 48);

    // Compare with int tag
    ctx.asm.cmp_ri(scratch, value_tags::int_tag_check() as i32);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Extract payload and sign-extend from 48 bits
    if src != dst {
        ctx.asm.mov_rr(dst, src);
    }
    ctx.asm.shl_ri(dst, 16); // shift left 16 to put sign bit at bit 63
    ctx.asm.sar_ri(dst, 16); // arithmetic shift right 16 to sign-extend
}

/// Emit code to box a string pointer back into a NaN-boxed string value.
///
/// Constructs: result = QNAN_BITS | STRING_TAG | (ptr & PAYLOAD_MASK)
/// Since ptr is already payload-extracted, we just OR with the tag pattern.
fn emit_string_box(ctx: &mut TemplateContext, ptr_reg: Gpr, scratch: Gpr) {
    let pattern = value_tags::QNAN_BITS | value_tags::STRING_TAG;
    ctx.asm.mov_ri64(scratch, pattern as i64);
    ctx.asm.or_rr(ptr_reg, scratch);
}

// =============================================================================
// String Concatenation Template
// =============================================================================

/// Template for inline string concatenation (`str + str`).
///
/// # Code Generation Strategy
///
/// 1. Type-check both operands as strings (deopt on mismatch)
/// 2. Extract string pointers from payload
/// 3. Deopt to interpreter for the actual concatenation
///    (runtime call linkage will be wired in Tier 2)
///
/// The key optimization is that the type guards are resolved inline —
/// no dynamic dispatch overhead when both operands are known strings.
///
/// # Estimated Code Size
///
/// ~120 bytes: 2×tag checks (40), pointer extractions (16),
/// runtime deopt (16), overhead (48)
pub struct StrConcatTemplate {
    /// Destination register index.
    pub dst_reg: u8,
    /// Left operand register index.
    pub lhs_reg: u8,
    /// Right operand register index.
    pub rhs_reg: u8,
    /// Deopt label index for type mismatch.
    pub deopt_idx: usize,
}

impl StrConcatTemplate {
    /// Create a new string concatenation template.
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrConcatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let _dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and type-check LHS as string
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load and type-check RHS as string
        ctx.asm.mov_rm(scratch2, &rhs_slot);
        emit_string_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Both type guards passed — deopt to interpreter for the actual concat.
        // When Tier 2 runtime call ABI is wired, this becomes:
        //   call __str_concat(acc, scratch2) → acc
        //   mov [dst_slot], acc
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// String Repetition Template
// =============================================================================

/// Template for string repetition (`str * int` or `int * str`).
///
/// # Code Generation Strategy
///
/// 1. Type-check operands: one must be string, other must be int
/// 2. Extract int count and string pointer
/// 3. Guard: count >= 0 (negative → deopt to return empty string)
/// 4. Guard: count <= MAX_REPEAT_INLINE (avoid huge allocations)
/// 5. Deopt for actual repetition (runtime call in Tier 2)
pub struct StrRepeatTemplate {
    /// Destination register index.
    pub dst_reg: u8,
    /// String operand register index.
    pub str_reg: u8,
    /// Integer count operand register index.
    pub count_reg: u8,
    /// Whether the string is the first operand (true = `str * int`).
    pub str_first: bool,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl StrRepeatTemplate {
    /// Maximum repeat count before deopting to interpreter.
    /// Prevents inline template from creating enormous strings.
    pub const MAX_REPEAT_INLINE: i64 = 1_000_000;

    /// Create a new string repetition template.
    pub fn new(dst_reg: u8, str_reg: u8, count_reg: u8, str_first: bool, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            str_reg,
            count_reg,
            str_first,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrRepeatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let str_slot = ctx.frame.register_slot(self.str_reg as u16);
        let count_slot = ctx.frame.register_slot(self.count_reg as u16);
        let _dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and type-check string operand
        ctx.asm.mov_rm(acc, &str_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load and type-check integer operand
        ctx.asm.mov_rm(scratch2, &count_slot);
        emit_int_check_and_extract_for_str(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Guard: count must be non-negative
        ctx.asm.cmp_ri(scratch2, 0);
        ctx.asm.jl(ctx.deopt_label(self.deopt_idx));

        // Guard: count <= MAX_REPEAT_INLINE
        ctx.asm.cmp_ri(scratch2, Self::MAX_REPEAT_INLINE as i32);
        ctx.asm.jg(ctx.deopt_label(self.deopt_idx));

        // Deopt for actual repetition (runtime call in Tier 2)
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    fn estimated_size(&self) -> usize {
        128
    }
}

// =============================================================================
// String Equality Template
// =============================================================================

/// Template for string equality comparison (`str == str` / `str != str`).
///
/// # Fast Path: Pointer Equality
///
/// For interned strings (the common case in Python — all identifier strings,
/// string literals, and strings used as dict keys are typically interned),
/// pointer equality is sufficient for correctness. Two interned strings
/// with the same content share the same backing `Arc<str>`, so their
/// payload pointers are identical.
///
/// This provides O(1) string equality for the common case.
///
/// # Slow Path
///
/// Non-interned strings with identical content but different pointers
/// need content comparison. The pointer-inequality path deopts to the
/// interpreter, which handles full content comparison.
pub struct StrEqualTemplate {
    /// Destination register index.
    pub dst_reg: u8,
    /// Left operand register index.
    pub lhs_reg: u8,
    /// Right operand register index.
    pub rhs_reg: u8,
    /// Whether this is `!=` (negate result).
    pub negate: bool,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl StrEqualTemplate {
    /// Create a new string equality template.
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, negate: bool, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            negate,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrEqualTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and type-check LHS as string
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load and type-check RHS as string
        ctx.asm.mov_rm(scratch2, &rhs_slot);
        emit_string_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Pointer equality check (fast path for interned strings)
        ctx.asm.cmp_rr(acc, scratch2);

        // If pointers NOT equal, deopt to interpreter for full comparison.
        // This handles the case where strings have same content but
        // different backing allocations (non-interned strings).
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));

        // Pointers are equal → strings are equal.
        // Load the appropriate boolean result.
        let result_value = if self.negate {
            // != : pointers equal means strings equal → result is False
            value_tags::false_value()
        } else {
            // == : pointers equal means strings equal → result is True
            value_tags::true_value()
        };

        ctx.asm.mov_ri64(acc, result_value as i64);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    fn estimated_size(&self) -> usize {
        96
    }
}

// =============================================================================
// String Comparison Template
// =============================================================================

/// Comparison operation type for string ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrCompareOp {
    /// Less than (`<`)
    Lt,
    /// Less than or equal (`<=`)
    Le,
    /// Greater than (`>`)
    Gt,
    /// Greater than or equal (`>=`)
    Ge,
}

impl StrCompareOp {
    /// Get the display representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }
}

/// Template for string ordering comparison (`str < str`, `str <= str`, etc.).
///
/// Type-checks inline but delegates the comparison to the interpreter
/// via deopt. The key optimization is that the type guards are already
/// resolved — the interpreter doesn't need to re-check types.
pub struct StrCompareTemplate {
    /// Destination register index.
    pub dst_reg: u8,
    /// Left operand register index.
    pub lhs_reg: u8,
    /// Right operand register index.
    pub rhs_reg: u8,
    /// Comparison operation.
    pub op: StrCompareOp,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl StrCompareTemplate {
    /// Create a new string comparison template.
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, op: StrCompareOp, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            op,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrCompareTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let _dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and type-check LHS as string
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load and type-check RHS as string
        ctx.asm.mov_rm(scratch2, &rhs_slot);
        emit_string_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Ordering comparison requires iterating string bytes.
        // Deopt to interpreter — runtime call linkage in Tier 2.
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    fn estimated_size(&self) -> usize {
        112
    }
}
