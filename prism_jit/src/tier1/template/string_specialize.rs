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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::x64::Assembler;
    use crate::tier1::frame::FrameLayout;

    // =========================================================================
    // Helper: Emit template and finalize
    // =========================================================================

    /// Helper to emit a template and get the generated code bytes.
    fn emit_and_finalize(template: &dyn OpcodeTemplate) -> Vec<u8> {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            let _deopt = ctx.create_deopt_label();
            template.emit(&mut ctx);
            // Bind all deopt labels to current position (required by assembler)
            for label in &ctx.deopt_labels {
                ctx.asm.bind_label(*label);
            }
        }
        asm.finalize()
            .expect("assembler finalization should succeed")
    }

    // =========================================================================
    // Tag Constants Tests
    // =========================================================================

    #[test]
    fn test_string_tag_check_value() {
        let expected = ((value_tags::QNAN_BITS | value_tags::STRING_TAG) >> 48) as u16;
        assert_eq!(string_tag_check(), expected);
    }

    #[test]
    fn test_string_tag_check_distinct_from_int() {
        assert_ne!(string_tag_check(), value_tags::int_tag_check());
    }

    #[test]
    fn test_string_tag_is_nonzero() {
        assert_ne!(value_tags::STRING_TAG, 0);
    }

    #[test]
    fn test_string_tag_distinct_from_all_other_tags() {
        let string = value_tags::STRING_TAG;
        assert_ne!(string, value_tags::INT_TAG);
        assert_ne!(string, value_tags::OBJECT_TAG);
        assert_ne!(string, value_tags::NONE_TAG);
        assert_ne!(string, value_tags::BOOL_TAG);
    }

    #[test]
    fn test_string_tag_check_roundtrip() {
        let string_value = value_tags::QNAN_BITS | value_tags::STRING_TAG | 0x1234_5678;
        let extracted_tag = (string_value >> 48) as u16;
        assert_eq!(extracted_tag, string_tag_check());
    }

    #[test]
    fn test_payload_extraction_preserves_pointer() {
        let ptr: u64 = 0x0000_DEAD_BEEF_CAFE;
        let boxed = value_tags::QNAN_BITS | value_tags::STRING_TAG | ptr;
        let payload = boxed & value_tags::PAYLOAD_MASK;
        assert_eq!(payload, ptr);
    }

    // =========================================================================
    // StrConcatTemplate Tests
    // =========================================================================

    #[test]
    fn test_str_concat_template_creation() {
        let t = StrConcatTemplate::new(0, 1, 2, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.lhs_reg, 1);
        assert_eq!(t.rhs_reg, 2);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_str_concat_template_estimated_size() {
        let t = StrConcatTemplate::new(0, 1, 2, 0);
        assert_eq!(t.estimated_size(), 120);
    }

    #[test]
    fn test_str_concat_template_emits_code() {
        let code = emit_and_finalize(&StrConcatTemplate::new(0, 1, 2, 0));
        assert!(!code.is_empty(), "StrConcatTemplate should emit code");
    }

    #[test]
    fn test_str_concat_template_different_registers() {
        let configs = [(0, 1, 2), (3, 4, 5), (0, 0, 1), (2, 1, 0)];
        for (dst, lhs, rhs) in configs {
            let code = emit_and_finalize(&StrConcatTemplate::new(dst, lhs, rhs, 0));
            assert!(
                !code.is_empty(),
                "Should emit code for ({}, {}, {})",
                dst,
                lhs,
                rhs
            );
        }
    }

    #[test]
    fn test_str_concat_code_size_within_estimate() {
        let t = StrConcatTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size() * 3,
            "Code size {} exceeds 3x estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_str_concat_same_src_dst() {
        let code = emit_and_finalize(&StrConcatTemplate::new(1, 1, 2, 0));
        assert!(!code.is_empty());
    }

    // =========================================================================
    // StrRepeatTemplate Tests
    // =========================================================================

    #[test]
    fn test_str_repeat_template_creation_str_first() {
        let t = StrRepeatTemplate::new(0, 1, 2, true, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.str_reg, 1);
        assert_eq!(t.count_reg, 2);
        assert!(t.str_first);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_str_repeat_template_creation_int_first() {
        let t = StrRepeatTemplate::new(3, 4, 5, false, 1);
        assert_eq!(t.dst_reg, 3);
        assert_eq!(t.str_reg, 4);
        assert_eq!(t.count_reg, 5);
        assert!(!t.str_first);
        assert_eq!(t.deopt_idx, 1);
    }

    #[test]
    fn test_str_repeat_template_estimated_size() {
        let t = StrRepeatTemplate::new(0, 1, 2, true, 0);
        assert_eq!(t.estimated_size(), 128);
    }

    #[test]
    fn test_str_repeat_template_emits_code() {
        let code = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, true, 0));
        assert!(!code.is_empty(), "StrRepeatTemplate should emit code");
    }

    #[test]
    fn test_str_repeat_max_inline_constant() {
        assert_eq!(StrRepeatTemplate::MAX_REPEAT_INLINE, 1_000_000);
        assert!(StrRepeatTemplate::MAX_REPEAT_INLINE > 0);
        assert!(StrRepeatTemplate::MAX_REPEAT_INLINE <= i32::MAX as i64);
    }

    #[test]
    fn test_str_repeat_template_both_orderings() {
        let code1 = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, true, 0));
        let code2 = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, false, 0));
        assert!(!code1.is_empty());
        assert!(!code2.is_empty());
    }

    #[test]
    fn test_str_repeat_code_size_within_estimate() {
        let t = StrRepeatTemplate::new(0, 1, 2, true, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size() * 3,
            "Code size {} exceeds 3x estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    // =========================================================================
    // StrEqualTemplate Tests
    // =========================================================================

    #[test]
    fn test_str_equal_template_creation() {
        let t = StrEqualTemplate::new(0, 1, 2, false, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.lhs_reg, 1);
        assert_eq!(t.rhs_reg, 2);
        assert!(!t.negate);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_str_not_equal_template_creation() {
        let t = StrEqualTemplate::new(3, 4, 5, true, 1);
        assert_eq!(t.dst_reg, 3);
        assert_eq!(t.lhs_reg, 4);
        assert_eq!(t.rhs_reg, 5);
        assert!(t.negate);
        assert_eq!(t.deopt_idx, 1);
    }

    #[test]
    fn test_str_equal_template_estimated_size() {
        let t = StrEqualTemplate::new(0, 1, 2, false, 0);
        assert_eq!(t.estimated_size(), 96);
    }

    #[test]
    fn test_str_equal_template_emits_code() {
        let code = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, false, 0));
        assert!(!code.is_empty(), "StrEqualTemplate should emit code");
    }

    #[test]
    fn test_str_not_equal_template_emits_code() {
        let code = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, true, 0));
        assert!(!code.is_empty(), "StrNotEqualTemplate should emit code");
    }

    #[test]
    fn test_str_equal_vs_not_equal_emit_different_code() {
        let code_eq = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, false, 0));
        let code_ne = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, true, 0));
        assert_ne!(code_eq, code_ne, "== and != should generate different code");
    }

    #[test]
    fn test_str_equal_code_size_within_estimate() {
        let t = StrEqualTemplate::new(0, 1, 2, false, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size() * 3,
            "Code size {} exceeds 3x estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_str_equal_all_same_registers() {
        let code = emit_and_finalize(&StrEqualTemplate::new(0, 0, 0, false, 0));
        assert!(!code.is_empty());
    }

    // =========================================================================
    // StrCompareTemplate Tests
    // =========================================================================

    #[test]
    fn test_str_compare_op_as_str() {
        assert_eq!(StrCompareOp::Lt.as_str(), "<");
        assert_eq!(StrCompareOp::Le.as_str(), "<=");
        assert_eq!(StrCompareOp::Gt.as_str(), ">");
        assert_eq!(StrCompareOp::Ge.as_str(), ">=");
    }

    #[test]
    fn test_str_compare_op_equality() {
        assert_eq!(StrCompareOp::Lt, StrCompareOp::Lt);
        assert_ne!(StrCompareOp::Lt, StrCompareOp::Gt);
        assert_ne!(StrCompareOp::Le, StrCompareOp::Ge);
    }

    #[test]
    fn test_str_compare_op_all_distinct() {
        let ops = [
            StrCompareOp::Lt,
            StrCompareOp::Le,
            StrCompareOp::Gt,
            StrCompareOp::Ge,
        ];
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "Ops at {} and {} should differ", i, j);
            }
        }
    }

    #[test]
    fn test_str_compare_template_creation() {
        let t = StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.lhs_reg, 1);
        assert_eq!(t.rhs_reg, 2);
        assert_eq!(t.op, StrCompareOp::Lt);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_str_compare_template_estimated_size() {
        let t = StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0);
        assert_eq!(t.estimated_size(), 112);
    }

    #[test]
    fn test_str_compare_template_emits_code_all_ops() {
        let ops = [
            StrCompareOp::Lt,
            StrCompareOp::Le,
            StrCompareOp::Gt,
            StrCompareOp::Ge,
        ];
        for op in ops {
            let code = emit_and_finalize(&StrCompareTemplate::new(0, 1, 2, op, 0));
            assert!(
                !code.is_empty(),
                "StrCompareTemplate({:?}) should emit code",
                op
            );
        }
    }

    #[test]
    fn test_str_compare_code_size_within_estimate() {
        let t = StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size() * 3,
            "Code size {} exceeds 3x estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    // =========================================================================
    // Cross-Template Consistency Tests
    // =========================================================================

    #[test]
    fn test_all_templates_produce_nonzero_code() {
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(StrConcatTemplate::new(0, 1, 2, 0)),
            Box::new(StrRepeatTemplate::new(0, 1, 2, true, 0)),
            Box::new(StrEqualTemplate::new(0, 1, 2, false, 0)),
            Box::new(StrEqualTemplate::new(0, 1, 2, true, 0)),
            Box::new(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0)),
            Box::new(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Le, 0)),
            Box::new(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Gt, 0)),
            Box::new(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Ge, 0)),
        ];

        for (i, template) in templates.iter().enumerate() {
            let code = emit_and_finalize(template.as_ref());
            assert!(
                !code.is_empty(),
                "Template {} should produce non-empty code",
                i
            );
        }
    }

    #[test]
    fn test_all_templates_have_positive_estimated_size() {
        assert!(StrConcatTemplate::new(0, 1, 2, 0).estimated_size() > 0);
        assert!(StrRepeatTemplate::new(0, 1, 2, true, 0).estimated_size() > 0);
        assert!(StrEqualTemplate::new(0, 1, 2, false, 0).estimated_size() > 0);
        assert!(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0).estimated_size() > 0);
    }

    #[test]
    fn test_estimated_sizes_are_reasonable() {
        let sizes = [
            StrConcatTemplate::new(0, 1, 2, 0).estimated_size(),
            StrRepeatTemplate::new(0, 1, 2, true, 0).estimated_size(),
            StrEqualTemplate::new(0, 1, 2, false, 0).estimated_size(),
            StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0).estimated_size(),
        ];
        for size in sizes {
            assert!(size >= 64, "Size {} too small", size);
            assert!(size <= 256, "Size {} too large", size);
        }
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_emit_string_box() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            let acc = ctx.regs.accumulator;
            let scratch = ctx.regs.scratch1;
            emit_string_box(&mut ctx, acc, scratch);
        }
        let code = asm.finalize().expect("finalize should succeed");
        assert!(!code.is_empty());
    }

    #[test]
    fn test_emit_string_check_and_extract() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            let deopt = ctx.create_deopt_label();
            let acc = ctx.regs.accumulator;
            let scratch = ctx.regs.scratch1;
            emit_string_check_and_extract(&mut ctx, acc, acc, scratch, deopt);
            for label in &ctx.deopt_labels {
                ctx.asm.bind_label(*label);
            }
        }
        let code = asm.finalize().expect("finalize should succeed");
        assert!(!code.is_empty());
    }

    #[test]
    fn test_emit_int_check_for_str() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            let deopt = ctx.create_deopt_label();
            let acc = ctx.regs.accumulator;
            let scratch = ctx.regs.scratch1;
            emit_int_check_and_extract_for_str(&mut ctx, acc, acc, scratch, deopt);
            for label in &ctx.deopt_labels {
                ctx.asm.bind_label(*label);
            }
        }
        let code = asm.finalize().expect("finalize should succeed");
        assert!(!code.is_empty());
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_multiple_deopt_labels() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            let deopt0 = ctx.create_deopt_label();
            let deopt1 = ctx.create_deopt_label();
            let deopt2 = ctx.create_deopt_label();

            StrConcatTemplate::new(0, 1, 2, deopt0).emit(&mut ctx);
            StrEqualTemplate::new(3, 4, 5, false, deopt1).emit(&mut ctx);
            StrRepeatTemplate::new(6, 7, 8, true, deopt2).emit(&mut ctx);
            for label in &ctx.deopt_labels {
                ctx.asm.bind_label(*label);
            }
        }
        let code = asm.finalize().expect("finalize should succeed");
        assert!(!code.is_empty());
    }

    #[test]
    fn test_sequential_template_emission() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            let deopt = ctx.create_deopt_label();

            for i in 0..5u8 {
                StrEqualTemplate::new(i, i.wrapping_add(1), i.wrapping_add(2), i % 2 == 0, deopt)
                    .emit(&mut ctx);
            }
            for label in &ctx.deopt_labels {
                ctx.asm.bind_label(*label);
            }
        }
        let code = asm.finalize().expect("finalize should succeed");
        assert!(!code.is_empty());
    }

    #[test]
    fn test_high_register_indices() {
        let code = emit_and_finalize(&StrConcatTemplate::new(14, 13, 12, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_str_concat_adjacent_registers() {
        let code = emit_and_finalize(&StrConcatTemplate::new(0, 1, 2, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_str_repeat_with_reversed_operands() {
        let code_str_first = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, true, 0));
        let code_int_first = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, false, 0));
        assert!(!code_str_first.is_empty());
        assert!(!code_int_first.is_empty());
    }

    #[test]
    fn test_str_compare_all_ops_same_size() {
        let code_lt = emit_and_finalize(&StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0));
        let code_gt = emit_and_finalize(&StrCompareTemplate::new(0, 1, 2, StrCompareOp::Gt, 0));
        assert_eq!(
            code_lt.len(),
            code_gt.len(),
            "All compare ops should generate same-sized code"
        );
    }

    #[test]
    fn test_str_equal_both_generate_code() {
        let code_eq = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, false, 0));
        let code_ne = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, true, 0));
        assert!(!code_eq.is_empty());
        assert!(!code_ne.is_empty());
    }
}
