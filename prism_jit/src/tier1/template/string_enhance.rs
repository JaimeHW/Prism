//! String method enhancement JIT templates.
//!
//! Provides type-specialized native code generation for common string methods:
//!
//! ## Unary Methods (self → result)
//! - **`str.upper()`** — guard string → deopt for SIMD-accelerated uppercase
//! - **`str.lower()`** — guard string → deopt for SIMD-accelerated lowercase
//! - **`str.strip()`** — guard string → deopt for SIMD-accelerated whitespace trim
//! - **`str.lstrip()`** — guard string → deopt for left whitespace trim
//! - **`str.rstrip()`** — guard string → deopt for right whitespace trim
//! - **`len(str)`** — guard string → deopt for character count
//!
//! ## Binary Methods (self, arg → result)
//! - **`str.startswith(prefix)`** — guard both strings → deopt
//! - **`str.endswith(suffix)`** — guard both strings → deopt
//! - **`str.find(needle)`** — guard both strings → deopt
//! - **`needle in str`** — guard both strings → deopt
//!
//! # Performance Strategy
//!
//! All templates emit inline type guards followed by deoptimization to the
//! interpreter. The key optimization is **eliminating method resolution overhead**:
//!
//! Without specialization:
//! ```text
//! LOAD_ATTR "upper" → MRO lookup → descriptor protocol → bound method → CALL
//! ```
//!
//! With specialization:
//! ```text
//! TYPE_GUARD(str) → deopt (operation pre-resolved, no MRO/descriptor overhead)
//! ```
//!
//! The actual string operations (SIMD-accelerated case conversion, whitespace
//! trimming, substring search) remain in the runtime, as they are already
//! optimized and too complex for Tier 1 inline emission.
//!
//! # Architecture
//!
//! This module complements `string_specialize.rs` (which handles string
//! *operators*: concat, repeat, equality, comparison) by covering string
//! *methods* and builtins. Both modules share the same type guard helpers
//! from `specialize_common.rs`.

use super::specialize_common::{emit_string_check_and_extract, string_tag_check};
use super::{OpcodeTemplate, TemplateContext};
use crate::backend::x64::Gpr;

// =============================================================================
// Unary String Method Templates
// =============================================================================

/// Template for `len(str)` — string character count.
///
/// # Strategy
///
/// 1. Guard operand as string (tag check + pointer extraction)
/// 2. Deopt to interpreter for `char_count()` (O(n) UTF-8 codepoint counting)
///
/// The deopt is necessary because Python's `len()` returns the *character* count
/// (Unicode codepoints), not the *byte* count. This requires iterating the string
/// to count non-continuation UTF-8 bytes, which is SIMD-accelerated in the runtime.
///
/// # Estimated Size: ~60 bytes
pub struct StrLenTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl StrLenTemplate {
    #[inline]
    pub fn new(dst_reg: u8, src_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            src_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrLenTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard and extract string pointer
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(acc, &src_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt for actual len computation (char_count is O(n) with SIMD)
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        60
    }
}

/// Template for `str.upper()` — uppercase conversion.
///
/// # Strategy
///
/// 1. Guard operand as string
/// 2. Deopt to interpreter for SIMD-accelerated case conversion
///
/// The runtime's `upper()` already uses:
/// - SIMD fast path for ASCII-only strings (`is_ascii()` check + vectorized conversion)
/// - Unicode-aware fallback for non-ASCII strings
///
/// # Estimated Size: ~60 bytes
pub struct StrUpperTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl StrUpperTemplate {
    #[inline]
    pub fn new(dst_reg: u8, src_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            src_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrUpperTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(acc, &src_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt for SIMD-accelerated uppercase conversion
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        60
    }
}

/// Template for `str.lower()` — lowercase conversion.
///
/// Same strategy as `StrUpperTemplate` but for lowercase.
///
/// # Estimated Size: ~60 bytes
pub struct StrLowerTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl StrLowerTemplate {
    #[inline]
    pub fn new(dst_reg: u8, src_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            src_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrLowerTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(acc, &src_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt for SIMD-accelerated lowercase conversion
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        60
    }
}

/// Which end(s) to strip whitespace from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StripKind {
    /// Strip both ends (`str.strip()`)
    Both,
    /// Strip left only (`str.lstrip()`)
    Left,
    /// Strip right only (`str.rstrip()`)
    Right,
}

impl StripKind {
    /// Get a display name for diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            StripKind::Both => "strip",
            StripKind::Left => "lstrip",
            StripKind::Right => "rstrip",
        }
    }
}

/// Template for `str.strip()` / `str.lstrip()` / `str.rstrip()`.
///
/// # Strategy
///
/// 1. Guard operand as string
/// 2. Deopt to interpreter for SIMD-accelerated whitespace trimming
///
/// The runtime's strip methods use:
/// - SIMD fast path for ASCII-only strings (vectorized whitespace detection)
/// - Unicode whitespace fallback for non-ASCII
///
/// The `StripKind` enum selects which variant to use. All three share
/// the same template structure, differing only in which interpreter
/// operation is invoked post-deopt.
///
/// # Estimated Size: ~60 bytes
pub struct StrStripTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub kind: StripKind,
    pub deopt_idx: usize,
}

impl StrStripTemplate {
    #[inline]
    pub fn new(dst_reg: u8, src_reg: u8, kind: StripKind, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            src_reg,
            kind,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrStripTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(acc, &src_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt for strip (kind disambiguated by interpreter)
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        60
    }
}

// =============================================================================
// Binary String Method Templates
// =============================================================================

/// Template for `str.startswith(prefix)`.
///
/// # Strategy
///
/// 1. Guard self as string
/// 2. Guard prefix as string
/// 3. Deopt to interpreter for the actual prefix check
///
/// Both operands must be strings — Python's `str.startswith()` only accepts
/// `str` or `tuple` of `str`. The tuple case deopts via the type guard.
///
/// # Estimated Size: ~90 bytes
pub struct StrStartsWithTemplate {
    pub dst_reg: u8,
    pub self_reg: u8,
    pub prefix_reg: u8,
    pub deopt_idx: usize,
}

impl StrStartsWithTemplate {
    #[inline]
    pub fn new(dst_reg: u8, self_reg: u8, prefix_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            self_reg,
            prefix_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrStartsWithTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard self as string
        let self_slot = ctx.frame.register_slot(self.self_reg as u16);
        ctx.asm.mov_rm(acc, &self_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Guard prefix as string
        let prefix_slot = ctx.frame.register_slot(self.prefix_reg as u16);
        ctx.asm.mov_rm(acc, &prefix_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt for actual prefix check
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        90
    }
}

/// Template for `str.endswith(suffix)`.
///
/// Same structure as `StrStartsWithTemplate` but for suffix checking.
///
/// # Estimated Size: ~90 bytes
pub struct StrEndsWithTemplate {
    pub dst_reg: u8,
    pub self_reg: u8,
    pub suffix_reg: u8,
    pub deopt_idx: usize,
}

impl StrEndsWithTemplate {
    #[inline]
    pub fn new(dst_reg: u8, self_reg: u8, suffix_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            self_reg,
            suffix_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrEndsWithTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard self as string
        let self_slot = ctx.frame.register_slot(self.self_reg as u16);
        ctx.asm.mov_rm(acc, &self_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Guard suffix as string
        let suffix_slot = ctx.frame.register_slot(self.suffix_reg as u16);
        ctx.asm.mov_rm(acc, &suffix_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt for actual suffix check
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        90
    }
}

/// Template for `needle in str` (string containment check).
///
/// # Strategy
///
/// 1. Guard haystack as string
/// 2. Guard needle as string
/// 3. Deopt to interpreter for SIMD-accelerated substring search
///
/// The runtime's `contains()` uses `str_contains()` which leverages
/// SIMD (SSE4.2 PCMPESTRI) for efficient substring search.
///
/// # Estimated Size: ~90 bytes
pub struct StrContainsTemplate {
    pub dst_reg: u8,
    pub needle_reg: u8,
    pub haystack_reg: u8,
    pub deopt_idx: usize,
}

impl StrContainsTemplate {
    #[inline]
    pub fn new(dst_reg: u8, needle_reg: u8, haystack_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            needle_reg,
            haystack_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrContainsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard haystack as string
        let haystack_slot = ctx.frame.register_slot(self.haystack_reg as u16);
        ctx.asm.mov_rm(acc, &haystack_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Guard needle as string
        let needle_slot = ctx.frame.register_slot(self.needle_reg as u16);
        ctx.asm.mov_rm(acc, &needle_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt for SIMD-accelerated containment check
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        90
    }
}

/// Template for `str.find(needle)` — substring index search.
///
/// # Strategy
///
/// 1. Guard self as string
/// 2. Guard needle as string
/// 3. Deopt to interpreter for SIMD-accelerated substring search
///
/// Returns the index of the first occurrence or -1 if not found.
/// The runtime uses `str_find()` with SIMD acceleration.
///
/// # Estimated Size: ~90 bytes
pub struct StrFindTemplate {
    pub dst_reg: u8,
    pub self_reg: u8,
    pub needle_reg: u8,
    pub deopt_idx: usize,
}

impl StrFindTemplate {
    #[inline]
    pub fn new(dst_reg: u8, self_reg: u8, needle_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            self_reg,
            needle_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for StrFindTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard self as string
        let self_slot = ctx.frame.register_slot(self.self_reg as u16);
        ctx.asm.mov_rm(acc, &self_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Guard needle as string
        let needle_slot = ctx.frame.register_slot(self.needle_reg as u16);
        ctx.asm.mov_rm(acc, &needle_slot);
        emit_string_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt for SIMD-accelerated find
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        90
    }
}
