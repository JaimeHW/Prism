//! Template JIT opcode templates - modular organization.
//!
//! Templates are organized by opcode category for maintainability:
//! - `values`: Load constants (None, True, False, integers, floats)
//! - `memory`: Load/store locals, globals, closures
//! - `arithmetic`: Integer, float, and generic arithmetic
//! - `comparison`: Comparison and identity operators
//! - `bitwise`: Bitwise and logical operations
//! - `control`: Jumps, branches, returns
//! - `objects`: Attribute and item access, iterators
//! - `calls`: Function calls and closures
//! - `containers`: List, tuple, dict, set construction
//!
//! Each template implements fast type-specialized code with deoptimization
//! guards for type mismatches.

pub mod arithmetic;
pub mod bitwise;
pub mod calls;
pub mod comparison;
pub mod containers;
pub mod control;
pub mod dict_specialize;
pub mod float_specialize;
pub mod guards;
pub mod ic_helpers;
pub mod int_specialize;
pub mod list_enhance;
pub mod list_specialize;
pub mod memory;
pub mod objects;
pub mod specialize_common;
pub mod string_enhance;
pub mod string_specialize;
pub mod tuple_specialize;
pub mod values;

use crate::backend::x64::{Assembler, Label};
use crate::tier1::frame::{FrameLayout, JitCallingConvention, RegisterAssignment};

// Re-export commonly used types
pub use arithmetic::*;
pub use bitwise::*;
pub use calls::*;
pub use comparison::*;
pub use containers::*;
pub use control::*;
pub use guards::*;
pub use memory::*;
pub use objects::*;
pub use values::*;

// =============================================================================
// Value Type Tags (matching NaN-boxing from prism_core::value)
// =============================================================================
//
// CANONICAL ENCODING (from prism_core::value):
//
// IEEE 754 quiet NaN + 3-bit tag in bits 48-50 + 48-bit payload in bits 0-47.
//
// | Tag | Type        | Full Pattern (QNAN | tag<<48) |
// |-----|-------------|-------------------------------|
// | 0   | None        | 0x7FF8_0000_0000_0000         |
// | 1   | Bool        | 0x7FF9_0000_0000_000{0,1}     |
// | 2   | Int (small) | 0x7FFA_xxxx_xxxx_xxxx         |
// | 3   | Object      | 0x7FFB_xxxx_xxxx_xxxx         |
// | 4   | String      | 0x7FFC_xxxx_xxxx_xxxx         |
// | 5-7 | Reserved    |                               |
//
// CRITICAL: These constants MUST match prism_core::value exactly.
// The JIT emits native code that creates/inspects values using these
// bit patterns. Any divergence causes silent data corruption at
// tier boundaries (interpreter ↔ JIT compiled code).

/// Tag bits for NaN-boxed values, unified with `prism_core::value`.
pub mod value_tags {
    // =========================================================================
    // Core Encoding Constants
    // =========================================================================

    /// Quiet NaN base: exponent all 1s + quiet NaN bit (bit 51).
    /// Bits 48-50 are free for our 3-bit type tag.
    pub const QNAN_BITS: u64 = 0x7FF8_0000_0000_0000;

    /// Tag shift position (bits 48-50 encode the type).
    pub const TAG_SHIFT: u32 = 48;

    /// 3-bit tag mask (bits 48-50 only, before shifting).
    pub const TAG_BITS_MASK: u64 = 0x0007_0000_0000_0000;

    // =========================================================================
    // Type Tags (matching prism_core::value TAG_* constants)
    // =========================================================================

    /// Tag for None value (tag = 0).
    pub const NONE_TAG: u64 = 0 << TAG_SHIFT;

    /// Tag for boolean values (tag = 1). Payload: 0 = false, 1 = true.
    pub const BOOL_TAG: u64 = 1 << TAG_SHIFT;

    /// Tag for integer values (tag = 2). Payload: 48-bit signed integer.
    pub const INT_TAG: u64 = 2 << TAG_SHIFT;

    /// Tag for object pointers (tag = 3). Payload: 48-bit pointer.
    pub const OBJECT_TAG: u64 = 3 << TAG_SHIFT;

    /// Tag for string values (tag = 4). Payload: 48-bit interned string pointer.
    pub const STRING_TAG: u64 = 4 << TAG_SHIFT;

    // =========================================================================
    // Masks
    // =========================================================================

    /// Payload mask (lower 48 bits).
    pub const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    /// Full tag mask: QNAN + tag bits (upper 16 bits effectively).
    /// Used for branchless type checks: `(bits & TAG_MASK) == (QNAN_BITS | XXX_TAG)`.
    pub const TAG_MASK: u64 = QNAN_BITS | TAG_BITS_MASK;

    // =========================================================================
    // Complete Value Constructors
    // =========================================================================

    /// Complete None value: `QNAN | (0 << 48)` = `0x7FF8_0000_0000_0000`.
    #[inline]
    pub const fn none_value() -> u64 {
        QNAN_BITS | NONE_TAG
    }

    /// Complete True value: `QNAN | (1 << 48) | 1` = `0x7FF9_0000_0000_0001`.
    #[inline]
    pub const fn true_value() -> u64 {
        QNAN_BITS | BOOL_TAG | 1
    }

    /// Complete False value: `QNAN | (1 << 48) | 0` = `0x7FF9_0000_0000_0000`.
    #[inline]
    pub const fn false_value() -> u64 {
        QNAN_BITS | BOOL_TAG
    }

    // =========================================================================
    // Tag Check Helpers (for JIT speculation guards)
    // =========================================================================

    /// Tag check value for integers (upper 16 bits of a boxed integer).
    /// Returns `((QNAN_BITS | INT_TAG) >> 48) as u16` = `0x7FFA`.
    #[inline]
    pub const fn int_tag_check() -> u16 {
        ((QNAN_BITS | INT_TAG) >> 48) as u16
    }

    /// Tag check value for strings (upper 16 bits of a boxed string).
    /// Returns `((QNAN_BITS | STRING_TAG) >> 48) as u16` = `0x7FFC`.
    #[inline]
    pub const fn string_tag_check() -> u16 {
        ((QNAN_BITS | STRING_TAG) >> 48) as u16
    }

    /// Tag check value for objects (upper 16 bits of a boxed object).
    /// Returns `((QNAN_BITS | OBJECT_TAG) >> 48) as u16` = `0x7FFB`.
    #[inline]
    pub const fn object_tag_check() -> u16 {
        ((QNAN_BITS | OBJECT_TAG) >> 48) as u16
    }

    /// Tag check value for booleans (upper 16 bits of a boxed bool).
    /// Returns `((QNAN_BITS | BOOL_TAG) >> 48) as u16` = `0x7FF9`.
    #[inline]
    pub const fn bool_tag_check() -> u16 {
        ((QNAN_BITS | BOOL_TAG) >> 48) as u16
    }

    /// Tag check value for None (upper 16 bits of None).
    /// Returns `((QNAN_BITS | NONE_TAG) >> 48) as u16` = `0x7FF8`.
    #[inline]
    pub const fn none_tag_check() -> u16 {
        ((QNAN_BITS | NONE_TAG) >> 48) as u16
    }

    // =========================================================================
    // Boxing Helpers
    // =========================================================================

    /// Box an integer value.
    #[inline]
    pub const fn box_int(value: i64) -> u64 {
        (QNAN_BITS | INT_TAG) | ((value as u64) & PAYLOAD_MASK)
    }

    /// Box a boolean value.
    #[inline]
    pub const fn box_bool(value: bool) -> u64 {
        QNAN_BITS | BOOL_TAG | (value as u64)
    }

    /// Box an object pointer.
    #[inline]
    pub const fn box_object(ptr: u64) -> u64 {
        (QNAN_BITS | OBJECT_TAG) | (ptr & PAYLOAD_MASK)
    }

    /// Box a string pointer.
    #[inline]
    pub const fn box_string(ptr: u64) -> u64 {
        (QNAN_BITS | STRING_TAG) | (ptr & PAYLOAD_MASK)
    }

    // =========================================================================
    // Full Tag Patterns (for branchless comparisons)
    // =========================================================================

    /// Complete tag pattern for int: `QNAN_BITS | INT_TAG` = `0x7FFA_0000_0000_0000`.
    pub const INT_PATTERN: u64 = QNAN_BITS | INT_TAG;

    /// Complete tag pattern for string: `QNAN_BITS | STRING_TAG` = `0x7FFC_0000_0000_0000`.
    pub const STRING_PATTERN: u64 = QNAN_BITS | STRING_TAG;

    /// Complete tag pattern for object: `QNAN_BITS | OBJECT_TAG` = `0x7FFB_0000_0000_0000`.
    pub const OBJECT_PATTERN: u64 = QNAN_BITS | OBJECT_TAG;

    /// Complete tag pattern for bool: `QNAN_BITS | BOOL_TAG` = `0x7FF9_0000_0000_0000`.
    pub const BOOL_PATTERN: u64 = QNAN_BITS | BOOL_TAG;

    /// Complete tag pattern for none: `QNAN_BITS | NONE_TAG` = `0x7FF8_0000_0000_0000`.
    pub const NONE_PATTERN: u64 = QNAN_BITS | NONE_TAG;
}

// =============================================================================
// Template Context
// =============================================================================

/// Context passed to all templates during code generation.
pub struct TemplateContext<'a> {
    /// The assembler to emit code to.
    pub asm: &'a mut Assembler,
    /// Frame layout for the current function.
    pub frame: &'a FrameLayout,
    /// Register assignments.
    pub regs: RegisterAssignment,
    /// JIT calling convention.
    pub cc: JitCallingConvention,
    /// Labels for deoptimization stubs.
    pub deopt_labels: Vec<Label>,
    /// Current bytecode offset (for debugging/deopt).
    pub bc_offset: usize,
}

impl<'a> TemplateContext<'a> {
    /// Create a new template context.
    #[inline]
    pub fn new(asm: &'a mut Assembler, frame: &'a FrameLayout) -> Self {
        TemplateContext {
            asm,
            frame,
            regs: RegisterAssignment::host(),
            cc: JitCallingConvention::host(),
            deopt_labels: Vec::new(),
            bc_offset: 0,
        }
    }

    /// Set the current bytecode offset.
    #[inline]
    pub fn set_bc_offset(&mut self, offset: usize) {
        self.bc_offset = offset;
    }

    /// Create a deoptimization label and return its index.
    #[inline]
    pub fn create_deopt_label(&mut self) -> usize {
        let idx = self.deopt_labels.len();
        let label = self.asm.create_label();
        self.deopt_labels.push(label);
        idx
    }

    /// Get a deoptimization label by index.
    #[inline]
    pub fn deopt_label(&self, idx: usize) -> Label {
        self.deopt_labels[idx]
    }
}

// =============================================================================
// Template Trait
// =============================================================================

/// Trait for opcode templates.
///
/// Each template generates native code for a specific bytecode operation.
/// Templates are designed to be:
/// - Fast to generate (minimal analysis)
/// - Type-specialized where profitable
/// - Deoptimization-capable for type mismatches
pub trait OpcodeTemplate {
    /// Emit the native code for this opcode.
    fn emit(&self, ctx: &mut TemplateContext);

    /// Get the estimated code size for this template (for allocation).
    fn estimated_size(&self) -> usize {
        32 // Default estimate
    }
}

// =============================================================================
// Template Registry
// =============================================================================

/// Centralized registry of all opcode templates.
///
/// Provides O(1) dispatch to the appropriate template emitter.
#[derive(Default)]
pub struct TemplateRegistry {
    // Statistics for monitoring
    total_emitted: usize,
    bytes_generated: usize,
}

impl TemplateRegistry {
    /// Create a new template registry.
    #[inline]
    pub const fn new() -> Self {
        TemplateRegistry {
            total_emitted: 0,
            bytes_generated: 0,
        }
    }

    /// Get statistics about template usage.
    #[inline]
    pub fn stats(&self) -> (usize, usize) {
        (self.total_emitted, self.bytes_generated)
    }

    /// Record template emission for statistics.
    #[inline]
    pub fn record_emission(&mut self, bytes: usize) {
        self.total_emitted += 1;
        self.bytes_generated += bytes;
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

    #[test]
    fn test_template_context_creation() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let ctx = TemplateContext::new(&mut asm, &frame);

        assert_eq!(ctx.bc_offset, 0);
        assert!(ctx.deopt_labels.is_empty());
    }

    #[test]
    fn test_deopt_label_creation() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let idx1 = ctx.create_deopt_label();
        let idx2 = ctx.create_deopt_label();

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(ctx.deopt_labels.len(), 2);
    }

    // =========================================================================
    // Value Tag Bit Pattern Tests
    // =========================================================================

    #[test]
    fn test_value_tags_none() {
        // None = QNAN | (0 << 48) = 0x7FF8_0000_0000_0000
        assert_eq!(value_tags::none_value(), 0x7FF8_0000_0000_0000);
        assert_eq!(value_tags::none_tag_check(), 0x7FF8);
    }

    #[test]
    fn test_value_tags_bool() {
        // True = QNAN | (1 << 48) | 1 = 0x7FF9_0000_0000_0001
        assert_eq!(value_tags::true_value(), 0x7FF9_0000_0000_0001);
        // False = QNAN | (1 << 48) | 0 = 0x7FF9_0000_0000_0000
        assert_eq!(value_tags::false_value(), 0x7FF9_0000_0000_0000);
        assert_eq!(value_tags::bool_tag_check(), 0x7FF9);
    }

    #[test]
    fn test_value_tags_int() {
        // Int tag check = upper 16 bits of (QNAN | INT_TAG) = 0x7FFA
        assert_eq!(value_tags::int_tag_check(), 0x7FFA);
        assert_eq!(value_tags::INT_PATTERN, 0x7FFA_0000_0000_0000);
    }

    #[test]
    fn test_value_tags_object() {
        assert_eq!(value_tags::object_tag_check(), 0x7FFB);
        assert_eq!(value_tags::OBJECT_PATTERN, 0x7FFB_0000_0000_0000);
    }

    #[test]
    fn test_value_tags_string() {
        assert_eq!(value_tags::string_tag_check(), 0x7FFC);
        assert_eq!(value_tags::STRING_PATTERN, 0x7FFC_0000_0000_0000);
    }

    #[test]
    fn test_value_tags_no_overlap() {
        // Verify all tag patterns are distinct
        let patterns = [
            value_tags::NONE_PATTERN,
            value_tags::BOOL_PATTERN,
            value_tags::INT_PATTERN,
            value_tags::OBJECT_PATTERN,
            value_tags::STRING_PATTERN,
        ];
        for i in 0..patterns.len() {
            for j in (i + 1)..patterns.len() {
                assert_ne!(
                    patterns[i], patterns[j],
                    "Tag patterns {} and {} must not overlap",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_value_tags_tag_mask_extracts_correctly() {
        // TAG_MASK should extract QNAN + 3-bit tag, stripping payload
        let int_42 = value_tags::box_int(42);
        assert_eq!(int_42 & value_tags::TAG_MASK, value_tags::INT_PATTERN);

        let none = value_tags::none_value();
        assert_eq!(none & value_tags::TAG_MASK, value_tags::NONE_PATTERN);

        let true_val = value_tags::true_value();
        assert_eq!(true_val & value_tags::TAG_MASK, value_tags::BOOL_PATTERN);
    }

    // =========================================================================
    // Boxing Tests
    // =========================================================================

    #[test]
    fn test_box_int() {
        let boxed = value_tags::box_int(42);
        assert_eq!(boxed & value_tags::PAYLOAD_MASK, 42);
        assert_eq!((boxed >> 48) as u16, value_tags::int_tag_check());
    }

    #[test]
    fn test_box_int_negative() {
        let boxed = value_tags::box_int(-1);
        // Payload should be the lower 48 bits of -1 (sign-extended)
        let payload = boxed & value_tags::PAYLOAD_MASK;
        assert_eq!(payload, 0x0000_FFFF_FFFF_FFFF);
        assert_eq!((boxed >> 48) as u16, value_tags::int_tag_check());
    }

    #[test]
    fn test_box_int_zero() {
        let boxed = value_tags::box_int(0);
        assert_eq!(boxed & value_tags::PAYLOAD_MASK, 0);
        assert_eq!((boxed >> 48) as u16, value_tags::int_tag_check());
    }

    #[test]
    fn test_box_bool() {
        assert_eq!(value_tags::box_bool(true), value_tags::true_value());
        assert_eq!(value_tags::box_bool(false), value_tags::false_value());
    }

    #[test]
    fn test_box_object() {
        let ptr = 0x0000_1234_5678_9ABCu64;
        let boxed = value_tags::box_object(ptr);
        assert_eq!(boxed & value_tags::PAYLOAD_MASK, ptr);
        assert_eq!((boxed >> 48) as u16, value_tags::object_tag_check());
    }

    #[test]
    fn test_box_string() {
        let ptr = 0x0000_DEAD_BEEF_CAFEu64;
        let boxed = value_tags::box_string(ptr);
        assert_eq!(boxed & value_tags::PAYLOAD_MASK, ptr);
        assert_eq!((boxed >> 48) as u16, value_tags::string_tag_check());
    }

    // =========================================================================
    // Cross-Validation with prism_core::Value
    // =========================================================================
    //
    // These tests are the ultimate proof of correctness: they verify that
    // JIT-produced bit patterns are byte-identical to interpreter-produced ones.
    // Value is #[repr(transparent)] wrapping u64, so transmute is sound.

    /// Extract raw u64 bits from a prism_core::Value.
    ///
    /// # Safety
    /// Value is `#[repr(transparent)]` around `u64`, so this is always sound.
    fn value_bits(v: prism_core::Value) -> u64 {
        // SAFETY: Value is #[repr(transparent)] wrapping u64
        unsafe { std::mem::transmute(v) }
    }

    #[test]
    fn test_cross_validate_none_with_core() {
        let core_none = prism_core::Value::none();
        assert_eq!(value_bits(core_none), value_tags::none_value());
    }

    #[test]
    fn test_cross_validate_true_with_core() {
        let core_true = prism_core::Value::bool(true);
        assert_eq!(value_bits(core_true), value_tags::true_value());
    }

    #[test]
    fn test_cross_validate_false_with_core() {
        let core_false = prism_core::Value::bool(false);
        assert_eq!(value_bits(core_false), value_tags::false_value());
    }

    #[test]
    fn test_cross_validate_int_with_core() {
        for i in [0i64, 1, -1, 42, -42, 255, 1000, -1000] {
            let core_int = prism_core::Value::int(i).unwrap();
            let jit_int = value_tags::box_int(i);
            let core_bits = value_bits(core_int);
            assert_eq!(
                core_bits, jit_int,
                "Mismatch for int {}: core={:#018x}, jit={:#018x}",
                i, core_bits, jit_int
            );
        }
    }

    #[test]
    fn test_cross_validate_int_tag_check_with_core() {
        let core_int = prism_core::Value::int(0).unwrap();
        let core_upper = (value_bits(core_int) >> 48) as u16;
        assert_eq!(core_upper, value_tags::int_tag_check());
    }

    #[test]
    fn test_cross_validate_string_tag_pattern_with_core() {
        assert_eq!(
            value_tags::STRING_PATTERN,
            prism_core::value::STRING_TAG_PATTERN
        );
    }

    #[test]
    fn test_cross_validate_int_tag_pattern_with_core() {
        assert_eq!(value_tags::INT_PATTERN, prism_core::value::INT_TAG_PATTERN);
    }

    #[test]
    fn test_cross_validate_payload_mask_with_core() {
        assert_eq!(
            value_tags::PAYLOAD_MASK,
            prism_core::value::VALUE_PAYLOAD_MASK
        );
    }

    #[test]
    fn test_cross_validate_tag_mask_with_core() {
        assert_eq!(value_tags::TAG_MASK, prism_core::value::TYPE_TAG_MASK);
    }

    #[test]
    fn test_template_registry() {
        let mut registry = TemplateRegistry::new();
        assert_eq!(registry.stats(), (0, 0));

        registry.record_emission(100);
        registry.record_emission(50);
        assert_eq!(registry.stats(), (2, 150));
    }
}
