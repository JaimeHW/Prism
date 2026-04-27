//! Dict-specialized JIT templates for high-performance dict operations.
//!
//! Provides type-specialized native code generation for:
//! - **Dict lookup** (`dict[key]`) — type guard + deopt for hash lookup
//! - **Dict store** (`dict[key] = v`) — type guard + deopt for hash insert
//! - **Dict contains** (`key in dict`) — type guard + deopt for hash probe
//! - **Dict merge** (`dict | dict`) — dual type guard + deopt
//! - **Dict length** (`len(dict)`) — type guard + deopt
//!
//! # Memory Model
//!
//! `DictObject` is `#[repr(C)]`:
//! ```text
//! Offset  Size  Field
//! ──────  ────  ─────────────────────
//!   0      4    ObjectHeader.type_id  (TypeId, u32)
//!   4      4    ObjectHeader.gc_flags (AtomicU32)
//!   8      8    ObjectHeader.hash     (u64)
//!  16      ?    FxHashMap<HashableValue, Value>
//! ```
//!
//! Unlike `ListObject`, the `FxHashMap` does not have a stable internal layout
//! suitable for direct memory access from JIT code. Therefore, all dict
//! operations use the following strategy:
//!
//! # Performance Strategy
//!
//! 1. **Inline type guard**: Verify the value is an OBJECT with `TypeId::DICT`.
//!    This eliminates interpreter-level type dispatch overhead.
//! 2. **Key type guard**: Where applicable, verify the key type inline.
//! 3. **Deopt to interpreter**: For the actual hash operation (lookup, insert,
//!    membership test), deopt to Tier 0/interpreter. Tier 2 will inline these
//!    via runtime helper calls.
//!
//! The inline type guard alone provides significant speedup:
//! - Eliminates polymorphic dispatch in the interpreter
//! - Enables the JIT to speculate on dict-only access patterns
//! - Provides a deopt point when types change (e.g., dict replaced by custom object)
//!
//! For the string-key fast path, we additionally inline a STRING_TAG check
//! on the key, enabling the JIT to skip the key type dispatch.

use super::specialize_common::{
    emit_int_check_and_extract, emit_object_check_and_extract, emit_string_check_and_extract,
    emit_typed_object_check_and_extract, type_ids,
};
use super::{OpcodeTemplate, TemplateContext};

// =============================================================================
// Dict Lookup Template
// =============================================================================

/// Template for dict item lookup (`dict[key]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for actual hash lookup
///
/// This provides a typed deopt — the JIT knows the receiver is a dict,
/// so re-entry to the interpreter can take a fast path.
///
/// # Estimated Code Size
///
/// ~40 bytes: object tag check (~20), type_id check (~12), jmp (~8)
pub struct DictGetTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key value.
    pub key_reg: u8,
    /// Destination register for the result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictGetTemplate {
    /// Create a new dict get template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictGetTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value into accumulator
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: verify dict is OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Hash lookup requires runtime support — deopt
        // The interpreter re-enters with the knowledge that the receiver is a dict
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict String-Key Lookup Template
// =============================================================================

/// Template for dict lookup with a known-string key (`dict[str_key]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Type-check key: STRING_TAG
/// 3. Deopt to interpreter for the actual hash probe
///
/// This double type guard is the most common dict access pattern in Python
/// (string keys account for >90% of dict accesses in typical programs).
/// The inline string-key verification enables the interpreter to skip the
/// key-type dispatch entirely on re-entry.
///
/// # Estimated Code Size
///
/// ~70 bytes: dict guard (~32), string guard (~20), jmp (~8), loads (~10)
pub struct DictGetStrTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key (should be a string).
    pub key_reg: u8,
    /// Destination register for the result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictGetStrTemplate {
    /// Create a new dict string-key get template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictGetStrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Load key value
        let key_slot = ctx.frame.register_slot(self.key_reg as u16);
        ctx.asm.mov_rm(scratch1, &key_slot);

        // Type guard: key must be a string
        emit_string_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt for actual lookup — both types verified
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        72
    }
}

// =============================================================================
// Dict Int-Key Lookup Template
// =============================================================================

/// Template for dict lookup with a known-integer key (`dict[int_key]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Type-check key: INT_TAG
/// 3. Deopt to interpreter for the actual hash probe
///
/// Integer keys are the second most common dict access pattern, especially
/// in numeric and data-processing code.
///
/// # Estimated Code Size
///
/// ~70 bytes: dict guard (~32), int guard (~20), jmp (~8), loads (~10)
pub struct DictGetIntTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key (should be an integer).
    pub key_reg: u8,
    /// Destination register for the result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictGetIntTemplate {
    /// Create a new dict int-key get template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictGetIntTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Load key value
        let key_slot = ctx.frame.register_slot(self.key_reg as u16);
        ctx.asm.mov_rm(scratch1, &key_slot);

        // Type guard: key must be an integer
        emit_int_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt for actual lookup — both types verified
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        72
    }
}

// =============================================================================
// Dict Store Template
// =============================================================================

/// Template for dict item store (`dict[key] = value`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for actual hash insertion
///
/// The type guard ensures we only deopt for true dict operations,
/// not for custom `__setitem__` on arbitrary objects.
///
/// # Estimated Code Size
///
/// ~40 bytes: dict guard (~32), jmp (~8)
pub struct DictSetFastTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key value.
    pub key_reg: u8,
    /// Register holding the value to store.
    pub value_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictSetFastTemplate {
    /// Create a new dict set template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictSetFastTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Hash insertion requires runtime support — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict String-Key Store Template
// =============================================================================

/// Template for dict store with a known-string key (`dict[str_key] = value`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Type-check key: STRING_TAG
/// 3. Deopt to interpreter for hash insertion
///
/// The double type guard eliminates both receiver-type and key-type
/// dispatch in the interpreter on re-entry.
///
/// # Estimated Code Size
///
/// ~70 bytes: dict guard (~32), string guard (~20), jmp (~8), loads (~10)
pub struct DictSetStrTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key (should be a string).
    pub key_reg: u8,
    /// Register holding the value to store.
    pub value_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictSetStrTemplate {
    /// Create a new dict string-key set template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictSetStrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Load key value
        let key_slot = ctx.frame.register_slot(self.key_reg as u16);
        ctx.asm.mov_rm(scratch1, &key_slot);

        // Type guard: key must be a string
        emit_string_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt for actual insertion — both types verified
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        72
    }
}

// =============================================================================
// Dict Contains Template
// =============================================================================

/// Template for dict containment check (`key in dict`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for has probe
///
/// Membership testing is a hot operation in many Python workloads.
/// The inline type guard enables the JIT to specialize the `in` operator
/// without falling through to the generic `__contains__` dispatch.
///
/// # Estimated Code Size
///
/// ~48 bytes: dict guard (~32), load (~8), jmp (~8)
pub struct DictContainsTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key to check.
    pub key_reg: u8,
    /// Destination register for the boolean result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictContainsTemplate {
    /// Create a new dict contains template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictContainsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Membership test requires runtime — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict Merge Template
// =============================================================================

/// Template for dict merge (`dict | dict` or `dict.update(dict)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check LHS: OBJECT_TAG + TypeId::DICT
/// 2. Type-check RHS: OBJECT_TAG + TypeId::DICT
/// 3. Deopt to interpreter for actual merge
///
/// This ensures that `|` is only applied to actual dict objects,
/// not custom `__or__` implementations on other types.
///
/// # Estimated Code Size
///
/// ~80 bytes: 2× dict guard (~64), jmp (~8), loads (~8)
pub struct DictMergeTemplate {
    /// Register holding the left dict.
    pub lhs_reg: u8,
    /// Register holding the right dict.
    pub rhs_reg: u8,
    /// Destination register for the merged dict.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictMergeTemplate {
    /// Create a new dict merge template.
    #[inline]
    pub fn new(lhs_reg: u8, rhs_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            lhs_reg,
            rhs_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictMergeTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and verify LHS dict
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Load and verify RHS dict
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(scratch1, &rhs_slot);
        emit_typed_object_check_and_extract(
            ctx,
            scratch1,
            scratch1,
            scratch2,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Merge requires allocation + hash copying — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80
    }
}

// =============================================================================
// Dict Delete Template
// =============================================================================

/// Template for dict item deletion (`del dict[key]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for actual hash removal
///
/// # Estimated Code Size
///
/// ~48 bytes: dict guard (~32), load (~8), jmp (~8)
pub struct DictDeleteTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key to delete.
    pub key_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictDeleteTemplate {
    /// Create a new dict delete template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictDeleteTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Deletion requires runtime — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict Keys/Values/Items Guard Template
// =============================================================================

/// Template for dict view operations (`dict.keys()`, `dict.values()`, `dict.items()`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for view construction
///
/// These operations always allocate a new view object, so they must deopt.
/// The type guard ensures we only handle actual dict objects.
///
/// # Estimated Code Size
///
/// ~40 bytes: dict guard (~32), jmp (~8)
pub struct DictViewGuardTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Destination register for the view result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictViewGuardTemplate {
    /// Create a new dict view guard template.
    #[inline]
    pub fn new(dict_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictViewGuardTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // View construction requires allocation — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict Get-Or-Default Template
// =============================================================================

/// Template for `dict.get(key, default)` with type guards.
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Type-check key: STRING_TAG (for the common string-key case)
/// 3. Deopt to interpreter for the actual hash probe + default handling
///
/// `dict.get()` is extremely common in Python code and is the second
/// most frequently called dict method after `__getitem__`.
///
/// # Estimated Code Size
///
/// ~70 bytes: dict guard (~32), string guard (~20), jmp (~8), loads (~10)
pub struct DictGetOrDefaultTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key.
    pub key_reg: u8,
    /// Register holding the default value.
    pub default_reg: u8,
    /// Destination register for the result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictGetOrDefaultTemplate {
    /// Create a new dict get-or-default template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, default_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            default_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictGetOrDefaultTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // .get() with default requires runtime hash probe — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}
