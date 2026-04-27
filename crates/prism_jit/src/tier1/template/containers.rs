//! Container construction templates - list, tuple, dict, set, string.
//!
//! These templates handle building Python container types.
//! Most require allocation and runtime support.

use super::{OpcodeTemplate, TemplateContext};

// =============================================================================
// Build Containers
// =============================================================================

/// Template for building a list.
pub struct BuildListTemplate {
    pub dst_reg: u8,
    pub start_reg: u8,
    pub count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BuildListTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // List construction requires allocation - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for building a tuple.
pub struct BuildTupleTemplate {
    pub dst_reg: u8,
    pub start_reg: u8,
    pub count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BuildTupleTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for building a set.
pub struct BuildSetTemplate {
    pub dst_reg: u8,
    pub start_reg: u8,
    pub count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BuildSetTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for building a dict.
pub struct BuildDictTemplate {
    pub dst_reg: u8,
    pub start_reg: u8,
    pub count: u8, // Number of key-value pairs
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BuildDictTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for building a string (f-string concatenation).
pub struct BuildStringTemplate {
    pub dst_reg: u8,
    pub start_reg: u8,
    pub count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BuildStringTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Container Mutation
// =============================================================================

/// Template for list append.
pub struct ListAppendTemplate {
    pub list_reg: u8,
    pub value_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ListAppendTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for set add.
pub struct SetAddTemplate {
    pub set_reg: u8,
    pub value_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for SetAddTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for dict set (key-value pair).
pub struct DictSetTemplate {
    pub dict_reg: u8,
    pub key_reg: u8,
    pub value_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for DictSetTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Unpacking
// =============================================================================

/// Template for unpacking a sequence.
pub struct UnpackSequenceTemplate {
    pub src_reg: u8,
    pub dst_start_reg: u8,
    pub count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for UnpackSequenceTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for extended unpacking (with *rest).
pub struct UnpackExTemplate {
    pub src_reg: u8,
    pub dst_start_reg: u8,
    pub before_count: u8,
    pub after_count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for UnpackExTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Slice
// =============================================================================

/// Template for building a slice object.
pub struct BuildSliceTemplate {
    pub dst_reg: u8,
    pub start_reg: u8,
    pub stop_reg: u8,
    pub step_reg: Option<u8>,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BuildSliceTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Import
// =============================================================================

/// Template for import name.
pub struct ImportNameTemplate {
    pub dst_reg: u8,
    pub name_idx: u16,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ImportNameTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for import from.
pub struct ImportFromTemplate {
    pub dst_reg: u8,
    pub module_reg: u8,
    pub name_idx: u16,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ImportFromTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for import star.
pub struct ImportStarTemplate {
    pub module_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ImportStarTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}
