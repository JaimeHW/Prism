//! Memory operation templates - load/store locals, globals, closures, moves.
//!
//! Templates for moving data between bytecode registers and other storage:
//! - Local variables (in frame)
//! - Global variables (via global scope)
//! - Closure variables (captured from outer scopes)
//! - Register-to-register moves

use super::{OpcodeTemplate, TemplateContext};
use crate::backend::x64::{Gpr, MemOperand};

// =============================================================================
// Move Between Registers
// =============================================================================

/// Template for moving a value between registers.
pub struct MoveTemplate {
    pub src_reg: u8,
    pub dst_reg: u8,
}

impl OpcodeTemplate for MoveTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Optimize self-moves to no-op
        if self.src_reg == self.dst_reg {
            return;
        }

        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        ctx.asm.mov_rm(ctx.regs.accumulator, &src_slot);
        ctx.asm.mov_mr(&dst_slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        16 // Two memory moves through register
    }
}

// =============================================================================
// Load/Store Local Variables
// =============================================================================

/// Template for loading a local variable.
///
/// Locals are stored in the frame at a fixed offset from RBP.
pub struct LoadLocalTemplate {
    pub dst_reg: u8,
    pub local_idx: u16,
}

impl OpcodeTemplate for LoadLocalTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let local_slot = ctx.frame.local_slot(self.local_idx);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        ctx.asm.mov_rm(ctx.regs.accumulator, &local_slot);
        ctx.asm.mov_mr(&dst_slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        16
    }
}

/// Template for storing to a local variable.
pub struct StoreLocalTemplate {
    pub src_reg: u8,
    pub local_idx: u16,
}

impl OpcodeTemplate for StoreLocalTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let local_slot = ctx.frame.local_slot(self.local_idx);

        ctx.asm.mov_rm(ctx.regs.accumulator, &src_slot);
        ctx.asm.mov_mr(&local_slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        16
    }
}

// =============================================================================
// Delete Local
// =============================================================================

/// Template for deleting a local variable (setting to undefined/uninitialized).
pub struct DeleteLocalTemplate {
    pub local_idx: u16,
}

impl OpcodeTemplate for DeleteLocalTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Set to a special "uninitialized" value (we use 0 as a sentinel)
        // In practice, accesses should check for this
        ctx.asm.xor_rr(ctx.regs.accumulator, ctx.regs.accumulator);
        let local_slot = ctx.frame.local_slot(self.local_idx);
        ctx.asm.mov_mr(&local_slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        12
    }
}

// =============================================================================
// Load/Store Global Variables
// =============================================================================

/// Template for loading a global variable.
///
/// Globals are accessed via a global scope pointer stored in the frame.
/// This uses an indirect lookup through the scope.
pub struct LoadGlobalTemplate {
    pub dst_reg: u8,
    pub name_idx: u16,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for LoadGlobalTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Load global scope pointer
        let scope_slot = ctx.frame.global_scope_slot();
        ctx.asm.mov_rm(ctx.regs.scratch1, &scope_slot);

        // Call runtime helper to look up global (would be inlined in full impl)
        // For now, emit a call to a stub address
        // In production, this would be an inline cache

        // Simplified: just load the name index and deopt
        // The actual implementation would do proper lookup
        ctx.asm.mov_ri32(ctx.regs.scratch2, self.name_idx as u32);

        // Deopt if global not found (placeholder - actual impl would check)
        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48 // Multiple loads + potential call
    }
}

/// Template for storing to a global variable.
pub struct StoreGlobalTemplate {
    pub src_reg: u8,
    pub name_idx: u16,
}

impl OpcodeTemplate for StoreGlobalTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Load global scope pointer
        let scope_slot = ctx.frame.global_scope_slot();
        ctx.asm.mov_rm(ctx.regs.scratch1, &scope_slot);

        // Load value to store
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(ctx.regs.accumulator, &src_slot);

        // Store name index
        ctx.asm.mov_ri32(ctx.regs.scratch2, self.name_idx as u32);

        // In production, this would call a runtime helper or use inline cache
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

/// Template for deleting a global variable.
pub struct DeleteGlobalTemplate {
    pub name_idx: u16,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for DeleteGlobalTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Load global scope
        let scope_slot = ctx.frame.global_scope_slot();
        ctx.asm.mov_rm(ctx.regs.scratch1, &scope_slot);

        // Name index
        ctx.asm.mov_ri32(ctx.regs.scratch2, self.name_idx as u32);

        // Would call runtime delete helper
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        32
    }
}

// =============================================================================
// Load/Store Closure Variables
// =============================================================================

/// Template for loading a closure variable.
///
/// Closure variables are stored in a closure environment linked from the frame.
pub struct LoadClosureTemplate {
    pub dst_reg: u8,
    pub closure_idx: u16,
}

impl OpcodeTemplate for LoadClosureTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Load closure environment pointer
        let closure_slot = ctx.frame.closure_env_slot();
        ctx.asm.mov_rm(ctx.regs.scratch1, &closure_slot);

        // Load value at closure index (each slot is 8 bytes)
        let offset = (self.closure_idx as i32) * 8;
        let closure_mem = MemOperand::base_disp(ctx.regs.scratch1, offset);
        ctx.asm.mov_rm(ctx.regs.accumulator, &closure_mem);

        // Store to destination register
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        24
    }
}

/// Template for storing to a closure variable.
pub struct StoreClosureTemplate {
    pub src_reg: u8,
    pub closure_idx: u16,
}

impl OpcodeTemplate for StoreClosureTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Load closure environment pointer
        let closure_slot = ctx.frame.closure_env_slot();
        ctx.asm.mov_rm(ctx.regs.scratch1, &closure_slot);

        // Load value to store
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(ctx.regs.accumulator, &src_slot);

        // Store at closure index
        let offset = (self.closure_idx as i32) * 8;
        let closure_mem = MemOperand::base_disp(ctx.regs.scratch1, offset);
        ctx.asm.mov_mr(&closure_mem, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        24
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
    fn test_move_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = MoveTemplate {
            src_reg: 0,
            dst_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_move_self_noop() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = MoveTemplate {
            src_reg: 0,
            dst_reg: 0,
        };
        template.emit(&mut ctx);

        // Self-move should produce no code
        assert_eq!(ctx.asm.offset(), 0);
    }

    #[test]
    fn test_load_local_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = LoadLocalTemplate {
            dst_reg: 0,
            local_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_store_local_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = StoreLocalTemplate {
            src_reg: 0,
            local_idx: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_load_closure_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = LoadClosureTemplate {
            dst_reg: 0,
            closure_idx: 2,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_store_closure_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = StoreClosureTemplate {
            src_reg: 1,
            closure_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }
}
