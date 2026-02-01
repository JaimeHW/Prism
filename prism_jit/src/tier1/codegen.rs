//! Template JIT compilation driver.
//!
//! This module orchestrates the compilation of bytecode to native code
//! using the template-based approach. The flow is:
//!
//! 1. Analyze bytecode to determine register count, jumps, etc.
//! 2. Create frame layout
//! 3. Emit prologue
//! 4. For each bytecode instruction, emit corresponding template
//! 5. Emit deopt stubs
//! 6. Emit epilogue
//! 7. Finalize and return executable code

use super::deopt::{DeoptInfo, DeoptReason, DeoptStubGenerator};
use super::frame::{FrameLayout, JitCallingConvention};
use super::template::*;
use crate::backend::x64::{Assembler, ExecutableBuffer, Gpr, Label};

use std::collections::HashMap;

// =============================================================================
// Compilation Result
// =============================================================================

/// Result of JIT compilation.
pub struct CompiledFunction {
    /// The executable code.
    pub code: ExecutableBuffer,
    /// Mapping from bytecode offset to native offset.
    pub bc_to_native: HashMap<u32, u32>,
    /// Deoptimization info for each deopt point.
    pub deopt_info: Vec<DeoptInfo>,
    /// Frame layout used.
    pub frame_layout: FrameLayout,
}

impl CompiledFunction {
    /// Get the entry point as a function pointer.
    ///
    /// # Safety
    /// The caller must ensure the function is called with the correct
    /// calling convention and argument types.
    pub unsafe fn as_fn<F: Copy>(&self) -> F {
        unsafe { self.code.as_fn() }
    }

    /// Get the native offset for a bytecode offset.
    pub fn native_offset(&self, bc_offset: u32) -> Option<u32> {
        self.bc_to_native.get(&bc_offset).copied()
    }

    /// Get deopt info by stub index.
    pub fn get_deopt_info(&self, stub_index: u16) -> Option<&DeoptInfo> {
        self.deopt_info.get(stub_index as usize)
    }
}

// =============================================================================
// Template Compiler
// =============================================================================

/// Main entry point for template JIT compilation.
pub struct TemplateCompiler {
    /// Address of the runtime deopt handler.
    deopt_handler: u64,
}

impl TemplateCompiler {
    /// Create a new template compiler.
    pub fn new(deopt_handler: u64) -> Self {
        TemplateCompiler { deopt_handler }
    }

    /// Create a compiler with a dummy deopt handler (for testing).
    pub fn new_for_testing() -> Self {
        // Use a placeholder address - actual deopts will crash but
        // we can still test code generation
        Self::new(0xDEAD_BEEF_DEAD_BEEF)
    }

    /// Compile a function.
    ///
    /// This is a simplified API that takes just the essential info.
    /// A real implementation would take a `CompiledCode` from the bytecode compiler.
    pub fn compile(
        &self,
        num_registers: u16,
        instructions: &[TemplateInstruction],
    ) -> Result<CompiledFunction, String> {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(num_registers);
        let cc = JitCallingConvention::host();
        let mut bc_to_native = HashMap::new();
        let mut deopt_gen = DeoptStubGenerator::new();

        // Create deopt labels first
        let mut deopt_labels = Vec::with_capacity(instructions.len());
        for _ in 0..instructions.len() {
            deopt_labels.push(asm.create_label());
        }

        // Collect all labels for jump targets
        let mut labels: HashMap<u32, Label> = HashMap::new();
        for instr in instructions {
            if let Some(target) = instr.jump_target() {
                if !labels.contains_key(&target) {
                    labels.insert(target, asm.create_label());
                }
            }
        }

        // Emit prologue
        self.emit_prologue(&mut asm, &frame, &cc);

        // Emit each instruction using a scoped context
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.deopt_labels = deopt_labels.clone();

            for (idx, instr) in instructions.iter().enumerate() {
                let bc_offset = instr.bc_offset();
                ctx.set_bc_offset(bc_offset as usize);

                // Record bc->native mapping
                bc_to_native.insert(bc_offset, ctx.asm.offset() as u32);

                // Bind label if this is a jump target
                if let Some(label) = labels.get(&bc_offset) {
                    ctx.asm.bind_label(*label);
                }

                // Register deopt for this instruction
                if instr.can_deopt() {
                    deopt_gen.register_deopt(deopt_labels[idx], bc_offset, instr.deopt_reason());
                }

                // Emit the template
                self.emit_instruction(&mut ctx, instr, &labels, idx);
            }
        }
        // ctx is dropped here, releasing the borrow on asm

        // Emit epilogue
        self.emit_epilogue(&mut asm, &frame);

        // Emit deopt stubs
        let deopt_info = deopt_gen.emit_stubs(&mut asm, &frame, self.deopt_handler);

        // Finalize - now we own asm
        let code = asm.finalize_executable()?;

        Ok(CompiledFunction {
            code,
            bc_to_native,
            deopt_info,
            frame_layout: frame,
        })
    }

    /// Emit function prologue.
    fn emit_prologue(&self, asm: &mut Assembler, frame: &FrameLayout, cc: &JitCallingConvention) {
        // Push callee-saved registers
        for reg in frame.saved_regs.iter() {
            asm.push(reg);
        }

        // Set up frame pointer
        asm.push(Gpr::Rbp);
        asm.mov_rr(Gpr::Rbp, Gpr::Rsp);

        // Allocate stack space
        let frame_size = frame.frame_size();
        if frame_size > 0 {
            asm.sub_ri(Gpr::Rsp, frame_size);
        }

        // Store context pointer (first argument)
        let ctx_slot = frame.context_slot();
        asm.mov_mr(&ctx_slot, cc.arg0);
    }

    /// Emit function epilogue.
    fn emit_epilogue(&self, asm: &mut Assembler, frame: &FrameLayout) {
        // Deallocate stack
        let frame_size = frame.frame_size();
        if frame_size > 0 {
            asm.add_ri(Gpr::Rsp, frame_size);
        }

        // Restore frame pointer
        asm.pop(Gpr::Rbp);

        // Pop callee-saved registers in reverse
        let saved_regs: Vec<Gpr> = frame.saved_regs.iter().collect();
        for reg in saved_regs.into_iter().rev() {
            asm.pop(reg);
        }

        // Return
        asm.ret();
    }

    /// Emit code for a single instruction.
    fn emit_instruction(
        &self,
        ctx: &mut TemplateContext,
        instr: &TemplateInstruction,
        labels: &HashMap<u32, Label>,
        deopt_idx: usize,
    ) {
        match instr {
            TemplateInstruction::LoadInt { dst, value, .. } => {
                LoadIntTemplate {
                    dst_reg: *dst,
                    value: *value,
                }
                .emit(ctx);
            }
            TemplateInstruction::LoadFloat { dst, value, .. } => {
                LoadFloatTemplate {
                    dst_reg: *dst,
                    value: *value,
                }
                .emit(ctx);
            }
            TemplateInstruction::LoadNone { dst, .. } => {
                LoadNoneTemplate { dst_reg: *dst }.emit(ctx);
            }
            TemplateInstruction::LoadBool { dst, value, .. } => {
                LoadBoolTemplate {
                    dst_reg: *dst,
                    value: *value,
                }
                .emit(ctx);
            }
            TemplateInstruction::Move { dst, src, .. } => {
                MoveTemplate {
                    dst_reg: *dst,
                    src_reg: *src,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntAdd { dst, lhs, rhs, .. } => {
                IntAddTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntSub { dst, lhs, rhs, .. } => {
                IntSubTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntMul { dst, lhs, rhs, .. } => {
                IntMulTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatAdd { dst, lhs, rhs, .. } => {
                FloatAddTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatSub { dst, lhs, rhs, .. } => {
                FloatSubTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatMul { dst, lhs, rhs, .. } => {
                FloatMulTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatDiv { dst, lhs, rhs, .. } => {
                FloatDivTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::Jump { target, .. } => {
                if let Some(label) = labels.get(target) {
                    JumpTemplate { target: *label }.emit(ctx);
                }
            }
            TemplateInstruction::BranchIfTrue { cond, target, .. } => {
                if let Some(label) = labels.get(target) {
                    BranchIfTrueTemplate {
                        condition_reg: *cond,
                        target: *label,
                        deopt_idx,
                    }
                    .emit(ctx);
                }
            }
            TemplateInstruction::BranchIfFalse { cond, target, .. } => {
                if let Some(label) = labels.get(target) {
                    BranchIfFalseTemplate {
                        condition_reg: *cond,
                        target: *label,
                        deopt_idx,
                    }
                    .emit(ctx);
                }
            }
            TemplateInstruction::Return { value, .. } => {
                ReturnTemplate { value_reg: *value }.emit(ctx);
            }
            TemplateInstruction::Nop { .. } => {
                // No-op: emit nothing
            }
        }
    }
}

// =============================================================================
// Template Instruction (IR for template compilation)
// =============================================================================

/// Intermediate representation for template-based compilation.
/// This is a simplified IR that maps 1:1 to bytecode operations.
#[derive(Debug, Clone)]
pub enum TemplateInstruction {
    // Value loading
    LoadInt {
        bc_offset: u32,
        dst: u8,
        value: i64,
    },
    LoadFloat {
        bc_offset: u32,
        dst: u8,
        value: f64,
    },
    LoadNone {
        bc_offset: u32,
        dst: u8,
    },
    LoadBool {
        bc_offset: u32,
        dst: u8,
        value: bool,
    },

    // Register operations
    Move {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },

    // Integer arithmetic
    IntAdd {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntSub {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntMul {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },

    // Float arithmetic
    FloatAdd {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatSub {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatMul {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatDiv {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },

    // Control flow
    Jump {
        bc_offset: u32,
        target: u32,
    },
    BranchIfTrue {
        bc_offset: u32,
        cond: u8,
        target: u32,
    },
    BranchIfFalse {
        bc_offset: u32,
        cond: u8,
        target: u32,
    },
    Return {
        bc_offset: u32,
        value: u8,
    },

    // No-op
    Nop {
        bc_offset: u32,
    },
}

impl TemplateInstruction {
    /// Get the bytecode offset of this instruction.
    pub fn bc_offset(&self) -> u32 {
        match self {
            TemplateInstruction::LoadInt { bc_offset, .. } => *bc_offset,
            TemplateInstruction::LoadFloat { bc_offset, .. } => *bc_offset,
            TemplateInstruction::LoadNone { bc_offset, .. } => *bc_offset,
            TemplateInstruction::LoadBool { bc_offset, .. } => *bc_offset,
            TemplateInstruction::Move { bc_offset, .. } => *bc_offset,
            TemplateInstruction::IntAdd { bc_offset, .. } => *bc_offset,
            TemplateInstruction::IntSub { bc_offset, .. } => *bc_offset,
            TemplateInstruction::IntMul { bc_offset, .. } => *bc_offset,
            TemplateInstruction::FloatAdd { bc_offset, .. } => *bc_offset,
            TemplateInstruction::FloatSub { bc_offset, .. } => *bc_offset,
            TemplateInstruction::FloatMul { bc_offset, .. } => *bc_offset,
            TemplateInstruction::FloatDiv { bc_offset, .. } => *bc_offset,
            TemplateInstruction::Jump { bc_offset, .. } => *bc_offset,
            TemplateInstruction::BranchIfTrue { bc_offset, .. } => *bc_offset,
            TemplateInstruction::BranchIfFalse { bc_offset, .. } => *bc_offset,
            TemplateInstruction::Return { bc_offset, .. } => *bc_offset,
            TemplateInstruction::Nop { bc_offset } => *bc_offset,
        }
    }

    /// Get the jump target if this is a branch/jump instruction.
    pub fn jump_target(&self) -> Option<u32> {
        match self {
            TemplateInstruction::Jump { target, .. } => Some(*target),
            TemplateInstruction::BranchIfTrue { target, .. } => Some(*target),
            TemplateInstruction::BranchIfFalse { target, .. } => Some(*target),
            _ => None,
        }
    }

    /// Check if this instruction can trigger deoptimization.
    pub fn can_deopt(&self) -> bool {
        matches!(
            self,
            TemplateInstruction::IntAdd { .. }
                | TemplateInstruction::IntSub { .. }
                | TemplateInstruction::IntMul { .. }
                | TemplateInstruction::BranchIfTrue { .. }
                | TemplateInstruction::BranchIfFalse { .. }
        )
    }

    /// Get the deopt reason for this instruction.
    pub fn deopt_reason(&self) -> DeoptReason {
        match self {
            TemplateInstruction::IntAdd { .. }
            | TemplateInstruction::IntSub { .. }
            | TemplateInstruction::IntMul { .. } => DeoptReason::TypeGuardFailed,
            TemplateInstruction::BranchIfTrue { .. }
            | TemplateInstruction::BranchIfFalse { .. } => DeoptReason::TypeGuardFailed,
            _ => DeoptReason::UncommonTrap,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_empty() {
        let compiler = TemplateCompiler::new_for_testing();
        let result = compiler.compile(4, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_load_int() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadInt {
                bc_offset: 0,
                dst: 0,
                value: 42,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 0,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());

        let func = result.unwrap();
        assert!(func.code.len() > 0);
        assert!(func.bc_to_native.contains_key(&0));
        assert!(func.bc_to_native.contains_key(&4));
    }

    #[test]
    fn test_compile_arithmetic() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadInt {
                bc_offset: 0,
                dst: 0,
                value: 10,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 4,
                dst: 1,
                value: 20,
            },
            TemplateInstruction::IntAdd {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 12,
                value: 2,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());

        let func = result.unwrap();
        // Should have deopt info for IntAdd
        assert!(!func.deopt_info.is_empty());
    }

    #[test]
    fn test_compile_float_arithmetic() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadFloat {
                bc_offset: 0,
                dst: 0,
                value: 1.5,
            },
            TemplateInstruction::LoadFloat {
                bc_offset: 4,
                dst: 1,
                value: 2.5,
            },
            TemplateInstruction::FloatAdd {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::FloatMul {
                bc_offset: 12,
                dst: 3,
                lhs: 2,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 16,
                value: 3,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_jumps() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadBool {
                bc_offset: 0,
                dst: 0,
                value: true,
            },
            TemplateInstruction::BranchIfTrue {
                bc_offset: 4,
                cond: 0,
                target: 12,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 8,
                dst: 1,
                value: 1,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 12,
                dst: 1,
                value: 2,
            },
            TemplateInstruction::Return {
                bc_offset: 16,
                value: 1,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_template_instruction_properties() {
        let jump = TemplateInstruction::Jump {
            bc_offset: 0,
            target: 100,
        };
        assert_eq!(jump.bc_offset(), 0);
        assert_eq!(jump.jump_target(), Some(100));
        assert!(!jump.can_deopt());

        let add = TemplateInstruction::IntAdd {
            bc_offset: 4,
            dst: 0,
            lhs: 1,
            rhs: 2,
        };
        assert_eq!(add.bc_offset(), 4);
        assert_eq!(add.jump_target(), None);
        assert!(add.can_deopt());
        assert_eq!(add.deopt_reason(), DeoptReason::TypeGuardFailed);
    }
}
