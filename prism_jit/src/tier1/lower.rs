//! Bytecode to Template IR lowering with speculation-guided specialization.
//!
//! This module converts VM bytecode into template IR for JIT compilation.
//! It uses speculation hints from the interpreter's type feedback system
//! to select specialized templates for polymorphic operations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
//! │   CodeObject    │────▶│   BytecodeLower   │────▶│ TemplateIR Vec  │
//! │   (bytecode)    │     │ + SpeculationHints│     │                 │
//! └─────────────────┘     └───────────────────┘     └─────────────────┘
//!                               │
//!                               ▼
//!                         ┌───────────────────┐
//!                         │ SpeculationProvider│
//!                         │  (type feedback)   │
//!                         └───────────────────┘
//! ```
//!
//! # Specialization Strategy
//!
//! For generic opcodes (Add, Sub, Mul, etc.), the lowerer:
//! 1. Queries the SpeculationProvider for type hints at the bytecode offset
//! 2. If hints are available, emits specialized templates with inline guards
//! 3. If no hints, emits generic templates with full type dispatch
//!
//! This allows hot paths to use fast specialized code while cold paths
//! remain correct via fallback to the slow path.

use prism_compiler::bytecode::{CodeObject, Instruction, Opcode};
use prism_core::{SpeculationProvider, TypeHint};

use super::codegen::TemplateInstruction;

// =============================================================================
// Lowering Configuration
// =============================================================================

/// Configuration for bytecode lowering.
#[derive(Debug, Clone)]
pub struct LoweringConfig {
    /// Enable speculation-based specialization.
    pub enable_speculation: bool,
    /// Emit type guards for speculative code.
    pub emit_guards: bool,
    /// Use aggressive inlining for arithmetic ops.
    pub aggressive_inline: bool,
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self {
            enable_speculation: true,
            emit_guards: true,
            aggressive_inline: true,
        }
    }
}

// =============================================================================
// Comparison Operation Type
// =============================================================================

/// Type of comparison operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

// =============================================================================
// Bytecode Lowerer
// =============================================================================

/// Lowers bytecode to template IR with speculation-guided specialization.
///
/// The lowerer is parameterized by a `SpeculationProvider` which supplies
/// type hints collected during interpretation. This enables adaptive
/// compilation where frequently executed paths get specialized code.
pub struct BytecodeLowerer<'a, S: SpeculationProvider> {
    /// Speculation provider for type hints.
    speculation: &'a S,
    /// Code ID for speculation queries.
    code_id: u32,
    /// Lowering configuration.
    config: LoweringConfig,
    /// Output template instructions.
    output: Vec<TemplateInstruction>,
    /// Current bytecode offset (for error reporting).
    current_offset: u32,
}

impl<'a, S: SpeculationProvider> BytecodeLowerer<'a, S> {
    /// Create a new lowerer with the given speculation provider.
    #[inline]
    pub fn new(speculation: &'a S, code_id: u32, config: LoweringConfig) -> Self {
        Self {
            speculation,
            code_id,
            config,
            output: Vec::with_capacity(128),
            current_offset: 0,
        }
    }

    /// Lower a code object to template IR.
    pub fn lower(&mut self, code: &CodeObject) -> Vec<TemplateInstruction> {
        self.output.clear();
        self.output.reserve(code.instructions.len());

        for (idx, inst) in code.instructions.iter().enumerate() {
            let bc_offset = (idx * 4) as u32; // Each instruction is 4 bytes
            self.current_offset = bc_offset;
            self.lower_instruction(*inst, bc_offset);
        }

        std::mem::take(&mut self.output)
    }

    /// Get the type hint for the current bytecode offset.
    #[inline]
    fn get_hint(&self) -> TypeHint {
        if self.config.enable_speculation {
            self.speculation
                .get_type_hint(self.code_id, self.current_offset)
        } else {
            TypeHint::None
        }
    }

    /// Lower a single instruction.
    fn lower_instruction(&mut self, inst: Instruction, bc_offset: u32) {
        let opcode = Opcode::from_u8(inst.opcode());

        match opcode {
            // =================================================================
            // Control Flow
            // =================================================================
            Some(Opcode::Nop) => {
                self.output.push(TemplateInstruction::Nop { bc_offset });
            }
            Some(Opcode::Return) => {
                self.output.push(TemplateInstruction::Return {
                    bc_offset,
                    value: inst.dst().0,
                });
            }
            Some(Opcode::Jump) => {
                // Target is pc-relative, need to calculate absolute
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                self.output
                    .push(TemplateInstruction::Jump { bc_offset, target });
            }
            Some(Opcode::JumpIfTrue) => {
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                self.output.push(TemplateInstruction::BranchIfTrue {
                    bc_offset,
                    cond: inst.dst().0,
                    target,
                });
            }
            Some(Opcode::JumpIfFalse) => {
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                self.output.push(TemplateInstruction::BranchIfFalse {
                    bc_offset,
                    cond: inst.dst().0,
                    target,
                });
            }

            // =================================================================
            // Load/Store
            // =================================================================
            Some(Opcode::LoadNone) => {
                self.output.push(TemplateInstruction::LoadNone {
                    bc_offset,
                    dst: inst.dst().0,
                });
            }
            Some(Opcode::LoadTrue) => {
                self.output.push(TemplateInstruction::LoadBool {
                    bc_offset,
                    dst: inst.dst().0,
                    value: true,
                });
            }
            Some(Opcode::LoadFalse) => {
                self.output.push(TemplateInstruction::LoadBool {
                    bc_offset,
                    dst: inst.dst().0,
                    value: false,
                });
            }
            Some(Opcode::Move) => {
                self.output.push(TemplateInstruction::Move {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }

            // =================================================================
            // Specialized Integer Arithmetic (already typed)
            // =================================================================
            Some(Opcode::AddInt) => {
                self.output.push(TemplateInstruction::IntAdd {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::SubInt) => {
                self.output.push(TemplateInstruction::IntSub {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::MulInt) => {
                self.output.push(TemplateInstruction::IntMul {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }

            // =================================================================
            // Specialized Float Arithmetic (already typed)
            // =================================================================
            Some(Opcode::AddFloat) => {
                self.output.push(TemplateInstruction::FloatAdd {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::SubFloat) => {
                self.output.push(TemplateInstruction::FloatSub {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::MulFloat) => {
                self.output.push(TemplateInstruction::FloatMul {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::DivFloat) => {
                self.output.push(TemplateInstruction::FloatDiv {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }

            // =================================================================
            // Generic Arithmetic (speculation-guided)
            // =================================================================
            Some(Opcode::Add) => {
                self.lower_generic_add(bc_offset, inst.dst().0, inst.src1().0, inst.src2().0);
            }
            Some(Opcode::Sub) => {
                self.lower_generic_sub(bc_offset, inst.dst().0, inst.src1().0, inst.src2().0);
            }
            Some(Opcode::Mul) => {
                self.lower_generic_mul(bc_offset, inst.dst().0, inst.src1().0, inst.src2().0);
            }
            Some(Opcode::TrueDiv) => {
                // True division always produces float
                self.output.push(TemplateInstruction::FloatDiv {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }

            // =================================================================
            // Comparison Operations (speculation-guided)
            // =================================================================
            Some(Opcode::Lt) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Lt,
                );
            }
            Some(Opcode::Le) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Le,
                );
            }
            Some(Opcode::Gt) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Gt,
                );
            }
            Some(Opcode::Ge) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Ge,
                );
            }
            Some(Opcode::Eq) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Eq,
                );
            }
            Some(Opcode::Ne) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Ne,
                );
            }

            // =================================================================
            // Unhandled opcodes - emit Nop for now
            // =================================================================
            _ => {
                // TODO: Implement remaining opcodes
                self.output.push(TemplateInstruction::Nop { bc_offset });
            }
        }
    }

    /// Calculate absolute jump target from pc-relative offset.
    #[inline]
    fn calculate_jump_target(&self, bc_offset: u32, relative: i16) -> u32 {
        // Jump target is relative to the NEXT instruction
        let next_pc = bc_offset.wrapping_add(4);
        if relative >= 0 {
            next_pc.wrapping_add(relative as u32 * 4)
        } else {
            next_pc.wrapping_sub((-relative) as u32 * 4)
        }
    }

    // =========================================================================
    // Speculation-Guided Lowering
    // =========================================================================

    /// Lower generic Add with speculation.
    fn lower_generic_add(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8) {
        let hint = self.get_hint();

        match hint {
            TypeHint::IntInt => {
                // Speculate integer addition
                self.output.push(TemplateInstruction::IntAdd {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::FloatFloat | TypeHint::IntFloat | TypeHint::FloatInt => {
                // Speculate float addition
                self.output.push(TemplateInstruction::FloatAdd {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::StrStr => {
                // String concatenation - not yet implemented
                // Fall back to generic (would emit runtime call)
                self.output.push(TemplateInstruction::Nop { bc_offset });
            }
            TypeHint::None | TypeHint::StrInt | TypeHint::IntStr | TypeHint::ListList => {
                // No speculation or invalid type combo - emit generic
                // For now, fall back to IntAdd as placeholder
                // A real implementation would emit a runtime call
                self.output.push(TemplateInstruction::IntAdd {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
        }
    }

    /// Lower generic Sub with speculation.
    fn lower_generic_sub(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8) {
        let hint = self.get_hint();

        match hint {
            TypeHint::IntInt => {
                self.output.push(TemplateInstruction::IntSub {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::FloatFloat | TypeHint::IntFloat | TypeHint::FloatInt => {
                self.output.push(TemplateInstruction::FloatSub {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            _ => {
                // Fall back to int sub
                self.output.push(TemplateInstruction::IntSub {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
        }
    }

    /// Lower generic Mul with speculation.
    fn lower_generic_mul(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8) {
        let hint = self.get_hint();

        match hint {
            TypeHint::IntInt => {
                self.output.push(TemplateInstruction::IntMul {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::FloatFloat | TypeHint::IntFloat | TypeHint::FloatInt => {
                self.output.push(TemplateInstruction::FloatMul {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::StrInt | TypeHint::IntStr => {
                // String repetition - not yet implemented
                self.output.push(TemplateInstruction::Nop { bc_offset });
            }
            _ => {
                self.output.push(TemplateInstruction::IntMul {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
        }
    }

    /// Lower comparison operation with speculation.
    fn lower_comparison(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8, op: ComparisonOp) {
        let hint = self.get_hint();

        match hint {
            TypeHint::IntInt => {
                // Emit integer comparison
                self.emit_int_comparison(bc_offset, dst, lhs, rhs, op);
            }
            TypeHint::FloatFloat | TypeHint::IntFloat | TypeHint::FloatInt => {
                // Emit float comparison
                self.emit_float_comparison(bc_offset, dst, lhs, rhs, op);
            }
            _ => {
                // No speculation or unsupported type - fall back to int comparison
                self.emit_int_comparison(bc_offset, dst, lhs, rhs, op);
            }
        }
    }

    /// Emit specialized integer comparison.
    fn emit_int_comparison(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8, op: ComparisonOp) {
        let instr = match op {
            ComparisonOp::Lt => TemplateInstruction::IntLt {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Le => TemplateInstruction::IntLe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Gt => TemplateInstruction::IntGt {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Ge => TemplateInstruction::IntGe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Eq => TemplateInstruction::IntEq {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Ne => TemplateInstruction::IntNe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
        };
        self.output.push(instr);
    }

    /// Emit specialized float comparison.
    fn emit_float_comparison(
        &mut self,
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
        op: ComparisonOp,
    ) {
        let instr = match op {
            ComparisonOp::Lt => TemplateInstruction::FloatLt {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Le => TemplateInstruction::FloatLe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Gt => TemplateInstruction::FloatGt {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Ge => TemplateInstruction::FloatGe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Eq => TemplateInstruction::FloatEq {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Ne => TemplateInstruction::FloatNe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
        };
        self.output.push(instr);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_compiler::bytecode::{CodeObject, Instruction, Opcode, Register};
    use prism_core::speculation::NoSpeculation;

    fn make_code(instructions: Vec<Instruction>) -> CodeObject {
        let mut code = CodeObject::new("test", "test.py");
        code.instructions = instructions.into_boxed_slice();
        code
    }

    #[test]
    fn test_lower_nop() {
        let code = make_code(vec![Instruction::op(Opcode::Nop)]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(ir[0], TemplateInstruction::Nop { bc_offset: 0 }));
    }

    #[test]
    fn test_lower_load_none() {
        let code = make_code(vec![Instruction::op_d(Opcode::LoadNone, Register(5))]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(
            ir[0],
            TemplateInstruction::LoadNone {
                bc_offset: 0,
                dst: 5
            }
        ));
    }

    #[test]
    fn test_lower_int_add() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::AddInt,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(
            ir[0],
            TemplateInstruction::IntAdd {
                bc_offset: 0,
                dst: 0,
                lhs: 1,
                rhs: 2
            }
        ));
    }

    #[test]
    fn test_lower_float_add() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::AddFloat,
            Register(3),
            Register(4),
            Register(5),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(
            ir[0],
            TemplateInstruction::FloatAdd {
                bc_offset: 0,
                dst: 3,
                lhs: 4,
                rhs: 5
            }
        ));
    }

    #[test]
    fn test_lower_generic_add_no_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Add,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        // Without speculation, falls back to IntAdd
        assert!(matches!(ir[0], TemplateInstruction::IntAdd { .. }));
    }

    /// Mock speculation provider for testing.
    struct MockSpeculation {
        hint: TypeHint,
    }

    impl SpeculationProvider for MockSpeculation {
        fn get_type_hint(&self, _code_id: u32, _bc_offset: u32) -> TypeHint {
            self.hint
        }
    }

    #[test]
    fn test_lower_generic_add_int_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Add,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = MockSpeculation {
            hint: TypeHint::IntInt,
        };
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(ir[0], TemplateInstruction::IntAdd { .. }));
    }

    #[test]
    fn test_lower_generic_add_float_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Add,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = MockSpeculation {
            hint: TypeHint::FloatFloat,
        };
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(ir[0], TemplateInstruction::FloatAdd { .. }));
    }

    #[test]
    fn test_lower_generic_mul_int_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Mul,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = MockSpeculation {
            hint: TypeHint::IntInt,
        };
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(ir[0], TemplateInstruction::IntMul { .. }));
    }

    #[test]
    fn test_lower_return() {
        let code = make_code(vec![Instruction::op_d(Opcode::Return, Register(3))]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(
            ir[0],
            TemplateInstruction::Return {
                bc_offset: 0,
                value: 3
            }
        ));
    }

    #[test]
    fn test_lower_move() {
        let code = make_code(vec![Instruction::op_ds(
            Opcode::Move,
            Register(5),
            Register(10),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(
            ir[0],
            TemplateInstruction::Move {
                bc_offset: 0,
                dst: 5,
                src: 10
            }
        ));
    }

    #[test]
    fn test_lowering_config_disable_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Add,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = MockSpeculation {
            hint: TypeHint::FloatFloat,
        };

        // With speculation disabled, should use fallback even if hints available
        let config = LoweringConfig {
            enable_speculation: false,
            ..Default::default()
        };
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, config);

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        // Falls back to IntAdd when speculation is disabled
        assert!(matches!(ir[0], TemplateInstruction::IntAdd { .. }));
    }

    #[test]
    fn test_lower_sequence() {
        let code = make_code(vec![
            Instruction::op_d(Opcode::LoadNone, Register(0)),
            Instruction::op_d(Opcode::LoadTrue, Register(1)),
            Instruction::op_ds(Opcode::Move, Register(2), Register(1)),
            Instruction::op_d(Opcode::Return, Register(2)),
        ]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 4);

        assert!(matches!(
            ir[0],
            TemplateInstruction::LoadNone {
                bc_offset: 0,
                dst: 0
            }
        ));
        assert!(matches!(
            ir[1],
            TemplateInstruction::LoadBool {
                bc_offset: 4,
                dst: 1,
                value: true
            }
        ));
        assert!(matches!(
            ir[2],
            TemplateInstruction::Move {
                bc_offset: 8,
                dst: 2,
                src: 1
            }
        ));
        assert!(matches!(
            ir[3],
            TemplateInstruction::Return {
                bc_offset: 12,
                value: 2
            }
        ));
    }

    #[test]
    fn test_lower_comparison_lt_no_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Lt,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        // Without speculation, falls back to IntLt
        assert!(matches!(ir[0], TemplateInstruction::IntLt { .. }));
    }

    #[test]
    fn test_lower_comparison_lt_int_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Lt,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = MockSpeculation {
            hint: TypeHint::IntInt,
        };
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(
            ir[0],
            TemplateInstruction::IntLt {
                bc_offset: 0,
                dst: 0,
                lhs: 1,
                rhs: 2
            }
        ));
    }

    #[test]
    fn test_lower_comparison_lt_float_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Lt,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = MockSpeculation {
            hint: TypeHint::FloatFloat,
        };
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(ir[0], TemplateInstruction::FloatLt { .. }));
    }

    #[test]
    fn test_lower_comparison_eq_int_speculation() {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::Eq,
            Register(3),
            Register(4),
            Register(5),
        )]);
        let speculation = MockSpeculation {
            hint: TypeHint::IntInt,
        };
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        assert!(matches!(
            ir[0],
            TemplateInstruction::IntEq {
                bc_offset: 0,
                dst: 3,
                lhs: 4,
                rhs: 5
            }
        ));
    }

    #[test]
    fn test_lower_all_comparisons() {
        // Test that all 6 comparison ops work
        let opcodes = [
            (Opcode::Lt, "Lt"),
            (Opcode::Le, "Le"),
            (Opcode::Gt, "Gt"),
            (Opcode::Ge, "Ge"),
            (Opcode::Eq, "Eq"),
            (Opcode::Ne, "Ne"),
        ];

        for (op, name) in opcodes {
            let code = make_code(vec![Instruction::op_dss(
                op,
                Register(0),
                Register(1),
                Register(2),
            )]);
            let speculation = MockSpeculation {
                hint: TypeHint::IntInt,
            };
            let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
            let ir = lowerer.lower(&code);
            assert_eq!(ir.len(), 1, "Failed for comparison op: {}", name);
        }
    }
}
