use super::*;

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

// =========================================================================
// Integer Floor Division Tests
// =========================================================================

#[test]
fn test_lower_floor_div_int() {
    let code = make_code(vec![Instruction::op_dss(
        Opcode::FloorDivInt,
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
        TemplateInstruction::IntDiv {
            bc_offset: 0,
            dst: 0,
            lhs: 1,
            rhs: 2
        }
    ));
}

#[test]
fn test_lower_floor_div_int_different_registers() {
    let code = make_code(vec![Instruction::op_dss(
        Opcode::FloorDivInt,
        Register(5),
        Register(10),
        Register(15),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntDiv {
            bc_offset: 0,
            dst: 5,
            lhs: 10,
            rhs: 15
        }
    ));
}

// =========================================================================
// Integer Modulo Tests
// =========================================================================

#[test]
fn test_lower_mod_int() {
    let code = make_code(vec![Instruction::op_dss(
        Opcode::ModInt,
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
        TemplateInstruction::IntMod {
            bc_offset: 0,
            dst: 0,
            lhs: 1,
            rhs: 2
        }
    ));
}

#[test]
fn test_lower_mod_int_different_registers() {
    let code = make_code(vec![Instruction::op_dss(
        Opcode::ModInt,
        Register(7),
        Register(8),
        Register(9),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntMod {
            bc_offset: 0,
            dst: 7,
            lhs: 8,
            rhs: 9
        }
    ));
}

// =========================================================================
// Integer Negation Tests
// =========================================================================

#[test]
fn test_lower_neg_int() {
    let code = make_code(vec![Instruction::op_ds(
        Opcode::NegInt,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntNeg {
            bc_offset: 0,
            dst: 0,
            src: 1
        }
    ));
}

#[test]
fn test_lower_neg_int_same_register() {
    // Test negation in-place: r0 = -r0
    let code = make_code(vec![Instruction::op_ds(
        Opcode::NegInt,
        Register(3),
        Register(3),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntNeg {
            bc_offset: 0,
            dst: 3,
            src: 3
        }
    ));
}

// =========================================================================
// Integer Positive (no-op) Tests
// =========================================================================

#[test]
fn test_lower_pos_int() {
    let code = make_code(vec![Instruction::op_ds(
        Opcode::PosInt,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    // PosInt should lower to a Move since +x == x for integers
    assert!(matches!(
        ir[0],
        TemplateInstruction::Move {
            bc_offset: 0,
            dst: 0,
            src: 1
        }
    ));
}

#[test]
fn test_lower_pos_int_same_register() {
    // +r0 stored to r0 should emit Move r0 <- r0 which is a no-op
    let code = make_code(vec![Instruction::op_ds(
        Opcode::PosInt,
        Register(5),
        Register(5),
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
            src: 5
        }
    ));
}

// =========================================================================
// Combined Integer Arithmetic Sequence Tests
// =========================================================================

#[test]
fn test_lower_arithmetic_sequence() {
    // Test a sequence of arithmetic operations
    let code = make_code(vec![
        Instruction::op_dss(Opcode::AddInt, Register(0), Register(1), Register(2)),
        Instruction::op_dss(Opcode::SubInt, Register(3), Register(0), Register(4)),
        Instruction::op_dss(Opcode::MulInt, Register(5), Register(3), Register(6)),
        Instruction::op_dss(Opcode::FloorDivInt, Register(7), Register(5), Register(8)),
        Instruction::op_dss(Opcode::ModInt, Register(9), Register(7), Register(10)),
        Instruction::op_ds(Opcode::NegInt, Register(11), Register(9)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 6);
    assert!(matches!(ir[0], TemplateInstruction::IntAdd { .. }));
    assert!(matches!(ir[1], TemplateInstruction::IntSub { .. }));
    assert!(matches!(ir[2], TemplateInstruction::IntMul { .. }));
    assert!(matches!(ir[3], TemplateInstruction::IntDiv { .. }));
    assert!(matches!(ir[4], TemplateInstruction::IntMod { .. }));
    assert!(matches!(ir[5], TemplateInstruction::IntNeg { .. }));
}

#[test]
fn test_lower_all_specialized_int_ops() {
    // Test that all specialized int ops work
    let opcodes = [
        (Opcode::AddInt, "AddInt"),
        (Opcode::SubInt, "SubInt"),
        (Opcode::MulInt, "MulInt"),
        (Opcode::FloorDivInt, "FloorDivInt"),
        (Opcode::ModInt, "ModInt"),
    ];

    for (op, name) in opcodes {
        let code = make_code(vec![Instruction::op_dss(
            op,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for int op: {}", name);
    }
}

// =========================================================================
// Float Arithmetic Tests (Phase 2)
// =========================================================================

#[test]
fn test_lower_floor_div_float() {
    let code = make_code(vec![Instruction::op_dss(
        Opcode::FloorDivFloat,
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
        TemplateInstruction::FloatFloorDiv {
            bc_offset: 0,
            dst: 0,
            lhs: 1,
            rhs: 2
        }
    ));
}

#[test]
fn test_lower_mod_float() {
    let code = make_code(vec![Instruction::op_dss(
        Opcode::ModFloat,
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
        TemplateInstruction::FloatMod {
            bc_offset: 0,
            dst: 0,
            lhs: 1,
            rhs: 2
        }
    ));
}

#[test]
fn test_lower_neg_float() {
    let code = make_code(vec![Instruction::op_ds(
        Opcode::NegFloat,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::FloatNeg {
            bc_offset: 0,
            dst: 0,
            src: 1
        }
    ));
}

#[test]
fn test_lower_floor_div_float_different_registers() {
    // Test with different register combinations
    let code = make_code(vec![Instruction::op_dss(
        Opcode::FloorDivFloat,
        Register(10),
        Register(20),
        Register(30),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::FloatFloorDiv {
            bc_offset: 0,
            dst: 10,
            lhs: 20,
            rhs: 30
        }
    ));
}

#[test]
fn test_lower_mod_float_same_register() {
    // Test self-modulo (x = x % y)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::ModFloat,
        Register(5),
        Register(5),
        Register(7),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::FloatMod {
            bc_offset: 0,
            dst: 5,
            lhs: 5,
            rhs: 7
        }
    ));
}

#[test]
fn test_lower_float_arithmetic_sequence() {
    // Test a sequence of float arithmetic operations
    let code = make_code(vec![
        Instruction::op_dss(Opcode::AddFloat, Register(0), Register(1), Register(2)),
        Instruction::op_dss(Opcode::SubFloat, Register(3), Register(0), Register(4)),
        Instruction::op_dss(Opcode::MulFloat, Register(5), Register(3), Register(6)),
        Instruction::op_dss(Opcode::DivFloat, Register(7), Register(5), Register(8)),
        Instruction::op_dss(
            Opcode::FloorDivFloat,
            Register(9),
            Register(7),
            Register(10),
        ),
        Instruction::op_dss(Opcode::ModFloat, Register(11), Register(9), Register(12)),
        Instruction::op_ds(Opcode::NegFloat, Register(13), Register(11)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 7);
    assert!(matches!(ir[0], TemplateInstruction::FloatAdd { .. }));
    assert!(matches!(ir[1], TemplateInstruction::FloatSub { .. }));
    assert!(matches!(ir[2], TemplateInstruction::FloatMul { .. }));
    assert!(matches!(ir[3], TemplateInstruction::FloatDiv { .. }));
    assert!(matches!(ir[4], TemplateInstruction::FloatFloorDiv { .. }));
    assert!(matches!(ir[5], TemplateInstruction::FloatMod { .. }));
    assert!(matches!(ir[6], TemplateInstruction::FloatNeg { .. }));
}

#[test]
fn test_lower_all_specialized_float_ops() {
    // Test that all specialized float ops work with binary operations
    let binary_float_opcodes = [
        (Opcode::AddFloat, "AddFloat"),
        (Opcode::SubFloat, "SubFloat"),
        (Opcode::MulFloat, "MulFloat"),
        (Opcode::DivFloat, "DivFloat"),
        (Opcode::FloorDivFloat, "FloorDivFloat"),
        (Opcode::ModFloat, "ModFloat"),
    ];

    for (op, name) in binary_float_opcodes {
        let code = make_code(vec![Instruction::op_dss(
            op,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for float op: {}", name);
    }
}

#[test]
fn test_lower_neg_float_in_place() {
    // Test negation with same src and dst (x = -x)
    let code = make_code(vec![Instruction::op_ds(
        Opcode::NegFloat,
        Register(8),
        Register(8),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::FloatNeg {
            bc_offset: 0,
            dst: 8,
            src: 8
        }
    ));
}

#[test]
fn test_lower_mixed_int_float_sequence() {
    // Test mixed integer and float operations in sequence
    let code = make_code(vec![
        Instruction::op_dss(Opcode::AddInt, Register(0), Register(1), Register(2)),
        Instruction::op_dss(Opcode::AddFloat, Register(3), Register(4), Register(5)),
        Instruction::op_dss(Opcode::FloorDivInt, Register(6), Register(0), Register(7)),
        Instruction::op_dss(Opcode::FloorDivFloat, Register(8), Register(3), Register(9)),
        Instruction::op_ds(Opcode::NegInt, Register(10), Register(6)),
        Instruction::op_ds(Opcode::NegFloat, Register(11), Register(8)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 6);
    assert!(matches!(ir[0], TemplateInstruction::IntAdd { .. }));
    assert!(matches!(ir[1], TemplateInstruction::FloatAdd { .. }));
    assert!(matches!(ir[2], TemplateInstruction::IntDiv { .. }));
    assert!(matches!(ir[3], TemplateInstruction::FloatFloorDiv { .. }));
    assert!(matches!(ir[4], TemplateInstruction::IntNeg { .. }));
    assert!(matches!(ir[5], TemplateInstruction::FloatNeg { .. }));
}

// =========================================================================
// Load/Store Local Tests (Phase 3)
// =========================================================================

#[test]
fn test_lower_load_local() {
    let code = make_code(vec![Instruction::op_di(Opcode::LoadLocal, Register(0), 42)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadLocal {
            bc_offset: 0,
            dst: 0,
            slot: 42
        }
    ));
}

#[test]
fn test_lower_store_local() {
    let code = make_code(vec![Instruction::op_di(
        Opcode::StoreLocal,
        Register(5),
        100,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::StoreLocal {
            bc_offset: 0,
            src: 5,
            slot: 100
        }
    ));
}

#[test]
fn test_lower_delete_local() {
    let code = make_code(vec![Instruction::op_di(
        Opcode::DeleteLocal,
        Register(0),
        7,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::DeleteLocal {
            bc_offset: 0,
            slot: 7
        }
    ));
}

// =========================================================================
// Load/Store Global Tests (Phase 3)
// =========================================================================

#[test]
fn test_lower_load_global() {
    let code = make_code(vec![Instruction::op_di(
        Opcode::LoadGlobal,
        Register(3),
        25,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadGlobal {
            bc_offset: 0,
            dst: 3,
            name_idx: 25
        }
    ));
}

#[test]
fn test_lower_store_global() {
    let code = make_code(vec![Instruction::op_di(
        Opcode::StoreGlobal,
        Register(8),
        99,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::StoreGlobal {
            bc_offset: 0,
            src: 8,
            name_idx: 99
        }
    ));
}

#[test]
fn test_lower_delete_global() {
    let code = make_code(vec![Instruction::op_di(
        Opcode::DeleteGlobal,
        Register(0),
        15,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::DeleteGlobal {
            bc_offset: 0,
            name_idx: 15
        }
    ));
}

#[test]
fn test_lower_load_store_local_sequence() {
    // x = locals[0]; y = x; locals[1] = y
    let code = make_code(vec![
        Instruction::op_di(Opcode::LoadLocal, Register(0), 0),
        Instruction::op_ds(Opcode::Move, Register(1), Register(0)),
        Instruction::op_di(Opcode::StoreLocal, Register(1), 1),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadLocal {
            dst: 0,
            slot: 0,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::Move { dst: 1, src: 0, .. }
    ));
    assert!(matches!(
        ir[2],
        TemplateInstruction::StoreLocal {
            src: 1,
            slot: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_global_ops_sequence() {
    // x = globals[name0]; y = x + 1; globals[name1] = y
    let code = make_code(vec![
        Instruction::op_di(Opcode::LoadGlobal, Register(0), 0),
        Instruction::op_dss(Opcode::AddInt, Register(1), Register(0), Register(2)),
        Instruction::op_di(Opcode::StoreGlobal, Register(1), 1),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadGlobal {
            dst: 0,
            name_idx: 0,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::IntAdd {
            dst: 1,
            lhs: 0,
            rhs: 2,
            ..
        }
    ));
    assert!(matches!(
        ir[2],
        TemplateInstruction::StoreGlobal {
            src: 1,
            name_idx: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_local_high_slot() {
    // Test with maximum slot index values
    let code = make_code(vec![
        Instruction::op_di(Opcode::LoadLocal, Register(0), 65535),
        Instruction::op_di(Opcode::StoreLocal, Register(0), 65534),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadLocal { slot: 65535, .. }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::StoreLocal { slot: 65534, .. }
    ));
}

#[test]
fn test_lower_mixed_local_global_sequence() {
    // Test mixed local and global operations
    let code = make_code(vec![
        Instruction::op_di(Opcode::LoadLocal, Register(0), 0),
        Instruction::op_di(Opcode::LoadGlobal, Register(1), 5),
        Instruction::op_dss(Opcode::AddInt, Register(2), Register(0), Register(1)),
        Instruction::op_di(Opcode::StoreLocal, Register(2), 2),
        Instruction::op_di(Opcode::StoreGlobal, Register(2), 10),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 5);
    assert!(matches!(ir[0], TemplateInstruction::LoadLocal { .. }));
    assert!(matches!(ir[1], TemplateInstruction::LoadGlobal { .. }));
    assert!(matches!(ir[2], TemplateInstruction::IntAdd { .. }));
    assert!(matches!(ir[3], TemplateInstruction::StoreLocal { .. }));
    assert!(matches!(ir[4], TemplateInstruction::StoreGlobal { .. }));
}

#[test]
fn test_lower_all_load_store_ops() {
    // Test each load/store operation in isolation
    let load_store_ops = [
        (Opcode::LoadLocal, "LoadLocal"),
        (Opcode::StoreLocal, "StoreLocal"),
        (Opcode::DeleteLocal, "DeleteLocal"),
        (Opcode::LoadGlobal, "LoadGlobal"),
        (Opcode::StoreGlobal, "StoreGlobal"),
        (Opcode::DeleteGlobal, "DeleteGlobal"),
    ];

    for (op, name) in load_store_ops {
        let inst = match op {
            Opcode::LoadLocal | Opcode::LoadGlobal => Instruction::op_di(op, Register(0), 0),
            Opcode::StoreLocal | Opcode::StoreGlobal => Instruction::op_di(op, Register(0), 0),
            Opcode::DeleteLocal | Opcode::DeleteGlobal => Instruction::op_di(op, Register(0), 0),
            _ => unreachable!(),
        };
        let code = make_code(vec![inst]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for load/store op: {}", name);
    }
}

// =========================================================================
// Bitwise Operations Tests
// =========================================================================

#[test]
fn test_lower_bitwise_and() {
    // BitwiseAnd: r2 = r0 & r1
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BitwiseAnd,
        Register(2),
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntAnd {
            dst: 2,
            lhs: 0,
            rhs: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_bitwise_or() {
    // BitwiseOr: r2 = r0 | r1
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BitwiseOr,
        Register(2),
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntOr {
            dst: 2,
            lhs: 0,
            rhs: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_bitwise_xor() {
    // BitwiseXor: r2 = r0 ^ r1
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BitwiseXor,
        Register(2),
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntXor {
            dst: 2,
            lhs: 0,
            rhs: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_bitwise_not() {
    // BitwiseNot: r1 = ~r0
    let code = make_code(vec![Instruction::op_ds(
        Opcode::BitwiseNot,
        Register(1),
        Register(0),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntNot { dst: 1, src: 0, .. }
    ));
}

#[test]
fn test_lower_shl() {
    // Shl: r2 = r0 << r1
    let code = make_code(vec![Instruction::op_dss(
        Opcode::Shl,
        Register(2),
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntShl {
            dst: 2,
            lhs: 0,
            rhs: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_shr() {
    // Shr: r2 = r0 >> r1
    let code = make_code(vec![Instruction::op_dss(
        Opcode::Shr,
        Register(2),
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntShr {
            dst: 2,
            lhs: 0,
            rhs: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_logical_not() {
    // Not (logical): r1 = not r0
    let code = make_code(vec![Instruction::op_ds(
        Opcode::Not,
        Register(1),
        Register(0),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LogicalNot { dst: 1, src: 0, .. }
    ));
}

#[test]
fn test_lower_bitwise_sequence() {
    // Test sequence: compute (a & b) | (c ^ d)
    let code = make_code(vec![
        Instruction::op_dss(Opcode::BitwiseAnd, Register(2), Register(0), Register(1)),
        Instruction::op_dss(Opcode::BitwiseXor, Register(5), Register(3), Register(4)),
        Instruction::op_dss(Opcode::BitwiseOr, Register(6), Register(2), Register(5)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(ir[0], TemplateInstruction::IntAnd { .. }));
    assert!(matches!(ir[1], TemplateInstruction::IntXor { .. }));
    assert!(matches!(ir[2], TemplateInstruction::IntOr { .. }));
}

#[test]
fn test_lower_shift_sequence() {
    // Test shift sequence: (a << 2) >> 1
    let code = make_code(vec![
        Instruction::op_dss(Opcode::Shl, Register(2), Register(0), Register(1)),
        Instruction::op_dss(Opcode::Shr, Register(4), Register(2), Register(3)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(ir[0], TemplateInstruction::IntShl { .. }));
    assert!(matches!(ir[1], TemplateInstruction::IntShr { .. }));
}

#[test]
fn test_lower_bitwise_with_not() {
    // Test: ~(a & b) | c
    let code = make_code(vec![
        Instruction::op_dss(Opcode::BitwiseAnd, Register(2), Register(0), Register(1)),
        Instruction::op_ds(Opcode::BitwiseNot, Register(3), Register(2)),
        Instruction::op_dss(Opcode::BitwiseOr, Register(4), Register(3), Register(1)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(ir[0], TemplateInstruction::IntAnd { .. }));
    assert!(matches!(ir[1], TemplateInstruction::IntNot { .. }));
    assert!(matches!(ir[2], TemplateInstruction::IntOr { .. }));
}

#[test]
fn test_lower_bitwise_self_operand() {
    // Test: r0 = r0 & r0 (same register for all operands)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BitwiseAnd,
        Register(0),
        Register(0),
        Register(0),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntAnd {
            dst: 0,
            lhs: 0,
            rhs: 0,
            ..
        }
    ));
}

#[test]
fn test_lower_all_bitwise_ops() {
    // Test all bitwise operations in a comprehensive sweep
    let bitwise_binary_ops = [
        (Opcode::BitwiseAnd, "BitwiseAnd"),
        (Opcode::BitwiseOr, "BitwiseOr"),
        (Opcode::BitwiseXor, "BitwiseXor"),
        (Opcode::Shl, "Shl"),
        (Opcode::Shr, "Shr"),
    ];

    for (op, name) in bitwise_binary_ops {
        let code = make_code(vec![Instruction::op_dss(
            op,
            Register(2),
            Register(0),
            Register(1),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for binary bitwise op: {}", name);
    }

    // Test unary operations
    let bitwise_unary_ops = [(Opcode::BitwiseNot, "BitwiseNot"), (Opcode::Not, "Not")];

    for (op, name) in bitwise_unary_ops {
        let code = make_code(vec![Instruction::op_ds(op, Register(1), Register(0))]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for unary bitwise op: {}", name);
    }
}

#[test]
fn test_lower_bitwise_combined_with_arithmetic() {
    // Test mixing bitwise and arithmetic operations
    let code = make_code(vec![
        Instruction::op_dss(Opcode::AddInt, Register(2), Register(0), Register(1)),
        Instruction::op_dss(Opcode::BitwiseAnd, Register(3), Register(2), Register(1)),
        Instruction::op_dss(Opcode::Shl, Register(4), Register(3), Register(0)),
        Instruction::op_dss(Opcode::MulInt, Register(5), Register(4), Register(2)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 4);
    assert!(matches!(ir[0], TemplateInstruction::IntAdd { .. }));
    assert!(matches!(ir[1], TemplateInstruction::IntAnd { .. }));
    assert!(matches!(ir[2], TemplateInstruction::IntShl { .. }));
    assert!(matches!(ir[3], TemplateInstruction::IntMul { .. }));
}

#[test]
fn test_lower_bitwise_high_registers() {
    // Test bitwise operations with high register numbers
    let code = make_code(vec![
        Instruction::op_dss(
            Opcode::BitwiseOr,
            Register(200),
            Register(100),
            Register(150),
        ),
        Instruction::op_ds(Opcode::BitwiseNot, Register(255), Register(200)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IntOr {
            dst: 200,
            lhs: 100,
            rhs: 150,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::IntNot {
            dst: 255,
            src: 200,
            ..
        }
    ));
}

// =========================================================================
// Identity Operations Tests
// =========================================================================

#[test]
fn test_lower_is() {
    // Is: r2 = r0 is r1
    let code = make_code(vec![Instruction::op_dss(
        Opcode::Is,
        Register(2),
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::Is {
            dst: 2,
            lhs: 0,
            rhs: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_is_not() {
    // IsNot: r2 = r0 is not r1
    let code = make_code(vec![Instruction::op_dss(
        Opcode::IsNot,
        Register(2),
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IsNot {
            dst: 2,
            lhs: 0,
            rhs: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_is_self_comparison() {
    // Is: r0 = r0 is r0 (identity with self)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::Is,
        Register(0),
        Register(0),
        Register(0),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::Is {
            dst: 0,
            lhs: 0,
            rhs: 0,
            ..
        }
    ));
}

#[test]
fn test_lower_identity_sequence() {
    // Test sequence: (a is b) and (c is not d)
    let code = make_code(vec![
        Instruction::op_dss(Opcode::Is, Register(2), Register(0), Register(1)),
        Instruction::op_dss(Opcode::IsNot, Register(5), Register(3), Register(4)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(ir[0], TemplateInstruction::Is { .. }));
    assert!(matches!(ir[1], TemplateInstruction::IsNot { .. }));
}

#[test]
fn test_lower_all_identity_ops() {
    // Test all identity operations in isolation
    let identity_ops = [(Opcode::Is, "Is"), (Opcode::IsNot, "IsNot")];

    for (op, name) in identity_ops {
        let code = make_code(vec![Instruction::op_dss(
            op,
            Register(2),
            Register(0),
            Register(1),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for identity op: {}", name);
    }
}

#[test]
fn test_lower_identity_combined_with_bitwise() {
    // Test mixing identity and bitwise operations (common pattern: x is None or y & z)
    let code = make_code(vec![
        Instruction::op_dss(Opcode::Is, Register(2), Register(0), Register(1)),
        Instruction::op_dss(Opcode::BitwiseAnd, Register(3), Register(0), Register(1)),
        Instruction::op_dss(Opcode::IsNot, Register(4), Register(2), Register(3)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(ir[0], TemplateInstruction::Is { .. }));
    assert!(matches!(ir[1], TemplateInstruction::IntAnd { .. }));
    assert!(matches!(ir[2], TemplateInstruction::IsNot { .. }));
}

#[test]
fn test_lower_identity_high_registers() {
    // Test identity operations with high register numbers
    let code = make_code(vec![
        Instruction::op_dss(Opcode::Is, Register(200), Register(100), Register(150)),
        Instruction::op_dss(Opcode::IsNot, Register(255), Register(200), Register(100)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::Is {
            dst: 200,
            lhs: 100,
            rhs: 150,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::IsNot {
            dst: 255,
            lhs: 200,
            rhs: 100,
            ..
        }
    ));
}

// =========================================================================
// Control Flow Lowering Tests
// =========================================================================

#[test]
fn test_lower_jump() {
    // Simple unconditional jump
    let code = make_code(vec![
        Instruction::op_di(Opcode::Jump, Register(0), 5), // Jump forward 5 instructions
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::Jump { bc_offset, target } = ir[0] {
        assert_eq!(bc_offset, 0);
        // Target = (bc_offset + 4) + (5 * 4) = 4 + 20 = 24
        assert_eq!(target, 24);
    } else {
        panic!("Expected Jump instruction");
    }
}

#[test]
fn test_lower_jump_if_true() {
    // Branch if condition is true
    let code = make_code(vec![Instruction::op_di(Opcode::JumpIfTrue, Register(1), 3)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::BranchIfTrue {
        bc_offset,
        cond,
        target,
    } = ir[0]
    {
        assert_eq!(bc_offset, 0);
        assert_eq!(cond, 1);
        // Target = (0 + 4) + (3 * 4) = 4 + 12 = 16
        assert_eq!(target, 16);
    } else {
        panic!("Expected BranchIfTrue instruction");
    }
}

#[test]
fn test_lower_jump_if_false() {
    // Branch if condition is false
    let code = make_code(vec![Instruction::op_di(
        Opcode::JumpIfFalse,
        Register(2),
        10,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::BranchIfFalse {
        bc_offset,
        cond,
        target,
    } = ir[0]
    {
        assert_eq!(bc_offset, 0);
        assert_eq!(cond, 2);
        // Target = (0 + 4) + (10 * 4) = 4 + 40 = 44
        assert_eq!(target, 44);
    } else {
        panic!("Expected BranchIfFalse instruction");
    }
}

#[test]
fn test_lower_return_value() {
    // Return with value
    let code = make_code(vec![Instruction::op_d(Opcode::Return, Register(5))]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::Return { bc_offset, value } = ir[0] {
        assert_eq!(bc_offset, 0);
        assert_eq!(value, 5);
    } else {
        panic!("Expected Return instruction");
    }
}

#[test]
fn test_lower_return_none() {
    // Return None (implicit)
    let code = make_code(vec![Instruction::op(Opcode::ReturnNone)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::ReturnNone { bc_offset } = ir[0] {
        assert_eq!(bc_offset, 0);
    } else {
        panic!("Expected ReturnNone instruction");
    }
}

#[test]
fn test_lower_jump_backward() {
    // Test backward jump (negative offset)
    // Offset is u16 cast to i16, so a high u16 value represents a negative offset
    let backward_offset: u16 = (-3i16) as u16;
    let code = make_code(vec![Instruction::op_di(
        Opcode::Jump,
        Register(0),
        backward_offset,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::Jump { bc_offset, target } = ir[0] {
        assert_eq!(bc_offset, 0);
        // Target = (0 + 4) - (3 * 4) = 4 - 12 (wraps around)
        // Using wrapping arithmetic: next_pc (4) - 12 should wrap
        let expected = 4u32.wrapping_sub(12);
        assert_eq!(target, expected);
    } else {
        panic!("Expected Jump instruction");
    }
}

#[test]
fn test_lower_control_flow_sequence() {
    // Test a realistic control flow pattern (if-else)
    let code = make_code(vec![
        Instruction::op_di(Opcode::JumpIfFalse, Register(0), 2), // Skip if false
        Instruction::op_dss(Opcode::Add, Register(1), Register(2), Register(3)), // if-branch
        Instruction::op_di(Opcode::Jump, Register(0), 1),        // Skip else
        Instruction::op_dss(Opcode::Sub, Register(1), Register(2), Register(3)), // else-branch
        Instruction::op_ds(Opcode::Return, Register(0), Register(1)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 5);
    assert!(matches!(ir[0], TemplateInstruction::BranchIfFalse { .. }));
    assert!(matches!(ir[1], TemplateInstruction::IntAdd { .. }));
    assert!(matches!(ir[2], TemplateInstruction::Jump { .. }));
    assert!(matches!(ir[3], TemplateInstruction::IntSub { .. }));
    assert!(matches!(ir[4], TemplateInstruction::Return { .. }));
}

#[test]
fn test_lower_loop_pattern() {
    // Test a loop pattern (while loop) - uses speculation-guided lowering
    let code = make_code(vec![
        // Loop header: check condition (uses Lt opcode)
        Instruction::op_dss(Opcode::Lt, Register(2), Register(0), Register(1)),
        Instruction::op_di(Opcode::JumpIfFalse, Register(2), 3), // Exit if false
        // Loop body
        Instruction::op_dss(Opcode::Add, Register(0), Register(0), Register(3)),
        // Loop back
        Instruction::op_di(Opcode::Jump, Register(0), (-4i16) as u16), // Back to header
        // After loop
        Instruction::op_ds(Opcode::Return, Register(0), Register(0)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 5);
    // Without speculation, Lt lowers to IntLt as default
    assert!(matches!(ir[0], TemplateInstruction::IntLt { .. }));
    assert!(matches!(ir[1], TemplateInstruction::BranchIfFalse { .. }));
    assert!(matches!(ir[2], TemplateInstruction::IntAdd { .. }));
    assert!(matches!(ir[3], TemplateInstruction::Jump { .. }));
    assert!(matches!(ir[4], TemplateInstruction::Return { .. }));

    // Verify backward jump target
    if let TemplateInstruction::Jump { target, .. } = ir[3] {
        // Instruction at offset 12, jumps back 4 instructions
        // Target = (12 + 4) - (4 * 4) = 16 - 16 = 0
        assert_eq!(target, 0);
    } else {
        panic!("Expected Jump instruction");
    }
}

#[test]
fn test_lower_control_flow_all_opcodes() {
    // Test all control flow opcodes in one sequence
    let all_control = vec![
        ("Jump", Opcode::Jump),
        ("JumpIfTrue", Opcode::JumpIfTrue),
        ("JumpIfFalse", Opcode::JumpIfFalse),
    ];

    for (name, opcode) in all_control {
        let code = make_code(vec![Instruction::op_di(opcode, Register(0), 1)]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for control flow op: {}", name);
    }
}

#[test]
fn test_lower_return_high_registers() {
    // Test Return with high register numbers
    let code = make_code(vec![Instruction::op_d(Opcode::Return, Register(255))]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::Return { value, .. } = ir[0] {
        assert_eq!(value, 255);
    } else {
        panic!("Expected Return instruction");
    }
}

#[test]
fn test_lower_branch_high_condition_register() {
    // Test BranchIfTrue/False with high condition register
    let code = make_code(vec![
        Instruction::op_di(Opcode::JumpIfTrue, Register(200), 5),
        Instruction::op_di(Opcode::JumpIfFalse, Register(255), 3),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    if let TemplateInstruction::BranchIfTrue { cond, .. } = ir[0] {
        assert_eq!(cond, 200);
    } else {
        panic!("Expected BranchIfTrue instruction");
    }
    if let TemplateInstruction::BranchIfFalse { cond, .. } = ir[1] {
        assert_eq!(cond, 255);
    } else {
        panic!("Expected BranchIfFalse instruction");
    }
}

// =========================================================================
// Phase 7: Control Flow Extensions - JumpIfNone/JumpIfNotNone
// =========================================================================

#[test]
fn test_lower_jump_if_none() {
    // Basic JumpIfNone lowering
    let code = make_code(vec![Instruction::op_di(Opcode::JumpIfNone, Register(5), 3)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::BranchIfNone {
        bc_offset,
        cond,
        target,
    } = ir[0]
    {
        assert_eq!(bc_offset, 0);
        assert_eq!(cond, 5);
        assert_eq!(target, 4 + 3 * 4); // next_pc + offset * 4
    } else {
        panic!("Expected BranchIfNone instruction");
    }
}

#[test]
fn test_lower_jump_if_not_none() {
    // Basic JumpIfNotNone lowering
    let code = make_code(vec![Instruction::op_di(
        Opcode::JumpIfNotNone,
        Register(7),
        4,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::BranchIfNotNone {
        bc_offset,
        cond,
        target,
    } = ir[0]
    {
        assert_eq!(bc_offset, 0);
        assert_eq!(cond, 7);
        assert_eq!(target, 4 + 4 * 4); // next_pc + offset * 4
    } else {
        panic!("Expected BranchIfNotNone instruction");
    }
}

#[test]
fn test_lower_jump_if_none_backward() {
    // Backward JumpIfNone (negative offset)
    let backward_offset = (-2i16) as u16;
    let code = make_code(vec![Instruction::op_di(
        Opcode::JumpIfNone,
        Register(0),
        backward_offset,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::BranchIfNone {
        bc_offset, target, ..
    } = ir[0]
    {
        assert_eq!(bc_offset, 0);
        // Target = next_pc (4) + (-2 * 4) = 4 - 8 (wraps)
        let expected = 4u32.wrapping_sub(8);
        assert_eq!(target, expected);
    } else {
        panic!("Expected BranchIfNone instruction");
    }
}

#[test]
fn test_lower_jump_if_not_none_backward() {
    // Backward JumpIfNotNone (negative offset)
    let backward_offset = (-3i16) as u16;
    let code = make_code(vec![Instruction::op_di(
        Opcode::JumpIfNotNone,
        Register(1),
        backward_offset,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::BranchIfNotNone {
        bc_offset, target, ..
    } = ir[0]
    {
        assert_eq!(bc_offset, 0);
        // Target = next_pc (4) + (-3 * 4) = 4 - 12 (wraps)
        let expected = 4u32.wrapping_sub(12);
        assert_eq!(target, expected);
    } else {
        panic!("Expected BranchIfNotNone instruction");
    }
}

#[test]
fn test_lower_null_check_pattern() {
    // Realistic null check pattern: if x is not None: use x
    let code = make_code(vec![
        Instruction::op_di(Opcode::JumpIfNone, Register(0), 2), // Skip if None
        Instruction::op_dss(Opcode::Add, Register(1), Register(0), Register(2)), // Use x
        Instruction::op(Opcode::ReturnNone),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BranchIfNone { cond: 0, .. }
    ));
    assert!(matches!(ir[1], TemplateInstruction::IntAdd { .. }));
    assert!(matches!(ir[2], TemplateInstruction::ReturnNone { .. }));
}

#[test]
fn test_lower_optional_unwrap_pattern() {
    // Optional unwrap pattern: x = default if x is None else x
    let code = make_code(vec![
        Instruction::op_di(Opcode::JumpIfNotNone, Register(0), 2), // Skip if not None
        Instruction::op_ds(Opcode::Move, Register(0), Register(1)), // Copy default
        Instruction::op_d(Opcode::Return, Register(0)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BranchIfNotNone { cond: 0, .. }
    ));
    assert!(matches!(ir[1], TemplateInstruction::Move { .. }));
    assert!(matches!(
        ir[2],
        TemplateInstruction::Return { value: 0, .. }
    ));
}

#[test]
fn test_lower_all_null_branches() {
    // Verify all null branch opcodes are lowered
    let null_ops = [
        ("JumpIfNone", Opcode::JumpIfNone),
        ("JumpIfNotNone", Opcode::JumpIfNotNone),
    ];

    for (name, opcode) in null_ops {
        let code = make_code(vec![Instruction::op_di(opcode, Register(0), 1)]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for null control flow op: {}", name);
    }
}

#[test]
fn test_lower_null_branch_high_registers() {
    // Test null branches with high register numbers
    let code = make_code(vec![
        Instruction::op_di(Opcode::JumpIfNone, Register(200), 5),
        Instruction::op_di(Opcode::JumpIfNotNone, Register(255), 3),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    if let TemplateInstruction::BranchIfNone { cond, .. } = ir[0] {
        assert_eq!(cond, 200);
    } else {
        panic!("Expected BranchIfNone instruction");
    }
    if let TemplateInstruction::BranchIfNotNone { cond, .. } = ir[1] {
        assert_eq!(cond, 255);
    } else {
        panic!("Expected BranchIfNotNone instruction");
    }
}

// =========================================================================
// Phase 8: Membership Tests - In/NotIn
// =========================================================================

#[test]
fn test_lower_in() {
    // Basic In lowering
    let code = make_code(vec![Instruction::op_dss(
        Opcode::In,
        Register(0),
        Register(1),
        Register(2),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::In {
        bc_offset,
        dst,
        lhs,
        rhs,
    } = ir[0]
    {
        assert_eq!(bc_offset, 0);
        assert_eq!(dst, 0);
        assert_eq!(lhs, 1);
        assert_eq!(rhs, 2);
    } else {
        panic!("Expected In instruction");
    }
}

#[test]
fn test_lower_not_in() {
    // Basic NotIn lowering
    let code = make_code(vec![Instruction::op_dss(
        Opcode::NotIn,
        Register(5),
        Register(3),
        Register(4),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    if let TemplateInstruction::NotIn {
        bc_offset,
        dst,
        lhs,
        rhs,
    } = ir[0]
    {
        assert_eq!(bc_offset, 0);
        assert_eq!(dst, 5);
        assert_eq!(lhs, 3);
        assert_eq!(rhs, 4);
    } else {
        panic!("Expected NotIn instruction");
    }
}

#[test]
fn test_lower_membership_high_registers() {
    // Test membership with high register numbers
    let code = make_code(vec![
        Instruction::op_dss(Opcode::In, Register(255), Register(200), Register(128)),
        Instruction::op_dss(Opcode::NotIn, Register(128), Register(255), Register(200)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    if let TemplateInstruction::In { dst, lhs, rhs, .. } = ir[0] {
        assert_eq!(dst, 255);
        assert_eq!(lhs, 200);
        assert_eq!(rhs, 128);
    } else {
        panic!("Expected In instruction");
    }
    if let TemplateInstruction::NotIn { dst, lhs, rhs, .. } = ir[1] {
        assert_eq!(dst, 128);
        assert_eq!(lhs, 255);
        assert_eq!(rhs, 200);
    } else {
        panic!("Expected NotIn instruction");
    }
}

#[test]
fn test_lower_all_membership_ops() {
    // Verify all membership opcodes are lowered
    let membership_ops = [("In", Opcode::In), ("NotIn", Opcode::NotIn)];

    for (name, opcode) in membership_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for membership op: {}", name);
    }
}

#[test]
fn test_lower_membership_sequence() {
    // Realistic pattern: multiple membership checks
    let code = make_code(vec![
        Instruction::op_dss(Opcode::In, Register(3), Register(0), Register(1)),
        Instruction::op_dss(Opcode::NotIn, Register(4), Register(0), Register(2)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::In {
            dst: 3,
            lhs: 0,
            rhs: 1,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::NotIn {
            dst: 4,
            lhs: 0,
            rhs: 2,
            ..
        }
    ));
}

// =========================================================================
// Phase 9: Closure Variable Operations
// =========================================================================
