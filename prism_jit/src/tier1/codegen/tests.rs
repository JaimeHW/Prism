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
fn test_unsupported_tier1_instruction_is_explicit_deopt() {
    let compiler = TemplateCompiler::new_for_testing();
    let instr = TemplateInstruction::BuildList {
        bc_offset: 12,
        dst: 0,
        start: 1,
        count: 2,
    };

    assert!(instr.requires_interpreter_in_tier1());
    assert!(!instr.can_deopt(), "inline deopt does not need a side stub");
    assert!(compiler.compile(4, &[instr]).is_ok());
}

#[test]
fn test_attribute_templates_register_deopt_stubs() {
    let instr = TemplateInstruction::GetAttr {
        bc_offset: 8,
        dst: 0,
        obj: 1,
        name_idx: 2,
        ic_site_idx: None,
    };

    assert!(!instr.requires_interpreter_in_tier1());
    assert!(instr.can_deopt());
    assert_eq!(instr.deopt_reason(), DeoptReason::UncommonTrap);
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
fn test_compile_jump_to_end_sentinel() {
    let compiler = TemplateCompiler::new_for_testing();
    let instrs = vec![
        TemplateInstruction::Jump {
            bc_offset: 0,
            target: 8,
        },
        TemplateInstruction::ReturnNone { bc_offset: 4 },
    ];

    let result = compiler.compile(1, &instrs);
    assert!(result.is_ok());
}

#[test]
fn test_compile_branch_to_end_sentinel() {
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
        TemplateInstruction::ReturnNone { bc_offset: 8 },
    ];

    let result = compiler.compile(1, &instrs);
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

    let branch_none = TemplateInstruction::BranchIfNone {
        bc_offset: 8,
        cond: 0,
        target: 20,
    };
    assert_eq!(branch_none.jump_target(), Some(20));

    let end_async_for = TemplateInstruction::EndAsyncFor {
        bc_offset: 12,
        dst: 0,
        target: 28,
    };
    assert_eq!(end_async_for.jump_target(), Some(28));

    let ret_none = TemplateInstruction::ReturnNone { bc_offset: 16 };
    assert_eq!(ret_none.bc_offset(), 16);
    assert_eq!(ret_none.jump_target(), None);
    assert!(!ret_none.can_deopt());
}

#[test]
fn test_compile_int_comparisons() {
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
        TemplateInstruction::IntLt {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::IntLe {
            bc_offset: 12,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::IntGt {
            bc_offset: 16,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::IntGe {
            bc_offset: 20,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::IntEq {
            bc_offset: 24,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::IntNe {
            bc_offset: 28,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::Return {
            bc_offset: 32,
            value: 2,
        },
    ];

    let result = compiler.compile(4, &instrs);
    assert!(result.is_ok());

    let func = result.unwrap();
    // Should have deopt info for all integer comparisons
    assert!(func.deopt_info.len() >= 6);
    // Code should be substantial (each comparison generates ~100 bytes)
    assert!(func.code.len() > 200);
}

#[test]
fn test_compile_float_comparisons() {
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
        TemplateInstruction::FloatLt {
            bc_offset: 8,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::FloatLe {
            bc_offset: 12,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::FloatGt {
            bc_offset: 16,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::FloatGe {
            bc_offset: 20,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::FloatEq {
            bc_offset: 24,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::FloatNe {
            bc_offset: 28,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        TemplateInstruction::Return {
            bc_offset: 32,
            value: 2,
        },
    ];

    let result = compiler.compile(4, &instrs);
    assert!(result.is_ok());

    let func = result.unwrap();
    // Float comparisons don't deopt (pure float math)
    // Code should be substantial
    assert!(func.code.len() > 100);
}

#[test]
fn test_compile_type_guards() {
    let compiler = TemplateCompiler::new_for_testing();
    let instrs = vec![
        TemplateInstruction::LoadInt {
            bc_offset: 0,
            dst: 0,
            value: 42,
        },
        TemplateInstruction::GuardInt {
            bc_offset: 4,
            reg: 0,
        },
        TemplateInstruction::LoadFloat {
            bc_offset: 8,
            dst: 1,
            value: 3.125,
        },
        TemplateInstruction::GuardFloat {
            bc_offset: 12,
            reg: 1,
        },
        TemplateInstruction::LoadBool {
            bc_offset: 16,
            dst: 2,
            value: true,
        },
        TemplateInstruction::GuardBool {
            bc_offset: 20,
            reg: 2,
        },
        TemplateInstruction::Return {
            bc_offset: 24,
            value: 0,
        },
    ];

    let result = compiler.compile(4, &instrs);
    assert!(result.is_ok());

    let func = result.unwrap();
    // Should have deopt info for all 3 guards
    assert!(func.deopt_info.len() >= 3);
    // Guards generate compact code
    assert!(func.code.len() > 50);
}

#[test]
fn test_int_comparison_deopt_properties() {
    // Verify IntLt can deopt
    let lt = TemplateInstruction::IntLt {
        bc_offset: 0,
        dst: 0,
        lhs: 1,
        rhs: 2,
    };
    assert!(lt.can_deopt());
    assert_eq!(lt.deopt_reason(), DeoptReason::TypeGuardFailed);

    // Verify IntEq can deopt
    let eq = TemplateInstruction::IntEq {
        bc_offset: 0,
        dst: 0,
        lhs: 1,
        rhs: 2,
    };
    assert!(eq.can_deopt());

    // Verify FloatLt does NOT deopt (pure float)
    let float_lt = TemplateInstruction::FloatLt {
        bc_offset: 0,
        dst: 0,
        lhs: 1,
        rhs: 2,
    };
    assert!(!float_lt.can_deopt());
}

#[test]
fn test_guard_deopt_properties() {
    let guard_int = TemplateInstruction::GuardInt {
        bc_offset: 0,
        reg: 0,
    };
    assert!(guard_int.can_deopt());
    assert_eq!(guard_int.deopt_reason(), DeoptReason::TypeGuardFailed);

    let guard_float = TemplateInstruction::GuardFloat {
        bc_offset: 0,
        reg: 0,
    };
    assert!(guard_float.can_deopt());

    let guard_bool = TemplateInstruction::GuardBool {
        bc_offset: 0,
        reg: 0,
    };
    assert!(guard_bool.can_deopt());
}
