use super::*;

// =========================================================================
// OperandRole Tests
// =========================================================================

#[test]
fn test_operand_role_is_use() {
    assert!(OperandRole::Use.is_use());
    assert!(OperandRole::UseDef.is_use());
    assert!(!OperandRole::Def.is_use());
    assert!(!OperandRole::EarlyClobber.is_use());
}

#[test]
fn test_operand_role_is_def() {
    assert!(OperandRole::Def.is_def());
    assert!(OperandRole::UseDef.is_def());
    assert!(OperandRole::EarlyClobber.is_def());
    assert!(!OperandRole::Use.is_def());
}

#[test]
fn test_operand_role_is_use_def() {
    assert!(OperandRole::UseDef.is_use_def());
    assert!(!OperandRole::Use.is_use_def());
    assert!(!OperandRole::Def.is_use_def());
}

// =========================================================================
// OperandConstraint Tests
// =========================================================================

#[test]
fn test_fixed_constraint() {
    let c = OperandConstraint::fixed_gpr(Gpr::Rax);
    assert!(c.is_fixed());
    assert_eq!(c.fixed_reg(), Some(PReg::Gpr(Gpr::Rax)));
}

#[test]
fn test_reg_class_constraint() {
    let c = OperandConstraint::ymm();
    assert!(!c.is_fixed());
    assert_eq!(c.get_reg_class(), Some(RegClass::Vec256));
    assert!(c.is_vector());
}

#[test]
fn test_constraint_satisfaction_gpr() {
    let c = OperandConstraint::gpr();
    assert!(c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
    assert!(c.is_satisfied_by(PReg::Gpr(Gpr::R15)));
    assert!(!c.is_satisfied_by(PReg::Xmm(Xmm::Xmm0)));
    assert!(!c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
}

#[test]
fn test_constraint_satisfaction_xmm() {
    let c = OperandConstraint::xmm();
    assert!(c.is_satisfied_by(PReg::Xmm(Xmm::Xmm0)));
    assert!(c.is_satisfied_by(PReg::Xmm(Xmm::Xmm15)));
    assert!(!c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
    assert!(!c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
}

#[test]
fn test_constraint_satisfaction_ymm() {
    let c = OperandConstraint::ymm();
    assert!(c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
    assert!(c.is_satisfied_by(PReg::Ymm(Ymm::Ymm15)));
    assert!(!c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
    assert!(!c.is_satisfied_by(PReg::Xmm(Xmm::Xmm0)));
}

#[test]
fn test_constraint_satisfaction_zmm() {
    let c = OperandConstraint::zmm();
    assert!(c.is_satisfied_by(PReg::Zmm(Zmm::Zmm0)));
    assert!(c.is_satisfied_by(PReg::Zmm(Zmm::Zmm31)));
    assert!(!c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
    assert!(!c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
}

#[test]
fn test_constraint_satisfaction_fixed() {
    let c = OperandConstraint::fixed_gpr(Gpr::Rax);
    assert!(c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
    assert!(!c.is_satisfied_by(PReg::Gpr(Gpr::Rbx)));
}

#[test]
fn test_constraint_satisfaction_any() {
    let c = OperandConstraint::Any;
    assert!(c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
    assert!(c.is_satisfied_by(PReg::Xmm(Xmm::Xmm0)));
    assert!(c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
    assert!(c.is_satisfied_by(PReg::Zmm(Zmm::Zmm0)));
}

#[test]
fn test_constraint_display() {
    assert_eq!(format!("{}", OperandConstraint::gpr()), "Int");
    assert_eq!(format!("{}", OperandConstraint::ymm()), "Vec256");
    assert!(format!("{}", OperandConstraint::fixed_gpr(Gpr::Rax)).contains("rax"));
    assert_eq!(format!("{}", OperandConstraint::tied(0)), "tied(0)");
}

// =========================================================================
// OperandDescriptor Tests
// =========================================================================

#[test]
fn test_operand_descriptor_use() {
    let v = VReg::new(0);
    let desc = OperandDescriptor::use_op(v, OperandConstraint::ymm());
    assert_eq!(desc.role, OperandRole::Use);
    assert!(desc.is_vector());
}

#[test]
fn test_operand_descriptor_def() {
    let v = VReg::new(1);
    let desc = OperandDescriptor::def(v, OperandConstraint::gpr());
    assert_eq!(desc.role, OperandRole::Def);
    assert!(!desc.is_vector());
}

#[test]
fn test_operand_descriptor_use_def() {
    let v = VReg::new(2);
    let desc = OperandDescriptor::use_def(v, OperandConstraint::zmm());
    assert_eq!(desc.role, OperandRole::UseDef);
    assert!(desc.is_vector());
}

// =========================================================================
// InstructionConstraint Tests
// =========================================================================

#[test]
fn test_instruction_constraint_creation() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let ic = InstructionConstraint::new("vaddpd_ymm")
        .def(v0, OperandConstraint::ymm())
        .use_op(v1, OperandConstraint::ymm())
        .use_op(v2, OperandConstraint::ymm())
        .commutative();

    assert_eq!(ic.name, "vaddpd_ymm");
    assert_eq!(ic.num_defs(), 1);
    assert_eq!(ic.num_uses(), 2);
    assert!(ic.is_commutative);
    assert!(!ic.is_two_address);
}

#[test]
fn test_instruction_constraint_two_address() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);

    let ic = InstructionConstraint::new("add_rr")
        .use_def(v0, OperandConstraint::gpr())
        .use_op(v1, OperandConstraint::gpr());

    assert!(ic.is_two_address);
    assert_eq!(ic.num_defs(), 1);
    assert_eq!(ic.num_uses(), 2); // use_def counts as both
}

#[test]
fn test_instruction_constraint_clobbers() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let ic = InstructionConstraint::new("div")
        .def(v0, OperandConstraint::fixed_gpr(Gpr::Rax))
        .def(v1, OperandConstraint::fixed_gpr(Gpr::Rdx))
        .use_op(v2, OperandConstraint::gpr())
        .clobber(PReg::Gpr(Gpr::Rax))
        .clobber(PReg::Gpr(Gpr::Rdx));

    assert_eq!(ic.clobbers.len(), 2);
}

#[test]
fn test_instruction_constraint_has_vector() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);

    let ic_vec = InstructionConstraint::new("vmovapd_ymm")
        .def(v0, OperandConstraint::ymm())
        .use_op(v1, OperandConstraint::ymm());
    assert!(ic_vec.has_vector_operands());

    let ic_gpr = InstructionConstraint::new("mov_rr")
        .def(v0, OperandConstraint::gpr())
        .use_op(v1, OperandConstraint::gpr());
    assert!(!ic_gpr.has_vector_operands());
}

#[test]
fn test_instruction_constraint_vector_width() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);

    let ic_zmm = InstructionConstraint::new("vaddpd_zmm")
        .def(v0, OperandConstraint::zmm())
        .use_op(v1, OperandConstraint::zmm());
    assert_eq!(ic_zmm.vector_width(), Some(512));

    let ic_ymm = InstructionConstraint::new("vaddpd_ymm")
        .def(v0, OperandConstraint::ymm())
        .use_op(v1, OperandConstraint::ymm());
    assert_eq!(ic_ymm.vector_width(), Some(256));

    let ic_xmm = InstructionConstraint::new("addsd")
        .def(v0, OperandConstraint::xmm())
        .use_op(v1, OperandConstraint::xmm());
    assert_eq!(ic_xmm.vector_width(), Some(128));

    let ic_gpr = InstructionConstraint::new("add")
        .def(v0, OperandConstraint::gpr())
        .use_op(v1, OperandConstraint::gpr());
    assert_eq!(ic_gpr.vector_width(), None);
}

// =========================================================================
// Template Tests
// =========================================================================

#[test]
fn test_simd_binary_template() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let ic = simd_binary_rrr("vaddpd_ymm", v0, v1, v2, RegClass::Vec256);
    assert_eq!(ic.name, "vaddpd_ymm");
    assert_eq!(ic.num_defs(), 1);
    assert_eq!(ic.num_uses(), 2);
    assert!(ic.is_commutative);
    assert!(ic.has_vector_operands());
}

#[test]
fn test_simd_unary_template() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);

    let ic = simd_unary_rr("vsqrtpd_ymm", v0, v1, RegClass::Vec256);
    assert_eq!(ic.num_defs(), 1);
    assert_eq!(ic.num_uses(), 1);
}

#[test]
fn test_simd_fma_template() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let ic = simd_fma("vfmadd213pd_zmm", v0, v1, v2, RegClass::Vec512);
    assert!(ic.is_two_address);
    assert_eq!(ic.vector_width(), Some(512));
}

#[test]
fn test_div_constraint_template() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let ic = div_constraint("idiv", v0, v1, v2);
    assert_eq!(ic.name, "idiv");
    assert_eq!(ic.num_defs(), 2);

    // Check fixed register constraints
    let defs: Vec<_> = ic.defs().collect();
    assert!(defs[0].constraint.is_fixed());
    assert_eq!(defs[0].constraint.fixed_reg(), Some(PReg::Gpr(Gpr::Rax)));
    assert!(defs[1].constraint.is_fixed());
    assert_eq!(defs[1].constraint.fixed_reg(), Some(PReg::Gpr(Gpr::Rdx)));
}

#[test]
fn test_shift_by_cl_template() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let ic = shift_by_cl_constraint("shl", v0, v1, v2);
    assert!(ic.is_two_address);

    // Count operand should be fixed to RCX
    let uses: Vec<_> = ic.uses().collect();
    assert!(
        uses.iter()
            .any(|op| op.constraint.fixed_reg() == Some(PReg::Gpr(Gpr::Rcx)))
    );
}

// =========================================================================
// ConstraintDatabase Tests
// =========================================================================

#[test]
fn test_constraint_database_register() {
    let mut db = ConstraintDatabase::new();
    let v0 = VReg::new(0);

    let ic = InstructionConstraint::new("test").def(v0, OperandConstraint::gpr());

    db.register(ic);
    assert_eq!(db.len(), 1);
    assert!(!db.is_empty());
}

#[test]
fn test_constraint_database_lookup() {
    let mut db = ConstraintDatabase::new();
    let v0 = VReg::new(0);

    let ic = InstructionConstraint::new("test_op").def(v0, OperandConstraint::ymm());

    db.register(ic);

    let found = db.get("test_op");
    assert!(found.is_some());
    assert_eq!(found.unwrap().name, "test_op");

    let not_found = db.get("nonexistent");
    assert!(not_found.is_none());
}
