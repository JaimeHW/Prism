use super::*;

#[test]
fn test_lower_load_closure_basic() {
    // LoadClosure uses op_di format: dst = closure[imm16].get()
    let code = make_code(vec![Instruction::op_di(
        Opcode::LoadClosure,
        Register(5),
        42,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadClosure {
            dst: 5,
            cell_idx: 42,
            ..
        }
    ));
}

#[test]
fn test_lower_store_closure_basic() {
    // StoreClosure uses op_di format: closure[imm16].set(src)
    let code = make_code(vec![Instruction::op_di(
        Opcode::StoreClosure,
        Register(3),
        17,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::StoreClosure {
            src: 3,
            cell_idx: 17,
            ..
        }
    ));
}

#[test]
fn test_lower_delete_closure_basic() {
    // DeleteClosure uses op_di format: closure[imm16].clear()
    let code = make_code(vec![Instruction::op_di(
        Opcode::DeleteClosure,
        Register(0), // Register ignored for delete
        8,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::DeleteClosure { cell_idx: 8, .. }
    ));
}

#[test]
fn test_lower_all_closure_ops() {
    // Verify all closure opcodes are lowered
    let closure_ops = [
        ("LoadClosure", Opcode::LoadClosure),
        ("StoreClosure", Opcode::StoreClosure),
        ("DeleteClosure", Opcode::DeleteClosure),
    ];

    for (name, opcode) in closure_ops {
        let code = make_code(vec![Instruction::op_di(opcode, Register(0), 5)]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for closure op: {}", name);
    }
}

#[test]
fn test_lower_closure_load_with_high_cell_index() {
    // Test with maximum cell index (u16::MAX - 1)
    let code = make_code(vec![Instruction::op_di(
        Opcode::LoadClosure,
        Register(0),
        65534,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadClosure {
            dst: 0,
            cell_idx: 65534,
            ..
        }
    ));
}

#[test]
fn test_lower_closure_sequence_pattern() {
    // Realistic pattern: load captured value, modify, store back
    let code = make_code(vec![
        // x = closure[0] (load captured variable)
        Instruction::op_di(Opcode::LoadClosure, Register(0), 0),
        // temp = x + 1 (arithmetic on captured value)
        Instruction::op_dss(Opcode::Add, Register(1), Register(0), Register(2)),
        // closure[0] = temp (store back)
        Instruction::op_di(Opcode::StoreClosure, Register(1), 0),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadClosure {
            dst: 0,
            cell_idx: 0,
            ..
        }
    ));
    // Middle instruction is Add (uses speculation-guided lowering)
    assert!(matches!(
        ir[2],
        TemplateInstruction::StoreClosure {
            src: 1,
            cell_idx: 0,
            ..
        }
    ));
}

#[test]
fn test_lower_multiple_closure_cells() {
    // Pattern: access multiple closure cells (nested function with multiple captures)
    let code = make_code(vec![
        Instruction::op_di(Opcode::LoadClosure, Register(0), 0), // First captured var
        Instruction::op_di(Opcode::LoadClosure, Register(1), 1), // Second captured var
        Instruction::op_di(Opcode::LoadClosure, Register(2), 2), // Third captured var
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadClosure {
            dst: 0,
            cell_idx: 0,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::LoadClosure {
            dst: 1,
            cell_idx: 1,
            ..
        }
    ));
    assert!(matches!(
        ir[2],
        TemplateInstruction::LoadClosure {
            dst: 2,
            cell_idx: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_closure_delete_sequence() {
    // Pattern: delete captured variable then reload (should fail at runtime)
    let code = make_code(vec![
        Instruction::op_di(Opcode::DeleteClosure, Register(0), 5),
        Instruction::op_di(Opcode::LoadClosure, Register(0), 5),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::DeleteClosure { cell_idx: 5, .. }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::LoadClosure {
            dst: 0,
            cell_idx: 5,
            ..
        }
    ));
}

// =========================================================================
// Phase 11: Object Attribute Operations
// =========================================================================

#[test]
fn test_lower_get_attr_basic() {
    // GetAttr uses DstSrcSrc format: dst = src1.attr[src2]
    let code = make_code(vec![Instruction::op_dss(
        Opcode::GetAttr,
        Register(5),
        Register(1),
        Register(3), // name_idx encoded as register
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::GetAttr {
            dst: 5,
            obj: 1,
            name_idx: 3,
            ..
        }
    ));
}

#[test]
fn test_lower_set_attr_basic() {
    // SetAttr uses DstSrcSrc format: dst.attr[src1] = src2
    let code = make_code(vec![Instruction::op_dss(
        Opcode::SetAttr,
        Register(2), // object
        Register(5), // name_idx
        Register(7), // value
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::SetAttr {
            obj: 2,
            name_idx: 5,
            value: 7,
            ..
        }
    ));
}

#[test]
fn test_lower_del_attr_basic() {
    // DelAttr uses DstSrcSrc format: del src1.attr[src2]
    let code = make_code(vec![Instruction::op_dss(
        Opcode::DelAttr,
        Register(0), // unused
        Register(3), // object
        Register(8), // name_idx
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::DelAttr {
            obj: 3,
            name_idx: 8,
            ..
        }
    ));
}

#[test]
fn test_lower_load_method_basic() {
    // LoadMethod uses DstSrcSrc format: dst = obj.method
    let code = make_code(vec![Instruction::op_dss(
        Opcode::LoadMethod,
        Register(4),
        Register(1),
        Register(6),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadMethod {
            dst: 4,
            obj: 1,
            name_idx: 6,
            ..
        }
    ));
}

#[test]
fn test_lower_all_object_attr_ops() {
    // Verify all object attribute opcodes are lowered
    let attr_ops = [
        ("GetAttr", Opcode::GetAttr),
        ("SetAttr", Opcode::SetAttr),
        ("DelAttr", Opcode::DelAttr),
        ("LoadMethod", Opcode::LoadMethod),
    ];

    for (name, opcode) in attr_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for object attr op: {}", name);
    }
}

#[test]
fn test_lower_object_attr_sequence() {
    // Realistic pattern: load method, then get attr
    let code = make_code(vec![
        // Method lookup: dst = obj.method
        Instruction::op_dss(Opcode::LoadMethod, Register(5), Register(0), Register(1)),
        // Attribute access: dst = obj.attr
        Instruction::op_dss(Opcode::GetAttr, Register(6), Register(0), Register(2)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadMethod {
            dst: 5,
            obj: 0,
            name_idx: 1,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::GetAttr {
            dst: 6,
            obj: 0,
            name_idx: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_set_del_attr_sequence() {
    // Pattern: set attribute then delete different attribute
    let code = make_code(vec![
        Instruction::op_dss(Opcode::SetAttr, Register(0), Register(1), Register(5)),
        Instruction::op_dss(Opcode::DelAttr, Register(0), Register(0), Register(2)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::SetAttr {
            obj: 0,
            name_idx: 1,
            value: 5,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::DelAttr {
            obj: 0,
            name_idx: 2,
            ..
        }
    ));
}

// =========================================================================
// Phase 11b: IC (Inline Caching) Allocation Tests
// =========================================================================

#[test]
fn test_lower_get_attr_with_ic_manager_allocates_site() {
    // When an IcManager is provided, GetAttr should allocate an IC site
    use crate::ic::{IcManager, ShapeVersion};

    let code = make_code(vec![Instruction::op_dss(
        Opcode::GetAttr,
        Register(5),
        Register(1),
        Register(3),
    )]);
    let speculation = NoSpeculation;
    let shape_version = ShapeVersion::new(1);
    let mut ic_manager = IcManager::new(shape_version);

    let mut lowerer = BytecodeLowerer::with_ic_manager(
        &speculation,
        0,
        LoweringConfig::default(),
        &mut ic_manager,
    );

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);

    // Verify IC site was allocated
    match &ir[0] {
        TemplateInstruction::GetAttr { ic_site_idx, .. } => {
            assert!(
                ic_site_idx.is_some(),
                "GetAttr should have ic_site_idx with IcManager"
            );
            assert_eq!(ic_site_idx.unwrap(), 0, "First IC site should have index 0");
        }
        _ => panic!("Expected GetAttr instruction"),
    }

    // Verify IcManager state
    assert_eq!(ic_manager.len(), 1, "IcManager should have 1 IC site");
}

#[test]
fn test_lower_set_attr_with_ic_manager_allocates_site() {
    // SetAttr should also allocate an IC site when IcManager is provided
    use crate::ic::{IcManager, ShapeVersion};

    let code = make_code(vec![Instruction::op_dss(
        Opcode::SetAttr,
        Register(2),
        Register(5),
        Register(7),
    )]);
    let speculation = NoSpeculation;
    let shape_version = ShapeVersion::new(1);
    let mut ic_manager = IcManager::new(shape_version);

    let mut lowerer = BytecodeLowerer::with_ic_manager(
        &speculation,
        0,
        LoweringConfig::default(),
        &mut ic_manager,
    );

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);

    match &ir[0] {
        TemplateInstruction::SetAttr { ic_site_idx, .. } => {
            assert!(
                ic_site_idx.is_some(),
                "SetAttr should have ic_site_idx with IcManager"
            );
        }
        _ => panic!("Expected SetAttr instruction"),
    }

    assert_eq!(ic_manager.len(), 1, "IcManager should have 1 IC site");
}

#[test]
fn test_lower_multiple_attrs_allocate_unique_sites() {
    // Multiple GetAttr/SetAttr should each get unique IC site indices
    use crate::ic::{IcManager, ShapeVersion};

    let code = make_code(vec![
        Instruction::op_dss(Opcode::GetAttr, Register(5), Register(1), Register(3)),
        Instruction::op_dss(Opcode::SetAttr, Register(2), Register(5), Register(7)),
        Instruction::op_dss(Opcode::GetAttr, Register(6), Register(0), Register(4)),
    ]);
    let speculation = NoSpeculation;
    let shape_version = ShapeVersion::new(1);
    let mut ic_manager = IcManager::new(shape_version);

    let mut lowerer = BytecodeLowerer::with_ic_manager(
        &speculation,
        0,
        LoweringConfig::default(),
        &mut ic_manager,
    );

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);

    // Verify each instruction has a unique IC site index
    let mut indices = Vec::new();
    for instr in &ir {
        match instr {
            TemplateInstruction::GetAttr {
                ic_site_idx: Some(idx),
                ..
            } => indices.push(*idx),
            TemplateInstruction::SetAttr {
                ic_site_idx: Some(idx),
                ..
            } => indices.push(*idx),
            _ => panic!("Expected GetAttr or SetAttr with IC site"),
        }
    }

    assert_eq!(indices.len(), 3, "Should have 3 IC sites");
    assert_eq!(
        indices,
        vec![0, 1, 2],
        "IC site indices should be sequential"
    );
    assert_eq!(ic_manager.len(), 3, "IcManager should have 3 IC sites");
}

#[test]
fn test_lower_without_ic_manager_no_ic_site() {
    // Without IcManager, ic_site_idx should be None
    let code = make_code(vec![Instruction::op_dss(
        Opcode::GetAttr,
        Register(5),
        Register(1),
        Register(3),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);

    match &ir[0] {
        TemplateInstruction::GetAttr { ic_site_idx, .. } => {
            assert!(
                ic_site_idx.is_none(),
                "Without IcManager, ic_site_idx should be None"
            );
        }
        _ => panic!("Expected GetAttr instruction"),
    }
}

#[test]
fn test_lower_with_ic_disabled_no_ic_site() {
    // Even with IcManager, if enable_ic is false, no IC sites should be allocated
    use crate::ic::{IcManager, ShapeVersion};

    let code = make_code(vec![Instruction::op_dss(
        Opcode::GetAttr,
        Register(5),
        Register(1),
        Register(3),
    )]);
    let speculation = NoSpeculation;
    let shape_version = ShapeVersion::new(1);
    let mut ic_manager = IcManager::new(shape_version);

    let config = LoweringConfig {
        enable_ic: false, // Explicitly disable IC
        ..LoweringConfig::default()
    };

    let mut lowerer = BytecodeLowerer::with_ic_manager(&speculation, 0, config, &mut ic_manager);

    let ir = lowerer.lower(&code);

    match &ir[0] {
        TemplateInstruction::GetAttr { ic_site_idx, .. } => {
            assert!(
                ic_site_idx.is_none(),
                "With IC disabled, ic_site_idx should be None"
            );
        }
        _ => panic!("Expected GetAttr instruction"),
    }

    assert_eq!(
        ic_manager.len(),
        0,
        "IcManager should have 0 IC sites when IC is disabled"
    );
}

#[test]
fn test_lower_get_attr_correct_bytecode_offset_in_ic() {
    // Verify that IC sites are allocated with correct bytecode offsets
    use crate::ic::{IcKind, IcManager, ShapeVersion};

    let code = make_code(vec![
        Instruction::op_dss(Opcode::GetAttr, Register(5), Register(1), Register(3)),
        Instruction::op_dss(Opcode::GetAttr, Register(6), Register(2), Register(4)),
    ]);
    let speculation = NoSpeculation;
    let shape_version = ShapeVersion::new(1);
    let mut ic_manager = IcManager::new(shape_version);

    let mut lowerer = BytecodeLowerer::with_ic_manager(
        &speculation,
        0,
        LoweringConfig::default(),
        &mut ic_manager,
    );

    let _ = lowerer.lower(&code);

    // Verify IC site was created with GetProperty kind
    assert_eq!(ic_manager.len(), 2, "Should have 2 IC sites");

    let site0 = ic_manager.get(0).expect("Should have site 0");
    assert_eq!(site0.header.kind, IcKind::GetProperty);
    assert_eq!(site0.header.bytecode_offset, 0);

    let site1 = ic_manager.get(1).expect("Should have site 1");
    assert_eq!(site1.header.kind, IcKind::GetProperty);
    assert_eq!(site1.header.bytecode_offset, 4);
}

#[test]
fn test_lower_set_attr_correct_ic_kind() {
    // Verify that SetAttr uses SetProperty IC kind
    use crate::ic::{IcKind, IcManager, ShapeVersion};

    let code = make_code(vec![Instruction::op_dss(
        Opcode::SetAttr,
        Register(2),
        Register(5),
        Register(7),
    )]);
    let speculation = NoSpeculation;
    let shape_version = ShapeVersion::new(1);
    let mut ic_manager = IcManager::new(shape_version);

    let mut lowerer = BytecodeLowerer::with_ic_manager(
        &speculation,
        0,
        LoweringConfig::default(),
        &mut ic_manager,
    );

    let _ = lowerer.lower(&code);

    let site = ic_manager.get(0).expect("Should have site 0");
    assert_eq!(
        site.header.kind,
        IcKind::SetProperty,
        "SetAttr should use SetProperty IC kind"
    );
}

#[test]
fn test_del_attr_does_not_allocate_ic() {
    // DelAttr should not allocate IC sites (no IC support for deletion)
    use crate::ic::{IcManager, ShapeVersion};

    let code = make_code(vec![Instruction::op_dss(
        Opcode::DelAttr,
        Register(0),
        Register(3),
        Register(4),
    )]);
    let speculation = NoSpeculation;
    let shape_version = ShapeVersion::new(1);
    let mut ic_manager = IcManager::new(shape_version);

    let mut lowerer = BytecodeLowerer::with_ic_manager(
        &speculation,
        0,
        LoweringConfig::default(),
        &mut ic_manager,
    );

    let _ = lowerer.lower(&code);

    assert_eq!(ic_manager.len(), 0, "DelAttr should not allocate IC sites");
}

// =========================================================================
// Phase 12: Container Item Operations
// =========================================================================

#[test]
fn test_lower_get_item_basic() {
    // GetItem: dst = container[key]
    let code = make_code(vec![Instruction::op_dss(
        Opcode::GetItem,
        Register(3),
        Register(1),
        Register(2),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::GetItem {
            dst: 3,
            container: 1,
            key: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_set_item_basic() {
    // SetItem: src1[dst] = src2 (using DstSrcSrc, key in dst field)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::SetItem,
        Register(2), // key
        Register(0), // container
        Register(5), // value
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::SetItem {
            container: 0,
            key: 2,
            value: 5,
            ..
        }
    ));
}

#[test]
fn test_lower_del_item_basic() {
    // DelItem: del src1[src2]
    let code = make_code(vec![Instruction::op_dss(
        Opcode::DelItem,
        Register(0), // unused dst
        Register(1), // container
        Register(4), // key
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::DelItem {
            container: 1,
            key: 4,
            ..
        }
    ));
}

#[test]
fn test_lower_get_iter_basic() {
    // GetIter: dst = iter(src)
    let code = make_code(vec![Instruction::op_ds(
        Opcode::GetIter,
        Register(2),
        Register(0),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::GetIter { dst: 2, src: 0, .. }
    ));
}

#[test]
fn test_lower_for_iter_basic() {
    // ForIter: dst = next(dst - 1), jump offset in imm16
    let code = make_code(vec![Instruction::op_di(Opcode::ForIter, Register(3), 10)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::ForIter {
            dst: 3,
            iter: 2,
            offset: 10,
            ..
        }
    ));
}

#[test]
fn test_lower_len_basic() {
    // Len: dst = len(src)
    let code = make_code(vec![Instruction::op_ds(
        Opcode::Len,
        Register(5),
        Register(2),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::Len { dst: 5, src: 2, .. }
    ));
}

#[test]
fn test_lower_is_callable_basic() {
    // IsCallable: dst = callable(src)
    let code = make_code(vec![Instruction::op_ds(
        Opcode::IsCallable,
        Register(4),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::IsCallable { dst: 4, src: 1, .. }
    ));
}

#[test]
fn test_lower_all_container_ops() {
    // Verify all container opcodes produce output
    let container_ops = [
        ("GetItem", Opcode::GetItem),
        ("SetItem", Opcode::SetItem),
        ("DelItem", Opcode::DelItem),
    ];

    for (name, opcode) in container_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for container op: {}", name);
    }
}

#[test]
fn test_lower_all_iteration_ops() {
    // Verify all iteration opcodes produce output
    let iter_ops = [
        (
            "GetIter",
            make_code(vec![Instruction::op_ds(
                Opcode::GetIter,
                Register(0),
                Register(1),
            )]),
        ),
        (
            "ForIter",
            make_code(vec![Instruction::op_di(Opcode::ForIter, Register(2), 7)]),
        ),
    ];

    for (name, code) in iter_ops {
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for iteration op: {}", name);
    }
}

#[test]
fn test_lower_all_utility_ops() {
    // Verify utility opcodes produce output
    let utility_ops = [("Len", Opcode::Len), ("IsCallable", Opcode::IsCallable)];

    for (name, opcode) in utility_ops {
        let code = make_code(vec![Instruction::op_ds(opcode, Register(0), Register(1))]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for utility op: {}", name);
    }
}

#[test]
fn test_lower_for_loop_pattern() {
    // Realistic for-loop pattern: GetIter + ForIter
    let code = make_code(vec![
        // iter = iter(list)
        Instruction::op_ds(Opcode::GetIter, Register(1), Register(0)),
        // item = next(iter), jump on StopIteration
        Instruction::op_di(Opcode::ForIter, Register(2), 5),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::GetIter { dst: 1, src: 0, .. }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::ForIter {
            dst: 2,
            iter: 1,
            offset: 5,
            ..
        }
    ));
}

#[test]
fn test_lower_container_access_sequence() {
    // Pattern: list[0], list[1] = value, len(list)
    let code = make_code(vec![
        Instruction::op_dss(Opcode::GetItem, Register(2), Register(0), Register(1)),
        Instruction::op_dss(Opcode::SetItem, Register(3), Register(0), Register(4)),
        Instruction::op_ds(Opcode::Len, Register(5), Register(0)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(ir[0], TemplateInstruction::GetItem { .. }));
    assert!(matches!(ir[1], TemplateInstruction::SetItem { .. }));
    assert!(matches!(ir[2], TemplateInstruction::Len { .. }));
}

#[test]
fn test_lower_negative_for_iter_offset() {
    // ForIter with negative offset (backward jump for continue)
    let code = make_code(vec![Instruction::op_di(
        Opcode::ForIter,
        Register(2),
        (-5i16) as u16,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::ForIter { offset: -5, .. }
    ));
}

// =========================================================================
// Phase 13: Container Building Operations
// =========================================================================

#[test]
fn test_lower_build_list_basic() {
    // BuildList: dst = [r(src1)..r(src1+src2)]
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BuildList,
        Register(0),
        Register(1),
        Register(3), // 3 elements starting at r1
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildList {
            dst: 0,
            start: 1,
            count: 3,
            ..
        }
    ));
}

#[test]
fn test_lower_build_tuple_basic() {
    // BuildTuple: dst = (r(src1)..r(src1+src2))
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BuildTuple,
        Register(2),
        Register(4),
        Register(5), // 5 elements starting at r4
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildTuple {
            dst: 2,
            start: 4,
            count: 5,
            ..
        }
    ));
}

#[test]
fn test_lower_build_set_basic() {
    // BuildSet: dst = {r(src1)..r(src1+src2)}
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BuildSet,
        Register(5),
        Register(0),
        Register(2),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildSet {
            dst: 5,
            start: 0,
            count: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_build_dict_basic() {
    // BuildDict: dst = {} with src2 key-value pairs starting at src1
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BuildDict,
        Register(10),
        Register(0),
        Register(4), // 4 key-value pairs = 8 registers
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildDict {
            dst: 10,
            start: 0,
            count: 4,
            ..
        }
    ));
}

#[test]
fn test_lower_build_string_basic() {
    // BuildString: dst = "".join(r(src1)..r(src1+src2))
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BuildString,
        Register(3),
        Register(0),
        Register(5),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildString {
            dst: 3,
            start: 0,
            count: 5,
            ..
        }
    ));
}

#[test]
fn test_lower_build_slice_basic() {
    // BuildSlice: dst = slice(src1, src2)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BuildSlice,
        Register(6),
        Register(1),
        Register(4),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildSlice {
            dst: 6,
            start: 1,
            stop: 4,
            ..
        }
    ));
}

#[test]
fn test_lower_list_append_basic() {
    // ListAppend: src1.append(src2)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::ListAppend,
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
        TemplateInstruction::ListAppend {
            list: 1,
            value: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_set_add_basic() {
    // SetAdd: src1.add(src2)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::SetAdd,
        Register(0),
        Register(3),
        Register(5),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::SetAdd {
            set: 3,
            value: 5,
            ..
        }
    ));
}

#[test]
fn test_lower_dict_set_basic() {
    // DictSet: src1[dst] = src2 (key in dst field)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::DictSet,
        Register(2), // key
        Register(1), // dict
        Register(3), // value
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::DictSet {
            dict: 1,
            key: 2,
            value: 3,
            ..
        }
    ));
}

#[test]
fn test_lower_unpack_sequence_basic() {
    // UnpackSequence: r(dst)..r(dst+src2) = unpack(src1)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::UnpackSequence,
        Register(0),
        Register(5),
        Register(3), // 3 elements
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::UnpackSequence {
            dst: 0,
            src: 5,
            count: 3,
            ..
        }
    ));
}

#[test]
fn test_lower_unpack_ex_basic() {
    // UnpackEx: unpack with *rest - before/after encoded in src2
    // 2 elements before, 1 after = 0x21
    let code = make_code(vec![Instruction::op_dss(
        Opcode::UnpackEx,
        Register(0),
        Register(10),
        Register(0x21), // before=2, after=1
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::UnpackEx {
            dst: 0,
            src: 10,
            before: 2,
            after: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_all_build_container_ops() {
    // Verify all container build opcodes produce output
    let build_ops = [
        ("BuildList", Opcode::BuildList),
        ("BuildTuple", Opcode::BuildTuple),
        ("BuildSet", Opcode::BuildSet),
        ("BuildDict", Opcode::BuildDict),
        ("BuildString", Opcode::BuildString),
        ("BuildSlice", Opcode::BuildSlice),
    ];

    for (name, opcode) in build_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for build op: {}", name);
    }
}

#[test]
fn test_lower_all_container_mutation_ops() {
    // Verify list/set/dict mutation opcodes produce output
    let mutation_ops = [
        ("ListAppend", Opcode::ListAppend),
        ("SetAdd", Opcode::SetAdd),
        ("DictSet", Opcode::DictSet),
    ];

    for (name, opcode) in mutation_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for mutation op: {}", name);
    }
}

#[test]
fn test_lower_all_unpack_ops() {
    // Verify unpack opcodes produce output
    let unpack_ops = [
        ("UnpackSequence", Opcode::UnpackSequence),
        ("UnpackEx", Opcode::UnpackEx),
    ];

    for (name, opcode) in unpack_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for unpack op: {}", name);
    }
}

#[test]
fn test_lower_list_comprehension_pattern() {
    // Pattern: build list, append in loop
    let code = make_code(vec![
        Instruction::op_dss(Opcode::BuildList, Register(0), Register(1), Register(0)), // empty list
        Instruction::op_dss(Opcode::ListAppend, Register(0), Register(0), Register(5)), // append value
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildList {
            dst: 0,
            count: 0,
            ..
        }
    ));
    assert!(matches!(ir[1], TemplateInstruction::ListAppend { .. }));
}

#[test]
fn test_lower_tuple_unpacking_pattern() {
    // Pattern: build tuple, unpack
    let code = make_code(vec![
        Instruction::op_dss(Opcode::BuildTuple, Register(0), Register(1), Register(3)),
        Instruction::op_dss(
            Opcode::UnpackSequence,
            Register(5),
            Register(0),
            Register(3),
        ),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildTuple { count: 3, .. }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::UnpackSequence { count: 3, .. }
    ));
}

#[test]
fn test_lower_dict_building_pattern() {
    // Pattern: build dict, set items
    let code = make_code(vec![
        Instruction::op_dss(Opcode::BuildDict, Register(0), Register(1), Register(0)), // empty dict
        Instruction::op_dss(Opcode::DictSet, Register(2), Register(0), Register(3)), // dict[key] = value
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildDict {
            dst: 0,
            count: 0,
            ..
        }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::DictSet {
            dict: 0,
            key: 2,
            value: 3,
            ..
        }
    ));
}

#[test]
fn test_lower_unpack_ex_edge_cases() {
    // Test UnpackEx with different before/after combinations
    let cases = [
        (0x00, 0, 0),  // no before, no after (all in *rest)
        (0x10, 1, 0),  // 1 before, 0 after
        (0x01, 0, 1),  // 0 before, 1 after
        (0x23, 2, 3),  // 2 before, 3 after
        (0xF0, 15, 0), // max before, 0 after
        (0x0F, 0, 15), // 0 before, max after
    ];

    for (encoded, expected_before, expected_after) in cases {
        let code = make_code(vec![Instruction::op_dss(
            Opcode::UnpackEx,
            Register(0),
            Register(1),
            Register(encoded),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1);
        match &ir[0] {
            TemplateInstruction::UnpackEx { before, after, .. } => {
                assert_eq!(
                    *before, expected_before,
                    "before mismatch for encoded 0x{:02X}",
                    encoded
                );
                assert_eq!(
                    *after, expected_after,
                    "after mismatch for encoded 0x{:02X}",
                    encoded
                );
            }
            _ => panic!("Expected UnpackEx"),
        }
    }
}

#[test]
fn test_lower_empty_container_builds() {
    // Test building empty containers (count = 0)
    let empty_ops = [
        ("BuildList", Opcode::BuildList),
        ("BuildTuple", Opcode::BuildTuple),
        ("BuildSet", Opcode::BuildSet),
        ("BuildDict", Opcode::BuildDict),
    ];

    for (name, opcode) in empty_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(0), // count = 0
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for empty {}", name);
    }
}

// =========================================================================
// Phase 14: Function Call Operations
// =========================================================================

#[test]
fn test_lower_call_basic() {
    // Call: dst = func(args...)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::Call,
        Register(0),
        Register(1),
        Register(3), // 3 args
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::Call {
            dst: 0,
            func: 1,
            argc: 3,
            ..
        }
    ));
}

#[test]
fn test_lower_call_kw_basic() {
    // CallKw: dst = func(args..., **kwargs)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::CallKw,
        Register(5),
        Register(2),
        Register(4),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::CallKw {
            dst: 5,
            func: 2,
            argc: 4,
            ..
        }
    ));
}

#[test]
fn test_lower_call_method_basic() {
    // CallMethod: dst = obj.method(args...)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::CallMethod,
        Register(0),
        Register(3),
        Register(2),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::CallMethod {
            dst: 0,
            method: 3,
            argc: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_tail_call_basic() {
    // TailCall: reuse current frame
    let code = make_code(vec![Instruction::op_dss(
        Opcode::TailCall,
        Register(0),
        Register(1),
        Register(4),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::TailCall {
            func: 1,
            argc: 4,
            ..
        }
    ));
}

#[test]
fn test_lower_make_function_basic() {
    // MakeFunction: dst = function(code_idx)
    let code = make_code(vec![Instruction::op_di(
        Opcode::MakeFunction,
        Register(0),
        5,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::MakeFunction {
            dst: 0,
            code_idx: 5,
            ..
        }
    ));
}

#[test]
fn test_lower_make_closure_basic() {
    // MakeClosure: dst = closure(code_idx)
    let code = make_code(vec![Instruction::op_di(
        Opcode::MakeClosure,
        Register(3),
        10,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::MakeClosure {
            dst: 3,
            code_idx: 10,
            ..
        }
    ));
}

#[test]
fn test_lower_call_ex_basic() {
    // CallEx: dst = func(*args_tuple, **kwargs_dict)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::CallEx,
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
        TemplateInstruction::CallEx {
            dst: 0,
            func: 1,
            args_tuple: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_build_tuple_unpack_basic() {
    // BuildTupleUnpack: dst = (*src1, *src2, ...)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BuildTupleUnpack,
        Register(5),
        Register(0),
        Register(3),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildTupleUnpack {
            dst: 5,
            start: 0,
            count: 3,
            ..
        }
    ));
}

#[test]
fn test_lower_build_dict_unpack_basic() {
    // BuildDictUnpack: dst = {**src1, **src2, ...}
    let code = make_code(vec![Instruction::op_dss(
        Opcode::BuildDictUnpack,
        Register(10),
        Register(0),
        Register(2),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BuildDictUnpack {
            dst: 10,
            start: 0,
            count: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_all_call_ops() {
    // Verify all call opcodes produce output
    let call_ops = [
        ("Call", Opcode::Call),
        ("CallKw", Opcode::CallKw),
        ("CallMethod", Opcode::CallMethod),
        ("TailCall", Opcode::TailCall),
        ("CallEx", Opcode::CallEx),
    ];

    for (name, opcode) in call_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for call op: {}", name);
    }
}

#[test]
fn test_lower_all_function_creation_ops() {
    // Verify function creation opcodes produce output
    let fn_ops = [
        ("MakeFunction", Opcode::MakeFunction),
        ("MakeClosure", Opcode::MakeClosure),
    ];

    for (name, opcode) in fn_ops {
        let code = make_code(vec![Instruction::op_di(opcode, Register(0), 5)]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for function op: {}", name);
    }
}

#[test]
fn test_lower_all_unpack_build_ops() {
    // Verify unpack build opcodes produce output
    let unpack_ops = [
        ("BuildTupleUnpack", Opcode::BuildTupleUnpack),
        ("BuildDictUnpack", Opcode::BuildDictUnpack),
    ];

    for (name, opcode) in unpack_ops {
        let code = make_code(vec![Instruction::op_dss(
            opcode,
            Register(0),
            Register(1),
            Register(2),
        )]);
        let speculation = NoSpeculation;
        let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());
        let ir = lowerer.lower(&code);
        assert_eq!(ir.len(), 1, "Failed for unpack op: {}", name);
    }
}

#[test]
fn test_lower_call_with_no_args() {
    // Call with 0 arguments
    let code = make_code(vec![Instruction::op_dss(
        Opcode::Call,
        Register(0),
        Register(1),
        Register(0), // 0 args
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(ir[0], TemplateInstruction::Call { argc: 0, .. }));
}

#[test]
fn test_lower_function_call_sequence() {
    // Typical pattern: make function, then call
    let code = make_code(vec![
        Instruction::op_di(Opcode::MakeFunction, Register(0), 1),
        Instruction::op_dss(Opcode::Call, Register(1), Register(0), Register(2)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(ir[0], TemplateInstruction::MakeFunction { .. }));
    assert!(matches!(ir[1], TemplateInstruction::Call { .. }));
}

// =========================================================================
// Exception Handling Tests (Phase 16)
// =========================================================================

#[test]
fn test_lower_raise_basic() {
    // Raise: raise exc_reg
    let code = make_code(vec![Instruction::op_ds(
        Opcode::Raise,
        Register(5),
        Register(0), // unused src
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(ir[0], TemplateInstruction::Raise { exc: 5, .. }));
}

#[test]
fn test_lower_reraise() {
    // Reraise: bare raise statement (re-raise current exception)
    let code = make_code(vec![Instruction::op(Opcode::Reraise)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(ir[0], TemplateInstruction::Reraise { .. }));
}

#[test]
fn test_lower_raise_from() {
    // RaiseFrom: raise exc from cause
    let code = make_code(vec![Instruction::op_ds(
        Opcode::RaiseFrom,
        Register(1),
        Register(2),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::RaiseFrom {
            exc: 1,
            cause: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_pop_except_handler() {
    // PopExceptHandler: pop exception handler from handler stack
    let code = make_code(vec![Instruction::op_di(
        Opcode::PopExceptHandler,
        Register(0),
        42,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::PopExceptHandler {
            handler_idx: 42,
            ..
        }
    ));
}

#[test]
fn test_lower_exception_match() {
    // ExceptionMatch: dst = isinstance(exc, type)
    let code = make_code(vec![Instruction::op_ds(
        Opcode::ExceptionMatch,
        Register(0),
        Register(3),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::ExceptionMatch {
            dst: 0,
            exc_type: 3,
            ..
        }
    ));
}

#[test]
fn test_lower_load_exception() {
    // LoadException: dst = current_exception
    let code = make_code(vec![Instruction::op_d(Opcode::LoadException, Register(7))]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadException { dst: 7, .. }
    ));
}

#[test]
fn test_lower_push_exc_info() {
    // PushExcInfo: push exception info to stack
    let code = make_code(vec![Instruction::op(Opcode::PushExcInfo)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(ir[0], TemplateInstruction::PushExcInfo { .. }));
}

#[test]
fn test_lower_pop_exc_info() {
    // PopExcInfo: pop exception info from stack
    let code = make_code(vec![Instruction::op(Opcode::PopExcInfo)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(ir[0], TemplateInstruction::PopExcInfo { .. }));
}

#[test]
fn test_lower_has_exc_info() {
    // HasExcInfo: dst = has_pending_exception()
    let code = make_code(vec![Instruction::op_d(Opcode::HasExcInfo, Register(0))]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::HasExcInfo { dst: 0, .. }
    ));
}

#[test]
fn test_lower_clear_exception() {
    // ClearException: clear exception state
    let code = make_code(vec![Instruction::op(Opcode::ClearException)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(ir[0], TemplateInstruction::ClearException { .. }));
}

#[test]
fn test_lower_end_finally() {
    // EndFinally: end finally block
    let code = make_code(vec![Instruction::op(Opcode::EndFinally)]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(ir[0], TemplateInstruction::EndFinally { .. }));
}

#[test]
fn test_lower_try_except_pattern() {
    // Typical pattern: try block with exception handling
    // LoadException, ExceptionMatch, conditional logic
    let code = make_code(vec![
        Instruction::op_d(Opcode::LoadException, Register(0)),
        Instruction::op_ds(Opcode::ExceptionMatch, Register(1), Register(2)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadException { dst: 0, .. }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::ExceptionMatch {
            dst: 1,
            exc_type: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_finally_block_pattern() {
    // Typical finally block pattern: PushExcInfo, ... finally logic ..., PopExcInfo, EndFinally
    let code = make_code(vec![
        Instruction::op(Opcode::PushExcInfo),
        Instruction::op_d(Opcode::HasExcInfo, Register(0)),
        Instruction::op(Opcode::PopExcInfo),
        Instruction::op(Opcode::EndFinally),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 4);
    assert!(matches!(ir[0], TemplateInstruction::PushExcInfo { .. }));
    assert!(matches!(
        ir[1],
        TemplateInstruction::HasExcInfo { dst: 0, .. }
    ));
    assert!(matches!(ir[2], TemplateInstruction::PopExcInfo { .. }));
    assert!(matches!(ir[3], TemplateInstruction::EndFinally { .. }));
}

#[test]
fn test_lower_chained_exception_pattern() {
    // Pattern: raise from (exception chaining)
    // LoadException to get current, RaiseFrom with new exc and cause
    let code = make_code(vec![
        Instruction::op_d(Opcode::LoadException, Register(1)),
        Instruction::op_ds(Opcode::RaiseFrom, Register(0), Register(1)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(
        ir[0],
        TemplateInstruction::LoadException { dst: 1, .. }
    ));
    assert!(matches!(
        ir[1],
        TemplateInstruction::RaiseFrom {
            exc: 0,
            cause: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_exception_handler_cleanup() {
    // Pattern: handler cleanup with ClearException and PopExceptHandler
    let code = make_code(vec![
        Instruction::op(Opcode::ClearException),
        Instruction::op_di(Opcode::PopExceptHandler, Register(0), 0),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(ir[0], TemplateInstruction::ClearException { .. }));
    assert!(matches!(
        ir[1],
        TemplateInstruction::PopExceptHandler { handler_idx: 0, .. }
    ));
}

// =========================================================================
// Phase 17: Generator Operations Tests
// =========================================================================

#[test]
fn test_lower_yield() {
    // Basic yield: yield value
    let code = make_code(vec![Instruction::op_d(Opcode::Yield, Register(0))]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(ir[0], TemplateInstruction::Yield { value: 0, .. }));
}

#[test]
fn test_lower_yield_from() {
    // Yield from: yield from sub_gen
    let code = make_code(vec![Instruction::op_ds(
        Opcode::YieldFrom,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::YieldFrom {
            dst: 0,
            iter: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_generator_pattern() {
    // Common generator pattern: yield then get sent value
    let code = make_code(vec![
        Instruction::op_d(Opcode::Yield, Register(0)),
        Instruction::op_d(Opcode::Yield, Register(1)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(ir[0], TemplateInstruction::Yield { value: 0, .. }));
    assert!(matches!(ir[1], TemplateInstruction::Yield { value: 1, .. }));
}

// =========================================================================
// Phase 18: Context Manager Operations Tests
// =========================================================================

#[test]
fn test_lower_before_with() {
    // Enter context manager: with mgr as val
    let code = make_code(vec![Instruction::op_ds(
        Opcode::BeforeWith,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::BeforeWith { dst: 0, mgr: 1, .. }
    ));
}

#[test]
fn test_lower_exit_with() {
    // Normal exit from with block
    let code = make_code(vec![Instruction::op_ds(
        Opcode::ExitWith,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::ExitWith { dst: 0, mgr: 1, .. }
    ));
}

#[test]
fn test_lower_with_cleanup() {
    // Exception exit from with block
    let code = make_code(vec![Instruction::op_ds(
        Opcode::WithCleanup,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::WithCleanup { dst: 0, mgr: 1, .. }
    ));
}

#[test]
fn test_lower_with_block_pattern() {
    // Complete with block pattern: enter, body, exit
    let code = make_code(vec![
        Instruction::op_ds(Opcode::BeforeWith, Register(0), Register(1)),
        Instruction::op(Opcode::Nop), // body placeholder
        Instruction::op_ds(Opcode::ExitWith, Register(2), Register(1)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(ir[0], TemplateInstruction::BeforeWith { .. }));
    assert!(matches!(ir[1], TemplateInstruction::Nop { .. }));
    assert!(matches!(ir[2], TemplateInstruction::ExitWith { .. }));
}

// =========================================================================
// Phase 19: Import Operations Tests
// =========================================================================

#[test]
fn test_lower_import_name() {
    // Import module: import foo
    let code = make_code(vec![Instruction::op_di(
        Opcode::ImportName,
        Register(0),
        42,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::ImportName {
            dst: 0,
            name_idx: 42,
            ..
        }
    ));
}

#[test]
fn test_lower_import_from() {
    // Import from: from foo import bar
    let code = make_code(vec![Instruction::op_di(
        Opcode::ImportFrom,
        Register(0),
        10,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::ImportFrom {
            dst: 0,
            name_idx: 10,
            ..
        }
    ));
}

#[test]
fn test_lower_import_star() {
    // Import star: from foo import *
    let code = make_code(vec![Instruction::op_ds(
        Opcode::ImportStar,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::ImportStar { module: 1, .. }
    ));
}

#[test]
fn test_lower_import_pattern() {
    // Common import pattern: import then extract
    let code = make_code(vec![
        Instruction::op_di(Opcode::ImportName, Register(0), 1),
        Instruction::op_di(Opcode::ImportFrom, Register(1), 2),
        Instruction::op_di(Opcode::ImportFrom, Register(2), 3),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(ir[0], TemplateInstruction::ImportName { .. }));
    assert!(matches!(ir[1], TemplateInstruction::ImportFrom { .. }));
    assert!(matches!(ir[2], TemplateInstruction::ImportFrom { .. }));
}

// =========================================================================
// Phase 20: Pattern Matching Operations Tests (PEP 634)
// =========================================================================

#[test]
fn test_lower_match_class() {
    // Match class pattern: case Point(x, y)
    let code = make_code(vec![Instruction::op_dss(
        Opcode::MatchClass,
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
        TemplateInstruction::MatchClass {
            dst: 0,
            subject: 1,
            cls: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_match_mapping() {
    // Match mapping pattern: case {"key": value}
    let code = make_code(vec![Instruction::op_ds(
        Opcode::MatchMapping,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::MatchMapping {
            dst: 0,
            subject: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_match_sequence() {
    // Match sequence pattern: case [a, b, c]
    let code = make_code(vec![Instruction::op_ds(
        Opcode::MatchSequence,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::MatchSequence {
            dst: 0,
            subject: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_match_keys() {
    // Extract values from mapping by keys
    let code = make_code(vec![Instruction::op_dss(
        Opcode::MatchKeys,
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
        TemplateInstruction::MatchKeys {
            dst: 0,
            mapping: 1,
            keys: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_copy_dict_without_keys() {
    // Copy dict for **rest capture
    let code = make_code(vec![Instruction::op_dss(
        Opcode::CopyDictWithoutKeys,
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
        TemplateInstruction::CopyDictWithoutKeys {
            dst: 0,
            mapping: 1,
            keys: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_get_match_args() {
    // Get __match_args__ for positional class pattern
    let code = make_code(vec![Instruction::op_ds(
        Opcode::GetMatchArgs,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::GetMatchArgs {
            dst: 0,
            subject: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_match_class_pattern() {
    // Complete class pattern matching sequence
    let code = make_code(vec![
        Instruction::op_ds(Opcode::GetMatchArgs, Register(0), Register(1)),
        Instruction::op_dss(Opcode::MatchClass, Register(2), Register(1), Register(3)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(ir[0], TemplateInstruction::GetMatchArgs { .. }));
    assert!(matches!(ir[1], TemplateInstruction::MatchClass { .. }));
}

#[test]
fn test_lower_match_mapping_pattern() {
    // Complete mapping pattern matching sequence with **rest
    let code = make_code(vec![
        Instruction::op_ds(Opcode::MatchMapping, Register(0), Register(1)),
        Instruction::op_dss(Opcode::MatchKeys, Register(2), Register(1), Register(3)),
        Instruction::op_dss(
            Opcode::CopyDictWithoutKeys,
            Register(4),
            Register(1),
            Register(3),
        ),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(ir[0], TemplateInstruction::MatchMapping { .. }));
    assert!(matches!(ir[1], TemplateInstruction::MatchKeys { .. }));
    assert!(matches!(
        ir[2],
        TemplateInstruction::CopyDictWithoutKeys { .. }
    ));
}

// =========================================================================
// Phase 21: Async/Coroutine Operations Tests (PEP 492/525/530)
// =========================================================================

#[test]
fn test_lower_get_awaitable() {
    // Get awaitable for await expression
    let code = make_code(vec![Instruction::op_ds(
        Opcode::GetAwaitable,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::GetAwaitable { dst: 0, obj: 1, .. }
    ));
}

#[test]
fn test_lower_get_aiter() {
    // Get async iterator: async for x in aiter
    let code = make_code(vec![Instruction::op_ds(
        Opcode::GetAIter,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::GetAIter { dst: 0, obj: 1, .. }
    ));
}

#[test]
fn test_lower_get_anext() {
    // Get next from async iterator
    let code = make_code(vec![Instruction::op_ds(
        Opcode::GetANext,
        Register(0),
        Register(1),
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::GetANext {
            dst: 0,
            iter: 1,
            ..
        }
    ));
}

#[test]
fn test_lower_end_async_for() {
    // Handle StopAsyncIteration
    let code = make_code(vec![Instruction::op_di(
        Opcode::EndAsyncFor,
        Register(0),
        100,
    )]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 1);
    assert!(matches!(
        ir[0],
        TemplateInstruction::EndAsyncFor {
            dst: 0,
            // Target = (0 + 4) + (100 * 4) = 404
            target: 404,
            ..
        }
    ));
}

#[test]
fn test_lower_send() {
    // Send value to coroutine/generator
    let code = make_code(vec![Instruction::op_dss(
        Opcode::Send,
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
        TemplateInstruction::Send {
            dst: 0,
            generator: 1,
            value: 2,
            ..
        }
    ));
}

#[test]
fn test_lower_async_for_pattern() {
    // Complete async for loop pattern
    let code = make_code(vec![
        Instruction::op_ds(Opcode::GetAIter, Register(0), Register(1)),
        Instruction::op_ds(Opcode::GetANext, Register(2), Register(0)),
        Instruction::op_ds(Opcode::GetAwaitable, Register(3), Register(2)),
        Instruction::op_di(Opcode::EndAsyncFor, Register(3), 50),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 4);
    assert!(matches!(ir[0], TemplateInstruction::GetAIter { .. }));
    assert!(matches!(ir[1], TemplateInstruction::GetANext { .. }));
    assert!(matches!(ir[2], TemplateInstruction::GetAwaitable { .. }));
    assert!(matches!(ir[3], TemplateInstruction::EndAsyncFor { .. }));
}

#[test]
fn test_lower_coroutine_send_pattern() {
    // Coroutine send/receive pattern
    let code = make_code(vec![
        Instruction::op_ds(Opcode::GetAwaitable, Register(0), Register(1)),
        Instruction::op_dss(Opcode::Send, Register(2), Register(0), Register(3)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(ir[0], TemplateInstruction::GetAwaitable { .. }));
    assert!(matches!(ir[1], TemplateInstruction::Send { .. }));
}

// =========================================================================
// Cross-Phase Integration Tests
// =========================================================================

#[test]
fn test_lower_async_generator_pattern() {
    // Async generator combining yield and await
    let code = make_code(vec![
        Instruction::op_ds(Opcode::GetAwaitable, Register(0), Register(1)),
        Instruction::op_d(Opcode::Yield, Register(0)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 2);
    assert!(matches!(ir[0], TemplateInstruction::GetAwaitable { .. }));
    assert!(matches!(ir[1], TemplateInstruction::Yield { .. }));
}

#[test]
fn test_lower_async_with_pattern() {
    // Async context manager pattern
    let code = make_code(vec![
        Instruction::op_ds(Opcode::BeforeWith, Register(0), Register(1)),
        Instruction::op_ds(Opcode::GetAwaitable, Register(2), Register(0)),
        Instruction::op_ds(Opcode::ExitWith, Register(3), Register(1)),
    ]);
    let speculation = NoSpeculation;
    let mut lowerer = BytecodeLowerer::new(&speculation, 0, LoweringConfig::default());

    let ir = lowerer.lower(&code);
    assert_eq!(ir.len(), 3);
    assert!(matches!(ir[0], TemplateInstruction::BeforeWith { .. }));
    assert!(matches!(ir[1], TemplateInstruction::GetAwaitable { .. }));
    assert!(matches!(ir[2], TemplateInstruction::ExitWith { .. }));
}
