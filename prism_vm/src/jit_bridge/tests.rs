use super::*;

#[test]
fn test_bridge_config_default() {
    let config = BridgeConfig::default();
    assert!(config.enabled);
    assert_eq!(config.tier1_threshold, 1_000);
    assert_eq!(config.tier2_threshold, 10_000);
    assert_eq!(config.max_code_size, 64 * 1024 * 1024);
}

#[test]
fn test_bridge_config_testing() {
    let config = BridgeConfig::for_testing();
    assert!(config.enabled);
    assert_eq!(config.tier1_threshold, 10);
    assert!(!config.background_compilation);
    assert_eq!(config.max_code_size, 1024 * 1024);
}

#[test]
fn test_bridge_creation() {
    let bridge = JitBridge::new(BridgeConfig::for_testing());
    assert!(bridge.is_enabled());
    assert_eq!(bridge.compiled_count(), 0);
}

#[test]
fn test_bridge_disabled() {
    let bridge = JitBridge::new(BridgeConfig::disabled());
    assert!(!bridge.is_enabled());
    assert!(bridge.lookup(123).is_none());
}

#[test]
fn test_compilation_state() {
    let state = CompilationState::Interpreted;
    assert_eq!(state, CompilationState::Interpreted);
}

#[test]
fn test_compile_tier2_sets_tier_and_raw_value_abi() {
    use prism_code::{Instruction, Opcode, Register};

    let mut bridge = JitBridge::new(BridgeConfig::for_testing());
    let mut code = CodeObject::new("tier2_test", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        Instruction::op_d(Opcode::LoadNone, Register::new(0)),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();
    let code = Arc::new(code);

    let entry = bridge.compile_tier2(&code).unwrap();
    assert_eq!(entry.tier(), 2);
    assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);
}

#[test]
fn test_compile_tier2_rejects_unsupported_generic_arithmetic() {
    use prism_code::{Constant, Instruction, Opcode, Register};
    use prism_core::Value;

    let mut bridge = JitBridge::new(BridgeConfig::for_testing());
    let mut code = CodeObject::new("tier2_unsupported", "<test>");
    code.register_count = 3;
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
        Instruction::op_dss(
            Opcode::Add,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();
    code.constants = vec![Value::int(10).unwrap(), Value::int(20).unwrap()]
        .into_iter()
        .map(Constant::Value)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let code = Arc::new(code);

    let err = bridge
        .compile_tier2(&code)
        .expect_err("unsupported generic arithmetic must not install tier2 code");
    assert!(err.contains("does not support operator"));
}

#[test]
fn test_compile_tier2_accepts_parameterized_graphs() {
    use prism_code::{Instruction, Opcode, Register};

    let mut bridge = JitBridge::new(BridgeConfig::for_testing());
    let mut code = CodeObject::new("tier2_param", "<test>");
    code.register_count = 1;
    code.arg_count = 1;
    code.instructions =
        vec![Instruction::op_d(Opcode::Return, Register::new(0))].into_boxed_slice();
    let code = Arc::new(code);

    let entry = bridge.compile_tier2(&code).expect(
        "parameterized code should compile once parameters are materialized from frame state",
    );
    assert_eq!(entry.tier(), 2);
    assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);
}

#[test]
fn test_compile_tier2_accepts_branching_control_flow() {
    use prism_code::{Instruction, Opcode, Register};

    let mut bridge = JitBridge::new(BridgeConfig::for_testing());
    let mut code = CodeObject::new("tier2_branch", "<test>");
    code.register_count = 1;
    code.arg_count = 1;
    code.instructions = vec![
        // if not r0: jump to offset 2
        Instruction::op_di(Opcode::JumpIfFalse, Register::new(0), 1),
        Instruction::op_d(Opcode::Return, Register::new(0)),
        Instruction::op_d(Opcode::LoadNone, Register::new(0)),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();
    let code = Arc::new(code);

    let entry = bridge
        .compile_tier2(&code)
        .expect("branching code should lower through If/Projection Tier2 support");
    assert_eq!(entry.tier(), 2);
    assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);
}

#[test]
fn test_compile_tier2_rejects_uninitialized_register_reads() {
    use prism_code::{Instruction, Opcode, Register};

    let mut bridge = JitBridge::new(BridgeConfig::for_testing());
    let mut code = CodeObject::new("tier2_uninit", "<test>");
    code.register_count = 3;
    code.instructions = vec![
        Instruction::op_dss(
            Opcode::Add,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();
    let code = Arc::new(code);

    let err = bridge
        .compile_tier2(&code)
        .expect_err("uninitialized register reads should fail translation");
    assert!(err.contains("uninitialized register"));
}

#[test]
fn test_gc_stack_bitmap_encodes_rbp_relative_slots() {
    let bitmap = gc_stack_bitmap(&[-8, -24, -8]).expect("valid aligned slots should encode");
    assert_eq!(bitmap, 0b101);
}

#[test]
fn test_gc_stack_bitmap_rejects_non_negative_and_unaligned_slots() {
    let err = gc_stack_bitmap(&[8]).expect_err("positive offsets are not RBP-relative locals");
    assert!(err.contains("non-negative"));

    let err = gc_stack_bitmap(&[-10]).expect_err("unaligned offsets must be rejected");
    assert!(err.contains("unaligned"));
}

#[test]
fn test_build_stack_map_for_entry_and_collect_deopt_sites() {
    let entries = vec![
        Tier2StackMapEntry {
            code_offset: 0x10,
            bc_offset: Some(3),
            gc_slots: vec![-8, -24],
            gc_regs: vec![Gpr::Rbx, Gpr::R12],
        },
        Tier2StackMapEntry {
            code_offset: 0x20,
            bc_offset: None,
            gc_slots: vec![-16],
            gc_regs: vec![Gpr::R13],
        },
        Tier2StackMapEntry {
            code_offset: 0x10,
            bc_offset: Some(3),
            gc_slots: vec![-8, -24],
            gc_regs: vec![Gpr::Rbx, Gpr::R12],
        },
    ];

    let map = build_stack_map_for_entry(0x1000, 0x80, 48, &entries)
        .expect("stack map conversion should succeed")
        .expect("non-empty stack map entries should produce metadata");
    assert_eq!(map.code_start, 0x1000);
    assert_eq!(map.code_size, 0x80);
    assert_eq!(map.frame_size, 48);
    assert_eq!(map.safepoint_count(), 3);

    let sp = map
        .lookup_offset(0x10)
        .expect("safepoint should be present");
    assert_eq!(
        sp.register_bitmap,
        (1 << Gpr::Rbx.encoding()) | (1 << Gpr::R12.encoding())
    );
    assert_eq!(sp.stack_bitmap, 0b101);

    let deopt_sites = collect_deopt_sites(&entries);
    assert_eq!(
        deopt_sites,
        vec![DeoptSite {
            code_offset: 0x10,
            bc_offset: 3
        }]
    );
}
