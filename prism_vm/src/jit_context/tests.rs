use super::*;

#[test]
fn test_jit_config_default() {
    let config = JitConfig::default();
    assert!(config.enabled);
    assert!(config.background_compilation);
    assert!(config.enable_osr);
}

#[test]
fn test_jit_config_disabled() {
    let config = JitConfig::disabled();
    assert!(!config.enabled);
}

#[test]
fn test_jit_config_testing() {
    let config = JitConfig::for_testing();
    assert!(config.enabled);
    assert!(!config.background_compilation);
    assert!(config.eager_compilation);
    assert_eq!(config.tier1_threshold, 10);
}

#[test]
fn test_jit_config_to_bridge_config_propagates_max_code_size() {
    let config = JitConfig {
        max_code_size: 2 * 1024 * 1024,
        ..JitConfig::for_testing()
    };
    let bridge = config.to_bridge_config();
    assert_eq!(bridge.max_code_size, 2 * 1024 * 1024);
}

#[test]
fn test_jit_context_creation() {
    let ctx = JitContext::with_defaults();
    assert!(ctx.is_enabled());
    assert_eq!(ctx.compiled_count(), 0);
}

#[test]
fn test_jit_context_lookup_miss() {
    let ctx = JitContext::with_defaults();
    assert!(ctx.lookup(12345).is_none());
}

#[test]
fn test_jit_stats_hit_rate() {
    let mut stats = JitStats::default();
    stats.cache_hits = 90;
    stats.cache_misses = 10;
    assert!((stats.hit_rate() - 0.9).abs() < 0.001);
}

#[test]
fn test_jit_stats_deopt_rate() {
    let mut stats = JitStats::default();
    stats.cache_hits = 100;
    stats.deopts = 5;
    assert!((stats.deopt_rate() - 0.05).abs() < 0.001);
}

#[test]
fn test_jit_context_tier_up_none() {
    let mut ctx = JitContext::with_defaults();
    let code = Arc::new(CodeObject::new("test", "<test>"));
    assert!(!ctx.handle_tier_up(&code, TierUpDecision::None));
}

#[test]
fn test_processed_result() {
    use prism_core::Value;

    let result = ProcessedResult::Return(Value::int(42).unwrap());
    match result {
        ProcessedResult::Return(v) => assert_eq!(v.as_int(), Some(42)),
        _ => panic!("Expected Return"),
    }

    let result = ProcessedResult::Resume { bc_offset: 100 };
    match result {
        ProcessedResult::Resume { bc_offset } => assert_eq!(bc_offset, 100),
        _ => panic!("Expected Resume"),
    }
}

#[test]
fn test_check_tier_up_uses_runtime_config_thresholds() {
    let ctx = JitContext::for_testing();
    let mut profiler = Profiler::new();
    let code_id = CodeId::new(42);

    for _ in 0..9 {
        profiler.record_call(code_id);
        assert_eq!(ctx.check_tier_up(&profiler, code_id), TierUpDecision::None);
    }

    profiler.record_call(code_id);
    assert_eq!(ctx.check_tier_up(&profiler, code_id), TierUpDecision::Tier1);

    for _ in 0..90 {
        profiler.record_call(code_id);
    }
    assert_eq!(ctx.check_tier_up(&profiler, code_id), TierUpDecision::Tier2);
}

#[test]
fn test_handle_tier_up_tier2_compiles_tier2_in_sync_mode() {
    use prism_code::{Instruction, Opcode, Register};
    use prism_jit::runtime::ReturnAbi;

    let mut ctx = JitContext::for_testing();
    let mut code = CodeObject::new("jit_ctx_tier2", "<test>");
    code.register_count = 1;
    code.instructions = vec![
        Instruction::op_d(Opcode::LoadNone, Register::new(0)),
        Instruction::op_d(Opcode::Return, Register::new(0)),
    ]
    .into_boxed_slice();

    let code = Arc::new(code);
    let code_id = Arc::as_ptr(&code) as u64;

    assert!(ctx.handle_tier_up(&code, TierUpDecision::Tier2));
    let entry = ctx.lookup(code_id).expect("tier2 entry should be cached");
    assert_eq!(entry.tier(), 2);
    assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);
}

#[test]
fn test_handle_tier_up_tier2_failure_is_not_retried_for_same_code() {
    use prism_code::{Constant, Instruction, Opcode, Register};
    use prism_core::Value;

    let mut ctx = JitContext::for_testing();
    let mut code = CodeObject::new("jit_ctx_tier2_fail_once", "<test>");
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
    code.constants = vec![Value::int(1).unwrap(), Value::int(2).unwrap()]
        .into_iter()
        .map(Constant::Value)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let code = Arc::new(code);

    assert!(!ctx.handle_tier_up(&code, TierUpDecision::Tier2));
    assert!(!ctx.handle_tier_up(&code, TierUpDecision::Tier2));

    let stats = ctx.stats();
    assert_eq!(stats.compilations_failed, 1);
    assert_eq!(stats.compilations_triggered, 1);
}
