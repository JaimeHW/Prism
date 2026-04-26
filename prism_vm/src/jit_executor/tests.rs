use super::*;
use prism_jit::runtime::DeoptSite;

unsafe extern "C" fn jit_stub_return_int30_bits(_state: *mut JitFrameState) -> u64 {
    Value::int(30).unwrap().to_bits()
}

unsafe extern "C" fn jit_stub_return_exit_return(_state: *mut JitFrameState) -> u64 {
    ExitReason::Return as u64
}

unsafe extern "C" fn jit_stub_return_deopt_site_index_1(_state: *mut JitFrameState) -> u64 {
    let exit = ExitReason::Deoptimize as u64;
    let deopt_index = 1u64; // Use deopt-site index fallback path.
    let reason = DeoptReason::TypeGuard as u64;
    let data = (deopt_index << 8) | reason;
    exit | (data << 8)
}

#[test]
fn test_deopt_reason_roundtrip() {
    for i in 0..=8 {
        let reason = DeoptReason::from_u8(i).unwrap();
        assert_eq!(reason as u8, i);
    }
    assert!(DeoptReason::from_u8(255).is_none());
}

#[test]
fn test_decode_jit_result() {
    // Normal return
    let (reason, _data) = decode_jit_result(0);
    assert_eq!(reason, ExitReason::Return);

    // Deopt with bc_offset
    let result = 2 | (100 << 8); // Deoptimize, offset 100
    let (reason, data) = decode_jit_result(result);
    assert_eq!(reason, ExitReason::Deoptimize);
    assert_eq!(data, 100);
}

#[test]
fn test_executor_creation() {
    let cache = Arc::new(CodeCache::new(1024 * 1024));
    let executor = JitExecutor::new(cache);
    assert!(executor.code_cache().is_empty());
}

#[test]
fn test_execute_raw_value_abi_uses_rax_bits() {
    let cache = Arc::new(CodeCache::new(1024 * 1024));
    let mut executor = JitExecutor::new(cache);

    let code = Arc::new(prism_code::CodeObject::new("jit_raw_abi_test", "<test>"));
    let mut frame = Frame::new(code, None, 0);

    let entry = CompiledEntry::new(1, jit_stub_return_int30_bits as *const u8, 1)
        .with_return_abi(ReturnAbi::RawValueBits);

    let result = executor.execute(&entry, &mut frame);
    match result {
        ExecutionResult::Return(value) => assert_eq!(value.as_int(), Some(30)),
        other => panic!("unexpected execution result: {:?}", other),
    }
}

#[test]
fn test_execute_encoded_return_reads_frame_slot_zero() {
    let cache = Arc::new(CodeCache::new(1024 * 1024));
    let mut executor = JitExecutor::new(cache);

    let code = Arc::new(prism_code::CodeObject::new(
        "jit_encoded_abi_test",
        "<test>",
    ));
    let mut frame = Frame::new(code, None, 0);
    frame.set_reg(0, Value::int(77).unwrap());

    let entry = CompiledEntry::new(2, jit_stub_return_exit_return as *const u8, 1)
        .with_return_abi(ReturnAbi::EncodedExitReason);

    let result = executor.execute(&entry, &mut frame);
    match result {
        ExecutionResult::Return(value) => assert_eq!(value.as_int(), Some(77)),
        other => panic!("unexpected execution result: {:?}", other),
    }
}

#[test]
fn test_deopt_recovery() {
    let result = ExecutionResult::Deopt {
        bc_offset: 42,
        reason: DeoptReason::TypeGuard,
    };
    let recovery = DeoptRecovery::from_result(&result).unwrap();
    assert_eq!(recovery.bc_offset, 42);
    assert_eq!(recovery.reason, DeoptReason::TypeGuard);
}

#[test]
fn test_deopt_bytecode_offsets_resume_at_instruction_index() {
    assert_eq!(normalize_deopt_resume_ip(0, 4), Some(0));
    assert_eq!(normalize_deopt_resume_ip(4, 4), Some(1));
    assert_eq!(normalize_deopt_resume_ip(12, 4), Some(3));
}

#[test]
fn test_deopt_resume_preserves_legacy_instruction_index_payloads() {
    assert_eq!(normalize_deopt_resume_ip(3, 8), Some(3));
    assert_eq!(normalize_deopt_resume_ip(9, 8), None);
    assert_eq!(normalize_deopt_resume_ip(4, 0), None);
}

#[test]
fn test_execute_deopt_uses_site_index_fallback_when_offset_out_of_range() {
    let cache = Arc::new(CodeCache::new(1024 * 1024));
    let mut executor = JitExecutor::new(cache);

    let code = Arc::new(prism_code::CodeObject::new(
        "jit_deopt_site_index_fallback",
        "<test>",
    ));
    let mut frame = Frame::new(code, None, 0);

    let entry = CompiledEntry::new(3, jit_stub_return_deopt_site_index_1 as *const u8, 1)
        .with_return_abi(ReturnAbi::EncodedExitReason)
        .with_deopt_sites(vec![
            DeoptSite {
                code_offset: 10,
                bc_offset: 7,
            },
            DeoptSite {
                code_offset: 20,
                bc_offset: 41,
            },
        ]);

    match executor.execute(&entry, &mut frame) {
        ExecutionResult::Deopt { bc_offset, reason } => {
            assert_eq!(reason, DeoptReason::TypeGuard);
            assert_eq!(bc_offset, 41);
        }
        other => panic!("unexpected execution result: {:?}", other),
    }
}

#[test]
fn test_setup_frame_state_wires_closure_env_pointer() {
    let cache = Arc::new(CodeCache::new(1024 * 1024));
    let mut executor = JitExecutor::new(cache);

    let code = Arc::new(prism_code::CodeObject::new(
        "jit_closure_pointer_test",
        "<test>",
    ));
    let closure = Arc::new(crate::frame::ClosureEnv::with_unbound_cells(2));
    let frame = Frame::with_closure(code, None, 0, Arc::clone(&closure));

    executor.setup_frame_state(&frame, std::ptr::null());
    assert_eq!(
        executor.frame_state.closure_env,
        Arc::as_ptr(&closure) as *const u64
    );
}

#[test]
fn test_setup_frame_state_clears_closure_env_without_closure() {
    let cache = Arc::new(CodeCache::new(1024 * 1024));
    let mut executor = JitExecutor::new(cache);

    let code = Arc::new(prism_code::CodeObject::new(
        "jit_no_closure_pointer_test",
        "<test>",
    ));
    let frame = Frame::new(code, None, 0);

    executor.setup_frame_state(&frame, std::ptr::null());
    assert!(executor.frame_state.closure_env.is_null());
}

#[test]
fn test_setup_frame_state_wires_global_scope_pointer() {
    let cache = Arc::new(CodeCache::new(1024 * 1024));
    let mut executor = JitExecutor::new(cache);

    let code = Arc::new(prism_code::CodeObject::new(
        "jit_global_pointer_test",
        "<test>",
    ));
    let frame = Frame::new(code, None, 0);
    let global_scope = 0x1000usize as *const u64;

    executor.setup_frame_state(&frame, global_scope);
    assert_eq!(executor.frame_state.global_scope, global_scope);
}
