//! Runtime helper entry points called by Tier 1 native templates.
//!
//! These helpers are deliberately narrow ABI shims. They do not implement
//! fallback semantics themselves; on a miss they tell native code to deopt so
//! the interpreter resumes at the original bytecode and raises or continues
//! with exact Python behavior.

use prism_code::{CodeObject, Opcode};
use prism_core::Value;
use smallvec::SmallVec;

use crate::VirtualMachine;
use crate::dispatch::{ControlFlow, get_handler};
use crate::error::RuntimeError;
use crate::import::ModuleObject;
use crate::jit_executor::JitFrameState;

const TIER1_HELPER_SUCCESS: u64 = 0;
const TIER1_HELPER_DEOPT: u64 = 1;
const TIER1_HELPER_EXCEPTION: u64 = 2;
const TIER1_HELPER_JUMP: u64 = 3;

pub(crate) fn tier1_load_global_addr() -> u64 {
    tier1_load_global as *const () as usize as u64
}

pub(crate) fn tier1_call_addr() -> u64 {
    tier1_call as *const () as usize as u64
}

pub(crate) fn tier1_bytecode_addr() -> u64 {
    tier1_execute_bytecode as *const () as usize as u64
}

/// Resolve `LOAD_GLOBAL` for Tier 1 code.
///
/// # Safety
///
/// `state` must point to a live `JitFrameState`, `state.vm_context` must be the
/// owning `VirtualMachine`, `state.code` must be the executing `CodeObject`,
/// and `out_value` must point at the destination slot in the native Tier 1
/// frame mirror.
pub(crate) unsafe extern "C" fn tier1_load_global(
    state: *mut JitFrameState,
    name_idx: u32,
    out_value: *mut u64,
) -> u64 {
    if state.is_null() || out_value.is_null() || name_idx > u16::MAX as u32 {
        return TIER1_HELPER_DEOPT;
    }

    let state = unsafe { &mut *state };
    if state.vm_context.is_null() || state.code.is_null() {
        return TIER1_HELPER_DEOPT;
    }

    let vm = unsafe { &mut *(state.vm_context as *mut VirtualMachine) };
    let code = unsafe { &*(state.code as *const CodeObject) };
    let module = if state.module.is_null() {
        None
    } else {
        Some(unsafe { &*(state.module as *const ModuleObject) })
    };

    let Some(value) = vm.load_global_for_jit(code, module, name_idx as u16) else {
        return TIER1_HELPER_DEOPT;
    };

    unsafe {
        *out_value = value.to_bits();
    }
    TIER1_HELPER_SUCCESS
}

/// Invoke a positional Python call for Tier 1 code.
///
/// Native code writes the register mirror back before entering this helper.
/// That makes `frame_base` authoritative for callable/argument reads and keeps
/// the caller frame visible to re-entrant Python code, exception handlers, and
/// traceback construction.
///
/// # Safety
///
/// `state` must point to the active `JitFrameState` for the current VM frame.
/// The helper only reads and writes slots within `state.frame_base` after
/// validating the bytecode register indexes against `state.num_registers`.
pub(crate) unsafe extern "C" fn tier1_call(
    state: *mut JitFrameState,
    dst: u32,
    func: u32,
    argc: u32,
) -> u64 {
    if state.is_null() || dst > u8::MAX as u32 || func > u8::MAX as u32 || argc > u8::MAX as u32 {
        return TIER1_HELPER_DEOPT;
    }

    let state = unsafe { &mut *state };
    if state.vm_context.is_null() || state.frame_base.is_null() {
        return TIER1_HELPER_DEOPT;
    }

    let num_registers = usize::from(state.num_registers);
    let dst = dst as usize;
    let func = func as usize;
    let argc = argc as usize;
    let Some(args_end) = dst.checked_add(1).and_then(|base| base.checked_add(argc)) else {
        return TIER1_HELPER_DEOPT;
    };
    if dst >= num_registers || func >= num_registers || args_end > num_registers {
        return TIER1_HELPER_DEOPT;
    }

    let vm = unsafe { &mut *(state.vm_context as *mut VirtualMachine) };
    sync_current_frame_ip_from_jit_state(vm, state);

    let frame_base = state.frame_base;
    let callable = unsafe { Value::from_bits(*frame_base.add(func)) };
    let mut args = SmallVec::<[Value; 8]>::with_capacity(argc);
    for index in 0..argc {
        let raw = unsafe { *frame_base.add(dst + 1 + index) };
        args.push(Value::from_bits(raw));
    }

    match crate::ops::calls::invoke_callable_value(vm, callable, &args) {
        Ok(value) => {
            let bits = value.to_bits();
            unsafe {
                *frame_base.add(dst) = bits;
            }
            mark_register_written(state, dst);
            TIER1_HELPER_SUCCESS
        }
        Err(err) => {
            sync_jit_state_ip_from_current_frame(vm, state);
            vm.record_jit_error(err);
            TIER1_HELPER_EXCEPTION
        }
    }
}

/// Execute a single fallthrough bytecode using the VM's canonical handler.
///
/// This is the exact-semantics bridge for high-impact dynamic operations while
/// Tier 1 grows specialized fast paths. Native code writes its register mirror
/// back before entry; on success the helper leaves the interpreter frame as the
/// source of truth and compiled code reloads its mirror before continuing.
///
/// # Safety
///
/// `state` must describe the active VM frame. `current_bc_offset` must identify
/// the bytecode that native code is replacing, and `expected_opcode` must match
/// the opcode in that frame.
pub(crate) unsafe extern "C" fn tier1_execute_bytecode(
    state: *mut JitFrameState,
    expected_opcode: u32,
    current_bc_offset: u32,
) -> u64 {
    if state.is_null() || expected_opcode > u8::MAX as u32 || current_bc_offset % 4 != 0 {
        return TIER1_HELPER_DEOPT;
    }

    let state = unsafe { &mut *state };
    if state.vm_context.is_null() || state.frame_base.is_null() || state.code.is_null() {
        return TIER1_HELPER_DEOPT;
    }

    let vm = unsafe { &mut *(state.vm_context as *mut VirtualMachine) };
    if vm.call_depth() == 0 {
        return TIER1_HELPER_DEOPT;
    }

    let current_index = current_bc_offset / 4;
    let inst = {
        let frame = vm.current_frame_mut();
        let code_ptr = std::sync::Arc::as_ptr(&frame.code) as *const ();
        if code_ptr != state.code {
            return TIER1_HELPER_DEOPT;
        }

        let Some(inst) = frame.code.instructions.get(current_index as usize).copied() else {
            return TIER1_HELPER_DEOPT;
        };

        if u32::from(inst.opcode()) != expected_opcode {
            return TIER1_HELPER_DEOPT;
        }

        frame.ip = current_index.saturating_add(1);
        inst
    };

    let handler = get_handler(inst.opcode());
    match handler(vm, inst) {
        ControlFlow::Continue => {
            sync_jit_state_ip_from_current_frame(vm, state);
            TIER1_HELPER_SUCCESS
        }
        ControlFlow::Jump(offset) => match apply_helper_relative_jump(vm, offset) {
            Ok(()) => {
                sync_jit_state_ip_from_current_frame(vm, state);
                TIER1_HELPER_JUMP
            }
            Err(err) => {
                sync_jit_state_ip_from_current_frame(vm, state);
                vm.record_jit_error(err);
                TIER1_HELPER_EXCEPTION
            }
        },
        ControlFlow::Error(err) => {
            sync_jit_state_ip_from_current_frame(vm, state);
            vm.record_jit_error(err);
            TIER1_HELPER_EXCEPTION
        }
        other => {
            sync_jit_state_ip_from_current_frame(vm, state);
            vm.record_jit_error(RuntimeError::internal(format!(
                "Tier 1 bytecode helper for {:?} produced unsupported control flow {:?}",
                Opcode::from_u8(inst.opcode()),
                other
            )));
            TIER1_HELPER_EXCEPTION
        }
    }
}

#[inline]
fn apply_helper_relative_jump(vm: &mut VirtualMachine, offset: i16) -> Result<(), RuntimeError> {
    if vm.call_depth() == 0 {
        return Err(RuntimeError::internal(
            "Tier 1 bytecode helper jump with no active frame",
        ));
    }

    let frame = vm.current_frame_mut();
    let next_ip = (frame.ip as i64 + i64::from(offset)).max(0);
    frame.ip = u32::try_from(next_ip)
        .map_err(|_| RuntimeError::internal("Tier 1 bytecode helper jump target out of range"))?;
    Ok(())
}

#[inline]
fn mark_register_written(state: &mut JitFrameState, register: usize) {
    if state.written_registers.is_null() {
        return;
    }

    let word_index = register / 64;
    let bit = 1u64 << (register % 64);
    unsafe {
        let word = state.written_registers.add(word_index);
        *word |= bit;
    }
}

#[inline]
fn sync_current_frame_ip_from_jit_state(vm: &mut VirtualMachine, state: &JitFrameState) {
    if vm.call_depth() == 0 || state.code.is_null() {
        return;
    }

    let frame = vm.current_frame_mut();
    let code_ptr = std::sync::Arc::as_ptr(&frame.code) as *const ();
    if code_ptr == state.code {
        frame.ip = state.bc_offset / 4;
    }
}

#[inline]
fn sync_jit_state_ip_from_current_frame(vm: &VirtualMachine, state: &mut JitFrameState) {
    if vm.call_depth() == 0 || state.code.is_null() {
        return;
    }

    let frame = vm.current_frame();
    let code_ptr = std::sync::Arc::as_ptr(&frame.code) as *const ();
    if code_ptr == state.code {
        state.bc_offset = frame.ip.saturating_mul(4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RuntimeErrorKind;
    use crate::ops::iteration::ensure_iterator_value;
    use prism_code::{FunctionBuilder, Instruction, Opcode, Register};
    use prism_core::intern::intern;
    use std::sync::Arc;

    fn helper_frame() -> (VirtualMachine, Arc<CodeObject>, u16) {
        let mut builder = FunctionBuilder::new("tier1_call_helper");
        let abs_name = builder.add_name("abs");
        builder.emit_return(Register::new(0));
        let mut code = builder.finish();
        code.register_count = 4;
        let code = Arc::new(code);

        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::clone(&code), 0)
            .expect("frame push should succeed");
        (vm, code, abs_name)
    }

    fn state_for_current_frame(vm: &mut VirtualMachine, code: &Arc<CodeObject>) -> JitFrameState {
        let vm_context = vm as *mut VirtualMachine as *mut ();
        let frame = vm.current_frame_mut();
        JitFrameState {
            frame_base: frame.registers.as_mut_ptr() as *mut u64,
            num_registers: frame.code.register_count,
            bc_offset: 4,
            const_pool: std::ptr::null(),
            closure_env: std::ptr::null(),
            global_scope: std::ptr::null(),
            written_registers: frame.written_registers_mut_ptr(),
            vm_context,
            code: Arc::as_ptr(code) as *const (),
            module: std::ptr::null(),
        }
    }

    fn len_helper_frame() -> (VirtualMachine, Arc<CodeObject>) {
        let mut builder = FunctionBuilder::new("tier1_bytecode_helper");
        builder.emit(Instruction::op_ds(
            Opcode::Len,
            Register::new(1),
            Register::new(0),
        ));
        builder.emit_return(Register::new(1));
        let mut code = builder.finish();
        code.register_count = 2;
        let code = Arc::new(code);

        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::clone(&code), 0)
            .expect("frame push should succeed");
        (vm, code)
    }

    fn for_iter_helper_frame() -> (VirtualMachine, Arc<CodeObject>) {
        let mut builder = FunctionBuilder::new("tier1_for_iter_helper");
        builder.emit(Instruction::op_di(Opcode::ForIter, Register::new(1), 1));
        builder.emit_return(Register::new(1));
        let mut code = builder.finish();
        code.register_count = 2;
        let code = Arc::new(code);

        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::clone(&code), 0)
            .expect("frame push should succeed");
        (vm, code)
    }

    #[test]
    fn tier1_call_helper_invokes_builtin_and_writes_result() {
        let (mut vm, code, abs_name) = helper_frame();
        let abs = vm
            .load_builtin_cached(abs_name)
            .expect("abs builtin should resolve");
        {
            let frame = vm.current_frame_mut();
            frame.set_reg(1, Value::int_unchecked(-7));
            frame.set_reg(3, abs);
        }

        let mut state = state_for_current_frame(&mut vm, &code);
        let status = unsafe { tier1_call(&mut state, 0, 3, 1) };

        assert_eq!(status, TIER1_HELPER_SUCCESS);
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(7));
        assert!(vm.current_frame().local_is_written(0));
        assert!(vm.take_last_jit_error().is_none());
    }

    #[test]
    fn tier1_call_helper_preserves_runtime_error_payload() {
        let (mut vm, code, _) = helper_frame();
        {
            let frame = vm.current_frame_mut();
            frame.set_reg(3, Value::int_unchecked(42));
        }

        let mut state = state_for_current_frame(&mut vm, &code);
        let status = unsafe { tier1_call(&mut state, 0, 3, 0) };

        assert_eq!(status, TIER1_HELPER_EXCEPTION);
        let err = vm
            .take_last_jit_error()
            .expect("helper should store the real VM error");
        assert!(matches!(
            err.kind(),
            RuntimeErrorKind::TypeError { .. } | RuntimeErrorKind::NotCallable { .. }
        ));
        assert_eq!(state.bc_offset, vm.current_frame().ip * 4);
    }

    #[test]
    fn tier1_bytecode_helper_executes_fallthrough_vm_opcode() {
        let (mut vm, code) = len_helper_frame();
        vm.current_frame_mut()
            .set_reg(0, Value::string(intern("hello")));

        let mut state = state_for_current_frame(&mut vm, &code);
        let status = unsafe { tier1_execute_bytecode(&mut state, Opcode::Len as u32, 0) };

        assert_eq!(status, TIER1_HELPER_SUCCESS);
        assert_eq!(vm.current_frame().get_reg(1).as_int(), Some(5));
        assert!(vm.current_frame().local_is_written(1));
        assert_eq!(state.bc_offset, 4);
        assert!(vm.take_last_jit_error().is_none());
    }

    #[test]
    fn tier1_bytecode_helper_preserves_vm_opcode_error() {
        let (mut vm, code) = len_helper_frame();
        vm.current_frame_mut().set_reg(0, Value::int_unchecked(42));

        let mut state = state_for_current_frame(&mut vm, &code);
        let status = unsafe { tier1_execute_bytecode(&mut state, Opcode::Len as u32, 0) };

        assert_eq!(status, TIER1_HELPER_EXCEPTION);
        let err = vm
            .take_last_jit_error()
            .expect("helper should store the real VM error");
        assert!(matches!(err.kind(), RuntimeErrorKind::TypeError { .. }));
        assert_eq!(state.bc_offset, vm.current_frame().ip * 4);
    }

    #[test]
    fn tier1_bytecode_helper_applies_relative_jump() {
        let (mut vm, code) = for_iter_helper_frame();
        let iterator = ensure_iterator_value(&mut vm, Value::string(intern("")))
            .expect("empty string should produce an iterator");
        vm.current_frame_mut().set_reg(0, iterator);

        let mut state = state_for_current_frame(&mut vm, &code);
        let status = unsafe { tier1_execute_bytecode(&mut state, Opcode::ForIter as u32, 0) };

        assert_eq!(status, TIER1_HELPER_JUMP);
        assert_eq!(vm.current_frame().ip, 2);
        assert_eq!(state.bc_offset, 8);
        assert!(vm.take_last_jit_error().is_none());
    }
}
