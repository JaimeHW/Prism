//! Runtime helper entry points called by Tier 1 native templates.
//!
//! These helpers are deliberately narrow ABI shims. They do not implement
//! fallback semantics themselves; on a miss they tell native code to deopt so
//! the interpreter resumes at the original bytecode and raises or continues
//! with exact Python behavior.

use prism_code::CodeObject;

use crate::VirtualMachine;
use crate::import::ModuleObject;
use crate::jit_executor::JitFrameState;

const TIER1_HELPER_SUCCESS: u64 = 0;
const TIER1_HELPER_DEOPT: u64 = 1;

pub(crate) fn tier1_load_global_addr() -> u64 {
    tier1_load_global as *const () as usize as u64
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
