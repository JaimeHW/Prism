//! Runtime helpers for Prism's native AOT module initialization ABI.

use std::sync::Arc;

use prism_core::Value;
use prism_core::aot::{
    AOT_IMPORT_FROM_SYMBOL, AOT_IMPORT_MODULE_SYMBOL, AOT_STORE_EXPR_SYMBOL, AotImmediate,
    AotImmediateKind, AotImportBinding, AotImportFromOp, AotImportModuleOp, AotOpStatus,
    AotOperand, AotOperandKind, AotStoreExprKind, AotStoreExprOp, AotStringRef,
};
use prism_core::intern::{intern, intern_owned, interned_by_ptr};

use crate::error::RuntimeError;
use crate::import::ModuleObject;
use crate::vm::VirtualMachine;

fn resolve_string(string: &AotStringRef) -> Result<&str, RuntimeError> {
    if string.len == 0 {
        return Ok("");
    }
    if string.data.is_null() {
        return Err(RuntimeError::internal(
            "AOT ABI received a null string pointer with non-zero length",
        ));
    }

    let bytes = unsafe { std::slice::from_raw_parts(string.data, string.len) };
    std::str::from_utf8(bytes)
        .map_err(|err| RuntimeError::internal(format!("AOT ABI received invalid UTF-8: {err}")))
}

fn decode_immediate(immediate: &AotImmediate) -> Result<Value, RuntimeError> {
    match immediate.kind {
        AotImmediateKind::ValueBits => Ok(Value::from_bits(immediate.bits)),
        AotImmediateKind::String => Ok(Value::string(intern(resolve_string(&immediate.string)?))),
    }
}

fn decode_operand(
    vm: &VirtualMachine,
    module: &Arc<ModuleObject>,
    operand: &AotOperand,
) -> Result<Value, RuntimeError> {
    match operand.kind {
        AotOperandKind::Immediate => decode_immediate(&operand.immediate),
        AotOperandKind::Name => {
            let name = resolve_string(&operand.name)?;
            vm.module_scope_value_for_module(module.as_ref(), name)
                .ok_or_else(|| RuntimeError::name_error(name))
        }
    }
}

fn add_values(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    if let (Some(left), Some(right)) = (lhs.as_int(), rhs.as_int()) {
        let sum = left
            .checked_add(right)
            .ok_or_else(|| RuntimeError::value_error("integer overflow in native AOT add"))?;
        return Value::int(sum).ok_or_else(|| {
            RuntimeError::value_error("integer result does not fit Prism inline representation")
        });
    }

    if let (Some(left), Some(right)) = (lhs.as_float_coerce(), rhs.as_float_coerce()) {
        return Ok(Value::float(left + right));
    }

    if let (Some(left_ptr), Some(right_ptr)) =
        (lhs.as_string_object_ptr(), rhs.as_string_object_ptr())
    {
        let left = interned_by_ptr(left_ptr as *const u8).ok_or_else(|| {
            RuntimeError::internal("left AOT string operand is not present in the global interner")
        })?;
        let right = interned_by_ptr(right_ptr as *const u8).ok_or_else(|| {
            RuntimeError::internal("right AOT string operand is not present in the global interner")
        })?;
        return Ok(Value::string(intern_owned(format!(
            "{}{}",
            left.as_str(),
            right.as_str()
        ))));
    }

    Err(RuntimeError::unsupported_operand(
        "+",
        lhs.type_name(),
        rhs.type_name(),
    ))
}

pub(crate) fn execute_import_module_op(
    vm: &mut VirtualMachine,
    module: &Arc<ModuleObject>,
    op: &AotImportModuleOp,
) -> Result<(), RuntimeError> {
    let target = resolve_string(&op.target)?;
    let module_spec = resolve_string(&op.module_spec)?;
    let imported = vm.import_module_with_context(module_spec, Some(module))?;
    let bound_module = match op.binding {
        AotImportBinding::Exact => imported,
        AotImportBinding::TopLevel => {
            let top_level = imported
                .name()
                .split('.')
                .next()
                .expect("imported module name should have a top-level segment");
            vm.import_module_with_context(top_level, Some(module))?
        }
    };

    vm.set_module_scope_value_for_module(
        module,
        target,
        Value::object_ptr(Arc::as_ptr(&bound_module) as *const ()),
    );
    Ok(())
}

pub(crate) fn execute_import_from_op(
    vm: &mut VirtualMachine,
    module: &Arc<ModuleObject>,
    op: &AotImportFromOp,
) -> Result<(), RuntimeError> {
    let target = resolve_string(&op.target)?;
    let module_spec = resolve_string(&op.module_spec)?;
    let attribute = resolve_string(&op.attribute)?;
    let value = vm.import_from_with_context(module_spec, attribute, Some(module))?;
    vm.set_module_scope_value_for_module(module, target, value);
    Ok(())
}

pub(crate) fn execute_store_expr_op(
    vm: &mut VirtualMachine,
    module: &Arc<ModuleObject>,
    op: &AotStoreExprOp,
) -> Result<(), RuntimeError> {
    let target = resolve_string(&op.target)?;
    let value = match op.kind {
        AotStoreExprKind::Operand => decode_operand(vm, module, &op.left)?,
        AotStoreExprKind::Add => {
            let left = decode_operand(vm, module, &op.left)?;
            let right = decode_operand(vm, module, &op.right)?;
            add_values(left, right)?
        }
    };

    vm.set_module_scope_value_for_module(module, target, value);
    Ok(())
}

fn with_aot_context<T>(
    vm_ptr: *mut VirtualMachine,
    module_ptr: *const ModuleObject,
    op_ptr: *const T,
    execute: impl FnOnce(&mut VirtualMachine, &Arc<ModuleObject>, &T) -> Result<(), RuntimeError>,
) -> AotOpStatus {
    if vm_ptr.is_null() || module_ptr.is_null() || op_ptr.is_null() {
        return AotOpStatus::Error;
    }

    let vm = unsafe { &mut *vm_ptr };
    vm.clear_last_aot_error();

    let module = match vm.import_resolver.module_from_ptr(module_ptr as *const ()) {
        Some(module) => module,
        None => {
            vm.record_aot_error(RuntimeError::internal(
                "AOT ABI received a module pointer that is not registered with the VM",
            ));
            return AotOpStatus::Error;
        }
    };

    let op = unsafe { &*op_ptr };
    match execute(vm, &module, op) {
        Ok(()) => AotOpStatus::Ok,
        Err(err) => {
            vm.record_aot_error(err);
            AotOpStatus::Error
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn prism_aot_op_import_module(
    vm: *mut VirtualMachine,
    module: *const ModuleObject,
    op: *const AotImportModuleOp,
) -> AotOpStatus {
    let _ = AOT_IMPORT_MODULE_SYMBOL;
    with_aot_context(vm, module, op, execute_import_module_op)
}

#[unsafe(no_mangle)]
pub extern "C" fn prism_aot_op_import_from(
    vm: *mut VirtualMachine,
    module: *const ModuleObject,
    op: *const AotImportFromOp,
) -> AotOpStatus {
    let _ = AOT_IMPORT_FROM_SYMBOL;
    with_aot_context(vm, module, op, execute_import_from_op)
}

#[unsafe(no_mangle)]
pub extern "C" fn prism_aot_op_store_expr(
    vm: *mut VirtualMachine,
    module: *const ModuleObject,
    op: *const AotStoreExprOp,
) -> AotOpStatus {
    let _ = AOT_STORE_EXPR_SYMBOL;
    with_aot_context(vm, module, op, execute_store_expr_op)
}

#[cfg(test)]
mod tests;
