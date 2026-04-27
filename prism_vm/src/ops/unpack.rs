//! Unpacking operations for call-site *args and **kwargs.
//!
//! This module implements VM handlers for:
//! - `BuildTupleUnpack`: Builds a tuple from multiple values, unpacking starred iterables
//! - `BuildDictUnpack`: Builds a dict from multiple values, merging **dict mappings
//! - `CallEx`: Calls a function with unpacked *args tuple and **kwargs dict

use prism_code::{Instruction, Opcode};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::ops::calls::{
    call_callable_value_with_keywords_from_values, invoke_builtin, invoke_callable_value,
};
use crate::ops::iteration::collect_iterable_values;
use crate::ops::objects::{dict_storage_ref_from_ptr, get_attribute_value};

fn keyword_key_to_name(key: Value) -> Result<Arc<str>, RuntimeError> {
    if let Some(ptr) = key.as_string_object_ptr() {
        if let Some(interned) = interned_by_ptr(ptr as *const u8) {
            return Ok(interned.get_arc());
        }
        return Err(RuntimeError::type_error(
            "keyword argument names must be strings",
        ));
    }

    if let Some(ptr) = key.as_object_ptr()
        && unsafe { (*(ptr as *const ObjectHeader)).type_id } == TypeId::STR
    {
        let string = unsafe { &*(ptr as *const StringObject) };
        return Ok(Arc::from(string.as_str()));
    }

    Err(RuntimeError::type_error(
        "keyword argument names must be strings",
    ))
}

fn read_unpack_flags(vm: &mut VirtualMachine, op_name: &'static str) -> Result<u32, RuntimeError> {
    let ext_inst = vm.current_frame_mut().fetch();
    if ext_inst.opcode() != Opcode::CallKwEx as u8 {
        return Err(RuntimeError::internal(format!(
            "{op_name}: missing extension instruction"
        )));
    }

    Ok(ext_inst.dst().0 as u32
        | ((ext_inst.src1().0 as u32) << 8)
        | ((ext_inst.src2().0 as u32) << 16))
}

fn collect_unpack_sources(
    vm: &mut VirtualMachine,
    base: u8,
    count: u8,
    unpack_flags: u32,
) -> Result<SmallVec<[Value; 16]>, RuntimeError> {
    let mut result: SmallVec<[Value; 16]> = SmallVec::new();

    for i in 0..count {
        let value = vm.current_frame().get_reg(base + i);

        if (unpack_flags & (1 << i)) != 0 {
            let values = collect_iterable_values(vm, value)?;
            result.extend(values);
        } else {
            result.push(value);
        }
    }

    Ok(result)
}

fn mapping_entries_for_unpack(
    vm: &mut VirtualMachine,
    mapping: Value,
) -> Result<Vec<(Value, Value)>, RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && let Some(dict) = dict_storage_ref_from_ptr(ptr)
    {
        return Ok(dict.iter().collect());
    }

    let keys = get_attribute_value(vm, mapping, &intern("keys")).map_err(|err| {
        if matches!(
            err.kind(),
            crate::error::RuntimeErrorKind::AttributeError { .. }
        ) {
            RuntimeError::type_error("keyword argument unpacking requires a mapping")
        } else {
            err
        }
    })?;
    let keys = invoke_callable_value(vm, keys, &[])?;
    let keys = collect_iterable_values(vm, keys)?;
    let getitem = get_attribute_value(vm, mapping, &intern("__getitem__"))?;
    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        entries.push((key, invoke_callable_value(vm, getitem, &[key])?));
    }
    Ok(entries)
}

/// Handle BuildTupleUnpack opcode.
///
/// Format: [BuildTupleUnpack][dst][base][count] + [CallKwEx][flags_lo][flags_mid][flags_hi]
///
/// Builds a tuple from `count` values starting at `base` register.
/// The extension instruction contains a 24-bit bitmap where bit i indicates
/// that register base+i contains an iterable that should be unpacked.
pub fn build_tuple_unpack(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let base = inst.src1().0;
    let count = inst.src2().0;

    let unpack_flags = match read_unpack_flags(vm, "BuildTupleUnpack") {
        Ok(flags) => flags,
        Err(err) => return ControlFlow::Error(err),
    };
    let result = match collect_unpack_sources(vm, base, count, unpack_flags) {
        Ok(values) => values,
        Err(err) => return ControlFlow::Error(err),
    };

    // Create result tuple
    let tuple = TupleObject::from_vec(result.to_vec());
    let tuple_ptr = Arc::into_raw(Arc::new(tuple)) as *const ();
    let tuple_value = Value::object_ptr(tuple_ptr);

    // Store result
    vm.current_frame_mut().set_reg(dst, tuple_value);

    ControlFlow::Continue
}

/// Handle BuildListUnpack opcode.
pub fn build_list_unpack(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let base = inst.src1().0;
    let count = inst.src2().0;

    let unpack_flags = match read_unpack_flags(vm, "BuildListUnpack") {
        Ok(flags) => flags,
        Err(err) => return ControlFlow::Error(err),
    };
    let values = match collect_unpack_sources(vm, base, count, unpack_flags) {
        Ok(values) => values,
        Err(err) => return ControlFlow::Error(err),
    };

    let list = ListObject::from_slice(&values);
    let ptr = match vm.allocator().alloc(list) {
        Some(ptr) => ptr as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate list",
            ));
        }
    };
    vm.current_frame_mut().set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// Handle BuildSetUnpack opcode.
pub fn build_set_unpack(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let base = inst.src1().0;
    let count = inst.src2().0;

    let unpack_flags = match read_unpack_flags(vm, "BuildSetUnpack") {
        Ok(flags) => flags,
        Err(err) => return ControlFlow::Error(err),
    };
    let values = match collect_unpack_sources(vm, base, count, unpack_flags) {
        Ok(values) => values,
        Err(err) => return ControlFlow::Error(err),
    };

    let set = match crate::ops::set_access::set_from_values(vm, values) {
        Ok(set) => set,
        Err(err) => return ControlFlow::Error(err),
    };
    let ptr = match vm.allocator().alloc(set) {
        Some(ptr) => ptr as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate set",
            ));
        }
    };
    vm.current_frame_mut().set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// Handle BuildDictUnpack opcode.
///
/// Format: [BuildDictUnpack][dst][base][count] + [CallKwEx][flags_lo][flags_mid][flags_hi]
///
/// Builds a dict from `count` values starting at `base` register.
/// The extension instruction contains a 24-bit bitmap where bit i indicates
/// that register base+i contains a mapping that should be merged (**dict).
pub fn build_dict_unpack(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let base = inst.src1().0;
    let count = inst.src2().0;

    let unpack_flags = match read_unpack_flags(vm, "BuildDictUnpack") {
        Ok(flags) => flags,
        Err(err) => return ControlFlow::Error(err),
    };

    // Create result dict
    let mut dict = DictObject::new();

    for i in 0..count {
        let src_reg = base + i;
        let value = vm.current_frame().get_reg(src_reg);

        let should_unpack = (unpack_flags & (1 << i)) != 0;

        if should_unpack {
            let entries = match mapping_entries_for_unpack(vm, value) {
                Ok(entries) => entries,
                Err(err) => return ControlFlow::Error(err),
            };
            for (key, val) in entries {
                dict.set(key, val);
            }
        }
    }

    // Store result
    let dict_ptr = Arc::into_raw(Arc::new(dict)) as *const ();
    let dict_value = Value::object_ptr(dict_ptr);
    vm.current_frame_mut().set_reg(dst, dict_value);

    ControlFlow::Continue
}

/// Handle CallEx opcode.
///
/// Format: [CallEx][dst][func][args_tuple] + [CallKwEx][kwargs_dict][0][0]
///
/// Calls a function with unpacked positional args from a tuple and keyword args from a dict.
/// This is used for call sites that contain *args or **kwargs unpacking.
pub fn call_ex(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let func_reg = inst.src1().0;
    let args_tuple_reg = inst.src2().0;

    // Read extension instruction to get kwargs dict register
    let ext_inst = vm.current_frame_mut().fetch();

    // Verify extension is CallKwEx
    if ext_inst.opcode() != Opcode::CallKwEx as u8 {
        return ControlFlow::Error(RuntimeError::internal(
            "CallEx: missing extension instruction",
        ));
    }

    // Get kwargs dict register (0xFF = no kwargs)
    let kwargs_dict_reg = ext_inst.dst().0;
    let has_kwargs = kwargs_dict_reg != 0xFF;

    // Get function and args
    let func_val = vm.current_frame().get_reg(func_reg);
    let args_tuple_val = vm.current_frame().get_reg(args_tuple_reg);
    let kwargs_dict_val = if has_kwargs {
        Some(vm.current_frame().get_reg(kwargs_dict_reg))
    } else {
        None
    };

    if !has_kwargs
        && crate::stdlib::_testcapi::is_method_descriptor_nop_get_instance(func_val)
        && args_tuple_val.as_object_ptr().is_some_and(|ptr| {
            (unsafe { (*(ptr as *const ObjectHeader)).type_id }) == TypeId::TUPLE
        })
    {
        vm.current_frame_mut().set_reg(dst, args_tuple_val);
        return ControlFlow::Continue;
    }

    // Extract positional arguments. Exact tuples are passed through without the
    // compiler copying them first; other starred values are expanded here.
    let args: Vec<Value> = if let Some(tuple_ptr) = args_tuple_val.as_object_ptr().filter(|ptr| {
        let raw = *ptr;
        (unsafe { (*(raw as *const ObjectHeader)).type_id }) == TypeId::TUPLE
    }) {
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        tuple.iter().copied().collect()
    } else {
        match collect_iterable_values(vm, args_tuple_val) {
            Ok(values) => values,
            Err(err) => return ControlFlow::Error(err),
        }
    };

    let keyword_entries = match kwargs_dict_val {
        None => Vec::new(),
        Some(value) if value.is_none() => Vec::new(),
        Some(value) => {
            let mapping_entries = match mapping_entries_for_unpack(vm, value) {
                Ok(entries) => entries,
                Err(err) => return ControlFlow::Error(err),
            };
            let mut entries = Vec::with_capacity(mapping_entries.len());
            for (key, value) in mapping_entries {
                let name = match keyword_key_to_name(key) {
                    Ok(name) => name,
                    Err(err) => return ControlFlow::Error(err),
                };
                entries.push((name, value));
            }
            entries
        }
    };

    let keywords: Vec<(&str, Value)> = keyword_entries
        .iter()
        .map(|(name, value)| (name.as_ref(), *value))
        .collect();
    call_callable_value_with_keywords_from_values(vm, func_val, dst, &args, &keywords)
}
