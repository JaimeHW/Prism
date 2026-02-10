//! Unpacking operations for call-site *args and **kwargs.
//!
//! This module implements VM handlers for:
//! - `BuildTupleUnpack`: Builds a tuple from multiple values, unpacking starred iterables
//! - `BuildDictUnpack`: Builds a dict from multiple values, merging **dict mappings
//! - `CallEx`: Calls a function with unpacked *args tuple and **kwargs dict

use prism_compiler::bytecode::{Instruction, Opcode};
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::tuple::TupleObject;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;

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

    // Read extension instruction to get unpack flags
    let ext_inst = vm.current_frame_mut().fetch();

    // Verify extension is CallKwEx (used for extension bytes)
    if ext_inst.opcode() != Opcode::CallKwEx as u8 {
        return ControlFlow::Error(RuntimeError::internal(
            "BuildTupleUnpack: missing extension instruction",
        ));
    }

    // Extract 24-bit unpack flags
    let unpack_flags: u32 = ext_inst.dst().0 as u32
        | ((ext_inst.src1().0 as u32) << 8)
        | ((ext_inst.src2().0 as u32) << 16);

    // Collect values into result tuple
    let mut result: SmallVec<[Value; 16]> = SmallVec::new();

    for i in 0..count {
        let src_reg = base + i;
        let value = vm.current_frame().get_reg(src_reg);

        let should_unpack = (unpack_flags & (1 << i)) != 0;

        if should_unpack {
            // Unpack iterable - for now, handle tuples directly
            if let Some(tuple_ptr) = value.as_object_ptr() {
                let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
                for j in 0..tuple.len() {
                    if let Some(item) = tuple.get(j as i64) {
                        result.push(item);
                    }
                }
            } else {
                // Single value - just add it
                result.push(value);
            }
        } else {
            // Regular value - add directly
            result.push(value);
        }
    }

    // Create result tuple
    let tuple = TupleObject::from_vec(result.to_vec());
    let tuple_ptr = Arc::into_raw(Arc::new(tuple)) as *const ();
    let tuple_value = Value::object_ptr(tuple_ptr);

    // Store result
    vm.current_frame_mut().set_reg(dst, tuple_value);

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

    // Read extension instruction to get unpack flags
    let ext_inst = vm.current_frame_mut().fetch();

    // Verify extension is CallKwEx
    if ext_inst.opcode() != Opcode::CallKwEx as u8 {
        return ControlFlow::Error(RuntimeError::internal(
            "BuildDictUnpack: missing extension instruction",
        ));
    }

    // Extract 24-bit unpack flags
    let unpack_flags: u32 = ext_inst.dst().0 as u32
        | ((ext_inst.src1().0 as u32) << 8)
        | ((ext_inst.src2().0 as u32) << 16);

    // Create result dict
    let mut dict = DictObject::new();

    for i in 0..count {
        let src_reg = base + i;
        let value = vm.current_frame().get_reg(src_reg);

        let should_unpack = (unpack_flags & (1 << i)) != 0;

        if should_unpack {
            // Merge dict - iterate and add all key-value pairs
            if let Some(dict_ptr) = value.as_object_ptr() {
                let src_dict = unsafe { &*(dict_ptr as *const DictObject) };
                for (key, val) in src_dict.iter() {
                    dict.set(key, val);
                }
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
    let _kwargs_dict_val = if has_kwargs {
        Some(vm.current_frame().get_reg(kwargs_dict_reg))
    } else {
        None
    };

    // Extract args tuple
    let args: Vec<Value> = if let Some(tuple_ptr) = args_tuple_val.as_object_ptr() {
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        (0..tuple.len())
            .filter_map(|i| tuple.get(i as i64))
            .collect()
    } else {
        vec![args_tuple_val] // Single value fallback
    };

    // Check if this is a callable object
    if let Some(ptr) = func_val.as_object_ptr() {
        let header_ptr = ptr as *const ObjectHeader;
        let type_id = unsafe { (*header_ptr).type_id };

        match type_id {
            TypeId::BUILTIN_FUNCTION => {
                let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
                match builtin.call(&args) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(RuntimeError::type_error(e.to_string())),
                }
            }
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                let code = Arc::clone(&func.code);

                // Push new frame for function execution
                if let Err(e) = vm.push_frame(Arc::clone(&code), dst) {
                    return ControlFlow::Error(e);
                }

                // Copy arguments to new frame's registers
                for (i, arg) in args.into_iter().enumerate() {
                    vm.current_frame_mut().set_reg(i as u8, arg);
                }

                ControlFlow::Continue
            }
            _ => ControlFlow::Error(RuntimeError::type_error(
                "object is not callable".to_string(),
            )),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("CallEx: not a callable object"))
    }
}
