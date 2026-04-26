//! Unpacking operations for call-site *args and **kwargs.
//!
//! This module implements VM handlers for:
//! - `BuildTupleUnpack`: Builds a tuple from multiple values, unpacking starred iterables
//! - `BuildDictUnpack`: Builds a dict from multiple values, merging **dict mappings
//! - `CallEx`: Calls a function with unpacked *args tuple and **kwargs dict

use prism_code::{Instruction, Opcode};
use prism_core::Value;
use prism_core::intern::interned_by_ptr;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
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

    let mut set = SetObject::new();
    for value in values {
        set.add(value);
    }
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
    let kwargs_dict_val = if has_kwargs {
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

    let keyword_entries = match kwargs_dict_val {
        None => Vec::new(),
        Some(value) if value.is_none() => Vec::new(),
        Some(value) => {
            let Some(ptr) = value.as_object_ptr() else {
                return ControlFlow::Error(RuntimeError::type_error(
                    "keyword argument unpacking requires a dict",
                ));
            };
            if unsafe { (*(ptr as *const ObjectHeader)).type_id } != TypeId::DICT {
                return ControlFlow::Error(RuntimeError::type_error(
                    "keyword argument unpacking requires a dict",
                ));
            }

            let dict = unsafe { &*(ptr as *const DictObject) };
            let mut entries = Vec::with_capacity(dict.len());
            for (key, value) in dict.iter() {
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

#[cfg(test)]
mod tests {
    use super::call_ex;
    use crate::VirtualMachine;
    use crate::builtins::{BuiltinError, BuiltinFunctionObject, builtin_iter, builtin_next};
    use crate::dispatch::ControlFlow;
    use prism_code::{CodeObject, Instruction, Opcode};
    use prism_core::Value;
    use prism_runtime::object::descriptor::BoundMethod;
    use prism_runtime::types::dict::DictObject;
    use prism_runtime::types::list::ListObject;
    use prism_runtime::types::tuple::TupleObject;
    use std::sync::Arc;

    fn bound_method_probe(args: &[Value]) -> Result<Value, BuiltinError> {
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].as_int(), Some(7));
        assert_eq!(args[1].as_int(), Some(11));
        Ok(args[1])
    }

    #[test]
    fn test_call_ex_executes_bound_method_with_empty_kwargs_dict() {
        let mut code = CodeObject::new("call_ex_bound_method", "<test>");
        code.instructions = vec![Instruction::new(Opcode::CallKwEx, 3, 0, 0)].into_boxed_slice();

        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::new(code), 0)
            .expect("frame push should succeed");

        let builtin_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
            Arc::from("test.bound_method_probe_call_ex"),
            bound_method_probe,
        )));
        let method_ptr = Box::into_raw(Box::new(BoundMethod::new(
            Value::object_ptr(builtin_ptr as *const ()),
            Value::int(7).unwrap(),
        )));
        let args_tuple_ptr =
            Box::into_raw(Box::new(TupleObject::from_slice(
                &[Value::int(11).unwrap()],
            )));
        let kwargs_ptr = Box::into_raw(Box::new(DictObject::new()));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(method_ptr as *const ()));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(args_tuple_ptr as *const ()));
        vm.current_frame_mut()
            .set_reg(3, Value::object_ptr(kwargs_ptr as *const ()));

        let inst = Instruction::new(Opcode::CallEx, 0, 1, 2);
        assert!(matches!(call_ex(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(11));

        unsafe {
            drop(Box::from_raw(method_ptr));
            drop(Box::from_raw(args_tuple_ptr));
            drop(Box::from_raw(kwargs_ptr));
            drop(Box::from_raw(builtin_ptr));
        }
    }

    #[test]
    fn test_build_tuple_unpack_expands_starred_list() {
        let mut code = CodeObject::new("build_tuple_unpack", "<test>");
        code.instructions = vec![Instruction::new(Opcode::CallKwEx, 1, 0, 0)].into_boxed_slice();

        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::new(code), 0)
            .expect("frame push should succeed");

        let list_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ])));
        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(list_ptr as *const ()));

        let inst = Instruction::new(Opcode::BuildTupleUnpack, 0, 1, 1);
        assert!(matches!(
            super::build_tuple_unpack(&mut vm, inst),
            ControlFlow::Continue
        ));

        let tuple_ptr = vm.current_frame().get_reg(0).as_object_ptr().unwrap();
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.get(0).unwrap().as_int(), Some(1));
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(2));

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_build_tuple_unpack_consumes_starred_iterator_in_place() {
        let mut code = CodeObject::new("build_tuple_unpack_iterator", "<test>");
        code.instructions = vec![Instruction::new(Opcode::CallKwEx, 1, 0, 0)].into_boxed_slice();

        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::new(code), 0)
            .expect("frame push should succeed");

        let list_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[
            Value::int(4).unwrap(),
            Value::int(5).unwrap(),
        ])));
        let iterator = builtin_iter(&[Value::object_ptr(list_ptr as *const ())])
            .expect("iter(list) should create an iterator");
        vm.current_frame_mut().set_reg(1, iterator);

        let inst = Instruction::new(Opcode::BuildTupleUnpack, 0, 1, 1);
        assert!(matches!(
            super::build_tuple_unpack(&mut vm, inst),
            ControlFlow::Continue
        ));

        let tuple_ptr = vm.current_frame().get_reg(0).as_object_ptr().unwrap();
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        assert_eq!(
            tuple.as_slice(),
            &[Value::int(4).unwrap(), Value::int(5).unwrap()]
        );

        let next_result = builtin_next(&[iterator]);
        assert!(matches!(next_result, Err(BuiltinError::StopIteration)));

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }
}
