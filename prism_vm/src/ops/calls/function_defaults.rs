use super::*;

/// SetFunctionDefaults: attach positional/kw-only default metadata to a function object.
///
/// - dst: function register
/// - src1: positional defaults tuple register (or None)
/// - src2: keyword-only defaults dict register (or None)
#[inline(always)]
pub fn set_function_defaults(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (func_val, pos_defaults_val, kw_defaults_val) = {
        let frame = vm.current_frame();
        (
            frame.get_reg(inst.dst().0),
            frame.get_reg(inst.src1().0),
            frame.get_reg(inst.src2().0),
        )
    };

    let defaults = match parse_positional_defaults(pos_defaults_val) {
        Ok(v) => v,
        Err(e) => return ControlFlow::Error(e),
    };
    let kwdefaults = match parse_kw_defaults(kw_defaults_val) {
        Ok(v) => v,
        Err(e) => return ControlFlow::Error(e),
    };

    let Some(func_ptr) = func_val.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::type_error(
            "SetFunctionDefaults target is not a function object",
        ));
    };
    let type_id = extract_type_id(func_ptr);
    if type_id != TypeId::FUNCTION && type_id != TypeId::CLOSURE {
        return ControlFlow::Error(RuntimeError::type_error(
            "SetFunctionDefaults target is not a function object",
        ));
    }

    let func = unsafe { &mut *(func_ptr as *mut FunctionObject) };
    func.defaults = defaults;
    func.kwdefaults = kwdefaults;
    ControlFlow::Continue
}

fn parse_positional_defaults(value: Value) -> Result<Option<Box<[Value]>>, RuntimeError> {
    if value.is_none() {
        return Ok(None);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "function positional defaults must be tuple or None",
        ));
    };
    if extract_type_id(ptr) != TypeId::TUPLE {
        return Err(RuntimeError::type_error(
            "function positional defaults must be tuple or None",
        ));
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let values: Vec<Value> = tuple.iter().copied().collect();
    Ok(Some(values.into_boxed_slice()))
}

fn parse_kw_defaults(value: Value) -> Result<Option<Box<[(Arc<str>, Value)]>>, RuntimeError> {
    if value.is_none() {
        return Ok(None);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "function keyword defaults must be dict or None",
        ));
    };
    if extract_type_id(ptr) != TypeId::DICT {
        return Err(RuntimeError::type_error(
            "function keyword defaults must be dict or None",
        ));
    }

    let dict = unsafe { &*(ptr as *const DictObject) };
    let mut entries = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key_name = kw_default_key_to_name(key)?;
        entries.push((key_name, value));
    }
    Ok(Some(entries.into_boxed_slice()))
}

fn kw_default_key_to_name(key: Value) -> Result<Arc<str>, RuntimeError> {
    if let Some(ptr) = key.as_string_object_ptr() {
        if let Some(interned) = interned_by_ptr(ptr as *const u8) {
            return Ok(interned.get_arc());
        }
        return Err(RuntimeError::type_error(
            "keyword defaults dict contains invalid interned string key",
        ));
    }

    if let Some(ptr) = key.as_object_ptr() {
        if extract_type_id(ptr) == TypeId::STR {
            let string = unsafe { &*(ptr as *const StringObject) };
            return Ok(Arc::from(string.as_str()));
        }
    }

    Err(RuntimeError::type_error(
        "function keyword defaults dict keys must be strings",
    ))
}
