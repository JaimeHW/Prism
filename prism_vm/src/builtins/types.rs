//! Type constructor builtins (int, float, str, bool, list, dict, etc.).

use super::BuiltinError;
use prism_core::Value;

/// Builtin int constructor.
pub fn builtin_int(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::int(0).unwrap());
    }
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "int() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }
    let arg = args[0];
    if let Some(i) = arg.as_int() {
        return Ok(arg);
    }
    if let Some(f) = arg.as_float() {
        return Value::int(f as i64)
            .ok_or_else(|| BuiltinError::OverflowError("int too large".to_string()));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Value::int(if b { 1 } else { 0 }).unwrap());
    }
    Err(BuiltinError::TypeError(
        "int() argument must be a string or number".to_string(),
    ))
}

/// Builtin float constructor.
pub fn builtin_float(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::float(0.0));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "float() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    let arg = args[0];
    if let Some(f) = arg.as_float() {
        return Ok(arg);
    }
    if let Some(i) = arg.as_int() {
        return Ok(Value::float(i as f64));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Value::float(if b { 1.0 } else { 0.0 }));
    }
    Err(BuiltinError::TypeError(
        "float() argument must be a string or number".to_string(),
    ))
}

/// Builtin str constructor.
pub fn builtin_str(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        // TODO: Return empty string Value
        return Ok(Value::none());
    }
    // TODO: Call __str__ on object
    Ok(Value::none())
}

/// Builtin bool constructor.
pub fn builtin_bool(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::bool(false));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "bool() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    Ok(Value::bool(args[0].is_truthy()))
}

/// Builtin list constructor.
pub fn builtin_list(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "list() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    // TODO: Create ListObject from iterable
    Ok(Value::none())
}

/// Builtin tuple constructor.
pub fn builtin_tuple(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "tuple() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    // TODO: Create TupleObject from iterable
    Ok(Value::none())
}

/// Builtin dict constructor.
pub fn builtin_dict(args: &[Value]) -> Result<Value, BuiltinError> {
    // TODO: Create DictObject from iterable of pairs or kwargs
    let _ = args;
    Ok(Value::none())
}

/// Builtin set constructor.
pub fn builtin_set(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "set() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    // TODO: Create SetObject from iterable
    Ok(Value::none())
}

/// Builtin frozenset constructor.
pub fn builtin_frozenset(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "frozenset() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    // TODO: Create FrozenSetObject
    Err(BuiltinError::NotImplemented(
        "frozenset() not yet implemented".to_string(),
    ))
}

/// Builtin type function.
pub fn builtin_type(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "type() takes 1 or 3 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Return TypeObject for the value
    Ok(Value::none())
}

/// Builtin isinstance function.
pub fn builtin_isinstance(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "isinstance() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Check type hierarchy
    Ok(Value::bool(false))
}

/// Builtin issubclass function.
pub fn builtin_issubclass(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "issubclass() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Check class hierarchy
    Ok(Value::bool(false))
}

/// Builtin object constructor.
pub fn builtin_object(args: &[Value]) -> Result<Value, BuiltinError> {
    let _ = args;
    // TODO: Create base object
    Ok(Value::none())
}

/// Builtin getattr function.
pub fn builtin_getattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "getattr() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Get attribute from object
    Err(BuiltinError::NotImplemented(
        "getattr() not yet implemented".to_string(),
    ))
}

/// Builtin setattr function.
pub fn builtin_setattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "setattr() takes 3 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Set attribute on object
    Err(BuiltinError::NotImplemented(
        "setattr() not yet implemented".to_string(),
    ))
}

/// Builtin hasattr function.
pub fn builtin_hasattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "hasattr() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Check if object has attribute
    Ok(Value::bool(false))
}

/// Builtin delattr function.
pub fn builtin_delattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "delattr() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Delete attribute from object
    Err(BuiltinError::NotImplemented(
        "delattr() not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_from_int() {
        let result = builtin_int(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_int_from_float() {
        let result = builtin_int(&[Value::float(3.9)]).unwrap();
        assert_eq!(result.as_int(), Some(3));
    }

    #[test]
    fn test_float_from_int() {
        let result = builtin_float(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_float(), Some(42.0));
    }

    #[test]
    fn test_bool_truthy() {
        let result = builtin_bool(&[Value::int(1).unwrap()]).unwrap();
        assert!(result.is_truthy());

        let result = builtin_bool(&[Value::int(0).unwrap()]).unwrap();
        assert!(!result.is_truthy());
    }
}
