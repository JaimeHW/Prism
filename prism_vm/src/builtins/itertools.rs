use super::BuiltinError;
use prism_core::Value;
use prism_runtime::types::range::RangeObject;

// =============================================================================
// range
// =============================================================================

/// Builtin range function.
///
/// range(stop) -> range object
/// range(start, stop[, step]) -> range object
///
/// Returns a range object representing a sequence of integers.
pub fn builtin_range(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "range expected 1 to 3 arguments, got {}",
            args.len()
        )));
    }

    let (start, stop, step) = match args.len() {
        1 => {
            // range(stop)
            let stop = args[0].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer end argument expected".to_string())
            })?;
            (0i64, stop, 1i64)
        }
        2 => {
            // range(start, stop)
            let start = args[0].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer start argument expected".to_string())
            })?;
            let stop = args[1].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer end argument expected".to_string())
            })?;
            (start, stop, 1i64)
        }
        3 => {
            // range(start, stop, step)
            let start = args[0].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer start argument expected".to_string())
            })?;
            let stop = args[1].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer end argument expected".to_string())
            })?;
            let step = args[2].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer step argument expected".to_string())
            })?;
            if step == 0 {
                return Err(BuiltinError::ValueError(
                    "range() arg 3 must not be zero".to_string(),
                ));
            }
            (start, stop, step)
        }
        _ => unreachable!(),
    };

    // Create RangeObject on heap and return as Value
    // TODO: Use GC allocator instead of Box::leak
    let range_obj = Box::new(RangeObject::new(start, stop, step));
    let ptr = Box::leak(range_obj) as *mut RangeObject as *const ();
    Ok(Value::object_ptr(ptr))
}

// =============================================================================
// iter
// =============================================================================

/// Builtin iter function.
///
/// iter(object) -> iterator
/// iter(callable, sentinel) -> iterator
pub fn builtin_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "iter expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    // TODO: Dispatch to TypeSlots.tp_iter for objects
    // For sentinel form, create a callable iterator
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "iter() not yet fully implemented".to_string(),
    ))
}

// =============================================================================
// next
// =============================================================================

/// Builtin next function.
///
/// next(iterator[, default]) -> next item from iterator
pub fn builtin_next(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "next expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    // TODO: Dispatch to TypeSlots.tp_next for iterator objects
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "next() not yet fully implemented".to_string(),
    ))
}

// =============================================================================
// enumerate
// =============================================================================

/// Builtin enumerate function.
///
/// enumerate(iterable, start=0) -> enumerate object
pub fn builtin_enumerate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "enumerate expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let _start = if args.len() == 2 {
        args[1].as_int().unwrap_or(0)
    } else {
        0
    };

    // TODO: Create enumerate iterator object
    Err(BuiltinError::NotImplemented(
        "enumerate() not yet implemented".to_string(),
    ))
}

// =============================================================================
// zip
// =============================================================================

/// Builtin zip function.
///
/// zip(*iterables) -> zip object
pub fn builtin_zip(args: &[Value]) -> Result<Value, BuiltinError> {
    // zip accepts zero or more iterables
    let _ = args;

    // TODO: Create zip iterator that pairs elements from multiple iterables
    Err(BuiltinError::NotImplemented(
        "zip() not yet implemented".to_string(),
    ))
}

// =============================================================================
// map
// =============================================================================

/// Builtin map function.
///
/// map(function, iterable, ...) -> map object
pub fn builtin_map(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 {
        return Err(BuiltinError::TypeError(format!(
            "map expected at least 2 arguments, got {}",
            args.len()
        )));
    }

    // TODO: Create map iterator that applies function to each element
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "map() not yet implemented".to_string(),
    ))
}

// =============================================================================
// filter
// =============================================================================

/// Builtin filter function.
///
/// filter(function, iterable) -> filter object
pub fn builtin_filter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "filter expected 2 arguments, got {}",
            args.len()
        )));
    }

    // TODO: Create filter iterator
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "filter() not yet implemented".to_string(),
    ))
}

// =============================================================================
// reversed
// =============================================================================

/// Builtin reversed function.
///
/// reversed(sequence) -> reverse iterator
pub fn builtin_reversed(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "reversed expected 1 argument, got {}",
            args.len()
        )));
    }

    // TODO: Create reversed iterator via TypeSlots.tp_reversed or sq_item
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "reversed() not yet implemented".to_string(),
    ))
}

// =============================================================================
// sorted
// =============================================================================

/// Builtin sorted function.
///
/// sorted(iterable, /, *, key=None, reverse=False) -> list
pub fn builtin_sorted(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "sorted expected 1 to 3 arguments, got {}",
            args.len()
        )));
    }

    // TODO: Collect iterable, sort, return as list
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "sorted() not yet implemented".to_string(),
    ))
}

// =============================================================================
// all / any
// =============================================================================

/// Builtin all function.
///
/// all(iterable) -> bool - True if all elements are truthy
pub fn builtin_all(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "all expected 1 argument, got {}",
            args.len()
        )));
    }

    // TODO: Iterate and check truthiness
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "all() not yet implemented".to_string(),
    ))
}

/// Builtin any function.
///
/// any(iterable) -> bool - True if any element is truthy
pub fn builtin_any(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "any expected 1 argument, got {}",
            args.len()
        )));
    }

    // TODO: Iterate and check truthiness
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "any() not yet implemented".to_string(),
    ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_validation() {
        // Zero step should error
        let result = builtin_range(&[
            Value::int(0).unwrap(),
            Value::int(10).unwrap(),
            Value::int(0).unwrap(),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_type_error() {
        // Non-integer should error
        let result = builtin_range(&[Value::float(3.14)]);
        assert!(result.is_err());
    }
}
