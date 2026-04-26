//! Set and frozenset builtin method resolution and implementations.

use super::*;

static SET_ADD_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.add"), set_add));
static SET_REMOVE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.remove"), set_remove));
static SET_DISCARD_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.discard"), set_discard));
static SET_POP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.pop"), set_pop));
static SET_CLEAR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.clear"), set_clear));
static SET_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("set.update"), set_update_with_vm));
static SET_DIFFERENCE_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("set.difference_update"),
        set_difference_update_with_vm,
    )
});
static SET_INTERSECTION_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("set.intersection_update"),
        set_intersection_update_with_vm,
    )
});
static SET_SYMMETRIC_DIFFERENCE_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm(
            Arc::from("set.symmetric_difference_update"),
            set_symmetric_difference_update_with_vm,
        )
    });
static SET_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.copy"), set_copy));
static SET_UNION_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("set.union"), set_union_with_vm));
static SET_INTERSECTION_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.intersection"), set_intersection_with_vm)
});
static SET_DIFFERENCE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.difference"), set_difference_with_vm)
});
static SET_SYMMETRIC_DIFFERENCE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("set.symmetric_difference"),
        set_symmetric_difference_with_vm,
    )
});
static SET_ISDISJOINT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.isdisjoint"), set_isdisjoint_with_vm)
});
static SET_ISSUBSET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.issubset"), set_issubset_with_vm)
});
static SET_ISSUPERSET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.issuperset"), set_issuperset_with_vm)
});
static SET_CONTAINS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.__contains__"), set_contains));
static FROZENSET_CONTAINS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("frozenset.__contains__"), frozenset_contains)
});
static FROZENSET_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("frozenset.copy"), frozenset_copy));
static FROZENSET_UNION_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("frozenset.union"), frozenset_union_with_vm)
});
static FROZENSET_INTERSECTION_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("frozenset.intersection"),
        frozenset_intersection_with_vm,
    )
});
static FROZENSET_DIFFERENCE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("frozenset.difference"),
        frozenset_difference_with_vm,
    )
});
static FROZENSET_SYMMETRIC_DIFFERENCE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm(
            Arc::from("frozenset.symmetric_difference"),
            frozenset_symmetric_difference_with_vm,
        )
    });
static FROZENSET_ISDISJOINT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("frozenset.isdisjoint"),
        frozenset_isdisjoint_with_vm,
    )
});
static FROZENSET_ISSUBSET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("frozenset.issubset"), frozenset_issubset_with_vm)
});
static FROZENSET_ISSUPERSET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("frozenset.issuperset"),
        frozenset_issuperset_with_vm,
    )
});

/// Resolve builtin set and frozenset methods backed by static builtin function objects.

/// Resolve builtin set and frozenset methods backed by static builtin function objects.
pub fn resolve_set_method(type_id: TypeId, name: &str) -> Option<CachedMethod> {
    match (type_id, name) {
        (TypeId::SET, "add") => Some(CachedMethod::simple(builtin_method_value(&SET_ADD_METHOD))),
        (TypeId::SET, "remove") => Some(CachedMethod::simple(builtin_method_value(
            &SET_REMOVE_METHOD,
        ))),
        (TypeId::SET, "discard") => Some(CachedMethod::simple(builtin_method_value(
            &SET_DISCARD_METHOD,
        ))),
        (TypeId::SET, "pop") => Some(CachedMethod::simple(builtin_method_value(&SET_POP_METHOD))),
        (TypeId::SET, "clear") => Some(CachedMethod::simple(builtin_method_value(
            &SET_CLEAR_METHOD,
        ))),
        (TypeId::SET, "update") => Some(CachedMethod::simple(builtin_method_value(
            &SET_UPDATE_METHOD,
        ))),
        (TypeId::SET, "difference_update") => Some(CachedMethod::simple(builtin_method_value(
            &SET_DIFFERENCE_UPDATE_METHOD,
        ))),
        (TypeId::SET, "intersection_update") => Some(CachedMethod::simple(builtin_method_value(
            &SET_INTERSECTION_UPDATE_METHOD,
        ))),
        (TypeId::SET, "symmetric_difference_update") => Some(CachedMethod::simple(
            builtin_method_value(&SET_SYMMETRIC_DIFFERENCE_UPDATE_METHOD),
        )),
        (TypeId::SET, "copy") => Some(CachedMethod::simple(builtin_method_value(&SET_COPY_METHOD))),
        (TypeId::SET, "union") => Some(CachedMethod::simple(builtin_method_value(
            &SET_UNION_METHOD,
        ))),
        (TypeId::SET, "intersection") => Some(CachedMethod::simple(builtin_method_value(
            &SET_INTERSECTION_METHOD,
        ))),
        (TypeId::SET, "difference") => Some(CachedMethod::simple(builtin_method_value(
            &SET_DIFFERENCE_METHOD,
        ))),
        (TypeId::SET, "symmetric_difference") => Some(CachedMethod::simple(builtin_method_value(
            &SET_SYMMETRIC_DIFFERENCE_METHOD,
        ))),
        (TypeId::SET, "isdisjoint") => Some(CachedMethod::simple(builtin_method_value(
            &SET_ISDISJOINT_METHOD,
        ))),
        (TypeId::SET, "issubset") => Some(CachedMethod::simple(builtin_method_value(
            &SET_ISSUBSET_METHOD,
        ))),
        (TypeId::SET, "issuperset") => Some(CachedMethod::simple(builtin_method_value(
            &SET_ISSUPERSET_METHOD,
        ))),
        (TypeId::SET, "__contains__") => Some(CachedMethod::simple(builtin_method_value(
            &SET_CONTAINS_METHOD,
        ))),
        (TypeId::FROZENSET, "union") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_UNION_METHOD,
        ))),
        (TypeId::FROZENSET, "intersection") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_INTERSECTION_METHOD,
        ))),
        (TypeId::FROZENSET, "difference") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_DIFFERENCE_METHOD,
        ))),
        (TypeId::FROZENSET, "symmetric_difference") => Some(CachedMethod::simple(
            builtin_method_value(&FROZENSET_SYMMETRIC_DIFFERENCE_METHOD),
        )),
        (TypeId::FROZENSET, "isdisjoint") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_ISDISJOINT_METHOD,
        ))),
        (TypeId::FROZENSET, "issubset") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_ISSUBSET_METHOD,
        ))),
        (TypeId::FROZENSET, "issuperset") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_ISSUPERSET_METHOD,
        ))),
        (TypeId::FROZENSET, "copy") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_COPY_METHOD,
        ))),
        (TypeId::FROZENSET, "__contains__") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_CONTAINS_METHOD,
        ))),
        _ => None,
    }
}

#[inline]
pub(super) fn set_add(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "add", args, 1)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "add")?;
    ensure_hashable(args[1])?;
    set.add(args[1]);
    Ok(Value::none())
}

#[inline]
pub(super) fn set_remove(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "remove", args, 1)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "remove")?;
    ensure_hashable(args[1])?;
    if set.remove(args[1]) {
        Ok(Value::none())
    } else {
        Err(BuiltinError::KeyError(args[1].to_string()))
    }
}

#[inline]
pub(super) fn set_discard(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "discard", args, 1)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "discard")?;
    ensure_hashable(args[1])?;
    set.discard(args[1]);
    Ok(Value::none())
}

#[inline]
pub(super) fn set_pop(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "pop", args, 0)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "pop")?;
    set.pop()
        .ok_or_else(|| BuiltinError::KeyError("pop from an empty set".to_string()))
}

#[inline]
pub(super) fn set_clear(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "clear", args, 0)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "clear")?;
    set.clear();
    Ok(Value::none())
}

#[inline]
pub(super) fn set_contains(args: &[Value]) -> Result<Value, BuiltinError> {
    contains_for_set_type(args, TypeId::SET, "set", "__contains__")
}

#[inline]
pub(super) fn frozenset_contains(args: &[Value]) -> Result<Value, BuiltinError> {
    contains_for_set_type(args, TypeId::FROZENSET, "frozenset", "__contains__")
}

#[inline]
pub(super) fn set_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "copy", args, 0)?;
    let set = expect_set_receiver(args[0], TypeId::SET, "copy")?;
    Ok(to_object_value(set.clone()))
}

#[inline]
fn set_result_value(mut set: SetObject, result_type: TypeId) -> Value {
    set.header.type_id = result_type;
    to_object_value(set)
}

#[inline]
fn hashable_iterable_values_with_vm(
    vm: &mut VirtualMachine,
    iterable: Value,
) -> Result<Vec<Value>, BuiltinError> {
    let values = collect_iterable_values_with_vm(vm, iterable)?;
    for value in values.iter().copied() {
        ensure_hashable(value)?;
    }
    Ok(values)
}

#[inline]
fn iterable_to_hashable_set_with_vm(
    vm: &mut VirtualMachine,
    iterable: Value,
) -> Result<SetObject, BuiltinError> {
    Ok(SetObject::from_iter(hashable_iterable_values_with_vm(
        vm, iterable,
    )?))
}

#[inline]
pub(super) fn set_update_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let set = expect_set_mut_receiver(
        *args
            .first()
            .ok_or_else(|| BuiltinError::TypeError("unbound set.update()".to_string()))?,
        TypeId::SET,
        "update",
    )?;

    for iterable in &args[1..] {
        for value in hashable_iterable_values_with_vm(vm, *iterable)? {
            set.add(value);
        }
    }

    Ok(Value::none())
}

#[inline]
pub(super) fn set_difference_update_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let set = expect_set_mut_receiver(
        *args.first().ok_or_else(|| {
            BuiltinError::TypeError("unbound set.difference_update()".to_string())
        })?,
        TypeId::SET,
        "difference_update",
    )?;

    for iterable in &args[1..] {
        for value in hashable_iterable_values_with_vm(vm, *iterable)? {
            set.discard(value);
        }
    }

    Ok(Value::none())
}

#[inline]
pub(super) fn set_intersection_update_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let set = expect_set_mut_receiver(
        *args.first().ok_or_else(|| {
            BuiltinError::TypeError("unbound set.intersection_update()".to_string())
        })?,
        TypeId::SET,
        "intersection_update",
    )?;

    for iterable in &args[1..] {
        let other = iterable_to_hashable_set_with_vm(vm, *iterable)?;
        set.intersection_update(&other);
    }

    Ok(Value::none())
}

#[inline]
pub(super) fn set_symmetric_difference_update_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "symmetric_difference_update", args, 1)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "symmetric_difference_update")?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    set.symmetric_difference_update(&other);
    Ok(Value::none())
}

#[inline]
fn set_union_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    let receiver = *args.first().ok_or_else(|| {
        BuiltinError::TypeError(format!("unbound {receiver_name}.{method_name}()"))
    })?;
    let mut result = expect_set_receiver(receiver, expected_type, method_name)?.clone();
    for iterable in &args[1..] {
        for value in hashable_iterable_values_with_vm(vm, *iterable)? {
            result.add(value);
        }
    }
    Ok(set_result_value(result, expected_type))
}

#[inline]
fn set_intersection_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    let receiver = *args.first().ok_or_else(|| {
        BuiltinError::TypeError(format!("unbound {receiver_name}.{method_name}()"))
    })?;
    let current = expect_set_receiver(receiver, expected_type, method_name)?;
    if args.len() == 1 {
        return if expected_type == TypeId::FROZENSET {
            Ok(receiver)
        } else {
            Ok(set_result_value(current.clone(), expected_type))
        };
    }

    let mut result = current.clone();
    for iterable in &args[1..] {
        let other = iterable_to_hashable_set_with_vm(vm, *iterable)?;
        result.intersection_update(&other);
    }
    Ok(set_result_value(result, expected_type))
}

#[inline]
fn set_difference_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    let receiver = *args.first().ok_or_else(|| {
        BuiltinError::TypeError(format!("unbound {receiver_name}.{method_name}()"))
    })?;
    let current = expect_set_receiver(receiver, expected_type, method_name)?;
    if args.len() == 1 {
        return if expected_type == TypeId::FROZENSET {
            Ok(receiver)
        } else {
            Ok(set_result_value(current.clone(), expected_type))
        };
    }

    let mut result = current.clone();
    for iterable in &args[1..] {
        let other = iterable_to_hashable_set_with_vm(vm, *iterable)?;
        result.difference_update(&other);
    }
    Ok(set_result_value(result, expected_type))
}

#[inline]
fn set_symmetric_difference_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    Ok(set_result_value(
        set.symmetric_difference(&other),
        expected_type,
    ))
}

#[inline]
fn set_isdisjoint_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    Ok(Value::bool(set.is_disjoint(&other)))
}

#[inline]
fn set_issubset_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    Ok(Value::bool(set.is_subset(&other)))
}

#[inline]
fn set_issuperset_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    Ok(Value::bool(set.is_superset(&other)))
}

#[inline]
pub(super) fn set_union_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_union_impl(vm, args, TypeId::SET, "set", "union")
}

#[inline]
pub(super) fn frozenset_union_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_union_impl(vm, args, TypeId::FROZENSET, "frozenset", "union")
}

#[inline]
pub(super) fn set_intersection_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_intersection_impl(vm, args, TypeId::SET, "set", "intersection")
}

#[inline]
pub(super) fn frozenset_intersection_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_intersection_impl(vm, args, TypeId::FROZENSET, "frozenset", "intersection")
}

#[inline]
pub(super) fn set_difference_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_difference_impl(vm, args, TypeId::SET, "set", "difference")
}

#[inline]
pub(super) fn frozenset_difference_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_difference_impl(vm, args, TypeId::FROZENSET, "frozenset", "difference")
}

#[inline]
pub(super) fn set_symmetric_difference_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_symmetric_difference_impl(vm, args, TypeId::SET, "set", "symmetric_difference")
}

#[inline]
pub(super) fn frozenset_symmetric_difference_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_symmetric_difference_impl(
        vm,
        args,
        TypeId::FROZENSET,
        "frozenset",
        "symmetric_difference",
    )
}

#[inline]
pub(super) fn set_isdisjoint_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_isdisjoint_impl(vm, args, TypeId::SET, "set", "isdisjoint")
}

#[inline]
pub(super) fn frozenset_isdisjoint_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_isdisjoint_impl(vm, args, TypeId::FROZENSET, "frozenset", "isdisjoint")
}

#[inline]
pub(super) fn set_issubset_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_issubset_impl(vm, args, TypeId::SET, "set", "issubset")
}

#[inline]
pub(super) fn frozenset_issubset_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_issubset_impl(vm, args, TypeId::FROZENSET, "frozenset", "issubset")
}

#[inline]
pub(super) fn set_issuperset_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_issuperset_impl(vm, args, TypeId::SET, "set", "issuperset")
}

#[inline]
pub(super) fn frozenset_issuperset_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_issuperset_impl(vm, args, TypeId::FROZENSET, "frozenset", "issuperset")
}

#[inline]
pub(super) fn frozenset_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("frozenset", "copy", args, 0)?;
    expect_set_receiver(args[0], TypeId::FROZENSET, "copy")?;
    Ok(args[0])
}

#[inline]
fn contains_for_set_type(
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    ensure_hashable(args[1])?;
    Ok(Value::bool(set.contains(args[1])))
}
