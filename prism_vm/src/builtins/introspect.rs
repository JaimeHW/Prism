//! Introspection builtins (dir, vars, globals, locals, help).
//!
//! Functions for runtime inspection of Python objects and namespaces.
//! All functions are Python 3.12 compatible.
//!
//! # Python Semantics
//!
//! - `dir([object])` - List attributes/valid names in scope
//! - `vars([object])` - Return __dict__ of object (or locals if no arg)
//! - `globals()` - Return global symbol table dict
//! - `locals()` - Return local symbol table dict
//! - `help([object])` - Display documentation

use super::BuiltinError;
use crate::VirtualMachine;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::import::ModuleObject;
use crate::ops::attribute::is_user_defined_type;
use crate::ops::objects::{
    alloc_heap_value, get_attribute_value, snapshot_current_globals_dict,
    snapshot_frame_locals_dict,
};
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{builtin_class_mro, class_id_to_type_id, global_class};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::MappingProxyObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashSet;
use std::sync::Arc;

// =============================================================================
// dir() - List Attributes
// =============================================================================

const TYPE_REFLECTION_NAMES: &[&str] = &[
    "__base__",
    "__bases__",
    "__dict__",
    "__module__",
    "__mro__",
    "__name__",
    "__qualname__",
];

/// Builtin dir([object]) function.
///
/// Without arguments, returns list of names in the current local scope.
/// With an object argument, returns list of valid attributes for that object.
///
/// # Python Semantics
/// - `dir()` → local scope names (sorted)
/// - `dir(int)` → ['__abs__', '__add__', '__and__', ...]
/// - `dir(obj)` → obj.__dir__() or sorted(obj.__dict__.keys())
///
/// # Implementation Note
/// Full implementation requires __dir__ protocol and __dict__ access.
pub fn builtin_dir(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "dir() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    if args.is_empty() {
        // No argument: return names in local scope
        // TODO: Requires access to the current frame's locals
        return Err(BuiltinError::NotImplemented(
            "dir() without argument requires frame introspection".to_string(),
        ));
    }

    // With argument: return attributes of object
    let obj = &args[0];
    dir_of_value(obj)
}

/// VM-aware dir([object]) function.
///
/// This implementation can resolve module pointers registered with the import
/// resolver and can enumerate heap type metadata for user-defined classes.
pub fn builtin_dir_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "dir() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    if args.is_empty() {
        return dir_of_current_scope(vm);
    }

    dir_of_value_vm(vm, args[0])
}

#[inline]
fn dir_of_current_scope(vm: &VirtualMachine) -> Result<Value, BuiltinError> {
    let mut names = Vec::new();
    let mut seen = FxHashSet::default();

    if let Some(module) = vm.current_module() {
        for name in module.dir() {
            push_unique_name(&mut names, &mut seen, name);
        }
    }

    let frame = vm.current_frame();
    if frame.locals_mapping().is_none() {
        for (slot, name) in frame.code.locals.iter().enumerate() {
            let slot = slot as u8;
            if frame.reg_is_written(slot) {
                push_unique_name(&mut names, &mut seen, intern(name.as_ref()));
            }
        }
    }

    dir_from_names(names)
}

/// Get directory (attributes) of a value.
///
/// This is the core implementation for dir(obj).
#[inline]
fn dir_of_value(value: &Value) -> Result<Value, BuiltinError> {
    if let Some(owner) = value_owner_type_id(*value) {
        return dir_from_names(collect_builtin_instance_dir_names(owner)?);
    }

    if let Some(ptr) = value.as_object_ptr() {
        if let Some(represented) = crate::builtins::builtin_type_object_type_id(ptr) {
            return dir_from_names(collect_builtin_type_dir_names(represented)?);
        }

        if crate::ops::objects::extract_type_id(ptr) == TypeId::TYPE {
            return dir_from_names(collect_heap_type_dir_names(unsafe {
                &*(ptr as *const PyClassObject)
            })?);
        }
    }

    // For objects, we need __dir__ or __dict__
    Err(BuiltinError::NotImplemented(
        "dir() for objects requires __dir__ or __dict__".to_string(),
    ))
}

#[inline]
fn dir_of_value_vm(vm: &VirtualMachine, value: Value) -> Result<Value, BuiltinError> {
    if let Some(owner) = value_owner_type_id(value) {
        return dir_from_names(collect_builtin_instance_dir_names(owner)?);
    }

    if let Some(ptr) = value.as_object_ptr() {
        if let Some(module) = vm.import_resolver.module_from_ptr(ptr) {
            return dir_from_names(module.dir());
        }

        if let Some(represented) = crate::builtins::builtin_type_object_type_id(ptr) {
            return dir_from_names(collect_builtin_type_dir_names(represented)?);
        }

        let type_id = crate::ops::objects::extract_type_id(ptr);
        if type_id == TypeId::TYPE {
            return dir_from_names(collect_heap_type_dir_names(unsafe {
                &*(ptr as *const PyClassObject)
            })?);
        }

        if is_user_defined_type(type_id) {
            let shaped = unsafe { &*(ptr as *const ShapedObject) };
            return dir_from_names(collect_heap_instance_dir_names(type_id, shaped)?);
        }

        return dir_from_names(collect_builtin_instance_dir_names(type_id)?);
    }

    Err(BuiltinError::NotImplemented(
        "dir() for this value is not implemented".to_string(),
    ))
}

#[inline]
fn value_owner_type_id(value: Value) -> Option<TypeId> {
    if value.is_none() {
        Some(TypeId::NONE)
    } else if value.is_bool() {
        Some(TypeId::BOOL)
    } else if value.is_int() {
        Some(TypeId::INT)
    } else if value.is_float() {
        Some(TypeId::FLOAT)
    } else if value.is_string() {
        Some(TypeId::STR)
    } else {
        None
    }
}

#[inline]
fn dir_from_names(mut names: Vec<InternedString>) -> Result<Value, BuiltinError> {
    names.sort_unstable_by(|left, right| left.as_str().cmp(right.as_str()));
    let list = ListObject::from_iter(names.into_iter().map(Value::string));
    Ok(crate::alloc_managed_value(list))
}

#[inline]
fn push_unique_name(
    names: &mut Vec<InternedString>,
    seen: &mut FxHashSet<InternedString>,
    name: InternedString,
) {
    if seen.insert(name.clone()) {
        names.push(name);
    }
}

#[inline]
fn value_to_name(value: Value) -> Result<InternedString, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("dir() expected string key".to_string()))?;
        return interned_by_ptr(ptr as *const u8)
            .or_else(|| {
                let string = unsafe { &*(ptr as *const StringObject) };
                Some(intern(string.as_str()))
            })
            .ok_or_else(|| BuiltinError::TypeError("dir() expected string key".to_string()));
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("dir() expected string key".to_string()))?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError(
            "dir() expected string key".to_string(),
        ));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(intern(string.as_str()))
}

fn runtime_error_to_builtin_error(err: RuntimeError) -> BuiltinError {
    let display = err.to_string();
    match err.into_kind() {
        RuntimeErrorKind::TypeError { message } => BuiltinError::TypeError(message.to_string()),
        RuntimeErrorKind::UnsupportedOperandTypes { op, left, right } => BuiltinError::TypeError(
            format!("unsupported operand type(s) for {op}: '{left}' and '{right}'"),
        ),
        RuntimeErrorKind::NotCallable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not callable", type_name))
        }
        RuntimeErrorKind::NotIterable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not iterable", type_name))
        }
        RuntimeErrorKind::NotSubscriptable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not subscriptable", type_name))
        }
        RuntimeErrorKind::AttributeError { type_name, attr } => BuiltinError::AttributeError(
            format!("'{}' object has no attribute '{}'", type_name, attr),
        ),
        RuntimeErrorKind::KeyError { key } => BuiltinError::KeyError(key.to_string()),
        RuntimeErrorKind::IndexError { index, length } => {
            BuiltinError::IndexError(format!("index {index} out of range for length {length}"))
        }
        RuntimeErrorKind::ValueError { message } => BuiltinError::ValueError(message.to_string()),
        RuntimeErrorKind::OverflowError { message } => {
            BuiltinError::OverflowError(message.to_string())
        }
        RuntimeErrorKind::StopIteration => BuiltinError::StopIteration,
        _ => BuiltinError::TypeError(display),
    }
}

fn append_builtin_direct_names(
    names: &mut Vec<InternedString>,
    seen: &mut FxHashSet<InternedString>,
    owner: TypeId,
) -> Result<(), BuiltinError> {
    let proxy = MappingProxyObject::for_builtin_type(owner);
    let keys = crate::builtins::builtin_mapping_proxy_keys(&proxy)
        .map_err(runtime_error_to_builtin_error)?;
    for key in keys {
        push_unique_name(names, seen, value_to_name(key)?);
    }
    Ok(())
}

fn append_class_mro_names(
    names: &mut Vec<InternedString>,
    seen: &mut FxHashSet<InternedString>,
    class: &PyClassObject,
) -> Result<(), BuiltinError> {
    for &class_id in class.mro() {
        if class_id == class.class_id() {
            class.for_each_attr(|name, _| push_unique_name(names, seen, name.clone()));
            continue;
        }

        if class_id.0 < TypeId::FIRST_USER_TYPE {
            append_builtin_direct_names(names, seen, class_id_to_type_id(class_id))?;
            continue;
        }

        if let Some(parent) = global_class(class_id) {
            parent.for_each_attr(|name, _| push_unique_name(names, seen, name.clone()));
        }
    }

    Ok(())
}

fn collect_builtin_instance_dir_names(owner: TypeId) -> Result<Vec<InternedString>, BuiltinError> {
    let mut names = Vec::new();
    let mut seen = FxHashSet::default();

    push_unique_name(&mut names, &mut seen, intern("__class__"));
    for class_id in builtin_class_mro(owner) {
        append_builtin_direct_names(&mut names, &mut seen, class_id_to_type_id(class_id))?;
    }

    Ok(names)
}

fn collect_builtin_type_dir_names(owner: TypeId) -> Result<Vec<InternedString>, BuiltinError> {
    let mut names = Vec::new();
    let mut seen = FxHashSet::default();

    for &name in TYPE_REFLECTION_NAMES {
        push_unique_name(&mut names, &mut seen, intern(name));
    }

    append_builtin_direct_names(&mut names, &mut seen, owner)?;
    Ok(names)
}

fn collect_heap_type_dir_names(class: &PyClassObject) -> Result<Vec<InternedString>, BuiltinError> {
    let mut names = Vec::new();
    let mut seen = FxHashSet::default();

    for &name in TYPE_REFLECTION_NAMES {
        push_unique_name(&mut names, &mut seen, intern(name));
    }

    append_class_mro_names(&mut names, &mut seen, class)?;
    Ok(names)
}

fn collect_heap_instance_dir_names(
    type_id: TypeId,
    shaped: &ShapedObject,
) -> Result<Vec<InternedString>, BuiltinError> {
    let mut names = Vec::new();
    let mut seen = FxHashSet::default();

    push_unique_name(&mut names, &mut seen, intern("__class__"));
    for name in shaped.property_names() {
        push_unique_name(&mut names, &mut seen, name);
    }

    let class = global_class(ClassId(type_id.raw())).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "missing heap type metadata for '{}'",
            type_id.name()
        ))
    })?;
    append_class_mro_names(&mut names, &mut seen, class.as_ref())?;
    Ok(names)
}

// =============================================================================
// vars() - Return __dict__
// =============================================================================

/// Builtin vars([object]) function.
///
/// Without arguments, acts like locals().
/// With an object, returns the __dict__ of the object.
///
/// # Python Semantics
/// - `vars()` → locals()
/// - `vars(obj)` → obj.__dict__
/// - `vars(None)` → TypeError
///
/// # Implementation Note
/// Full implementation requires __dict__ attribute access.
pub fn builtin_vars(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "vars() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    if args.is_empty() {
        // No argument: return locals()
        return Err(BuiltinError::NotImplemented(
            "vars() without argument requires frame introspection".to_string(),
        ));
    }

    let obj = &args[0];

    // Check for objects without __dict__
    if obj.is_none() {
        return Err(BuiltinError::TypeError(
            "vars() argument must have __dict__ attribute".to_string(),
        ));
    }

    if obj.is_int() || obj.is_float() || obj.is_bool() {
        return Err(BuiltinError::TypeError(
            "vars() argument must have __dict__ attribute".to_string(),
        ));
    }

    // TODO: Access object's __dict__ when available
    Err(BuiltinError::NotImplemented(
        "vars() for objects requires __dict__ access".to_string(),
    ))
}

/// VM-aware `vars([object])` implementation.
///
/// This returns the same live namespace object exposed by `obj.__dict__`.
/// With no argument it follows CPython and delegates to `locals()`.
pub fn builtin_vars_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "vars() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    if args.is_empty() {
        return builtin_locals_vm(vm, args);
    }

    get_attribute_value(vm, args[0], &intern("__dict__")).map_err(|err| {
        if err.is_attribute_error() {
            BuiltinError::TypeError("vars() argument must have __dict__ attribute".to_string())
        } else {
            super::runtime_error_to_builtin_error(err)
        }
    })
}

// =============================================================================
// globals() - Global Symbol Table
// =============================================================================

/// Builtin globals() function.
///
/// Returns a dictionary representing the current global symbol table.
///
/// # Python Semantics
/// - Always returns the globals dict of the current module
/// - Modifications to the returned dict affect the actual globals
///
/// # Implementation Note
/// Requires access to the module's global namespace.
pub fn builtin_globals(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "globals() takes no arguments ({} given)",
            args.len()
        )));
    }

    // TODO: Return actual globals dict from current frame
    Err(BuiltinError::NotImplemented(
        "globals() requires frame introspection".to_string(),
    ))
}

/// VM-aware `globals()` implementation.
///
/// Module-backed frames return the module's live dictionary, so mutations made
/// through the returned object are immediately visible to global name lookup.
pub fn builtin_globals_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "globals() takes no arguments ({} given)",
            args.len()
        )));
    }

    let module = vm
        .current_frame()
        .module
        .as_ref()
        .cloned()
        .or_else(|| vm.current_module_cloned());
    if let Some(module) = module {
        return Ok(module.dict_value());
    }

    alloc_heap_value(
        vm,
        snapshot_current_globals_dict(vm),
        "globals namespace dict",
    )
    .map_err(super::runtime_error_to_builtin_error)
}

// =============================================================================
// locals() - Local Symbol Table
// =============================================================================

/// Builtin locals() function.
///
/// Returns a dictionary representing the current local symbol table.
///
/// # Python Semantics
/// - In function: returns a copy of local variables
/// - At module level: same as globals()
/// - Modifications may not affect actual locals (implementation-defined)
///
/// # Implementation Note  
/// Requires access to the current frame's local namespace.
pub fn builtin_locals(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "locals() takes no arguments ({} given)",
            args.len()
        )));
    }

    // TODO: Return actual locals dict from current frame
    Err(BuiltinError::NotImplemented(
        "locals() requires frame introspection".to_string(),
    ))
}

/// VM-aware `locals()` implementation.
///
/// At module scope CPython returns the same dictionary as `globals()`. In
/// function/class scopes without a live mapping, Prism returns a freshly
/// materialized locals dictionary from the current frame's assigned locals.
pub fn builtin_locals_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "locals() takes no arguments ({} given)",
            args.len()
        )));
    }

    let frame = vm.current_frame();
    if let Some(mapping) = frame.locals_mapping() {
        return Ok(mapping);
    }

    if frame.return_frame.is_none()
        && let Some(module) = frame.module.as_ref().cloned()
    {
        return Ok(module.dict_value());
    }

    let locals = snapshot_frame_locals_dict(frame);
    alloc_heap_value(vm, locals, "locals namespace dict")
        .map_err(super::runtime_error_to_builtin_error)
}

// =============================================================================
// help() - Interactive Help
// =============================================================================

/// Builtin help([object]) function.
///
/// Invokes the built-in help system.
///
/// # Python Semantics
/// - `help()` → Start interactive help
/// - `help(obj)` → Show help for object
/// - `help('topic')` → Show help for topic
///
/// # Implementation Note
/// Full interactive help is not implemented.
/// This provides a stub that returns NotImplemented.
pub fn builtin_help(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "help() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    // For now, help() is a stub
    // In a full implementation, this would:
    // 1. Access object's __doc__
    // 2. Format and display documentation
    // 3. Use pydoc for topics

    Err(BuiltinError::NotImplemented(
        "help() is not implemented".to_string(),
    ))
}

// =============================================================================
// __import__() - Import System Hook
// =============================================================================

/// Builtin __import__(name, ...) function.
///
/// This function is invoked by the import statement.
///
/// # Python Semantics
/// - `__import__('os')` → <module 'os'>
/// - Usually not called directly; use import statement
///
pub fn builtin_import(args: &[Value]) -> Result<Value, BuiltinError> {
    validate_import_arity(args)?;
    Err(BuiltinError::TypeError(
        "__import__() requires VM context".to_string(),
    ))
}

/// VM-backed `__import__` implementation that delegates to Prism's import
/// resolver rather than maintaining a second import pipeline inside builtins.
pub fn builtin_import_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    validate_import_arity(args)?;

    let name = value_to_python_string(args[0], "__import__() argument 'name'")?;
    let level = import_level_arg(args.get(4).copied())?;
    let module_spec = import_module_spec(name.as_ref(), level);
    let current_module = vm.current_module_cloned();
    let imported = vm
        .import_module_with_context(&module_spec, current_module.as_ref())
        .map_err(runtime_error_to_builtin_import_error)?;

    let fromlist = import_fromlist_items(args.get(3).copied())?;
    if !fromlist.is_empty() {
        import_requested_fromlist_members(vm, &imported, current_module.as_ref(), &fromlist)?;
        return Ok(module_value(&imported));
    }

    if let Some((top_level, _)) = imported.name().split_once('.') {
        let top = vm
            .import_module_with_context(top_level, current_module.as_ref())
            .map_err(runtime_error_to_builtin_import_error)?;
        return Ok(module_value(&top));
    }

    Ok(module_value(&imported))
}

#[inline]
fn validate_import_arity(args: &[Value]) -> Result<(), BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "__import__() missing required argument: 'name'".to_string(),
        ));
    }
    if args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "__import__() takes at most 5 arguments ({} given)",
            args.len()
        )));
    }
    Ok(())
}

#[inline]
fn value_to_python_string(value: Value, context: &str) -> Result<Arc<str>, BuiltinError> {
    if let Some(ptr) = value.as_string_object_ptr() {
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a str")))?;
        return Ok(Arc::from(interned.as_ref()));
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!("{context} must be a str")));
    };

    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError(format!("{context} must be a str")));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(Arc::from(string.as_str()))
}

#[inline]
fn import_level_arg(value: Option<Value>) -> Result<usize, BuiltinError> {
    let Some(value) = value else {
        return Ok(0);
    };

    let Some(level) = value.as_int() else {
        return Err(BuiltinError::TypeError(
            "__import__() argument 'level' must be an int".to_string(),
        ));
    };

    usize::try_from(level).map_err(|_| BuiltinError::ValueError("level must be >= 0".to_string()))
}

#[inline]
fn import_module_spec(name: &str, level: usize) -> String {
    if level == 0 {
        name.to_string()
    } else {
        ".".repeat(level) + name
    }
}

fn import_fromlist_items(value: Option<Value>) -> Result<Vec<Arc<str>>, BuiltinError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };

    if !crate::truthiness::is_truthy(value) {
        return Ok(Vec::new());
    }

    if value.is_string()
        || value
            .as_object_ptr()
            .is_some_and(|ptr| crate::ops::objects::extract_type_id(ptr) == TypeId::STR)
    {
        return Ok(vec![value_to_python_string(
            value,
            "__import__() argument 'fromlist'",
        )?]);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "__import__() argument 'fromlist' must be a sequence of strings".to_string(),
        ));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::LIST => {
            let list = unsafe { &*(ptr as *const ListObject) };
            list.as_slice()
                .iter()
                .copied()
                .map(|item| value_to_python_string(item, "__import__() argument 'fromlist'"))
                .collect()
        }
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple
                .as_slice()
                .iter()
                .copied()
                .map(|item| value_to_python_string(item, "__import__() argument 'fromlist'"))
                .collect()
        }
        _ => Err(BuiltinError::TypeError(
            "__import__() argument 'fromlist' must be a sequence of strings".to_string(),
        )),
    }
}

fn import_requested_fromlist_members(
    vm: &mut VirtualMachine,
    module: &Arc<ModuleObject>,
    current_module: Option<&Arc<ModuleObject>>,
    fromlist: &[Arc<str>],
) -> Result<(), BuiltinError> {
    if !is_package_module(module) {
        return Ok(());
    }

    for item in fromlist {
        if item.as_ref() == "*" || module.has_attr(item.as_ref()) {
            continue;
        }

        let qualified_name = format!("{}.{}", module.name(), item);
        if let Ok(submodule) = vm.import_module_with_context(&qualified_name, current_module) {
            module.set_attr(item.as_ref(), module_value(&submodule));
        }
    }

    Ok(())
}

#[inline]
fn is_package_module(module: &ModuleObject) -> bool {
    module.package_name() == Some(module.name())
}

#[inline]
fn module_value(module: &Arc<ModuleObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(module) as *const ())
}

fn runtime_error_to_builtin_import_error(err: RuntimeError) -> BuiltinError {
    match err.into_kind() {
        RuntimeErrorKind::ImportError {
            message, missing, ..
        } => {
            if missing {
                BuiltinError::ModuleNotFoundError(message.to_string())
            } else {
                BuiltinError::ImportError(message.to_string())
            }
        }
        other => BuiltinError::ImportError(RuntimeError::new(other).to_string()),
    }
}

// =============================================================================
// hasattr, getattr, setattr, delattr - Already in types.rs
// =============================================================================

// Note: hasattr, getattr, setattr, delattr are already implemented
// in types.rs. They are core attribute access functions.

// =============================================================================
// isinstance, issubclass - Already in types.rs
// =============================================================================

// Note: isinstance, issubclass are already implemented in types.rs.
// They are type checking functions.

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
