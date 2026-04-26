//! Native `_warnings` bootstrap module.
//!
//! CPython's `warnings.py` expects a small C-accelerated compatibility surface
//! to exist during startup. Prism still relies on the Python stdlib module for
//! higher-level warning semantics, but this native module provides the stable
//! bootstrap state and callables that let that import complete cleanly.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, ExceptionValue, exception_proxy_class,
    exception_proxy_class_id_from_ptr, get_exception_type,
};
use crate::error::RuntimeError;
use crate::import::ModuleObject;
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{
    dict_storage_mut_from_ptr, get_attribute_value, list_storage_ref_from_ptr,
};
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::truthiness::try_is_truthy;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class_bitmap;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

static FILTERS_MUTATED_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_warnings._filters_mutated"), filters_mutated)
});
static WARN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm_kw(Arc::from("_warnings.warn"), warn_kw));
static WARN_EXPLICIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_warnings.warn_explicit"), warn_explicit)
});

static FILTERS_VERSION: AtomicUsize = AtomicUsize::new(1);

pub(crate) const BOOL_INVERT_DEPRECATION_MESSAGE: &str = "Bitwise inversion '~' on bool is deprecated and will be removed in Python 3.16. This returns the bitwise inversion of the underlying int object and is usually not what you expect from negating a bool. Use the 'not' operator for boolean negation or ~int(x) if you really want the bitwise inversion of the underlying int.";

/// Native `_warnings` module descriptor.
#[derive(Debug, Clone)]
pub struct WarningsModule {
    filters_value: Value,
    defaultaction_value: Value,
    onceregistry_value: Value,
    attrs: Vec<Arc<str>>,
}

impl WarningsModule {
    /// Create a new `_warnings` module descriptor.
    pub fn new() -> Self {
        Self {
            filters_value: leak_object_value(ListObject::new()),
            defaultaction_value: Value::string(intern("default")),
            onceregistry_value: leak_object_value(DictObject::new()),
            attrs: vec![
                Arc::from("_defaultaction"),
                Arc::from("_filters_mutated"),
                Arc::from("_onceregistry"),
                Arc::from("filters"),
                Arc::from("warn"),
                Arc::from("warn_explicit"),
            ],
        }
    }
}

impl Default for WarningsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for WarningsModule {
    fn name(&self) -> &str {
        "_warnings"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "filters" => Ok(self.filters_value),
            "_defaultaction" => Ok(self.defaultaction_value),
            "_onceregistry" => Ok(self.onceregistry_value),
            "_filters_mutated" => Ok(builtin_value(&FILTERS_MUTATED_FUNCTION)),
            "warn" => Ok(builtin_value(&WARN_FUNCTION)),
            "warn_explicit" => Ok(builtin_value(&WARN_EXPLICIT_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_warnings' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn filters_mutated(_args: &[Value]) -> Result<Value, BuiltinError> {
    FILTERS_VERSION.fetch_add(1, Ordering::Relaxed);
    Ok(Value::none())
}

pub(crate) fn emit_bool_invert_deprecation_warning(
    vm: &mut VirtualMachine,
) -> Result<(), RuntimeError> {
    let context = WarningContext::capture(vm)?;
    emit_warning(
        vm,
        builtin_warning_category(ExceptionTypeId::DeprecationWarning)?,
        BOOL_INVERT_DEPRECATION_MESSAGE,
        &context,
    )
}

#[derive(Clone, Copy, Debug)]
struct WarnCall {
    message: Value,
    category: Option<Value>,
    stacklevel: i64,
}

fn warn_kw(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let call = bind_warn_args(args, keywords)?;
    let message = warning_message_text(call.message).map_err(BuiltinError::TypeError)?;
    let category = warning_category_from_value(call.category)
        .map_err(BuiltinError::TypeError)?
        .unwrap_or_else(|| {
            builtin_warning_category(ExceptionTypeId::UserWarning)
                .expect("UserWarning must be a valid warning category")
        });
    let context = WarningContext::capture_at_depth(vm, warning_stack_depth(call.stacklevel))
        .map_err(BuiltinError::Raised)?;
    emit_warning(vm, category, &message, &context).map_err(BuiltinError::Raised)?;
    Ok(Value::none())
}

fn bind_warn_args(args: &[Value], keywords: &[(&str, Value)]) -> Result<WarnCall, BuiltinError> {
    if args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "warn() takes at most 5 arguments ({} given)",
            args.len()
        )));
    }

    let mut message = args.first().copied();
    let mut category = args.get(1).copied();
    let mut stacklevel = args.get(2).copied();
    let mut _source = args.get(3).copied();
    let mut _skip_file_prefixes = args.get(4).copied();

    for (name, value) in keywords {
        match *name {
            "message" => assign_warn_keyword(&mut message, *value, "message")?,
            "category" => assign_warn_keyword(&mut category, *value, "category")?,
            "stacklevel" => assign_warn_keyword(&mut stacklevel, *value, "stacklevel")?,
            "source" => assign_warn_keyword(&mut _source, *value, "source")?,
            "skip_file_prefixes" => {
                assign_warn_keyword(&mut _skip_file_prefixes, *value, "skip_file_prefixes")?
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "warn() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let message = message.ok_or_else(|| {
        BuiltinError::TypeError("warn() missing required argument 'message'".to_string())
    })?;
    let stacklevel = stacklevel
        .filter(|value| !value.is_none())
        .map(|value| value_to_i64(value, "warn() argument 3 must be int"))
        .transpose()
        .map_err(BuiltinError::TypeError)?
        .unwrap_or(1);

    Ok(WarnCall {
        message,
        category,
        stacklevel,
    })
}

fn assign_warn_keyword(
    slot: &mut Option<Value>,
    value: Value,
    name: &'static str,
) -> Result<(), BuiltinError> {
    if slot.is_some() {
        return Err(BuiltinError::TypeError(format!(
            "warn() got multiple values for argument '{}'",
            name
        )));
    }
    *slot = Some(value);
    Ok(())
}

#[inline]
fn warning_stack_depth(stacklevel: i64) -> usize {
    usize::try_from(stacklevel.saturating_sub(1)).unwrap_or(0)
}

fn warn_explicit(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 4 {
        return Err(BuiltinError::TypeError(
            "warn_explicit() requires at least 4 arguments".to_string(),
        ));
    }

    let message = warning_message_text(args[0]).map_err(BuiltinError::TypeError)?;
    let category = warning_category_from_value(args.get(1).copied())
        .map_err(BuiltinError::TypeError)?
        .unwrap_or_else(|| {
            builtin_warning_category(ExceptionTypeId::UserWarning)
                .expect("UserWarning must be a valid warning category")
        });
    let filename = value_to_string(args[2], "warn_explicit() arg 3 must be str")
        .map_err(BuiltinError::TypeError)?;
    let lineno = value_to_i64(args[3], "warn_explicit() arg 4 must be int")
        .map_err(BuiltinError::TypeError)? as u32;
    let module_name = args
        .get(4)
        .copied()
        .filter(|value| !value.is_none())
        .map(|value| value_to_string(value, "warn_explicit() arg 5 must be str"))
        .transpose()
        .map_err(BuiltinError::TypeError)?
        .unwrap_or_else(|| module_name_from_filename(&filename));
    let registry = args
        .get(5)
        .copied()
        .filter(|value| !value.is_none())
        .and_then(dict_ptr_from_value);

    let context = WarningContext {
        filename,
        lineno,
        module_name,
        registry,
    };
    emit_warning(vm, category, &message, &context).map_err(BuiltinError::Raised)?;
    Ok(Value::none())
}

#[derive(Debug)]
struct WarningContext {
    filename: String,
    lineno: u32,
    module_name: String,
    registry: Option<*mut DictObject>,
}

impl WarningContext {
    fn capture(vm: &mut VirtualMachine) -> Result<Self, RuntimeError> {
        Self::capture_at_depth(vm, 0)
    }

    fn capture_at_depth(vm: &mut VirtualMachine, depth: usize) -> Result<Self, RuntimeError> {
        let (filename, lineno, module) = {
            let frame = vm
                .frame_at_depth(depth)
                .unwrap_or_else(|| vm.current_frame());
            let pc = frame.ip.saturating_sub(1);
            let line = frame
                .code
                .line_for_pc(pc)
                .unwrap_or(frame.code.first_lineno);
            (frame.code.filename.to_string(), line, frame.module.clone())
        };

        let Some(module) = module else {
            return Ok(Self {
                filename,
                lineno,
                module_name: "<string>".to_string(),
                registry: None,
            });
        };

        let registry = ensure_warning_registry(vm, &module)?;
        Ok(Self {
            filename,
            lineno,
            module_name: module.name().to_string(),
            registry: Some(registry),
        })
    }
}

fn emit_warning(
    vm: &mut VirtualMachine,
    category: WarningCategory,
    message: &str,
    context: &WarningContext,
) -> Result<(), RuntimeError> {
    let warnings = match vm.import_module_named("warnings") {
        Ok(module) => module,
        Err(_) => return Ok(()),
    };

    let text_value = Value::string(intern(message));
    let warning_instance = instantiate_warning_instance(vm, category, message)?;

    let action = resolve_warning_action(
        vm,
        &warnings,
        text_value,
        category,
        &context.module_name,
        context.lineno,
        context.registry,
    )?;

    match action.as_str() {
        "ignore" => return Ok(()),
        "error" => {
            return Err(RuntimeError::raised_exception(
                category.exception_type_id_for_raise(),
                warning_instance,
                Arc::from(message),
            ));
        }
        "once" => {
            let Some(onceregistry_ptr) = warnings
                .get_attr("onceregistry")
                .and_then(dict_ptr_from_value)
            else {
                return Ok(());
            };
            let once_key = alloc_tuple_value(
                vm,
                &[text_value, category.value],
                "warning once registry key",
            )?;
            let onceregistry = unsafe { &mut *onceregistry_ptr };
            if onceregistry.get(once_key).is_some() {
                return Ok(());
            }
            onceregistry.set(once_key, Value::int(1).unwrap());
        }
        _ => {}
    }

    if let Some(registry) = context.registry {
        match action.as_str() {
            "once" | "default" => {
                write_warning_registry_entry(
                    vm,
                    registry,
                    &[
                        text_value,
                        category.value,
                        Value::int(context.lineno as i64).unwrap(),
                    ],
                )?;
            }
            "module" => {
                write_warning_registry_entry(
                    vm,
                    registry,
                    &[
                        text_value,
                        category.value,
                        Value::int(context.lineno as i64).unwrap(),
                    ],
                )?;
                let altkey = alloc_tuple_value(
                    vm,
                    &[
                        text_value,
                        category.value,
                        Value::int(0).expect("zero should fit in tagged int"),
                    ],
                    "warning module registry key",
                )?;
                let registry = unsafe { &mut *registry };
                if registry.get(altkey).is_some() {
                    return Ok(());
                }
                registry.set(altkey, Value::int(1).unwrap());
            }
            _ => {}
        }
    }

    let warning_message = instantiate_warning_message(
        vm,
        &warnings,
        warning_instance,
        category.value,
        &context.filename,
        context.lineno,
    )?;
    let showwarnmsg = warnings
        .get_attr("_showwarnmsg")
        .ok_or_else(|| RuntimeError::attribute_error("module", "_showwarnmsg"))?;
    invoke_callable_value(vm, showwarnmsg, &[warning_message])?;
    Ok(())
}

fn resolve_warning_action(
    vm: &mut VirtualMachine,
    warnings: &Arc<ModuleObject>,
    text: Value,
    category: WarningCategory,
    module_name: &str,
    lineno: u32,
    registry: Option<*mut DictObject>,
) -> Result<String, RuntimeError> {
    if let Some(registry) = registry {
        refresh_warning_registry(registry);
        let key = alloc_tuple_value(
            vm,
            &[text, category.value, Value::int(lineno as i64).unwrap()],
            "warning registry key",
        )?;
        if unsafe { &mut *registry }.get(key).is_some() {
            return Ok("ignore".to_string());
        }
    }

    let filters = warnings
        .get_attr("filters")
        .and_then(|value| value.as_object_ptr())
        .and_then(list_storage_ref_from_ptr);
    if let Some(filters) = filters {
        for item in filters.as_slice() {
            if let Some(action) = match_filter(vm, *item, text, category, module_name, lineno)? {
                return Ok(action);
            }
        }
    }

    warnings
        .get_attr("defaultaction")
        .ok_or_else(|| RuntimeError::attribute_error("module", "defaultaction"))
        .and_then(|value| warning_action_string(value))
}

fn match_filter(
    vm: &mut VirtualMachine,
    item: Value,
    text: Value,
    category: WarningCategory,
    module_name: &str,
    lineno: u32,
) -> Result<Option<String>, RuntimeError> {
    let Some(tuple_ptr) = item.as_object_ptr() else {
        return Ok(None);
    };
    let header = unsafe { &*(tuple_ptr as *const ObjectHeader) };
    if header.type_id != TypeId::TUPLE {
        return Ok(None);
    }

    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
    if tuple.len() != 5 {
        return Ok(None);
    }

    let action = tuple.get(0).unwrap();
    let msg = tuple.get(1).unwrap();
    let cat = tuple.get(2).unwrap();
    let module = tuple.get(3).unwrap();
    let line = tuple.get(4).unwrap();

    if !filter_pattern_matches(vm, msg, text)? {
        return Ok(None);
    }
    if !warning_category_matches(category, cat)? {
        return Ok(None);
    }
    if !filter_pattern_matches(vm, module, Value::string(intern(module_name)))? {
        return Ok(None);
    }
    let line = value_to_i64(line, "warnings filter line number must be int")
        .map_err(RuntimeError::type_error)?;
    if line != 0 && line as u32 != lineno {
        return Ok(None);
    }

    warning_action_string(action).map(Some)
}

fn filter_pattern_matches(
    vm: &mut VirtualMachine,
    pattern: Value,
    text: Value,
) -> Result<bool, RuntimeError> {
    if pattern.is_none() {
        return Ok(true);
    }

    let matcher = get_attribute_value(vm, pattern, &intern("match"))?;
    let matched = invoke_callable_value(vm, matcher, &[text])?;
    try_is_truthy(vm, matched)
}

fn warning_category_matches(
    actual: WarningCategory,
    expected: Value,
) -> Result<bool, RuntimeError> {
    let Some(expected_class_id) = warning_category_class_id(expected) else {
        return Ok(false);
    };
    Ok(global_class_bitmap(actual.class_id)
        .map(|bitmap| bitmap.is_subclass_of(TypeId::from_raw(expected_class_id.0)))
        .unwrap_or(actual.class_id == expected_class_id))
}

fn instantiate_warning_instance(
    vm: &mut VirtualMachine,
    category: WarningCategory,
    message: &str,
) -> Result<Value, RuntimeError> {
    if let Some(kind) = category.exception_type_id {
        return Ok(crate::builtins::create_exception(
            kind,
            Some(Arc::from(message)),
        ));
    }

    invoke_callable_value(vm, category.value, &[Value::string(intern(message))])
}

fn instantiate_warning_message(
    vm: &mut VirtualMachine,
    warnings: &Arc<ModuleObject>,
    warning_instance: Value,
    category: Value,
    filename: &str,
    lineno: u32,
) -> Result<Value, RuntimeError> {
    let warning_message = warnings
        .get_attr("WarningMessage")
        .ok_or_else(|| RuntimeError::attribute_error("module", "WarningMessage"))?;
    invoke_callable_value(
        vm,
        warning_message,
        &[
            warning_instance,
            category,
            Value::string(intern(filename)),
            Value::int(lineno as i64).unwrap(),
        ],
    )
}

fn ensure_warning_registry(
    vm: &mut VirtualMachine,
    module: &Arc<ModuleObject>,
) -> Result<*mut DictObject, RuntimeError> {
    if let Some(existing) = module.get_attr("__warningregistry__")
        && let Some(ptr) = dict_ptr_from_value(existing)
    {
        return Ok(ptr);
    }

    let ptr = vm.allocator().alloc(DictObject::new()).ok_or_else(|| {
        RuntimeError::internal("out of memory: failed to allocate warning registry")
    })?;
    module.set_attr("__warningregistry__", Value::object_ptr(ptr as *const ()));
    Ok(ptr)
}

fn refresh_warning_registry(registry_ptr: *mut DictObject) {
    let registry = unsafe { &mut *registry_ptr };
    let version_key = Value::string(intern("version"));
    let current_version = FILTERS_VERSION.load(Ordering::Relaxed) as i64;
    if registry.get(version_key).and_then(|value| value.as_int()) != Some(current_version) {
        registry.clear();
        registry.set(version_key, Value::int(current_version).unwrap());
    }
}

fn write_warning_registry_entry(
    vm: &mut VirtualMachine,
    registry_ptr: *mut DictObject,
    key_items: &[Value],
) -> Result<(), RuntimeError> {
    let key = alloc_tuple_value(vm, key_items, "warning registry key")?;
    let registry = unsafe { &mut *registry_ptr };
    registry.set(key, Value::int(1).unwrap());
    Ok(())
}

fn alloc_tuple_value(
    vm: &mut VirtualMachine,
    items: &[Value],
    context: &'static str,
) -> Result<Value, RuntimeError> {
    vm.allocator()
        .alloc(TupleObject::from_slice(items))
        .map(|ptr| Value::object_ptr(ptr as *const ()))
        .ok_or_else(|| {
            RuntimeError::internal(format!("out of memory: failed to allocate {context}"))
        })
}

fn dict_ptr_from_value(value: Value) -> Option<*mut DictObject> {
    let ptr = value.as_object_ptr()?;
    dict_storage_mut_from_ptr(ptr).map(|dict| dict as *mut DictObject)
}

fn category_value(category: ExceptionTypeId) -> Result<Value, RuntimeError> {
    let exception_type = get_exception_type(category.name())
        .ok_or_else(|| RuntimeError::name_error(category.name()))?;
    Ok(Value::object_ptr(
        exception_type as *const crate::builtins::ExceptionTypeObject as *const (),
    ))
}

#[derive(Clone, Copy, Debug)]
struct WarningCategory {
    value: Value,
    class_id: ClassId,
    exception_type_id: Option<ExceptionTypeId>,
}

impl WarningCategory {
    #[inline]
    fn exception_type_id_for_raise(self) -> u16 {
        self.exception_type_id.unwrap_or(ExceptionTypeId::Warning) as u16
    }
}

fn builtin_warning_category(category: ExceptionTypeId) -> Result<WarningCategory, RuntimeError> {
    let value = category_value(category)?;
    let class_id = warning_category_class_id(value).ok_or_else(|| {
        RuntimeError::internal(format!(
            "{} did not publish an exception proxy class",
            category.name()
        ))
    })?;
    Ok(WarningCategory {
        value,
        class_id,
        exception_type_id: Some(category),
    })
}

fn warning_category_from_value(value: Option<Value>) -> Result<Option<WarningCategory>, String> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(None);
    };

    let Some(class_id) = warning_category_class_id(value) else {
        return Err("warning category must be a Warning subclass".to_string());
    };
    if !is_warning_category_class_id(class_id) {
        return Err("warning category must be a Warning subclass".to_string());
    }

    Ok(Some(WarningCategory {
        value,
        class_id,
        exception_type_id: warning_category_exception_type_id(value),
    }))
}

fn warning_category_exception_type_id(value: Value) -> Option<ExceptionTypeId> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != crate::builtins::EXCEPTION_TYPE_ID {
        return None;
    }

    let exception_type = unsafe { &*(ptr as *const crate::builtins::ExceptionTypeObject) };
    ExceptionTypeId::from_u8(exception_type.exception_type_id as u8)
}

fn warning_category_class_id(value: Value) -> Option<ClassId> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id == crate::builtins::EXCEPTION_TYPE_ID {
        return exception_proxy_class_id_from_ptr(ptr);
    }
    if header.type_id == TypeId::TYPE && crate::builtins::builtin_type_object_type_id(ptr).is_none()
    {
        let class = unsafe { &*(ptr as *const PyClassObject) };
        return Some(class.class_id());
    }
    None
}

fn is_warning_category_class_id(class_id: ClassId) -> bool {
    let warning_id = exception_proxy_class(ExceptionTypeId::Warning).class_id();
    global_class_bitmap(class_id)
        .map(|bitmap| bitmap.is_subclass_of(TypeId::from_raw(warning_id.0)))
        .unwrap_or(class_id == warning_id)
}

fn warning_message_text(value: Value) -> Result<String, String> {
    if let Some(exception) = unsafe { ExceptionValue::from_value(value) }
        && exception.is_subclass_of(ExceptionTypeId::Warning)
    {
        return Ok(exception.display_text());
    }

    value_to_string(value, "warning message must be str or Warning instance")
}

fn value_to_string(value: Value, context: &str) -> Result<String, String> {
    value_as_string_ref(value)
        .map(|text| text.as_str().to_string())
        .ok_or_else(|| context.to_string())
}

fn value_to_i64(value: Value, context: &str) -> Result<i64, String> {
    value.as_int().ok_or_else(|| context.to_string())
}

fn warning_action_string(value: Value) -> Result<String, RuntimeError> {
    value_as_string_ref(value)
        .map(|text| text.as_str().to_string())
        .ok_or_else(|| RuntimeError::type_error("warning action must be str"))
}

fn module_name_from_filename(filename: &str) -> String {
    filename.strip_suffix(".py").unwrap_or(filename).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::{intern, interned_by_ptr};

    #[test]
    fn test_warnings_module_imports_as_builtin_anchor() {
        let module = WarningsModule::new();
        assert_eq!(module.name(), "_warnings");
        assert!(module.dir().contains(&Arc::from("filters")));
    }

    #[test]
    fn test_warnings_module_exposes_bootstrap_state() {
        let module = WarningsModule::new();

        let defaultaction = module
            .get_attr("_defaultaction")
            .expect("_defaultaction should exist");
        let defaultaction_ptr = defaultaction
            .as_string_object_ptr()
            .expect("_defaultaction should be an interned string");
        assert_eq!(
            interned_by_ptr(defaultaction_ptr as *const u8)
                .unwrap()
                .as_str(),
            "default"
        );

        let filters = module.get_attr("filters").expect("filters should exist");
        let filters_ptr = filters
            .as_object_ptr()
            .expect("filters should be a list object");
        let filters = unsafe { &*(filters_ptr as *const ListObject) };
        assert!(filters.is_empty());

        let onceregistry = module
            .get_attr("_onceregistry")
            .expect("_onceregistry should exist");
        let onceregistry_ptr = onceregistry
            .as_object_ptr()
            .expect("_onceregistry should be a dict object");
        let onceregistry = unsafe { &*(onceregistry_ptr as *const DictObject) };
        assert!(onceregistry.is_empty());
    }

    #[test]
    fn test_warnings_module_exposes_callable_bootstrap_functions() {
        let module = WarningsModule::new();

        for name in ["_filters_mutated", "warn", "warn_explicit"] {
            let value = module.get_attr(name).expect("callable should exist");
            let ptr = value
                .as_object_ptr()
                .expect("callable should be a builtin function object");
            let header = unsafe { &*(ptr as *const ObjectHeader) };
            assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
        }
    }

    #[test]
    fn test_warn_accepts_cpython_keyword_surface() {
        let message = Value::string(intern("hello"));
        let category = category_value(ExceptionTypeId::DeprecationWarning)
            .expect("DeprecationWarning should be available");

        let call = bind_warn_args(
            &[],
            &[
                ("message", message),
                ("category", category),
                ("stacklevel", Value::int(3).unwrap()),
                ("source", Value::none()),
                ("skip_file_prefixes", Value::none()),
            ],
        )
        .expect("warn should bind CPython keyword arguments");

        assert_eq!(call.message, message);
        assert_eq!(call.category, Some(category));
        assert_eq!(call.stacklevel, 3);
        assert_eq!(warning_stack_depth(call.stacklevel), 2);
    }

    #[test]
    fn test_warn_keyword_binding_rejects_duplicates_and_unknowns() {
        let message = Value::string(intern("hello"));
        let duplicate = bind_warn_args(&[message], &[("message", message)])
            .expect_err("duplicate message should be rejected");
        assert!(
            matches!(duplicate, BuiltinError::TypeError(ref msg) if msg.contains("multiple values"))
        );

        let unknown = bind_warn_args(&[message], &[("bogus", Value::none())])
            .expect_err("unknown keywords should be rejected");
        assert!(
            matches!(unknown, BuiltinError::TypeError(ref msg) if msg.contains("unexpected keyword"))
        );
    }

    #[test]
    fn test_warning_category_accepts_supplemental_heap_warning_classes() {
        let resource_warning = crate::builtins::supplemental_exception_class("ResourceWarning")
            .expect("ResourceWarning supplemental class should exist");
        let value =
            Value::object_ptr(resource_warning.as_ref() as *const PyClassObject as *const ());

        let category = warning_category_from_value(Some(value))
            .expect("ResourceWarning should be a valid category")
            .expect("category should be present");

        assert_eq!(category.value, value);
        assert_eq!(category.class_id, resource_warning.class_id());
        assert_eq!(category.exception_type_id, None);
        assert!(is_warning_category_class_id(category.class_id));
    }

    #[test]
    fn test_warning_category_matches_builtin_and_heap_category_hierarchy() {
        let resource_warning = crate::builtins::supplemental_exception_class("ResourceWarning")
            .expect("ResourceWarning supplemental class should exist");
        let resource_value =
            Value::object_ptr(resource_warning.as_ref() as *const PyClassObject as *const ());
        let category = warning_category_from_value(Some(resource_value))
            .expect("ResourceWarning should be a valid category")
            .expect("category should be present");

        let warning_value = category_value(ExceptionTypeId::Warning).expect("Warning exists");
        let runtime_warning_value =
            category_value(ExceptionTypeId::RuntimeWarning).expect("RuntimeWarning exists");

        assert!(warning_category_matches(category, warning_value).unwrap());
        assert!(warning_category_matches(category, resource_value).unwrap());
        assert!(!warning_category_matches(category, runtime_warning_value).unwrap());
    }

    #[test]
    fn test_warning_category_rejects_non_warning_heap_classes() {
        let plain_class = PyClassObject::new_simple(intern("PlainCategory"));
        let value = Value::object_ptr(&plain_class as *const PyClassObject as *const ());

        let error = warning_category_from_value(Some(value))
            .expect_err("plain heap classes are not warning categories");

        assert_eq!(error, "warning category must be a Warning subclass");
    }
}
