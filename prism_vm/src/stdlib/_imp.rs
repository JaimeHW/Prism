//! Native `_imp` bootstrap module.
//!
//! This module provides the low-level import hooks that CPython's
//! `importlib` bootstrap expects to exist. Prism's import pipeline remains the
//! authoritative implementation; `_imp` exposes the compatibility surface
//! importlib uses to cooperate with that pipeline.

use super::{Module, ModuleError, ModuleResult, is_builtin_module_name};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::import::ModuleObject;
use crate::ops::attribute::is_user_defined_type;
use crate::ops::objects::extract_type_id;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread::ThreadId;

static ACQUIRE_LOCK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_imp.acquire_lock"), acquire_lock));
static RELEASE_LOCK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_imp.release_lock"), release_lock));
static LOCK_HELD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_imp.lock_held"), lock_held));
static IS_BUILTIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_imp.is_builtin"), is_builtin));
static CREATE_BUILTIN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_imp.create_builtin"), create_builtin)
});
static EXEC_BUILTIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_imp.exec_builtin"), exec_builtin));
static EXTENSION_SUFFIXES_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_imp.extension_suffixes"), extension_suffixes)
});
static IS_FROZEN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_imp.is_frozen"), is_frozen));
static IS_FROZEN_PACKAGE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_imp.is_frozen_package"), is_frozen_package)
});
static FIND_FROZEN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_imp.find_frozen"), find_frozen));
static GET_FROZEN_OBJECT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_imp.get_frozen_object"), get_frozen_object)
});
static SOURCE_HASH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_imp.source_hash"), source_hash));
static FIX_CO_FILENAME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_imp._fix_co_filename"), fix_co_filename)
});
static OVERRIDE_MULTI_INTERP_EXTENSIONS_CHECK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_imp._override_multi_interp_extensions_check"),
            override_multi_interp_extensions_check,
        )
    });
static CREATE_DYNAMIC_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_imp.create_dynamic"), create_dynamic)
});
static EXEC_DYNAMIC_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_imp.exec_dynamic"), exec_dynamic));

static CHECK_HASH_BASED_PYCS_VALUE: LazyLock<Value> =
    LazyLock::new(|| Value::string(intern("never")));
static MULTI_INTERP_EXTENSIONS_CHECK: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));

static IMPORT_LOCK: LazyLock<(Mutex<ImportLockState>, Condvar)> =
    LazyLock::new(|| (Mutex::new(ImportLockState::default()), Condvar::new()));

#[derive(Debug, Default)]
struct ImportLockState {
    owner: Option<ThreadId>,
    depth: usize,
}

/// Native `_imp` module descriptor.
#[derive(Debug, Clone)]
pub struct ImpModule {
    attrs: Vec<Arc<str>>,
}

impl ImpModule {
    /// Create a new `_imp` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("_fix_co_filename"),
                Arc::from("_override_multi_interp_extensions_check"),
                Arc::from("acquire_lock"),
                Arc::from("check_hash_based_pycs"),
                Arc::from("create_builtin"),
                Arc::from("create_dynamic"),
                Arc::from("exec_builtin"),
                Arc::from("exec_dynamic"),
                Arc::from("extension_suffixes"),
                Arc::from("find_frozen"),
                Arc::from("get_frozen_object"),
                Arc::from("is_builtin"),
                Arc::from("is_frozen"),
                Arc::from("is_frozen_package"),
                Arc::from("lock_held"),
                Arc::from("release_lock"),
                Arc::from("source_hash"),
            ],
        }
    }
}

impl Default for ImpModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ImpModule {
    fn name(&self) -> &str {
        "_imp"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "_fix_co_filename" => Ok(builtin_value(&FIX_CO_FILENAME_FUNCTION)),
            "_override_multi_interp_extensions_check" => Ok(builtin_value(
                &OVERRIDE_MULTI_INTERP_EXTENSIONS_CHECK_FUNCTION,
            )),
            "acquire_lock" => Ok(builtin_value(&ACQUIRE_LOCK_FUNCTION)),
            "check_hash_based_pycs" => Ok(*CHECK_HASH_BASED_PYCS_VALUE),
            "create_builtin" => Ok(builtin_value(&CREATE_BUILTIN_FUNCTION)),
            "create_dynamic" => Ok(builtin_value(&CREATE_DYNAMIC_FUNCTION)),
            "exec_builtin" => Ok(builtin_value(&EXEC_BUILTIN_FUNCTION)),
            "exec_dynamic" => Ok(builtin_value(&EXEC_DYNAMIC_FUNCTION)),
            "extension_suffixes" => Ok(builtin_value(&EXTENSION_SUFFIXES_FUNCTION)),
            "find_frozen" => Ok(builtin_value(&FIND_FROZEN_FUNCTION)),
            "get_frozen_object" => Ok(builtin_value(&GET_FROZEN_OBJECT_FUNCTION)),
            "is_builtin" => Ok(builtin_value(&IS_BUILTIN_FUNCTION)),
            "is_frozen" => Ok(builtin_value(&IS_FROZEN_FUNCTION)),
            "is_frozen_package" => Ok(builtin_value(&IS_FROZEN_PACKAGE_FUNCTION)),
            "lock_held" => Ok(builtin_value(&LOCK_HELD_FUNCTION)),
            "release_lock" => Ok(builtin_value(&RELEASE_LOCK_FUNCTION)),
            "source_hash" => Ok(builtin_value(&SOURCE_HASH_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_imp' has no attribute '{}'",
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
fn leak_object_value<T: prism_runtime::Trace>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn expect_no_args(args: &[Value], name: &str) -> Result<(), BuiltinError> {
    if args.is_empty() {
        return Ok(());
    }
    Err(BuiltinError::TypeError(format!(
        "{name}() takes 0 positional arguments but {} were given",
        args.len()
    )))
}

fn expect_arg_count(args: &[Value], expected: usize, name: &str) -> Result<(), BuiltinError> {
    if args.len() == expected {
        return Ok(());
    }
    Err(BuiltinError::TypeError(format!(
        "{name}() takes {expected} positional arguments but {} were given",
        args.len()
    )))
}

fn value_to_python_string(value: Value, context: &str) -> Result<Arc<str>, BuiltinError> {
    if let Some(ptr) = value.as_string_object_ptr() {
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a str")))?;
        return Ok(Arc::from(interned.as_ref()));
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!("{context} must be a str")));
    };

    if extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError(format!("{context} must be a str")));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(Arc::from(string.as_str()))
}

fn value_to_bytes(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like"
        )));
    };

    match extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            Ok(bytes.as_bytes().to_vec())
        }
        _ => Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like"
        ))),
    }
}

fn get_instance_property(value: Value, attr_name: &str) -> Option<Value> {
    let ptr = value.as_object_ptr()?;
    let type_id = extract_type_id(ptr);

    if type_id == TypeId::MODULE {
        let module = unsafe { &*(ptr as *const ModuleObject) };
        return module.get_attr(attr_name);
    }

    if type_id == TypeId::OBJECT || is_user_defined_type(type_id) {
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        return shaped.get_property(attr_name);
    }

    None
}

fn spec_name(spec: Value) -> Result<Arc<str>, BuiltinError> {
    let name = get_instance_property(spec, "name").ok_or_else(|| {
        BuiltinError::AttributeError("spec.name is required for _imp.create_builtin".to_string())
    })?;
    value_to_python_string(name, "spec.name")
}

fn module_value(module: &Arc<ModuleObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(module) as *const ())
}

fn builtin_module_value(vm: &mut VirtualMachine, name: &str) -> Result<Value, BuiltinError> {
    let module = vm.import_module_named(name).map_err(|err| {
        BuiltinError::ValueError(format!("failed to load built-in module '{name}': {err}"))
    })?;
    Ok(module_value(&module))
}

fn acquire_lock(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "_imp.acquire_lock")?;

    let current = std::thread::current().id();
    let (mutex, condvar) = &*IMPORT_LOCK;
    let mut state = mutex.lock().expect("import lock should not be poisoned");

    loop {
        match state.owner {
            None => {
                state.owner = Some(current);
                state.depth = 1;
                return Ok(Value::none());
            }
            Some(ref owner) if *owner == current => {
                state.depth += 1;
                return Ok(Value::none());
            }
            Some(_) => {
                state = condvar
                    .wait(state)
                    .expect("import lock wait should not be poisoned");
            }
        }
    }
}

fn release_lock(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "_imp.release_lock")?;

    let current = std::thread::current().id();
    let (mutex, condvar) = &*IMPORT_LOCK;
    let mut state = mutex.lock().expect("import lock should not be poisoned");

    match state.owner {
        Some(ref owner) if *owner == current => {
            state.depth -= 1;
            if state.depth == 0 {
                state.owner = None;
                condvar.notify_one();
            }
            Ok(Value::none())
        }
        _ => Err(BuiltinError::ValueError(
            "not holding the import lock".to_string(),
        )),
    }
}

fn lock_held(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "_imp.lock_held")?;

    let (mutex, _) = &*IMPORT_LOCK;
    let state = mutex.lock().expect("import lock should not be poisoned");
    Ok(Value::from(state.owner.is_some()))
}

fn is_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.is_builtin")?;
    let name = value_to_python_string(args[0], "module name")?;
    Ok(Value::int(if is_builtin_module_name(name.as_ref()) {
        1
    } else {
        0
    })
    .expect("small builtin flags should fit in i64"))
}

fn create_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.create_builtin")?;

    let name = spec_name(args[0])?;
    if !is_builtin_module_name(name.as_ref()) {
        return Ok(Value::none());
    }

    builtin_module_value(vm, name.as_ref())
}

fn exec_builtin(_vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.exec_builtin")?;
    Ok(Value::int(0).expect("zero should fit in i64"))
}

fn extension_suffixes(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "_imp.extension_suffixes")?;
    Ok(leak_object_value(ListObject::new()))
}

fn is_frozen(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.is_frozen")?;
    Ok(Value::from(false))
}

fn is_frozen_package(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.is_frozen_package")?;
    Ok(Value::from(false))
}

fn find_frozen(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.find_frozen")?;
    Ok(Value::none())
}

fn get_frozen_object(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.get_frozen_object")?;
    Err(BuiltinError::NotImplemented(
        "frozen modules are not available".to_string(),
    ))
}

fn fix_co_filename(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 2, "_imp._fix_co_filename")?;
    Ok(Value::none())
}

fn override_multi_interp_extensions_check(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp._override_multi_interp_extensions_check")?;

    let Some(override_value) = args[0]
        .as_bool()
        .or_else(|| args[0].as_int().map(|v| v != 0))
    else {
        return Err(BuiltinError::TypeError(
            "_override_multi_interp_extensions_check() argument must be bool or int".to_string(),
        ));
    };

    let mut state = MULTI_INTERP_EXTENSIONS_CHECK
        .lock()
        .expect("multi-interpreter flag should not be poisoned");
    let previous = *state;
    *state = override_value;
    Ok(Value::from(previous))
}

fn create_dynamic(_vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.create_dynamic")?;
    Err(BuiltinError::NotImplemented(
        "dynamic extension modules are not supported".to_string(),
    ))
}

fn exec_dynamic(_vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 1, "_imp.exec_dynamic")?;
    Err(BuiltinError::NotImplemented(
        "dynamic extension modules are not supported".to_string(),
    ))
}

fn source_hash(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, 2, "_imp.source_hash")?;

    let Some(key) = args[0].as_int() else {
        return Err(BuiltinError::TypeError(
            "_imp.source_hash() key must be an int".to_string(),
        ));
    };
    let source = value_to_bytes(args[1], "source")?;
    let hash = siphash13(key as u64, 0, &source);
    Ok(leak_object_value(BytesObject::from_slice(
        &hash.to_le_bytes(),
    )))
}

#[inline]
fn rotl64(value: u64, shift: u32) -> u64 {
    value.rotate_left(shift)
}

#[inline]
fn half_round(a: &mut u64, b: &mut u64, c: &mut u64, d: &mut u64, s: u32, t: u32) {
    *a = a.wrapping_add(*b);
    *c = c.wrapping_add(*d);
    *b = rotl64(*b, s) ^ *a;
    *d = rotl64(*d, t) ^ *c;
    *a = rotl64(*a, 32);
}

#[inline]
fn single_round(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    half_round(v0, v1, v2, v3, 13, 16);
    half_round(v2, v1, v0, v3, 17, 21);
}

fn siphash13(k0: u64, k1: u64, source: &[u8]) -> u64 {
    let mut v0 = k0 ^ 0x736f_6d65_7073_6575;
    let mut v1 = k1 ^ 0x646f_7261_6e64_6f6d;
    let mut v2 = k0 ^ 0x6c79_6765_6e65_7261;
    let mut v3 = k1 ^ 0x7465_6462_7974_6573;

    let mut chunks = source.chunks_exact(8);
    for chunk in &mut chunks {
        let mut word = [0_u8; 8];
        word.copy_from_slice(chunk);
        let message = u64::from_le_bytes(word);
        v3 ^= message;
        single_round(&mut v0, &mut v1, &mut v2, &mut v3);
        v0 ^= message;
    }

    let remainder = chunks.remainder();
    let mut tail = (source.len() as u64) << 56;
    for (index, byte) in remainder.iter().enumerate() {
        tail |= (*byte as u64) << (index * 8);
    }

    v3 ^= tail;
    single_round(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= tail;
    v2 ^= 0xff;
    single_round(&mut v0, &mut v1, &mut v2, &mut v3);
    single_round(&mut v0, &mut v1, &mut v2, &mut v3);
    single_round(&mut v0, &mut v1, &mut v2, &mut v3);

    (v0 ^ v1) ^ (v2 ^ v3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualMachine;
    use crate::import::ImportResolver;
    use prism_core::intern::intern;
    use prism_runtime::types::bytes::BytesObject;
    use prism_runtime::types::list::ListObject;

    fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
        let ptr = value
            .as_object_ptr()
            .expect("builtin function should be an object");
        unsafe { &*(ptr as *const BuiltinFunctionObject) }
    }

    #[test]
    fn test_imp_module_exposes_bootstrap_surface() {
        let module = ImpModule::new();

        assert!(module.get_attr("acquire_lock").is_ok());
        assert!(module.get_attr("create_builtin").is_ok());
        assert!(module.get_attr("exec_builtin").is_ok());
        assert!(module.get_attr("source_hash").is_ok());
        assert!(module.get_attr("check_hash_based_pycs").is_ok());
    }

    #[test]
    fn test_imp_is_builtin_reports_native_bootstrap_modules() {
        let builtin = builtin_from_value(
            ImpModule::new()
                .get_attr("is_builtin")
                .expect("is_builtin should exist"),
        );

        let thread_result = builtin
            .call(&[Value::string(intern("_thread"))])
            .expect("builtin lookup should succeed");
        assert_eq!(thread_result.as_int(), Some(1));

        let re_result = builtin
            .call(&[Value::string(intern("re"))])
            .expect("source-backed lookup should succeed");
        assert_eq!(re_result.as_int(), Some(0));
    }

    #[test]
    fn test_imp_extension_suffixes_returns_empty_list() {
        let builtin = builtin_from_value(
            ImpModule::new()
                .get_attr("extension_suffixes")
                .expect("extension_suffixes should exist"),
        );

        let result = builtin
            .call(&[])
            .expect("extension_suffixes should succeed");
        let ptr = result
            .as_object_ptr()
            .expect("extension_suffixes should return a list");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert!(list.is_empty());
    }

    #[test]
    fn test_imp_source_hash_returns_eight_bytes() {
        let builtin = builtin_from_value(
            ImpModule::new()
                .get_attr("source_hash")
                .expect("source_hash should exist"),
        );
        let source = leak_object_value(BytesObject::from_slice(b"print('hello')"));
        let result = builtin
            .call(&[Value::int(123).unwrap(), source])
            .expect("source_hash should succeed");
        let ptr = result
            .as_object_ptr()
            .expect("source_hash should return bytes");
        let bytes = unsafe { &*(ptr as *const BytesObject) };
        assert_eq!(bytes.as_bytes().len(), 8);
    }

    #[test]
    fn test_imp_create_builtin_loads_native_module_through_vm() {
        let mut vm = VirtualMachine::new();
        let builtin = builtin_from_value(
            ImpModule::new()
                .get_attr("create_builtin")
                .expect("create_builtin should exist"),
        );

        let spec = Box::into_raw(Box::new({
            let registry = prism_runtime::object::shape::shape_registry();
            let mut object = ShapedObject::with_empty_shape(registry.empty_shape());
            object.set_property(intern("name"), Value::string(intern("_thread")), registry);
            object
        }));

        let value = builtin
            .call_with_vm(&mut vm, &[Value::object_ptr(spec as *const ())])
            .expect("create_builtin should succeed");
        let module_ptr = value
            .as_object_ptr()
            .expect("create_builtin should return a module object");
        let module = unsafe { &*(module_ptr as *const ModuleObject) };
        assert_eq!(module.name(), "_thread");
    }

    #[test]
    fn test_import_resolver_can_load_imp_module() {
        let resolver = ImportResolver::new();
        let module = resolver
            .import_module("_imp")
            .expect("_imp should be importable");
        assert!(module.get_attr("create_builtin").is_some());
        assert_eq!(
            module
                .get_attr("check_hash_based_pycs")
                .and_then(|value| value.as_string_object_ptr())
                .and_then(|ptr| interned_by_ptr(ptr as *const u8))
                .map(|value| value.as_ref().to_string()),
            Some("never".to_string())
        );
    }
}
