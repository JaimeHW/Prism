use super::*;
use prism_compiler::Compiler;
use prism_core::intern::intern;
use prism_parser::parse;
use prism_runtime::types::list::ListObject;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

struct TestTempDir {
    path: PathBuf,
}

impl TestTempDir {
    fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "prism_builtin_import_tests_{}_{}_{}",
            std::process::id(),
            nanos,
            unique
        ));
        std::fs::create_dir_all(&path).expect("failed to create temp dir");
        Self { path }
    }
}

impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn write_file(path: &std::path::Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create parent dir");
    }
    std::fs::write(path, content).expect("failed to write test file");
}

fn string_list(items: &[&str]) -> Value {
    let list = ListObject::from_iter(
        items
            .iter()
            .copied()
            .map(|item| Value::string(intern(item))),
    );
    Value::object_ptr(Box::into_raw(Box::new(list)) as *const ())
}

fn cpython_lib_dir() -> PathBuf {
    let root = std::env::var_os("PRISM_CPYTHON_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\James\Desktop\cpython-3.12"));
    let lib_dir = root.join("Lib");
    assert!(
        lib_dir.is_dir(),
        "CPython Lib directory not found at {}. Set PRISM_CPYTHON_ROOT to override.",
        lib_dir.display()
    );
    lib_dir
}

fn execute_with_search_paths_and_step_limit(
    source: &str,
    search_paths: &[&Path],
    step_limit: u64,
) -> Result<Value, String> {
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;
    let code = Compiler::compile_module(&module, "<test>")
        .map_err(|e| format!("Compile error: {:?}", e))?;

    let mut vm = VirtualMachine::new();
    if let Some(verbosity) = std::env::var_os("PRISM_TEST_IMPORT_VERBOSITY")
        .and_then(|value| value.to_str().and_then(|s| s.parse::<u32>().ok()))
    {
        vm.set_import_verbosity(verbosity);
    }
    for path in search_paths {
        let path = Arc::<str>::from(path.to_string_lossy().into_owned());
        vm.import_resolver.add_search_path(path);
    }
    vm.set_execution_step_limit(Some(step_limit));
    vm.execute(Arc::new(code))
        .map_err(|e| format!("Runtime error: {:?}", e))
}

fn execute_with_cpython_lib_and_step_limit(source: &str, step_limit: u64) -> Result<Value, String> {
    let lib_dir = cpython_lib_dir();
    execute_with_search_paths_and_step_limit(source, &[lib_dir.as_path()], step_limit)
}

// =========================================================================
// dir() Argument Validation Tests
// =========================================================================

#[test]
fn test_dir_too_many_args() {
    let result = builtin_dir(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("at most 1 argument"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_dir_no_args_not_implemented() {
    // dir() without args requires frame introspection
    let result = builtin_dir(&[]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::NotImplemented(_)) => {}
        _ => panic!("Expected NotImplemented"),
    }
}

#[test]
fn test_dir_with_none() {
    // dir(None) should return NoneType attributes
    let result = builtin_dir(&[Value::none()]);
    // Currently returns placeholder
    assert!(result.is_ok() || matches!(result, Err(BuiltinError::NotImplemented(_))));
}

#[test]
fn test_dir_with_int() {
    let result = builtin_dir(&[Value::int(42).unwrap()]);
    assert!(result.is_ok());
}

#[test]
fn test_dir_with_float() {
    let result = builtin_dir(&[Value::float(3.14)]);
    assert!(result.is_ok());
}

#[test]
fn test_dir_with_bool() {
    let result = builtin_dir(&[Value::bool(true)]);
    assert!(result.is_ok());
}

#[test]
fn test_dir_vm_lists_unittest_module_and_heap_testcase_attrs() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest

class Smoke(unittest.TestCase):
    marker = 1

    def test_ok(self):
        pass

module_names = dir(unittest)
class_names = dir(Smoke)

assert "TestCase" in module_names
assert "TextTestRunner" in module_names
assert "marker" in class_names
assert "test_ok" in class_names
assert "assertTrue" in class_names
"#,
        160_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_abc_with_cpython_lib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import abc
"#,
        40_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_contextlib_with_cpython_lib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import contextlib
"#,
        60_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_traceback_with_cpython_lib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import traceback
"#,
        80_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_unittest_result_with_cpython_lib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest.result
"#,
        160_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_inspect_with_cpython_lib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import inspect
"#,
        120_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_dataclasses_with_cpython_lib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import dataclasses
"#,
        150_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =========================================================================
// vars() Argument Validation Tests
// =========================================================================

#[test]
fn test_vars_too_many_args() {
    let result = builtin_vars(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("at most 1 argument"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_vars_no_args_not_implemented() {
    let result = builtin_vars(&[]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::NotImplemented(_)) => {}
        _ => panic!("Expected NotImplemented"),
    }
}

#[test]
fn test_vars_with_none() {
    // vars(None) should be TypeError (no __dict__)
    let result = builtin_vars(&[Value::none()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("__dict__"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_vars_with_int() {
    // vars(42) should be TypeError (int has no __dict__)
    let result = builtin_vars(&[Value::int(42).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("__dict__"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_vars_with_float() {
    let result = builtin_vars(&[Value::float(3.14)]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(_)) => {}
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_vars_with_bool() {
    let result = builtin_vars(&[Value::bool(true)]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(_)) => {}
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_vars_vm_returns_live_module_dict() {
    let mut vm = VirtualMachine::new();
    let module = Arc::new(ModuleObject::new("vars_probe"));
    module.set_attr("answer", Value::int(42).unwrap());
    vm.import_resolver
        .insert_module("vars_probe", Arc::clone(&module));

    let dict_value =
        builtin_vars_vm(&mut vm, &[module_value(&module)]).expect("module vars should work");
    assert_eq!(
        dict_value.as_object_ptr(),
        module.dict_value().as_object_ptr()
    );

    let dict = crate::ops::objects::dict_storage_ref_from_ptr(
        dict_value
            .as_object_ptr()
            .expect("vars(module) should return a dictionary"),
    )
    .expect("vars(module) should return dict storage");
    assert_eq!(
        dict.get(Value::string(intern("answer")))
            .expect("module dict should expose attributes")
            .as_int(),
        Some(42)
    );
}

#[test]
fn test_vars_vm_rejects_objects_without_dict() {
    let mut vm = VirtualMachine::new();
    let result = builtin_vars_vm(&mut vm, &[Value::int(42).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("__dict__"));
        }
        _ => panic!("Expected TypeError"),
    }
}

// =========================================================================
// globals() Argument Validation Tests
// =========================================================================

#[test]
fn test_globals_with_args() {
    let result = builtin_globals(&[Value::int(1).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("takes no arguments"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_globals_not_implemented() {
    let result = builtin_globals(&[]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::NotImplemented(_)) => {}
        _ => panic!("Expected NotImplemented"),
    }
}

// =========================================================================
// locals() Argument Validation Tests
// =========================================================================

#[test]
fn test_locals_with_args() {
    let result = builtin_locals(&[Value::int(1).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("takes no arguments"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_locals_not_implemented() {
    let result = builtin_locals(&[]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::NotImplemented(_)) => {}
        _ => panic!("Expected NotImplemented"),
    }
}

// =========================================================================
// help() Argument Validation Tests
// =========================================================================

#[test]
fn test_help_too_many_args() {
    let result = builtin_help(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("at most 1 argument"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_help_not_implemented() {
    let result = builtin_help(&[]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::NotImplemented(_)) => {}
        _ => panic!("Expected NotImplemented"),
    }
}

#[test]
fn test_help_with_arg_not_implemented() {
    let result = builtin_help(&[Value::int(42).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::NotImplemented(_)) => {}
        _ => panic!("Expected NotImplemented"),
    }
}

// =========================================================================
// __import__() Argument Validation Tests
// =========================================================================

#[test]
fn test_import_no_args() {
    let result = builtin_import(&[]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("missing required argument"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_import_requires_vm_context_without_vm_dispatch() {
    let result = builtin_import(&[Value::int(1).unwrap()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("requires VM context"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_import_vm_imports_builtin_module() {
    let mut vm = VirtualMachine::new();
    let value = builtin_import_vm(&mut vm, &[Value::string(intern("builtins"))])
        .expect("__import__ should load builtins");
    let module_ptr = value
        .as_object_ptr()
        .expect("__import__ should return a module object");
    let module = unsafe { &*(module_ptr as *const ModuleObject) };

    assert_eq!(module.name(), "builtins");
    assert!(module.get_attr("open").is_some());
}

#[test]
fn test_import_vm_returns_top_level_module_for_dotted_import_without_fromlist() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg").join("__init__.py"), "");
    write_file(&temp.path.join("pkg").join("child.py"), "VALUE = 1\n");

    let mut vm = VirtualMachine::new();
    vm.import_resolver
        .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

    let value = builtin_import_vm(&mut vm, &[Value::string(intern("pkg.child"))])
        .expect("__import__ should load the dotted module");
    let module_ptr = value
        .as_object_ptr()
        .expect("__import__ should return a module object");
    let module = unsafe { &*(module_ptr as *const ModuleObject) };

    assert_eq!(module.name(), "pkg");
    let child = module
        .get_attr("child")
        .expect("top-level package should expose imported child module");
    let child_ptr = child
        .as_object_ptr()
        .expect("child attribute should be a module object");
    let child_module = unsafe { &*(child_ptr as *const ModuleObject) };
    assert_eq!(child_module.name(), "pkg.child");
}

#[test]
fn test_import_vm_returns_leaf_module_when_fromlist_is_present() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg").join("__init__.py"), "");
    write_file(&temp.path.join("pkg").join("child.py"), "VALUE = 1\n");

    let mut vm = VirtualMachine::new();
    vm.import_resolver
        .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

    let value = builtin_import_vm(
        &mut vm,
        &[
            Value::string(intern("pkg.child")),
            Value::none(),
            Value::none(),
            string_list(&["VALUE"]),
        ],
    )
    .expect("__import__ should return the leaf module when fromlist is set");
    let module_ptr = value
        .as_object_ptr()
        .expect("__import__ should return a module object");
    let module = unsafe { &*(module_ptr as *const ModuleObject) };

    assert_eq!(module.name(), "pkg.child");
    assert_eq!(
        module.get_attr("VALUE").and_then(|value| value.as_int()),
        Some(1)
    );
}

#[test]
fn test_import_vm_fromlist_loads_requested_package_submodule() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg").join("__init__.py"), "");
    write_file(&temp.path.join("pkg").join("child.py"), "VALUE = 1\n");

    let mut vm = VirtualMachine::new();
    vm.import_resolver
        .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

    let value = builtin_import_vm(
        &mut vm,
        &[
            Value::string(intern("pkg")),
            Value::none(),
            Value::none(),
            string_list(&["child"]),
        ],
    )
    .expect("__import__ should honor package fromlist requests");
    let module_ptr = value
        .as_object_ptr()
        .expect("__import__ should return a module object");
    let module = unsafe { &*(module_ptr as *const ModuleObject) };

    assert_eq!(module.name(), "pkg");
    let child = module
        .get_attr("child")
        .expect("fromlist should materialize requested submodule");
    let child_ptr = child
        .as_object_ptr()
        .expect("child attribute should be a module object");
    let child_module = unsafe { &*(child_ptr as *const ModuleObject) };
    assert_eq!(child_module.name(), "pkg.child");
}

// =========================================================================
// dir_of_value() Implementation Tests
// =========================================================================

#[test]
fn test_dir_of_value_none() {
    let result = dir_of_value(&Value::none());
    assert!(result.is_ok());
}

#[test]
fn test_dir_of_value_int() {
    let result = dir_of_value(&Value::int(42).unwrap());
    assert!(result.is_ok());
}

#[test]
fn test_dir_of_value_float() {
    let result = dir_of_value(&Value::float(3.14));
    assert!(result.is_ok());
}

#[test]
fn test_dir_of_value_bool() {
    let result = dir_of_value(&Value::bool(true));
    assert!(result.is_ok());
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_dir_preserves_type_info() {
    // Ensure dir() doesn't corrupt the value
    let val = Value::int(42).unwrap();
    let _ = builtin_dir(&[val.clone()]);
    assert!(val.is_int());
    assert_eq!(val.as_int(), Some(42));
}

#[test]
fn test_vars_preserves_type_info() {
    let val = Value::int(42).unwrap();
    let _ = builtin_vars(&[val.clone()]);
    assert!(val.is_int());
}
