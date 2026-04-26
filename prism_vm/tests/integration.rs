//! End-to-end integration tests for Prism VM.
//!
//! These tests verify the complete parse → compile → execute pipeline.

use prism_compiler::Compiler;
use prism_core::Value;
use prism_core::intern::interned_by_ptr;
use prism_parser::parse;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use prism_vm::VirtualMachine;
use prism_vm::imports::ModuleObject;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

const RE_IGNORECASE: i64 = 2;
const RE_UNICODE: i64 = 32;

/// Helper to run Python source code and return result.
fn execute(source: &str) -> Result<Value, String> {
    execute_with_search_paths(source, &[])
}

/// Helper to run Python source code with additional import search paths.
fn execute_with_search_paths(source: &str, search_paths: &[&Path]) -> Result<Value, String> {
    execute_with_search_paths_and_step_limit(source, search_paths, None)
}

fn execute_with_search_paths_and_step_limit(
    source: &str,
    search_paths: &[&Path],
    step_limit: Option<u64>,
) -> Result<Value, String> {
    let code = prism_compiler::compile_source_code(
        source,
        "<test>",
        prism_compiler::OptimizationLevel::None,
    )
    .map_err(|e| format!("Compile error: {:?}", e))?;

    let mut vm = VirtualMachine::new();
    for path in search_paths {
        let path = Arc::<str>::from(path.to_string_lossy().into_owned());
        vm.add_import_search_path(path);
    }
    vm.set_execution_step_limit(step_limit);
    let main = Arc::new(ModuleObject::new("__main__"));
    let value = vm
        .execute_in_module_runtime(code, Arc::clone(&main))
        .map_err(|e| format!("Runtime error: {:?}", e))?;
    Ok(main.get_attr("RESULT").unwrap_or(value))
}

fn cpython_lib_dir() -> std::path::PathBuf {
    let root = std::env::var_os("PRISM_CPYTHON_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Users\James\Desktop\cpython-3.12"));
    let lib_dir = root.join("Lib");
    assert!(
        lib_dir.is_dir(),
        "CPython Lib directory not found at {}. Set PRISM_CPYTHON_ROOT to override.",
        lib_dir.display()
    );
    lib_dir
}

fn execute_with_cpython_lib(source: &str) -> Result<Value, String> {
    let lib_dir = cpython_lib_dir();
    execute_with_search_paths(source, &[lib_dir.as_path()])
}

fn execute_with_cpython_lib_and_step_limit(source: &str, step_limit: u64) -> Result<Value, String> {
    let lib_dir = cpython_lib_dir();
    execute_with_search_paths_and_step_limit(source, &[lib_dir.as_path()], Some(step_limit))
}

fn execute_in_main_module_with_search_paths(
    source: &str,
    search_paths: &[&Path],
) -> Result<(VirtualMachine, Arc<ModuleObject>), String> {
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;
    let code = Compiler::compile_module(&module, "<test>")
        .map_err(|e| format!("Compile error: {:?}", e))?;

    let mut vm = VirtualMachine::new();
    for path in search_paths {
        let path = Arc::<str>::from(path.to_string_lossy().into_owned());
        vm.add_import_search_path(path);
    }

    let main = Arc::new(ModuleObject::new("__main__"));
    vm.execute_in_module(Arc::new(code), Arc::clone(&main))
        .map_err(|e| format!("Runtime error: {:?}", e))?;
    Ok((vm, main))
}

fn value_is_python_string(value: Value, expected: &str) -> bool {
    if value.is_string() {
        let Some(ptr) = value.as_string_object_ptr() else {
            return false;
        };
        return interned_by_ptr(ptr as *const u8)
            .is_some_and(|interned| interned.as_str() == expected);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::STR {
        return false;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    string.as_str() == expected
}

fn python_string_value(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        return interned_by_ptr(ptr as *const u8).map(|interned| interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(string.as_str().to_string())
}

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("prism_{label}_{}_{}", std::process::id(), nonce))
}

#[path = "integration/doctest_io.rs"]
mod doctest_io;
#[path = "integration/language_core.rs"]
mod language_core;
#[path = "integration/primitives_and_protocols.rs"]
mod primitives_and_protocols;
#[path = "integration/runtime_threads.rs"]
mod runtime_threads;
#[path = "integration/stdlib_bootstrap.rs"]
mod stdlib_bootstrap;
#[path = "integration/stdlib_imports.rs"]
mod stdlib_imports;
#[path = "integration/type_metaclass.rs"]
mod type_metaclass;
