//! Minimal native `_testcapi` compatibility hooks used by CPython regression tests.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, runtime_error_to_builtin_error};
use crate::import::ModuleObject;
use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_core::Value;
use prism_runtime::types::string::value_as_string_ref;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

static RUN_IN_SUBINTERP_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("_testcapi.run_in_subinterp"),
        testcapi_run_in_subinterp,
    )
});
static RUN_IN_SUBINTERP_WITH_CONFIG_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm_kw(
            Arc::from("_testcapi.run_in_subinterp_with_config"),
            testcapi_run_in_subinterp_with_config,
        )
    });

#[derive(Debug, Clone)]
pub struct TestCapiModule;

impl TestCapiModule {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TestCapiModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TestCapiModule {
    fn name(&self) -> &str {
        "_testcapi"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "run_in_subinterp" => Ok(builtin_value(&RUN_IN_SUBINTERP_FUNCTION)),
            "run_in_subinterp_with_config" => {
                Ok(builtin_value(&RUN_IN_SUBINTERP_WITH_CONFIG_FUNCTION))
            }
            _ => Err(ModuleError::AttributeError(format!(
                "module '_testcapi' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            Arc::from("run_in_subinterp"),
            Arc::from("run_in_subinterp_with_config"),
        ]
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn source_arg(value: Value, fn_name: &str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|text| text.as_str().to_string())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "{fn_name}() argument 1 must be str, not {}",
                value.type_name()
            ))
        })
}

fn execute_subinterpreter_source(
    vm: &mut VirtualMachine,
    source: String,
) -> Result<Value, BuiltinError> {
    let code = compile_source_code(&source, "<subinterpreter>", OptimizationLevel::None)
        .map_err(|err| BuiltinError::SyntaxError(err.to_string()))?;
    let module = Arc::new(ModuleObject::new("__subinterp__"));
    let search_paths = vm.import_resolver.search_paths();
    let mut sub_vm =
        VirtualMachine::with_shared_heap(vm.shared_heap(), vm.thread_interrupt_target());
    for path in search_paths {
        sub_vm.import_resolver.add_search_path(path);
    }

    let status = match sub_vm.execute_in_module_runtime(code, module) {
        Ok(_) => 0,
        Err(err) => {
            wait_for_subinterpreter_threads(&sub_vm);
            return Err(runtime_error_to_builtin_error(err));
        }
    };

    wait_for_subinterpreter_threads(&sub_vm);
    Ok(Value::int(status).expect("subinterpreter status should fit"))
}

fn wait_for_subinterpreter_threads(sub_vm: &VirtualMachine) {
    crate::threading_runtime::blocking_operation(|| {
        let _ = sub_vm.join_owned_threads(Duration::from_secs(5));
    });
}

fn testcapi_run_in_subinterp(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "run_in_subinterp() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    execute_subinterpreter_source(vm, source_arg(args[0], "run_in_subinterp")?)
}

fn testcapi_run_in_subinterp_with_config(
    vm: &mut VirtualMachine,
    args: &[Value],
    _keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "run_in_subinterp_with_config() takes exactly one positional argument ({} given)",
            args.len()
        )));
    }

    execute_subinterpreter_source(vm, source_arg(args[0], "run_in_subinterp_with_config")?)
}
