//! Execution builtins (exec, eval, compile).
//!
//! Functions for dynamic code execution and compilation.
//! All functions are Python 3.12 compatible.
//!
//! # Python Semantics
//!
//! - `exec(code, globals, locals)` - Execute code in given namespaces
//! - `eval(expression, globals, locals)` - Evaluate expression
//! - `compile(source, filename, mode)` - Compile source to code object
//!
//! # Security Note
//!
//! These functions execute arbitrary code. In production, they should
//! be used with caution and potentially sandboxed.

use super::BuiltinError;
use crate::VirtualMachine;
use crate::import::ModuleObject;
use crate::ops::objects::{dict_storage_mut_from_ptr, dict_storage_ref_from_ptr};
use prism_compiler::{Compiler, ModuleNamespaceMode, OptimizationLevel};
use prism_core::Value;
use prism_core::intern::intern;
use prism_parser::parse as parse_module_source;
use prism_runtime::object::class::ClassDict;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::value_as_string_ref;
use std::sync::Arc;

const DYNAMIC_EXEC_FILENAME: &str = "<string>";
const EVAL_RESULT_NAME: &str = "__prism_eval_result__";
const DYNAMIC_EXEC_RETURN_REG: u8 = 255;

// =============================================================================
// exec() - Execute Python Code
// =============================================================================

/// Builtin exec(code[, globals[, locals]]) function.
///
/// Execute the code in the optional globals and locals namespaces.
///
/// # Python Semantics
/// - `exec('x = 1')` → executes in current namespace
/// - `exec('x = 1', globals())` → executes in global namespace
/// - `exec('x = 1', globals(), locals())` → executes in local namespace
/// - Returns None
///
/// # Code Argument Types
/// - String: source code to compile and execute
/// - Code object: pre-compiled code to execute
///
/// # Implementation Note
/// Full implementation requires the compiler and frame system.
pub fn builtin_exec(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "exec() missing required argument: 'source'".to_string(),
        ));
    }
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "exec() takes at most 3 arguments ({} given)",
            args.len()
        )));
    }

    let source = &args[0];
    let _globals = args.get(1);
    let _locals = args.get(2);

    // Validate source type
    if source.is_none() {
        return Err(BuiltinError::TypeError(
            "exec() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }

    // For primitive types, we can't execute them
    if source.is_int() || source.is_float() || source.is_bool() {
        return Err(BuiltinError::TypeError(
            "exec() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }

    // TODO: Implement actual execution
    // 1. If source is a string, compile it with mode='exec'
    // 2. If source is a code object, use it directly
    // 3. Execute the code in the given namespaces
    Err(BuiltinError::NotImplemented(
        "exec() requires compiler integration".to_string(),
    ))
}

/// VM-aware exec implementation.
pub fn builtin_exec_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    validate_exec_args(args)?;
    let source = dynamic_source_text(
        args[0],
        "exec() arg 1 must be a string, bytes or code object",
    )?;
    let globals_arg = args.get(1).copied();
    let locals_arg = args.get(2).copied();
    let code = compile_dynamic_module(
        &source,
        DYNAMIC_EXEC_FILENAME,
        dynamic_module_namespace_mode(globals_arg, locals_arg),
    )?;
    let mut execution = execute_dynamic_module(vm, code, globals_arg, locals_arg)?;
    execution.write_back(None)?;
    Ok(Value::none())
}

// =============================================================================
// eval() - Evaluate Python Expression
// =============================================================================

/// Builtin eval(expression[, globals[, locals]]) function.
///
/// Evaluate a Python expression and return the result.
///
/// # Python Semantics
/// - `eval('1 + 2')` → 3
/// - `eval('x', {'x': 10})` → 10
/// - Returns the result of the expression
///
/// # Expression Argument Types
/// - String: source code to compile and evaluate
/// - Code object: pre-compiled code to evaluate
///
/// # Implementation Note
/// Full implementation requires the compiler and frame system.
pub fn builtin_eval(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "eval() missing required argument: 'source'".to_string(),
        ));
    }
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "eval() takes at most 3 arguments ({} given)",
            args.len()
        )));
    }

    let source = &args[0];
    let _globals = args.get(1);
    let _locals = args.get(2);

    // Validate source type
    if source.is_none() {
        return Err(BuiltinError::TypeError(
            "eval() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }

    if source.is_int() || source.is_float() || source.is_bool() {
        return Err(BuiltinError::TypeError(
            "eval() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }

    // TODO: Implement actual evaluation
    // 1. If source is a string, compile it with mode='eval'
    // 2. If source is a code object, use it directly
    // 3. Evaluate the code and return the result
    Err(BuiltinError::NotImplemented(
        "eval() requires compiler integration".to_string(),
    ))
}

/// VM-aware eval implementation.
pub fn builtin_eval_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    validate_eval_args(args)?;
    let source = dynamic_source_text(
        args[0],
        "eval() arg 1 must be a string, bytes or code object",
    )?;
    let wrapped = format!("{EVAL_RESULT_NAME} = ({source})\n");
    let globals_arg = args.get(1).copied();
    let locals_arg = args.get(2).copied();
    let code = compile_dynamic_module(
        &wrapped,
        DYNAMIC_EXEC_FILENAME,
        dynamic_module_namespace_mode(globals_arg, locals_arg),
    )?;
    let mut execution = execute_dynamic_module(vm, code, globals_arg, locals_arg)?;
    let result = execution
        .read_name(EVAL_RESULT_NAME)
        .ok_or_else(|| BuiltinError::SyntaxError("eval() did not produce a result".to_string()))?;
    execution.delete_name(EVAL_RESULT_NAME);
    execution.module.del_attr(EVAL_RESULT_NAME);
    execution.write_back(Some(EVAL_RESULT_NAME))?;
    Ok(result)
}

fn validate_exec_args(args: &[Value]) -> Result<(), BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "exec() missing required argument: 'source'".to_string(),
        ));
    }
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "exec() takes at most 3 arguments ({} given)",
            args.len()
        )));
    }
    if args[0].is_none() || args[0].is_int() || args[0].is_float() || args[0].is_bool() {
        return Err(BuiltinError::TypeError(
            "exec() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }
    Ok(())
}

fn validate_eval_args(args: &[Value]) -> Result<(), BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "eval() missing required argument: 'source'".to_string(),
        ));
    }
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "eval() takes at most 3 arguments ({} given)",
            args.len()
        )));
    }
    if args[0].is_none() || args[0].is_int() || args[0].is_float() || args[0].is_bool() {
        return Err(BuiltinError::TypeError(
            "eval() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }
    Ok(())
}

fn dynamic_source_text(value: Value, error_message: &str) -> Result<String, BuiltinError> {
    if let Some(string) = value_as_string_ref(value) {
        return Ok(string.as_str().to_string());
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(error_message.to_string()));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            std::str::from_utf8(bytes.as_bytes())
                .map(|text| text.to_string())
                .map_err(|_| {
                    BuiltinError::SyntaxError(
                        "source code string cannot contain undecodable bytes".to_string(),
                    )
                })
        }
        _ => Err(BuiltinError::TypeError(error_message.to_string())),
    }
}

fn compile_dynamic_module(
    source: &str,
    filename: &str,
    namespace_mode: ModuleNamespaceMode,
) -> Result<Arc<prism_code::CodeObject>, BuiltinError> {
    let parsed =
        parse_module_source(source).map_err(|err| BuiltinError::SyntaxError(err.to_string()))?;
    Compiler::compile_module_with_namespace_mode(
        &parsed,
        filename,
        OptimizationLevel::None,
        namespace_mode,
    )
    .map(Arc::new)
    .map_err(|err| BuiltinError::SyntaxError(err.to_string()))
}

fn dynamic_module_namespace_mode(
    globals_arg: Option<Value>,
    locals_arg: Option<Value>,
) -> ModuleNamespaceMode {
    if uses_dynamic_locals_mapping(locals_arg) {
        return ModuleNamespaceMode::DynamicLocals;
    }

    if locals_arg.is_none() && uses_dynamic_locals_mapping(globals_arg) {
        return ModuleNamespaceMode::DynamicLocals;
    }

    ModuleNamespaceMode::Standard
}

fn uses_dynamic_locals_mapping(arg: Option<Value>) -> bool {
    let Some(value) = arg.filter(|value| !value.is_none()) else {
        return false;
    };

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    dict_storage_ref_from_ptr(ptr).is_some()
}

enum NamespaceTarget {
    Dict(*mut DictObject),
    Module(Arc<ModuleObject>),
}

struct DynamicExecution {
    module: Arc<ModuleObject>,
    globals_target: Option<NamespaceTarget>,
    locals_target: Option<NamespaceTarget>,
}

impl DynamicExecution {
    fn read_name(&self, name: &str) -> Option<Value> {
        if let Some(target) = self.locals_target.as_ref() {
            if let Some(value) = target_get_name(target, name) {
                return Some(value);
            }
        } else if let Some(NamespaceTarget::Dict(_)) = self.globals_target.as_ref()
            && let Some(value) = target_get_name(
                self.globals_target
                    .as_ref()
                    .expect("dict target must exist"),
                name,
            )
        {
            return Some(value);
        }

        self.module.get_attr(name)
    }

    fn delete_name(&self, name: &str) {
        if let Some(target) = self.locals_target.as_ref() {
            target_delete_name(target, name);
            return;
        }

        if let Some(NamespaceTarget::Dict(_)) = self.globals_target.as_ref() {
            target_delete_name(
                self.globals_target
                    .as_ref()
                    .expect("dict target must exist"),
                name,
            );
        }
    }

    fn write_back(&mut self, skip_name: Option<&str>) -> Result<(), BuiltinError> {
        if let Some(target) = self.globals_target.as_ref() {
            sync_module_to_target(self.module.as_ref(), target, skip_name)?;
        }

        if let Some(target) = self.locals_target.as_ref() {
            if !matches!(target, NamespaceTarget::Dict(_)) {
                sync_module_to_target(self.module.as_ref(), target, skip_name)?;
            }
        }

        Ok(())
    }
}

fn execute_dynamic_module(
    vm: &mut VirtualMachine,
    code: Arc<prism_code::CodeObject>,
    globals_arg: Option<Value>,
    locals_arg: Option<Value>,
) -> Result<DynamicExecution, BuiltinError> {
    let globals_arg = globals_arg.filter(|value| !value.is_none());
    let locals_arg = locals_arg.filter(|value| !value.is_none());

    if globals_arg.is_none() && locals_arg.is_none() {
        let module = vm
            .current_module_cloned()
            .unwrap_or_else(|| Arc::new(ModuleObject::new("__main__")));
        if vm.call_depth() == 0 {
            vm.execute_in_module(code, Arc::clone(&module))
                .map_err(|err| BuiltinError::Raised(err.into()))?;
        } else {
            let stop_depth = vm.call_depth();
            let saved_return_reg = vm.current_frame().get_reg(DYNAMIC_EXEC_RETURN_REG);

            vm.push_frame_with_module(code, DYNAMIC_EXEC_RETURN_REG, Some(Arc::clone(&module)))
                .map_err(BuiltinError::Raised)?;

            let nested_result = vm
                .execute_until_stack_depth_restored(stop_depth, DYNAMIC_EXEC_RETURN_REG)
                .map_err(BuiltinError::Raised);

            if vm.call_depth() == stop_depth {
                vm.current_frame_mut()
                    .set_reg(DYNAMIC_EXEC_RETURN_REG, saved_return_reg);
            }

            nested_result?;
        }
        return Ok(DynamicExecution {
            module,
            globals_target: None,
            locals_target: None,
        });
    }

    let globals_target = globals_arg
        .map(|value| namespace_target_for_value(vm, value, "globals"))
        .transpose()?;
    let locals_target = locals_arg
        .map(|value| namespace_target_for_value(vm, value, "locals"))
        .transpose()?;

    let module = Arc::new(ModuleObject::new("__dynamic_exec__"));
    if let Some(target) = globals_target.as_ref() {
        seed_module_from_target(module.as_ref(), target)?;
    }

    let locals_mapping = locals_mapping_value(globals_target.as_ref(), locals_target.as_ref());
    let namespace = vm
        .execute_code_collect_locals_namespace_in_module(
            Arc::clone(&code),
            Arc::clone(&module),
            locals_mapping,
        )
        .map_err(|err| BuiltinError::Raised(err.into()))?;
    publish_namespace_to_module(module.as_ref(), &namespace.namespace);

    Ok(DynamicExecution {
        module,
        globals_target,
        locals_target,
    })
}

fn namespace_target_for_value(
    vm: &VirtualMachine,
    value: Value,
    context: &str,
) -> Result<NamespaceTarget, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be a dict or module"
        )));
    };

    if dict_storage_ref_from_ptr(ptr).is_some() {
        return Ok(NamespaceTarget::Dict(ptr as *mut DictObject));
    }

    if crate::ops::objects::extract_type_id(ptr) == TypeId::MODULE {
        if let Some(module) = vm.module_from_globals_ptr(ptr) {
            return Ok(NamespaceTarget::Module(module));
        }
    }

    Err(BuiltinError::TypeError(format!(
        "{context} must be a dict or module"
    )))
}

fn seed_module_from_target(
    module: &ModuleObject,
    target: &NamespaceTarget,
) -> Result<(), BuiltinError> {
    match target {
        NamespaceTarget::Dict(dict_ptr) => {
            let dict = unsafe { &**dict_ptr };
            for (key, value) in dict.iter() {
                let Some(name) = value_as_string_ref(key) else {
                    return Err(BuiltinError::TypeError(
                        "dynamic execution namespace keys must be strings".to_string(),
                    ));
                };
                module.set_attr(name.as_str(), value);
            }
            Ok(())
        }
        NamespaceTarget::Module(source) => {
            for (name, value) in source.all_attrs() {
                module.set_attr(name.as_ref(), value);
            }
            Ok(())
        }
    }
}

fn locals_mapping_value(
    globals_target: Option<&NamespaceTarget>,
    locals_target: Option<&NamespaceTarget>,
) -> Option<Value> {
    match locals_target {
        Some(NamespaceTarget::Dict(dict_ptr)) => Some(Value::object_ptr(*dict_ptr as *const ())),
        Some(NamespaceTarget::Module(_)) => None,
        None => match globals_target {
            Some(NamespaceTarget::Dict(dict_ptr)) => {
                Some(Value::object_ptr(*dict_ptr as *const ()))
            }
            Some(NamespaceTarget::Module(_)) | None => None,
        },
    }
}

fn publish_namespace_to_module(module: &ModuleObject, namespace: &ClassDict) {
    namespace.for_each(|name, value| {
        module.set_attr(name.as_ref(), value);
    });
}

fn target_get_name(target: &NamespaceTarget, name: &str) -> Option<Value> {
    match target {
        NamespaceTarget::Dict(dict_ptr) => {
            let dict = unsafe { &**dict_ptr };
            dict.get(Value::string(intern(name)))
        }
        NamespaceTarget::Module(module) => module.get_attr(name),
    }
}

fn target_delete_name(target: &NamespaceTarget, name: &str) {
    match target {
        NamespaceTarget::Dict(dict_ptr) => {
            let dict = unsafe { &mut **dict_ptr };
            let _ = dict.remove(Value::string(intern(name)));
        }
        NamespaceTarget::Module(module) => {
            module.del_attr(name);
        }
    }
}

fn sync_module_to_target(
    module: &ModuleObject,
    target: &NamespaceTarget,
    skip_name: Option<&str>,
) -> Result<(), BuiltinError> {
    match target {
        NamespaceTarget::Dict(dict_ptr) => {
            let dict = dict_storage_mut_from_ptr(*dict_ptr as *const ()).ok_or_else(|| {
                BuiltinError::TypeError(
                    "dynamic execution namespace is no longer mutable".to_string(),
                )
            })?;
            for (name, value) in module.all_attrs() {
                if skip_name.is_some_and(|skip| name.as_ref() == skip) {
                    continue;
                }
                dict.set(Value::string(intern(name.as_ref())), value);
            }
            Ok(())
        }
        NamespaceTarget::Module(target_module) => {
            for (name, value) in module.all_attrs() {
                if skip_name.is_some_and(|skip| name.as_ref() == skip) {
                    continue;
                }
                target_module.set_attr(name.as_ref(), value);
            }
            Ok(())
        }
    }
}

// =============================================================================
// compile() - Compile Source to Code Object
// =============================================================================

/// Compile mode for Python code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    /// Compile as a module (sequence of statements)
    Exec,
    /// Compile as a single expression
    Eval,
    /// Compile as a single interactive statement
    Single,
}

impl CompileMode {
    /// Parse mode from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "exec" => Some(CompileMode::Exec),
            "eval" => Some(CompileMode::Eval),
            "single" => Some(CompileMode::Single),
            _ => None,
        }
    }
}

/// Builtin compile(source, filename, mode, ...) function.
///
/// Compile source code into a code object.
///
/// # Python Semantics
/// - `compile('x+1', '<string>', 'eval')` → code object
/// - `compile('x=1', '<string>', 'exec')` → code object
/// - `compile('x=1', '<string>', 'single')` → code object
///
/// # Arguments
/// - source: String, bytes, or AST object
/// - filename: Name for error messages
/// - mode: 'exec', 'eval', or 'single'
/// - flags: Optional compiler flags (default 0)
/// - dont_inherit: Don't inherit future flags (default False)
/// - optimize: Optimization level (default -1)
///
/// # Implementation Note
/// Full implementation requires the compiler.
pub fn builtin_compile(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 3 {
        return Err(BuiltinError::TypeError(format!(
            "compile() takes at least 3 arguments ({} given)",
            args.len()
        )));
    }
    if args.len() > 6 {
        return Err(BuiltinError::TypeError(format!(
            "compile() takes at most 6 arguments ({} given)",
            args.len()
        )));
    }

    let source = &args[0];
    let _filename = &args[1];
    let _mode = &args[2];

    // Validate source type
    if source.is_none() {
        return Err(BuiltinError::TypeError(
            "compile() source must be a string, bytes, or AST object".to_string(),
        ));
    }

    if source.is_int() || source.is_float() || source.is_bool() {
        return Err(BuiltinError::TypeError(
            "compile() source must be a string, bytes, or AST object".to_string(),
        ));
    }

    // TODO: Validate filename is a string
    // TODO: Validate mode is 'exec', 'eval', or 'single'
    // TODO: Validate optional arguments

    // TODO: Implement actual compilation
    // 1. Parse the source code
    // 2. Compile to bytecode
    // 3. Return a code object
    Err(BuiltinError::NotImplemented(
        "compile() requires compiler integration".to_string(),
    ))
}

// =============================================================================
// breakpoint() - Invoke Debugger
// =============================================================================

/// Builtin breakpoint(*args, **kws) function.
///
/// Calls sys.breakpointhook() to enter the debugger.
///
/// # Python Semantics
/// - `breakpoint()` → enters pdb debugger
/// - Can be customized via PYTHONBREAKPOINT env var
///
/// # Implementation Note
/// This is a stub for debugger integration.
pub fn builtin_breakpoint(args: &[Value]) -> Result<Value, BuiltinError> {
    // breakpoint() is typically a no-op if no debugger is attached
    // For now, we just ignore any arguments and return None
    let _ = args;

    // In a full implementation:
    // 1. Check PYTHONBREAKPOINT environment variable
    // 2. If set to '0', do nothing
    // 3. Otherwise, call the configured hook (default: pdb.set_trace)

    Ok(Value::none())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn leaked_dict_value() -> (*mut DictObject, Value) {
        let dict_ptr = Box::into_raw(Box::new(DictObject::new()));
        (dict_ptr, Value::object_ptr(dict_ptr as *const ()))
    }

    fn dict_get(dict_ptr: *mut DictObject, name: &str) -> Option<Value> {
        let dict = unsafe { &*dict_ptr };
        dict.get(Value::string(intern(name)))
    }

    fn dict_set(dict_ptr: *mut DictObject, name: &str, value: Value) {
        let dict = unsafe { &mut *dict_ptr };
        dict.set(Value::string(intern(name)), value);
    }

    // =========================================================================
    // exec() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_exec_no_args() {
        let result = builtin_exec(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("missing required argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_too_many_args() {
        let result = builtin_exec(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 3 arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_with_none() {
        let result = builtin_exec(&[Value::none()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("string, bytes or code object"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_with_int() {
        let result = builtin_exec(&[Value::int(42).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("string, bytes or code object"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_with_float() {
        let result = builtin_exec(&[Value::float(3.14)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_with_bool() {
        let result = builtin_exec(&[Value::bool(true)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_vm_writes_assignments_into_explicit_globals_dict() {
        let mut vm = VirtualMachine::new();
        let (globals_ptr, globals) = leaked_dict_value();

        builtin_exec_vm(&mut vm, &[Value::string(intern("x = 1\n")), globals])
            .expect("exec should succeed");

        assert_eq!(
            dict_get(globals_ptr, "x").and_then(|value| value.as_int()),
            Some(1)
        );
    }

    #[test]
    fn test_exec_vm_publishes_generated_functions_into_namespace_dict() {
        let mut vm = VirtualMachine::new();
        let (globals_ptr, globals) = leaked_dict_value();
        let source = Value::string(intern(
            "def __create_fn__():\n    def generated(self):\n        return self.x\n    return generated\n",
        ));

        builtin_exec_vm(&mut vm, &[source, globals]).expect("exec should succeed");

        assert!(
            dict_get(globals_ptr, "__create_fn__").is_some(),
            "exec() must publish generated helper factories into the provided namespace",
        );
    }

    #[test]
    fn test_exec_vm_respects_distinct_globals_and_locals_dicts() {
        let mut vm = VirtualMachine::new();
        let (globals_ptr, globals) = leaked_dict_value();
        let (locals_ptr, locals) = leaked_dict_value();
        let source = Value::string(intern("global y\nx = 1\ny = 2\n"));

        builtin_exec_vm(&mut vm, &[source, globals, locals]).expect("exec should succeed");

        assert_eq!(
            dict_get(locals_ptr, "x").and_then(|value| value.as_int()),
            Some(1)
        );
        assert_eq!(
            dict_get(globals_ptr, "y").and_then(|value| value.as_int()),
            Some(2)
        );
        assert!(
            dict_get(locals_ptr, "y").is_none(),
            "global assignments must not be reflected back into the local mapping",
        );
    }

    // =========================================================================
    // eval() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_eval_no_args() {
        let result = builtin_eval(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("missing required argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_too_many_args() {
        let result = builtin_eval(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 3 arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_with_none() {
        let result = builtin_eval(&[Value::none()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("string, bytes or code object"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_with_int() {
        let result = builtin_eval(&[Value::int(42).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_with_float() {
        let result = builtin_eval(&[Value::float(3.14)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_with_bool() {
        let result = builtin_eval(&[Value::bool(true)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_vm_reads_from_globals_dict_and_cleans_internal_result_name() {
        let mut vm = VirtualMachine::new();
        let (globals_ptr, globals) = leaked_dict_value();
        dict_set(globals_ptr, "x", Value::int(40).unwrap());

        let result = builtin_eval_vm(&mut vm, &[Value::string(intern("x + 2")), globals])
            .expect("eval should succeed");

        assert_eq!(result.as_int(), Some(42));
        assert!(
            dict_get(globals_ptr, EVAL_RESULT_NAME).is_none(),
            "eval() must remove its internal result binding from the provided namespace",
        );
    }

    #[test]
    fn test_eval_vm_reads_from_explicit_locals_dict() {
        let mut vm = VirtualMachine::new();
        let (_globals_ptr, globals) = leaked_dict_value();
        let (locals_ptr, locals) = leaked_dict_value();
        dict_set(locals_ptr, "x", Value::int(40).unwrap());

        let result = builtin_eval_vm(&mut vm, &[Value::string(intern("x + 2")), globals, locals])
            .expect("eval should resolve names from the explicit locals mapping");

        assert_eq!(result.as_int(), Some(42));
        assert!(
            dict_get(locals_ptr, EVAL_RESULT_NAME).is_none(),
            "eval() must not leak its internal result binding into the locals mapping",
        );
    }

    #[test]
    fn test_eval_vm_propagates_missing_name_errors_without_panicking() {
        let mut vm = VirtualMachine::new();

        let result = builtin_eval_vm(&mut vm, &[Value::string(intern("missing_name"))]);

        match result {
            Err(BuiltinError::Raised(err)) => {
                let is_name_error =
                    matches!(err.kind, crate::error::RuntimeErrorKind::NameError { .. })
                        || matches!(
                            err.kind,
                            crate::error::RuntimeErrorKind::Exception { type_id, .. }
                                if type_id
                                    == crate::stdlib::exceptions::ExceptionTypeId::NameError.as_u8()
                                        as u16
                        );

                assert!(
                    is_name_error,
                    "expected missing eval names to surface as NameError, got {err:?}",
                );
            }
            other => panic!("expected eval() to raise NameError, got {other:?}"),
        }
    }

    // =========================================================================
    // compile() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_compile_too_few_args() {
        let result = builtin_compile(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at least 3 arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_compile_too_many_args() {
        let result = builtin_compile(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
            Value::int(5).unwrap(),
            Value::int(6).unwrap(),
            Value::int(7).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 6 arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_compile_with_none_source() {
        let result = builtin_compile(&[
            Value::none(),
            Value::int(0).unwrap(),
            Value::int(0).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("string, bytes, or AST"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_compile_with_int_source() {
        let result = builtin_compile(&[
            Value::int(42).unwrap(),
            Value::int(0).unwrap(),
            Value::int(0).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    // =========================================================================
    // breakpoint() Tests
    // =========================================================================

    #[test]
    fn test_breakpoint_no_args() {
        let result = builtin_breakpoint(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_breakpoint_with_args() {
        // breakpoint accepts arbitrary args and ignores them
        let result = builtin_breakpoint(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_ok());
    }

    // =========================================================================
    // CompileMode Tests
    // =========================================================================

    #[test]
    fn test_compile_mode_from_str() {
        assert_eq!(CompileMode::from_str("exec"), Some(CompileMode::Exec));
        assert_eq!(CompileMode::from_str("eval"), Some(CompileMode::Eval));
        assert_eq!(CompileMode::from_str("single"), Some(CompileMode::Single));
        assert_eq!(CompileMode::from_str("invalid"), None);
        assert_eq!(CompileMode::from_str(""), None);
    }

    #[test]
    fn test_compile_mode_equality() {
        assert_eq!(CompileMode::Exec, CompileMode::Exec);
        assert_ne!(CompileMode::Exec, CompileMode::Eval);
        assert_ne!(CompileMode::Eval, CompileMode::Single);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_exec_preserves_input() {
        let val = Value::int(42).unwrap();
        let _ = builtin_exec(&[val.clone()]);
        assert!(val.is_int());
        assert_eq!(val.as_int(), Some(42));
    }

    #[test]
    fn test_eval_preserves_input() {
        let val = Value::float(3.14);
        let _ = builtin_eval(&[val.clone()]);
        assert!(val.is_float());
    }

    #[test]
    fn test_compile_preserves_input() {
        let val = Value::bool(true);
        let _ = builtin_compile(&[val.clone(), Value::int(0).unwrap(), Value::int(0).unwrap()]);
        assert!(val.is_bool());
    }
}
