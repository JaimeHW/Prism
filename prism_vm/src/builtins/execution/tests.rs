use super::*;
use prism_code::CodeObject;
use prism_runtime::object::ObjectHeader;
use prism_runtime::types::TupleObject;
use std::sync::Arc;

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

fn code_view_from_value(value: Value) -> &'static CodeObjectView {
    let ptr = value.as_object_ptr().expect("expected code object view");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::CODE);
    unsafe { &*(ptr as *const CodeObjectView) }
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
fn test_eval_vm_defaults_to_current_frame_locals_when_nested() {
    let mut vm = VirtualMachine::new();
    let mut code = CodeObject::new("outer", "<test>");
    code.register_count = 4;
    code.locals = vec![Arc::from("args")].into_boxed_slice();
    vm.push_frame(Arc::new(code), 0)
        .expect("frame push should succeed");

    let args_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(7).unwrap(),
        Value::int(42).unwrap(),
    ])));
    let args_value = Value::object_ptr(args_ptr as *const ());
    vm.current_frame_mut().set_reg(0, args_value);

    let result = builtin_eval_vm(&mut vm, &[Value::string(intern("args[1]"))])
        .expect("nested eval should see the caller's locals");

    assert_eq!(result.as_int(), Some(42));
    unsafe {
        drop(Box::from_raw(args_ptr));
    }
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

#[test]
fn test_compile_returns_code_object_for_eval_mode() {
    let code = builtin_compile(&[
        Value::string(intern("40 + 2")),
        Value::string(intern("<test>")),
        Value::string(intern("eval")),
    ])
    .expect("compile should succeed");

    let view = code_view_from_value(code);
    assert!(code_contains_internal_eval_result(view.code().as_ref()));
}

#[test]
fn test_compile_accepts_future_flags_argument() {
    let code = builtin_compile(&[
        Value::string(intern("pass")),
        Value::string(intern("<test>")),
        Value::string(intern("exec")),
        Value::int(0x20_000).unwrap(),
    ])
    .expect("compile should accept future flags");

    let ptr = code
        .as_object_ptr()
        .expect("compile should return a code object");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::CODE);
}

#[test]
fn test_compile_accepts_cpython_keyword_arguments() {
    let code = builtin_compile_kw(
        &[Value::string(intern("pass"))],
        &[
            ("filename", Value::string(intern("<keyword>"))),
            ("mode", Value::string(intern("exec"))),
            ("dont_inherit", Value::bool(true)),
            ("optimize", Value::int(-1).unwrap()),
            ("_feature_version", Value::int(-1).unwrap()),
        ],
    )
    .expect("compile should accept CPython keyword arguments");

    let ptr = code
        .as_object_ptr()
        .expect("compile should return a code object");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::CODE);
}

#[test]
fn test_compile_keyword_validation_rejects_duplicates_and_unknown_names() {
    let duplicate = builtin_compile_kw(
        &[
            Value::string(intern("pass")),
            Value::string(intern("<test>")),
            Value::string(intern("exec")),
        ],
        &[("mode", Value::string(intern("eval")))],
    )
    .expect_err("duplicate mode keyword should fail");
    assert!(
        duplicate
            .to_string()
            .contains("compile() got multiple values for argument 'mode'")
    );

    let unexpected = builtin_compile_kw(
        &[
            Value::string(intern("pass")),
            Value::string(intern("<test>")),
            Value::string(intern("exec")),
        ],
        &[("bogus", Value::none())],
    )
    .expect_err("unexpected keyword should fail");
    assert!(
        unexpected
            .to_string()
            .contains("compile() got an unexpected keyword argument 'bogus'")
    );
}

#[test]
fn test_compile_syntax_error_preserves_filename_metadata() {
    let err = builtin_compile(&[
        Value::string(intern("if True\n    pass\n")),
        Value::string(intern("demo.py")),
        Value::string(intern("exec")),
    ])
    .expect_err("compile should raise SyntaxError metadata");

    let BuiltinError::Raised(runtime) = err else {
        panic!("expected raised SyntaxError, got {err:?}");
    };
    let raised = runtime
        .raised_value
        .expect("raised SyntaxError should preserve exception object");
    let exc = unsafe {
        crate::builtins::ExceptionValue::from_value(raised)
            .expect("raised value should be an exception")
    };
    assert_eq!(
        exc.type_id(),
        crate::stdlib::exceptions::ExceptionTypeId::SyntaxError
    );
    assert_eq!(exc.syntax_filename(), Some("demo.py"));
    assert_eq!(exc.syntax_lineno(), Some(1));
    assert!(exc.syntax_offset().is_some());
    assert_eq!(exc.syntax_text(), Some("if True\n"));
}

#[test]
fn test_eval_vm_accepts_compiled_code_object() {
    let mut vm = VirtualMachine::new();
    let (globals_ptr, globals) = leaked_dict_value();
    dict_set(globals_ptr, "x", Value::int(40).unwrap());

    let code = builtin_compile(&[
        Value::string(intern("x + 2")),
        Value::string(intern("<test>")),
        Value::string(intern("eval")),
    ])
    .expect("compile should succeed");

    let result = builtin_eval_vm(&mut vm, &[code, globals]).expect("eval should succeed");
    assert_eq!(result.as_int(), Some(42));
    assert!(dict_get(globals_ptr, EVAL_RESULT_NAME).is_none());
}

#[test]
fn test_eval_vm_returns_none_for_exec_mode_code_object() {
    let mut vm = VirtualMachine::new();
    let code = builtin_compile(&[
        Value::string(intern("x = 1")),
        Value::string(intern("<test>")),
        Value::string(intern("exec")),
    ])
    .expect("compile should succeed");

    let result = builtin_eval_vm(&mut vm, &[code]).expect("eval should accept code object");
    assert!(result.is_none());
}

#[test]
fn test_exec_vm_cleans_internal_binding_from_compiled_eval_code_object() {
    let mut vm = VirtualMachine::new();
    let (globals_ptr, globals) = leaked_dict_value();

    let code = builtin_compile(&[
        Value::string(intern("40 + 2")),
        Value::string(intern("<test>")),
        Value::string(intern("eval")),
    ])
    .expect("compile should succeed");

    builtin_exec_vm(&mut vm, &[code, globals]).expect("exec should accept code object");
    assert!(dict_get(globals_ptr, EVAL_RESULT_NAME).is_none());
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
