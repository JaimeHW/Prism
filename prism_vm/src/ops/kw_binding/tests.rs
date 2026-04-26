use super::*;
use prism_code::{CodeFlags, CodeObject};

fn make_test_code(
    arg_count: u16,
    kwonly_count: u16,
    locals: Vec<&str>,
    flags: CodeFlags,
) -> Arc<CodeObject> {
    let mut code = CodeObject::new("test_func", "test.py");
    code.arg_count = arg_count;
    code.kwonlyarg_count = kwonly_count;
    code.locals = locals.into_iter().map(|s| Arc::from(s)).collect();
    code.flags = flags;
    Arc::new(code)
}

fn make_test_func(
    code: Arc<CodeObject>,
    defaults: Option<Vec<Value>>,
    kwdefaults: Option<Vec<(Arc<str>, Value)>>,
) -> FunctionObject {
    let mut func = FunctionObject::new(
        code,
        "test_func".into(),
        defaults.map(|v| v.into_boxed_slice()),
        None,
    );
    func.kwdefaults = kwdefaults.map(|v| v.into_boxed_slice());
    func
}

// =========================================================================
// Basic Positional Argument Tests
// =========================================================================

#[test]
fn test_bind_exact_positional_args() {
    let code = make_test_code(2, 0, vec!["a", "b"], CodeFlags::NONE);
    let func = make_test_func(code, None, None);

    let args = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert_eq!(bound.parameters.len(), 2);
    assert_eq!(bound.parameters[0].as_int(), Some(1));
    assert_eq!(bound.parameters[1].as_int(), Some(2));
    assert!(bound.varargs.is_none());
    assert!(bound.varkw.is_none());
}

#[test]
fn test_bind_too_many_positional_args_error() {
    let code = make_test_code(2, 0, vec!["a", "b"], CodeFlags::NONE);
    let func = make_test_func(code, None, None);

    let args = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_err());
    match result.unwrap_err() {
        BindingError::TooManyPositional {
            expected, given, ..
        } => {
            assert_eq!(expected, 2);
            assert_eq!(given, 3);
        }
        _ => panic!("Expected TooManyPositional error"),
    }
}

// =========================================================================
// Varargs (*args) Tests
// =========================================================================

#[test]
fn test_bind_with_varargs_empty() {
    let code = make_test_code(2, 0, vec!["a", "b", "args"], CodeFlags::VARARGS);
    let func = make_test_func(code, None, None);

    let args = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert_eq!(bound.parameters.len(), 2);
    assert!(bound.varargs.is_some());
    assert_eq!(bound.varargs.as_ref().unwrap().len(), 0);
}

#[test]
fn test_bind_with_varargs_populated() {
    let code = make_test_code(2, 0, vec!["a", "b", "args"], CodeFlags::VARARGS);
    let func = make_test_func(code, None, None);

    let args = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
        Value::int(5).unwrap(),
    ];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert_eq!(bound.parameters[0].as_int(), Some(1));
    assert_eq!(bound.parameters[1].as_int(), Some(2));

    let varargs = bound.varargs.as_ref().unwrap();
    assert_eq!(varargs.len(), 3);
    assert_eq!(varargs.get(0).unwrap().as_int(), Some(3));
    assert_eq!(varargs.get(1).unwrap().as_int(), Some(4));
    assert_eq!(varargs.get(2).unwrap().as_int(), Some(5));
}

// =========================================================================
// Keyword Argument Tests
// =========================================================================

#[test]
fn test_bind_keyword_args_only() {
    let code = make_test_code(2, 0, vec!["a", "b"], CodeFlags::NONE);
    let func = make_test_func(code, None, None);

    let kwargs = vec![
        ("b", Value::int(20).unwrap()),
        ("a", Value::int(10).unwrap()),
    ];
    let result = ArgumentBinder::bind(&func, std::iter::empty(), kwargs.into_iter());

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert_eq!(bound.parameters[0].as_int(), Some(10)); // a
    assert_eq!(bound.parameters[1].as_int(), Some(20)); // b
}

#[test]
fn test_bind_mixed_positional_and_keyword() {
    // def f(a, b=2, c=3): pass
    // f(1, c=3) -> a=1, b=2 (default), c=3 (keyword)
    let code = make_test_code(3, 0, vec!["a", "b", "c"], CodeFlags::NONE);
    // Defaults for b and c (last 2 params)
    let func = make_test_func(
        code,
        Some(vec![Value::int(2).unwrap(), Value::int(30).unwrap()]),
        None,
    );

    let args = vec![Value::int(1).unwrap()];
    let kwargs = vec![("c", Value::int(3).unwrap())]; // Override c's default

    let result = ArgumentBinder::bind(&func, args.into_iter(), kwargs.into_iter());

    assert!(result.is_ok(), "Binding failed: {:?}", result);
    let bound = result.unwrap();
    assert_eq!(bound.parameters[0].as_int(), Some(1)); // a (positional)
    assert_eq!(bound.parameters[1].as_int(), Some(2)); // b (default)
    assert_eq!(bound.parameters[2].as_int(), Some(3)); // c (keyword override)
}

#[test]
fn test_bind_duplicate_argument_error() {
    let code = make_test_code(2, 0, vec!["a", "b"], CodeFlags::NONE);
    let func = make_test_func(code, None, None);

    let args = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let kwargs = vec![("a", Value::int(10).unwrap())];

    let result = ArgumentBinder::bind(&func, args.into_iter(), kwargs.into_iter());

    assert!(result.is_err());
    match result.unwrap_err() {
        BindingError::DuplicateArgument { param_name, .. } => {
            assert_eq!(param_name.as_ref(), "a");
        }
        _ => panic!("Expected DuplicateArgument error"),
    }
}

#[test]
fn test_bind_unexpected_keyword_error() {
    let code = make_test_code(2, 0, vec!["a", "b"], CodeFlags::NONE);
    let func = make_test_func(code, None, None);

    let args = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let kwargs = vec![("unknown", Value::int(99).unwrap())];

    let result = ArgumentBinder::bind(&func, args.into_iter(), kwargs.into_iter());

    assert!(result.is_err());
    match result.unwrap_err() {
        BindingError::UnexpectedKeyword { keyword, .. } => {
            assert_eq!(keyword.as_ref(), "unknown");
        }
        _ => panic!("Expected UnexpectedKeyword error"),
    }
}

// =========================================================================
// Varkw (**kwargs) Tests
// =========================================================================

#[test]
fn test_bind_with_varkw_empty() {
    let code = make_test_code(2, 0, vec!["a", "b", "kwargs"], CodeFlags::VARKEYWORDS);
    let func = make_test_func(code, None, None);

    let args = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert!(bound.varkw.is_some());
    assert_eq!(bound.varkw.as_ref().unwrap().len(), 0);
}

#[test]
fn test_bind_with_varkw_populated() {
    let code = make_test_code(1, 0, vec!["a", "kwargs"], CodeFlags::VARKEYWORDS);
    let func = make_test_func(code, None, None);

    let args = vec![Value::int(1).unwrap()];
    let kwargs = vec![
        ("extra1", Value::int(100).unwrap()),
        ("extra2", Value::int(200).unwrap()),
    ];

    let result = ArgumentBinder::bind(&func, args.into_iter(), kwargs.into_iter());

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert_eq!(bound.parameters[0].as_int(), Some(1));

    let varkw = bound.varkw.as_ref().unwrap();
    assert_eq!(varkw.len(), 2);
    assert_eq!(
        varkw
            .get(Value::string(intern("extra1")))
            .and_then(|value| value.as_int()),
        Some(100)
    );
    assert_eq!(
        varkw
            .get(Value::string(intern("extra2")))
            .and_then(|value| value.as_int()),
        Some(200)
    );
}

// =========================================================================
// Keyword-Only Parameter Tests
// =========================================================================

#[test]
fn test_bind_keyword_only_args() {
    // def f(a, *args, kwonly1, kwonly2): pass
    // locals = [a, args, kwonly1, kwonly2] with VARARGS flag
    let code = make_test_code(
        1,                                       // arg_count = 1 positional
        2,                                       // kwonlyarg_count = 2 kwonly
        vec!["a", "args", "kwonly1", "kwonly2"], // include *args slot
        CodeFlags::VARARGS,
    );
    let func = make_test_func(code, None, None);

    let args = vec![Value::int(1).unwrap()];
    let kwargs = vec![
        ("kwonly1", Value::int(10).unwrap()),
        ("kwonly2", Value::int(20).unwrap()),
    ];

    let result = ArgumentBinder::bind(&func, args.into_iter(), kwargs.into_iter());

    assert!(result.is_ok(), "Binding failed: {:?}", result);
    let bound = result.unwrap();
    assert_eq!(bound.parameters[0].as_int(), Some(1)); // a
    assert_eq!(bound.parameters[1].as_int(), Some(10)); // kwonly1 (now at param_idx 1)
    assert_eq!(bound.parameters[2].as_int(), Some(20)); // kwonly2 (now at param_idx 2)
}

#[test]
fn test_bind_keyword_only_with_default() {
    // def f(a, *args, kwonly=99): pass
    let code = make_test_code(1, 1, vec!["a", "args", "kwonly"], CodeFlags::VARARGS);
    let kwdefaults = Some(vec![("kwonly".into(), Value::int(99).unwrap())]);
    let func = make_test_func(code, None, kwdefaults);

    let args = vec![Value::int(1).unwrap()];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_ok(), "Binding failed: {:?}", result);
    let bound = result.unwrap();
    assert_eq!(bound.parameters[0].as_int(), Some(1)); // a
    assert_eq!(bound.parameters[1].as_int(), Some(99)); // kwonly (default)
}

#[test]
fn test_bind_missing_keyword_only_error() {
    // def f(a, *args, kwonly): pass - kwonly has no default
    let code = make_test_code(1, 1, vec!["a", "args", "kwonly"], CodeFlags::VARARGS);
    let func = make_test_func(code, None, None);

    let args = vec![Value::int(1).unwrap()];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_err());
    match result.unwrap_err() {
        BindingError::MissingKeywordOnly { param_name, .. } => {
            assert_eq!(param_name.as_ref(), "kwonly");
        }
        e => panic!("Expected MissingKeywordOnly error, got {:?}", e),
    }
}

// =========================================================================
// Default Value Tests
// =========================================================================

#[test]
fn test_bind_with_positional_defaults() {
    let code = make_test_code(3, 0, vec!["a", "b", "c"], CodeFlags::NONE);
    // Defaults for b and c
    let defaults = Some(vec![Value::int(20).unwrap(), Value::int(30).unwrap()]);
    let func = make_test_func(code, defaults, None);

    let args = vec![Value::int(1).unwrap()];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert_eq!(bound.parameters[0].as_int(), Some(1)); // a (provided)
    assert_eq!(bound.parameters[1].as_int(), Some(20)); // b (default)
    assert_eq!(bound.parameters[2].as_int(), Some(30)); // c (default)
}

#[test]
fn test_bind_missing_required_positional_error() {
    let code = make_test_code(3, 0, vec!["a", "b", "c"], CodeFlags::NONE);
    // Only default for c
    let defaults = Some(vec![Value::int(30).unwrap()]);
    let func = make_test_func(code, defaults, None);

    let args = vec![Value::int(1).unwrap()];
    let result = ArgumentBinder::bind(&func, args.into_iter(), std::iter::empty());

    assert!(result.is_err());
    match result.unwrap_err() {
        BindingError::MissingPositional { param_name, .. } => {
            assert_eq!(param_name.as_ref(), "b");
        }
        _ => panic!("Expected MissingPositional error"),
    }
}

// =========================================================================
// Complex Combined Tests
// =========================================================================

#[test]
fn test_bind_complex_signature() {
    // def func(a, b, *args, kwonly, **kwargs):
    let code = make_test_code(
        2,
        1,
        vec!["a", "b", "args", "kwonly", "kwargs"],
        CodeFlags::VARARGS.union(CodeFlags::VARKEYWORDS),
    );
    let kwdefaults = Some(vec![("kwonly".into(), Value::int(999).unwrap())]);
    let func = make_test_func(code, None, kwdefaults);

    let args = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ];
    let kwargs = vec![("extra", Value::int(100).unwrap())];

    let result = ArgumentBinder::bind(&func, args.into_iter(), kwargs.into_iter());

    assert!(result.is_ok(), "Binding failed: {:?}", result);
    let bound = result.unwrap();

    // Regular params
    assert_eq!(bound.parameters[0].as_int(), Some(1)); // a
    assert_eq!(bound.parameters[1].as_int(), Some(2)); // b
    assert_eq!(bound.parameters[2].as_int(), Some(999)); // kwonly (default)

    // Varargs
    let varargs = bound.varargs.as_ref().unwrap();
    assert_eq!(varargs.len(), 2);
    assert_eq!(varargs.get(0).unwrap().as_int(), Some(3));
    assert_eq!(varargs.get(1).unwrap().as_int(), Some(4));

    // Varkw
    let varkw = bound.varkw.as_ref().unwrap();
    assert_eq!(varkw.len(), 1);
}

#[test]
fn test_bind_no_args_function() {
    let code = make_test_code(0, 0, vec![], CodeFlags::NONE);
    let func = make_test_func(code, None, None);

    let result = ArgumentBinder::bind(&func, std::iter::empty(), std::iter::empty());

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert!(bound.parameters.is_empty());
    assert!(bound.varargs.is_none());
    assert!(bound.varkw.is_none());
}

// =========================================================================
// Error Message Formatting Tests
// =========================================================================

#[test]
fn test_error_message_too_many_positional() {
    let err = BindingError::TooManyPositional {
        func_name: "foo".into(),
        expected: 2,
        given: 5,
    };
    assert_eq!(
        err.to_error_message(),
        "foo() takes 2 positional arguments but 5 were given"
    );
}

#[test]
fn test_error_message_duplicate_argument() {
    let err = BindingError::DuplicateArgument {
        func_name: "foo".into(),
        param_name: "x".into(),
    };
    assert_eq!(
        err.to_error_message(),
        "foo() got multiple values for argument 'x'"
    );
}

#[test]
fn test_error_message_unexpected_keyword() {
    let err = BindingError::UnexpectedKeyword {
        func_name: "foo".into(),
        keyword: "unknown".into(),
    };
    assert_eq!(
        err.to_error_message(),
        "foo() got an unexpected keyword argument 'unknown'"
    );
}

#[test]
fn test_error_message_missing_positional() {
    let err = BindingError::MissingPositional {
        func_name: "foo".into(),
        param_name: "x".into(),
    };
    assert_eq!(
        err.to_error_message(),
        "foo() missing required positional argument: 'x'"
    );
}

#[test]
fn test_error_message_missing_keyword_only() {
    let err = BindingError::MissingKeywordOnly {
        func_name: "foo".into(),
        param_name: "x".into(),
    };
    assert_eq!(
        err.to_error_message(),
        "foo() missing required keyword-only argument: 'x'"
    );
}
