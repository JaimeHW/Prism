use super::*;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_module_name() {
    let m = MathModule::new();
    assert_eq!(m.name(), "math");
}

#[test]
fn test_get_pi() {
    let m = MathModule::new();
    let pi = m.get_attr("pi").unwrap();
    assert!(pi.is_float());
    assert!((pi.as_float().unwrap() - std::f64::consts::PI).abs() < 1e-15);
}

#[test]
fn test_get_e() {
    let m = MathModule::new();
    let e = m.get_attr("e").unwrap();
    assert!(e.is_float());
    assert!((e.as_float().unwrap() - std::f64::consts::E).abs() < 1e-15);
}

#[test]
fn test_get_tau() {
    let m = MathModule::new();
    let tau = m.get_attr("tau").unwrap();
    assert!(tau.is_float());
    assert!((tau.as_float().unwrap() - std::f64::consts::TAU).abs() < 1e-15);
}

#[test]
fn test_get_inf() {
    let m = MathModule::new();
    let inf = m.get_attr("inf").unwrap();
    assert!(inf.is_float());
    assert!(inf.as_float().unwrap().is_infinite());
    assert!(inf.as_float().unwrap().is_sign_positive());
}

#[test]
fn test_get_nan() {
    let m = MathModule::new();
    let nan = m.get_attr("nan").unwrap();
    assert!(nan.is_float());
    assert!(nan.as_float().unwrap().is_nan());
}

#[test]
fn test_unknown_attr() {
    let m = MathModule::new();
    let result = m.get_attr("nonexistent");
    assert!(result.is_err());
    match result {
        Err(ModuleError::AttributeError(msg)) => {
            assert!(msg.contains("no attribute 'nonexistent'"));
        }
        _ => panic!("Expected AttributeError"),
    }
}

#[test]
fn test_dir() {
    let m = MathModule::new();
    let attrs = m.dir();
    assert!(attrs.contains(&Arc::from("pi")));
    assert!(attrs.contains(&Arc::from("sin")));
    assert!(attrs.contains(&Arc::from("sqrt")));
    assert!(attrs.contains(&Arc::from("factorial")));
    assert!(attrs.len() >= 40); // We have 42+ functions
}

#[test]
fn test_get_attr_exposes_callable_math_exports() {
    let module = MathModule::new();
    assert!(module.get_attr("fabs").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("gcd").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("log").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("sqrt").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("sin").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("lgamma").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("erf").unwrap().as_object_ptr().is_some());
}

#[test]
fn test_math_gcd_supports_zero_and_variadic_arguments() {
    let builtin = builtin_from_value(MathModule::new().get_attr("gcd").unwrap());
    assert_eq!(builtin.call(&[]).unwrap().as_int(), Some(0));
    assert_eq!(
        builtin
            .call(&[Value::int(48).unwrap(), Value::int(18).unwrap()])
            .unwrap()
            .as_int(),
        Some(6)
    );
    assert_eq!(
        builtin
            .call(&[
                Value::int(48).unwrap(),
                Value::int(18).unwrap(),
                Value::bool(true),
            ])
            .unwrap()
            .as_int(),
        Some(1)
    );
}

#[test]
fn test_extract_float_from_float() {
    let v = Value::float(3.14);
    assert!((extract_float(&v).unwrap() - 3.14).abs() < 1e-15);
}

#[test]
fn test_extract_float_from_int() {
    let v = Value::int(42).unwrap();
    assert!((extract_float(&v).unwrap() - 42.0).abs() < 1e-15);
}

#[test]
fn test_extract_float_from_bool() {
    let t = Value::bool(true);
    let f = Value::bool(false);
    assert!((extract_float(&t).unwrap() - 1.0).abs() < 1e-15);
    assert!((extract_float(&f).unwrap()).abs() < 1e-15);
}

#[test]
fn test_extract_float_from_none() {
    let v = Value::none();
    assert!(extract_float(&v).is_err());
}

#[test]
fn test_extract_int_from_int() {
    let v = Value::int(42).unwrap();
    assert_eq!(extract_int(&v).unwrap(), 42);
}

#[test]
fn test_extract_int_from_bool() {
    let t = Value::bool(true);
    let f = Value::bool(false);
    assert_eq!(extract_int(&t).unwrap(), 1);
    assert_eq!(extract_int(&f).unwrap(), 0);
}

#[test]
fn test_extract_int_from_float_fails() {
    let v = Value::float(3.14);
    assert!(extract_int(&v).is_err());
}

#[test]
fn test_log_builtin_supports_default_base_and_exp_inverse() {
    let module = MathModule::new();
    let log_fn = builtin_from_value(module.get_attr("log").unwrap());
    let exp_fn = builtin_from_value(module.get_attr("exp").unwrap());

    let natural = log_fn
        .call(&[Value::float(std::f64::consts::E)])
        .expect("math.log should work");
    assert!((natural.as_float().unwrap() - 1.0).abs() < 1e-12);

    let base10 = log_fn
        .call(&[Value::float(1000.0), Value::float(10.0)])
        .expect("math.log with base should work");
    assert!((base10.as_float().unwrap() - 3.0).abs() < 1e-12);

    let exp_value = exp_fn
        .call(&[Value::float(2.0)])
        .expect("math.exp should work");
    assert!((exp_value.as_float().unwrap() - std::f64::consts::E.powf(2.0)).abs() < 1e-12);
}

#[test]
fn test_floor_and_ceil_builtins_return_ints() {
    let module = MathModule::new();
    let floor_fn = builtin_from_value(module.get_attr("floor").unwrap());
    let ceil_fn = builtin_from_value(module.get_attr("ceil").unwrap());

    assert_eq!(
        floor_fn.call(&[Value::float(2.9)]).unwrap().as_int(),
        Some(2)
    );
    assert_eq!(
        ceil_fn.call(&[Value::float(-2.1)]).unwrap().as_int(),
        Some(-2)
    );
}

#[test]
fn test_special_math_builtins_match_known_values() {
    let module = MathModule::new();
    let fabs_fn = builtin_from_value(module.get_attr("fabs").unwrap());
    let log2_fn = builtin_from_value(module.get_attr("log2").unwrap());
    let log10_fn = builtin_from_value(module.get_attr("log10").unwrap());
    let gamma_fn = builtin_from_value(module.get_attr("gamma").unwrap());
    let lgamma_fn = builtin_from_value(module.get_attr("lgamma").unwrap());
    let erf_fn = builtin_from_value(module.get_attr("erf").unwrap());
    let erfc_fn = builtin_from_value(module.get_attr("erfc").unwrap());

    assert_eq!(
        fabs_fn.call(&[Value::int(-7).unwrap()]).unwrap().as_float(),
        Some(7.0)
    );
    assert!(
        (log2_fn
            .call(&[Value::float(8.0)])
            .unwrap()
            .as_float()
            .unwrap()
            - 3.0)
            .abs()
            < 1e-12
    );
    assert!(
        (log10_fn
            .call(&[Value::float(1000.0)])
            .unwrap()
            .as_float()
            .unwrap()
            - 3.0)
            .abs()
            < 1e-12
    );
    assert!(
        (gamma_fn
            .call(&[Value::float(5.0)])
            .unwrap()
            .as_float()
            .unwrap()
            - 24.0)
            .abs()
            < 1e-10
    );
    assert!(
        (lgamma_fn
            .call(&[Value::float(5.0)])
            .unwrap()
            .as_float()
            .unwrap()
            - 3.1780538303479458)
            .abs()
            < 1e-12
    );
    assert!(
        (erf_fn
            .call(&[Value::float(0.0)])
            .unwrap()
            .as_float()
            .unwrap()
            - 0.0)
            .abs()
            < 1e-12
    );
    assert!(
        (erfc_fn
            .call(&[Value::float(0.0)])
            .unwrap()
            .as_float()
            .unwrap()
            - 1.0)
            .abs()
            < 1e-12
    );
}
