//! Python `math` module implementation.
//!
//! Provides mathematical functions with maximum performance through:
//! - Direct hardware intrinsics (sin, cos, exp → SSE/x87 instructions)
//! - Branch-free classification (isinf, isnan via bit patterns)
//! - Const lookup tables (factorial 0-20)
//! - Binary algorithms (GCD, integer pow)
//!
//! # Performance Characteristics
//!
//! - Zero heap allocations for all operations
//! - All functions are `#[inline]` for call elimination
//! - Special value handling without branching where possible
//!
//! # Python 3.12 Compatibility
//!
//! All functions match Python 3.12 semantics including:
//! - Domain error handling (ValueError for invalid inputs)
//! - Special value propagation (inf, nan)
//! - Exact output format matching

mod angular;
mod basic;
mod classify;
mod combinatorics;
mod constants;
mod exp_log;
mod hyperbolic;
mod power;
mod special;
mod trig;

pub use angular::*;
pub use basic::*;
pub use classify::*;
pub use combinatorics::*;
pub use constants::*;
pub use exp_log::*;
pub use hyperbolic::*;
pub use power::*;
pub use special::*;
pub use trig::*;

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use std::sync::{Arc, LazyLock};

static CEIL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.ceil"), math_ceil_builtin));
static FLOOR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.floor"), math_floor_builtin));
static FABS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.fabs"), math_fabs_builtin));
static SIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.sin"), math_sin_builtin));
static COS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.cos"), math_cos_builtin));
static ACOS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.acos"), math_acos_builtin));
static EXP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.exp"), math_exp_builtin));
static LOG_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.log"), math_log_builtin));
static LOG2_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.log2"), math_log2_builtin));
static LOG10_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.log10"), math_log10_builtin));
static SQRT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.sqrt"), math_sqrt_builtin));
static GAMMA_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.gamma"), math_gamma_builtin));
static LGAMMA_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.lgamma"), math_lgamma_builtin));
static ERF_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.erf"), math_erf_builtin));
static ERFC_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.erfc"), math_erfc_builtin));
static ISFINITE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.isfinite"), math_isfinite_builtin));
static GCD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.gcd"), math_gcd_builtin));

/// The math module instance.
pub struct MathModule;

impl MathModule {
    /// Create a new math module.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for MathModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for MathModule {
    fn name(&self) -> &str {
        "math"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            // Constants
            "pi" => Ok(Value::float(PI)),
            "e" => Ok(Value::float(E)),
            "tau" => Ok(Value::float(TAU)),
            "inf" => Ok(Value::float(INFINITY)),
            "nan" => Ok(Value::float(NAN)),
            "ceil" => Ok(builtin_value(&CEIL_FUNCTION)),
            "floor" => Ok(builtin_value(&FLOOR_FUNCTION)),
            "fabs" => Ok(builtin_value(&FABS_FUNCTION)),
            "sin" => Ok(builtin_value(&SIN_FUNCTION)),
            "cos" => Ok(builtin_value(&COS_FUNCTION)),
            "acos" => Ok(builtin_value(&ACOS_FUNCTION)),
            "exp" => Ok(builtin_value(&EXP_FUNCTION)),
            "log" => Ok(builtin_value(&LOG_FUNCTION)),
            "log2" => Ok(builtin_value(&LOG2_FUNCTION)),
            "log10" => Ok(builtin_value(&LOG10_FUNCTION)),
            "sqrt" => Ok(builtin_value(&SQRT_FUNCTION)),
            "gamma" => Ok(builtin_value(&GAMMA_FUNCTION)),
            "lgamma" => Ok(builtin_value(&LGAMMA_FUNCTION)),
            "erf" => Ok(builtin_value(&ERF_FUNCTION)),
            "erfc" => Ok(builtin_value(&ERFC_FUNCTION)),
            "isfinite" => Ok(builtin_value(&ISFINITE_FUNCTION)),
            "gcd" => Ok(builtin_value(&GCD_FUNCTION)),

            // Functions are returned as None for now
            // Full implementation would return callable objects
            "trunc" | "copysign" | "fmod" | "modf" | "remainder" | "tan" | "asin" | "atan"
            | "atan2" | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "exp2"
            | "expm1" | "log1p" | "pow" | "isqrt" | "hypot" | "factorial" | "comb" | "perm"
            | "lcm" | "degrees" | "radians" | "isinf" | "isnan" => {
                // TODO: Return actual function objects when callable system is ready
                Err(ModuleError::AttributeError(format!(
                    "math.{} is not yet callable as an object",
                    name
                )))
            }

            _ => Err(ModuleError::AttributeError(format!(
                "module 'math' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            // Constants
            Arc::from("pi"),
            Arc::from("e"),
            Arc::from("tau"),
            Arc::from("inf"),
            Arc::from("nan"),
            // Basic
            Arc::from("ceil"),
            Arc::from("floor"),
            Arc::from("trunc"),
            Arc::from("fabs"),
            Arc::from("copysign"),
            Arc::from("fmod"),
            Arc::from("modf"),
            Arc::from("remainder"),
            // Trig
            Arc::from("sin"),
            Arc::from("cos"),
            Arc::from("tan"),
            Arc::from("asin"),
            Arc::from("acos"),
            Arc::from("atan"),
            Arc::from("atan2"),
            // Hyperbolic
            Arc::from("sinh"),
            Arc::from("cosh"),
            Arc::from("tanh"),
            Arc::from("asinh"),
            Arc::from("acosh"),
            Arc::from("atanh"),
            // Exp/Log
            Arc::from("exp"),
            Arc::from("exp2"),
            Arc::from("expm1"),
            Arc::from("log"),
            Arc::from("log2"),
            Arc::from("log10"),
            Arc::from("log1p"),
            // Power
            Arc::from("pow"),
            Arc::from("sqrt"),
            Arc::from("isqrt"),
            Arc::from("hypot"),
            // Special
            Arc::from("factorial"),
            Arc::from("gamma"),
            Arc::from("lgamma"),
            Arc::from("erf"),
            Arc::from("erfc"),
            // Combinatorics
            Arc::from("comb"),
            Arc::from("perm"),
            Arc::from("gcd"),
            Arc::from("lcm"),
            // Angular
            Arc::from("degrees"),
            Arc::from("radians"),
            // Classification
            Arc::from("isinf"),
            Arc::from("isnan"),
            Arc::from("isfinite"),
        ]
    }
}

// =============================================================================
// Builtin Function Wrappers
// =============================================================================

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn map_math_error(err: ModuleError) -> BuiltinError {
    match err {
        ModuleError::AttributeError(message) => BuiltinError::AttributeError(message),
        ModuleError::ValueError(message) | ModuleError::MathDomainError(message) => {
            BuiltinError::ValueError(message)
        }
        ModuleError::TypeError(message) => BuiltinError::TypeError(message),
        ModuleError::MathRangeError(message) => BuiltinError::OverflowError(message),
        ModuleError::OSError(message) => BuiltinError::OSError(message),
    }
}

#[inline]
fn expect_math_arg_count(
    args: &[Value],
    fn_name: &'static str,
    expected: usize,
) -> Result<(), BuiltinError> {
    if args.len() == expected {
        return Ok(());
    }

    Err(BuiltinError::TypeError(format!(
        "math.{fn_name}() takes exactly {expected} argument{} ({} given)",
        if expected == 1 { "" } else { "s" },
        args.len()
    )))
}

#[inline]
fn extract_float_builtin(value: &Value) -> Result<f64, BuiltinError> {
    extract_float(value).map_err(map_math_error)
}

#[inline]
fn extract_int_builtin(value: &Value) -> Result<i64, BuiltinError> {
    extract_int(value).map_err(map_math_error)
}

#[inline]
fn float_to_integral_value(value: f64) -> Result<Value, BuiltinError> {
    if value.is_nan() {
        return Err(BuiltinError::ValueError(
            "cannot convert float NaN to integer".to_string(),
        ));
    }
    if value.is_infinite() {
        return Err(BuiltinError::OverflowError(
            "cannot convert float infinity to integer".to_string(),
        ));
    }
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(BuiltinError::OverflowError(
            "math result out of range".to_string(),
        ));
    }

    Value::int(value as i64)
        .ok_or_else(|| BuiltinError::OverflowError("math result out of range".to_string()))
}

#[inline]
fn math_ceil_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "ceil", 1)?;
    float_to_integral_value(ceil(extract_float_builtin(&args[0])?))
}

#[inline]
fn math_floor_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "floor", 1)?;
    float_to_integral_value(floor(extract_float_builtin(&args[0])?))
}

#[inline]
fn math_fabs_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "fabs", 1)?;
    Ok(Value::float(extract_float_builtin(&args[0])?.abs()))
}

#[inline]
fn math_sin_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "sin", 1)?;
    Ok(Value::float(sin(extract_float_builtin(&args[0])?)))
}

#[inline]
fn math_cos_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "cos", 1)?;
    Ok(Value::float(cos(extract_float_builtin(&args[0])?)))
}

#[inline]
fn math_acos_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "acos", 1)?;
    Ok(Value::float(
        acos(extract_float_builtin(&args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_exp_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "exp", 1)?;
    Ok(Value::float(exp(extract_float_builtin(&args[0])?)))
}

#[inline]
fn math_log_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "math.log() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    let value = extract_float_builtin(&args[0])?;
    let result = match args.get(1) {
        Some(base_value) => log_base(value, extract_float_builtin(base_value)?),
        None => log(value),
    }
    .map_err(map_math_error)?;
    Ok(Value::float(result))
}

#[inline]
fn math_log2_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "log2", 1)?;
    Ok(Value::float(
        log2(extract_float_builtin(&args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_log10_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "log10", 1)?;
    Ok(Value::float(
        log10(extract_float_builtin(&args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_sqrt_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "sqrt", 1)?;
    Ok(Value::float(
        sqrt(extract_float_builtin(&args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_gamma_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "gamma", 1)?;
    Ok(Value::float(
        gamma(extract_float_builtin(&args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_lgamma_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "lgamma", 1)?;
    Ok(Value::float(
        lgamma(extract_float_builtin(&args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_erf_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "erf", 1)?;
    Ok(Value::float(erf(extract_float_builtin(&args[0])?)))
}

#[inline]
fn math_erfc_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "erfc", 1)?;
    Ok(Value::float(erfc(extract_float_builtin(&args[0])?)))
}

#[inline]
fn math_isfinite_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "isfinite", 1)?;
    Ok(Value::bool(isfinite(extract_float_builtin(&args[0])?)))
}

#[inline]
fn math_gcd_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    let mut current = 0i64;
    for arg in args {
        current = gcd(current, extract_int_builtin(arg)?);
    }
    Value::int(current)
        .ok_or_else(|| BuiltinError::OverflowError("math result out of range".to_string()))
}

/// Helper to extract a float from Value.
#[inline]
pub fn extract_float(value: &Value) -> Result<f64, ModuleError> {
    if let Some(f) = value.as_float() {
        Ok(f)
    } else if let Some(i) = value.as_int() {
        Ok(i as f64)
    } else if value.is_bool() {
        Ok(if value.as_bool().unwrap_or(false) {
            1.0
        } else {
            0.0
        })
    } else {
        Err(ModuleError::TypeError(
            "must be real number, not 'NoneType'".to_string(),
        ))
    }
}

/// Helper to extract an integer from Value.
#[inline]
pub fn extract_int(value: &Value) -> Result<i64, ModuleError> {
    if let Some(i) = value.as_int() {
        Ok(i)
    } else if value.is_bool() {
        Ok(if value.as_bool().unwrap_or(false) {
            1
        } else {
            0
        })
    } else if let Some(_f) = value.as_float() {
        Err(ModuleError::TypeError(format!(
            "'float' object cannot be interpreted as an integer"
        )))
    } else {
        Err(ModuleError::TypeError(
            "'NoneType' object cannot be interpreted as an integer".to_string(),
        ))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}
