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
static COPYSIGN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.copysign"), math_copysign_builtin));
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
static ISINF_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.isinf"), math_isinf_builtin));
static ISNAN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.isnan"), math_isnan_builtin));
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
            "copysign" => Ok(builtin_value(&COPYSIGN_FUNCTION)),
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
            "isinf" => Ok(builtin_value(&ISINF_FUNCTION)),
            "isnan" => Ok(builtin_value(&ISNAN_FUNCTION)),
            "gcd" => Ok(builtin_value(&GCD_FUNCTION)),

            // Functions are returned as None for now
            // Full implementation would return callable objects
            "trunc" | "fmod" | "modf" | "remainder" | "tan" | "asin" | "atan" | "atan2"
            | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "exp2" | "expm1"
            | "log1p" | "pow" | "isqrt" | "hypot" | "factorial" | "comb" | "perm" | "lcm"
            | "degrees" | "radians" => {
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
fn math_copysign_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "copysign", 2)?;
    Ok(Value::float(copysign(
        extract_float_builtin(&args[0])?,
        extract_float_builtin(&args[1])?,
    )))
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
fn math_isinf_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "isinf", 1)?;
    Ok(Value::bool(isinf(extract_float_builtin(&args[0])?)))
}

#[inline]
fn math_isnan_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "isnan", 1)?;
    Ok(Value::bool(isnan(extract_float_builtin(&args[0])?)))
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
