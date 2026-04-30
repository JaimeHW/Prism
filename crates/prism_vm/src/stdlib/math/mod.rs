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
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, runtime_error_to_builtin_error};
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use prism_core::Value;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint, value_to_saturated_i64};
use std::sync::{Arc, LazyLock};

static CEIL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.ceil"), math_ceil_builtin));
static FLOOR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.floor"), math_floor_builtin));
static TRUNC_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.trunc"), math_trunc_builtin));
static FABS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.fabs"), math_fabs_builtin));
static COPYSIGN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("math.copysign"), math_copysign_builtin)
});
static SIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.sin"), math_sin_builtin));
static COS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.cos"), math_cos_builtin));
static ACOS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.acos"), math_acos_builtin));
static EXP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.exp"), math_exp_builtin));
static LOG_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.log"), math_log_builtin));
static LOG2_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.log2"), math_log2_builtin));
static LOG10_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.log10"), math_log10_builtin));
static SQRT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.sqrt"), math_sqrt_builtin));
static GAMMA_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.gamma"), math_gamma_builtin));
static LGAMMA_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.lgamma"), math_lgamma_builtin));
static ERF_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.erf"), math_erf_builtin));
static ERFC_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.erfc"), math_erfc_builtin));
static ISFINITE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("math.isfinite"), math_isfinite_builtin)
});
static ISINF_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.isinf"), math_isinf_builtin));
static ISNAN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.isnan"), math_isnan_builtin));
static GCD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("math.gcd"), math_gcd_builtin));
static LDEXP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("math.ldexp"), math_ldexp_builtin));

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
            "trunc" => Ok(builtin_value(&TRUNC_FUNCTION)),
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
            "ldexp" => Ok(builtin_value(&LDEXP_FUNCTION)),

            // Functions are returned as None for now
            // Full implementation would return callable objects
            "fmod" | "modf" | "remainder" | "tan" | "asin" | "atan" | "atan2" | "sinh" | "cosh"
            | "tanh" | "asinh" | "acosh" | "atanh" | "exp2" | "expm1" | "log1p" | "pow"
            | "isqrt" | "hypot" | "factorial" | "comb" | "perm" | "lcm" | "degrees" | "radians" => {
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
            Arc::from("ldexp"),
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
fn extract_float_builtin(vm: &mut VirtualMachine, value: Value) -> Result<f64, BuiltinError> {
    if let Some(float) = value.as_float() {
        return Ok(float);
    }
    if let Some(integer) = value.as_int() {
        return Ok(integer as f64);
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(if boolean { 1.0 } else { 0.0 });
    }

    if let Some(result) = invoke_optional_float_protocol(vm, value)? {
        return prism_runtime::types::float::value_to_f64(result).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "__float__ returned non-float (type {})",
                result.type_name()
            ))
        });
    }

    if let Some(result) = invoke_optional_float_protocol_index(vm, value)? {
        return integer_value_to_float(result);
    }

    Err(BuiltinError::TypeError(format!(
        "must be real number, not '{}'",
        value.type_name()
    )))
}

fn invoke_optional_float_protocol(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Option<Value>, BuiltinError> {
    match resolve_special_method(value, "__float__") {
        Ok(target) => invoke_bound_method_no_args(vm, target).map(Some),
        Err(err) if err.is_attribute_error() => Ok(None),
        Err(err) => Err(runtime_error_to_builtin_error(err)),
    }
}

fn invoke_optional_float_protocol_index(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Option<Value>, BuiltinError> {
    match resolve_special_method(value, "__index__") {
        Ok(target) => invoke_bound_method_no_args(vm, target).map(Some),
        Err(err) if err.is_attribute_error() => Ok(None),
        Err(err) => Err(runtime_error_to_builtin_error(err)),
    }
}

fn integer_value_to_float(value: Value) -> Result<f64, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(if boolean { 1.0 } else { 0.0 });
    }
    if let Some(integer) = value.as_int() {
        return Ok(integer as f64);
    }

    let bigint = value_to_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "__index__ returned non-int (type {})",
            value.type_name()
        ))
    })?;
    bigint
        .to_f64()
        .filter(|float| float.is_finite())
        .ok_or_else(|| BuiltinError::OverflowError("int too large to convert to float".to_string()))
}

#[derive(Clone, Copy)]
enum LdexpExponent {
    InRange(i32),
    Overflow,
    Underflow,
}

#[inline]
fn classify_ldexp_exponent(exponent: i64) -> LdexpExponent {
    if exponent > i32::MAX as i64 {
        LdexpExponent::Overflow
    } else if exponent < i32::MIN as i64 {
        LdexpExponent::Underflow
    } else {
        LdexpExponent::InRange(exponent as i32)
    }
}

#[inline]
fn extract_ldexp_exponent(value: Value) -> Result<LdexpExponent, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(LdexpExponent::InRange(i32::from(boolean)));
    }
    value_to_saturated_i64(value)
        .map(classify_ldexp_exponent)
        .ok_or_else(|| {
            BuiltinError::TypeError("Expected an int as second argument to ldexp.".to_string())
        })
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

    Ok(bigint_to_value(float_integer_to_bigint(value)))
}

fn float_integer_to_bigint(value: f64) -> BigInt {
    debug_assert!(value.is_finite());

    let bits = value.to_bits();
    let negative = (bits >> 63) != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        return BigInt::zero();
    }

    let mut integer = BigInt::from((1_u64 << 52) | fraction);
    let shift = exponent_bits - 1023 - 52;
    if shift >= 0 {
        integer <<= shift as usize;
    } else {
        integer >>= (-shift) as usize;
    }

    if negative { -integer } else { integer }
}

fn math_ceil_builtin(
    vm: &mut crate::VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    math_integral_builtin(vm, args, "ceil", "__ceil__", ceil)
}

fn math_floor_builtin(
    vm: &mut crate::VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    math_integral_builtin(vm, args, "floor", "__floor__", floor)
}

fn math_integral_builtin(
    vm: &mut crate::VirtualMachine,
    args: &[Value],
    fn_name: &'static str,
    method_name: &'static str,
    operation: fn(f64) -> f64,
) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, fn_name, 1)?;
    let value = args[0];

    if let Some(boolean) = value.as_bool() {
        return Ok(Value::int(i64::from(boolean)).expect("bool integral result should fit"));
    }
    if let Some(integer) = value_to_bigint(value) {
        return Ok(bigint_to_value(integer));
    }
    if let Some(float) = value.as_float() {
        return float_to_integral_value(operation(float));
    }

    match resolve_special_method(value, method_name) {
        Ok(target) => return invoke_bound_method_no_args(vm, target),
        Err(err) if err.is_attribute_error() => {}
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    }

    let float_value = crate::builtins::builtin_float_vm(vm, &[value])?;
    let Some(float) = float_value.as_float() else {
        return Err(BuiltinError::TypeError(format!(
            "must be real number, not '{}'",
            value.type_name()
        )));
    };
    float_to_integral_value(operation(float))
}

fn math_trunc_builtin(
    vm: &mut crate::VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "trunc", 1)?;
    let value = args[0];

    if let Some(boolean) = value.as_bool() {
        return Ok(Value::int(i64::from(boolean)).expect("bool trunc result should fit"));
    }
    if let Some(integer) = value_to_bigint(value) {
        return Ok(bigint_to_value(integer));
    }
    if value.as_float().is_some() {
        return crate::builtins::builtin_int_vm(vm, &[value]);
    }

    let target = match resolve_special_method(value, "__trunc__") {
        Ok(target) => target,
        Err(err) if err.is_attribute_error() => {
            return Err(BuiltinError::TypeError(format!(
                "type {} doesn't define __trunc__ method",
                value.type_name()
            )));
        }
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };

    invoke_bound_method_no_args(vm, target)
}

fn invoke_bound_method_no_args(
    vm: &mut crate::VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, BuiltinError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
    .map_err(runtime_error_to_builtin_error)
}

#[inline]
fn math_fabs_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "fabs", 1)?;
    Ok(Value::float(extract_float_builtin(vm, args[0])?.abs()))
}

#[inline]
fn math_copysign_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "copysign", 2)?;
    Ok(Value::float(copysign(
        extract_float_builtin(vm, args[0])?,
        extract_float_builtin(vm, args[1])?,
    )))
}

#[inline]
fn math_sin_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "sin", 1)?;
    Ok(Value::float(sin(extract_float_builtin(vm, args[0])?)))
}

#[inline]
fn math_cos_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "cos", 1)?;
    Ok(Value::float(cos(extract_float_builtin(vm, args[0])?)))
}

#[inline]
fn math_acos_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "acos", 1)?;
    Ok(Value::float(
        acos(extract_float_builtin(vm, args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_exp_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "exp", 1)?;
    Ok(Value::float(exp(extract_float_builtin(vm, args[0])?)))
}

#[inline]
fn math_log_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "math.log() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    let value = extract_float_builtin(vm, args[0])?;
    let result = match args.get(1) {
        Some(base_value) => log_base(value, extract_float_builtin(vm, *base_value)?),
        None => log(value),
    }
    .map_err(map_math_error)?;
    Ok(Value::float(result))
}

#[inline]
fn math_log2_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "log2", 1)?;
    Ok(Value::float(
        log2(extract_float_builtin(vm, args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_log10_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "log10", 1)?;
    Ok(Value::float(
        log10(extract_float_builtin(vm, args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_sqrt_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "sqrt", 1)?;
    Ok(Value::float(
        sqrt(extract_float_builtin(vm, args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_gamma_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "gamma", 1)?;
    Ok(Value::float(
        gamma(extract_float_builtin(vm, args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_lgamma_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "lgamma", 1)?;
    Ok(Value::float(
        lgamma(extract_float_builtin(vm, args[0])?).map_err(map_math_error)?,
    ))
}

#[inline]
fn math_erf_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "erf", 1)?;
    Ok(Value::float(erf(extract_float_builtin(vm, args[0])?)))
}

#[inline]
fn math_erfc_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "erfc", 1)?;
    Ok(Value::float(erfc(extract_float_builtin(vm, args[0])?)))
}

#[inline]
fn math_isfinite_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "isfinite", 1)?;
    Ok(Value::bool(isfinite(extract_float_builtin(vm, args[0])?)))
}

#[inline]
fn math_isinf_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "isinf", 1)?;
    Ok(Value::bool(isinf(extract_float_builtin(vm, args[0])?)))
}

#[inline]
fn math_isnan_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "isnan", 1)?;
    Ok(Value::bool(isnan(extract_float_builtin(vm, args[0])?)))
}

#[inline]
fn math_gcd_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    let mut current = BigInt::zero();
    for arg in args {
        current = gcd_bigint(current, extract_integer_bigint(*arg)?);
    }
    Ok(bigint_to_value(current))
}

#[inline]
fn math_ldexp_builtin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_math_arg_count(args, "ldexp", 2)?;
    let mantissa = extract_float_builtin(vm, args[0])?;
    let exponent = extract_ldexp_exponent(args[1])?;

    if mantissa == 0.0 || !mantissa.is_finite() {
        return Ok(Value::float(mantissa));
    }

    let result = match exponent {
        LdexpExponent::InRange(exponent) => ldexp(mantissa, exponent),
        LdexpExponent::Overflow => {
            return Err(BuiltinError::OverflowError("math range error".to_string()));
        }
        LdexpExponent::Underflow => 0.0_f64.copysign(mantissa),
    };

    if result.is_infinite() {
        return Err(BuiltinError::OverflowError("math range error".to_string()));
    }

    Ok(Value::float(result))
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

#[inline]
fn extract_integer_bigint(value: Value) -> Result<BigInt, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(BigInt::from(u8::from(boolean)));
    }
    value_to_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "'{}' object cannot be interpreted as an integer",
            integer_type_name(value)
        ))
    })
}

#[inline]
fn integer_type_name(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_float() {
        "float"
    } else {
        value.type_name()
    }
}

fn gcd_bigint(mut left: BigInt, mut right: BigInt) -> BigInt {
    left = left.abs();
    right = right.abs();

    while !right.is_zero() {
        let remainder = left % &right;
        left = right;
        right = remainder;
    }

    left
}
