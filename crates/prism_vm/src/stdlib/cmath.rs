//! Native Python `cmath` module implementation.
//!
//! Prism keeps complex values as compact native runtime objects, so exposing the
//! core transcendental operations here avoids Python-level adapter overhead on
//! numeric-heavy import paths and user code.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::python_numeric::{ComplexParts, complex_like_parts};
use prism_core::Value;
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = std::f64::consts::TAU;
const E: f64 = std::f64::consts::E;
const LOG10_E: f64 = std::f64::consts::LOG10_E;

static ACOS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.acos"), cmath_acos));
static ACOSH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.acosh"), cmath_acosh));
static ASIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.asin"), cmath_asin));
static ASINH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.asinh"), cmath_asinh));
static ATAN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.atan"), cmath_atan));
static ATANH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.atanh"), cmath_atanh));
static COS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.cos"), cmath_cos));
static COSH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.cosh"), cmath_cosh));
static EXP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.exp"), cmath_exp));
static ISFINITE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.isfinite"), cmath_isfinite));
static ISINF_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.isinf"), cmath_isinf));
static ISNAN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.isnan"), cmath_isnan));
static LOG_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.log"), cmath_log));
static LOG10_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.log10"), cmath_log10));
static PHASE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.phase"), cmath_phase));
static POLAR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.polar"), cmath_polar));
static RECT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("cmath.rect"), cmath_rect));
static SIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.sin"), cmath_sin));
static SINH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.sinh"), cmath_sinh));
static SQRT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.sqrt"), cmath_sqrt));
static TAN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.tan"), cmath_tan));
static TANH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("cmath.tanh"), cmath_tanh));

const EXPORTS: &[&str] = &[
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosh", "e", "exp", "inf", "infj",
    "isfinite", "isinf", "isnan", "log", "log10", "nan", "nanj", "phase", "pi", "polar", "rect",
    "sin", "sinh", "sqrt", "tan", "tanh", "tau",
];

pub struct CMathModule;

impl CMathModule {
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for CMathModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CMathModule {
    fn name(&self) -> &str {
        "cmath"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "pi" => Ok(Value::float(PI)),
            "e" => Ok(Value::float(E)),
            "tau" => Ok(Value::float(TAU)),
            "inf" => Ok(Value::float(f64::INFINITY)),
            "nan" => Ok(Value::float(f64::NAN)),
            "infj" => Ok(complex_value(0.0, f64::INFINITY)),
            "nanj" => Ok(complex_value(0.0, f64::NAN)),
            "acos" => Ok(builtin_value(&ACOS_FUNCTION)),
            "acosh" => Ok(builtin_value(&ACOSH_FUNCTION)),
            "asin" => Ok(builtin_value(&ASIN_FUNCTION)),
            "asinh" => Ok(builtin_value(&ASINH_FUNCTION)),
            "atan" => Ok(builtin_value(&ATAN_FUNCTION)),
            "atanh" => Ok(builtin_value(&ATANH_FUNCTION)),
            "cos" => Ok(builtin_value(&COS_FUNCTION)),
            "cosh" => Ok(builtin_value(&COSH_FUNCTION)),
            "exp" => Ok(builtin_value(&EXP_FUNCTION)),
            "isfinite" => Ok(builtin_value(&ISFINITE_FUNCTION)),
            "isinf" => Ok(builtin_value(&ISINF_FUNCTION)),
            "isnan" => Ok(builtin_value(&ISNAN_FUNCTION)),
            "log" => Ok(builtin_value(&LOG_FUNCTION)),
            "log10" => Ok(builtin_value(&LOG10_FUNCTION)),
            "phase" => Ok(builtin_value(&PHASE_FUNCTION)),
            "polar" => Ok(builtin_value(&POLAR_FUNCTION)),
            "rect" => Ok(builtin_value(&RECT_FUNCTION)),
            "sin" => Ok(builtin_value(&SIN_FUNCTION)),
            "sinh" => Ok(builtin_value(&SINH_FUNCTION)),
            "sqrt" => Ok(builtin_value(&SQRT_FUNCTION)),
            "tan" => Ok(builtin_value(&TAN_FUNCTION)),
            "tanh" => Ok(builtin_value(&TANH_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'cmath' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        EXPORTS.iter().map(|name| Arc::from(*name)).collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    #[inline]
    const fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    #[inline]
    fn abs(self) -> f64 {
        self.real.hypot(self.imag)
    }

    #[inline]
    fn phase(self) -> f64 {
        self.imag.atan2(self.real)
    }

    #[inline]
    fn square(self) -> Self {
        Self::new(
            self.real.mul_add(self.real, -(self.imag * self.imag)),
            2.0 * self.real * self.imag,
        )
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imag + other.imag)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.real - other.real, self.imag - other.imag)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.real.mul_add(other.real, -(self.imag * other.imag)),
            self.real.mul_add(other.imag, self.imag * other.real),
        )
    }

    #[inline]
    fn div(self, other: Self) -> Result<Self, BuiltinError> {
        let denominator = other.real.mul_add(other.real, other.imag * other.imag);
        if denominator == 0.0 {
            return Err(math_domain_error());
        }
        Ok(Self::new(
            self.real.mul_add(other.real, self.imag * other.imag) / denominator,
            self.imag.mul_add(other.real, -(self.real * other.imag)) / denominator,
        ))
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        Self::new(self.real * factor, self.imag * factor)
    }

    #[inline]
    fn isfinite(self) -> bool {
        self.real.is_finite() && self.imag.is_finite()
    }

    #[inline]
    fn isinf(self) -> bool {
        self.real.is_infinite() || self.imag.is_infinite()
    }

    #[inline]
    fn isnan(self) -> bool {
        self.real.is_nan() || self.imag.is_nan()
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    let ptr = function as *const BuiltinFunctionObject as *const ();
    Value::object_ptr(ptr)
}

#[inline]
fn complex_value(real: f64, imag: f64) -> Value {
    crate::alloc_managed_value(ComplexObject::new(real, imag))
}

#[inline]
fn tuple_value(items: Vec<Value>) -> Value {
    crate::alloc_managed_value(TupleObject::from_vec(items))
}

#[inline]
fn complex_result(value: Complex) -> Value {
    complex_value(value.real, value.imag)
}

fn complex_arg(
    vm: &mut VirtualMachine,
    args: &[Value],
    function: &'static str,
) -> Result<Complex, BuiltinError> {
    expect_arg_count(args, function, 1)?;
    complex_from_value(vm, args[0])
}

fn complex_from_value(vm: &mut VirtualMachine, value: Value) -> Result<Complex, BuiltinError> {
    if let Some(parts) = complex_like_parts(value) {
        return Ok(parts.into());
    }

    let converted = crate::builtins::builtin_complex_vm(vm, &[value])?;
    complex_like_parts(converted)
        .map(Into::into)
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "complex() returned non-complex (type {})",
                converted.type_name()
            ))
        })
}

impl From<ComplexParts> for Complex {
    #[inline]
    fn from(value: ComplexParts) -> Self {
        Self::new(value.real, value.imag)
    }
}

#[inline]
fn expect_arg_count(
    args: &[Value],
    function: &'static str,
    expected: usize,
) -> Result<(), BuiltinError> {
    if args.len() != expected {
        return Err(BuiltinError::TypeError(format!(
            "cmath.{function}() takes exactly {expected} argument{} ({} given)",
            if expected == 1 { "" } else { "s" },
            args.len()
        )));
    }
    Ok(())
}

#[inline]
fn float_arg(value: Value, name: &'static str) -> Result<f64, BuiltinError> {
    if let Some(float) = value.as_float() {
        return Ok(float);
    }
    if let Some(integer) = value.as_int() {
        return Ok(integer as f64);
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(if boolean { 1.0 } else { 0.0 });
    }
    Err(BuiltinError::TypeError(format!(
        "must be real number, not {}",
        name
    )))
}

#[inline]
fn math_domain_error() -> BuiltinError {
    BuiltinError::ValueError("math domain error".to_string())
}

#[inline]
fn one() -> Complex {
    Complex::new(1.0, 0.0)
}

#[inline]
fn i() -> Complex {
    Complex::new(0.0, 1.0)
}

#[inline]
fn neg_i() -> Complex {
    Complex::new(0.0, -1.0)
}

fn complex_sqrt(value: Complex) -> Complex {
    if value.real == 0.0 && value.imag == 0.0 {
        return Complex::new(0.0, value.imag);
    }

    let magnitude = value.abs();
    let real = ((magnitude + value.real) * 0.5).sqrt();
    let imag = ((magnitude - value.real) * 0.5).sqrt().copysign(value.imag);
    Complex::new(real, imag)
}

fn complex_log(value: Complex) -> Result<Complex, BuiltinError> {
    if value.real == 0.0 && value.imag == 0.0 {
        return Err(math_domain_error());
    }
    Ok(Complex::new(value.abs().ln(), value.phase()))
}

#[inline]
fn complex_exp(value: Complex) -> Complex {
    let scale = value.real.exp();
    Complex::new(scale * value.imag.cos(), scale * value.imag.sin())
}

#[inline]
fn complex_sin(value: Complex) -> Complex {
    Complex::new(
        value.real.sin() * value.imag.cosh(),
        value.real.cos() * value.imag.sinh(),
    )
}

#[inline]
fn complex_cos(value: Complex) -> Complex {
    Complex::new(
        value.real.cos() * value.imag.cosh(),
        -(value.real.sin() * value.imag.sinh()),
    )
}

#[inline]
fn complex_sinh(value: Complex) -> Complex {
    Complex::new(
        value.real.sinh() * value.imag.cos(),
        value.real.cosh() * value.imag.sin(),
    )
}

#[inline]
fn complex_cosh(value: Complex) -> Complex {
    Complex::new(
        value.real.cosh() * value.imag.cos(),
        value.real.sinh() * value.imag.sin(),
    )
}

fn complex_asin(value: Complex) -> Result<Complex, BuiltinError> {
    let iz = i().mul(value);
    let root = complex_sqrt(one().sub(value.square()));
    Ok(neg_i().mul(complex_log(iz.add(root))?))
}

fn complex_atan(value: Complex) -> Result<Complex, BuiltinError> {
    let iz = i().mul(value);
    let numerator = complex_log(one().add(iz))?;
    let denominator = complex_log(one().sub(iz))?;
    Ok(neg_i().scale(0.5).mul(numerator.sub(denominator)))
}

fn complex_acosh(value: Complex) -> Result<Complex, BuiltinError> {
    complex_log(value.add(complex_sqrt(value.add(one())).mul(complex_sqrt(value.sub(one())))))
}

fn complex_atanh(value: Complex) -> Result<Complex, BuiltinError> {
    let numerator = complex_log(one().add(value))?;
    let denominator = complex_log(one().sub(value))?;
    Ok(numerator.sub(denominator).scale(0.5))
}

fn cmath_acos(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let z = complex_arg(vm, args, "acos")?;
    let asin = complex_asin(z)?;
    Ok(complex_result(Complex::new(
        PI * 0.5 - asin.real,
        -asin.imag,
    )))
}

fn cmath_acosh(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_acosh(complex_arg(
        vm, args, "acosh",
    )?)?))
}

fn cmath_asin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_asin(complex_arg(
        vm, args, "asin",
    )?)?))
}

fn cmath_asinh(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let z = complex_arg(vm, args, "asinh")?;
    Ok(complex_result(complex_log(
        z.add(complex_sqrt(z.square().add(one()))),
    )?))
}

fn cmath_atan(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_atan(complex_arg(
        vm, args, "atan",
    )?)?))
}

fn cmath_atanh(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_atanh(complex_arg(
        vm, args, "atanh",
    )?)?))
}

fn cmath_cos(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_cos(complex_arg(vm, args, "cos")?)))
}

fn cmath_cosh(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_cosh(complex_arg(vm, args, "cosh")?)))
}

fn cmath_exp(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_exp(complex_arg(vm, args, "exp")?)))
}

fn cmath_isfinite(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::bool(complex_arg(vm, args, "isfinite")?.isfinite()))
}

fn cmath_isinf(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::bool(complex_arg(vm, args, "isinf")?.isinf()))
}

fn cmath_isnan(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::bool(complex_arg(vm, args, "isnan")?.isnan()))
}

fn cmath_log(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "cmath.log() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    let result = complex_log(complex_from_value(vm, args[0])?)?;
    match args.get(1) {
        Some(base) => Ok(complex_result(
            result.div(complex_log(complex_from_value(vm, *base)?)?)?,
        )),
        None => Ok(complex_result(result)),
    }
}

fn cmath_log10(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let result = complex_log(complex_arg(vm, args, "log10")?)?;
    Ok(complex_result(result.scale(LOG10_E)))
}

fn cmath_phase(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::float(complex_arg(vm, args, "phase")?.phase()))
}

fn cmath_polar(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let z = complex_arg(vm, args, "polar")?;
    Ok(tuple_value(vec![
        Value::float(z.abs()),
        Value::float(z.phase()),
    ]))
}

fn cmath_rect(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count(args, "rect", 2)?;
    let radius = float_arg(args[0], args[0].type_name())?;
    let phase = float_arg(args[1], args[1].type_name())?;
    Ok(complex_result(Complex::new(
        radius * phase.cos(),
        radius * phase.sin(),
    )))
}

fn cmath_sin(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_sin(complex_arg(vm, args, "sin")?)))
}

fn cmath_sinh(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_sinh(complex_arg(vm, args, "sinh")?)))
}

fn cmath_sqrt(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(complex_result(complex_sqrt(complex_arg(vm, args, "sqrt")?)))
}

fn cmath_tan(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let z = complex_arg(vm, args, "tan")?;
    Ok(complex_result(complex_sin(z).div(complex_cos(z))?))
}

fn cmath_tanh(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let z = complex_arg(vm, args, "tanh")?;
    Ok(complex_result(complex_sinh(z).div(complex_cosh(z))?))
}
