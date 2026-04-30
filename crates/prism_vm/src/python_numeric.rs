//! Shared helpers for Python's numeric tower semantics.
//!
//! CPython models `bool` as a proper subtype of `int`. Prism keeps bool and int
//! as distinct tagged values for representation efficiency, so operator and
//! builtin implementations use these helpers when they need Python-compatible
//! numeric coercion.

use crate::error::RuntimeError;
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::float::value_to_f64;
use prism_runtime::types::int::{value_to_bigint, value_to_i64};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexParts {
    pub real: f64,
    pub imag: f64,
}

/// Convert a value to its Python integer slot value.
#[inline(always)]
pub fn int_like_value(value: Value) -> Option<i64> {
    value
        .as_bool()
        .map(i64::from)
        .or_else(|| value_to_i64(value))
}

/// Convert a value to its Python real-number slot value.
#[inline(always)]
pub fn float_like_value(value: Value) -> Option<f64> {
    if let Some(float) = value_to_f64(value) {
        return Some(float);
    }
    if let Some(integer) = int_like_value(value) {
        return Some(integer as f64);
    }

    value_to_bigint(value).and_then(|integer| integer.to_f64().filter(|float| float.is_finite()))
}

/// Check whether a value is an exact complex object.
#[inline(always)]
pub fn is_complex_value(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    let type_id = unsafe { (*(ptr as *const prism_runtime::object::ObjectHeader)).type_id };
    type_id == TypeId::COMPLEX
}

/// Convert a value to Python complex components when it participates in the
/// numeric tower as a real or complex number.
#[inline(always)]
pub fn complex_like_parts(value: Value) -> Option<ComplexParts> {
    if let Some(float_value) = value_to_f64(value) {
        return Some(ComplexParts {
            real: float_value,
            imag: 0.0,
        });
    }

    if let Some(int_value) = float_like_value(value) {
        return Some(ComplexParts {
            real: int_value,
            imag: 0.0,
        });
    }

    let ptr = value.as_object_ptr()?;
    let type_id = unsafe { (*(ptr as *const prism_runtime::object::ObjectHeader)).type_id };
    if type_id != TypeId::COMPLEX {
        return None;
    }

    let complex = unsafe { &*(ptr as *const ComplexObject) };
    Some(ComplexParts {
        real: complex.real(),
        imag: complex.imag(),
    })
}

/// Python-compatible float modulo for `%`, `operator.mod`, and speculative
/// numeric fast paths.
#[inline]
pub fn python_float_modulo(left: f64, right: f64) -> Result<f64, RuntimeError> {
    if right == 0.0 {
        return Err(RuntimeError::zero_division_with_message("float modulo"));
    }

    if right.is_infinite() && left.is_finite() {
        if left == 0.0 {
            return Ok(0.0_f64.copysign(right));
        }
        return Ok(if (left < 0.0) != (right < 0.0) {
            right
        } else {
            left
        });
    }

    let result = left - right * (left / right).floor();
    if result == 0.0 {
        Ok(0.0_f64.copysign(right))
    } else {
        Ok(result)
    }
}

/// Fast-path IEEE float power when CPython does not require non-libm behavior.
#[inline(always)]
pub fn python_float_pow_fast_path(base: f64, exponent: f64) -> Option<f64> {
    if exponent == 0.0 || base == 1.0 || (base == -1.0 && exponent.is_infinite()) {
        return Some(1.0);
    }

    if float_pow_needs_runtime_path(base, exponent) {
        None
    } else {
        Some(base.powf(exponent))
    }
}

/// Python-compatible float power, including CPython's explicit edge semantics.
#[inline]
pub fn python_float_pow_value(base: f64, exponent: f64) -> Result<Value, RuntimeError> {
    if let Some(result) = python_float_pow_fast_path(base, exponent) {
        return Ok(Value::float(result));
    }

    if base == 0.0 && exponent.is_finite() && exponent < 0.0 {
        return Err(RuntimeError::zero_division_with_message(
            "0.0 cannot be raised to a negative power",
        ));
    }

    debug_assert!(base.is_finite() && base < 0.0);
    debug_assert!(exponent.is_finite() && !float_is_integral(exponent));

    let magnitude = (-base).powf(exponent);
    let phase = std::f64::consts::PI * exponent;
    Ok(crate::alloc_managed_value(ComplexObject::new(
        magnitude * phase.cos(),
        magnitude * phase.sin(),
    )))
}

/// Exact complex multiplication for Python's numeric tower.
///
/// Returns `None` unless at least one operand is an exact native complex. That
/// keeps int/float fast paths in their native representation and leaves user
/// numeric classes to the special-method protocol.
#[inline]
pub fn python_complex_mul_value(left: Value, right: Value) -> Option<Value> {
    let (left, right) = exact_complex_operands(left, right)?;
    Some(complex_parts_to_value(ComplexParts {
        real: left.real.mul_add(right.real, -(left.imag * right.imag)),
        imag: left.real.mul_add(right.imag, left.imag * right.real),
    }))
}

/// Exact complex true division for Python's numeric tower.
#[inline]
pub fn python_complex_true_div_value(
    left: Value,
    right: Value,
) -> Option<Result<Value, RuntimeError>> {
    let (left, right) = exact_complex_operands(left, right)?;
    Some(complex_div_parts(left, right).map(complex_parts_to_value))
}

/// Exact complex exponentiation for Python's numeric tower.
#[inline]
pub fn python_complex_pow_value(
    base: Value,
    exponent: Value,
) -> Option<Result<Value, RuntimeError>> {
    let (base, exponent) = exact_complex_operands(base, exponent)?;
    Some(complex_pow_parts(base, exponent).map(complex_parts_to_value))
}

#[inline]
fn exact_complex_operands(left: Value, right: Value) -> Option<(ComplexParts, ComplexParts)> {
    if !is_complex_value(left) && !is_complex_value(right) {
        return None;
    }

    Some((complex_like_parts(left)?, complex_like_parts(right)?))
}

#[inline]
fn complex_parts_to_value(value: ComplexParts) -> Value {
    crate::alloc_managed_value(ComplexObject::new(value.real, value.imag))
}

#[inline]
fn complex_div_parts(
    left: ComplexParts,
    right: ComplexParts,
) -> Result<ComplexParts, RuntimeError> {
    if right.real == 0.0 && right.imag == 0.0 {
        return Err(RuntimeError::zero_division_with_message(
            "complex division by zero",
        ));
    }

    if right.real.abs() >= right.imag.abs() {
        let ratio = right.imag / right.real;
        let denominator = right.real + right.imag * ratio;
        Ok(ComplexParts {
            real: (left.real + left.imag * ratio) / denominator,
            imag: (left.imag - left.real * ratio) / denominator,
        })
    } else {
        let ratio = right.real / right.imag;
        let denominator = right.real * ratio + right.imag;
        Ok(ComplexParts {
            real: (left.real * ratio + left.imag) / denominator,
            imag: (left.imag * ratio - left.real) / denominator,
        })
    }
}

fn complex_pow_parts(
    base: ComplexParts,
    exponent: ComplexParts,
) -> Result<ComplexParts, RuntimeError> {
    if exponent.real == 0.0 && exponent.imag == 0.0 {
        return Ok(ComplexParts {
            real: 1.0,
            imag: 0.0,
        });
    }

    if base.real == 0.0 && base.imag == 0.0 {
        if exponent.imag != 0.0 || exponent.real < 0.0 {
            return Err(RuntimeError::zero_division_with_message(
                "0.0 to a negative or complex power",
            ));
        }
        return Ok(ComplexParts {
            real: 0.0,
            imag: 0.0,
        });
    }

    if exponent.imag == 0.0
        && base.imag == 0.0
        && (base.real >= 0.0 || exponent.real.trunc() == exponent.real)
    {
        return Ok(ComplexParts {
            real: base.real.powf(exponent.real),
            imag: 0.0,
        });
    }

    let log_abs = base.real.hypot(base.imag).ln();
    let phase = base.imag.atan2(base.real);
    let scaled_real = exponent.real.mul_add(log_abs, -(exponent.imag * phase));
    let scaled_imag = exponent.real.mul_add(phase, exponent.imag * log_abs);
    let magnitude = scaled_real.exp();

    Ok(ComplexParts {
        real: magnitude * scaled_imag.cos(),
        imag: magnitude * scaled_imag.sin(),
    })
}

#[inline(always)]
fn float_pow_needs_runtime_path(base: f64, exponent: f64) -> bool {
    (base == 0.0 && exponent.is_finite() && exponent < 0.0)
        || (base.is_finite() && base < 0.0 && exponent.is_finite() && !float_is_integral(exponent))
}

#[inline(always)]
fn float_is_integral(value: f64) -> bool {
    value.trunc() == value
}
