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

#[inline(always)]
fn float_pow_needs_runtime_path(base: f64, exponent: f64) -> bool {
    (base == 0.0 && exponent.is_finite() && exponent < 0.0)
        || (base.is_finite() && base < 0.0 && exponent.is_finite() && !float_is_integral(exponent))
}

#[inline(always)]
fn float_is_integral(value: f64) -> bool {
    value.trunc() == value
}
