//! Shared helpers for Python's numeric tower semantics.
//!
//! CPython models `bool` as a proper subtype of `int`. Prism keeps bool and int
//! as distinct tagged values for representation efficiency, so operator and
//! builtin implementations use these helpers when they need Python-compatible
//! numeric coercion.

use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::int::value_to_i64;

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
    value
        .as_float()
        .or_else(|| int_like_value(value).map(|v| v as f64))
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
    if let Some(float_value) = value.as_float() {
        return Some(ComplexParts {
            real: float_value,
            imag: 0.0,
        });
    }

    if let Some(int_value) = int_like_value(value) {
        return Some(ComplexParts {
            real: int_value as f64,
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
