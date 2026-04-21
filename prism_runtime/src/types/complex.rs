//! Python complex number object implementation.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use std::fmt;

#[repr(C)]
pub struct ComplexObject {
    pub header: ObjectHeader,
    real: f64,
    imag: f64,
}

impl ComplexObject {
    #[inline]
    pub fn new(real: f64, imag: f64) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::COMPLEX),
            real,
            imag,
        }
    }

    #[inline]
    pub fn real(&self) -> f64 {
        self.real
    }

    #[inline]
    pub fn imag(&self) -> f64 {
        self.imag
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.real == 0.0 && self.imag == 0.0
    }
}

#[inline]
fn format_component(value: f64) -> String {
    if value.is_nan() {
        return if value.is_sign_negative() {
            "-nan".to_string()
        } else {
            "nan".to_string()
        };
    }

    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }

    if value.fract() == 0.0 {
        return format!("{value:.0}");
    }

    value.to_string()
}

fn format_complex(real: f64, imag: f64) -> String {
    let real_is_positive_zero = real == 0.0 && !real.is_sign_negative();
    let imag_text = format_component(imag);

    if real_is_positive_zero {
        return format!("{imag_text}j");
    }

    let real_text = format_component(real);
    let sign = if imag.is_sign_negative() { "" } else { "+" };
    format!("({real_text}{sign}{imag_text}j)")
}

impl fmt::Display for ComplexObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_complex(self.real, self.imag))
    }
}

impl fmt::Debug for ComplexObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComplexObject({})", self)
    }
}

impl PyObject for ComplexObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[cfg(test)]
mod tests {
    use super::ComplexObject;
    use crate::object::PyObject;
    use crate::object::type_obj::TypeId;

    #[test]
    fn test_complex_object_accessors() {
        let complex = ComplexObject::new(1.5, -2.0);
        assert_eq!(complex.header().type_id, TypeId::COMPLEX);
        assert_eq!(complex.real(), 1.5);
        assert_eq!(complex.imag(), -2.0);
        assert!(!complex.is_zero());
    }

    #[test]
    fn test_complex_object_display_matches_python_style() {
        assert_eq!(ComplexObject::new(0.0, 0.0).to_string(), "0j");
        assert_eq!(ComplexObject::new(0.0, 2.0).to_string(), "2j");
        assert_eq!(ComplexObject::new(1.0, 0.0).to_string(), "(1+0j)");
        assert_eq!(ComplexObject::new(1.5, -2.25).to_string(), "(1.5-2.25j)");
    }
}
