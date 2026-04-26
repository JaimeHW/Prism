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
