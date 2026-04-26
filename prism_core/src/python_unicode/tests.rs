use super::*;

#[test]
fn test_surrogate_roundtrip_uses_internal_carrier_range() {
    let surrogate = 0xDC80;
    let carrier = surrogate_to_carrier(surrogate).expect("surrogate should map");
    assert!(is_surrogate_carrier(carrier));
    assert_eq!(carrier_to_surrogate(carrier), Some(surrogate));
    assert_eq!(logical_python_code_point(carrier), surrogate);
}

#[test]
fn test_encode_python_code_point_accepts_scalar_and_surrogate_inputs() {
    assert_eq!(encode_python_code_point('A' as u32), Some('A'));

    let surrogate = encode_python_code_point(0xDC80).expect("surrogate should map");
    assert_eq!(logical_python_code_point(surrogate as u32), 0xDC80);
}

#[test]
fn test_python_code_point_escape_uses_python_visible_value() {
    assert_eq!(python_code_point_escape(0x41), "\\x41");
    assert_eq!(python_code_point_escape(0x20AC), "\\u20ac");
    assert_eq!(python_code_point_escape(0xDC80), "\\udc80");

    let surrogate = encode_python_code_point(0xDC80).expect("surrogate should map");
    assert_eq!(python_char_escape(surrogate), "\\udc80");
}
