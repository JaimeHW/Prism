use super::*;

#[test]
fn test_is_match_sequence_type() {
    assert!(is_match_sequence_type(TypeId::LIST));
    assert!(is_match_sequence_type(TypeId::TUPLE));
    assert!(!is_match_sequence_type(TypeId::STR));
    assert!(!is_match_sequence_type(TypeId::DICT));
    assert!(!is_match_sequence_type(TypeId::INT));
}

#[test]
fn test_is_match_mapping_type() {
    assert!(is_match_mapping_type(TypeId::DICT));
    assert!(!is_match_mapping_type(TypeId::LIST));
    assert!(!is_match_mapping_type(TypeId::TUPLE));
    assert!(!is_match_mapping_type(TypeId::STR));
}

#[test]
fn test_get_type_id_tagged_int() {
    let val = Value::int(42).unwrap();
    assert_eq!(get_type_id_from_value(val), Some(TypeId::INT));
}

#[test]
fn test_get_type_id_bool() {
    assert_eq!(
        get_type_id_from_value(Value::bool(true)),
        Some(TypeId::BOOL)
    );
    assert_eq!(
        get_type_id_from_value(Value::bool(false)),
        Some(TypeId::BOOL)
    );
}

#[test]
fn test_get_type_id_none() {
    assert_eq!(get_type_id_from_value(Value::none()), Some(TypeId::NONE));
}
