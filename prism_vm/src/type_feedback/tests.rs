use super::*;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;

#[test]
fn test_operand_type_classify() {
    assert_eq!(
        OperandType::classify(Value::int(42).unwrap()),
        OperandType::Int
    );
    assert_eq!(
        OperandType::classify(Value::float(3.14)),
        OperandType::Float
    );
    assert_eq!(OperandType::classify(Value::bool(true)), OperandType::Bool);
    assert_eq!(OperandType::classify(Value::none()), OperandType::None);
}

#[test]
fn test_operand_pair_pack() {
    let pair = OperandType::pack(OperandType::Int, OperandType::Float);
    assert_eq!(pair.left(), OperandType::Int);
    assert_eq!(pair.right(), OperandType::Float);
}

#[test]
fn test_operand_pair_constants() {
    assert!(OperandPair::INT_INT.is_int_int());
    assert!(OperandPair::FLOAT_FLOAT.is_float_float());
    assert!(OperandPair::INT_FLOAT.is_mixed_numeric());
    assert!(OperandPair::FLOAT_INT.is_mixed_numeric());
}

#[test]
fn test_operand_pair_from_values() {
    let pair = OperandPair::from_values(Value::int(1).unwrap(), Value::int(2).unwrap());
    assert!(pair.is_int_int());
    assert!(pair.is_numeric());
}

#[test]
fn test_operand_pair_from_tagged_strings() {
    let pair = OperandPair::from_values(Value::string(intern("a")), Value::string(intern("b")));
    assert_eq!(pair, OperandPair::STR_STR);
}

#[test]
fn test_operand_pair_from_lists() {
    let left = Box::into_raw(Box::new(ListObject::new()));
    let right = Box::into_raw(Box::new(ListObject::new()));

    let pair = OperandPair::from_values(
        Value::object_ptr(left as *const ()),
        Value::object_ptr(right as *const ()),
    );
    assert_eq!(pair, OperandPair::LIST_LIST);

    unsafe {
        drop(Box::from_raw(left));
        drop(Box::from_raw(right));
    }
}

#[test]
fn test_operand_pair_numeric_check() {
    // All numeric combinations should pass is_numeric()
    assert!(OperandPair::INT_INT.is_numeric());
    assert!(OperandPair::FLOAT_FLOAT.is_numeric());
    assert!(OperandPair::INT_FLOAT.is_numeric());
    assert!(OperandPair::FLOAT_INT.is_numeric());

    // Non-numeric should fail
    let bool_int = OperandType::pack(OperandType::Bool, OperandType::Int);
    assert!(!bool_int.is_numeric());
}

#[test]
fn test_specialization_query() {
    let mut ic_manager = ICManager::new();

    // Record some int+int accesses
    let site = ICSiteId::new(CodeId::new(1), 0);
    for _ in 0..5 {
        ic_manager.record_binary_op(site, OperandPair::INT_INT.0 as u32);
    }

    let query = TypeSpecializationQuery::new(&ic_manager);
    assert!(query.should_specialize_int(site));
    assert_eq!(query.specialization_for(site), Specialization::Integer);
}

#[test]
fn test_binary_op_feedback() {
    let mut ic_manager = ICManager::new();
    let code_id = CodeId::new(1);

    // Record feedback
    let feedback =
        BinaryOpFeedback::new(code_id, 10, Value::int(1).unwrap(), Value::int(2).unwrap());
    feedback.record(&mut ic_manager);

    // Verify it was recorded
    let query = TypeSpecializationQuery::new(&ic_manager);
    let site = ICSiteId::new(code_id, 10);
    assert!(query.should_specialize_int(site));
}
