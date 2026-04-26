use super::*;

#[test]
fn test_aot_status_ok() {
    assert!(AotOpStatus::Ok.is_ok());
    assert!(!AotOpStatus::Error.is_ok());
}

#[test]
fn test_aot_string_ref_empty() {
    let string = AotStringRef::empty();
    assert!(string.data.is_null());
    assert_eq!(string.len, 0);
}

#[test]
fn test_native_module_init_entry_layout() {
    let entry = AotNativeModuleInitEntry {
        module_name: AotStringRef::empty(),
        init_fn: core::ptr::null(),
    };
    assert_eq!(entry.module_name.len, 0);
    assert!(entry.init_fn.is_null());
}

#[test]
fn test_aot_immediate_constructors() {
    let bits = AotImmediate::value_bits(123);
    assert_eq!(bits.kind, AotImmediateKind::ValueBits);
    assert_eq!(bits.bits, 123);

    let string = AotImmediate::string(AotStringRef::empty());
    assert_eq!(string.kind, AotImmediateKind::String);
}

#[test]
fn test_aot_operand_constructors() {
    let immediate = AotOperand::immediate(AotImmediate::value_bits(7));
    assert_eq!(immediate.kind, AotOperandKind::Immediate);

    let name = AotOperand::name(AotStringRef::empty());
    assert_eq!(name.kind, AotOperandKind::Name);
}
