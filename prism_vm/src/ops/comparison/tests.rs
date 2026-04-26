use super::*;
use crate::VirtualMachine;
use crate::builtins::builtin_type_object_for_type_id;
use num_bigint::BigInt;
use prism_code::{CodeObject, Instruction, Opcode, Register};
use prism_compiler::Compiler;
use prism_core::intern::intern;
use prism_parser::parse;
use prism_runtime::object::shape::Shape;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::string::StringObject;
use prism_runtime::types::{DictObject, ListObject, SetObject, TupleObject};
use std::path::Path;
use std::sync::Arc;

fn vm_with_frame() -> VirtualMachine {
    let mut vm = VirtualMachine::new();
    let code = Arc::new(CodeObject::new("test_compare", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");
    vm
}

fn boxed_string_value(value: &str) -> (Value, *mut StringObject) {
    let ptr = Box::into_raw(Box::new(StringObject::new(value)));
    (Value::object_ptr(ptr as *const ()), ptr)
}

fn boxed_object_value<T>(value: T) -> (Value, *mut T) {
    let ptr = Box::into_raw(Box::new(value));
    (Value::object_ptr(ptr as *const ()), ptr)
}

fn promoted_int(value: BigInt) -> Value {
    bigint_to_value(value)
}

fn execute(source: &str) -> Result<Value, String> {
    execute_with_search_paths(source, &[])
}

fn binary_inst(opcode: Opcode) -> Instruction {
    Instruction::op_dss(opcode, Register::new(3), Register::new(1), Register::new(2))
}

fn execute_with_search_paths(source: &str, search_paths: &[&Path]) -> Result<Value, String> {
    let module = parse(source).map_err(|err| format!("parse error: {err:?}"))?;
    let code = Compiler::compile_module(&module, "<comparison-test>")
        .map_err(|err| format!("compile error: {err:?}"))?;

    let mut vm = VirtualMachine::new();
    for path in search_paths {
        let path = Arc::<str>::from(path.to_string_lossy().into_owned());
        vm.import_resolver.add_search_path(path);
    }

    vm.execute(Arc::new(code))
        .map_err(|err| format!("runtime error: {err:?}"))
}

unsafe fn drop_boxed<T>(ptr: *mut T) {
    drop(unsafe { Box::from_raw(ptr) });
}

#[test]
fn test_string_values_equal_across_tagged_and_heap_representations() {
    let tagged = Value::string(intern("inter"));
    let (heap_same, heap_same_ptr) = boxed_string_value("inter");
    let (heap_other, heap_other_ptr) = boxed_string_value("inner");

    assert_eq!(string_values_equal(tagged, tagged), Some(true));
    assert_eq!(string_values_equal(tagged, heap_same), Some(true));
    assert_eq!(string_values_equal(heap_same, tagged), Some(true));
    assert_eq!(string_values_equal(heap_same, heap_other), Some(false));

    unsafe { drop_boxed(heap_same_ptr) };
    unsafe { drop_boxed(heap_other_ptr) };
}

#[test]
fn test_values_equal_uses_string_contents() {
    let left = Value::string(intern("left"));
    let (right, right_ptr) = boxed_string_value("left");
    let (other, other_ptr) = boxed_string_value("right");

    assert!(values_equal(left, right));
    assert!(!values_equal(left, other));

    unsafe { drop_boxed(right_ptr) };
    unsafe { drop_boxed(other_ptr) };
}

#[test]
fn test_values_equal_supports_heap_backed_ints() {
    let big = (BigInt::from(1_u8) << 80_u32) + BigInt::from(7_u8);
    let left = promoted_int(big.clone());
    let right = promoted_int(big.clone());
    let other = promoted_int(big + BigInt::from(1_u8));

    assert!(values_equal(left, right));
    assert!(!values_equal(left, other));
}

#[test]
fn test_eq_and_lt_opcodes_support_heap_backed_ints() {
    let mut eq_vm = vm_with_frame();
    let big = (BigInt::from(1_u8) << 80_u32) + BigInt::from(5_u8);
    eq_vm
        .current_frame_mut()
        .set_reg(1, promoted_int(big.clone()));
    eq_vm
        .current_frame_mut()
        .set_reg(2, promoted_int(big.clone()));

    assert!(matches!(
        eq(&mut eq_vm, binary_inst(Opcode::Eq)),
        ControlFlow::Continue
    ));
    assert_eq!(eq_vm.current_frame().get_reg(3).as_bool(), Some(true));

    let mut lt_vm = vm_with_frame();
    lt_vm.current_frame_mut().set_reg(1, promoted_int(big));
    lt_vm.current_frame_mut().set_reg(
        2,
        promoted_int((BigInt::from(1_u8) << 80_u32) + BigInt::from(6_u8)),
    );

    assert!(matches!(
        lt(&mut lt_vm, binary_inst(Opcode::Lt)),
        ControlFlow::Continue
    ));
    assert_eq!(lt_vm.current_frame().get_reg(3).as_bool(), Some(true));
}

#[test]
fn test_values_equal_uses_bytes_contents_across_bytes_and_bytearray() {
    let (left, left_ptr) = boxed_object_value(BytesObject::from_slice(b"01\n"));
    let (right, right_ptr) = boxed_object_value(BytesObject::from_slice(b"01\n"));
    let (bytearray, bytearray_ptr) = boxed_object_value(BytesObject::bytearray_from_slice(b"01\n"));
    let (other, other_ptr) = boxed_object_value(BytesObject::from_slice(b"00\n"));

    assert!(values_equal(left, right));
    assert!(values_equal(left, bytearray));
    assert!(!values_equal(left, other));

    unsafe { drop_boxed(left_ptr) };
    unsafe { drop_boxed(right_ptr) };
    unsafe { drop_boxed(bytearray_ptr) };
    unsafe { drop_boxed(other_ptr) };
}

#[test]
fn test_execute_supports_bytes_equality_and_ordering() {
    let result = execute(
        r#"
assert b"abc" == b"abc"
assert b"abc" == bytearray(b"abc")
assert b"abc" < b"abd"
assert b"abd" > b"abc"
"#,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_values_equal_uses_list_contents() {
    let (left, left_ptr) = boxed_object_value(ListObject::from_slice(&[
        Value::string(intern("1")),
        Value::string(intern("2")),
        Value::string(intern("3")),
    ]));
    let (right, right_ptr) = boxed_object_value(ListObject::from_slice(&[
        Value::string(intern("1")),
        Value::string(intern("2")),
        Value::string(intern("3")),
    ]));
    let (other, other_ptr) = boxed_object_value(ListObject::from_slice(&[
        Value::string(intern("1")),
        Value::string(intern("2")),
    ]));

    assert!(values_equal(left, right));
    assert!(!values_equal(left, other));

    unsafe { drop_boxed(left_ptr) };
    unsafe { drop_boxed(right_ptr) };
    unsafe { drop_boxed(other_ptr) };
}

#[test]
fn test_values_equal_uses_tuple_contents() {
    let (left, left_ptr) = boxed_object_value(TupleObject::from_slice(&[
        Value::string(intern("1")),
        Value::string(intern("2")),
        Value::string(intern("3")),
    ]));
    let (right, right_ptr) = boxed_object_value(TupleObject::from_slice(&[
        Value::string(intern("1")),
        Value::string(intern("2")),
        Value::string(intern("3")),
    ]));
    let (other, other_ptr) = boxed_object_value(TupleObject::from_slice(&[
        Value::string(intern("1")),
        Value::string(intern("2")),
        Value::string(intern("4")),
    ]));

    assert!(values_equal(left, right));
    assert!(!values_equal(left, other));

    unsafe { drop_boxed(left_ptr) };
    unsafe { drop_boxed(right_ptr) };
    unsafe { drop_boxed(other_ptr) };
}

#[test]
fn test_values_equal_uses_dict_and_set_contents() {
    let mut left_dict = DictObject::new();
    left_dict.set(Value::string(intern("a")), Value::int_unchecked(1));
    left_dict.set(Value::string(intern("b")), Value::int_unchecked(2));
    let mut right_dict = DictObject::new();
    right_dict.set(Value::string(intern("a")), Value::int_unchecked(1));
    right_dict.set(Value::string(intern("b")), Value::int_unchecked(2));
    let mut other_dict = DictObject::new();
    other_dict.set(Value::string(intern("a")), Value::int_unchecked(1));
    other_dict.set(Value::string(intern("b")), Value::int_unchecked(3));

    let (left_dict_value, left_dict_ptr) = boxed_object_value(left_dict);
    let (right_dict_value, right_dict_ptr) = boxed_object_value(right_dict);
    let (other_dict_value, other_dict_ptr) = boxed_object_value(other_dict);

    assert!(values_equal(left_dict_value, right_dict_value));
    assert!(!values_equal(left_dict_value, other_dict_value));

    let (left_set_value, left_set_ptr) = boxed_object_value(SetObject::from_slice(&[
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]));
    let (right_set_value, right_set_ptr) = boxed_object_value(SetObject::from_slice(&[
        Value::int_unchecked(3),
        Value::int_unchecked(2),
        Value::int_unchecked(1),
    ]));
    let (other_set_value, other_set_ptr) = boxed_object_value(SetObject::from_slice(&[
        Value::int_unchecked(1),
        Value::int_unchecked(2),
    ]));

    assert!(values_equal(left_set_value, right_set_value));
    assert!(!values_equal(left_set_value, other_set_value));

    unsafe { drop_boxed(left_dict_ptr) };
    unsafe { drop_boxed(right_dict_ptr) };
    unsafe { drop_boxed(other_dict_ptr) };
    unsafe { drop_boxed(left_set_ptr) };
    unsafe { drop_boxed(right_set_ptr) };
    unsafe { drop_boxed(other_set_ptr) };
}

#[test]
fn test_eq_opcode_supports_string_values() {
    let mut vm = vm_with_frame();
    let left = Value::string(intern("inter"));
    let (right, right_ptr) = boxed_string_value("inter");
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::Eq,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(eq(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

    unsafe { drop_boxed(right_ptr) };
}

#[test]
fn test_is_and_is_not_treat_interned_strings_as_identical() {
    let mut is_vm = vm_with_frame();
    let left = Value::string(intern("decorator-name"));
    let right = Value::string(intern("decorator-name"));
    is_vm.current_frame_mut().set_reg(1, left);
    is_vm.current_frame_mut().set_reg(2, right);

    assert!(matches!(
        is(&mut is_vm, binary_inst(Opcode::Is)),
        ControlFlow::Continue
    ));
    assert_eq!(is_vm.current_frame().get_reg(3), Value::bool(true));

    let mut is_not_vm = vm_with_frame();
    is_not_vm.current_frame_mut().set_reg(1, left);
    is_not_vm.current_frame_mut().set_reg(2, right);

    assert!(matches!(
        is_not(&mut is_not_vm, binary_inst(Opcode::IsNot)),
        ControlFlow::Continue
    ));
    assert_eq!(is_not_vm.current_frame().get_reg(3), Value::bool(false));
}

#[test]
fn test_ne_opcode_supports_string_values() {
    let mut vm = vm_with_frame();
    let left = Value::string(intern("inter"));
    let (right, right_ptr) = boxed_string_value("inner");
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::Ne,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(ne(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

    unsafe { drop_boxed(right_ptr) };
}

#[test]
fn test_lt_opcode_supports_string_values() {
    let mut vm = vm_with_frame();
    let left = Value::string(intern("alpha"));
    let (right, right_ptr) = boxed_string_value("beta");
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::Lt,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(lt(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

    unsafe { drop_boxed(right_ptr) };
}

#[test]
fn test_le_opcode_supports_string_values() {
    let mut vm = vm_with_frame();
    let (left, left_ptr) = boxed_string_value("beta");
    let right = Value::string(intern("beta"));
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::Le,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(le(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

    unsafe { drop_boxed(left_ptr) };
}

#[test]
fn test_gt_opcode_supports_string_values() {
    let mut vm = vm_with_frame();
    let (left, left_ptr) = boxed_string_value("gamma");
    let right = Value::string(intern("beta"));
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::Gt,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(gt(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

    unsafe { drop_boxed(left_ptr) };
}

#[test]
fn test_ge_opcode_supports_string_values() {
    let mut vm = vm_with_frame();
    let left = Value::string(intern("beta"));
    let (right, right_ptr) = boxed_string_value("beta");
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::Ge,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(ge(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

    unsafe { drop_boxed(right_ptr) };
}

#[test]
fn test_bitwise_or_on_type_objects_produces_union_type() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut()
        .set_reg(1, builtin_type_object_for_type_id(TypeId::INT));
    vm.current_frame_mut()
        .set_reg(2, builtin_type_object_for_type_id(TypeId::STR));

    let inst = Instruction::op_dss(
        Opcode::BitwiseOr,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(bitwise_or(&mut vm, inst), ControlFlow::Continue));

    let ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::UNION);
}

#[test]
fn test_bitwise_and_on_sets_returns_intersection() {
    let mut vm = vm_with_frame();
    let (left, left_ptr) = boxed_object_value(SetObject::from_slice(&[
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]));
    let (right, right_ptr) = boxed_object_value(SetObject::from_slice(&[
        Value::int_unchecked(2),
        Value::int_unchecked(3),
        Value::int_unchecked(4),
    ]));
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::BitwiseAnd,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(bitwise_and(&mut vm, inst), ControlFlow::Continue));

    let result_ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
    let result = unsafe { &*(result_ptr as *const SetObject) };
    assert_eq!(
        crate::ops::objects::extract_type_id(result_ptr),
        TypeId::SET
    );
    assert!(result.contains(Value::int_unchecked(2)));
    assert!(result.contains(Value::int_unchecked(3)));
    assert_eq!(result.len(), 2);

    unsafe { drop_boxed(left_ptr) };
    unsafe { drop_boxed(right_ptr) };
}

#[test]
fn test_bitwise_or_on_frozenset_preserves_left_operand_type() {
    let mut vm = vm_with_frame();
    let mut left = SetObject::from_slice(&[Value::int_unchecked(1), Value::int_unchecked(2)]);
    left.header.type_id = TypeId::FROZENSET;
    let mut right = SetObject::from_slice(&[Value::int_unchecked(2), Value::int_unchecked(3)]);
    right.header.type_id = TypeId::FROZENSET;
    let (left, left_ptr) = boxed_object_value(left);
    let (right, right_ptr) = boxed_object_value(right);
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::BitwiseOr,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(bitwise_or(&mut vm, inst), ControlFlow::Continue));

    let result_ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
    let result = unsafe { &*(result_ptr as *const SetObject) };
    assert_eq!(
        crate::ops::objects::extract_type_id(result_ptr),
        TypeId::FROZENSET
    );
    assert!(result.contains(Value::int_unchecked(1)));
    assert!(result.contains(Value::int_unchecked(2)));
    assert!(result.contains(Value::int_unchecked(3)));
    assert_eq!(result.len(), 3);

    unsafe { drop_boxed(left_ptr) };
    unsafe { drop_boxed(right_ptr) };
}

#[test]
fn test_bitwise_or_on_dicts_returns_merged_dict_with_right_overrides() {
    let mut vm = vm_with_frame();
    let mut left = DictObject::new();
    left.set(Value::int_unchecked(1), Value::int_unchecked(10));
    left.set(Value::int_unchecked(2), Value::int_unchecked(20));
    let mut right = DictObject::new();
    right.set(Value::int_unchecked(2), Value::int_unchecked(200));
    right.set(Value::int_unchecked(3), Value::int_unchecked(30));
    let (left, left_ptr) = boxed_object_value(left);
    let (right, right_ptr) = boxed_object_value(right);
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::BitwiseOr,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(bitwise_or(&mut vm, inst), ControlFlow::Continue));

    let result_ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
    let result = unsafe { &*(result_ptr as *const DictObject) };
    let items = result
        .iter()
        .map(|(key, value)| (key.as_int(), value.as_int()))
        .collect::<Vec<_>>();
    assert_eq!(
        items,
        vec![
            (Some(1), Some(10)),
            (Some(2), Some(200)),
            (Some(3), Some(30)),
        ]
    );

    unsafe { drop_boxed(left_ptr) };
    unsafe { drop_boxed(right_ptr) };
}

#[test]
fn test_bitwise_xor_on_sets_returns_symmetric_difference() {
    let mut vm = vm_with_frame();
    let (left, left_ptr) = boxed_object_value(SetObject::from_slice(&[
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]));
    let (right, right_ptr) = boxed_object_value(SetObject::from_slice(&[
        Value::int_unchecked(3),
        Value::int_unchecked(4),
    ]));
    vm.current_frame_mut().set_reg(1, left);
    vm.current_frame_mut().set_reg(2, right);

    let inst = Instruction::op_dss(
        Opcode::BitwiseXor,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(bitwise_xor(&mut vm, inst), ControlFlow::Continue));

    let result_ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
    let result = unsafe { &*(result_ptr as *const SetObject) };
    assert!(result.contains(Value::int_unchecked(1)));
    assert!(result.contains(Value::int_unchecked(2)));
    assert!(result.contains(Value::int_unchecked(4)));
    assert_eq!(result.len(), 3);

    unsafe { drop_boxed(left_ptr) };
    unsafe { drop_boxed(right_ptr) };
}

#[test]
fn test_shl_promotes_large_results_to_heap_backed_ints() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::int(1).unwrap());
    vm.current_frame_mut().set_reg(2, Value::int(1000).unwrap());

    let inst = Instruction::op_dss(
        Opcode::Shl,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(shl(&mut vm, inst), ControlFlow::Continue));

    let result = vm.current_frame().get_reg(3);
    let expected = num_bigint::BigInt::from(1_u8) << 1000_u32;
    assert_eq!(value_to_bigint(result), Some(expected));
}

#[test]
fn test_bitwise_ops_support_heap_backed_ints() {
    let all_ones = (BigInt::from(1_u8) << 128_u32) - BigInt::from(1_u8);
    let low_ones = (BigInt::from(1_u8) << 64_u32) - BigInt::from(1_u8);
    let high_bit = BigInt::from(1_u8) << 127_u32;

    let mut vm = vm_with_frame();
    vm.current_frame_mut()
        .set_reg(1, promoted_int(all_ones.clone()));
    vm.current_frame_mut()
        .set_reg(2, promoted_int(low_ones.clone()));

    assert!(matches!(
        bitwise_xor(&mut vm, binary_inst(Opcode::BitwiseXor)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(vm.current_frame().get_reg(3)),
        Some(all_ones.clone() ^ low_ones.clone())
    );

    vm.current_frame_mut()
        .set_reg(1, promoted_int(all_ones.clone()));
    vm.current_frame_mut()
        .set_reg(2, promoted_int(low_ones.clone()));
    assert!(matches!(
        bitwise_and(&mut vm, binary_inst(Opcode::BitwiseAnd)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(vm.current_frame().get_reg(3)),
        Some(low_ones.clone())
    );

    vm.current_frame_mut()
        .set_reg(1, promoted_int(high_bit.clone()));
    vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());
    assert!(matches!(
        bitwise_or(&mut vm, binary_inst(Opcode::BitwiseOr)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(vm.current_frame().get_reg(3)),
        Some(high_bit.clone() | BigInt::from(3_u8))
    );

    vm.current_frame_mut()
        .set_reg(1, promoted_int(high_bit.clone()));
    let inst = Instruction::op_ds(Opcode::BitwiseNot, Register::new(3), Register::new(1));
    assert!(matches!(bitwise_not(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(
        value_to_bigint(vm.current_frame().get_reg(3)),
        Some(!high_bit)
    );
}

#[test]
fn test_shift_ops_accept_bool_operands_as_ints() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(true));
    vm.current_frame_mut().set_reg(2, Value::int(6).unwrap());
    assert!(matches!(
        shl(&mut vm, binary_inst(Opcode::Shl)),
        ControlFlow::Continue
    ));
    let result = vm.current_frame().get_reg(3);
    assert_eq!(result.as_int(), Some(64));
    assert!(!result.is_bool());

    vm.current_frame_mut().set_reg(1, Value::int(8).unwrap());
    vm.current_frame_mut().set_reg(2, Value::bool(true));
    assert!(matches!(
        shr(&mut vm, binary_inst(Opcode::Shr)),
        ControlFlow::Continue
    ));
    let result = vm.current_frame().get_reg(3);
    assert_eq!(result.as_int(), Some(4));
    assert!(!result.is_bool());
}

#[test]
fn test_contains_value_supports_heap_dict_subclasses_with_native_backing() {
    let mut vm = vm_with_frame();
    let mut instance = ShapedObject::new_dict_backed(TypeId::from_raw(600), Shape::empty());
    instance
        .dict_backing_mut()
        .expect("dict backing should exist")
        .set(Value::string(intern("present")), Value::int(1).unwrap());
    let (value, ptr) = boxed_object_value(instance);

    assert_eq!(
        contains_value(&mut vm, Value::string(intern("present")), value).unwrap(),
        true
    );
    assert_eq!(
        contains_value(&mut vm, Value::string(intern("missing")), value).unwrap(),
        false
    );

    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_contains_value_supports_interned_string_containment() {
    let mut vm = vm_with_frame();

    assert!(
        contains_value(
            &mut vm,
            Value::string(intern("a")),
            Value::string(intern("ab"))
        )
        .unwrap()
    );
    assert!(
        !contains_value(
            &mut vm,
            Value::string(intern("z")),
            Value::string(intern("ab"))
        )
        .unwrap()
    );
}

#[test]
fn test_contains_value_supports_bytes_membership_protocol() {
    let mut vm = vm_with_frame();
    let (haystack, haystack_ptr) = boxed_object_value(BytesObject::from_slice(b"abc"));
    let (needle, needle_ptr) = boxed_object_value(BytesObject::from_slice(b"bc"));

    assert!(contains_value(&mut vm, Value::int_unchecked(97), haystack).unwrap());
    assert!(!contains_value(&mut vm, Value::int_unchecked(122), haystack).unwrap());
    assert!(contains_value(&mut vm, needle, haystack).unwrap());

    unsafe { drop_boxed(needle_ptr) };
    unsafe { drop_boxed(haystack_ptr) };
}

#[test]
fn test_membership_uses_user_defined_contains_protocol() {
    let result = execute(
        r#"
class Bucket:
    def __contains__(self, item):
        return item == 42

bucket = Bucket()
assert 42 in bucket
assert 7 not in bucket
"#,
    );

    assert!(result.is_ok(), "membership protocol failed: {result:?}");
}

#[test]
fn test_membership_falls_back_to_iterator_protocol() {
    let result = execute(
        r#"
class Bucket:
    def __iter__(self):
        return iter((1, 2, 3))

bucket = Bucket()
assert 2 in bucket
assert 5 not in bucket
"#,
    );

    assert!(result.is_ok(), "iterator fallback failed: {result:?}");
}

#[test]
fn test_membership_supports_builtin_dict_key_views() {
    let result = execute(
        r#"
view = {"alpha": 1, "beta": 2}.keys()
assert "alpha" in view
assert "gamma" not in view
"#,
    );

    assert!(result.is_ok(), "dict_keys membership failed: {result:?}");
}

#[test]
fn test_membership_supports_iterator_objects_and_consumes_progress() {
    let result = execute(
        r#"
it = iter([1, 2, 3])
assert 2 in it
assert list(it) == [3]
"#,
    );

    assert!(result.is_ok(), "iterator membership failed: {result:?}");
}

#[test]
fn test_membership_treats_identical_nan_value_as_present() {
    let result = execute(
        r#"
needle = float("nan")
items = [needle]
assert needle in items
"#,
    );

    assert!(result.is_ok(), "nan membership identity failed: {result:?}");
}

#[test]
fn test_bool_ordering_uses_int_subtype_semantics() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(false));
    vm.current_frame_mut().set_reg(2, Value::int(1).unwrap());

    assert!(matches!(
        lt(&mut vm, binary_inst(Opcode::Lt)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));
}

#[test]
fn test_bitwise_and_with_bool_and_int_returns_int() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(true));
    vm.current_frame_mut().set_reg(2, Value::int(1).unwrap());

    assert!(matches!(
        bitwise_and(&mut vm, binary_inst(Opcode::BitwiseAnd)),
        ControlFlow::Continue
    ));
    let result = vm.current_frame().get_reg(3);
    assert_eq!(result.as_int(), Some(1));
    assert!(!result.is_bool());
}

#[test]
fn test_bitwise_xor_with_bool_pair_preserves_bool_result_type() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(true));
    vm.current_frame_mut().set_reg(2, Value::bool(true));

    assert!(matches!(
        bitwise_xor(&mut vm, binary_inst(Opcode::BitwiseXor)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(false));
}

#[test]
fn test_eq_and_ne_use_rich_comparison_protocols() {
    let result = execute(
        r#"
calls = []

class Left:
    def __eq__(self, other):
        calls.append("Left.__eq__")
        return NotImplemented

class Right:
    def __eq__(self, other):
        calls.append("Right.__eq__")
        return True

class OnlyEq:
    def __eq__(self, other):
        return self.value == other.value

class LeftNe:
    def __eq__(self, other):
        calls.append("LeftNe.__eq__")
        return NotImplemented

class RightNe:
    def __eq__(self, other):
        calls.append("RightNe.__eq__")
        return NotImplemented
    def __ne__(self, other):
        calls.append("RightNe.__ne__")
        return NotImplemented

a = Left()
b = Right()
assert a == b
assert calls == ["Left.__eq__", "Right.__eq__"]

x = OnlyEq()
y = OnlyEq()
x.value = 1
y.value = 1
assert (x != y) is False

calls = []
assert LeftNe() != RightNe()
assert calls == ["LeftNe.__eq__", "RightNe.__ne__"]
"#,
    );

    assert!(result.is_ok(), "rich equality protocol failed: {result:?}");
}

#[test]
fn test_native_tuple_subclasses_use_tuple_comparison_semantics() {
    let result = execute(
        r#"
class Pair(tuple):
    pass

left = Pair((1, 2))
same = Pair((1, 2))
larger = Pair((1, 3))

assert left == same
assert not (left != same)
assert left == (1, 2)
assert (1, 2) == left
assert left < larger
assert larger > left
"#,
    );

    assert!(
        result.is_ok(),
        "tuple subclass comparison failed: {result:?}"
    );
}

#[test]
fn test_ordering_uses_rich_comparison_protocols() {
    let result = execute(
        r#"
class Value:
    def __init__(self, value):
        self.value = value
    def __lt__(self, other):
        return self.value < other.value
    def __gt__(self, other):
        return self.value > other.value
    def __le__(self, other):
        return self.value <= other.value
    def __ge__(self, other):
        return self.value >= other.value

assert Value(1) < Value(2)
assert Value(2) > Value(1)
assert Value(2) >= Value(2)
assert Value(2) <= Value(2)
"#,
    );

    assert!(result.is_ok(), "rich ordering protocol failed: {result:?}");
}

#[test]
fn test_membership_falls_back_to_sequence_getitem_protocol() {
    let result = execute(
        r#"
class Bucket:
    def __getitem__(self, index):
        return [1, 2, 3][index]

bucket = Bucket()
assert 2 in bucket
assert 5 not in bucket
"#,
    );

    assert!(
        result.is_ok(),
        "sequence membership fallback failed: {result:?}"
    );
}

#[test]
fn test_tuple_ordering_uses_lexicographic_semantics() {
    let result = execute(
        r#"
assert (1, 2) < (1, 3)
assert (1, 2) <= (1, 2)
assert (2, 0) > (1, 99)
assert (2, 0) >= (2, 0)
"#,
    );

    assert!(result.is_ok(), "tuple ordering failed: {result:?}");
}
