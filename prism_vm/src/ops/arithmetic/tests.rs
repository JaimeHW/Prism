use super::{add, floor_div, modulo, mul, neg, pos, pos_int, pow, sub, true_div};
use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use num_bigint::BigInt;
use prism_code::{CodeObject, Instruction, Opcode, Register};
use prism_compiler::Compiler;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_core::value::SMALL_INT_MAX;
use prism_parser::parse;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use std::sync::Arc;

fn vm_with_frame() -> VirtualMachine {
    let mut code = CodeObject::new("test_add", "<test>");
    code.register_count = 16;
    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0).expect("frame push failed");
    vm
}

fn value_to_rust_string(value: Value) -> String {
    let string = value_as_string_ref(value).expect("value should be a Python string");
    string.as_str().to_string()
}

fn value_to_byte_vec(value: Value) -> Vec<u8> {
    let ptr = value
        .as_object_ptr()
        .expect("byte sequence should be object-backed");
    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert!(matches!(header.type_id, TypeId::BYTES | TypeId::BYTEARRAY));
    let bytes = unsafe { &*(ptr as *const BytesObject) };
    bytes.as_bytes().to_vec()
}

fn byte_sequence_type(value: Value) -> TypeId {
    let ptr = value
        .as_object_ptr()
        .expect("byte sequence should be object-backed");
    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    header.type_id
}

fn binary_inst(opcode: Opcode) -> Instruction {
    Instruction::op_dss(opcode, Register::new(0), Register::new(1), Register::new(2))
}

fn unary_inst(opcode: Opcode) -> Instruction {
    Instruction::op_ds(opcode, Register::new(0), Register::new(1))
}

fn promoted_int(value: BigInt) -> Value {
    bigint_to_value(value)
}

fn execute(source: &str) -> Result<Value, String> {
    let module = parse(source).map_err(|err| format!("parse error: {err:?}"))?;
    let code = Compiler::compile_module(&module, "<arithmetic-test>")
        .map_err(|err| format!("compile error: {err:?}"))?;

    let mut vm = VirtualMachine::new();
    vm.execute(Arc::new(code))
        .map_err(|err| format!("runtime error: {err:?}"))
}

#[test]
fn test_add_concatenates_tuples() {
    let mut vm = vm_with_frame();
    let left_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])));
    let right_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ])));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(left_ptr as *const ()));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(right_ptr as *const ()));

    let inst = Instruction::op_dss(
        Opcode::Add,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(add(&mut vm, inst), ControlFlow::Continue));

    let result_ptr = vm
        .current_frame()
        .get_reg(0)
        .as_object_ptr()
        .expect("tuple concat should return tuple object");
    let result = unsafe { &*(result_ptr as *const TupleObject) };
    assert_eq!(result.len(), 4);
    assert_eq!(result.get(0).unwrap().as_int(), Some(1));
    assert_eq!(result.get(3).unwrap().as_int(), Some(4));

    unsafe {
        drop(Box::from_raw(left_ptr));
        drop(Box::from_raw(right_ptr));
    }
}

#[test]
fn test_add_concatenates_tagged_strings() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut()
        .set_reg(1, Value::string(intern("hello")));
    vm.current_frame_mut()
        .set_reg(2, Value::string(intern(" world")));

    let inst = Instruction::op_dss(
        Opcode::Add,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(add(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(
        value_to_rust_string(vm.current_frame().get_reg(0)),
        "hello world"
    );
}

#[test]
fn test_add_promotes_inline_integer_overflow_to_heap_int() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(
        1,
        Value::int(SMALL_INT_MAX).expect("small int max fits inline"),
    );
    vm.current_frame_mut()
        .set_reg(2, Value::int(1).expect("small int fits inline"));

    assert!(matches!(
        add(&mut vm, binary_inst(Opcode::Add)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(vm.current_frame().get_reg(0)),
        Some(BigInt::from(SMALL_INT_MAX) + BigInt::from(1_i64))
    );
}

#[test]
fn test_sub_supports_heap_backed_integers() {
    let mut vm = vm_with_frame();
    let left = (BigInt::from(1_u8) << 80_u32) + BigInt::from(9_u8);
    vm.current_frame_mut()
        .set_reg(1, promoted_int(left.clone()));
    vm.current_frame_mut()
        .set_reg(2, Value::int(4).expect("small int fits inline"));

    assert!(matches!(
        sub(&mut vm, binary_inst(Opcode::Sub)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(vm.current_frame().get_reg(0)),
        Some(left - BigInt::from(4_u8))
    );
}

#[test]
fn test_sub_on_sets_returns_difference_with_left_operand_type() {
    let mut vm = vm_with_frame();
    let mut left_set = SetObject::from_slice(&[
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]);
    left_set.header.type_id = TypeId::FROZENSET;
    let right_set = SetObject::from_slice(&[Value::int_unchecked(2), Value::int_unchecked(4)]);
    let left_ptr = Box::into_raw(Box::new(left_set));
    let right_ptr = Box::into_raw(Box::new(right_set));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(left_ptr as *const ()));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(right_ptr as *const ()));

    assert!(matches!(
        sub(&mut vm, binary_inst(Opcode::Sub)),
        ControlFlow::Continue
    ));

    let result_ptr = vm.current_frame().get_reg(0).as_object_ptr().unwrap();
    let result = unsafe { &*(result_ptr as *const SetObject) };
    assert_eq!(
        crate::ops::objects::extract_type_id(result_ptr),
        TypeId::FROZENSET
    );
    assert!(result.contains(Value::int_unchecked(1)));
    assert!(result.contains(Value::int_unchecked(3)));
    assert_eq!(result.len(), 2);
}

#[test]
fn test_mul_supports_heap_backed_integers() {
    let mut vm = vm_with_frame();
    let left = (BigInt::from(1_u8) << 72_u32) + BigInt::from(3_u8);
    vm.current_frame_mut()
        .set_reg(1, promoted_int(left.clone()));
    vm.current_frame_mut()
        .set_reg(2, Value::int(8).expect("small int fits inline"));

    assert!(matches!(
        mul(&mut vm, binary_inst(Opcode::Mul)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(vm.current_frame().get_reg(0)),
        Some(left * BigInt::from(8_u8))
    );
}

#[test]
fn test_modulo_and_floor_div_support_heap_backed_integers() {
    let dividend = -((BigInt::from(1_u8) << 80_u32) + BigInt::from(5_u8));

    let mut div_vm = vm_with_frame();
    div_vm
        .current_frame_mut()
        .set_reg(1, promoted_int(dividend.clone()));
    div_vm
        .current_frame_mut()
        .set_reg(2, Value::int(8).expect("small int fits inline"));

    assert!(matches!(
        super::floor_div(&mut div_vm, binary_inst(Opcode::FloorDiv)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(div_vm.current_frame().get_reg(0)),
        Some(-((BigInt::from(1_u8) << 77_u32) + BigInt::from(1_u8)))
    );

    let mut mod_vm = vm_with_frame();
    mod_vm
        .current_frame_mut()
        .set_reg(1, promoted_int(dividend));
    mod_vm
        .current_frame_mut()
        .set_reg(2, Value::int(8).expect("small int fits inline"));

    assert!(matches!(
        modulo(&mut mod_vm, binary_inst(Opcode::Mod)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(mod_vm.current_frame().get_reg(0)),
        Some(BigInt::from(3_u8))
    );
}

#[test]
fn test_pow_promotes_large_integer_results() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut()
        .set_reg(1, Value::int(2).expect("small int fits inline"));
    vm.current_frame_mut()
        .set_reg(2, Value::int(100).expect("small int fits inline"));

    assert!(matches!(
        pow(&mut vm, binary_inst(Opcode::Pow)),
        ControlFlow::Continue
    ));
    assert_eq!(
        value_to_bigint(vm.current_frame().get_reg(0)),
        Some(BigInt::from(1_u8) << 100_u32)
    );
}

#[test]
fn test_pos_preserves_heap_backed_int_value() {
    let mut vm = vm_with_frame();
    let value = promoted_int(BigInt::from(1_u8) << 90_u32);
    vm.current_frame_mut().set_reg(1, value);

    assert!(matches!(
        pos(&mut vm, unary_inst(Opcode::Pos)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0), value);
}

#[test]
fn test_add_concatenates_bytes() {
    let mut vm = vm_with_frame();
    let left_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"abc")));
    let right_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"def")));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(left_ptr as *const ()));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(right_ptr as *const ()));

    assert!(matches!(
        add(&mut vm, binary_inst(Opcode::Add)),
        ControlFlow::Continue
    ));

    let result = vm.current_frame().get_reg(0);
    assert_eq!(byte_sequence_type(result), TypeId::BYTES);
    assert_eq!(value_to_byte_vec(result), b"abcdef");

    unsafe {
        drop(Box::from_raw(left_ptr));
        drop(Box::from_raw(right_ptr));
    }
}

#[test]
fn test_add_concatenates_mixed_byte_sequences_using_left_operand_type() {
    let mut vm = vm_with_frame();
    let left_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"abc")));
    let right_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"def")));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(left_ptr as *const ()));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(right_ptr as *const ()));

    assert!(matches!(
        add(&mut vm, binary_inst(Opcode::Add)),
        ControlFlow::Continue
    ));

    let first_result = vm.current_frame().get_reg(0);
    assert_eq!(byte_sequence_type(first_result), TypeId::BYTES);
    assert_eq!(value_to_byte_vec(first_result), b"abcdef");

    let second_left_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"ghi")));
    let second_right_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"jkl")));
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(second_left_ptr as *const ()));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(second_right_ptr as *const ()));

    assert!(matches!(
        add(&mut vm, binary_inst(Opcode::Add)),
        ControlFlow::Continue
    ));

    let second_result = vm.current_frame().get_reg(0);
    assert_eq!(byte_sequence_type(second_result), TypeId::BYTEARRAY);
    assert_eq!(value_to_byte_vec(second_result), b"ghijkl");

    unsafe {
        drop(Box::from_raw(left_ptr));
        drop(Box::from_raw(right_ptr));
        drop(Box::from_raw(second_left_ptr));
        drop(Box::from_raw(second_right_ptr));
    }
}

#[test]
fn test_mul_repeats_tagged_strings() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut()
        .set_reg(1, Value::string(intern("ab")));
    vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());

    let inst = Instruction::op_dss(
        Opcode::Mul,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(
        value_to_rust_string(vm.current_frame().get_reg(0)),
        "ababab"
    );
}

#[test]
fn test_mul_negative_string_repeat_returns_empty() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut()
        .set_reg(1, Value::string(intern("ab")));
    vm.current_frame_mut().set_reg(2, Value::int(-1).unwrap());

    let inst = Instruction::op_dss(
        Opcode::Mul,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));

    let result = vm.current_frame().get_reg(0);
    assert!(result.is_string());
    let ptr = result
        .as_string_object_ptr()
        .expect("empty repeat should return tagged empty string") as *const u8;
    assert_eq!(
        interned_by_ptr(ptr)
            .expect("empty repeat result should resolve through the interner")
            .as_str(),
        ""
    );
}

#[test]
fn test_mul_repeats_tuples() {
    let mut vm = vm_with_frame();
    let tuple_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(tuple_ptr as *const ()));
    vm.current_frame_mut().set_reg(2, Value::int(2).unwrap());

    let inst = Instruction::op_dss(
        Opcode::Mul,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));

    let result_ptr = vm
        .current_frame()
        .get_reg(0)
        .as_object_ptr()
        .expect("tuple repeat should return tuple object");
    let result = unsafe { &*(result_ptr as *const TupleObject) };
    assert_eq!(
        result.as_slice(),
        &[
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(1),
            Value::int_unchecked(2)
        ]
    );

    unsafe {
        drop(Box::from_raw(tuple_ptr));
    }
}

#[test]
fn test_mul_zero_repeats_tuples_to_empty() {
    let mut vm = vm_with_frame();
    let tuple_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(tuple_ptr as *const ()));
    vm.current_frame_mut().set_reg(2, Value::int(0).unwrap());

    let inst = Instruction::op_dss(
        Opcode::Mul,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));

    let result_ptr = vm
        .current_frame()
        .get_reg(0)
        .as_object_ptr()
        .expect("tuple repeat should return tuple object");
    let result = unsafe { &*(result_ptr as *const TupleObject) };
    assert!(result.is_empty());

    unsafe {
        drop(Box::from_raw(tuple_ptr));
    }
}

#[test]
fn test_mul_repeats_lists() {
    let mut vm = vm_with_frame();
    let list_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(list_ptr as *const ()));
    vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());

    let inst = Instruction::op_dss(
        Opcode::Mul,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));

    let result_ptr = vm
        .current_frame()
        .get_reg(0)
        .as_object_ptr()
        .expect("list repeat should return list object");
    let result = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(
        result.as_slice(),
        &[
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
        ]
    );

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_mul_repeats_bytes() {
    let mut vm = vm_with_frame();
    let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"ab")));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(bytes_ptr as *const ()));
    vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());

    assert!(matches!(
        mul(&mut vm, binary_inst(Opcode::Mul)),
        ControlFlow::Continue
    ));

    let result = vm.current_frame().get_reg(0);
    assert_eq!(byte_sequence_type(result), TypeId::BYTES);
    assert_eq!(value_to_byte_vec(result), b"ababab");

    unsafe {
        drop(Box::from_raw(bytes_ptr));
    }
}

#[test]
fn test_mul_repeats_bytearray_and_preserves_type() {
    let mut vm = vm_with_frame();
    let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"xy")));

    vm.current_frame_mut().set_reg(1, Value::int(2).unwrap());
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(bytearray_ptr as *const ()));

    assert!(matches!(
        mul(&mut vm, binary_inst(Opcode::Mul)),
        ControlFlow::Continue
    ));

    let result = vm.current_frame().get_reg(0);
    assert_eq!(byte_sequence_type(result), TypeId::BYTEARRAY);
    assert_eq!(value_to_byte_vec(result), b"xyxy");

    unsafe {
        drop(Box::from_raw(bytearray_ptr));
    }
}

#[test]
fn test_mul_negative_byte_repeat_returns_empty_same_type() {
    let mut vm = vm_with_frame();
    let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"xy")));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(bytearray_ptr as *const ()));
    vm.current_frame_mut().set_reg(2, Value::int(-1).unwrap());

    assert!(matches!(
        mul(&mut vm, binary_inst(Opcode::Mul)),
        ControlFlow::Continue
    ));

    let result = vm.current_frame().get_reg(0);
    assert_eq!(byte_sequence_type(result), TypeId::BYTEARRAY);
    assert_eq!(value_to_byte_vec(result), b"");

    unsafe {
        drop(Box::from_raw(bytearray_ptr));
    }
}

#[test]
fn test_modulo_formats_tagged_string_with_single_argument() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut()
        .set_reg(1, Value::string(intern("hello %s")));
    vm.current_frame_mut()
        .set_reg(2, Value::string(intern("world")));

    let inst = Instruction::op_dss(
        Opcode::Mod,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(modulo(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(
        value_to_rust_string(vm.current_frame().get_reg(0)),
        "hello world"
    );
}

#[test]
fn test_modulo_formats_tagged_string_with_tuple_arguments() {
    let mut vm = vm_with_frame();
    let tuple_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::string(intern("value")),
        Value::int(7).unwrap(),
    ])));
    vm.current_frame_mut()
        .set_reg(1, Value::string(intern("%s = %d")));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(tuple_ptr as *const ()));

    let inst = Instruction::op_dss(
        Opcode::Mod,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(modulo(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(
        value_to_rust_string(vm.current_frame().get_reg(0)),
        "value = 7"
    );

    unsafe {
        drop(Box::from_raw(tuple_ptr));
    }
}

#[test]
fn test_modulo_formats_bytes_with_raw_byte_argument() {
    let mut vm = vm_with_frame();
    let template_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"[xxx%sxxx]")));
    let argument_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"libc.so.1.2.5")));
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(template_ptr as *const ()));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(argument_ptr as *const ()));

    assert!(matches!(
        modulo(&mut vm, binary_inst(Opcode::Mod)),
        ControlFlow::Continue
    ));

    let result = vm.current_frame().get_reg(0);
    assert_eq!(byte_sequence_type(result), TypeId::BYTES);
    assert_eq!(value_to_byte_vec(result), b"[xxxlibc.so.1.2.5xxx]");

    unsafe {
        drop(Box::from_raw(template_ptr));
        drop(Box::from_raw(argument_ptr));
    }
}

#[test]
fn test_bool_sub_uses_int_semantics() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(true));
    vm.current_frame_mut().set_reg(2, Value::bool(false));

    assert!(matches!(
        sub(&mut vm, binary_inst(Opcode::Sub)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(1));
}

#[test]
fn test_bool_true_div_uses_real_number_semantics() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(true));
    vm.current_frame_mut().set_reg(2, Value::int(1).unwrap());

    assert!(matches!(
        true_div(&mut vm, binary_inst(Opcode::TrueDiv)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0).as_float(), Some(1.0));
}

#[test]
fn test_bool_pow_returns_tagged_int_result() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(true));
    vm.current_frame_mut().set_reg(2, Value::int(2).unwrap());

    assert!(matches!(
        pow(&mut vm, binary_inst(Opcode::Pow)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(1));
}

#[test]
fn test_bool_neg_returns_int() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(true));

    assert!(matches!(
        neg(&mut vm, unary_inst(Opcode::Neg)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(-1));
}

#[test]
fn test_unary_neg_uses_user_defined_special_method() {
    let result = execute(
        r#"
class Delta:
    def __init__(self, value):
        self.value = value

    def __neg__(self):
        return Delta(-self.value)

class Zone:
    max = Delta(24)
    min = -max

assert Zone.min.value == -24
"#,
    );

    assert!(result.is_ok(), "unary __neg__ dispatch failed: {result:?}");
}

#[test]
fn test_bool_pos_int_returns_int_not_bool() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(false));

    assert!(matches!(
        pos_int(&mut vm, unary_inst(Opcode::PosInt)),
        ControlFlow::Continue
    ));
    let result = vm.current_frame().get_reg(0);
    assert_eq!(result.as_int(), Some(0));
    assert!(!result.is_bool());
}

#[test]
fn test_bool_generic_pos_returns_int_not_bool() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::bool(true));

    assert!(matches!(
        pos(&mut vm, unary_inst(Opcode::Pos)),
        ControlFlow::Continue
    ));
    let result = vm.current_frame().get_reg(0);
    assert_eq!(result.as_int(), Some(1));
    assert!(!result.is_bool());
}

#[test]
fn test_binary_ops_use_user_defined_special_methods() {
    let result = execute(
        r#"
class Left:
    def __sub__(self, other):
        return 7

class Right:
    def __rtruediv__(self, other):
        return 11

class Mul:
    def __mul__(self, other):
        return 13

class Floor:
    def __rfloordiv__(self, other):
        return 17

class Mod:
    def __rmod__(self, other):
        return 19

assert Left() - 2 == 7
assert 5 / Right() == 11
assert Mul() * 3 == 13
assert 23 // Floor() == 17
assert 29 % Mod() == 19
"#,
    );

    assert!(result.is_ok(), "special-method dispatch failed: {result:?}");
}

#[test]
fn test_binary_ops_prefer_reflected_methods_for_proper_subtypes() {
    let result = execute(
        r#"
calls = []

class Base:
    def __add__(self, other):
        calls.append("Base.__add__")
        return NotImplemented

class Derived(Base):
    def __radd__(self, other):
        calls.append("Derived.__radd__")
        return 41

assert Base() + Derived() == 41
assert calls == ["Derived.__radd__"]
"#,
    );

    assert!(
        result.is_ok(),
        "reflected dispatch ordering failed: {result:?}"
    );
}

#[test]
fn test_floor_div_and_mod_follow_python_sign_rules() {
    let mut vm = vm_with_frame();

    vm.current_frame_mut().set_reg(1, Value::int(10).unwrap());
    vm.current_frame_mut().set_reg(2, Value::int(-3).unwrap());
    assert!(matches!(
        floor_div(&mut vm, binary_inst(Opcode::FloorDiv)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(-4));

    assert!(matches!(
        modulo(&mut vm, binary_inst(Opcode::Mod)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(-2));

    vm.current_frame_mut().set_reg(1, Value::int(-10).unwrap());
    vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());
    assert!(matches!(
        floor_div(&mut vm, binary_inst(Opcode::FloorDiv)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(-4));

    assert!(matches!(
        modulo(&mut vm, binary_inst(Opcode::Mod)),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(2));
}
