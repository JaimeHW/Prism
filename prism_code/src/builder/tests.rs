use super::*;

#[test]
fn test_simple_function() {
    let mut builder = FunctionBuilder::new("add");
    builder.set_filename("test.py");
    builder.set_arg_count(2);

    let x = builder.alloc_register();
    let y = builder.alloc_register();
    let result = builder.alloc_register();

    builder.emit_load_local(x, LocalSlot::new(0));
    builder.emit_load_local(y, LocalSlot::new(1));
    builder.emit_add(result, x, y);
    builder.emit_return(result);

    let code = builder.finish();

    assert_eq!(&*code.name, "add");
    assert_eq!(code.instructions.len(), 4);
    assert_eq!(code.register_count, 3);
}

#[test]
fn test_constant_deduplication() {
    let mut builder = FunctionBuilder::new("test");

    let idx1 = builder.add_int(42);
    let idx2 = builder.add_int(42);
    let idx3 = builder.add_int(100);

    assert_eq!(idx1.0, idx2.0); // Same constant, same index
    assert_ne!(idx1.0, idx3.0); // Different constant, different index
}

#[test]
fn test_code_object_constants_are_not_deduplicated_by_name() {
    let mut builder = FunctionBuilder::new("test");

    let mut first = CodeObject::new("same_name", "test.py");
    first.arg_count = 1;
    let mut second = CodeObject::new("same_name", "test.py");
    second.arg_count = 2;

    let first_idx = builder.add_code_object(Arc::new(first));
    let second_idx = builder.add_code_object(Arc::new(second));
    let code = builder.finish();

    assert_ne!(
        first_idx, second_idx,
        "distinct nested functions must keep distinct code constants even when names match"
    );
    assert_eq!(code.nested_code_objects.len(), 2);
    assert_eq!(code.nested_code_objects[0].arg_count, 1);
    assert_eq!(code.nested_code_objects[1].arg_count, 2);
}

#[test]
fn test_labels() {
    let mut builder = FunctionBuilder::new("loop");

    let loop_start = builder.create_label();
    let loop_end = builder.create_label();

    let r0 = builder.alloc_register();

    builder.bind_label(loop_start);
    builder.emit_jump_if_false(r0, loop_end);
    builder.emit_nop();
    builder.emit_jump(loop_start);
    builder.bind_label(loop_end);
    builder.emit_return_none();

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 4);
}

#[test]
fn test_emit_for_iter_preserves_full_i16_jump_offset() {
    let mut builder = FunctionBuilder::new("for_iter_long_jump");
    let pair = builder.alloc_register_block(2);
    let item = Register::new(pair.0 + 1);
    let loop_end = builder.create_label();

    builder.emit_for_iter(item, loop_end);
    for _ in 0..300 {
        builder.emit_nop();
    }
    builder.bind_label(loop_end);
    builder.emit_return_none();

    let code = builder.finish();
    let inst = code.instructions[0];
    assert_eq!(inst.opcode(), Opcode::ForIter as u8);
    assert_eq!(inst.dst().0, item.0);
    assert_eq!(inst.imm16() as i16, 300);
}

#[test]
fn test_register_allocation() {
    let mut builder = FunctionBuilder::new("test");

    let r0 = builder.alloc_register();
    let r1 = builder.alloc_register();
    builder.free_register(r0);
    let r2 = builder.alloc_register(); // Should reuse r0

    assert_eq!(r0.0, 0);
    assert_eq!(r1.0, 1);
    assert_eq!(r2.0, 0); // Reused
}

#[test]
fn test_register_block_allocation_reuses_contiguous_free_span() {
    let mut builder = FunctionBuilder::new("test");

    let block = builder.alloc_register_block(4);
    builder.free_register_block(block, 4);

    let reused = builder.alloc_register_block(3);

    assert_eq!(reused, Register(0));
    assert_eq!(builder.next_register, 4);
}

#[test]
fn test_emit_get_attr_uses_compact_name_index() {
    let mut builder = FunctionBuilder::new("attr");
    let dst = builder.alloc_register();
    let obj = builder.alloc_register();

    builder.emit_get_attr(dst, obj, 42);
    let code = builder.finish();
    let inst = code.instructions[0];

    assert_eq!(code.instructions.len(), 1);
    assert_eq!(inst.opcode(), Opcode::GetAttr as u8);
    assert_eq!(inst.src2().0, 42);
}

#[test]
fn test_emit_get_attr_uses_full_name_index_extension() {
    let mut builder = FunctionBuilder::new("attr");
    let dst = builder.alloc_register();
    let obj = builder.alloc_register();

    builder.emit_get_attr(dst, obj, 0x0123);
    let code = builder.finish();
    let inst = code.instructions[0];
    let ext = code.instructions[1];

    assert_eq!(inst.opcode(), Opcode::GetAttr as u8);
    assert_eq!(inst.src2().0, u8::MAX);
    assert_eq!(ext.opcode(), Opcode::AttrName as u8);
    assert_eq!(ext.imm16(), 0x0123);
}

#[test]
fn test_emit_set_attr_uses_full_name_index_extension() {
    let mut builder = FunctionBuilder::new("set_attr");
    let obj = builder.alloc_register();
    let value = builder.alloc_register();

    builder.emit_set_attr(obj, 0x0123, value);
    let code = builder.finish();
    let inst = code.instructions[0];

    assert_eq!(inst.opcode(), Opcode::SetAttr as u8);
    assert_eq!(inst.dst().0, obj.0);
    assert_eq!(inst.src1().0, u8::MAX);
    assert_eq!(inst.src2().0, value.0);
    assert_eq!(code.instructions[1].opcode(), Opcode::AttrName as u8);
    assert_eq!(code.instructions[1].imm16(), 0x0123);
}

#[test]
fn test_emit_load_method_uses_full_name_index_extension() {
    let mut builder = FunctionBuilder::new("load_method");
    let dst = builder.alloc_register();
    let obj = builder.alloc_register();

    builder.emit_load_method(dst, obj, 0x0123);
    let code = builder.finish();
    let inst = code.instructions[0];
    let ext = code.instructions[1];

    assert_eq!(inst.opcode(), Opcode::LoadMethod as u8);
    assert_eq!(inst.dst().0, dst.0);
    assert_eq!(inst.src1().0, obj.0);
    assert_eq!(inst.src2().0, u8::MAX);
    assert_eq!(ext.opcode(), Opcode::AttrName as u8);
    assert_eq!(ext.imm16(), 0x0123);
}

#[test]
fn test_emit_import_from_uses_full_name_index_extension() {
    let mut builder = FunctionBuilder::new("import_from");
    let dst = builder.alloc_register();
    let module = builder.alloc_register();

    builder.emit_import_from(dst, module, 0x0123);
    let code = builder.finish();
    let inst = code.instructions[0];
    let ext = code.instructions[1];

    assert_eq!(inst.opcode(), Opcode::ImportFrom as u8);
    assert_eq!(inst.dst().0, dst.0);
    assert_eq!(inst.src1().0, module.0);
    assert_eq!(inst.src2().0, u8::MAX);
    assert_eq!(ext.opcode(), Opcode::AttrName as u8);
    assert_eq!(ext.imm16(), 0x0123);
}

#[test]
fn test_emit_set_item_encodes_key_container_and_value_registers() {
    let mut builder = FunctionBuilder::new("set_item");
    let obj = builder.alloc_register();
    let key = builder.alloc_register();
    let value = builder.alloc_register();

    builder.emit_set_item(obj, key, value);
    let code = builder.finish();
    let inst = code.instructions[0];

    assert_eq!(inst.opcode(), Opcode::SetItem as u8);
    assert_eq!(inst.dst().0, key.0);
    assert_eq!(inst.src1().0, obj.0);
    assert_eq!(inst.src2().0, value.0);
}

#[test]
fn test_emit_del_attr_uses_full_name_index_extension() {
    let mut builder = FunctionBuilder::new("del_attr");
    let obj = builder.alloc_register();

    builder.emit_del_attr(obj, 0x0123);
    let code = builder.finish();
    let inst = code.instructions[0];

    assert_eq!(inst.opcode(), Opcode::DelAttr as u8);
    assert_eq!(inst.dst().0, 0);
    assert_eq!(inst.src1().0, obj.0);
    assert_eq!(inst.src2().0, u8::MAX);
    assert_eq!(code.instructions[1].opcode(), Opcode::AttrName as u8);
    assert_eq!(code.instructions[1].imm16(), 0x0123);
}

#[test]
fn test_emit_del_item_encodes_container_and_key_registers() {
    let mut builder = FunctionBuilder::new("del_item");
    let obj = builder.alloc_register();
    let key = builder.alloc_register();

    builder.emit_del_item(obj, key);
    let code = builder.finish();
    let inst = code.instructions[0];

    assert_eq!(inst.opcode(), Opcode::DelItem as u8);
    assert_eq!(inst.dst().0, 0);
    assert_eq!(inst.src1().0, obj.0);
    assert_eq!(inst.src2().0, key.0);
}

#[test]
fn test_emit_build_class_encodes_code_index_and_base_count() {
    let mut builder = FunctionBuilder::new("class_builder");
    let dst = builder.alloc_register();

    builder.emit_build_class(dst, 0x2A, 3);
    let code = builder.finish();
    let inst = code.instructions[0];
    let meta = code.instructions[1];

    assert_eq!(inst.opcode(), Opcode::BuildClass as u8);
    assert_eq!(inst.dst().0, dst.0);
    assert_eq!(inst.imm16(), 0x2A);
    assert_eq!(meta.opcode(), Opcode::ClassMeta as u8);
    assert_eq!(meta.dst().0, 3);
}

#[test]
fn test_emit_build_class_with_meta_encodes_code_index_and_base_count() {
    let mut builder = FunctionBuilder::new("class_builder_meta");
    let dst = builder.alloc_register();

    builder.emit_build_class_with_meta(dst, 0x19, 2);
    let code = builder.finish();
    let inst = code.instructions[0];
    let meta = code.instructions[1];

    assert_eq!(inst.opcode(), Opcode::BuildClassWithMeta as u8);
    assert_eq!(inst.dst().0, dst.0);
    assert_eq!(inst.imm16(), 0x19);
    assert_eq!(meta.opcode(), Opcode::ClassMeta as u8);
    assert_eq!(meta.dst().0, 2);
}

#[test]
fn test_emit_build_class_supports_full_u16_code_index() {
    let mut builder = FunctionBuilder::new("class_builder_u16");
    let dst = builder.alloc_register();
    builder.emit_build_class(dst, 256, 0);
    let code = builder.finish();
    assert_eq!(code.instructions[0].imm16(), 256);
    assert_eq!(code.instructions[1].opcode(), Opcode::ClassMeta as u8);
}

// =========================================================================
// Closure Variable Tests
// =========================================================================

#[test]
fn test_cellvar_registration() {
    let mut builder = FunctionBuilder::new("outer");

    assert!(!builder.has_closure());
    assert_eq!(builder.cellvar_count(), 0);

    let slot0 = builder.add_cellvar("x");
    let slot1 = builder.add_cellvar("y");
    let slot2 = builder.add_cellvar("z");

    assert_eq!(slot0, 0);
    assert_eq!(slot1, 1);
    assert_eq!(slot2, 2);
    assert_eq!(builder.cellvar_count(), 3);
    assert!(builder.has_closure());

    let code = builder.finish();
    assert_eq!(code.cellvars.len(), 3);
    assert_eq!(&*code.cellvars[0], "x");
    assert_eq!(&*code.cellvars[1], "y");
    assert_eq!(&*code.cellvars[2], "z");
}

#[test]
fn test_freevar_registration() {
    let mut builder = FunctionBuilder::new("inner");

    let slot0 = builder.add_freevar("captured");
    let slot1 = builder.add_freevar("outer_var");

    assert_eq!(slot0, 0);
    assert_eq!(slot1, 1);
    assert_eq!(builder.freevar_count(), 2);
    assert!(builder.has_closure());

    let code = builder.finish();
    assert_eq!(code.freevars.len(), 2);
    assert_eq!(&*code.freevars[0], "captured");
    assert_eq!(&*code.freevars[1], "outer_var");
}

#[test]
fn test_mixed_cell_and_free_vars() {
    let mut builder = FunctionBuilder::new("middle");

    let cell0 = builder.add_cellvar("local_captured");
    let cell1 = builder.add_cellvar("another");

    let free0 = builder.add_freevar("from_outer");
    let free1 = builder.add_freevar("also_outer");

    assert_eq!(cell0, 0);
    assert_eq!(cell1, 1);
    assert_eq!(free0, 2); // After 2 cell vars
    assert_eq!(free1, 3);

    assert_eq!(builder.cellvar_count(), 2);
    assert_eq!(builder.freevar_count(), 2);
}

#[test]
fn test_emit_load_closure() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    builder.add_cellvar("x");
    builder.emit_load_closure(r0, 0);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    let inst = code.instructions[0];
    assert_eq!(inst.opcode(), Opcode::LoadClosure as u8);
    assert_eq!(inst.dst().0, r0.0);
    assert_eq!(inst.imm16(), 0);
}

#[test]
fn test_emit_store_closure() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    builder.add_cellvar("x");
    builder.emit_store_closure(0, r0);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    let inst = code.instructions[0];
    assert_eq!(inst.opcode(), Opcode::StoreClosure as u8);
    assert_eq!(inst.imm16(), 0);
}

#[test]
fn test_emit_delete_closure() {
    let mut builder = FunctionBuilder::new("test");

    builder.add_cellvar("x");
    builder.emit_delete_closure(0);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    let inst = code.instructions[0];
    assert_eq!(inst.opcode(), Opcode::DeleteClosure as u8);
    assert_eq!(inst.imm16(), 0);
}

#[test]
fn test_closure_function_pattern() {
    let mut outer = FunctionBuilder::new("make_counter");
    outer.set_arg_count(1);

    let count_slot = outer.add_cellvar("count");
    assert_eq!(count_slot, 0);

    let r0 = outer.alloc_register();
    outer.emit_load_local(r0, LocalSlot::new(0));
    outer.emit_store_closure(count_slot, r0);
    outer.emit_return_none();

    let outer_code = outer.finish();
    assert_eq!(outer_code.cellvars.len(), 1);
    assert_eq!(&*outer_code.cellvars[0], "count");
    assert_eq!(outer_code.instructions.len(), 3);
}

#[test]
fn test_closure_instruction_sequence() {
    let mut builder = FunctionBuilder::new("increment_closure");

    builder.add_cellvar("counter");
    let r0 = builder.alloc_register();
    let r1 = builder.alloc_register();

    builder.emit_load_closure(r0, 0);
    let one_idx = builder.add_int(1);
    builder.emit_load_const(r1, one_idx);
    builder.emit_add(r0, r0, r1);
    builder.emit_store_closure(0, r0);
    builder.emit_return(r0);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 5);

    assert_eq!(code.instructions[0].opcode(), Opcode::LoadClosure as u8);
    assert_eq!(code.instructions[1].opcode(), Opcode::LoadConst as u8);
    assert_eq!(code.instructions[2].opcode(), Opcode::Add as u8);
    assert_eq!(code.instructions[3].opcode(), Opcode::StoreClosure as u8);
    assert_eq!(code.instructions[4].opcode(), Opcode::Return as u8);
}

#[test]
fn test_multiple_closure_slots() {
    let mut builder = FunctionBuilder::new("multi_closure");

    for i in 0..8 {
        let slot = builder.add_cellvar(format!("var{}", i));
        assert_eq!(slot, i as u16);
    }

    let r0 = builder.alloc_register();

    for i in 0..8u16 {
        builder.emit_load_closure(r0, i);
    }

    let code = builder.finish();
    assert_eq!(code.cellvars.len(), 8);
    assert_eq!(code.instructions.len(), 8);

    for i in 0..8 {
        let inst = code.instructions[i];
        assert_eq!(inst.opcode(), Opcode::LoadClosure as u8);
        assert_eq!(inst.imm16(), i as u16);
    }
}

#[test]
fn test_no_closure_by_default() {
    let builder = FunctionBuilder::new("simple");
    assert!(!builder.has_closure());
    assert_eq!(builder.cellvar_count(), 0);
    assert_eq!(builder.freevar_count(), 0);

    let code = builder.finish();
    assert!(code.cellvars.is_empty());
    assert!(code.freevars.is_empty());
}

// =========================================================================
// String Constant Tests
// =========================================================================

#[test]
fn test_add_string_basic() {
    let mut builder = FunctionBuilder::new("test");
    let idx = builder.add_string("hello");
    assert_eq!(idx.0, 0);
}

#[test]
fn test_add_string_deduplication() {
    let mut builder = FunctionBuilder::new("test");

    let idx1 = builder.add_string("hello");
    let idx2 = builder.add_string("hello");
    let idx3 = builder.add_string("world");

    // Same string should return same index
    assert_eq!(idx1.0, idx2.0);
    // Different strings should have different indices
    assert_ne!(idx1.0, idx3.0);
}

#[test]
fn test_add_string_empty() {
    let mut builder = FunctionBuilder::new("test");
    let idx1 = builder.add_string("");
    let idx2 = builder.add_string("");

    assert_eq!(idx1.0, idx2.0);
}

#[test]
fn test_add_string_unicode() {
    let mut builder = FunctionBuilder::new("test");

    let idx1 = builder.add_string("こんにちは");
    let idx2 = builder.add_string("こんにちは");
    let idx3 = builder.add_string("世界");

    assert_eq!(idx1.0, idx2.0);
    assert_ne!(idx1.0, idx3.0);
}

#[test]
fn test_add_string_emoji() {
    let mut builder = FunctionBuilder::new("test");

    let idx1 = builder.add_string("🦀🐍💻");
    let idx2 = builder.add_string("🦀🐍💻");

    assert_eq!(idx1.0, idx2.0);
}

#[test]
fn test_add_string_escape_sequences() {
    let mut builder = FunctionBuilder::new("test");

    let idx1 = builder.add_string("line1\nline2\ttab");
    let idx2 = builder.add_string("line1\nline2\ttab");
    let idx3 = builder.add_string("line1\\nline2\\ttab"); // Different: escaped

    assert_eq!(idx1.0, idx2.0);
    assert_ne!(idx1.0, idx3.0);
}

#[test]
fn test_add_string_whitespace_significant() {
    let mut builder = FunctionBuilder::new("test");

    let idx1 = builder.add_string("hello");
    let idx2 = builder.add_string("hello ");
    let idx3 = builder.add_string(" hello");
    let idx4 = builder.add_string("hello");

    // Whitespace matters
    assert_ne!(idx1.0, idx2.0);
    assert_ne!(idx1.0, idx3.0);
    assert_ne!(idx2.0, idx3.0);
    // Same string should deduplicate
    assert_eq!(idx1.0, idx4.0);
}

#[test]
fn test_add_string_case_sensitive() {
    let mut builder = FunctionBuilder::new("test");

    let idx1 = builder.add_string("Hello");
    let idx2 = builder.add_string("hello");
    let idx3 = builder.add_string("HELLO");

    assert_ne!(idx1.0, idx2.0);
    assert_ne!(idx1.0, idx3.0);
    assert_ne!(idx2.0, idx3.0);
}

#[test]
fn test_add_string_long() {
    let mut builder = FunctionBuilder::new("test");

    let long_string = "x".repeat(10000);
    let idx1 = builder.add_string(&long_string);
    let idx2 = builder.add_string(&long_string);

    assert_eq!(idx1.0, idx2.0);
}

#[test]
fn test_add_string_multiple_distinct() {
    let mut builder = FunctionBuilder::new("test");

    for i in 0..100 {
        let s = format!("string_{}", i);
        let idx = builder.add_string(&s);
        assert_eq!(idx.0, i as u16);
    }
}

#[test]
fn test_add_string_with_load_const() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    let str_idx = builder.add_string("test_string");
    builder.emit_load_const(r0, str_idx);
    builder.emit_return_none();

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 2);
    assert_eq!(code.constants.len(), 1);

    let inst = code.instructions[0];
    assert_eq!(inst.opcode(), Opcode::LoadConst as u8);
    assert_eq!(inst.dst().0, r0.0);
    assert_eq!(inst.imm16(), str_idx.0);
}

#[test]
fn test_add_string_constant_value_is_string() {
    let mut builder = FunctionBuilder::new("test");
    builder.add_string("hello");

    let code = builder.finish();
    assert_eq!(code.constants.len(), 1);

    assert!(matches!(
        &code.constants[0],
        Constant::Value(value) if value.is_string()
    ));
}

#[test]
fn test_mixed_constant_types() {
    let mut builder = FunctionBuilder::new("test");

    let int_idx = builder.add_int(42);
    let str_idx = builder.add_string("hello");
    let float_idx = builder.add_float(3.125);
    let str_idx2 = builder.add_string("world");

    // All should have unique indices
    assert_ne!(int_idx.0, str_idx.0);
    assert_ne!(str_idx.0, float_idx.0);
    assert_ne!(float_idx.0, str_idx2.0);

    let code = builder.finish();
    assert_eq!(code.constants.len(), 4);
}

#[test]
fn test_string_dedup_does_not_affect_other_types() {
    let mut builder = FunctionBuilder::new("test");

    // Add string "42" and int 42 - should not deduplicate
    let str_idx = builder.add_string("42");
    let int_idx = builder.add_int(42);

    assert_ne!(str_idx.0, int_idx.0);

    let code = builder.finish();
    assert_eq!(code.constants.len(), 2);
}

#[test]
fn test_add_int_promotes_wide_i64_to_heap_backed_constant() {
    let mut builder = FunctionBuilder::new("test");

    let value = 2_305_843_009_213_693_952_i64;
    let idx = builder.add_int(value);
    let code = builder.finish();

    assert_eq!(idx.0, 0);
    assert!(matches!(
        &code.constants[0],
        Constant::BigInt(constant) if constant == &BigInt::from(value)
    ));
}

#[test]
fn test_add_bigint_deduplicates_large_constants() {
    let mut builder = FunctionBuilder::new("test");
    let value = BigInt::from(1_u8) << 100_u32;

    let first = builder.add_bigint(value.clone());
    let second = builder.add_bigint(value.clone());
    let code = builder.finish();

    assert_eq!(first.0, second.0);
    assert_eq!(code.constants.len(), 1);
    assert!(matches!(
        &code.constants[0],
        Constant::BigInt(constant) if constant == &value
    ));
}

#[test]
fn test_add_string_from_string_type() {
    let mut builder = FunctionBuilder::new("test");

    let owned_string = String::from("owned_string");
    let idx1 = builder.add_string(&owned_string);
    let idx2 = builder.add_string("owned_string");

    // Should deduplicate even when coming from different source types
    assert_eq!(idx1.0, idx2.0);
}

#[test]
fn test_string_constant_pool_ordering() {
    let mut builder = FunctionBuilder::new("test");

    let idx_a = builder.add_string("aaa");
    let idx_b = builder.add_string("bbb");
    let idx_c = builder.add_string("ccc");

    assert_eq!(idx_a.0, 0);
    assert_eq!(idx_b.0, 1);
    assert_eq!(idx_c.0, 2);

    // Re-adding should return original indices
    assert_eq!(builder.add_string("bbb").0, 1);
    assert_eq!(builder.add_string("aaa").0, 0);
    assert_eq!(builder.add_string("ccc").0, 2);
}
