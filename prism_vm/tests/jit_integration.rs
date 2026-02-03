use prism_compiler::bytecode::{
    CodeFlags, CodeObject, ExceptionEntry, Instruction, LineTableEntry, Opcode, Register,
};
use prism_core::Value;
use prism_vm::{JitConfig, JitContext, VirtualMachine};
use std::sync::Arc;

// Helper to create a simple code object that returns a constant
fn create_return_const_code(const_val: Value) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // LOAD_CONST 0 -> r0
    constants.push(const_val);
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // RETURN r0
    instructions.push(Instruction::op_d(Opcode::Return, Register(0)));

    Arc::new(CodeObject {
        name: "test_return_const".into(),
        register_count: 1,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        exception_table: Box::new([]),
        filename: "test.py".into(),
        qualname: "test_return_const".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
        nested_code_objects: Box::new([]),
    })
}

#[test]
fn test_jit_simple_execution() {
    // Configure JIT for testing (eager compilation, low thresholds)
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let expected = Value::int(42).unwrap();
    let code = create_return_const_code(expected.clone());

    // Run multiple times to trigger tier-up (threshold is 10)
    for _ in 0..15 {
        let result = vm.execute(Arc::clone(&code)).unwrap();
        assert_eq!(result.as_int(), expected.as_int());
    }
}

// Helper for a loop code object:
// def test_loop():
//     n = 100
//     i = 0
//     while i < n:
//         i = i + 1
//     return i
fn create_loop_code() -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // r1: n (constant 100)
    // r2: i
    // r3: temp constant 1
    // r4: comparison result

    // 0: LOAD_CONST 0 (100) -> r1
    constants.push(Value::int(100).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 0));

    // 1: LOAD_CONST 1 (0) -> r2 (i = 0)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 1));

    // Loop header (Label L1)
    // 2: LOAD_CONST 2 (1) -> r3 (const 1)
    // Note: Moving LoadConst out of loop body would be optimization, but for test logic
    // we keep it here to match typical unoptimized bytecode.
    // Wait, constant load is cheap, but let's just do it.
    constants.push(Value::int(1).unwrap());
    let loop_header_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(3), 2));

    // 3: Compare i < n (LT r2, r1 -> r4)
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(4),
        Register(2),
        Register(1),
    ));

    // 4: JUMP_IF_FALSE r4, exit_target
    let jump_if_false_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(4), 0));

    // Body:
    // 5: ADD r2, r3 -> r2 (i = i + 1)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(2),
        Register(2),
        Register(3),
    ));

    // 6: JUMP loop_header
    let jump_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Fixup backward jump
    let next_ip = jump_back_idx + 1;
    let offset = (loop_header_idx as i32) - (next_ip as i32);
    instructions[jump_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    // Exit target (Label L2)
    let exit_idx = instructions.len();

    // Fixup forward jump
    let next_ip_fwd = jump_if_false_idx + 1;
    let fwd_offset = (exit_idx as i32) - (next_ip_fwd as i32);
    instructions[jump_if_false_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(4), fwd_offset as i16 as u16);

    // RETURN r2
    instructions.push(Instruction::op_d(Opcode::Return, Register(2)));

    Arc::new(CodeObject {
        name: "test_loop_local".into(),
        register_count: 5,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        exception_table: Box::new([]),
        filename: "test.py".into(),
        qualname: "test_loop_local".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
        nested_code_objects: Box::new([]),
    })
}

#[test]
fn test_jit_loop_execution_and_osr() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let code = create_loop_code();

    // Execute loop with n=100
    // This should trigger OSR at loop header since loop iterations > threshold
    let result = vm.execute(code).unwrap();
    assert_eq!(result.as_int(), Some(100));
}
