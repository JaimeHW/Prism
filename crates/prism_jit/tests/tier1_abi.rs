use prism_core::Value;
use prism_core::value::{SMALL_INT_MAX, SMALL_INT_MIN};
use prism_jit::runtime::ExitReason;
use prism_jit::tier1::TemplateCompiler;
use prism_jit::tier1::codegen::TemplateInstruction;

#[repr(C)]
struct TestFrameState {
    frame_base: *mut u64,
    num_registers: u16,
    bc_offset: u32,
    const_pool: *const u64,
    closure_env: *const u64,
    global_scope: *const u64,
    written_registers: *mut u64,
}

type Entry = unsafe extern "C" fn(*mut TestFrameState) -> u64;

fn run_tier1(registers: &mut [u64], instructions: &[TemplateInstruction]) -> u64 {
    run_tier1_at(registers, instructions, 0, &mut [0])
}

fn run_tier1_at(
    registers: &mut [u64],
    instructions: &[TemplateInstruction],
    bc_offset: u32,
    written_registers: &mut [u64],
) -> u64 {
    let compiler = TemplateCompiler::new_runtime();
    let compiled = compiler
        .compile(registers.len() as u16, instructions)
        .expect("tier1 compilation should succeed");
    let entry: Entry = unsafe { compiled.as_fn() };
    let mut state = TestFrameState {
        frame_base: registers.as_mut_ptr(),
        num_registers: registers.len() as u16,
        bc_offset,
        const_pool: std::ptr::null(),
        closure_env: std::ptr::null(),
        global_scope: std::ptr::null(),
        written_registers: written_registers.as_mut_ptr(),
    };

    unsafe { entry(&mut state) }
}

#[test]
fn tier1_returns_value_through_frame_register_zero() {
    let mut registers = vec![Value::none().to_bits(); 4];
    let result = run_tier1(
        &mut registers,
        &[
            TemplateInstruction::LoadInt {
                bc_offset: 0,
                dst: 1,
                value: 2,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 4,
                dst: 2,
                value: 3,
            },
            TemplateInstruction::IntAdd {
                bc_offset: 8,
                dst: 3,
                lhs: 1,
                rhs: 2,
            },
            TemplateInstruction::Return {
                bc_offset: 12,
                value: 3,
            },
        ],
    );

    assert_eq!(result & 0xFF, ExitReason::Return as u64);
    assert_eq!(Value::from_bits(registers[0]).as_int(), Some(5));
}

#[test]
fn tier1_deopt_exit_is_encoded_not_returned_as_value() {
    let mut registers = vec![Value::none().to_bits(), Value::int(1).unwrap().to_bits(), 0];
    let result = run_tier1(
        &mut registers,
        &[
            TemplateInstruction::IntAdd {
                bc_offset: 0,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 2,
            },
        ],
    );

    assert_eq!(result & 0xFF, ExitReason::Deoptimize as u64);
}

#[test]
fn tier1_int_add_deopts_on_small_int_payload_overflow() {
    let mut registers = vec![
        Value::int(SMALL_INT_MAX).unwrap().to_bits(),
        Value::int(1).unwrap().to_bits(),
        Value::none().to_bits(),
    ];
    let result = run_tier1(
        &mut registers,
        &[
            TemplateInstruction::IntAdd {
                bc_offset: 0,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 2,
            },
        ],
    );

    assert_eq!(result & 0xFF, ExitReason::Deoptimize as u64);
}

#[test]
fn tier1_int_neg_deopts_on_small_int_payload_overflow() {
    let mut registers = vec![
        Value::int(SMALL_INT_MIN).unwrap().to_bits(),
        Value::none().to_bits(),
    ];
    let result = run_tier1(
        &mut registers,
        &[
            TemplateInstruction::IntNeg {
                bc_offset: 0,
                dst: 1,
                src: 0,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 1,
            },
        ],
    );

    assert_eq!(result & 0xFF, ExitReason::Deoptimize as u64);
}

#[test]
fn tier1_int_floor_div_uses_python_sign_semantics() {
    let mut registers = vec![
        Value::int(-3).unwrap().to_bits(),
        Value::int(2).unwrap().to_bits(),
        Value::none().to_bits(),
    ];
    let result = run_tier1(
        &mut registers,
        &[
            TemplateInstruction::IntDiv {
                bc_offset: 0,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 2,
            },
        ],
    );

    assert_eq!(result & 0xFF, ExitReason::Return as u64);
    assert_eq!(Value::from_bits(registers[0]).as_int(), Some(-2));
}

#[test]
fn tier1_int_mod_uses_python_sign_semantics() {
    let mut registers = vec![
        Value::int(-3).unwrap().to_bits(),
        Value::int(2).unwrap().to_bits(),
        Value::none().to_bits(),
    ];
    let result = run_tier1(
        &mut registers,
        &[
            TemplateInstruction::IntMod {
                bc_offset: 0,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 2,
            },
        ],
    );

    assert_eq!(result & 0xFF, ExitReason::Return as u64);
    assert_eq!(Value::from_bits(registers[0]).as_int(), Some(1));
}

#[test]
fn tier1_interpreter_fallback_is_an_encoded_deopt() {
    let mut registers = vec![Value::int(1).unwrap().to_bits(); 3];
    let result = run_tier1(
        &mut registers,
        &[TemplateInstruction::GenericEq {
            bc_offset: 0,
            dst: 2,
            lhs: 0,
            rhs: 1,
        }],
    );

    assert_eq!(result & 0xFF, ExitReason::Deoptimize as u64);
}

#[test]
fn tier1_entry_dispatches_to_nonzero_bytecode_offset_for_osr() {
    let mut registers = vec![Value::none().to_bits(); 2];
    let mut written_registers = [0u64; 1];

    let result = run_tier1_at(
        &mut registers,
        &[
            TemplateInstruction::LoadInt {
                bc_offset: 0,
                dst: 0,
                value: 1,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 4,
                dst: 0,
                value: 2,
            },
            TemplateInstruction::Return {
                bc_offset: 8,
                value: 0,
            },
        ],
        4,
        &mut written_registers,
    );

    assert_eq!(result & 0xFF, ExitReason::Return as u64);
    assert_eq!(Value::from_bits(registers[0]).as_int(), Some(2));
}

#[test]
fn tier1_store_local_updates_interpreter_written_bitset() {
    let mut registers = vec![Value::none().to_bits(); 72];
    registers[1] = Value::int(42).unwrap().to_bits();
    let mut written_registers = [0u64; 2];

    let result = run_tier1_at(
        &mut registers,
        &[
            TemplateInstruction::StoreLocal {
                bc_offset: 0,
                src: 1,
                slot: 70,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 70,
            },
        ],
        0,
        &mut written_registers,
    );

    assert_eq!(result & 0xFF, ExitReason::Return as u64);
    assert_eq!(Value::from_bits(registers[70]).as_int(), Some(42));
    assert_ne!(written_registers[1] & (1 << 6), 0);
}

#[test]
fn tier1_delete_local_clears_interpreter_written_bitset() {
    let mut registers = vec![Value::none().to_bits(); 72];
    registers[70] = Value::int(42).unwrap().to_bits();
    let mut written_registers = [0u64, 1 << 6];

    let result = run_tier1_at(
        &mut registers,
        &[
            TemplateInstruction::DeleteLocal {
                bc_offset: 0,
                slot: 70,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 0,
            },
        ],
        0,
        &mut written_registers,
    );

    assert_eq!(result & 0xFF, ExitReason::Return as u64);
    assert_eq!(registers[70], 0);
    assert_eq!(written_registers[1] & (1 << 6), 0);
}
