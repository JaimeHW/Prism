use prism_core::Value;
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
}

type Entry = unsafe extern "C" fn(*mut TestFrameState) -> u64;

fn run_tier1(registers: &mut [u64], instructions: &[TemplateInstruction]) -> u64 {
    let compiler = TemplateCompiler::new_runtime();
    let compiled = compiler
        .compile(registers.len() as u16, instructions)
        .expect("tier1 compilation should succeed");
    let entry: Entry = unsafe { compiled.as_fn() };
    let mut state = TestFrameState {
        frame_base: registers.as_mut_ptr(),
        num_registers: registers.len() as u16,
        bc_offset: 0,
        const_pool: std::ptr::null(),
        closure_env: std::ptr::null(),
        global_scope: std::ptr::null(),
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
