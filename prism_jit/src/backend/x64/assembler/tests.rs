use super::*;

#[test]
fn test_assembler_basic() {
    let mut asm = Assembler::new();
    asm.mov_ri64(Gpr::Rax, 42);
    asm.ret();

    let code = asm.finalize().unwrap();
    assert!(!code.is_empty());
}

#[test]
fn test_assembler_labels_backward() {
    let mut asm = Assembler::new();
    let loop_label = asm.create_label();

    asm.bind_label(loop_label);
    asm.add_ri(Gpr::Rax, 1);
    asm.cmp_ri(Gpr::Rax, 10);
    asm.jl(loop_label);
    asm.ret();

    let code = asm.finalize().unwrap();
    assert!(!code.is_empty());
}

#[test]
fn test_assembler_labels_forward() {
    let mut asm = Assembler::new();
    let skip = asm.create_label();

    asm.cmp_ri(Gpr::Rax, 0);
    asm.je(skip);
    asm.mov_ri64(Gpr::Rax, 1);
    asm.bind_label(skip);
    asm.ret();

    let code = asm.finalize().unwrap();
    assert!(!code.is_empty());
}

#[test]
fn test_assembler_mov_optimization() {
    let mut asm = Assembler::new();

    // mov rax, 0 should become xor rax, rax
    asm.mov_ri64(Gpr::Rax, 0);
    let code_zero = asm.code().len();

    // mov rax, 1 should use 32-bit move
    asm.mov_ri64(Gpr::Rax, 1);
    let code_one = asm.code().len() - code_zero;

    // mov rax, imm64 for large values
    asm.mov_ri64(Gpr::Rax, 0x123456789ABCDEF0u64 as i64);
    let code_large = asm.code().len() - code_zero - code_one;

    // xor rax, rax is 3 bytes
    assert!(code_zero <= 4);
    // mov eax, 1 is 5 bytes
    assert!(code_one <= 6);
    // movabs is 10 bytes
    assert_eq!(code_large, 10);
}

#[test]
fn test_constant_pool() {
    let mut pool = ConstantPool::new();

    let idx1 = pool.add_f64(3.125);
    let idx2 = pool.add_f64(3.125); // Should be deduplicated
    let idx3 = pool.add_i64(42);

    assert_eq!(idx1, idx2);
    assert_ne!(idx1, idx3);
    assert_eq!(pool.len(), 2);
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn test_execute_simple() {
    let mut asm = Assembler::new();
    asm.mov_ri64(Gpr::Rax, 42);
    asm.ret();

    let buf = asm.finalize_executable().unwrap();

    type Fn = unsafe extern "C" fn() -> i64;
    let f: Fn = unsafe { buf.as_fn() };
    let result = unsafe { f() };

    assert_eq!(result, 42);
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn test_execute_loop() {
    let mut asm = Assembler::new();
    let loop_label = asm.create_label();
    let exit = asm.create_label();

    // int sum = 0;
    asm.xor_rr(Gpr::Rax, Gpr::Rax);
    // int i = 10;
    asm.mov_ri64(Gpr::Rcx, 10);

    // loop:
    asm.bind_label(loop_label);
    // if (i == 0) goto exit;
    asm.test_rr(Gpr::Rcx, Gpr::Rcx);
    asm.je(exit);
    // sum += i;
    asm.add_rr(Gpr::Rax, Gpr::Rcx);
    // i--;
    asm.dec(Gpr::Rcx);
    // goto loop;
    asm.jmp(loop_label);

    // exit:
    asm.bind_label(exit);
    asm.ret();

    let buf = asm.finalize_executable().unwrap();

    type Fn = unsafe extern "C" fn() -> i64;
    let f: Fn = unsafe { buf.as_fn() };
    let result = unsafe { f() };

    // sum = 10 + 9 + 8 + ... + 1 = 55
    assert_eq!(result, 55);
}

#[test]
fn test_prologue_epilogue() {
    let mut asm = Assembler::new();
    asm.emit_prologue(32, &[Gpr::Rbx, Gpr::R12]);
    asm.mov_ri64(Gpr::Rax, 0);
    asm.emit_epilogue(32, &[Gpr::Rbx, Gpr::R12]);

    let code = asm.finalize().unwrap();
    assert!(!code.is_empty());
}

#[test]
fn test_sse_operations() {
    let mut asm = Assembler::new();
    asm.zero_xmm(Xmm::Xmm0);
    asm.cvtsi2sd(Xmm::Xmm1, Gpr::Rax);
    asm.addsd(Xmm::Xmm0, Xmm::Xmm1);
    asm.mulsd(Xmm::Xmm0, Xmm::Xmm1);
    asm.cvttsd2si(Gpr::Rax, Xmm::Xmm0);
    asm.ret();

    let code = asm.finalize().unwrap();
    assert!(!code.is_empty());
}
