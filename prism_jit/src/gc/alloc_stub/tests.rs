use super::*;

#[test]
fn test_tlab_offsets_default() {
    let offsets = TlabOffsets::default();
    assert_eq!(offsets.ptr_offset, 0);
    assert_eq!(offsets.end_offset, 8);
}

#[test]
fn test_alloc_stub_new() {
    let stub = AllocStub::new();
    assert_eq!(stub.thread_reg(), Gpr::R14);
}

#[test]
fn test_alloc_stub_custom_config() {
    let offsets = TlabOffsets {
        ptr_offset: 16,
        end_offset: 24,
    };
    let stub = AllocStub::with_config(offsets, Gpr::R13);
    assert_eq!(stub.thread_reg(), Gpr::R13);
    assert_eq!(stub.offsets.ptr_offset, 16);
}

#[test]
fn test_emit_fast_path_const_size() {
    let stub = AllocStub::new();
    let mut asm = Assembler::new();

    let slow_path_offset = stub.emit_fast_path_const_size(
        &mut asm,
        64, // 64 bytes
        Gpr::Rax,
        Gpr::Rcx,
    );

    // Should have emitted some code
    assert!(asm.offset() > 0);
    // Should have returned an offset for patching
    assert!(slow_path_offset > 0);
}

#[test]
fn test_emit_fast_path_dynamic_size() {
    let stub = AllocStub::new();
    let mut asm = Assembler::new();

    let slow_path_offset = stub.emit_fast_path_dynamic_size(
        &mut asm,
        Gpr::Rcx, // size in RCX
        Gpr::Rax, // result in RAX
    );

    assert!(asm.offset() > 0);
    assert!(slow_path_offset > 0);
}

#[test]
fn test_size_alignment() {
    // Check our alignment logic
    let sizes = [1, 7, 8, 9, 15, 16, 17, 63, 64, 65];
    let expected = [8, 8, 8, 16, 16, 16, 24, 64, 64, 72];

    for (size, aligned) in sizes.iter().zip(expected.iter()) {
        let result = (*size + 7) & !7;
        assert_eq!(result, *aligned, "Alignment failed for size {}", size);
    }
}

#[test]
fn test_emit_slow_path() {
    let stub = AllocStub::new();
    let mut asm = Assembler::new();

    // Mock slow path function address
    let slow_path_fn = 0x12345678usize;

    stub.emit_slow_path(&mut asm, Gpr::Rcx, Gpr::Rax, slow_path_fn);

    assert!(asm.offset() > 0);
}

#[test]
fn test_patch_slow_path_offset() {
    // Test the patching logic
    let mut code = vec![0u8; 16];
    // Set up bytes like: [JA opcode bytes] [placeholder]
    code[0] = 0x0F;
    code[1] = 0x87;
    // Placeholder for offset at bytes 2-5

    // Patch to jump to offset 100 from jump at offset 0
    AllocStub::patch_slow_path_offset(&mut code, 0, 100);

    // Relative = 100 - (0 + 6) = 94
    let expected = 94i32.to_le_bytes();
    assert_eq!(&code[2..6], &expected);
}
