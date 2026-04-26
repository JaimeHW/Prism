use super::*;

#[test]
fn test_encode_jmp_rel32() {
    // Test forward jump
    let from = 0x1000 as *const u8;
    let to = 0x1100 as *const u8;
    let bytes = encode_jmp_rel32(from, to).expect("Should encode");
    assert_eq!(bytes[0], JMP_REL32_OPCODE);

    // Offset should be 0x100 - 5 = 0xFB
    let offset = i32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
    assert_eq!(offset, 0xFB);
}

#[test]
fn test_encode_jmp_rel32_backward() {
    // Test backward jump
    let from = 0x1100 as *const u8;
    let to = 0x1000 as *const u8;
    let bytes = encode_jmp_rel32(from, to).expect("Should encode");
    assert_eq!(bytes[0], JMP_REL32_OPCODE);

    let offset = i32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
    assert_eq!(offset, -0x105); // -0x100 - 5
}

#[test]
fn test_patcher_creation() {
    let patcher = Patcher::new();
    let stats = patcher.stats();
    assert_eq!(stats.patches_applied, 0);
    assert_eq!(stats.patches_rolled_back, 0);
}

#[test]
fn test_nop5_bytes() {
    // Verify NOP5 is correct encoding for 5-byte NOP
    assert_eq!(NOP5_BYTES[0], 0x0F);
    assert_eq!(NOP5_BYTES[1], 0x1F);
    assert_eq!(NOP5_SIZE, 5);
}

#[test]
fn test_page_alignment() {
    let patcher = Patcher::new();
    let page_size = patcher.page_size;

    assert_eq!(patcher.page_align(0x1000), 0x1000 & !(page_size - 1));
    assert_eq!(patcher.page_align(0x1001), 0x1000 & !(page_size - 1));
}
