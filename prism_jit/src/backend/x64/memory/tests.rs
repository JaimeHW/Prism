use super::*;

#[test]
fn test_executable_buffer_creation() {
    let buf = ExecutableBuffer::new(1024).expect("Failed to allocate");
    assert!(buf.capacity() >= 1024);
    assert_eq!(buf.len(), 0);
    assert!(!buf.is_executable());
}

#[test]
fn test_executable_buffer_emit() {
    let mut buf = ExecutableBuffer::new(1024).expect("Failed to allocate");

    buf.emit_u8(0x90); // NOP
    buf.emit_u8(0xC3); // RET

    assert_eq!(buf.len(), 2);
    assert_eq!(buf.as_slice(), &[0x90, 0xC3]);
}

#[test]
fn test_executable_buffer_emit_multi() {
    let mut buf = ExecutableBuffer::new(1024).expect("Failed to allocate");

    buf.emit_bytes(&[0x48, 0x89, 0xC1]); // mov rcx, rax
    buf.emit_u32(0x12345678);

    assert_eq!(buf.len(), 7);
}

#[test]
fn test_executable_buffer_patch() {
    let mut buf = ExecutableBuffer::new(1024).expect("Failed to allocate");

    buf.emit_u32(0); // Placeholder
    buf.emit_u8(0xC3);

    buf.patch_u32(0, 0xDEADBEEF);

    assert_eq!(&buf.as_slice()[0..4], &0xDEADBEEFu32.to_le_bytes());
}

#[test]
fn test_executable_buffer_permissions() {
    let mut buf = ExecutableBuffer::new(1024).expect("Failed to allocate");

    // Write some code
    buf.emit_bytes(&[0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00]); // mov rax, 42
    buf.emit_u8(0xC3); // ret

    assert!(!buf.is_executable());
    assert!(buf.make_executable());
    assert!(buf.is_executable());

    // Should be able to make writable again
    assert!(buf.make_writable());
    assert!(!buf.is_executable());
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn test_executable_buffer_execute() {
    let mut buf = ExecutableBuffer::new(1024).expect("Failed to allocate");

    // Write: mov eax, 42; ret
    buf.emit_bytes(&[
        0xB8, 0x2A, 0x00, 0x00, 0x00, // mov eax, 42
        0xC3, // ret
    ]);

    assert!(buf.make_executable());

    // Execute the code
    type Fn = unsafe extern "C" fn() -> i32;
    let f: Fn = unsafe { buf.as_fn() };
    let result = unsafe { f() };

    assert_eq!(result, 42);
}

#[test]
fn test_page_alignment() {
    assert_eq!(ExecutableBuffer::align_to_page(1), PAGE_SIZE);
    assert_eq!(ExecutableBuffer::align_to_page(PAGE_SIZE), PAGE_SIZE);
    assert_eq!(
        ExecutableBuffer::align_to_page(PAGE_SIZE + 1),
        2 * PAGE_SIZE
    );
}

#[test]
fn test_code_cache_stats() {
    let stats = CodeCacheStats::new();

    stats.record_allocation(1024);
    stats.record_allocation(2048);
    assert_eq!(stats.total_bytes(), 3072);
    assert_eq!(stats.compilations(), 2);

    stats.record_eviction(1024);
    assert_eq!(stats.total_bytes(), 2048);
    assert_eq!(stats.evictions(), 1);
}
