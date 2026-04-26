use super::*;

// ════════════════════════════════════════════════════════════════════════
// SuspendResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_suspend_result_new() {
    let result = SuspendResult::new(42, 0b1010);
    assert_eq!(result.resume_index, 42);
    assert_eq!(result.liveness, 0b1010);
    assert_eq!(result.live_count, 2);
    assert!(result.is_empty()); // No registers captured yet
}

#[test]
fn test_suspend_result_empty_liveness() {
    let result = SuspendResult::new(0, 0);
    assert_eq!(result.live_count, 0);
    assert!(result.is_empty());
}

#[test]
fn test_suspend_result_full_liveness() {
    let result = SuspendResult::new(0, u64::MAX);
    assert_eq!(result.live_count, 64);
}

// ════════════════════════════════════════════════════════════════════════
// capture_generator_frame Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_capture_empty() {
    let registers: Vec<Value> = vec![];
    let result = capture_generator_frame(&registers, 0, 0).unwrap();

    assert_eq!(result.resume_index, 0);
    assert_eq!(result.live_count, 0);
    assert!(result.is_empty());
}

#[test]
fn test_capture_all_live() {
    let registers = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    let result = capture_generator_frame(&registers, 0b111, 5).unwrap();

    assert_eq!(result.resume_index, 5);
    assert_eq!(result.live_count, 3);
    assert_eq!(result.len(), 3);
    assert_eq!(result.registers[0].as_int(), Some(1));
    assert_eq!(result.registers[1].as_int(), Some(2));
    assert_eq!(result.registers[2].as_int(), Some(3));
}

#[test]
fn test_capture_sparse() {
    let registers = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ];
    // Only registers 0 and 3 are live
    let result = capture_generator_frame(&registers, 0b1001, 10).unwrap();

    assert_eq!(result.live_count, 2);
    assert_eq!(result.len(), 2);
    assert_eq!(result.registers[0].as_int(), Some(1)); // Register 0
    assert_eq!(result.registers[1].as_int(), Some(4)); // Register 3
}

#[test]
fn test_capture_with_none_values() {
    let registers = vec![
        Value::none(),
        Value::int(42).unwrap(),
        Value::none(),
        Value::bool(true),
    ];
    let result = capture_generator_frame(&registers, 0b1111, 0).unwrap();

    assert_eq!(result.len(), 4);
    assert!(result.registers[0].is_none());
    assert_eq!(result.registers[1].as_int(), Some(42));
    assert!(result.registers[2].is_none());
    assert_eq!(result.registers[3].as_bool(), Some(true));
}

#[test]
fn test_capture_preserves_resume_index() {
    let registers = vec![Value::int(1).unwrap()];

    for idx in [0, 1, 100, u32::MAX] {
        let result = capture_generator_frame(&registers, 0b1, idx).unwrap();
        assert_eq!(result.resume_index, idx);
    }
}

// ════════════════════════════════════════════════════════════════════════
// save_live_registers Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_save_live_registers_empty() {
    let src: Vec<Value> = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let mut dst = Vec::new();

    let count = save_live_registers(&src, &mut dst, 0);

    assert_eq!(count, 0);
    assert!(dst.is_empty());
}

#[test]
fn test_save_live_registers_all() {
    let src = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let mut dst = Vec::new();

    let count = save_live_registers(&src, &mut dst, 0b11);

    assert_eq!(count, 2);
    assert_eq!(dst[0].as_int(), Some(1));
    assert_eq!(dst[1].as_int(), Some(2));
}

#[test]
fn test_save_live_registers_sparse() {
    let src: Vec<Value> = (0..8).map(|i| Value::int(i).unwrap()).collect();
    let mut dst = Vec::new();

    // Even registers only
    let count = save_live_registers(&src, &mut dst, 0b01010101);

    assert_eq!(count, 4);
    assert_eq!(dst[0].as_int(), Some(0));
    assert_eq!(dst[1].as_int(), Some(2));
    assert_eq!(dst[2].as_int(), Some(4));
    assert_eq!(dst[3].as_int(), Some(6));
}

// ════════════════════════════════════════════════════════════════════════
// compute_liveness_bitmap Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_compute_liveness_empty() {
    let bitmap = compute_liveness_bitmap(std::iter::empty());
    assert_eq!(bitmap, 0);
}

#[test]
fn test_compute_liveness_single() {
    let bitmap = compute_liveness_bitmap([5]);
    assert_eq!(bitmap, 0b100000);
}

#[test]
fn test_compute_liveness_multiple() {
    let bitmap = compute_liveness_bitmap([0, 2, 4, 6]);
    assert_eq!(bitmap, 0b01010101);
}

#[test]
fn test_compute_liveness_large_index_ignored() {
    let bitmap = compute_liveness_bitmap([0, 64, 65, 100]);
    assert_eq!(bitmap, 0b1); // Only index 0 counted
}

#[test]
fn test_compute_liveness_duplicates() {
    let bitmap = compute_liveness_bitmap([1, 1, 1, 1]);
    assert_eq!(bitmap, 0b10);
}

// ════════════════════════════════════════════════════════════════════════
// count_live_registers Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_count_live_zero() {
    assert_eq!(count_live_registers(0), 0);
}

#[test]
fn test_count_live_one() {
    assert_eq!(count_live_registers(0b1), 1);
    assert_eq!(count_live_registers(0b100), 1);
}

#[test]
fn test_count_live_multiple() {
    assert_eq!(count_live_registers(0b1010), 2);
    assert_eq!(count_live_registers(0b1111), 4);
    assert_eq!(count_live_registers(0xFF), 8);
}

#[test]
fn test_count_live_max() {
    assert_eq!(count_live_registers(u64::MAX), 64);
}

// ════════════════════════════════════════════════════════════════════════
// is_register_live Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_register_live_empty() {
    assert!(!is_register_live(0, 0));
    assert!(!is_register_live(0, 63));
}

#[test]
fn test_is_register_live_full() {
    for i in 0..64 {
        assert!(is_register_live(u64::MAX, i));
    }
}

#[test]
fn test_is_register_live_sparse() {
    let liveness = 0b1010;
    assert!(!is_register_live(liveness, 0));
    assert!(is_register_live(liveness, 1));
    assert!(!is_register_live(liveness, 2));
    assert!(is_register_live(liveness, 3));
}

#[test]
fn test_is_register_live_out_of_range() {
    assert!(!is_register_live(u64::MAX, 64));
    assert!(!is_register_live(u64::MAX, 100));
}

// ════════════════════════════════════════════════════════════════════════
// nth_live_register Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_nth_live_empty() {
    assert_eq!(nth_live_register(0, 0), None);
}

#[test]
fn test_nth_live_first() {
    assert_eq!(nth_live_register(0b1, 0), Some(0));
    assert_eq!(nth_live_register(0b100, 0), Some(2));
}

#[test]
fn test_nth_live_sequential() {
    let liveness = 0b1010_1010;
    assert_eq!(nth_live_register(liveness, 0), Some(1));
    assert_eq!(nth_live_register(liveness, 1), Some(3));
    assert_eq!(nth_live_register(liveness, 2), Some(5));
    assert_eq!(nth_live_register(liveness, 3), Some(7));
    assert_eq!(nth_live_register(liveness, 4), None);
}

#[test]
fn test_nth_live_out_of_range() {
    assert_eq!(nth_live_register(0b1, 1), None);
    assert_eq!(nth_live_register(0b11, 2), None);
}

// ════════════════════════════════════════════════════════════════════════
// SuspendError Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_suspend_error_display() {
    assert_eq!(
        SuspendError::NotSuspendable.to_string(),
        "Generator is not in a suspendable state"
    );
    assert_eq!(
        SuspendError::NoActiveGenerator.to_string(),
        "No active generator to suspend"
    );
    assert_eq!(
        SuspendError::InvalidResumeIndex(42).to_string(),
        "Invalid resume index: 42"
    );
    assert_eq!(
        SuspendError::TooManyRegisters(100).to_string(),
        "Too many live registers: 100 (max 64)"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_capture_more_registers_than_liveness_bits() {
    // Registers beyond bit 63 are ignored
    let registers: Vec<Value> = (0..100).map(|i| Value::int(i).unwrap()).collect();
    let result = capture_generator_frame(&registers, u64::MAX, 0).unwrap();

    // Only 64 registers captured
    assert_eq!(result.len(), 64);
}

#[test]
fn test_capture_fewer_registers_than_liveness_bits() {
    // Liveness marks more registers than exist
    let registers = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let result = capture_generator_frame(&registers, 0b1111, 0).unwrap();

    // Only 2 actually captured
    assert_eq!(result.len(), 2);
}

#[test]
fn test_liveness_bitmap_boundary() {
    // Test bit 63
    let bitmap = compute_liveness_bitmap([63]);
    assert_eq!(bitmap, 1 << 63);
    assert!(is_register_live(bitmap, 63));
    assert_eq!(nth_live_register(bitmap, 0), Some(63));
}

// ════════════════════════════════════════════════════════════════════════
// capture_to_pooled_frame Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_capture_to_pooled_frame() {
    use crate::ops::generator::frame_pool::{GeneratorFramePool, PooledFrame};

    let mut pool = GeneratorFramePool::new();
    let mut frame = pool.allocate(4);

    let registers = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    let liveness = 0b101; // Registers 0 and 2

    let count = capture_to_pooled_frame(&registers, liveness, 42, &mut frame);

    assert_eq!(count, 2);
    assert_eq!(frame.resume_index(), 42);
    assert_eq!(frame.liveness(), 0b101);
}
