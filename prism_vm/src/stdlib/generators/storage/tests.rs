use super::*;

// ════════════════════════════════════════════════════════════════════════
// LivenessMap Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_liveness_size() {
    assert_eq!(std::mem::size_of::<LivenessMap>(), 8);
}

#[test]
fn test_liveness_empty() {
    let map = LivenessMap::EMPTY;
    assert_eq!(map.count(), 0);
    assert!(!map.is_live(0));
    assert!(!map.is_live(63));
}

#[test]
fn test_liveness_all() {
    let map = LivenessMap::ALL;
    assert_eq!(map.count(), 64);
    assert!(map.is_live(0));
    assert!(map.is_live(63));
}

#[test]
fn test_liveness_from_bits() {
    let map = LivenessMap::from_bits(0b101);
    assert!(map.is_live(0));
    assert!(!map.is_live(1));
    assert!(map.is_live(2));
    assert_eq!(map.count(), 2);
}

#[test]
fn test_liveness_with_live() {
    let map = LivenessMap::EMPTY.with_live(5).with_live(10);
    assert!(map.is_live(5));
    assert!(map.is_live(10));
    assert!(!map.is_live(0));
    assert_eq!(map.count(), 2);
}

#[test]
fn test_liveness_without() {
    let map = LivenessMap::from_bits(0b111).without(1);
    assert!(map.is_live(0));
    assert!(!map.is_live(1));
    assert!(map.is_live(2));
}

#[test]
fn test_liveness_iter() {
    let map = LivenessMap::from_bits(0b10101);
    let regs: Vec<_> = map.iter().collect();
    assert_eq!(regs, vec![0, 2, 4]);
}

#[test]
fn test_liveness_iter_empty() {
    let map = LivenessMap::EMPTY;
    let regs: Vec<_> = map.iter().collect();
    assert!(regs.is_empty());
}

#[test]
fn test_liveness_iter_exact_size() {
    let map = LivenessMap::from_bits(0b1111);
    let iter = map.iter();
    assert_eq!(iter.len(), 4);
}

#[test]
fn test_liveness_compact_index() {
    let map = LivenessMap::from_bits(0b10101);
    // Live: 0, 2, 4 → compact indices 0, 1, 2
    assert_eq!(map.compact_index(0), 0);
    assert_eq!(map.compact_index(2), 1);
    assert_eq!(map.compact_index(4), 2);
}

#[test]
fn test_liveness_fits_inline() {
    assert!(LivenessMap::from_bits(0b11111111).fits_inline()); // 8
    assert!(!LivenessMap::from_bits(0b111111111).fits_inline()); // 9
}

#[test]
fn test_liveness_debug() {
    let map = LivenessMap::from_bits(0xFF);
    let debug = format!("{:?}", map);
    assert!(debug.contains("8 live"));
}

#[test]
fn test_liveness_high_bit_registers() {
    let map = LivenessMap::EMPTY.with_live(63);
    assert!(map.is_live(63));
    assert_eq!(map.count(), 1);
    assert_eq!(map.compact_index(63), 0);
}

// ════════════════════════════════════════════════════════════════════════
// FrameStorage Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_storage_size() {
    // Should be reasonably small
    let size = std::mem::size_of::<FrameStorage>();
    assert!(size <= 128, "FrameStorage too large: {}", size);
}

#[test]
fn test_storage_new() {
    let storage = FrameStorage::new();
    assert_eq!(storage.len(), 0);
    assert!(storage.is_empty());
    assert!(!storage.is_boxed());
    assert_eq!(storage.capacity(), INLINE_CAPACITY);
}

#[test]
fn test_storage_with_capacity_small() {
    let storage = FrameStorage::with_capacity(4);
    assert!(!storage.is_boxed());
    assert_eq!(storage.capacity(), INLINE_CAPACITY);
}

#[test]
fn test_storage_with_capacity_large() {
    let storage = FrameStorage::with_capacity(20);
    assert!(storage.is_boxed());
    assert_eq!(storage.capacity(), 20);
}

#[test]
fn test_storage_set_get_inline() {
    let mut storage = FrameStorage::new();
    storage.set(0, Value::int(42).unwrap());
    storage.set(1, Value::float(3.14));

    assert_eq!(storage.len(), 2);
    assert_eq!(storage.get(0).as_int().unwrap(), 42);
    assert!((storage.get(1).as_float().unwrap() - 3.14).abs() < 1e-10);
}

#[test]
fn test_storage_set_get_boxed() {
    let mut storage = FrameStorage::with_capacity(20);
    storage.set(0, Value::int(100).unwrap());
    storage.set(15, Value::int(200).unwrap());

    assert_eq!(storage.get(0).as_int().unwrap(), 100);
    assert_eq!(storage.get(15).as_int().unwrap(), 200);
}

#[test]
fn test_storage_capture_inline() {
    let mut registers = [Value::none(); 256];
    registers[0] = Value::int(10).unwrap();
    registers[2] = Value::int(20).unwrap();
    registers[5] = Value::int(30).unwrap();

    let liveness = LivenessMap::from_bits(0b100101); // regs 0, 2, 5
    let mut storage = FrameStorage::new();
    storage.capture(&registers, liveness);

    assert_eq!(storage.len(), 3);
    assert_eq!(storage.get(0).as_int().unwrap(), 10);
    assert_eq!(storage.get(1).as_int().unwrap(), 20);
    assert_eq!(storage.get(2).as_int().unwrap(), 30);
}

#[test]
fn test_storage_restore() {
    let mut storage = FrameStorage::new();
    storage.set(0, Value::int(100).unwrap());
    storage.set(1, Value::int(200).unwrap());
    storage.set(2, Value::int(300).unwrap());

    let liveness = LivenessMap::from_bits(0b100101); // regs 0, 2, 5
    let mut registers = [Value::none(); 256];
    storage.restore(&mut registers, liveness);

    assert_eq!(registers[0].as_int().unwrap(), 100);
    assert_eq!(registers[2].as_int().unwrap(), 200);
    assert_eq!(registers[5].as_int().unwrap(), 300);
}

#[test]
fn test_storage_capture_restore_roundtrip() {
    let mut orig_registers = [Value::none(); 256];
    orig_registers[1] = Value::int(42).unwrap();
    orig_registers[3] = Value::float(2.718);
    orig_registers[7] = Value::bool(true);

    let liveness = LivenessMap::from_bits(0b10001010); // regs 1, 3, 7

    let mut storage = FrameStorage::new();
    storage.capture(&orig_registers, liveness);

    let mut new_registers = [Value::none(); 256];
    storage.restore(&mut new_registers, liveness);

    assert_eq!(new_registers[1].as_int().unwrap(), 42);
    assert!((new_registers[3].as_float().unwrap() - 2.718).abs() < 1e-10);
    assert!(new_registers[7].as_bool().unwrap());
}

#[test]
fn test_storage_capture_spills_to_boxed() {
    let mut registers = [Value::none(); 256];
    for i in 0..16 {
        registers[i] = Value::int(i as i64).unwrap();
    }

    // 16 live registers - exceeds inline capacity
    let liveness = LivenessMap::from_bits(0xFFFF);
    let mut storage = FrameStorage::new();
    storage.capture(&registers, liveness);

    assert!(storage.is_boxed());
    assert_eq!(storage.len(), 16);

    for i in 0..16 {
        assert_eq!(storage.get(i).as_int().unwrap(), i as i64);
    }
}

#[test]
fn test_storage_clear() {
    let mut storage = FrameStorage::new();
    storage.set(0, Value::int(1).unwrap());
    storage.set(1, Value::int(2).unwrap());
    assert_eq!(storage.len(), 2);

    storage.clear();
    assert_eq!(storage.len(), 0);
    assert!(storage.is_empty());
}

#[test]
fn test_storage_clone_inline() {
    let mut storage = FrameStorage::new();
    storage.set(0, Value::int(42).unwrap());
    storage.set(1, Value::int(99).unwrap());

    let cloned = storage.clone();
    assert_eq!(cloned.len(), 2);
    assert_eq!(cloned.get(0).as_int().unwrap(), 42);
    assert_eq!(cloned.get(1).as_int().unwrap(), 99);
}

#[test]
fn test_storage_clone_boxed() {
    let mut storage = FrameStorage::with_capacity(20);
    storage.set(0, Value::int(1).unwrap());
    storage.set(10, Value::int(2).unwrap());

    let cloned = storage.clone();
    assert!(cloned.is_boxed());
    assert_eq!(cloned.get(0).as_int().unwrap(), 1);
    assert_eq!(cloned.get(10).as_int().unwrap(), 2);
}

#[test]
fn test_storage_debug() {
    let storage = FrameStorage::new();
    let debug = format!("{:?}", storage);
    assert!(debug.contains("FrameStorage"));
    assert!(debug.contains("len"));
}

// ════════════════════════════════════════════════════════════════════════
// Edge Cases
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_empty_liveness_capture() {
    let registers = [Value::none(); 256];
    let liveness = LivenessMap::EMPTY;
    let mut storage = FrameStorage::new();
    storage.capture(&registers, liveness);

    assert_eq!(storage.len(), 0);
}

#[test]
fn test_single_register_liveness() {
    let mut registers = [Value::none(); 256];
    registers[42] = Value::int(12345).unwrap();

    let liveness = LivenessMap::EMPTY.with_live(42);
    let mut storage = FrameStorage::new();
    storage.capture(&registers, liveness);

    assert_eq!(storage.len(), 1);
    assert_eq!(storage.get(0).as_int().unwrap(), 12345);

    let mut restored = [Value::none(); 256];
    storage.restore(&mut restored, liveness);
    assert_eq!(restored[42].as_int().unwrap(), 12345);
}

#[test]
fn test_max_inline_capacity() {
    let mut registers = [Value::none(); 256];
    for i in 0..8 {
        registers[i] = Value::int(i as i64 * 10).unwrap();
    }

    let liveness = LivenessMap::from_bits(0xFF); // Exactly 8 registers
    let mut storage = FrameStorage::new();
    storage.capture(&registers, liveness);

    assert!(!storage.is_boxed());
    assert_eq!(storage.len(), 8);
}

#[test]
fn test_storage_overwrite() {
    let mut storage = FrameStorage::new();
    storage.set(0, Value::int(1).unwrap());
    storage.set(0, Value::int(2).unwrap());

    assert_eq!(storage.get(0).as_int().unwrap(), 2);
}
