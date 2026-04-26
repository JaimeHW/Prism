use super::*;

#[test]
fn test_call_counting() {
    let mut profiler = Profiler::new();
    let code_id = CodeId(12345);

    // Call count should increase monotonically.
    for i in 1..=16 {
        assert_eq!(profiler.record_call(code_id), i);
    }

    assert_eq!(profiler.call_count(code_id), 16);
}

#[test]
fn test_type_bitmap() {
    let mut bitmap = TypeBitmap::default();

    bitmap.record(TypeBitmap::INT);
    assert!(bitmap.is_monomorphic());
    assert_eq!(bitmap.single_type(), Some(TypeBitmap::INT));

    bitmap.record(TypeBitmap::FLOAT);
    assert!(!bitmap.is_monomorphic());
    assert!(bitmap.is_polymorphic());

    // Add more types to make megamorphic
    bitmap.record(TypeBitmap::STRING);
    bitmap.record(TypeBitmap::LIST);
    bitmap.record(TypeBitmap::DICT);
    assert!(bitmap.is_megamorphic());
}

#[test]
fn test_loop_counting() {
    let mut profiler = Profiler::new();
    let code_id = CodeId(0);

    for i in 0..TIER1_THRESHOLD {
        let is_hot = profiler.record_loop(code_id, 10);
        if i < TIER1_THRESHOLD - 1 {
            assert!(!is_hot);
        } else {
            assert!(is_hot);
        }
    }
}
