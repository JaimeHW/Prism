use super::*;

// =========================================================================
// Basic Operations
// =========================================================================

#[test]
fn test_cell_new_with_int() {
    let cell = Cell::new(Value::int(42).unwrap());
    assert!(cell.is_bound());
    assert!(!cell.is_empty());
    assert_eq!(cell.get().unwrap().as_int(), Some(42));
}

#[test]
fn test_cell_new_with_float() {
    let cell = Cell::new(Value::float(3.125));
    assert!(cell.is_bound());
    let val = cell.get().unwrap();
    assert!((val.as_float().unwrap() - 3.125).abs() < 1e-10);
}

#[test]
fn test_cell_new_with_bool() {
    let cell_true = Cell::new(Value::bool(true));
    let cell_false = Cell::new(Value::bool(false));

    assert_eq!(cell_true.get().unwrap().as_bool(), Some(true));
    assert_eq!(cell_false.get().unwrap().as_bool(), Some(false));
}

#[test]
fn test_cell_new_with_none() {
    let cell = Cell::new(Value::none());
    assert!(cell.is_bound());
    assert!(cell.get().unwrap().is_none());
}

#[test]
fn test_cell_unbound() {
    let cell = Cell::unbound();
    assert!(cell.is_empty());
    assert!(!cell.is_bound());
    assert!(cell.get().is_none());
}

// =========================================================================
// Set Operations
// =========================================================================

#[test]
fn test_cell_set_updates_value() {
    let cell = Cell::new(Value::int(1).unwrap());
    assert_eq!(cell.get().unwrap().as_int(), Some(1));

    cell.set(Value::int(2).unwrap());
    assert_eq!(cell.get().unwrap().as_int(), Some(2));

    cell.set(Value::int(100).unwrap());
    assert_eq!(cell.get().unwrap().as_int(), Some(100));
}

#[test]
fn test_cell_set_on_unbound() {
    let cell = Cell::unbound();
    assert!(cell.is_empty());

    cell.set(Value::int(42).unwrap());
    assert!(cell.is_bound());
    assert_eq!(cell.get().unwrap().as_int(), Some(42));
}

#[test]
fn test_cell_clear() {
    let cell = Cell::new(Value::int(42).unwrap());
    assert!(cell.is_bound());

    cell.clear();
    assert!(cell.is_empty());
    assert!(cell.get().is_none());
}

#[test]
fn test_cell_clear_already_unbound() {
    let cell = Cell::unbound();
    cell.clear(); // Should not panic
    assert!(cell.is_empty());
}

// =========================================================================
// Swap Operations
// =========================================================================

#[test]
fn test_cell_swap_bound() {
    let cell = Cell::new(Value::int(1).unwrap());

    let old = cell.swap(Value::int(2).unwrap());
    assert_eq!(old.unwrap().as_int(), Some(1));
    assert_eq!(cell.get().unwrap().as_int(), Some(2));
}

#[test]
fn test_cell_swap_unbound() {
    let cell = Cell::unbound();

    let old = cell.swap(Value::int(42).unwrap());
    assert!(old.is_none());
    assert_eq!(cell.get().unwrap().as_int(), Some(42));
}

#[test]
fn test_cell_swap_multiple_times() {
    let cell = Cell::new(Value::int(1).unwrap());

    for i in 2..=10 {
        let old = cell.swap(Value::int(i).unwrap());
        assert_eq!(old.unwrap().as_int(), Some(i - 1));
    }
    assert_eq!(cell.get().unwrap().as_int(), Some(10));
}

// =========================================================================
// Convenience Methods
// =========================================================================

#[test]
fn test_cell_get_or_none_bound() {
    let cell = Cell::new(Value::int(42).unwrap());
    let value = cell.get_or_none();
    assert_eq!(value.as_int(), Some(42));
}

#[test]
fn test_cell_get_or_none_unbound() {
    let cell = Cell::unbound();
    let value = cell.get_or_none();
    assert!(value.is_none());
}

#[test]
fn test_cell_get_unchecked() {
    let cell = Cell::new(Value::int(42).unwrap());
    let value = cell.get_unchecked();
    assert_eq!(value.as_int(), Some(42));
}

// =========================================================================
// Clone and Debug
// =========================================================================

#[test]
fn test_cell_clone_creates_copy() {
    let cell1 = Cell::new(Value::int(42).unwrap());
    let cell2 = cell1.clone();

    // Both have same value initially
    assert_eq!(cell1.get().unwrap().as_int(), Some(42));
    assert_eq!(cell2.get().unwrap().as_int(), Some(42));

    // Modifying one doesn't affect the other
    cell1.set(Value::int(100).unwrap());
    assert_eq!(cell1.get().unwrap().as_int(), Some(100));
    assert_eq!(cell2.get().unwrap().as_int(), Some(42));
}

#[test]
fn test_cell_clone_unbound() {
    let cell1 = Cell::unbound();
    let cell2 = cell1.clone();

    assert!(cell1.is_empty());
    assert!(cell2.is_empty());

    // Setting one doesn't affect the other
    cell1.set(Value::int(42).unwrap());
    assert!(cell1.is_bound());
    assert!(cell2.is_empty());
}

#[test]
fn test_cell_debug_bound() {
    let cell = Cell::new(Value::int(42).unwrap());
    let debug = format!("{:?}", cell);
    assert!(debug.contains("Cell"));
}

#[test]
fn test_cell_debug_unbound() {
    let cell = Cell::unbound();
    let debug = format!("{:?}", cell);
    assert!(debug.contains("unbound"));
}

#[test]
fn test_cell_display_bound() {
    let cell = Cell::new(Value::int(42).unwrap());
    let display = format!("{}", cell);
    assert!(display.contains("cell"));
}

#[test]
fn test_cell_display_empty() {
    let cell = Cell::unbound();
    let display = format!("{}", cell);
    assert!(display.contains("empty"));
}

// =========================================================================
// Header Access
// =========================================================================

#[test]
fn test_cell_header_type_id() {
    let cell = Cell::new(Value::int(42).unwrap());
    assert_eq!(cell.header().type_id, TypeId::CELL);
}

#[test]
fn test_cell_header_mut() {
    let mut cell = Cell::new(Value::int(42).unwrap());
    let header = cell.header_mut();
    // Can mutate header fields if needed
    assert_eq!(header.type_id, TypeId::CELL);
}

// =========================================================================
// Size and Alignment
// =========================================================================

#[test]
fn test_cell_size() {
    // Cell should fit in a cache line (64 bytes)
    assert_eq!(std::mem::size_of::<Cell>(), 64);
}

#[test]
fn test_cell_alignment() {
    // Cell should be aligned to cache line boundary
    assert_eq!(std::mem::align_of::<Cell>(), 64);
}

// =========================================================================
// Thread Safety (basic checks)
// =========================================================================

#[test]
fn test_cell_concurrent_read_write() {
    use std::sync::Arc;
    use std::thread;

    let cell = Arc::new(Cell::new(Value::int(0).unwrap()));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let cell = Arc::clone(&cell);
            thread::spawn(move || {
                for i in 0..1000 {
                    cell.set(Value::int(i).unwrap());
                    let _ = cell.get();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Cell should have some valid value
    assert!(cell.is_bound());
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_cell_large_int() {
    // Use a large value that's clearly within any integer representation
    let large = 1_000_000_000i64;
    let cell = Cell::new(Value::int(large).unwrap());
    assert_eq!(cell.get().unwrap().as_int(), Some(large));
}

#[test]
fn test_cell_negative_int() {
    let cell = Cell::new(Value::int(-42).unwrap());
    assert_eq!(cell.get().unwrap().as_int(), Some(-42));
}

#[test]
fn test_cell_zero() {
    let cell = Cell::new(Value::int(0).unwrap());
    assert_eq!(cell.get().unwrap().as_int(), Some(0));
    assert!(cell.is_bound());
}

#[test]
fn test_cell_nan() {
    let cell = Cell::new(Value::float(f64::NAN));
    let val = cell.get().unwrap();
    assert!(val.as_float().unwrap().is_nan());
}

#[test]
fn test_cell_infinity() {
    let cell = Cell::new(Value::float(f64::INFINITY));
    assert_eq!(cell.get().unwrap().as_float(), Some(f64::INFINITY));
}

#[test]
fn test_cell_negative_zero() {
    let cell = Cell::new(Value::float(-0.0));
    let val = cell.get().unwrap().as_float().unwrap();
    // -0.0 should be preserved
    assert!(val.is_sign_negative() || val == 0.0);
}

// =========================================================================
// Closure Simulation
// =========================================================================

#[test]
fn test_cell_closure_simulation() {
    use std::sync::Arc;
    // Simulate: x = 10; def inner(): nonlocal x; x += 1; return x
    let cell = Arc::new(Cell::new(Value::int(10).unwrap()));

    // First "call" to inner
    {
        let cell = Arc::clone(&cell);
        let x = cell.get().unwrap().as_int().unwrap();
        cell.set(Value::int(x + 1).unwrap());
    }
    assert_eq!(cell.get().unwrap().as_int(), Some(11));

    // Second "call"
    {
        let cell = Arc::clone(&cell);
        let x = cell.get().unwrap().as_int().unwrap();
        cell.set(Value::int(x + 1).unwrap());
    }
    assert_eq!(cell.get().unwrap().as_int(), Some(12));
}

#[test]
fn test_cell_nested_closure_simulation() {
    use std::sync::Arc;

    // Simulate nested closures sharing a cell
    let outer_cell = Arc::new(Cell::new(Value::int(0).unwrap()));

    // "outer" function creates and modifies
    outer_cell.set(Value::int(100).unwrap());

    // "middle" function shares the cell
    let middle_cell = Arc::clone(&outer_cell);
    let val = middle_cell.get().unwrap().as_int().unwrap();
    middle_cell.set(Value::int(val + 10).unwrap());

    // "inner" function also shares the cell
    let inner_cell = Arc::clone(&outer_cell);
    let val = inner_cell.get().unwrap().as_int().unwrap();
    inner_cell.set(Value::int(val + 1).unwrap());

    // All closures see the final value
    assert_eq!(outer_cell.get().unwrap().as_int(), Some(111));
    assert_eq!(middle_cell.get().unwrap().as_int(), Some(111));
    assert_eq!(inner_cell.get().unwrap().as_int(), Some(111));
}
