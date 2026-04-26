use super::*;

#[test]
fn test_card_table_creation() {
    let table = CardTable::new(0x1000, 0x10000, 512);
    assert_eq!(table.len(), 0x10000 / 512);
    assert_eq!(table.card_size(), 512);
}

#[test]
fn test_card_marking() {
    let base = 0x1000usize;
    let table = CardTable::new(base, 0x10000, 512);

    let ptr = (base + 100) as *const ();
    assert!(!table.is_dirty(ptr));

    table.mark(ptr);
    assert!(table.is_dirty(ptr));

    table.clear(ptr);
    assert!(!table.is_dirty(ptr));
}

#[test]
fn test_card_same_card() {
    let base = 0x1000usize;
    let table = CardTable::new(base, 0x10000, 512);

    let ptr1 = (base + 100) as *const ();
    let ptr2 = (base + 200) as *const (); // Same card

    table.mark(ptr1);
    assert!(table.is_dirty(ptr2)); // Both in same card
}

#[test]
fn test_card_different_cards() {
    let base = 0x1000usize;
    let table = CardTable::new(base, 0x10000, 512);

    let ptr1 = (base + 100) as *const ();
    let ptr2 = (base + 600) as *const (); // Different card

    table.mark(ptr1);
    assert!(!table.is_dirty(ptr2));
}

#[test]
fn test_clear_all() {
    let base = 0x1000usize;
    let table = CardTable::new(base, 0x10000, 512);

    // Mark several cards
    for i in 0..10 {
        table.mark((base + i * 600) as *const ());
    }
    assert!(table.dirty_count() > 0);

    table.clear_all();
    assert_eq!(table.dirty_count(), 0);
}

#[test]
fn test_for_each_dirty() {
    let base = 0x1000usize;
    let table = CardTable::new(base, 0x10000, 512);

    table.mark((base + 100) as *const ());
    table.mark((base + 1500) as *const ());

    let mut dirty_ranges = Vec::new();
    table.for_each_dirty(|start, end| {
        dirty_ranges.push((start, end));
    });

    assert_eq!(dirty_ranges.len(), 2);
}
