use super::*;

#[test]
fn test_card_table_creation() {
    let table = CardTable::new(0x10000, 0x10000);
    assert_eq!(table.card_count(), 128); // 64KB / 512B = 128 cards
    assert_eq!(table.heap_base(), 0x10000);
}

#[test]
fn test_card_index() {
    let table = CardTable::new(0x10000, 0x10000);

    // First byte of first card
    assert_eq!(table.card_index(0x10000), Some(0));

    // Last byte of first card
    assert_eq!(table.card_index(0x101FF), Some(0));

    // First byte of second card
    assert_eq!(table.card_index(0x10200), Some(1));

    // Out of range
    assert_eq!(table.card_index(0x5000), None);
    assert_eq!(table.card_index(0x25000), None);
}

#[test]
fn test_card_address_range() {
    let table = CardTable::new(0x10000, 0x10000);

    assert_eq!(table.card_address_range(0), Some((0x10000, 0x10200)));
    assert_eq!(table.card_address_range(1), Some((0x10200, 0x10400)));
    assert_eq!(table.card_address_range(200), None);
}

#[test]
fn test_mark_and_check() {
    let table = CardTable::new(0x10000, 0x10000);

    assert!(!table.is_dirty(0));
    assert!(!table.is_dirty(1));

    table.mark_dirty(0x10100); // Card 0
    table.mark_dirty(0x10400); // Card 2

    assert!(table.is_dirty(0));
    assert!(!table.is_dirty(1));
    assert!(table.is_dirty(2));
}

#[test]
fn test_dirty_card_iteration() {
    let table = CardTable::new(0x10000, 0x10000);

    table.mark_dirty_index(0);
    table.mark_dirty_index(5);
    table.mark_dirty_index(10);

    let dirty: Vec<usize> = table.dirty_card_indices().collect();
    assert_eq!(dirty, vec![0, 5, 10]);
    assert_eq!(table.dirty_count(), 3);
}

#[test]
fn test_clear() {
    let table = CardTable::new(0x10000, 0x10000);

    table.mark_dirty_index(0);
    table.mark_dirty_index(1);
    assert_eq!(table.dirty_count(), 2);

    table.clear(0);
    assert_eq!(table.dirty_count(), 1);

    table.clear_all();
    assert_eq!(table.dirty_count(), 0);
}

#[test]
fn test_barrier_info() {
    let table = CardTable::new(0x10000, 0x10000);
    let info = table.barrier_info();

    assert_eq!(info.heap_base, 0x10000);
    assert_eq!(info.card_shift, CARD_SHIFT);
}

#[test]
fn test_constants() {
    // Verify card size is power of 2
    assert!(CARD_SIZE.is_power_of_two());
    assert_eq!(1 << CARD_SHIFT, CARD_SIZE);
}
