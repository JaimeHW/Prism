//! Card table for remembered set tracking.
//!
//! The card table divides the heap into fixed-size "cards" (typically 512 bytes).
//! Each card has a corresponding byte that indicates whether the card contains
//! a pointer to the young generation.
//!
//! During minor GC, we only scan cards that are marked dirty.

use std::sync::atomic::{AtomicU8, Ordering};

/// Card state values.
pub const CARD_CLEAN: u8 = 0;
pub const CARD_DIRTY: u8 = 1;

/// Card table for write barrier tracking.
///
/// Each byte in the table represents a "card" of heap memory.
/// When a pointer to young generation is stored in that card,
/// the byte is set to DIRTY.
pub struct CardTable {
    /// The card bytes.
    cards: Box<[AtomicU8]>,
    /// Start address of the covered region.
    base: usize,
    /// Size of each card in bytes.
    card_size: usize,
    /// Log2 of card size for fast division.
    card_shift: u32,
}

impl CardTable {
    /// Create a new card table covering the given address range.
    ///
    /// # Arguments
    ///
    /// * `base` - Start address of the covered region
    /// * `size` - Size of the covered region in bytes
    /// * `card_size` - Size of each card (must be power of 2)
    pub fn new(base: usize, size: usize, card_size: usize) -> Self {
        assert!(card_size.is_power_of_two(), "Card size must be power of 2");

        let card_shift = card_size.trailing_zeros();
        let num_cards = (size + card_size - 1) / card_size;

        // Allocate card bytes
        let cards: Vec<AtomicU8> = (0..num_cards).map(|_| AtomicU8::new(CARD_CLEAN)).collect();

        Self {
            cards: cards.into_boxed_slice(),
            base,
            card_size,
            card_shift,
        }
    }

    /// Get the card index for an address.
    #[inline]
    fn card_index(&self, addr: usize) -> Option<usize> {
        if addr < self.base {
            return None;
        }
        let offset = addr - self.base;
        let index = offset >> self.card_shift;
        if index < self.cards.len() {
            Some(index)
        } else {
            None
        }
    }

    /// Mark a card as dirty.
    #[inline]
    pub fn mark(&self, ptr: *const ()) {
        if let Some(index) = self.card_index(ptr as usize) {
            self.cards[index].store(CARD_DIRTY, Ordering::Relaxed);
        }
    }

    /// Check if a card is dirty.
    #[inline]
    pub fn is_dirty(&self, ptr: *const ()) -> bool {
        self.card_index(ptr as usize)
            .map(|i| self.cards[i].load(Ordering::Relaxed) == CARD_DIRTY)
            .unwrap_or(false)
    }

    /// Clear a single card.
    #[inline]
    pub fn clear(&self, ptr: *const ()) {
        if let Some(index) = self.card_index(ptr as usize) {
            self.cards[index].store(CARD_CLEAN, Ordering::Relaxed);
        }
    }

    /// Clear all cards.
    pub fn clear_all(&self) {
        for card in self.cards.iter() {
            card.store(CARD_CLEAN, Ordering::Relaxed);
        }
    }

    /// Iterate over dirty cards, calling the closure with the card's base address.
    pub fn for_each_dirty<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize), // (card_start, card_end)
    {
        for (i, card) in self.cards.iter().enumerate() {
            if card.load(Ordering::Relaxed) == CARD_DIRTY {
                let card_start = self.base + (i << self.card_shift);
                let card_end = card_start + self.card_size;
                f(card_start, card_end);
            }
        }
    }

    /// Count dirty cards.
    pub fn dirty_count(&self) -> usize {
        self.cards
            .iter()
            .filter(|c| c.load(Ordering::Relaxed) == CARD_DIRTY)
            .count()
    }

    /// Get total number of cards.
    pub fn len(&self) -> usize {
        self.cards.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Get card size.
    pub fn card_size(&self) -> usize {
        self.card_size
    }

    /// Get the base address.
    pub fn base(&self) -> usize {
        self.base
    }
}

#[cfg(test)]
mod tests {
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
}
