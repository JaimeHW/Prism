//! Dispatch Table Patching for JIT Entry.
//!
//! Provides atomic replacement of dispatch table entries to redirect
//! execution from interpreter handlers to compiled JIT code.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Dispatch Table Patching                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  Before JIT compilation:                                                │
//! │  ┌──────────────────┐                                                   │
//! │  │ dispatch[opcode] │ ─────▶ interpreter_handler                        │
//! │  └──────────────────┘                                                   │
//! │                                                                         │
//! │  After JIT compilation:                                                 │
//! │  ┌──────────────────┐                                                   │
//! │  │ dispatch[opcode] │ ─────▶ jit_entry_stub                             │
//! │  └──────────────────┘              │                                    │
//! │                                    ▼                                    │
//! │                            ┌──────────────┐                             │
//! │                            │ Check if     │                             │
//! │                            │ code hot?    │                             │
//! │                            └──────┬───────┘                             │
//! │                                   │                                     │
//! │                       ┌───────────┴───────────┐                         │
//! │                       │ yes                   │ no                      │
//! │                       ▼                       ▼                         │
//! │               ┌───────────────┐      ┌────────────────┐                 │
//! │               │ Execute JIT   │      │ Execute interp │                 │
//! │               │ code          │      │ handler        │                 │
//! │               └───────────────┘      └────────────────┘                 │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

// =============================================================================
// Handler Entry
// =============================================================================

/// A dispatch table entry that can be atomically replaced.
#[repr(C)]
pub struct HandlerEntry {
    /// Current handler function pointer.
    handler: AtomicPtr<()>,
    /// Number of times this entry has been patched.
    patch_count: AtomicU64,
}

impl HandlerEntry {
    /// Create a new entry with the given handler.
    #[inline]
    pub const fn new(handler: *const ()) -> Self {
        Self {
            handler: AtomicPtr::new(handler as *mut ()),
            patch_count: AtomicU64::new(0),
        }
    }

    /// Get the current handler.
    #[inline]
    pub fn get(&self) -> *const () {
        self.handler.load(Ordering::Acquire)
    }

    /// Set a new handler atomically.
    #[inline]
    pub fn set(&self, handler: *const ()) {
        self.handler.store(handler as *mut (), Ordering::Release);
        self.patch_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Compare-and-swap handler.
    #[inline]
    pub fn compare_exchange(
        &self,
        current: *const (),
        new: *const (),
    ) -> Result<*const (), *const ()> {
        self.handler
            .compare_exchange(
                current as *mut (),
                new as *mut (),
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .map(|p| {
                self.patch_count.fetch_add(1, Ordering::Relaxed);
                p as *const ()
            })
            .map_err(|p| p as *const ())
    }

    /// Get patch count.
    #[inline]
    pub fn patch_count(&self) -> u64 {
        self.patch_count.load(Ordering::Relaxed)
    }
}

// SAFETY: HandlerEntry uses atomic operations for all mutations.
unsafe impl Send for HandlerEntry {}
unsafe impl Sync for HandlerEntry {}

// =============================================================================
// Dynamic Dispatch Table
// =============================================================================

/// Maximum number of opcodes.
pub const MAX_OPCODES: usize = 256;

/// A dynamic dispatch table that supports atomic handler replacement.
///
/// This table shadows the static dispatch table and allows for runtime
/// patching to redirect execution to JIT code.
#[repr(C, align(64))] // Cache-line aligned
pub struct DynamicDispatchTable {
    /// Handler entries, one per opcode.
    entries: [HandlerEntry; MAX_OPCODES],
    /// Whether this table is active (vs static table).
    active: AtomicU64,
    /// Total patches applied.
    total_patches: AtomicU64,
}

impl DynamicDispatchTable {
    /// Create a new dynamic dispatch table.
    ///
    /// All entries initially point to the provided default handler.
    pub fn new(default_handler: *const ()) -> Self {
        const INIT: HandlerEntry = HandlerEntry::new(ptr::null());
        let mut table = Self {
            entries: [INIT; MAX_OPCODES],
            active: AtomicU64::new(0),
            total_patches: AtomicU64::new(0),
        };

        // Initialize all entries with default handler
        for entry in &mut table.entries {
            entry.handler = AtomicPtr::new(default_handler as *mut ());
        }

        table
    }

    /// Initialize from a static dispatch table.
    pub fn from_static_table(handlers: &[*const (); MAX_OPCODES]) -> Self {
        const INIT: HandlerEntry = HandlerEntry::new(ptr::null());
        let mut table = Self {
            entries: [INIT; MAX_OPCODES],
            active: AtomicU64::new(0),
            total_patches: AtomicU64::new(0),
        };

        for (i, &handler) in handlers.iter().enumerate() {
            table.entries[i].handler = AtomicPtr::new(handler as *mut ());
        }

        table
    }

    /// Get the handler for an opcode.
    #[inline(always)]
    pub fn get(&self, opcode: u8) -> *const () {
        // Safety: opcode is u8, always in bounds
        unsafe { self.entries.get_unchecked(opcode as usize).get() }
    }

    /// Patch an opcode's handler atomically.
    ///
    /// Returns the previous handler.
    #[inline]
    pub fn patch(&self, opcode: u8, new_handler: *const ()) -> *const () {
        let entry = &self.entries[opcode as usize];
        let old = entry.get();
        entry.set(new_handler);
        self.total_patches.fetch_add(1, Ordering::Relaxed);
        old
    }

    /// Patch only if the current handler matches expected.
    ///
    /// Returns Ok(old) if swap succeeded, Err(current) if not.
    #[inline]
    pub fn patch_if(
        &self,
        opcode: u8,
        expected: *const (),
        new_handler: *const (),
    ) -> Result<*const (), *const ()> {
        let entry = &self.entries[opcode as usize];
        let result = entry.compare_exchange(expected, new_handler);
        if result.is_ok() {
            self.total_patches.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Restore an opcode to its original handler.
    #[inline]
    pub fn restore(&self, opcode: u8, original: *const ()) {
        self.entries[opcode as usize].set(original);
    }

    /// Mark this table as active.
    #[inline]
    pub fn activate(&self) {
        self.active.store(1, Ordering::Release);
    }

    /// Mark this table as inactive.
    #[inline]
    pub fn deactivate(&self) {
        self.active.store(0, Ordering::Release);
    }

    /// Check if table is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire) != 0
    }

    /// Get the entry for an opcode.
    #[inline]
    pub fn entry(&self, opcode: u8) -> &HandlerEntry {
        &self.entries[opcode as usize]
    }

    /// Get statistics.
    pub fn stats(&self) -> DispatchTableStats {
        let mut patched_count = 0;
        for entry in &self.entries {
            if entry.patch_count() > 0 {
                patched_count += 1;
            }
        }

        DispatchTableStats {
            total_patches: self.total_patches.load(Ordering::Relaxed),
            patched_entries: patched_count,
            is_active: self.is_active(),
        }
    }
}

/// Statistics for the dynamic dispatch table.
#[derive(Debug, Clone, Copy, Default)]
pub struct DispatchTableStats {
    /// Total patches applied.
    pub total_patches: u64,
    /// Number of entries that have been patched at least once.
    pub patched_entries: usize,
    /// Whether the table is currently active.
    pub is_active: bool,
}

// =============================================================================
// JIT Entry Stub Registry
// =============================================================================

/// Registry of JIT entry stubs for code objects.
///
/// Maps (code_id, bc_offset) -> JIT entry point.
#[derive(Debug, Default)]
pub struct JitEntryRegistry {
    /// Entry points indexed by code ID.
    entries: std::collections::HashMap<u64, JitCodeEntry>,
    /// Total registered entry points.
    count: u64,
}

/// JIT entry information for a code object.
#[derive(Debug, Clone)]
pub struct JitCodeEntry {
    /// Code object ID.
    pub code_id: u64,
    /// Entry point address.
    pub entry_point: *const u8,
    /// Exit stub for returning to interpreter.
    pub exit_stub: *const u8,
    /// Whether this entry is active.
    pub active: bool,
}

// SAFETY: JitCodeEntry contains raw pointers but is only used within
// the JIT runtime's synchronization boundaries.
unsafe impl Send for JitCodeEntry {}
unsafe impl Sync for JitCodeEntry {}

impl JitEntryRegistry {
    /// Create a new registry.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            count: 0,
        }
    }

    /// Register a JIT entry point.
    pub fn register(&mut self, code_id: u64, entry_point: *const u8, exit_stub: *const u8) {
        let entry = JitCodeEntry {
            code_id,
            entry_point,
            exit_stub,
            active: true,
        };
        self.entries.insert(code_id, entry);
        self.count += 1;
    }

    /// Get entry point for a code object.
    #[inline]
    pub fn get(&self, code_id: u64) -> Option<&JitCodeEntry> {
        self.entries.get(&code_id).filter(|e| e.active)
    }

    /// Invalidate a JIT entry (on deoptimization).
    pub fn invalidate(&mut self, code_id: u64) -> bool {
        if let Some(entry) = self.entries.get_mut(&code_id) {
            entry.active = false;
            true
        } else {
            false
        }
    }

    /// Remove a JIT entry entirely.
    pub fn remove(&mut self, code_id: u64) -> Option<JitCodeEntry> {
        self.entries.remove(&code_id)
    }

    /// Number of registered entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_entry_new() {
        let handler = 0x12345678 as *const ();
        let entry = HandlerEntry::new(handler);

        assert_eq!(entry.get(), handler);
        assert_eq!(entry.patch_count(), 0);
    }

    #[test]
    fn test_handler_entry_set() {
        let entry = HandlerEntry::new(ptr::null());
        let new_handler = 0x12345678 as *const ();

        entry.set(new_handler);

        assert_eq!(entry.get(), new_handler);
        assert_eq!(entry.patch_count(), 1);
    }

    #[test]
    fn test_handler_entry_compare_exchange() {
        let initial = 0x1000 as *const ();
        let entry = HandlerEntry::new(initial);

        let new = 0x2000 as *const ();
        let result = entry.compare_exchange(initial, new);

        assert!(result.is_ok());
        assert_eq!(entry.get(), new);

        // Try with wrong expected
        let wrong = 0x3000 as *const ();
        let result = entry.compare_exchange(initial, wrong);

        assert!(result.is_err());
        assert_eq!(entry.get(), new);
    }

    #[test]
    fn test_dynamic_dispatch_table_new() {
        let default = 0x1000 as *const ();
        let table = DynamicDispatchTable::new(default);

        assert!(!table.is_active());

        for i in 0..MAX_OPCODES {
            assert_eq!(table.get(i as u8), default);
        }
    }

    #[test]
    fn test_dynamic_dispatch_table_patch() {
        let default = 0x1000 as *const ();
        let table = DynamicDispatchTable::new(default);

        let new_handler = 0x2000 as *const ();
        let old = table.patch(0x10, new_handler);

        assert_eq!(old, default);
        assert_eq!(table.get(0x10), new_handler);

        let stats = table.stats();
        assert_eq!(stats.total_patches, 1);
        assert_eq!(stats.patched_entries, 1);
    }

    #[test]
    fn test_dynamic_dispatch_table_patch_if() {
        let default = 0x1000 as *const ();
        let table = DynamicDispatchTable::new(default);

        let new = 0x2000 as *const ();
        let result = table.patch_if(0x20, default, new);

        assert!(result.is_ok());
        assert_eq!(table.get(0x20), new);

        // Try with wrong expected
        let wrong = 0x3000 as *const ();
        let result = table.patch_if(0x20, default, wrong);

        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_dispatch_table_activate() {
        let table = DynamicDispatchTable::new(ptr::null());

        assert!(!table.is_active());
        table.activate();
        assert!(table.is_active());
        table.deactivate();
        assert!(!table.is_active());
    }

    #[test]
    fn test_jit_entry_registry() {
        let mut registry = JitEntryRegistry::new();

        assert!(registry.is_empty());

        registry.register(1, 0x1000 as *const u8, 0x1100 as *const u8);
        registry.register(2, 0x2000 as *const u8, 0x2100 as *const u8);

        assert_eq!(registry.len(), 2);

        let entry = registry.get(1);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().code_id, 1);
    }

    #[test]
    fn test_jit_entry_registry_invalidate() {
        let mut registry = JitEntryRegistry::new();

        registry.register(1, 0x1000 as *const u8, 0x1100 as *const u8);

        assert!(registry.get(1).is_some());

        registry.invalidate(1);

        assert!(registry.get(1).is_none()); // Inactive entries not returned
    }
}
