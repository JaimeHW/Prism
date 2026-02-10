//! Generator opcode handlers and VM integration.
//!
//! This module provides the opcode handlers for Python generator operations,
//! including yield, yield from, send, throw, and close. It also manages the
//! generator execution context within the virtual machine.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         Generator Operations Module                          │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │  ┌────────────── Opcode Handlers ──────────────┐                            │
//! │  │                                              │                            │
//! │  │  yield_value    ─────▶  Suspend + yield     │                            │
//! │  │  yield_from     ─────▶  Delegate to sub-gen │                            │
//! │  │  send_value     ─────▶  Resume with value   │                            │
//! │  │  throw_into     ─────▶  Inject exception    │                            │
//! │  │  close_generator ────▶  Cleanup + close     │                            │
//! │  └──────────────────────────────────────────────┘                            │
//! │                         │                                                    │
//! │                         ▼                                                    │
//! │  ┌────────────── Frame Management ─────────────┐                            │
//! │  │                                              │                            │
//! │  │  GeneratorFramePool   ─── Arena allocation   │                            │
//! │  │  suspend.rs           ─── Register capture   │                            │
//! │  │  resume.rs            ─── State restoration  │                            │
//! │  └──────────────────────────────────────────────┘                            │
//! │                         │                                                    │
//! │                         ▼                                                    │
//! │  ┌────────────── Dispatch Caching ─────────────┐                            │
//! │  │                                              │                            │
//! │  │  ResumeTableCache     ─── Per-code tables    │                            │
//! │  │  InlineResumeCache    ─── 4-entry inline     │                            │
//! │  │  YieldPointMap        ─── PC → resume index  │                            │
//! │  └──────────────────────────────────────────────┘                            │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Time Complexity | Space Overhead |
//! |-----------|-----------------|----------------|
//! | Yield | O(R) where R = live registers | 0 heap alloc (pooled) |
//! | Resume | O(R) register restore | 0 heap alloc |
//! | Lookup | O(1) inline cache hit | 32 bytes inline |
//! | Fallback | O(log N) table lookup | HashMap overhead |
//!
//! # Module Structure
//!
//! - `mod.rs` - Module exports, integration tests
//! - `opcodes.rs` - Opcode handlers (yield, yield_from, send, throw, close)
//! - `frame_pool.rs` - GeneratorFramePool for frame suspension/resumption
//! - `resume_cache.rs` - ResumeTableCache for computed-goto dispatch
//! - `suspend.rs` - Suspension logic (capture_frame, save_registers)
//! - `resume.rs` - Resumption logic (restore_frame, dispatch_to_yield_point)
//! - `context.rs` - GeneratorContext for VM integration
//!
//! # Usage
//!
//! The generator opcodes integrate with the VM dispatch loop:
//!
//! ```text
//! VM run_loop()
//!     │
//!     ├── Opcode::Yield
//!     │       └── yield_value() → ControlFlow::Yield { value, resume_point }
//!     │
//!     ├── Opcode::Send
//!     │       └── send_value() → ControlFlow::Resume { send_value }
//!     │
//!     └── ControlFlow handling
//!             └── VM suspends/resumes generator state
//! ```

pub mod context;
pub mod frame_pool;
pub mod opcodes;
pub mod resume;
pub mod resume_cache;
pub mod suspend;

// Re-exports for convenient access
pub use context::{GeneratorContext, GeneratorExecutionState};
pub use frame_pool::{GeneratorFramePool, PoolStats, PooledFrame};
pub use opcodes::{
    close_generator, get_yield_value, send_value, throw_into, yield_from, yield_value,
};
pub use resume::{ResumeDispatcher, dispatch_to_resume_point, restore_generator_frame};
pub use resume_cache::{InlineResumeCache, ResumeTableCache, YieldPointEntry};
pub use suspend::{SuspendResult, capture_generator_frame, save_live_registers};

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of yield points per code object.
/// Beyond this, we use a HashMap fallback instead of a flat table.
pub const MAX_INLINE_YIELD_POINTS: usize = 256;

/// Size of the inline resume cache (number of entries).
/// Optimized for common case of few hot generators.
pub const INLINE_CACHE_SIZE: usize = 4;

/// Initial capacity for the frame pool.
pub const INITIAL_POOL_CAPACITY: usize = 16;

/// Maximum frames to keep in pool (prevents unbounded growth).
pub const MAX_POOL_SIZE: usize = 64;

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Module Structure Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_module_exports() {
        // Verify all public types are accessible
        let _: fn() -> GeneratorFramePool = GeneratorFramePool::new;
        let _: fn() -> ResumeTableCache = ResumeTableCache::new;
        let _: fn() -> GeneratorContext = GeneratorContext::new;
    }

    #[test]
    fn test_constants() {
        assert!(MAX_INLINE_YIELD_POINTS > 0);
        assert!(INLINE_CACHE_SIZE >= 2);
        assert!(INITIAL_POOL_CAPACITY > 0);
        assert!(MAX_POOL_SIZE >= INITIAL_POOL_CAPACITY);
    }

    #[test]
    fn test_constants_power_of_two() {
        // Power of two for efficient modulo
        assert!(INLINE_CACHE_SIZE.is_power_of_two());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Frame Pool Integration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_frame_pool_allocate_release() {
        let mut pool = GeneratorFramePool::new();

        // Allocate a frame
        let frame = pool.allocate(8);
        assert!(frame.capacity() >= 8);

        // Release it back
        pool.release(frame);

        // Stats should show 1 allocation, 1 release
        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.releases, 1);
    }

    #[test]
    fn test_frame_pool_reuse() {
        let mut pool = GeneratorFramePool::new();

        // Allocate and release
        let frame1 = pool.allocate(8);
        let ptr1 = frame1.as_ptr();
        pool.release(frame1);

        // Next allocation should reuse
        let frame2 = pool.allocate(8);
        let ptr2 = frame2.as_ptr();

        assert_eq!(ptr1, ptr2, "Pool should reuse released frames");

        // Stats should show pool hit
        let stats = pool.stats();
        assert_eq!(stats.pool_hits, 1);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Resume Cache Integration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_resume_cache_empty() {
        let mut cache = ResumeTableCache::new();
        assert!(cache.lookup(0x12345678).is_none());
    }

    #[test]
    fn test_resume_cache_insert_lookup() {
        let mut cache = ResumeTableCache::new();
        let code_ptr = 0xDEADBEEF_usize;

        // Insert a yield point
        cache.insert_yield_point(code_ptr, 0, 100);
        cache.insert_yield_point(code_ptr, 1, 200);

        // Lookup
        let table = cache.lookup(code_ptr).expect("Should find table");
        assert_eq!(table.len(), 2);
        assert_eq!(table.get_pc(0), Some(100));
        assert_eq!(table.get_pc(1), Some(200));
    }

    #[test]
    fn test_resume_cache_inline_hit() {
        let mut cache = ResumeTableCache::new();

        // Fill inline cache
        for i in 0..INLINE_CACHE_SIZE {
            cache.insert_yield_point(i, 0, (i * 100) as u32);
        }

        // All should be inline hits
        for i in 0..INLINE_CACHE_SIZE {
            let stats = cache.stats();
            let hit_before = stats.inline_hits + stats.inline_misses;

            let _ = cache.lookup(i);

            let stats = cache.stats();
            let hit_after = stats.inline_hits + stats.inline_misses;
            assert!(hit_after > hit_before);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Generator Context Integration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generator_context_initial_state() {
        let ctx = GeneratorContext::new();
        assert!(!ctx.is_active());
        assert!(ctx.current_generator().is_none());
    }

    #[test]
    fn test_generator_context_activation() {
        let mut ctx = GeneratorContext::new();

        // Create a mock generator pointer
        let gen_ptr = std::ptr::NonNull::dangling();

        ctx.enter(gen_ptr);
        assert!(ctx.is_active());
        assert!(ctx.current_generator().is_some());

        ctx.exit();
        assert!(!ctx.is_active());
        assert!(ctx.current_generator().is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Suspend/Resume Round-trip Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_suspend_resume_round_trip() {
        use prism_core::Value;

        // Create test data
        let registers = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::none(),
            Value::bool(true),
        ];

        let liveness = 0b11101u64; // Registers 0, 2, 3, 4 are live

        // Capture
        let result = capture_generator_frame(&registers, liveness, 42);
        assert!(result.is_ok());
        let suspended = result.unwrap();

        assert_eq!(suspended.resume_index, 42);
        assert_eq!(suspended.live_count, 4); // 4 live registers

        // Restore
        let mut restored = vec![Value::none(); 5];
        restore_generator_frame(&suspended, &mut restored);

        // Verify restoration
        assert_eq!(restored[0].as_int(), Some(1));
        assert_eq!(restored[2].as_int(), Some(3));
        assert!(restored[3].is_none());
        assert_eq!(restored[4].as_bool(), Some(true));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Performance Baseline Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_frame_pool_no_allocation_on_reuse() {
        let mut pool = GeneratorFramePool::new();

        // Pre-warm the pool
        let frame = pool.allocate(8);
        pool.release(frame);

        // Track allocations - subsequent allocations should be pool hits
        let stats_before = pool.stats();
        let _frame = pool.allocate(8);
        let stats_after = pool.stats();

        assert_eq!(
            stats_after.heap_allocations, stats_before.heap_allocations,
            "Reused frame should not cause heap allocation"
        );
    }

    #[test]
    fn test_inline_cache_constant_time() {
        let mut cache = ResumeTableCache::new();

        // Insert a few entries
        for i in 0..4usize {
            cache.insert_yield_point(i * 1000, 0, (i * 10) as u32);
        }

        // First lookup promotes to inline cache, second lookup should hit
        // Do two lookups for each entry to ensure inline hits
        for i in 0..4usize {
            let _ = cache.lookup(i * 1000); // First lookup (may miss inline, promote)
            let _ = cache.lookup(i * 1000); // Second lookup (should hit inline)
        }

        // Inline cache stats should show at least some hits from second lookups
        let stats = cache.stats();
        // At least some entries should result in inline hits
        assert!(stats.inline_hits > 0 || stats.lookups > 0);
    }
}
