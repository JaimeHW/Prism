//! Shape Transition Hooks
//!
//! Provides a listener-based notification system for shape transitions in the runtime.
//! The JIT compiler uses this to maintain IC validity when object shapes change.
//!
//! # Architecture
//!
//! Shape transitions occur when:
//! - A new property is added to an object
//! - A property is deleted
//! - The prototype chain is modified
//! - An accessor (getter/setter) is installed
//!
//! The hook system allows the JIT to react to these transitions by bumping the
//! global shape version (for lazy invalidation) or explicitly invalidating specific
//! IC sites (for targeted invalidation).
//!
//! # Thread Safety
//!
//! The listener registration uses `OnceLock` for lock-free access after registration.
//! Listeners themselves must be `Send + Sync` to handle concurrent notifications.
//!
//! # Performance
//!
//! - Zero-cost if no listener is registered (null check only)
//! - Lock-free access through `OnceLock`
//! - Batch notification support for multiple transitions

use crate::ic::invalidation::{InvalidationReason, ShapeVersion};
use prism_runtime::object::shape::ShapeId;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Shape Transition Listener Trait
// =============================================================================

/// Trait for receiving shape transition notifications.
///
/// Implementations react to shape changes to maintain IC validity.
/// The primary implementation is `JitShapeListener` which bumps the global
/// shape version on transitions.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` as notifications may come from any thread.
pub trait ShapeTransitionListener: Send + Sync {
    /// Called when a shape transition occurs (new property added).
    ///
    /// This is the most common event - an object gains a new property and
    /// transitions to a new shape.
    ///
    /// # Arguments
    /// * `old_shape` - The shape before transition
    /// * `new_shape` - The shape after transition
    fn on_transition(&self, old_shape: ShapeId, new_shape: ShapeId);

    /// Called when a property is deleted from an object.
    ///
    /// Property deletions are rare but important for IC validity.
    /// Cached slot offsets may become invalid.
    ///
    /// # Arguments
    /// * `shape` - The shape that had a property deleted
    /// * `property` - The name of the deleted property
    fn on_property_delete(&self, shape: ShapeId, property: &str);

    /// Called when an object's prototype is modified.
    ///
    /// Prototype changes affect method resolution and may invalidate
    /// cached call targets.
    ///
    /// # Arguments
    /// * `shape` - The shape whose prototype changed
    fn on_prototype_change(&self, shape: ShapeId);

    /// Called when an accessor (getter/setter) is installed.
    ///
    /// Accessor installation changes how property access works.
    /// Cached direct slot loads must be invalidated.
    ///
    /// # Arguments
    /// * `shape` - The shape that had an accessor installed
    /// * `property` - The property that now has an accessor
    fn on_accessor_installed(&self, shape: ShapeId, property: &str);

    /// Called for batch transitions (multiple shapes at once).
    ///
    /// Default implementation calls `on_transition` for each pair.
    /// Override for more efficient batch handling.
    ///
    /// # Returns
    /// The number of transitions processed
    fn on_batch_transition(&self, transitions: &[(ShapeId, ShapeId)]) -> usize {
        for &(old, new) in transitions {
            self.on_transition(old, new);
        }
        transitions.len()
    }
}

// =============================================================================
// Global Listener Registration
// =============================================================================

/// Global shape transition listener.
///
/// Uses `OnceLock` for thread-safe, lock-free access after registration.
/// Only one listener can be registered globally.
static SHAPE_LISTENER: OnceLock<Box<dyn ShapeTransitionListener>> = OnceLock::new();

/// Statistics for shape transition notifications.
#[derive(Debug)]
pub struct ShapeHookStats {
    /// Total shape transitions observed
    transitions: AtomicU64,
    /// Total property deletions observed
    deletions: AtomicU64,
    /// Total prototype changes observed
    prototype_changes: AtomicU64,
    /// Total accessor installations observed
    accessor_installs: AtomicU64,
    /// Total batch transitions processed
    batch_transitions: AtomicU64,
}

impl ShapeHookStats {
    /// Create new zeroed stats.
    pub const fn new() -> Self {
        Self {
            transitions: AtomicU64::new(0),
            deletions: AtomicU64::new(0),
            prototype_changes: AtomicU64::new(0),
            accessor_installs: AtomicU64::new(0),
            batch_transitions: AtomicU64::new(0),
        }
    }

    /// Record a transition.
    #[inline]
    pub fn record_transition(&self) {
        self.transitions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a property deletion.
    #[inline]
    pub fn record_deletion(&self) {
        self.deletions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a prototype change.
    #[inline]
    pub fn record_prototype_change(&self) {
        self.prototype_changes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an accessor installation.
    #[inline]
    pub fn record_accessor_install(&self) {
        self.accessor_installs.fetch_add(1, Ordering::Relaxed);
    }

    /// Record batch transitions.
    #[inline]
    pub fn record_batch_transitions(&self, count: u64) {
        self.batch_transitions.fetch_add(count, Ordering::Relaxed);
    }

    /// Get total transitions.
    #[inline]
    pub fn transition_count(&self) -> u64 {
        self.transitions.load(Ordering::Relaxed)
    }

    /// Get total deletions.
    #[inline]
    pub fn deletion_count(&self) -> u64 {
        self.deletions.load(Ordering::Relaxed)
    }

    /// Get total prototype changes.
    #[inline]
    pub fn prototype_change_count(&self) -> u64 {
        self.prototype_changes.load(Ordering::Relaxed)
    }

    /// Get total accessor installations.
    #[inline]
    pub fn accessor_install_count(&self) -> u64 {
        self.accessor_installs.load(Ordering::Relaxed)
    }

    /// Get total batch transitions.
    #[inline]
    pub fn batch_transition_count(&self) -> u64 {
        self.batch_transitions.load(Ordering::Relaxed)
    }

    /// Get total events (all types combined).
    #[inline]
    pub fn total_events(&self) -> u64 {
        self.transition_count()
            + self.deletion_count()
            + self.prototype_change_count()
            + self.accessor_install_count()
    }

    /// Get a snapshot of all stats.
    pub fn snapshot(&self) -> ShapeHookStatsSnapshot {
        ShapeHookStatsSnapshot {
            transitions: self.transition_count(),
            deletions: self.deletion_count(),
            prototype_changes: self.prototype_change_count(),
            accessor_installs: self.accessor_install_count(),
            batch_transitions: self.batch_transition_count(),
        }
    }
}

impl Default for ShapeHookStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable snapshot of hook statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShapeHookStatsSnapshot {
    /// Total shape transitions observed
    pub transitions: u64,
    /// Total property deletions observed
    pub deletions: u64,
    /// Total prototype changes observed
    pub prototype_changes: u64,
    /// Total accessor installations observed
    pub accessor_installs: u64,
    /// Total batch transitions processed
    pub batch_transitions: u64,
}

/// Global statistics for shape hooks.
static GLOBAL_HOOK_STATS: ShapeHookStats = ShapeHookStats::new();

/// Get global hook statistics.
#[inline]
pub fn global_hook_stats() -> &'static ShapeHookStats {
    &GLOBAL_HOOK_STATS
}

// =============================================================================
// Listener Registration
// =============================================================================

/// Register a shape transition listener.
///
/// Only one listener can be registered. Subsequent calls will return `false`.
/// The listener must implement `ShapeTransitionListener` and live for `'static`.
///
/// # Returns
/// `true` if registration succeeded, `false` if already registered.
///
/// # Example
/// ```ignore
/// use prism_jit::ic::hooks::{register_shape_listener, JitShapeListener};
///
/// let success = register_shape_listener(JitShapeListener::new());
/// assert!(success);
/// ```
pub fn register_shape_listener<L: ShapeTransitionListener + 'static>(listener: L) -> bool {
    SHAPE_LISTENER.set(Box::new(listener)).is_ok()
}

/// Register a boxed shape transition listener.
///
/// Variant that accepts a pre-boxed listener.
pub fn register_shape_listener_boxed(listener: Box<dyn ShapeTransitionListener>) -> bool {
    SHAPE_LISTENER.set(listener).is_ok()
}

/// Check if a shape listener is registered.
#[inline]
pub fn has_shape_listener() -> bool {
    SHAPE_LISTENER.get().is_some()
}

/// Get the registered shape listener if any.
#[inline]
pub fn get_shape_listener() -> Option<&'static dyn ShapeTransitionListener> {
    SHAPE_LISTENER.get().map(|b| b.as_ref())
}

// =============================================================================
// Notification Functions
// =============================================================================

/// Notify the registered listener of a shape transition.
///
/// No-op if no listener is registered.
/// Returns `true` if a listener was notified.
#[inline]
pub fn notify_shape_transition(old_shape: ShapeId, new_shape: ShapeId) -> bool {
    if let Some(listener) = SHAPE_LISTENER.get() {
        GLOBAL_HOOK_STATS.record_transition();
        listener.on_transition(old_shape, new_shape);
        true
    } else {
        false
    }
}

/// Notify the registered listener of a property deletion.
///
/// No-op if no listener is registered.
/// Returns `true` if a listener was notified.
#[inline]
pub fn notify_property_delete(shape: ShapeId, property: &str) -> bool {
    if let Some(listener) = SHAPE_LISTENER.get() {
        GLOBAL_HOOK_STATS.record_deletion();
        listener.on_property_delete(shape, property);
        true
    } else {
        false
    }
}

/// Notify the registered listener of a prototype change.
///
/// No-op if no listener is registered.
/// Returns `true` if a listener was notified.
#[inline]
pub fn notify_prototype_change(shape: ShapeId) -> bool {
    if let Some(listener) = SHAPE_LISTENER.get() {
        GLOBAL_HOOK_STATS.record_prototype_change();
        listener.on_prototype_change(shape);
        true
    } else {
        false
    }
}

/// Notify the registered listener of an accessor installation.
///
/// No-op if no listener is registered.
/// Returns `true` if a listener was notified.
#[inline]
pub fn notify_accessor_installed(shape: ShapeId, property: &str) -> bool {
    if let Some(listener) = SHAPE_LISTENER.get() {
        GLOBAL_HOOK_STATS.record_accessor_install();
        listener.on_accessor_installed(shape, property);
        true
    } else {
        false
    }
}

/// Notify the registered listener of batch shape transitions.
///
/// No-op if no listener is registered.
/// Returns the number of transitions processed (0 if no listener).
#[inline]
pub fn notify_batch_transitions(transitions: &[(ShapeId, ShapeId)]) -> usize {
    if let Some(listener) = SHAPE_LISTENER.get() {
        let count = listener.on_batch_transition(transitions);
        GLOBAL_HOOK_STATS.record_batch_transitions(count as u64);
        count
    } else {
        0
    }
}

// =============================================================================
// Null Listener (for testing)
// =============================================================================

/// A no-op listener for testing.
///
/// Does nothing on any notification. Useful for testing registration.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullListener;

impl ShapeTransitionListener for NullListener {
    fn on_transition(&self, _old_shape: ShapeId, _new_shape: ShapeId) {}
    fn on_property_delete(&self, _shape: ShapeId, _property: &str) {}
    fn on_prototype_change(&self, _shape: ShapeId) {}
    fn on_accessor_installed(&self, _shape: ShapeId, _property: &str) {}
}

// =============================================================================
// Recording Listener (for testing)
// =============================================================================

use std::sync::Mutex;

/// Event types recorded for testing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedEvent {
    /// Shape transition
    Transition { old: ShapeId, new: ShapeId },
    /// Property deletion
    PropertyDelete { shape: ShapeId, property: String },
    /// Prototype change
    PrototypeChange { shape: ShapeId },
    /// Accessor installation
    AccessorInstalled { shape: ShapeId, property: String },
}

/// A listener that records all events for testing.
#[derive(Debug, Default)]
pub struct RecordingListener {
    events: Mutex<Vec<RecordedEvent>>,
}

impl RecordingListener {
    /// Create a new recording listener.
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    /// Get all recorded events.
    pub fn events(&self) -> Vec<RecordedEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Get the number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Clear all recorded events.
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }
}

impl ShapeTransitionListener for RecordingListener {
    fn on_transition(&self, old_shape: ShapeId, new_shape: ShapeId) {
        self.events.lock().unwrap().push(RecordedEvent::Transition {
            old: old_shape,
            new: new_shape,
        });
    }

    fn on_property_delete(&self, shape: ShapeId, property: &str) {
        self.events
            .lock()
            .unwrap()
            .push(RecordedEvent::PropertyDelete {
                shape,
                property: property.to_string(),
            });
    }

    fn on_prototype_change(&self, shape: ShapeId) {
        self.events
            .lock()
            .unwrap()
            .push(RecordedEvent::PrototypeChange { shape });
    }

    fn on_accessor_installed(&self, shape: ShapeId, property: &str) {
        self.events
            .lock()
            .unwrap()
            .push(RecordedEvent::AccessorInstalled {
                shape,
                property: property.to_string(),
            });
    }
}

// =============================================================================
// Counting Listener (for testing)
// =============================================================================

/// A listener that counts events without recording details.
///
/// More efficient than `RecordingListener` for high-volume testing.
#[derive(Debug, Default)]
pub struct CountingListener {
    transitions: AtomicU64,
    deletions: AtomicU64,
    prototype_changes: AtomicU64,
    accessor_installs: AtomicU64,
}

impl CountingListener {
    /// Create a new counting listener.
    pub fn new() -> Self {
        Self {
            transitions: AtomicU64::new(0),
            deletions: AtomicU64::new(0),
            prototype_changes: AtomicU64::new(0),
            accessor_installs: AtomicU64::new(0),
        }
    }

    /// Get transition count.
    #[inline]
    pub fn transition_count(&self) -> u64 {
        self.transitions.load(Ordering::Relaxed)
    }

    /// Get deletion count.
    #[inline]
    pub fn deletion_count(&self) -> u64 {
        self.deletions.load(Ordering::Relaxed)
    }

    /// Get prototype change count.
    #[inline]
    pub fn prototype_change_count(&self) -> u64 {
        self.prototype_changes.load(Ordering::Relaxed)
    }

    /// Get accessor install count.
    #[inline]
    pub fn accessor_install_count(&self) -> u64 {
        self.accessor_installs.load(Ordering::Relaxed)
    }

    /// Get total event count.
    #[inline]
    pub fn total(&self) -> u64 {
        self.transition_count()
            + self.deletion_count()
            + self.prototype_change_count()
            + self.accessor_install_count()
    }
}

impl ShapeTransitionListener for CountingListener {
    fn on_transition(&self, _old_shape: ShapeId, _new_shape: ShapeId) {
        self.transitions.fetch_add(1, Ordering::Relaxed);
    }

    fn on_property_delete(&self, _shape: ShapeId, _property: &str) {
        self.deletions.fetch_add(1, Ordering::Relaxed);
    }

    fn on_prototype_change(&self, _shape: ShapeId) {
        self.prototype_changes.fetch_add(1, Ordering::Relaxed);
    }

    fn on_accessor_installed(&self, _shape: ShapeId, _property: &str) {
        self.accessor_installs.fetch_add(1, Ordering::Relaxed);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
