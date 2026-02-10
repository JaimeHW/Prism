//! Call Dispatcher Module
//!
//! High-performance class call protocol implementation with JIT-friendly
//! specializations for different constructor patterns.
//!
//! Handles: `cls.__call__(*args, **kwargs)`
//! Protocol: `cls.__new__(cls) → obj, then obj.__init__(*args, **kwargs)`
//!
//! ## Performance Design
//!
//! - Pre-computed specializations per class
//! - Fast-path for common patterns (default new, no init)
//! - JIT can specialize based on `CallSpecialization`

use prism_core::Value;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;

use super::class::{ClassFlags, PyClassObject};
use super::mro::ClassId;

// =============================================================================
// Call Errors
// =============================================================================

/// Errors during class call.
#[derive(Debug, Clone, PartialEq)]
pub enum CallError {
    /// Class not found in registry.
    ClassNotFound { class_id: ClassId },

    /// __new__ returned wrong type.
    NewReturnedWrongType { expected: String, actual: String },

    /// __init__ returned non-None.
    InitReturnedNonNone,

    /// Argument error.
    ArgumentError { message: String },

    /// Method not callable.
    NotCallable { type_name: String },

    /// Generic error.
    Other { message: String },
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClassNotFound { class_id } => {
                write!(f, "class not found: {:?}", class_id)
            }
            Self::NewReturnedWrongType { expected, actual } => {
                write!(f, "__new__ returned {}, expected {}", actual, expected)
            }
            Self::InitReturnedNonNone => {
                write!(f, "__init__ should return None")
            }
            Self::ArgumentError { message } => {
                write!(f, "argument error: {}", message)
            }
            Self::NotCallable { type_name } => {
                write!(f, "'{}' object is not callable", type_name)
            }
            Self::Other { message } => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for CallError {}

/// Result type for call operations.
pub type CallResult<T> = Result<T, CallError>;

// =============================================================================
// Method Slot
// =============================================================================

/// Reference to a method in the class hierarchy.
///
/// Used by CallDispatcher to cache method locations for fast dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MethodSlot {
    /// Class that defines this method.
    pub class_id: ClassId,

    /// Offset in the class's method table.
    /// We use interned string for name-based lookup.
    pub method_name: u32,
}

impl MethodSlot {
    /// Create a new method slot.
    pub const fn new(class_id: ClassId, method_name: u32) -> Self {
        Self {
            class_id,
            method_name,
        }
    }
}

// =============================================================================
// Call Specialization
// =============================================================================

/// JIT-friendly call specialization.
///
/// Encodes the constructor pattern for a class, enabling fast-path
/// dispatch without runtime method resolution.
///
/// ## Performance
///
/// - `DefaultBoth`: Fastest - just allocate, no method calls
/// - `DefaultNew`: Medium - allocate + call __init__
/// - `CustomNew`: Medium - call __new__ + no __init__
/// - `CustomBoth`: Slowest - full protocol
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallSpecialization {
    /// Default `__new__` + no `__init__` (fastest).
    ///
    /// Just allocates the instance with no initialization.
    DefaultBoth,

    /// Default `__new__` + custom `__init__`.
    ///
    /// Allocates then calls the custom initializer.
    DefaultNew {
        /// Slot for the __init__ method.
        init_class: ClassId,
    },

    /// Custom `__new__` + no `__init__`.
    ///
    /// Calls custom allocator but no initialization.
    CustomNew {
        /// Slot for the __new__ method.
        new_class: ClassId,
    },

    /// Custom `__new__` + custom `__init__` (slowest).
    ///
    /// Full Python protocol.
    CustomBoth {
        /// Slot for the __new__ method.
        new_class: ClassId,
        /// Slot for the __init__ method.
        init_class: ClassId,
    },

    /// Immutable singleton (e.g., None, True, False, small ints).
    ///
    /// Returns the cached singleton value.
    Singleton(Value),

    /// Metaclass with custom `__call__`.
    ///
    /// Delegates to the metaclass's __call__.
    MetaCall {
        /// Slot for the metaclass __call__.
        meta_class: ClassId,
    },

    /// Not yet analyzed.
    Unknown,
}

impl Default for CallSpecialization {
    fn default() -> Self {
        Self::Unknown
    }
}

impl CallSpecialization {
    /// Check if this is a fast-path specialization.
    #[inline]
    pub fn is_fast_path(&self) -> bool {
        matches!(self, Self::DefaultBoth | Self::Singleton(_))
    }

    /// Check if this specialization requires __new__ call.
    #[inline]
    pub fn needs_new(&self) -> bool {
        matches!(self, Self::CustomNew { .. } | Self::CustomBoth { .. })
    }

    /// Check if this specialization requires __init__ call.
    #[inline]
    pub fn needs_init(&self) -> bool {
        matches!(self, Self::DefaultNew { .. } | Self::CustomBoth { .. })
    }

    /// Compute specialization from class flags.
    pub fn from_class(class: &PyClassObject) -> Self {
        let flags = class.flags();

        let has_new = flags.contains(ClassFlags::HAS_NEW);
        let has_init = flags.contains(ClassFlags::HAS_INIT);

        match (has_new, has_init) {
            (false, false) => Self::DefaultBoth,
            (false, true) => Self::DefaultNew {
                init_class: class.class_id(),
            },
            (true, false) => Self::CustomNew {
                new_class: class.class_id(),
            },
            (true, true) => Self::CustomBoth {
                new_class: class.class_id(),
                init_class: class.class_id(),
            },
        }
    }
}

// =============================================================================
// Call Dispatcher
// =============================================================================

/// Call dispatcher for class instantiation.
///
/// Manages per-class call specializations and provides fast-path
/// dispatch for common constructor patterns.
///
/// ## Usage
///
/// ```ignore
/// let dispatcher = CallDispatcher::new();
/// let result = dispatcher.call(&class, args, kwargs, registry)?;
/// ```
///
/// ## JIT Integration
///
/// The JIT can query `get_specialization()` and emit specialized code
/// for each pattern, eliminating dispatch overhead.
pub struct CallDispatcher {
    /// Per-class call specialization cache.
    specializations: std::sync::RwLock<FxHashMap<ClassId, CallSpecialization>>,
}

impl Default for CallDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl CallDispatcher {
    /// Create a new call dispatcher.
    pub fn new() -> Self {
        Self {
            specializations: std::sync::RwLock::new(FxHashMap::default()),
        }
    }

    /// Get cached specialization for a class.
    #[inline]
    pub fn get_specialization(&self, class_id: ClassId) -> Option<CallSpecialization> {
        self.specializations.read().unwrap().get(&class_id).copied()
    }

    /// Register specialization for a class.
    pub fn register(&self, class_id: ClassId, spec: CallSpecialization) {
        self.specializations.write().unwrap().insert(class_id, spec);
    }

    /// Compute and cache specialization for a class.
    pub fn analyze(&self, class: &PyClassObject) -> CallSpecialization {
        let class_id = class.class_id();

        // Check cache first
        if let Some(spec) = self.get_specialization(class_id) {
            return spec;
        }

        // Compute specialization
        let spec = CallSpecialization::from_class(class);

        // Cache it
        self.register(class_id, spec);

        spec
    }

    /// Invalidate specialization (e.g., when class is modified).
    pub fn invalidate(&self, class_id: ClassId) {
        self.specializations.write().unwrap().remove(&class_id);
    }

    /// Clear all cached specializations.
    pub fn clear(&self) {
        self.specializations.write().unwrap().clear();
    }

    /// Number of cached specializations.
    pub fn len(&self) -> usize {
        self.specializations.read().unwrap().len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// =============================================================================
// Instance Allocator Trait
// =============================================================================

/// Trait for instance allocation.
///
/// Implemented by the runtime to provide actual allocation.
/// CallDispatcher uses this to allocate instances.
pub trait InstanceAllocator {
    /// Allocate a new instance of the given class.
    fn alloc_instance(&self, class: &PyClassObject) -> CallResult<Value>;

    /// Allocate from pool if available.
    fn alloc_from_pool(&self, class_id: ClassId) -> Option<Value>;

    /// Return instance to pool.
    fn return_to_pool(&self, class_id: ClassId, instance: Value);
}

// =============================================================================
// Call Context
// =============================================================================

/// Context for a call operation.
///
/// Contains all information needed to execute a class call.
#[derive(Debug)]
pub struct CallContext<'a> {
    /// Positional arguments.
    pub args: &'a [Value],

    /// Keyword arguments (if any).
    pub kwargs: Option<&'a [(InternedString, Value)]>,

    /// Whether this is a super() call.
    pub is_super_call: bool,
}

impl<'a> CallContext<'a> {
    /// Create a new call context.
    pub fn new(args: &'a [Value]) -> Self {
        Self {
            args,
            kwargs: None,
            is_super_call: false,
        }
    }

    /// Create with keyword arguments.
    pub fn with_kwargs(args: &'a [Value], kwargs: &'a [(InternedString, Value)]) -> Self {
        Self {
            args,
            kwargs: Some(kwargs),
            is_super_call: false,
        }
    }

    /// Mark as super call.
    pub fn as_super_call(mut self) -> Self {
        self.is_super_call = true;
        self
    }

    /// Check if there are any arguments.
    #[inline]
    pub fn has_args(&self) -> bool {
        !self.args.is_empty()
    }

    /// Check if there are any kwargs.
    #[inline]
    pub fn has_kwargs(&self) -> bool {
        self.kwargs.map_or(false, |k| !k.is_empty())
    }

    /// Total argument count.
    #[inline]
    pub fn arg_count(&self) -> usize {
        self.args.len() + self.kwargs.map_or(0, |k| k.len())
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Call dispatch statistics for profiling.
#[derive(Default, Debug)]
pub struct CallStats {
    /// Total calls via DefaultBoth path.
    pub default_both_calls: std::sync::atomic::AtomicU64,

    /// Total calls via DefaultNew path.
    pub default_new_calls: std::sync::atomic::AtomicU64,

    /// Total calls via CustomNew path.
    pub custom_new_calls: std::sync::atomic::AtomicU64,

    /// Total calls via CustomBoth path.
    pub custom_both_calls: std::sync::atomic::AtomicU64,

    /// Total singleton returns.
    pub singleton_returns: std::sync::atomic::AtomicU64,

    /// Total metacall dispatches.
    pub meta_calls: std::sync::atomic::AtomicU64,

    /// Cache hits.
    pub cache_hits: std::sync::atomic::AtomicU64,

    /// Cache misses (requiring analysis).
    pub cache_misses: std::sync::atomic::AtomicU64,
}

impl CallStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a call for the given specialization.
    pub fn record_call(&self, spec: &CallSpecialization) {
        use std::sync::atomic::Ordering;
        match spec {
            CallSpecialization::DefaultBoth => {
                self.default_both_calls.fetch_add(1, Ordering::Relaxed);
            }
            CallSpecialization::DefaultNew { .. } => {
                self.default_new_calls.fetch_add(1, Ordering::Relaxed);
            }
            CallSpecialization::CustomNew { .. } => {
                self.custom_new_calls.fetch_add(1, Ordering::Relaxed);
            }
            CallSpecialization::CustomBoth { .. } => {
                self.custom_both_calls.fetch_add(1, Ordering::Relaxed);
            }
            CallSpecialization::Singleton(_) => {
                self.singleton_returns.fetch_add(1, Ordering::Relaxed);
            }
            CallSpecialization::MetaCall { .. } => {
                self.meta_calls.fetch_add(1, Ordering::Relaxed);
            }
            CallSpecialization::Unknown => {}
        }
    }

    /// Record cache hit.
    pub fn record_cache_hit(&self) {
        use std::sync::atomic::Ordering;
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss.
    pub fn record_cache_miss(&self) {
        use std::sync::atomic::Ordering;
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::intern;

    // =========================================================================
    // CallError Tests
    // =========================================================================

    #[test]
    fn test_call_error_display() {
        let err = CallError::ClassNotFound {
            class_id: ClassId(123),
        };
        assert!(err.to_string().contains("123"));

        let err = CallError::NewReturnedWrongType {
            expected: "MyClass".to_string(),
            actual: "OtherClass".to_string(),
        };
        assert!(err.to_string().contains("MyClass"));
        assert!(err.to_string().contains("OtherClass"));

        let err = CallError::InitReturnedNonNone;
        assert!(err.to_string().contains("None"));

        let err = CallError::ArgumentError {
            message: "too many args".to_string(),
        };
        assert!(err.to_string().contains("too many args"));

        let err = CallError::NotCallable {
            type_name: "int".to_string(),
        };
        assert!(err.to_string().contains("int"));
    }

    // =========================================================================
    // MethodSlot Tests
    // =========================================================================

    #[test]
    fn test_method_slot_new() {
        let slot = MethodSlot::new(ClassId(42), 5);
        assert_eq!(slot.class_id, ClassId(42));
        assert_eq!(slot.method_name, 5);
    }

    // =========================================================================
    // CallSpecialization Tests
    // =========================================================================

    #[test]
    fn test_specialization_default() {
        let spec = CallSpecialization::default();
        assert_eq!(spec, CallSpecialization::Unknown);
    }

    #[test]
    fn test_specialization_is_fast_path() {
        assert!(CallSpecialization::DefaultBoth.is_fast_path());
        assert!(CallSpecialization::Singleton(Value::int_unchecked(0)).is_fast_path());
        assert!(
            !CallSpecialization::DefaultNew {
                init_class: ClassId(1)
            }
            .is_fast_path()
        );
        assert!(
            !CallSpecialization::CustomNew {
                new_class: ClassId(1)
            }
            .is_fast_path()
        );
        assert!(
            !CallSpecialization::CustomBoth {
                new_class: ClassId(1),
                init_class: ClassId(2),
            }
            .is_fast_path()
        );
    }

    #[test]
    fn test_specialization_needs_new() {
        assert!(!CallSpecialization::DefaultBoth.needs_new());
        assert!(
            !CallSpecialization::DefaultNew {
                init_class: ClassId(1)
            }
            .needs_new()
        );
        assert!(
            CallSpecialization::CustomNew {
                new_class: ClassId(1)
            }
            .needs_new()
        );
        assert!(
            CallSpecialization::CustomBoth {
                new_class: ClassId(1),
                init_class: ClassId(2),
            }
            .needs_new()
        );
    }

    #[test]
    fn test_specialization_needs_init() {
        assert!(!CallSpecialization::DefaultBoth.needs_init());
        assert!(
            CallSpecialization::DefaultNew {
                init_class: ClassId(1)
            }
            .needs_init()
        );
        assert!(
            !CallSpecialization::CustomNew {
                new_class: ClassId(1)
            }
            .needs_init()
        );
        assert!(
            CallSpecialization::CustomBoth {
                new_class: ClassId(1),
                init_class: ClassId(2),
            }
            .needs_init()
        );
    }

    #[test]
    fn test_specialization_from_class_default_both() {
        use super::super::class::PyClassObject;
        let class = PyClassObject::new_simple(intern("Simple"));
        let spec = CallSpecialization::from_class(&class);
        assert_eq!(spec, CallSpecialization::DefaultBoth);
    }

    #[test]
    fn test_specialization_from_class_with_init() {
        use super::super::class::PyClassObject;
        let mut class = PyClassObject::new_simple(intern("WithInit"));
        class.mark_has_init();
        let spec = CallSpecialization::from_class(&class);
        match spec {
            CallSpecialization::DefaultNew { init_class } => {
                assert_eq!(init_class, class.class_id());
            }
            _ => panic!("Expected DefaultNew, got {:?}", spec),
        }
    }

    #[test]
    fn test_specialization_from_class_with_new() {
        use super::super::class::PyClassObject;
        let mut class = PyClassObject::new_simple(intern("WithNew"));
        class.mark_has_new();
        let spec = CallSpecialization::from_class(&class);
        match spec {
            CallSpecialization::CustomNew { new_class } => {
                assert_eq!(new_class, class.class_id());
            }
            _ => panic!("Expected CustomNew, got {:?}", spec),
        }
    }

    #[test]
    fn test_specialization_from_class_with_both() {
        use super::super::class::PyClassObject;
        let mut class = PyClassObject::new_simple(intern("WithBoth"));
        class.mark_has_new();
        class.mark_has_init();
        let spec = CallSpecialization::from_class(&class);
        match spec {
            CallSpecialization::CustomBoth {
                new_class,
                init_class,
            } => {
                assert_eq!(new_class, class.class_id());
                assert_eq!(init_class, class.class_id());
            }
            _ => panic!("Expected CustomBoth, got {:?}", spec),
        }
    }

    // =========================================================================
    // CallDispatcher Tests
    // =========================================================================

    #[test]
    fn test_dispatcher_new() {
        let dispatcher = CallDispatcher::new();
        assert!(dispatcher.is_empty());
        assert_eq!(dispatcher.len(), 0);
    }

    #[test]
    fn test_dispatcher_register_and_get() {
        let dispatcher = CallDispatcher::new();

        dispatcher.register(ClassId(100), CallSpecialization::DefaultBoth);
        dispatcher.register(
            ClassId(101),
            CallSpecialization::DefaultNew {
                init_class: ClassId(101),
            },
        );

        assert_eq!(
            dispatcher.get_specialization(ClassId(100)),
            Some(CallSpecialization::DefaultBoth)
        );
        assert_eq!(
            dispatcher.get_specialization(ClassId(101)),
            Some(CallSpecialization::DefaultNew {
                init_class: ClassId(101)
            })
        );
        assert_eq!(dispatcher.get_specialization(ClassId(999)), None);
        assert_eq!(dispatcher.len(), 2);
    }

    #[test]
    fn test_dispatcher_analyze() {
        use super::super::class::PyClassObject;

        let dispatcher = CallDispatcher::new();
        let class = PyClassObject::new_simple(intern("AnalyzeTest"));

        let spec = dispatcher.analyze(&class);
        assert_eq!(spec, CallSpecialization::DefaultBoth);

        // Should be cached now
        assert_eq!(
            dispatcher.get_specialization(class.class_id()),
            Some(CallSpecialization::DefaultBoth)
        );
        assert_eq!(dispatcher.len(), 1);
    }

    #[test]
    fn test_dispatcher_invalidate() {
        let dispatcher = CallDispatcher::new();

        dispatcher.register(ClassId(100), CallSpecialization::DefaultBoth);
        assert!(dispatcher.get_specialization(ClassId(100)).is_some());

        dispatcher.invalidate(ClassId(100));
        assert!(dispatcher.get_specialization(ClassId(100)).is_none());
    }

    #[test]
    fn test_dispatcher_clear() {
        let dispatcher = CallDispatcher::new();

        dispatcher.register(ClassId(100), CallSpecialization::DefaultBoth);
        dispatcher.register(ClassId(101), CallSpecialization::DefaultBoth);
        assert_eq!(dispatcher.len(), 2);

        dispatcher.clear();
        assert!(dispatcher.is_empty());
    }

    // =========================================================================
    // CallContext Tests
    // =========================================================================

    #[test]
    fn test_call_context_new() {
        let args = vec![Value::int_unchecked(1), Value::int_unchecked(2)];
        let ctx = CallContext::new(&args);

        assert!(ctx.has_args());
        assert!(!ctx.has_kwargs());
        assert_eq!(ctx.arg_count(), 2);
        assert!(!ctx.is_super_call);
    }

    #[test]
    fn test_call_context_empty() {
        let args: Vec<Value> = vec![];
        let ctx = CallContext::new(&args);

        assert!(!ctx.has_args());
        assert!(!ctx.has_kwargs());
        assert_eq!(ctx.arg_count(), 0);
    }

    #[test]
    fn test_call_context_with_kwargs() {
        let args = vec![Value::int_unchecked(1)];
        let kwargs = vec![
            (intern("x"), Value::int_unchecked(10)),
            (intern("y"), Value::int_unchecked(20)),
        ];
        let ctx = CallContext::with_kwargs(&args, &kwargs);

        assert!(ctx.has_args());
        assert!(ctx.has_kwargs());
        assert_eq!(ctx.arg_count(), 3);
    }

    #[test]
    fn test_call_context_as_super_call() {
        let args = vec![Value::int_unchecked(1)];
        let ctx = CallContext::new(&args).as_super_call();

        assert!(ctx.is_super_call);
    }

    // =========================================================================
    // CallStats Tests
    // =========================================================================

    #[test]
    fn test_call_stats_new() {
        let stats = CallStats::new();
        use std::sync::atomic::Ordering;
        assert_eq!(stats.default_both_calls.load(Ordering::Relaxed), 0);
        assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_call_stats_record_call() {
        let stats = CallStats::new();
        use std::sync::atomic::Ordering;

        stats.record_call(&CallSpecialization::DefaultBoth);
        stats.record_call(&CallSpecialization::DefaultBoth);
        stats.record_call(&CallSpecialization::DefaultNew {
            init_class: ClassId(1),
        });

        assert_eq!(stats.default_both_calls.load(Ordering::Relaxed), 2);
        assert_eq!(stats.default_new_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_call_stats_record_cache() {
        let stats = CallStats::new();
        use std::sync::atomic::Ordering;

        stats.record_cache_hit();
        stats.record_cache_hit();
        stats.record_cache_miss();

        assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 2);
        assert_eq!(stats.cache_misses.load(Ordering::Relaxed), 1);
    }
}
