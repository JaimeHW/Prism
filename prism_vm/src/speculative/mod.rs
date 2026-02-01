//! Speculative Dispatch System for Type-Specialized Arithmetic.
//!
//! This module provides O(1) speculation lookup for inline fast-paths in
//! arithmetic operations. When type feedback indicates monomorphic behavior,
//! handlers can skip expensive type checks and execute specialized code.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Generic Add Handler                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │  1. Fetch operands                                           │
//! │  2. O(1) speculation cache lookup                            │
//! │  3. If INT_INT → inline int fast path                        │
//! │  4. If FLOAT_FLOAT → inline float fast path                  │
//! │  5. Else → full type check + record feedback                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! - **Cache Hit**: O(1) array lookup + single u8 comparison + inline execution
//! - **Cache Miss**: Falls back to full type check (no overhead if cold)
//! - **Memory**: Fixed 256-slot cache (4KB total)

use crate::ic_manager::ICSiteId;
use crate::type_feedback::OperandPair;

// =============================================================================
// Speculation Types
// =============================================================================

/// Speculation hint for binary operations.
///
/// Derived from IC type feedback, indicates expected operand types.
/// Stored in packed form for cache-line efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Speculation {
    /// No speculation available (cold or polymorphic site).
    #[default]
    None = 0,
    /// Both operands are integers (most common).
    IntInt = 1,
    /// Both operands are floats.
    FloatFloat = 2,
    /// Mixed int/float (will be promoted to float).
    IntFloat = 3,
    /// Float + Int (will be promoted to float).
    FloatInt = 4,
    /// Both operands are strings (concatenation).
    StrStr = 5,
    /// String + Int (repetition: "a" * 3).
    StrInt = 6,
    /// Int + String (repetition: 3 * "a").
    IntStr = 7,
}

impl Speculation {
    /// Create speculation from an OperandPair.
    #[inline(always)]
    pub const fn from_operand_pair(pair: OperandPair) -> Self {
        // OperandPair constants: INT_INT=0x11, FLOAT_FLOAT=0x22, etc.
        match pair.0 {
            0x11 => Speculation::IntInt,
            0x22 => Speculation::FloatFloat,
            0x12 => Speculation::IntFloat,
            0x21 => Speculation::FloatInt,
            0x55 => Speculation::StrStr, // STR_STR
            0x51 => Speculation::StrInt, // STR_INT
            0x15 => Speculation::IntStr, // INT_STR
            _ => Speculation::None,
        }
    }

    /// Check if this is an integer speculation.
    #[inline(always)]
    pub const fn is_int(self) -> bool {
        matches!(self, Speculation::IntInt)
    }

    /// Check if this is a float speculation (pure or mixed).
    #[inline(always)]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt
        )
    }

    /// Check if this is a string speculation.
    #[inline(always)]
    pub const fn is_string(self) -> bool {
        matches!(
            self,
            Speculation::StrStr | Speculation::StrInt | Speculation::IntStr
        )
    }

    /// Convert to a `TypeHint` for JIT integration.
    ///
    /// This allows the speculation data to be shared with the JIT compiler
    /// without exposing VM internals.
    #[inline]
    pub const fn to_type_hint(self) -> prism_core::TypeHint {
        use prism_core::TypeHint;
        match self {
            Speculation::None => TypeHint::None,
            Speculation::IntInt => TypeHint::IntInt,
            Speculation::FloatFloat => TypeHint::FloatFloat,
            Speculation::IntFloat => TypeHint::IntFloat,
            Speculation::FloatInt => TypeHint::FloatInt,
            Speculation::StrStr => TypeHint::StrStr,
            Speculation::StrInt => TypeHint::StrInt,
            Speculation::IntStr => TypeHint::IntStr,
        }
    }

    /// Create from a `TypeHint`.
    #[inline]
    pub const fn from_type_hint(hint: prism_core::TypeHint) -> Self {
        use prism_core::TypeHint;
        match hint {
            TypeHint::None => Speculation::None,
            TypeHint::IntInt => Speculation::IntInt,
            TypeHint::FloatFloat => Speculation::FloatFloat,
            TypeHint::IntFloat => Speculation::IntFloat,
            TypeHint::FloatInt => Speculation::FloatInt,
            TypeHint::StrStr => Speculation::StrStr,
            TypeHint::StrInt => Speculation::StrInt,
            TypeHint::IntStr => Speculation::IntStr,
        }
    }
}

// =============================================================================
// Speculation Cache
// =============================================================================

/// Cache size (must be power of 2 for fast modulo).
const CACHE_SIZE: usize = 256;

/// Mask for cache indexing (CACHE_SIZE - 1).
const CACHE_MASK: usize = CACHE_SIZE - 1;

/// Entry in the speculation cache.
///
/// Packed for cache-line efficiency. 16 bytes per entry.
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct CacheEntry {
    /// Code ID (64 bits).
    code_id: u64,
    /// Bytecode offset (32 bits).
    bc_offset: u32,
    /// Speculation hint (8 bits).
    speculation: Speculation,
    /// Padding for alignment.
    _pad: [u8; 3],
}

impl CacheEntry {
    /// Check if this entry matches a site ID.
    #[inline(always)]
    fn matches(&self, site: ICSiteId) -> bool {
        self.code_id == site.code_id.as_u64() && self.bc_offset == site.bc_offset
    }

    /// Update this entry with a new site and speculation.
    #[inline(always)]
    fn update(&mut self, site: ICSiteId, speculation: Speculation) {
        self.code_id = site.code_id.as_u64();
        self.bc_offset = site.bc_offset;
        self.speculation = speculation;
    }
}

/// O(1) speculation cache for hot path lookup.
///
/// Uses direct-mapped caching with site hash as index.
/// Collisions are handled by simply overwriting (LRU-like behavior
/// since most recently accessed site will be cached).
///
/// # Memory Layout
///
/// 256 entries × 16 bytes = 4KB (fits in L1 cache).
pub struct SpeculationCache {
    /// Fixed-size cache array.
    slots: Box<[CacheEntry; CACHE_SIZE]>,
}

impl SpeculationCache {
    /// Create a new empty speculation cache.
    pub fn new() -> Self {
        // Use box to avoid stack overflow with large array
        Self {
            slots: Box::new([CacheEntry::default(); CACHE_SIZE]),
        }
    }

    /// O(1) speculation lookup for a site.
    ///
    /// Returns the cached speculation if the site matches, None otherwise.
    #[inline(always)]
    pub fn get(&self, site: ICSiteId) -> Option<Speculation> {
        let index = Self::hash(site);
        let entry = &self.slots[index];

        if entry.matches(site) && entry.speculation != Speculation::None {
            Some(entry.speculation)
        } else {
            None
        }
    }

    /// Insert or update speculation for a site.
    ///
    /// Overwrites any existing entry at this hash index.
    #[inline(always)]
    pub fn insert(&mut self, site: ICSiteId, speculation: Speculation) {
        let index = Self::hash(site);
        self.slots[index].update(site, speculation);
    }

    /// Invalidate a cache entry (e.g., on deoptimization).
    #[inline]
    pub fn invalidate(&mut self, site: ICSiteId) {
        let index = Self::hash(site);
        if self.slots[index].matches(site) {
            self.slots[index].speculation = Speculation::None;
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        for slot in self.slots.iter_mut() {
            *slot = CacheEntry::default();
        }
    }

    /// Compute hash index for a site.
    ///
    /// Uses FNV-1a inspired mixing for good distribution.
    #[inline(always)]
    fn hash(site: ICSiteId) -> usize {
        // Mix code_id and bc_offset for better distribution
        let h = site.code_id.as_u64() ^ (site.bc_offset as u64).wrapping_mul(0x9E3779B9);
        (h as usize) & CACHE_MASK
    }
}

impl Default for SpeculationCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SpeculationCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let active_count = self
            .slots
            .iter()
            .filter(|e| e.speculation != Speculation::None)
            .count();
        f.debug_struct("SpeculationCache")
            .field("size", &CACHE_SIZE)
            .field("active_entries", &active_count)
            .finish()
    }
}

// Implement the SpeculationProvider trait for JIT integration
impl prism_core::SpeculationProvider for SpeculationCache {
    /// Get type hint for a bytecode site.
    ///
    /// This allows the JIT compiler to query speculation data collected
    /// by the VM's profiling system.
    #[inline]
    fn get_type_hint(&self, code_id: u32, bc_offset: u32) -> prism_core::TypeHint {
        use crate::profiler::CodeId;
        let site = ICSiteId::new(CodeId(code_id as u64), bc_offset);
        self.get(site)
            .map(|s| s.to_type_hint())
            .unwrap_or(prism_core::TypeHint::None)
    }
}

// =============================================================================
// Inline Speculation Helpers
// =============================================================================

use prism_core::Value;

/// Result of a speculative operation.
///
/// Designed for efficient branch prediction:
/// - Success is the expected path (no allocation)
/// - Deopt triggers slow path fallback
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecResult {
    /// Speculation succeeded, result stored in output.
    Success,
    /// Type mismatch, caller should use slow path.
    Deopt,
    /// Overflow occurred (integers only).
    Overflow,
}

impl SpecResult {
    /// Check if speculation succeeded.
    #[inline(always)]
    pub fn is_success(self) -> bool {
        self == SpecResult::Success
    }
}

/// Speculative integer addition.
///
/// Attempts int+int fast path. Returns Success and stores result if both
/// operands are integers and no overflow occurs.
#[inline(always)]
pub fn spec_add_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if let Some(result) = x.checked_add(y) {
            if let Some(v) = Value::int(result) {
                return (SpecResult::Success, v);
            }
        }
        return (SpecResult::Overflow, Value::none());
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer subtraction.
#[inline(always)]
pub fn spec_sub_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if let Some(result) = x.checked_sub(y) {
            if let Some(v) = Value::int(result) {
                return (SpecResult::Success, v);
            }
        }
        return (SpecResult::Overflow, Value::none());
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer multiplication.
#[inline(always)]
pub fn spec_mul_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if let Some(result) = x.checked_mul(y) {
            if let Some(v) = Value::int(result) {
                return (SpecResult::Success, v);
            }
        }
        return (SpecResult::Overflow, Value::none());
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer floor division.
#[inline(always)]
pub fn spec_floor_div_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if y == 0 {
            return (SpecResult::Overflow, Value::none()); // Division by zero
        }
        // Python floor division: always rounds toward negative infinity
        let result = x.div_euclid(y);
        if let Some(v) = Value::int(result) {
            return (SpecResult::Success, v);
        }
        return (SpecResult::Overflow, Value::none());
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer modulo.
#[inline(always)]
pub fn spec_mod_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if y == 0 {
            return (SpecResult::Overflow, Value::none()); // Division by zero
        }
        let result = x.rem_euclid(y);
        if let Some(v) = Value::int(result) {
            return (SpecResult::Success, v);
        }
        return (SpecResult::Overflow, Value::none());
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer power.
#[inline(always)]
pub fn spec_pow_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(base), Some(exp)) = (a.as_int(), b.as_int()) {
        // Only handle small positive exponents
        if exp >= 0 && exp <= 63 {
            if let Some(result) = (base as i128).checked_pow(exp as u32) {
                if result >= i64::MIN as i128 && result <= i64::MAX as i128 {
                    if let Some(v) = Value::int(result as i64) {
                        return (SpecResult::Success, v);
                    }
                }
            }
        }
        return (SpecResult::Overflow, Value::none());
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float addition.
#[inline(always)]
pub fn spec_add_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::float(x + y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float subtraction.
#[inline(always)]
pub fn spec_sub_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::float(x - y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float multiplication.
#[inline(always)]
pub fn spec_mul_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::float(x * y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float division.
#[inline(always)]
pub fn spec_div_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        if y == 0.0 {
            return (SpecResult::Overflow, Value::none()); // Division by zero
        }
        return (SpecResult::Success, Value::float(x / y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float floor division.
#[inline(always)]
pub fn spec_floor_div_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        if y == 0.0 {
            return (SpecResult::Overflow, Value::none());
        }
        return (SpecResult::Success, Value::float((x / y).floor()));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float modulo.
#[inline(always)]
pub fn spec_mod_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        if y == 0.0 {
            return (SpecResult::Overflow, Value::none());
        }
        // Python modulo: x - y * floor(x/y)
        let result = x - y * (x / y).floor();
        return (SpecResult::Success, Value::float(result));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float power.
#[inline(always)]
pub fn spec_pow_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::float(x.powf(y)));
    }
    (SpecResult::Deopt, Value::none())
}

// =============================================================================
// Comparison Speculation Helpers
// =============================================================================

/// Speculative integer less-than comparison.
#[inline(always)]
pub fn spec_lt_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return (SpecResult::Success, Value::bool(x < y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer less-than-or-equal comparison.
#[inline(always)]
pub fn spec_le_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return (SpecResult::Success, Value::bool(x <= y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer greater-than comparison.
#[inline(always)]
pub fn spec_gt_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return (SpecResult::Success, Value::bool(x > y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer greater-than-or-equal comparison.
#[inline(always)]
pub fn spec_ge_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return (SpecResult::Success, Value::bool(x >= y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer equality comparison.
#[inline(always)]
pub fn spec_eq_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return (SpecResult::Success, Value::bool(x == y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative integer not-equal comparison.
#[inline(always)]
pub fn spec_ne_int(a: Value, b: Value) -> (SpecResult, Value) {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return (SpecResult::Success, Value::bool(x != y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float less-than comparison.
#[inline(always)]
pub fn spec_lt_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::bool(x < y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float less-than-or-equal comparison.
#[inline(always)]
pub fn spec_le_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::bool(x <= y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float greater-than comparison.
#[inline(always)]
pub fn spec_gt_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::bool(x > y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float greater-than-or-equal comparison.
#[inline(always)]
pub fn spec_ge_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::bool(x >= y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float equality comparison.
#[inline(always)]
pub fn spec_eq_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::bool(x == y));
    }
    (SpecResult::Deopt, Value::none())
}

/// Speculative float not-equal comparison.
#[inline(always)]
pub fn spec_ne_float(a: Value, b: Value) -> (SpecResult, Value) {
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));

    if let (Some(x), Some(y)) = (x, y) {
        return (SpecResult::Success, Value::bool(x != y));
    }
    (SpecResult::Deopt, Value::none())
}

// =============================================================================
// String Speculation Helpers
// =============================================================================

use prism_runtime::TypeId;
use prism_runtime::types::StringObject;

/// Check if a Value is a StringObject (heap object with STR type).
///
/// This performs a two-step check:
/// 1. Check if Value is an object pointer (TAG_OBJECT)
/// 2. Read the object header to verify TypeId::STR
///
/// Returns the pointer to the StringObject if valid, None otherwise.
#[inline(always)]
fn extract_string_object(v: Value) -> Option<*const StringObject> {
    if !v.is_object() {
        return None;
    }

    if let Some(ptr) = v.as_object_ptr() {
        // Read the object header to check the type
        let header_ptr = ptr as *const prism_runtime::ObjectHeader;
        // SAFETY: We've verified this is an object pointer. The first field
        // of any Python object is the ObjectHeader.
        let type_id = unsafe { (*header_ptr).type_id };

        if type_id == TypeId::STR {
            return Some(ptr as *const StringObject);
        }
    }

    None
}

/// Speculative string concatenation (str + str).
///
/// # Performance
///
/// Uses type header inspection for StringObject detection:
/// 1. Check both values are object pointers with STR type
/// 2. Direct pointer extraction
/// 3. Delegates to optimized `StringObject::concat` with identity fast-paths
///
/// # Note
///
/// This handles StringObject heap objects. For InternedString (TAG_STRING),
/// a different path should be used that works with the interned string data.
#[inline(always)]
pub fn spec_str_concat(a: Value, b: Value) -> (SpecResult, Value) {
    // Extract StringObject pointers with type verification
    let a_ptr = match extract_string_object(a) {
        Some(p) => p,
        None => return (SpecResult::Deopt, Value::none()),
    };
    let b_ptr = match extract_string_object(b) {
        Some(p) => p,
        None => return (SpecResult::Deopt, Value::none()),
    };

    // SAFETY: Type checks above ensure these are valid StringObject pointers.
    // StringObjects are immutable so no data races are possible.
    let result = unsafe { (*a_ptr).concat(&*b_ptr) };

    // Box the result and create a Value::object for it
    // Note: In production, this would integrate with the GC allocator
    let boxed = Box::new(result);
    let ptr = Box::into_raw(boxed) as *const ();
    (SpecResult::Success, Value::object_ptr(ptr))
}

/// Speculative string repetition (str * int).
///
/// Handles both `str * int` and `int * str` patterns.
///
/// # Performance
///
/// 1. Type header check for StringObject detection
/// 2. Early return for common edge cases (n=0, n=1)
/// 3. Delegates to optimized `StringObject::repeat` with SSO/ASCII fast-paths
#[inline(always)]
pub fn spec_str_repeat(a: Value, b: Value) -> (SpecResult, Value) {
    // Check for str * int pattern
    if let Some(str_ptr) = extract_string_object(a) {
        if let Some(n) = b.as_int() {
            // Negative repetition count: Python returns empty string, but we signal overflow
            // to let the slow path handle it consistently
            if n < 0 {
                return (SpecResult::Overflow, Value::none());
            }

            // SAFETY: Type check ensures valid StringObject pointer
            let result = unsafe { (*str_ptr).repeat(n as usize) };
            let boxed = Box::new(result);
            let ptr = Box::into_raw(boxed) as *const ();
            return (SpecResult::Success, Value::object_ptr(ptr));
        }
    }

    // Check for int * str pattern (Python supports both orderings)
    if let Some(n) = a.as_int() {
        if let Some(str_ptr) = extract_string_object(b) {
            if n < 0 {
                return (SpecResult::Overflow, Value::none());
            }

            // SAFETY: Type check ensures valid StringObject pointer
            let result = unsafe { (*str_ptr).repeat(n as usize) };
            let boxed = Box::new(result);
            let ptr = Box::into_raw(boxed) as *const ();
            return (SpecResult::Success, Value::object_ptr(ptr));
        }
    }

    (SpecResult::Deopt, Value::none())
}

/// Speculative string length.
///
/// Returns the byte length of a string as an integer.
///
/// # Performance
///
/// Direct pointer access to StringObject for length retrieval.
#[inline(always)]
pub fn spec_str_len(a: Value) -> (SpecResult, Value) {
    if let Some(str_ptr) = extract_string_object(a) {
        // SAFETY: Type check ensures valid StringObject pointer
        let len = unsafe { (*str_ptr).len() };

        if let Some(v) = Value::int(len as i64) {
            return (SpecResult::Success, v);
        }
        // String too long to fit in integer (very unlikely but handle it)
        return (SpecResult::Overflow, Value::none());
    }

    (SpecResult::Deopt, Value::none())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::CodeId;

    #[test]
    fn test_speculation_from_operand_pair() {
        assert_eq!(
            Speculation::from_operand_pair(OperandPair(0x11)),
            Speculation::IntInt
        );
        assert_eq!(
            Speculation::from_operand_pair(OperandPair(0x22)),
            Speculation::FloatFloat
        );
        assert_eq!(
            Speculation::from_operand_pair(OperandPair(0x12)),
            Speculation::IntFloat
        );
        assert_eq!(
            Speculation::from_operand_pair(OperandPair(0x21)),
            Speculation::FloatInt
        );
        assert_eq!(
            Speculation::from_operand_pair(OperandPair(0x99)),
            Speculation::None
        );
    }

    #[test]
    fn test_speculation_cache_insert_get() {
        let mut cache = SpeculationCache::new();
        let site = ICSiteId::new(CodeId::new(123), 42);

        assert!(cache.get(site).is_none());

        cache.insert(site, Speculation::IntInt);
        assert_eq!(cache.get(site), Some(Speculation::IntInt));

        cache.insert(site, Speculation::FloatFloat);
        assert_eq!(cache.get(site), Some(Speculation::FloatFloat));
    }

    #[test]
    fn test_speculation_cache_invalidate() {
        let mut cache = SpeculationCache::new();
        let site = ICSiteId::new(CodeId::new(456), 100);

        cache.insert(site, Speculation::IntInt);
        assert_eq!(cache.get(site), Some(Speculation::IntInt));

        cache.invalidate(site);
        assert!(cache.get(site).is_none());
    }

    #[test]
    fn test_spec_add_int_success() {
        let a = Value::int(10).unwrap();
        let b = Value::int(20).unwrap();

        let (result, value) = spec_add_int(a, b);
        assert_eq!(result, SpecResult::Success);
        assert_eq!(value.as_int(), Some(30));
    }

    #[test]
    fn test_spec_add_int_deopt_on_float() {
        let a = Value::int(10).unwrap();
        let b = Value::float(20.5);

        let (result, _) = spec_add_int(a, b);
        assert_eq!(result, SpecResult::Deopt);
    }

    #[test]
    fn test_spec_add_float_success() {
        let a = Value::float(10.5);
        let b = Value::float(20.5);

        let (result, value) = spec_add_float(a, b);
        assert_eq!(result, SpecResult::Success);
        assert_eq!(value.as_float(), Some(31.0));
    }

    #[test]
    fn test_spec_add_float_with_int_promotion() {
        let a = Value::int(10).unwrap();
        let b = Value::float(20.5);

        let (result, value) = spec_add_float(a, b);
        assert_eq!(result, SpecResult::Success);
        assert_eq!(value.as_float(), Some(30.5));
    }

    #[test]
    fn test_spec_div_zero_overflow() {
        let a = Value::float(10.0);
        let b = Value::float(0.0);

        let (result, _) = spec_div_float(a, b);
        assert_eq!(result, SpecResult::Overflow);
    }

    #[test]
    fn test_spec_floor_div_int() {
        let a = Value::int(7).unwrap();
        let b = Value::int(3).unwrap();

        let (result, value) = spec_floor_div_int(a, b);
        assert_eq!(result, SpecResult::Success);
        assert_eq!(value.as_int(), Some(2));
    }

    #[test]
    fn test_spec_mod_int() {
        let a = Value::int(7).unwrap();
        let b = Value::int(3).unwrap();

        let (result, value) = spec_mod_int(a, b);
        assert_eq!(result, SpecResult::Success);
        assert_eq!(value.as_int(), Some(1));
    }

    #[test]
    fn test_spec_pow_int() {
        let a = Value::int(2).unwrap();
        let b = Value::int(10).unwrap();

        let (result, value) = spec_pow_int(a, b);
        assert_eq!(result, SpecResult::Success);
        assert_eq!(value.as_int(), Some(1024));
    }

    // =========================================================================
    // String Speculation Tests
    // =========================================================================

    #[test]
    fn test_spec_str_concat_deopt_on_non_strings() {
        // Test that non-string operands cause deopt
        let a = Value::int(10).unwrap();
        let b = Value::int(20).unwrap();

        let (result, _) = spec_str_concat(a, b);
        assert_eq!(result, SpecResult::Deopt);
    }

    #[test]
    fn test_spec_str_concat_deopt_on_mixed_types() {
        // String + int should deopt (not supported for concat)
        let str_obj = StringObject::new("hello");
        let boxed = Box::new(str_obj);
        let ptr = Box::into_raw(boxed) as *const ();
        let a = Value::object_ptr(ptr);
        let b = Value::int(5).unwrap();

        let (result, _) = spec_str_concat(a, b);
        assert_eq!(result, SpecResult::Deopt);

        // Cleanup
        unsafe {
            drop(Box::from_raw(ptr as *mut StringObject));
        }
    }

    #[test]
    fn test_spec_str_repeat_deopt_on_non_string_int() {
        // float * int should deopt
        let a = Value::float(3.14);
        let b = Value::int(5).unwrap();

        let (result, _) = spec_str_repeat(a, b);
        assert_eq!(result, SpecResult::Deopt);
    }

    #[test]
    fn test_spec_str_repeat_negative_count() {
        // Negative repetition should return Overflow
        let str_obj = StringObject::new("hello");
        let boxed = Box::new(str_obj);
        let ptr = Box::into_raw(boxed) as *const ();
        let a = Value::object_ptr(ptr);
        let b = Value::int(-1).unwrap();

        let (result, _) = spec_str_repeat(a, b);
        assert_eq!(result, SpecResult::Overflow);

        // Cleanup
        unsafe {
            drop(Box::from_raw(ptr as *mut StringObject));
        }
    }

    #[test]
    fn test_spec_str_repeat_int_str_order() {
        // int * str should also work (Python supports both orderings)
        let str_obj = StringObject::new("ab");
        let boxed = Box::new(str_obj);
        let ptr = Box::into_raw(boxed) as *const ();
        let a = Value::int(3).unwrap();
        let b = Value::object_ptr(ptr);

        let (result, value) = spec_str_repeat(a, b);
        assert_eq!(result, SpecResult::Success);

        // Verify the result is a valid object pointer
        assert!(value.is_object());

        // Cleanup both string objects
        unsafe {
            let result_ptr = value.as_object_ptr().unwrap() as *mut StringObject;
            let result_str = &*result_ptr;
            assert_eq!(result_str.as_str(), "ababab");
            drop(Box::from_raw(result_ptr));
            drop(Box::from_raw(ptr as *mut StringObject));
        }
    }

    #[test]
    fn test_spec_str_len_success() {
        let str_obj = StringObject::new("hello world");
        let boxed = Box::new(str_obj);
        let ptr = Box::into_raw(boxed) as *const ();
        let a = Value::object_ptr(ptr);

        let (result, value) = spec_str_len(a);
        assert_eq!(result, SpecResult::Success);
        assert_eq!(value.as_int(), Some(11));

        // Cleanup
        unsafe {
            drop(Box::from_raw(ptr as *mut StringObject));
        }
    }

    #[test]
    fn test_spec_str_len_deopt_on_non_string() {
        let a = Value::int(42).unwrap();

        let (result, _) = spec_str_len(a);
        assert_eq!(result, SpecResult::Deopt);
    }

    #[test]
    fn test_spec_str_len_empty_string() {
        let str_obj = StringObject::new("");
        let boxed = Box::new(str_obj);
        let ptr = Box::into_raw(boxed) as *const ();
        let a = Value::object_ptr(ptr);

        let (result, value) = spec_str_len(a);
        assert_eq!(result, SpecResult::Success);
        assert_eq!(value.as_int(), Some(0));

        // Cleanup
        unsafe {
            drop(Box::from_raw(ptr as *mut StringObject));
        }
    }
}
