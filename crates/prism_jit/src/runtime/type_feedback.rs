//! Type Feedback Oracle for Speculative Optimization.
//!
//! Provides a high-level query interface over raw [`ProfileData`] type profiles
//! to produce speculation decisions for the Tier 2 optimizing JIT.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
//! │  ProfileData │────►│  TypeFeedbackOracle   │────►│  Graph Builder   │
//! │  (raw data)  │     │  (decision engine)    │     │  (speculative    │
//! │              │     │                       │     │   guards)        │
//! └──────────────┘     └──────────────────────┘     └──────────────────┘
//! ```
//!
//! # Decision Flow
//!
//! For each bytecode offset with type feedback:
//! 1. Retrieve `TypeProfile` from `ProfileData`
//! 2. Classify stability (Monomorphic/Polymorphic/Megamorphic/Unknown)
//! 3. Compute confidence based on sample count and dominant frequency
//! 4. Produce `SpeculationDecision` with appropriate guard type
//!
//! # Performance
//!
//! - All lookups are O(1) hash table access via `ProfileData`
//! - Decision computation is O(k) where k = number of observed types (typically ≤ 5)
//! - Oracle is stateless and can be shared across compilation units

use super::profile_data::{CallProfile, ProfileData, TypeProfile, TypeProfileEntry};

// =============================================================================
// Observed Types
// =============================================================================

/// Canonical type classification used by the oracle.
///
/// Maps from the raw `type_id: u8` stored in `TypeProfileEntry` to a
/// semantic type that the JIT can reason about. The discriminant values
/// match the `TypeHint` repr used by the interpreter's profile instrumentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ObservedType {
    /// Small integer (i64, NaN-boxed).
    Int = 0,
    /// Double-precision float (f64, NaN-boxed).
    Float = 1,
    /// Boolean value.
    Bool = 2,
    /// None/null sentinel.
    None = 3,
    /// Heap-allocated string.
    String = 4,
    /// List (dynamic array).
    List = 5,
    /// Dictionary (hash map).
    Dict = 6,
    /// Tuple (immutable sequence).
    Tuple = 7,
    /// Set.
    Set = 8,
    /// User-defined class instance.
    Object = 9,
    /// Callable (function, method, closure).
    Callable = 10,
    /// Unknown/unrecognized type.
    Unknown = 255,
}

impl ObservedType {
    /// Convert from the raw `type_id` stored in profile data.
    #[inline]
    pub fn from_type_id(id: u8) -> Self {
        match id {
            0 => Self::Int,
            1 => Self::Float,
            2 => Self::Bool,
            3 => Self::None,
            4 => Self::String,
            5 => Self::List,
            6 => Self::Dict,
            7 => Self::Tuple,
            8 => Self::Set,
            9 => Self::Object,
            10 => Self::Callable,
            _ => Self::Unknown,
        }
    }

    /// Whether this type can benefit from speculative integer operations.
    #[inline]
    pub fn is_numeric(self) -> bool {
        matches!(self, Self::Int | Self::Float)
    }

    /// Whether this type is a collection that benefits from inline length checks.
    #[inline]
    pub fn is_collection(self) -> bool {
        matches!(self, Self::List | Self::Dict | Self::Tuple | Self::Set)
    }

    /// Whether this type is a heap-allocated object (needs GC barriers).
    #[inline]
    pub fn is_heap_allocated(self) -> bool {
        matches!(
            self,
            Self::String
                | Self::List
                | Self::Dict
                | Self::Tuple
                | Self::Set
                | Self::Object
                | Self::Callable
        )
    }
}

// =============================================================================
// Type Stability Classification
// =============================================================================

/// Classification of type behavior at a given site.
///
/// Determines whether speculation is safe and what kind of guard to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeStability {
    /// Single type observed >95% of the time. Safe for monomorphic guard.
    Monomorphic(ObservedType),
    /// 2-4 types observed with reasonable frequency. Suitable for type switch.
    Polymorphic,
    /// 5+ distinct types or no clear winner. Don't speculate.
    Megamorphic,
    /// Insufficient data (< minimum sample count). Don't speculate yet.
    Unstable,
}

impl TypeStability {
    /// Whether it's safe to emit speculative code for this site.
    #[inline]
    pub fn can_speculate(self) -> bool {
        matches!(self, Self::Monomorphic(_))
    }

    /// Whether a polymorphic type switch is worthwhile.
    #[inline]
    pub fn can_type_switch(self) -> bool {
        matches!(self, Self::Monomorphic(_) | Self::Polymorphic)
    }
}

// =============================================================================
// Speculation Decision
// =============================================================================

/// The oracle's recommendation for how to compile a given operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeculationDecision {
    /// Emit a monomorphic guard + specialized code path.
    ///
    /// Fields:
    /// - `guard_type`: The type to guard for
    /// - `confidence`: How confident we are (0.0 - 1.0)
    /// - `deopt_count_estimate`: Expected deopt frequency (lower = better)
    Speculate {
        guard_type: ObservedType,
        confidence: f64,
        deopt_count_estimate: f64,
    },

    /// Emit a polymorphic type switch with multiple specialized paths.
    ///
    /// Fields:
    /// - `types`: Ordered list of (type, probability) pairs
    TypeSwitch { types: Vec<(ObservedType, f64)> },

    /// Don't speculate — emit generic code.
    Generic { reason: GenericReason },
}

/// Reason for falling back to generic code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericReason {
    /// Not enough samples to be confident.
    InsufficientData,
    /// Too many types observed (megamorphic).
    Megamorphic,
    /// The operation doesn't benefit from specialization.
    NotSpecializable,
    /// No profile data available for this offset.
    NoProfile,
    /// Confidence below threshold.
    LowConfidence,
}

// =============================================================================
// Call Speculation Decision
// =============================================================================

/// The oracle's recommendation for a call site.
#[derive(Debug, Clone, PartialEq)]
pub enum CallSpeculation {
    /// Monomorphic call — inline or devirtualize.
    Monomorphic { target_id: u32, confidence: f64 },
    /// Polymorphic dispatch — type-switch with multiple targets.
    Polymorphic { targets: Vec<(u32, f64)> },
    /// Megamorphic — use generic dispatch.
    Megamorphic,
    /// No call profile data.
    Unknown,
}

// =============================================================================
// Oracle Configuration
// =============================================================================

/// Configuration for the type feedback oracle's decision thresholds.
#[derive(Debug, Clone)]
pub struct OracleConfig {
    /// Minimum number of observations before speculation (default: 30).
    ///
    /// Lower values enable earlier speculation but with higher deopt risk.
    /// V8 uses ~100, HotSpot uses ~10000. We use 30 for a balanced approach
    /// that works well with our 3-tier system.
    pub min_samples: u64,

    /// Confidence threshold for monomorphic speculation (default: 0.90).
    ///
    /// The dominant type must represent at least this fraction of observations.
    pub monomorphic_threshold: f64,

    /// Confidence threshold for polymorphic type switch (default: 0.70).
    ///
    /// The top-N types must collectively represent this fraction.
    pub polymorphic_threshold: f64,

    /// Maximum number of types in a polymorphic switch (default: 4).
    pub max_polymorphic_types: usize,

    /// Minimum confidence for call site devirtualization (default: 0.95).
    pub call_monomorphic_threshold: f64,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            min_samples: 30,
            monomorphic_threshold: 0.90,
            polymorphic_threshold: 0.70,
            max_polymorphic_types: 4,
            call_monomorphic_threshold: 0.95,
        }
    }
}

impl OracleConfig {
    /// Conservative configuration — higher thresholds, fewer deopts.
    pub fn conservative() -> Self {
        Self {
            min_samples: 100,
            monomorphic_threshold: 0.98,
            polymorphic_threshold: 0.85,
            max_polymorphic_types: 3,
            call_monomorphic_threshold: 0.99,
        }
    }

    /// Aggressive configuration — lower thresholds, more speculation.
    pub fn aggressive() -> Self {
        Self {
            min_samples: 10,
            monomorphic_threshold: 0.80,
            polymorphic_threshold: 0.60,
            max_polymorphic_types: 6,
            call_monomorphic_threshold: 0.90,
        }
    }
}

// =============================================================================
// Type Feedback Oracle
// =============================================================================

/// The type feedback oracle — the central decision engine for speculative
/// optimization in the Tier 2 JIT.
///
/// Given a `ProfileData` reference and a bytecode offset, the oracle
/// determines whether to speculate, what type to guard for, and how
/// confident the prediction is.
///
/// # Thread Safety
///
/// The oracle borrows `ProfileData` immutably and is stateless — it can
/// safely be used from the compilation thread without synchronization.
///
/// # Example
///
/// ```ignore
/// let oracle = TypeFeedbackOracle::new(&profile_data);
///
/// match oracle.query_type(bc_offset) {
///     SpeculationDecision::Speculate { guard_type, .. } => {
///         // Emit type guard + specialized code
///         emit_type_guard(guard_type);
///         emit_specialized_op(guard_type);
///     }
///     SpeculationDecision::Generic { .. } => {
///         // Emit generic code path
///         emit_generic_op();
///     }
///     _ => { /* handle other cases */ }
/// }
/// ```
pub struct TypeFeedbackOracle<'a> {
    /// The profile data to query.
    profile: &'a ProfileData,
    /// Decision configuration.
    config: OracleConfig,
}

impl<'a> TypeFeedbackOracle<'a> {
    /// Create a new oracle with default configuration.
    #[inline]
    pub fn new(profile: &'a ProfileData) -> Self {
        Self {
            profile,
            config: OracleConfig::default(),
        }
    }

    /// Create a new oracle with custom configuration.
    #[inline]
    pub fn with_config(profile: &'a ProfileData, config: OracleConfig) -> Self {
        Self { profile, config }
    }

    /// Get the oracle's configuration.
    #[inline]
    pub fn config(&self) -> &OracleConfig {
        &self.config
    }

    // =========================================================================
    // Type Queries
    // =========================================================================

    /// Query the type feedback for a given bytecode offset.
    ///
    /// Returns a `SpeculationDecision` indicating whether to speculate
    /// and what type guard to emit.
    pub fn query_type(&self, offset: u32) -> SpeculationDecision {
        let type_profile = match self.profile.type_at(offset) {
            Some(tp) => tp,
            None => {
                return SpeculationDecision::Generic {
                    reason: GenericReason::NoProfile,
                };
            }
        };

        // Insufficient data — don't speculate yet
        if type_profile.total() < self.config.min_samples {
            return SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData,
            };
        }

        let stability = self.classify_type_stability(type_profile);

        match stability {
            TypeStability::Monomorphic(observed) => {
                let dominant = type_profile.dominant_type().unwrap();
                let confidence = dominant.count as f64 / type_profile.total() as f64;

                if confidence >= self.config.monomorphic_threshold {
                    let deopt_estimate = 1.0 - confidence;
                    SpeculationDecision::Speculate {
                        guard_type: observed,
                        confidence,
                        deopt_count_estimate: deopt_estimate,
                    }
                } else {
                    // Confidence too low for monomorphic — try polymorphic
                    self.try_polymorphic_decision(type_profile)
                }
            }
            TypeStability::Polymorphic => self.try_polymorphic_decision(type_profile),
            TypeStability::Megamorphic => SpeculationDecision::Generic {
                reason: GenericReason::Megamorphic,
            },
            TypeStability::Unstable => SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData,
            },
        }
    }

    /// Query the dominant type at a given offset (fast path for simple checks).
    ///
    /// Returns `None` if no profile data or insufficient samples.
    #[inline]
    pub fn dominant_type_at(&self, offset: u32) -> Option<ObservedType> {
        let tp = self.profile.type_at(offset)?;
        if tp.total() < self.config.min_samples {
            return None;
        }
        tp.dominant_type()
            .map(|entry| ObservedType::from_type_id(entry.type_id))
    }

    /// Check if a site is monomorphic with sufficient confidence.
    #[inline]
    pub fn is_monomorphic_at(&self, offset: u32) -> bool {
        self.profile.type_at(offset).map_or(false, |tp| {
            tp.is_monomorphic() && tp.total() >= self.config.min_samples
        })
    }

    /// Get the type stability classification at a given offset.
    pub fn stability_at(&self, offset: u32) -> TypeStability {
        match self.profile.type_at(offset) {
            Some(tp) if tp.total() >= self.config.min_samples => self.classify_type_stability(tp),
            _ => TypeStability::Unstable,
        }
    }

    /// Compute confidence score for the dominant type at an offset.
    ///
    /// Returns a value between 0.0 (no confidence) and 1.0 (perfect confidence).
    /// Returns 0.0 if no profile data or insufficient samples.
    pub fn confidence_at(&self, offset: u32) -> f64 {
        match self.profile.type_at(offset) {
            Some(tp) if tp.total() >= self.config.min_samples => tp
                .dominant_type()
                .map_or(0.0, |d| d.count as f64 / tp.total() as f64),
            _ => 0.0,
        }
    }

    // =========================================================================
    // Call Site Queries
    // =========================================================================

    /// Query call site feedback for speculation/devirtualization decisions.
    pub fn query_call(&self, offset: u32) -> CallSpeculation {
        let call_profile = match self.profile.call_at(offset) {
            Some(cp) => cp,
            None => return CallSpeculation::Unknown,
        };

        if call_profile.total() < self.config.min_samples {
            return CallSpeculation::Unknown;
        }

        self.classify_call_site(call_profile)
    }

    /// Check if a call site is monomorphic (single target).
    #[inline]
    pub fn is_monomorphic_call_at(&self, offset: u32) -> bool {
        self.profile.call_at(offset).map_or(false, |cp| {
            cp.is_monomorphic() && cp.total() >= self.config.min_samples
        })
    }

    // =========================================================================
    // Loop Queries
    // =========================================================================

    /// Get the iteration count for a loop header.
    #[inline]
    pub fn loop_trip_count(&self, header_offset: u32) -> u64 {
        self.profile.loop_count(header_offset)
    }

    /// Whether a loop is hot (iterated more than threshold).
    #[inline]
    pub fn is_hot_loop(&self, header_offset: u32, threshold: u64) -> bool {
        self.loop_trip_count(header_offset) >= threshold
    }

    // =========================================================================
    // Internal Classification Methods
    // =========================================================================

    /// Classify the type stability of a profile.
    fn classify_type_stability(&self, profile: &TypeProfile) -> TypeStability {
        if profile.total() < self.config.min_samples {
            return TypeStability::Unstable;
        }

        if profile.is_megamorphic() {
            return TypeStability::Megamorphic;
        }

        if let Some(dominant) = profile.dominant_type() {
            let ratio = dominant.count as f64 / profile.total() as f64;
            let observed = ObservedType::from_type_id(dominant.type_id);

            if ratio >= self.config.monomorphic_threshold {
                return TypeStability::Monomorphic(observed);
            }
        }

        if profile.is_polymorphic() {
            return TypeStability::Polymorphic;
        }

        // Has data but no clear pattern
        TypeStability::Megamorphic
    }

    /// Attempt to produce a polymorphic type switch decision.
    fn try_polymorphic_decision(&self, profile: &TypeProfile) -> SpeculationDecision {
        let total = profile.total() as f64;
        if total == 0.0 {
            return SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData,
            };
        }

        // Collect entries sorted by frequency (descending)
        let mut entries: Vec<&TypeProfileEntry> = profile.entries().iter().collect();
        entries.sort_unstable_by(|a, b| b.count.cmp(&a.count));

        // Take top-N types that pass the significance threshold (> 1%)
        let significant: Vec<(ObservedType, f64)> = entries
            .iter()
            .take(self.config.max_polymorphic_types)
            .filter(|e| e.count as f64 / total > 0.01)
            .map(|e| {
                (
                    ObservedType::from_type_id(e.type_id),
                    e.count as f64 / total,
                )
            })
            .collect();

        if significant.is_empty() {
            return SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData,
            };
        }

        // Check if collective coverage meets threshold
        let coverage: f64 = significant.iter().map(|(_, p)| p).sum();
        if coverage >= self.config.polymorphic_threshold {
            if significant.len() == 1 {
                // Actually monomorphic with lower confidence
                SpeculationDecision::Speculate {
                    guard_type: significant[0].0,
                    confidence: significant[0].1,
                    deopt_count_estimate: 1.0 - significant[0].1,
                }
            } else {
                SpeculationDecision::TypeSwitch { types: significant }
            }
        } else {
            SpeculationDecision::Generic {
                reason: GenericReason::LowConfidence,
            }
        }
    }

    /// Classify a call site for devirtualization decisions.
    fn classify_call_site(&self, profile: &CallProfile) -> CallSpeculation {
        let total = profile.total() as f64;
        if total == 0.0 {
            return CallSpeculation::Unknown;
        }

        let targets = profile.targets();
        if targets.is_empty() {
            return CallSpeculation::Unknown;
        }

        // Check if the primary target dominates
        let primary = &targets[0];
        let primary_confidence = primary.count as f64 / total;

        if primary_confidence >= self.config.call_monomorphic_threshold {
            return CallSpeculation::Monomorphic {
                target_id: primary.target_id,
                confidence: primary_confidence,
            };
        }

        // Check if top-N targets cover enough
        let top_targets: Vec<(u32, f64)> = targets
            .iter()
            .take(self.config.max_polymorphic_types)
            .filter(|t| t.count as f64 / total > 0.01)
            .map(|t| (t.target_id, t.count as f64 / total))
            .collect();

        let coverage: f64 = top_targets.iter().map(|(_, p)| p).sum();

        if coverage >= self.config.polymorphic_threshold && top_targets.len() <= 4 {
            CallSpeculation::Polymorphic {
                targets: top_targets,
            }
        } else {
            CallSpeculation::Megamorphic
        }
    }
}
