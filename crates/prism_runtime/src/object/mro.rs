//! MRO (Method Resolution Order) computation using C3 linearization.
//!
//! Implements Python's C3 linearization algorithm for computing the method
//! resolution order of a class with multiple inheritance.
//!
//! # C3 Linearization
//!
//! The C3 algorithm ensures:
//! 1. **Local Precedence Order**: A class comes before its parents
//! 2. **Monotonicity**: If A precedes B in one linearization, A precedes B everywhere
//! 3. **Extended Precedence Graph**: Respects parent ordering as declared
//!
//! # Example
//!
//! ```text
//! class A: pass
//! class B(A): pass
//! class C(A): pass
//! class D(B, C): pass
//!
//! MRO of D: [D, B, C, A, object]
//! ```
//!
//! # Performance
//!
//! - O(n²) worst case where n = total classes in hierarchy
//! - Uses SmallVec to avoid allocations for typical hierarchies (≤8 classes)
//! - MRO is cached in the TypeObject after first computation

use smallvec::SmallVec;
use std::fmt;

use crate::object::type_obj::TypeId;

/// Type representing a class/type in the MRO computation.
/// Uses a newtype pattern for type safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ClassId(pub u32);

impl ClassId {
    /// Built-in object class ID.
    pub const OBJECT: ClassId = ClassId(TypeId::OBJECT.raw());

    /// Built-in type class ID.
    pub const TYPE: ClassId = ClassId(TypeId::TYPE.raw());

    /// Sentinel value for no class / invalid class.
    pub const NONE: ClassId = ClassId(u32::MAX);
}

/// Error during MRO computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MroError {
    /// C3 linearization failed - inconsistent method resolution.
    /// This happens when there's no valid ordering that satisfies all constraints.
    InconsistentMro {
        /// Class that caused the failure.
        class_name: String,
        /// Conflicting classes that couldn't be ordered.
        conflicting: Vec<String>,
    },

    /// Duplicate base class detected.
    DuplicateBase {
        class_name: String,
        duplicate: String,
    },

    /// Inheritance cycle detected.
    InheritanceCycle { classes: Vec<String> },
}

impl fmt::Display for MroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MroError::InconsistentMro {
                class_name,
                conflicting,
            } => {
                write!(
                    f,
                    "Cannot create a consistent method resolution order (MRO) for class '{}'. \
                     Conflicting classes: {:?}",
                    class_name, conflicting
                )
            }
            MroError::DuplicateBase {
                class_name,
                duplicate,
            } => {
                write!(
                    f,
                    "Duplicate base class '{}' in class '{}'",
                    duplicate, class_name
                )
            }
            MroError::InheritanceCycle { classes } => {
                write!(f, "Inheritance cycle detected: {:?}", classes)
            }
        }
    }
}

impl std::error::Error for MroError {}

/// Result type for MRO computation.
pub type MroResult<T> = Result<T, MroError>;

/// Stack-allocated MRO for typical class hierarchies.
/// Most Python classes have ≤8 classes in their MRO.
pub type Mro = SmallVec<[ClassId; 8]>;

/// Compute the C3 linearization (MRO) for a class.
///
/// # Arguments
///
/// * `class_id` - The class for which to compute the MRO
/// * `bases` - The direct base classes (in declaration order)
/// * `get_mro` - Function to get the MRO of a parent class
///
/// # Returns
///
/// The computed MRO as a list of class IDs, or an error if C3 fails.
///
/// # Algorithm
///
/// The C3 algorithm merges:
/// 1. The class itself [C]
/// 2. The MROs of each parent L[P1], L[P2], ...
/// 3. The list of parents [P1, P2, ...]
///
/// At each step, pick the first head that doesn't appear in any tail.
/// A "good head" is one that is not in the tail of any other list.
pub fn compute_c3_mro<F>(class_id: ClassId, bases: &[ClassId], get_mro: F) -> MroResult<Mro>
where
    F: Fn(ClassId) -> Option<Mro>,
{
    // No bases → MRO is just [class, object]
    if bases.is_empty() {
        let mut mro = Mro::new();
        mro.push(class_id);
        if class_id != ClassId::OBJECT {
            mro.push(ClassId::OBJECT);
        }
        return Ok(mro);
    }

    // Check for duplicate bases
    for (i, base) in bases.iter().enumerate() {
        if bases[i + 1..].contains(base) {
            return Err(MroError::DuplicateBase {
                class_name: format!("class_{}", class_id.0),
                duplicate: format!("class_{}", base.0),
            });
        }
    }

    // Collect MROs of all parents
    let parent_mros: Vec<Mro> = bases
        .iter()
        .map(|&base| {
            get_mro(base).unwrap_or_else(|| {
                // Fallback: just the base class and object
                let mut mro = Mro::new();
                mro.push(base);
                if base != ClassId::OBJECT {
                    mro.push(ClassId::OBJECT);
                }
                mro
            })
        })
        .collect();

    // Build the merge lists:
    // [L[P1], L[P2], ..., [P1, P2, ...]]
    let mut merge_lists: Vec<Vec<ClassId>> = parent_mros.iter().map(|mro| mro.to_vec()).collect();

    // Add the list of bases (parents in declaration order)
    merge_lists.push(bases.to_vec());

    // Result MRO starts with the class itself
    let mut result = Mro::new();
    result.push(class_id);

    // C3 merge loop
    loop {
        // Remove empty lists
        merge_lists.retain(|list| !list.is_empty());

        // If all lists are empty, we're done
        if merge_lists.is_empty() {
            break;
        }

        // Find a "good head": a head that is not in the tail of any list
        let mut found_good_head = false;

        for list_idx in 0..merge_lists.len() {
            let candidate = merge_lists[list_idx][0];

            // Check if candidate is in the tail of any list
            let in_any_tail = merge_lists
                .iter()
                .any(|list| list.len() > 1 && list[1..].contains(&candidate));

            if !in_any_tail {
                // Found a good head - add to result and remove from all lists
                result.push(candidate);

                for list in &mut merge_lists {
                    if !list.is_empty() && list[0] == candidate {
                        list.remove(0);
                    }
                }

                found_good_head = true;
                break;
            }
        }

        if !found_good_head {
            // C3 linearization failed - inconsistent hierarchy
            let conflicting: Vec<String> = merge_lists
                .iter()
                .filter(|list| !list.is_empty())
                .map(|list| format!("class_{}", list[0].0))
                .collect();

            return Err(MroError::InconsistentMro {
                class_name: format!("class_{}", class_id.0),
                conflicting,
            });
        }
    }

    Ok(result)
}

/// Compute MRO from class names (convenience function for error messages).
pub fn compute_mro_with_names<F, G>(
    class_name: &str,
    base_names: &[&str],
    get_bases: F,
    get_mro: G,
) -> MroResult<Vec<String>>
where
    F: Fn(&str) -> Vec<String>,
    G: Fn(&str) -> Option<Vec<String>>,
{
    // This is a simplified version for use in error messages and testing.
    // The actual implementation uses ClassId for efficiency.

    let mut result = vec![class_name.to_string()];

    if base_names.is_empty() {
        result.push("object".to_string());
        return Ok(result);
    }

    // For now, use simple right-to-left resolution for single inheritance
    // Full C3 implementation uses the numeric version above
    for base in base_names {
        if let Some(base_mro) = get_mro(base) {
            for class in base_mro {
                if !result.contains(&class) {
                    result.push(class);
                }
            }
        } else {
            if !result.contains(&base.to_string()) {
                result.push(base.to_string());
            }
        }
    }

    if !result.contains(&"object".to_string()) {
        result.push("object".to_string());
    }

    Ok(result)
}
