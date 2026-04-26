//! Keyword argument binding utilities.
//!
//! This module provides high-performance argument binding for function calls
//! with keyword arguments, implementing Python's complex parameter resolution.
//!
//! # Binding Algorithm
//!
//! The argument binding follows Python 3.12 semantics:
//! 1. Bind positional arguments to positional parameters
//! 2. Collect excess positional arguments into *args tuple (if present)
//! 3. Bind keyword arguments to matching parameters
//! 4. Collect unmatched keyword arguments into **kwargs dict (if present)
//! 5. Fill missing parameters with default values
//! 6. Error on missing required arguments
//!
//! # Performance Considerations
//!
//! - Uses stack-allocated arrays where possible to avoid heap allocation
//! - O(n) parameter lookup using linear search (suitable for typical arg counts)
//! - Minimizes Value copies through careful borrowing
//! - Pre-allocates containers with exact capacity when sizes are known

use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::Arc;

// =============================================================================
// Error Types
// =============================================================================

/// Argument binding error.
#[derive(Debug)]
pub enum BindingError {
    /// Too many positional arguments provided.
    TooManyPositional {
        func_name: Arc<str>,
        expected: u16,
        given: usize,
    },
    /// Duplicate value for parameter (passed both positionally and by keyword).
    DuplicateArgument {
        func_name: Arc<str>,
        param_name: Arc<str>,
    },
    /// Unexpected keyword argument (no matching parameter and no **kwargs).
    UnexpectedKeyword {
        func_name: Arc<str>,
        keyword: Arc<str>,
    },
    /// Missing required positional argument.
    MissingPositional {
        func_name: Arc<str>,
        param_name: Arc<str>,
    },
    /// Missing required keyword-only argument.
    MissingKeywordOnly {
        func_name: Arc<str>,
        param_name: Arc<str>,
    },
    /// Internal error during binding.
    Internal(&'static str),
}

impl BindingError {
    /// Format as a Python-style TypeError message.
    pub fn to_error_message(&self) -> String {
        match self {
            BindingError::TooManyPositional {
                func_name,
                expected,
                given,
            } => {
                format!(
                    "{}() takes {} positional arguments but {} were given",
                    func_name, expected, given
                )
            }
            BindingError::DuplicateArgument {
                func_name,
                param_name,
            } => {
                format!(
                    "{}() got multiple values for argument '{}'",
                    func_name, param_name
                )
            }
            BindingError::UnexpectedKeyword { func_name, keyword } => {
                format!(
                    "{}() got an unexpected keyword argument '{}'",
                    func_name, keyword
                )
            }
            BindingError::MissingPositional {
                func_name,
                param_name,
            } => {
                format!(
                    "{}() missing required positional argument: '{}'",
                    func_name, param_name
                )
            }
            BindingError::MissingKeywordOnly {
                func_name,
                param_name,
            } => {
                format!(
                    "{}() missing required keyword-only argument: '{}'",
                    func_name, param_name
                )
            }
            BindingError::Internal(msg) => msg.to_string(),
        }
    }
}

// =============================================================================
// Bound Arguments Result
// =============================================================================

/// Result of argument binding, containing resolved values for all parameters.
///
/// This struct supports functions with:
/// - Regular positional parameters
/// - Keyword-only parameters
/// - *args variadic positional parameter
/// - **kwargs variadic keyword parameter
pub struct BoundArguments {
    /// Values for positional and keyword-only parameters.
    /// Length = arg_count + kwonlyarg_count
    pub parameters: Vec<Value>,

    /// Collected *args tuple (if function has VARARGS flag).
    /// None if function doesn't accept *args.
    pub varargs: Option<Box<TupleObject>>,

    /// Collected **kwargs dict (if function has VARKEYWORDS flag).
    /// None if function doesn't accept **kwargs.
    pub varkw: Option<Box<DictObject>>,
}

impl std::fmt::Debug for BoundArguments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundArguments")
            .field("parameters", &self.parameters.len())
            .field("varargs_len", &self.varargs.as_ref().map(|t| t.len()))
            .field("varkw_len", &self.varkw.as_ref().map(|d| d.len()))
            .finish()
    }
}

impl BoundArguments {
    /// Create new bound arguments with capacity for parameters.
    #[inline]
    fn new(param_count: usize) -> Self {
        Self {
            parameters: vec![Value::none(); param_count],
            varargs: None,
            varkw: None,
        }
    }
}

// =============================================================================
// Binding Engine
// =============================================================================

/// Performs high-performance argument binding for function calls.
///
/// This is the core implementation that matches Python's argument resolution
/// semantics while optimizing for the common case of small argument counts.
pub struct ArgumentBinder;

impl ArgumentBinder {
    /// Bind positional and keyword arguments to function parameters.
    ///
    /// # Arguments
    ///
    /// * `func` - The function being called
    /// * `positional_args` - Iterator over positional argument values
    /// * `keyword_args` - Iterator over (name, value) pairs for keyword arguments
    ///
    /// # Returns
    ///
    /// `BoundArguments` containing resolved parameter values, or `BindingError`.
    ///
    /// # Performance
    ///
    /// - O(P + K) where P = positional args, K = keyword args
    /// - Single pass over each argument source
    /// - Pre-allocated result vectors
    #[inline]
    pub fn bind<'a, P, K>(
        func: &FunctionObject,
        positional_args: P,
        keyword_args: K,
    ) -> Result<BoundArguments, BindingError>
    where
        P: Iterator<Item = Value>,
        K: Iterator<Item = (&'a str, Value)>,
    {
        let code = &func.code;
        let arg_count = code.arg_count as usize;
        let kwonly_count = code.kwonlyarg_count as usize;
        let total_params = arg_count + kwonly_count;
        let has_varargs = func.has_varargs();
        let has_varkw = func.has_varkw();

        // Pre-allocate result
        let mut result = BoundArguments::new(total_params);

        // Track which parameters have been bound
        // Using a simple bool array for small counts, bitvec for larger
        let mut bound_flags = vec![false; total_params];

        // =========================================================================
        // Phase 1: Bind positional arguments
        // =========================================================================

        let mut excess_positional: Vec<Value> = Vec::new();
        let mut pos_idx = 0;

        for arg_value in positional_args {
            if pos_idx < arg_count {
                // Bind to regular positional parameter
                result.parameters[pos_idx] = arg_value;
                bound_flags[pos_idx] = true;
            } else if has_varargs {
                // Collect into *args
                excess_positional.push(arg_value);
            } else {
                // Too many positional arguments
                return Err(BindingError::TooManyPositional {
                    func_name: Arc::clone(&code.name),
                    expected: code.arg_count,
                    given: pos_idx + 1 + excess_positional.len(),
                });
            }
            pos_idx += 1;
        }

        // Create *args tuple if function accepts varargs
        if has_varargs {
            result.varargs = Some(Box::new(TupleObject::from_slice(&excess_positional)));
        }

        // =========================================================================
        // Phase 2: Bind keyword arguments
        // =========================================================================

        let mut extra_kwargs: Option<Box<DictObject>> = if has_varkw {
            Some(Box::new(DictObject::with_capacity(4)))
        } else {
            None
        };

        for (kw_name, kw_value) in keyword_args {
            // Find parameter index by name
            if let Some(param_idx) = Self::find_param_index(func, kw_name) {
                // Check for duplicate binding
                if bound_flags[param_idx] {
                    let locals_idx = Self::param_to_locals_index(func, param_idx);
                    let param_name = code
                        .locals
                        .get(locals_idx)
                        .map(|s| Arc::clone(s))
                        .unwrap_or_else(|| "?".into());
                    return Err(BindingError::DuplicateArgument {
                        func_name: Arc::clone(&code.name),
                        param_name,
                    });
                }

                result.parameters[param_idx] = kw_value;
                bound_flags[param_idx] = true;
            } else if let Some(ref mut kwargs_dict) = extra_kwargs {
                // Collect into **kwargs
                // Create a string Value for the key
                let key = Self::create_string_key(kw_name);
                kwargs_dict.set(key, kw_value);
            } else {
                // Unexpected keyword argument
                return Err(BindingError::UnexpectedKeyword {
                    func_name: Arc::clone(&code.name),
                    keyword: kw_name.into(),
                });
            }
        }

        result.varkw = extra_kwargs;

        // =========================================================================
        // Phase 3: Fill defaults for unbound positional parameters
        // =========================================================================

        for i in 0..arg_count {
            if !bound_flags[i] {
                if let Some(default_val) = func.get_default(i) {
                    result.parameters[i] = default_val;
                    bound_flags[i] = true;
                } else {
                    // For positional params, locals_idx == param_idx
                    let param_name = code
                        .locals
                        .get(i)
                        .map(|s| Arc::clone(s))
                        .unwrap_or_else(|| "?".into());
                    return Err(BindingError::MissingPositional {
                        func_name: Arc::clone(&code.name),
                        param_name,
                    });
                }
            }
        }

        // =========================================================================
        // Phase 4: Fill defaults for unbound keyword-only parameters
        // =========================================================================

        for i in arg_count..total_params {
            if !bound_flags[i] {
                // Get locals index accounting for varargs slot offset
                let locals_idx = Self::param_to_locals_index(func, i);
                let param_name = code
                    .locals
                    .get(locals_idx)
                    .map(|s| Arc::clone(s))
                    .unwrap_or_else(|| "?".into());

                // Check kwdefaults
                let found_default = if let Some(kwdefaults) = &func.kwdefaults {
                    kwdefaults
                        .iter()
                        .find(|(n, _)| n.as_ref() == param_name.as_ref())
                        .map(|(_, val)| *val)
                } else {
                    None
                };

                if let Some(default_val) = found_default {
                    result.parameters[i] = default_val;
                } else {
                    return Err(BindingError::MissingKeywordOnly {
                        func_name: Arc::clone(&code.name),
                        param_name,
                    });
                }
            }
        }

        Ok(result)
    }

    /// Find parameter index by name.
    ///
    /// Returns the *parameter index* (into the bound arguments array), not the locals index.
    /// Accounts for the varargs slot offset when looking up keyword-only parameters.
    ///
    /// Locals layout with varargs:
    /// - [0..arg_count): positional params
    /// - [arg_count]: *args name (if VARARGS flag)
    /// - [arg_count + varargs_offset..arg_count + varargs_offset + kwonlyarg_count): kwonly params
    ///
    /// Uses linear search which is optimal for typical parameter counts (< 10).
    #[inline]
    fn find_param_index(func: &FunctionObject, name: &str) -> Option<usize> {
        let code = &func.code;
        let arg_count = code.arg_count as usize;
        let kwonly_count = code.kwonlyarg_count as usize;
        let has_varargs = func.has_varargs();

        // Search positional parameters first (locals[0..arg_count])
        for i in 0..arg_count {
            if let Some(param_name) = code.locals.get(i) {
                if param_name.as_ref() == name {
                    return Some(i); // param_idx == locals_idx for positional
                }
            }
        }

        // Search keyword-only parameters (after potential *args slot)
        // locals_offset points to first kwonly param in locals
        let kwonly_locals_start = arg_count + if has_varargs { 1 } else { 0 };

        for i in 0..kwonly_count {
            let locals_idx = kwonly_locals_start + i;
            if let Some(param_name) = code.locals.get(locals_idx) {
                if param_name.as_ref() == name {
                    // param_idx for kwonly is arg_count + i
                    return Some(arg_count + i);
                }
            }
        }

        None
    }

    /// Get the locals index for a given parameter index.
    ///
    /// This is the inverse of find_param_index, accounting for varargs offset.
    #[inline]
    fn param_to_locals_index(func: &FunctionObject, param_idx: usize) -> usize {
        let code = &func.code;
        let arg_count = code.arg_count as usize;

        if param_idx < arg_count {
            // Positional param - direct mapping
            param_idx
        } else {
            // Keyword-only param - account for varargs slot
            let kwonly_offset = param_idx - arg_count;
            let varargs_slot = if func.has_varargs() { 1 } else { 0 };
            arg_count + varargs_slot + kwonly_offset
        }
    }

    /// Create a string key Value for use in kwargs dict.
    ///
    /// Note: This creates a temporary Value. For high-frequency paths,
    /// consider string interning.
    #[inline]
    fn create_string_key(name: &str) -> Value {
        Value::string(intern(name))
    }
}
