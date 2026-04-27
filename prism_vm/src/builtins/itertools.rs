use super::BuiltinError;
use crate::VirtualMachine;
use crate::ops::calls::value_supports_call_protocol;
use crate::ops::iteration::{IterStep, collect_iterable_values, ensure_iterator_value, next_step};
use crate::stdlib::collections::deque::{DequeObject, value_as_deque};
use num_traits::{One, Zero};
use prism_core::Value;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::list::value_as_list_ref;
use prism_runtime::types::range::RangeObject;

// =============================================================================
// range
// =============================================================================

/// Builtin range function.
///
/// range(stop) -> range object
/// range(start, stop[, step]) -> range object
///
/// Returns a range object representing a sequence of integers.
pub fn builtin_range(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "range expected 1 to 3 arguments, got {}",
            args.len()
        )));
    }

    let parse_int = |value: Value, position: &'static str| {
        value_to_bigint(value).ok_or_else(|| {
            BuiltinError::TypeError(format!("range() integer {position} argument expected"))
        })
    };

    let (start, stop, step) = match args.len() {
        1 => {
            // range(stop)
            let stop = parse_int(args[0], "end")?;
            (num_bigint::BigInt::zero(), stop, num_bigint::BigInt::one())
        }
        2 => {
            // range(start, stop)
            let start = parse_int(args[0], "start")?;
            let stop = parse_int(args[1], "end")?;
            (start, stop, num_bigint::BigInt::one())
        }
        3 => {
            // range(start, stop, step)
            let start = parse_int(args[0], "start")?;
            let stop = parse_int(args[1], "end")?;
            let step = parse_int(args[2], "step")?;
            if step.is_zero() {
                return Err(BuiltinError::ValueError(
                    "range() arg 3 must not be zero".to_string(),
                ));
            }
            (start, stop, step)
        }
        _ => unreachable!(),
    };

    // Create RangeObject on heap and return as Value
    // TODO: Use GC allocator instead of Box::leak
    let range_obj = Box::new(RangeObject::from_bigints(start, stop, step));
    let ptr = Box::leak(range_obj) as *mut RangeObject as *const ();
    Ok(Value::object_ptr(ptr))
}

// =============================================================================
// iter
// =============================================================================

/// Builtin iter function.
///
/// iter(object) -> iterator
/// iter(callable, sentinel) -> iterator (sentinel form)
///
/// # Performance
///
/// - Built-in types: O(1) TypeId dispatch (~16 cycles)
/// - User-defined types: O(n) protocol lookup
///
/// # Examples
///
/// ```python
/// iter([1, 2, 3])      # Returns list_iterator
/// iter(range(5))       # Returns range_iterator
/// iter("hello")        # Returns str_iterator
/// iter({1: 'a'})       # Returns dict_keys iterator
/// ```
pub fn builtin_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    match args.len() {
        1 => {
            // iter(object) - standard form
            // Python semantics: iter(iterator) returns the same iterator object.
            if super::iter_dispatch::is_iterator(&args[0]) {
                return Ok(args[0]);
            }
            let iter = super::iter_dispatch::value_to_iterator(&args[0])?;
            Ok(super::iter_dispatch::iterator_to_value(iter))
        }
        2 => {
            if !value_supports_call_protocol(args[0]) {
                return Err(BuiltinError::TypeError(format!(
                    "iter(v, w): v must be callable (got '{}')",
                    get_type_name(&args[0])
                )));
            }

            let iter =
                prism_runtime::types::iter::IteratorObject::from_call_sentinel(args[0], args[1]);
            Ok(super::iter_dispatch::iterator_to_value(iter))
        }
        _ => Err(BuiltinError::TypeError(format!(
            "iter() expected 1 or 2 arguments, got {}",
            args.len()
        ))),
    }
}

/// VM-aware iter builtin.
pub fn builtin_iter_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    match args.len() {
        1 => ensure_iterator_value(vm, args[0]).map_err(super::runtime_error_to_builtin_error),
        2 => {
            if !value_supports_call_protocol(args[0]) {
                return Err(BuiltinError::TypeError(format!(
                    "iter(v, w): v must be callable (got '{}')",
                    get_type_name(&args[0])
                )));
            }

            let iter =
                prism_runtime::types::iter::IteratorObject::from_call_sentinel(args[0], args[1]);
            Ok(super::iter_dispatch::iterator_to_value(iter))
        }
        _ => Err(BuiltinError::TypeError(format!(
            "iter() expected 1 or 2 arguments, got {}",
            args.len()
        ))),
    }
}

// =============================================================================
// next
// =============================================================================

/// Builtin next function.
///
/// next(iterator[, default]) -> next item from iterator
///
/// # Performance
///
/// O(1) for iterator objects - single method call.
///
/// # Raises
///
/// - TypeError: if argument is not an iterator
/// - StopIteration: if iterator is exhausted and no default provided
///
/// # Examples
///
/// ```python
/// it = iter([1, 2])
/// next(it)        # Returns 1
/// next(it)        # Returns 2
/// next(it)        # Raises StopIteration
/// next(it, None)  # Returns None (default)
/// ```
pub fn builtin_next(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "next() expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let default = args.get(1).copied();

    // Get mutable reference to iterator
    let iter_obj = super::iter_dispatch::get_iterator_mut(&args[0]).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "'{}' object is not an iterator",
            get_type_name(&args[0])
        ))
    })?;

    // Get next value
    match iter_obj.next() {
        Some(value) => Ok(value),
        None => match default {
            Some(d) => Ok(d),
            None => Err(BuiltinError::StopIteration),
        },
    }
}

/// VM-aware next builtin.
pub fn builtin_next_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "next() expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let default = args.get(1).copied();
    match next_step(vm, args[0]).map_err(super::runtime_error_to_builtin_error)? {
        IterStep::Yielded(value) => Ok(value),
        IterStep::Exhausted => match default {
            Some(d) => Ok(d),
            None => Err(BuiltinError::StopIteration),
        },
    }
}

/// Get type name for error messages.
#[inline]
fn get_type_name(value: &Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.as_bool().is_some() {
        "bool"
    } else if value.as_int().is_some() {
        "int"
    } else if value.as_float().is_some() {
        "float"
    } else if value.is_string() {
        "str"
    } else {
        "object"
    }
}

// =============================================================================
// enumerate
// =============================================================================

/// Builtin enumerate function.
///
/// enumerate(iterable, start=0) -> enumerate object
pub fn builtin_enumerate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "enumerate expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    // Parse start argument (default 0)
    let start = if args.len() == 2 {
        args[1].as_int().ok_or_else(|| {
            BuiltinError::TypeError("'start' argument must be an integer".to_string())
        })?
    } else {
        0
    };

    // Convert iterable to iterator using O(1) TypeId dispatch
    let inner = super::iter_dispatch::value_to_iterator(&args[0])?;

    // Create enumerate iterator
    let enumerate = prism_runtime::types::iter::IteratorObject::enumerate(inner, start);
    Ok(super::iter_dispatch::iterator_to_value(enumerate))
}

/// VM-aware enumerate builtin.
pub fn builtin_enumerate_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "enumerate expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let start = if args.len() == 2 {
        args[1].as_int().ok_or_else(|| {
            BuiltinError::TypeError("'start' argument must be an integer".to_string())
        })?
    } else {
        0
    };

    let iterator =
        ensure_iterator_value(vm, args[0]).map_err(super::runtime_error_to_builtin_error)?;
    let enumerate = prism_runtime::types::iter::IteratorObject::enumerate(
        prism_runtime::types::iter::IteratorObject::from_existing_iterator(iterator),
        start,
    );
    Ok(super::iter_dispatch::iterator_to_value(enumerate))
}

// =============================================================================
// zip
// =============================================================================

/// Builtin zip function.
///
/// zip(*iterables) -> zip object
///
/// # Performance
///
/// - O(k) construction where k = number of iterables
/// - O(k) per iteration step
/// - Terminates when any iterator is exhausted (shortest-first)
///
/// # Examples
///
/// ```python
/// zip([1, 2, 3], ['a', 'b', 'c'])  # -> [(1, 'a'), (2, 'b'), (3, 'c')]
/// zip([1, 2], ['a', 'b', 'c'])     # -> [(1, 'a'), (2, 'b')] (shortest)
/// zip()                            # -> empty iterator
/// ```
pub fn builtin_zip(args: &[Value]) -> Result<Value, BuiltinError> {
    // Convert all arguments to iterators
    let mut iterators = Vec::with_capacity(args.len());
    for arg in args {
        let iter = super::iter_dispatch::value_to_iterator(arg)?;
        iterators.push(iter);
    }

    // Create zip iterator
    let zip_iter = prism_runtime::types::iter::IteratorObject::zip(iterators);
    Ok(super::iter_dispatch::iterator_to_value(zip_iter))
}

/// VM-aware zip builtin for protocol-based iterables.
pub fn builtin_zip_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let mut iterators = Vec::with_capacity(args.len());
    for &arg in args {
        let iterator =
            ensure_iterator_value(vm, arg).map_err(super::runtime_error_to_builtin_error)?;
        iterators
            .push(prism_runtime::types::iter::IteratorObject::from_existing_iterator(iterator));
    }

    let zip_iter = prism_runtime::types::iter::IteratorObject::zip(iterators);
    Ok(super::iter_dispatch::iterator_to_value(zip_iter))
}

// =============================================================================
// map
// =============================================================================

/// Builtin map function.
///
/// map(function, iterable, ...) -> map object
///
/// # Note
///
/// Currently only supports single-iterable map. The function is stored and
/// must be called by the VM when iterating (lazy evaluation).
pub fn builtin_map(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 {
        return Err(BuiltinError::TypeError(format!(
            "map expected at least 2 arguments, got {}",
            args.len()
        )));
    }

    let func = args[0];

    // For now, only support single iterable
    if args.len() > 2 {
        return Err(BuiltinError::NotImplemented(
            "map() with multiple iterables not yet implemented".to_string(),
        ));
    }

    let inner = super::iter_dispatch::value_to_iterator(&args[1])?;

    // Create map iterator (function call handled by VM on iteration)
    let map_iter = prism_runtime::types::iter::IteratorObject::map(func, inner);
    Ok(super::iter_dispatch::iterator_to_value(map_iter))
}

/// VM-aware map builtin for protocol-based iterables.
pub fn builtin_map_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 {
        return Err(BuiltinError::TypeError(format!(
            "map expected at least 2 arguments, got {}",
            args.len()
        )));
    }

    if args.len() > 2 {
        return Err(BuiltinError::NotImplemented(
            "map() with multiple iterables not yet implemented".to_string(),
        ));
    }

    let iterator =
        ensure_iterator_value(vm, args[1]).map_err(super::runtime_error_to_builtin_error)?;
    let inner = prism_runtime::types::iter::IteratorObject::from_existing_iterator(iterator);
    let map_iter = prism_runtime::types::iter::IteratorObject::map(args[0], inner);
    Ok(super::iter_dispatch::iterator_to_value(map_iter))
}

// =============================================================================
// filter
// =============================================================================

/// Builtin filter function.
///
/// filter(function, iterable) -> filter object
///
/// # Performance
///
/// - O(1) per iteration when predicate returns truthy
/// - O(n) when many consecutive falsy values (skipped)
///
/// # Special Cases
///
/// - filter(None, iterable) filters out falsy values (identity filter)
pub fn builtin_filter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "filter expected 2 arguments, got {}",
            args.len()
        )));
    }

    // Check if predicate is None (identity filter)
    let func = if args[0].is_none() {
        None
    } else {
        Some(args[0])
    };

    let inner = super::iter_dispatch::value_to_iterator(&args[1])?;

    // Create filter iterator
    let filter_iter = prism_runtime::types::iter::IteratorObject::filter(func, inner);
    Ok(super::iter_dispatch::iterator_to_value(filter_iter))
}

/// VM-aware filter builtin for protocol-based iterables.
pub fn builtin_filter_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "filter expected 2 arguments, got {}",
            args.len()
        )));
    }

    let func = (!args[0].is_none()).then_some(args[0]);
    let iterator =
        ensure_iterator_value(vm, args[1]).map_err(super::runtime_error_to_builtin_error)?;
    let inner = prism_runtime::types::iter::IteratorObject::from_existing_iterator(iterator);
    let filter_iter = prism_runtime::types::iter::IteratorObject::filter(func, inner);
    Ok(super::iter_dispatch::iterator_to_value(filter_iter))
}

// =============================================================================
// reversed
// =============================================================================

/// Builtin reversed function.
///
/// reversed(sequence) -> reverse iterator
///
/// # Performance
///
/// - Lists use a live reverse iterator: O(1) construction and CPython-compatible
///   shrink handling.
/// - Deques use a guarded snapshot: O(n) construction and O(1) mutation checks.
/// - Other iterables currently materialize into a compact reverse iterator.
pub fn builtin_reversed(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "reversed expected 1 argument, got {}",
            args.len()
        )));
    }

    if value_as_list_ref(args[0]).is_some() {
        let reversed = prism_runtime::types::iter::IteratorObject::reversed_list(args[0]);
        return Ok(super::iter_dispatch::iterator_to_value(reversed));
    }

    if let Some(deque) = value_as_deque(&args[0]) {
        let values = deque.deque().iter().copied().collect();
        let reversed = prism_runtime::types::iter::IteratorObject::guarded_reversed_values(
            args[0],
            values,
            deque_len_guard,
            "deque mutated during iteration",
        );
        return Ok(super::iter_dispatch::iterator_to_value(reversed));
    }

    // Convert to iterator and collect all values
    let mut iter = super::iter_dispatch::value_to_iterator(&args[0])?;
    let values = iter.collect_remaining();

    // Create reversed iterator
    let reversed = prism_runtime::types::iter::IteratorObject::reversed(values);
    Ok(super::iter_dispatch::iterator_to_value(reversed))
}

#[inline]
fn deque_len_guard(value: Value) -> Option<usize> {
    value_as_deque(&value).map(DequeObject::len)
}

// =============================================================================
// sorted
// =============================================================================

/// Builtin sorted function.
///
/// sorted(iterable, /, *, key=None, reverse=False) -> list
///
/// # Performance
///
/// - O(n log n) time complexity using Rust's Timsort
/// - O(n) space for collecting iterable + O(log n) for sort
/// - Stable sort: equal elements maintain relative order
///
/// # Arguments
///
/// - `iterable`: Any iterable to sort
/// - `key`: Optional function to extract comparison key (currently placeholder)
/// - `reverse`: If `True`, sort in descending order
///
/// # Examples
///
/// ```python
/// sorted([3, 1, 2])                    # [1, 2, 3]
/// sorted([3, 1, 2], reverse=True)      # [3, 2, 1]
/// sorted("hello")                       # ['e', 'h', 'l', 'l', 'o']
/// sorted(range(5, 0, -1))              # [1, 2, 3, 4, 5]
/// ```
pub fn builtin_sorted(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "sorted expected 1 to 3 arguments, got {}",
            args.len()
        )));
    }

    // Parse optional key function (arg[1]) and reverse flag (arg[2])
    let key_func = args.get(1).filter(|v| !v.is_none()).copied();
    let reverse = args.get(2).and_then(|v| v.as_bool()).unwrap_or(false);

    // Collect iterable elements
    let mut iter = super::iter_dispatch::value_to_iterator(&args[0])?;
    let mut values = iter.collect_remaining();

    // If key function is provided, we need VM integration.
    // For now, return NotImplemented for key function.
    if key_func.is_some() {
        return Err(BuiltinError::NotImplemented(
            "sorted() with key function requires VM integration".to_string(),
        ));
    }

    // Sort using Python-compatible Value comparison
    // Uses stable sort (timsort) which is O(n log n)
    values.sort_by(|a, b| compare_values(a, b));

    // Reverse if requested
    if reverse {
        values.reverse();
    }

    // Return as list (allocate on heap)
    let list = prism_runtime::types::list::ListObject::from_slice(&values);
    let ptr = Box::leak(Box::new(list)) as *mut prism_runtime::types::list::ListObject as *const ();
    Ok(Value::object_ptr(ptr))
}

/// VM-aware sorted builtin.
pub fn builtin_sorted_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "sorted expected 1 to 3 arguments, got {}",
            args.len()
        )));
    }

    let key_func = args.get(1).filter(|v| !v.is_none()).copied();
    let reverse = args.get(2).and_then(|v| v.as_bool()).unwrap_or(false);
    let mut values =
        collect_iterable_values(vm, args[0]).map_err(super::runtime_error_to_builtin_error)?;

    if key_func.is_some() {
        return Err(BuiltinError::NotImplemented(
            "sorted() with key function requires VM integration".to_string(),
        ));
    }

    values.sort_by(|a, b| compare_values(a, b));

    if reverse {
        values.reverse();
    }

    let list = prism_runtime::types::list::ListObject::from_slice(&values);
    let ptr = Box::leak(Box::new(list)) as *mut prism_runtime::types::list::ListObject as *const ();
    Ok(Value::object_ptr(ptr))
}

/// Compare two Values for sorting.
///
/// # Comparison Order (Python semantics)
///
/// 1. None < bool < int/float < str < other
/// 2. Within types: natural ordering
/// 3. Cross-type numeric: int vs float uses float comparison
///
/// # Performance
///
/// - O(1) for numeric types
/// - O(n) for strings (lexicographic)
#[inline]
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    // None is smallest
    match (a.is_none(), b.is_none()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }

    // Boolean comparison
    if let (Some(a_bool), Some(b_bool)) = (a.as_bool(), b.as_bool()) {
        return a_bool.cmp(&b_bool);
    }

    // Cross-type bool vs numeric: bool < numeric
    match (
        a.as_bool(),
        b.as_int().or_else(|| b.as_float().map(|f| f as i64)),
    ) {
        (Some(_), Some(_)) => return Ordering::Less,
        _ => {}
    }
    match (
        a.as_int().or_else(|| a.as_float().map(|f| f as i64)),
        b.as_bool(),
    ) {
        (Some(_), Some(_)) => return Ordering::Greater,
        _ => {}
    }

    // Integer comparison
    if let (Some(a_int), Some(b_int)) = (a.as_int(), b.as_int()) {
        return a_int.cmp(&b_int);
    }

    // Float comparison
    if let (Some(a_float), Some(b_float)) = (a.as_float(), b.as_float()) {
        return a_float.partial_cmp(&b_float).unwrap_or(Ordering::Equal);
    }

    // Cross-type int vs float: promote int to float
    if let (Some(a_int), Some(b_float)) = (a.as_int(), b.as_float()) {
        return (a_int as f64)
            .partial_cmp(&b_float)
            .unwrap_or(Ordering::Equal);
    }
    if let (Some(a_float), Some(b_int)) = (a.as_float(), b.as_int()) {
        return a_float
            .partial_cmp(&(b_int as f64))
            .unwrap_or(Ordering::Equal);
    }

    // String comparison (lexicographic)
    if a.is_string() && b.is_string() {
        // Get string data for comparison
        // For now, compare by pointer address as placeholder
        // TODO: Proper string content comparison
        let a_ptr = a.as_object_ptr().unwrap_or(std::ptr::null());
        let b_ptr = b.as_object_ptr().unwrap_or(std::ptr::null());
        return (a_ptr as usize).cmp(&(b_ptr as usize));
    }

    // String vs numeric: numeric < string
    if a.is_string() {
        if b.as_int().is_some() || b.as_float().is_some() {
            return Ordering::Greater;
        }
    }
    if b.is_string() {
        if a.as_int().is_some() || a.as_float().is_some() {
            return Ordering::Less;
        }
    }

    // Default: compare by pointer (stable but arbitrary)
    let a_ptr = a.as_object_ptr().unwrap_or(std::ptr::null());
    let b_ptr = b.as_object_ptr().unwrap_or(std::ptr::null());
    (a_ptr as usize).cmp(&(b_ptr as usize))
}

// =============================================================================
// all / any
// =============================================================================

/// Builtin all function.
///
/// all(iterable) -> bool - True if all elements are truthy
///
/// # Performance
///
/// - O(n) worst case when all elements truthy
/// - O(1) best case with early exit on first falsy element
/// - Uses iterator dispatch for ~16 cycle setup
///
/// # Examples
///
/// ```python
/// all([1, 2, 3])      # True - all truthy
/// all([1, 0, 3])      # False - 0 is falsy, exits early
/// all([])             # True - vacuous truth
/// all(range(1, 100))  # True - all positive ints
/// ```
pub fn builtin_all(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "all expected 1 argument, got {}",
            args.len()
        )));
    }

    // Convert to iterator using O(1) TypeId dispatch
    let mut iter = super::iter_dispatch::value_to_iterator(&args[0])?;

    // Iterate with early exit on first falsy element
    while let Some(value) = iter.next() {
        if !crate::truthiness::is_truthy(value) {
            return Ok(Value::bool(false));
        }
    }

    // All elements were truthy (or empty iterable - vacuous truth)
    Ok(Value::bool(true))
}

/// VM-aware all builtin.
pub fn builtin_all_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "all expected 1 argument, got {}",
            args.len()
        )));
    }

    let iterator =
        ensure_iterator_value(vm, args[0]).map_err(super::runtime_error_to_builtin_error)?;
    loop {
        match next_step(vm, iterator).map_err(super::runtime_error_to_builtin_error)? {
            IterStep::Yielded(value) => {
                if !crate::truthiness::try_is_truthy(vm, value)
                    .map_err(super::runtime_error_to_builtin_error)?
                {
                    return Ok(Value::bool(false));
                }
            }
            IterStep::Exhausted => return Ok(Value::bool(true)),
        }
    }
}

/// Builtin any function.
///
/// any(iterable) -> bool - True if any element is truthy
///
/// # Performance
///
/// - O(n) worst case when all elements falsy
/// - O(1) best case with early exit on first truthy element
/// - Uses iterator dispatch for ~16 cycle setup
///
/// # Examples
///
/// ```python
/// any([0, 0, 1])      # True - 1 is truthy, exits early
/// any([0, 0, 0])      # False - all falsy
/// any([])             # False - empty iterable
/// any(range(10))      # True - 1-9 are truthy
/// ```
pub fn builtin_any(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "any expected 1 argument, got {}",
            args.len()
        )));
    }

    // Convert to iterator using O(1) TypeId dispatch
    let mut iter = super::iter_dispatch::value_to_iterator(&args[0])?;

    // Iterate with early exit on first truthy element
    while let Some(value) = iter.next() {
        if crate::truthiness::is_truthy(value) {
            return Ok(Value::bool(true));
        }
    }

    // No truthy elements found
    Ok(Value::bool(false))
}

/// VM-aware any builtin.
pub fn builtin_any_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "any expected 1 argument, got {}",
            args.len()
        )));
    }

    let iterator =
        ensure_iterator_value(vm, args[0]).map_err(super::runtime_error_to_builtin_error)?;
    loop {
        match next_step(vm, iterator).map_err(super::runtime_error_to_builtin_error)? {
            IterStep::Yielded(value) => {
                if crate::truthiness::try_is_truthy(vm, value)
                    .map_err(super::runtime_error_to_builtin_error)?
                {
                    return Ok(Value::bool(true));
                }
            }
            IterStep::Exhausted => return Ok(Value::bool(false)),
        }
    }
}
