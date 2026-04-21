//! Python `itertools` module implementation.
//!
//! Provides a comprehensive, high-performance implementation of Python's
//! `itertools` standard library module, split into logical submodules:
//!
//! - [`infinite`] — `count`, `cycle`, `repeat`
//! - [`terminating`] — `chain`, `compress`, `dropwhile`, `takewhile`,
//!   `filterfalse`, `islice`, `starmap`, `zip_longest`
//! - [`combinatoric`] — `product`, `permutations`, `combinations`,
//!   `combinations_with_replacement`
//! - [`grouping`] — `groupby`, `pairwise`, `batched`, `triplewise`
//! - [`recipes`] — `flatten`, `unique_everseen`, `unique_justseen`,
//!   `sliding_window`, `roundrobin`, `accumulate`, `partition`, `quantify`
//!
//! # Performance
//!
//! All iterators are zero-allocation where possible, implement `FusedIterator`,
//! provide accurate `size_hint()`, and use `#[inline]` on hot paths. Combinatoric
//! iterators use `SmallVec` for stack-allocated index arrays.

pub mod combinatoric;
pub mod grouping;
pub mod infinite;
pub mod recipes;
pub mod terminating;

#[cfg(test)]
mod tests;

// Re-export all public types for convenience
pub use combinatoric::{Combinations, CombinationsWithReplacement, Permutations, Product};
pub use grouping::{Batched, GroupBy, Pairwise, Triplewise};
pub use infinite::{Count, Cycle, Repeat};
pub use recipes::{Accumulate, Flatten, RoundRobin, SlidingWindow, UniqueEverseen, UniqueJustseen};
pub use terminating::{
    Chain, Compress, DropWhile, EitherOrBoth, FilterFalse, ISlice, Starmap, ZipLongest,
    chain_from_iterable,
};

use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::stdlib::{Module, ModuleResult};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::int::value_to_i64;
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::Arc;
use std::sync::LazyLock;

/// The `itertools` module implementation.
pub struct ItertoolsModule;

macro_rules! itertools_stub {
    ($static_name:ident, $func_name:ident, $qualified_name:literal, $display_name:literal) => {
        static $static_name: LazyLock<BuiltinFunctionObject> =
            LazyLock::new(|| BuiltinFunctionObject::new(Arc::from($qualified_name), $func_name));

        fn $func_name(_args: &[Value]) -> Result<Value, BuiltinError> {
            Err(BuiltinError::NotImplemented(format!(
                "{}() is not implemented yet",
                $display_name
            )))
        }
    };
}

static COUNT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("itertools.count"), builtin_count));
itertools_stub!(CYCLE_FUNCTION, builtin_cycle, "itertools.cycle", "cycle");
itertools_stub!(
    REPEAT_FUNCTION,
    builtin_repeat,
    "itertools.repeat",
    "repeat"
);
itertools_stub!(
    ACCUMULATE_FUNCTION,
    builtin_accumulate,
    "itertools.accumulate",
    "accumulate"
);
itertools_stub!(CHAIN_FUNCTION, builtin_chain, "itertools.chain", "chain");
itertools_stub!(
    COMPRESS_FUNCTION,
    builtin_compress,
    "itertools.compress",
    "compress"
);
itertools_stub!(
    DROPWHILE_FUNCTION,
    builtin_dropwhile,
    "itertools.dropwhile",
    "dropwhile"
);
itertools_stub!(
    TAKEWHILE_FUNCTION,
    builtin_takewhile,
    "itertools.takewhile",
    "takewhile"
);
itertools_stub!(
    FILTERFALSE_FUNCTION,
    builtin_filterfalse,
    "itertools.filterfalse",
    "filterfalse"
);
static ISLICE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("itertools.islice"), builtin_islice));
itertools_stub!(
    STARMAP_FUNCTION,
    builtin_starmap,
    "itertools.starmap",
    "starmap"
);
itertools_stub!(
    ZIP_LONGEST_FUNCTION,
    builtin_zip_longest,
    "itertools.zip_longest",
    "zip_longest"
);
static PRODUCT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("itertools.product"), builtin_product));
static PERMUTATIONS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("itertools.permutations"), builtin_permutations)
});
itertools_stub!(
    COMBINATIONS_FUNCTION,
    builtin_combinations,
    "itertools.combinations",
    "combinations"
);
itertools_stub!(
    COMBINATIONS_WR_FUNCTION,
    builtin_combinations_with_replacement,
    "itertools.combinations_with_replacement",
    "combinations_with_replacement"
);
itertools_stub!(
    GROUPBY_FUNCTION,
    builtin_groupby,
    "itertools.groupby",
    "groupby"
);
itertools_stub!(
    PAIRWISE_FUNCTION,
    builtin_pairwise,
    "itertools.pairwise",
    "pairwise"
);
itertools_stub!(
    BATCHED_FUNCTION,
    builtin_batched,
    "itertools.batched",
    "batched"
);

impl ItertoolsModule {
    /// Create a new itertools module.
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl Module for ItertoolsModule {
    fn name(&self) -> &str {
        "itertools"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            // Module-level constants/info
            "__name__" => Ok(Value::string(intern("itertools"))),
            "count" => Ok(builtin_value(&COUNT_FUNCTION)),
            "cycle" => Ok(builtin_value(&CYCLE_FUNCTION)),
            "repeat" => Ok(builtin_value(&REPEAT_FUNCTION)),
            "accumulate" => Ok(builtin_value(&ACCUMULATE_FUNCTION)),
            "chain" => Ok(builtin_value(&CHAIN_FUNCTION)),
            "compress" => Ok(builtin_value(&COMPRESS_FUNCTION)),
            "dropwhile" => Ok(builtin_value(&DROPWHILE_FUNCTION)),
            "takewhile" => Ok(builtin_value(&TAKEWHILE_FUNCTION)),
            "filterfalse" => Ok(builtin_value(&FILTERFALSE_FUNCTION)),
            "islice" => Ok(builtin_value(&ISLICE_FUNCTION)),
            "starmap" => Ok(builtin_value(&STARMAP_FUNCTION)),
            "zip_longest" => Ok(builtin_value(&ZIP_LONGEST_FUNCTION)),
            "product" => Ok(builtin_value(&PRODUCT_FUNCTION)),
            "permutations" => Ok(builtin_value(&PERMUTATIONS_FUNCTION)),
            "combinations" => Ok(builtin_value(&COMBINATIONS_FUNCTION)),
            "combinations_with_replacement" => Ok(builtin_value(&COMBINATIONS_WR_FUNCTION)),
            "groupby" => Ok(builtin_value(&GROUPBY_FUNCTION)),
            "pairwise" => Ok(builtin_value(&PAIRWISE_FUNCTION)),
            "batched" => Ok(builtin_value(&BATCHED_FUNCTION)),
            _ => Err(crate::stdlib::ModuleError::AttributeError(format!(
                "module 'itertools' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            Arc::from("count"),
            Arc::from("cycle"),
            Arc::from("repeat"),
            Arc::from("accumulate"),
            Arc::from("chain"),
            Arc::from("compress"),
            Arc::from("dropwhile"),
            Arc::from("takewhile"),
            Arc::from("filterfalse"),
            Arc::from("islice"),
            Arc::from("starmap"),
            Arc::from("zip_longest"),
            Arc::from("product"),
            Arc::from("permutations"),
            Arc::from("combinations"),
            Arc::from("combinations_with_replacement"),
            Arc::from("groupby"),
            Arc::from("pairwise"),
            Arc::from("batched"),
        ]
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn iterator_value(iter: IteratorObject) -> Value {
    crate::builtins::iterator_to_value(iter)
}

#[inline]
fn tuple_value(values: Vec<Value>) -> Value {
    let tuple = TupleObject::from_slice(&values);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    Value::object_ptr(ptr)
}

fn collect_iterable_values(value: Value) -> Result<Vec<Value>, BuiltinError> {
    let mut iterator = crate::builtins::value_to_iterator(&value).map_err(BuiltinError::from)?;
    Ok(iterator.collect_remaining())
}

fn parse_non_negative_usize(value: Value, name: &str) -> Result<usize, BuiltinError> {
    let Some(raw) = value_to_i64(value) else {
        return Err(BuiltinError::TypeError(format!(
            "{name} must be an integer"
        )));
    };
    if raw < 0 {
        return Err(BuiltinError::ValueError(format!(
            "{name} must be non-negative"
        )));
    }
    Ok(raw as usize)
}

fn parse_optional_non_negative_usize(
    value: Value,
    name: &str,
) -> Result<Option<usize>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }

    parse_non_negative_usize(value, name).map(Some)
}

fn builtin_count(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "count expected at most 2 arguments, got {}",
            args.len()
        )));
    }

    let start = args
        .first()
        .copied()
        .unwrap_or_else(|| Value::int(0).unwrap());
    let step = args
        .get(1)
        .copied()
        .unwrap_or_else(|| Value::int(1).unwrap());
    let iter = IteratorObject::count(start, step).ok_or_else(|| {
        BuiltinError::TypeError("count() requires integer or float arguments".to_string())
    })?;
    Ok(iterator_value(iter))
}

fn builtin_islice(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=4).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "islice expected 2 to 4 arguments, got {}",
            args.len()
        )));
    }

    let iterator = crate::builtins::value_to_iterator(&args[0]).map_err(BuiltinError::from)?;
    let (start, stop, step) = match args.len() {
        2 => (0, parse_optional_non_negative_usize(args[1], "stop")?, 1),
        3 => (
            parse_non_negative_usize(args[1], "start")?,
            parse_optional_non_negative_usize(args[2], "stop")?,
            1,
        ),
        4 => {
            let step = parse_non_negative_usize(args[3], "step")?;
            if step == 0 {
                return Err(BuiltinError::ValueError(
                    "step must be greater than zero".to_string(),
                ));
            }
            (
                parse_non_negative_usize(args[1], "start")?,
                parse_optional_non_negative_usize(args[2], "stop")?,
                step,
            )
        }
        _ => unreachable!("argument count validated above"),
    };

    Ok(iterator_value(IteratorObject::islice(
        iterator, start, stop, step,
    )))
}

fn builtin_product(args: &[Value]) -> Result<Value, BuiltinError> {
    let pools: Result<Vec<Vec<Value>>, BuiltinError> =
        args.iter().copied().map(collect_iterable_values).collect();
    let tuples = Product::new(pools?).map(tuple_value).collect::<Vec<_>>();
    Ok(iterator_value(IteratorObject::from_values(tuples)))
}

fn builtin_permutations(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "permutations expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let pool = collect_iterable_values(args[0])?;
    let r = match args.get(1).copied() {
        Some(value) => parse_non_negative_usize(value, "r")?,
        None => pool.len(),
    };
    let tuples = Permutations::new(pool, r)
        .map(tuple_value)
        .collect::<Vec<_>>();
    Ok(iterator_value(IteratorObject::from_values(tuples)))
}

#[cfg(test)]
mod mod_tests {
    use super::*;
    use crate::builtins::get_iterator_mut;
    use prism_core::intern::intern;

    #[test]
    fn test_module_name() {
        let m = ItertoolsModule::new();
        assert_eq!(m.name(), "itertools");
    }

    #[test]
    fn test_module_dir() {
        let m = ItertoolsModule::new();
        let attrs = m.dir();
        assert!(attrs.iter().any(|a| a.as_ref() == "count"));
        assert!(attrs.iter().any(|a| a.as_ref() == "chain"));
        assert!(attrs.iter().any(|a| a.as_ref() == "product"));
        assert!(attrs.iter().any(|a| a.as_ref() == "groupby"));
        assert!(attrs.iter().any(|a| a.as_ref() == "pairwise"));
        assert!(attrs.iter().any(|a| a.as_ref() == "batched"));
    }

    #[test]
    fn test_module_exports_bootstrap_callables() {
        let m = ItertoolsModule::new();

        for name in ["chain", "repeat", "starmap", "islice"] {
            assert!(
                m.get_attr(name)
                    .expect("callable should exist")
                    .as_object_ptr()
                    .is_some(),
                "{name} should be exposed as a callable object"
            );
        }
    }

    #[test]
    fn test_module_unknown_attr() {
        let m = ItertoolsModule::new();
        assert!(m.get_attr("nonexistent").is_err());
    }

    #[test]
    fn test_builtin_product_returns_cartesian_tuples() {
        let left = TupleObject::from_slice(&[Value::int_unchecked(1), Value::int_unchecked(2)]);
        let right =
            TupleObject::from_slice(&[Value::string(intern("a")), Value::string(intern("b"))]);
        let left_ptr = Box::leak(Box::new(left)) as *mut TupleObject as *const ();
        let right_ptr = Box::leak(Box::new(right)) as *mut TupleObject as *const ();

        let iter_value =
            builtin_product(&[Value::object_ptr(left_ptr), Value::object_ptr(right_ptr)])
                .expect("product() should succeed");
        let iter = get_iterator_mut(&iter_value).expect("product() should return an iterator");

        let first = iter.next().expect("expected first tuple");
        let second = iter.next().expect("expected second tuple");
        let first_ptr = first
            .as_object_ptr()
            .expect("product entries should be tuples");
        let second_ptr = second
            .as_object_ptr()
            .expect("product entries should be tuples");
        let first_tuple = unsafe { &*(first_ptr as *const TupleObject) };
        let second_tuple = unsafe { &*(second_ptr as *const TupleObject) };

        assert_eq!(first_tuple.get(0).unwrap().as_int(), Some(1));
        assert!(first_tuple.get(1).unwrap().is_string());
        assert_eq!(second_tuple.get(0).unwrap().as_int(), Some(1));
        assert!(second_tuple.get(1).unwrap().is_string());
    }

    #[test]
    fn test_builtin_product_with_no_args_yields_single_empty_tuple() {
        let iter_value = builtin_product(&[]).expect("product() should succeed");
        let iter = get_iterator_mut(&iter_value).expect("product() should return an iterator");
        let only = iter.next().expect("empty product should yield one tuple");
        let tuple_ptr = only
            .as_object_ptr()
            .expect("product entry should be a tuple");
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        assert_eq!(tuple.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_builtin_islice_slices_count_iterators_lazily() {
        let count = builtin_count(&[]).expect("count() should succeed");
        let iter_value = builtin_islice(&[
            count,
            Value::int_unchecked(3),
            Value::int_unchecked(8),
            Value::int_unchecked(2),
        ])
        .expect("islice() should succeed");
        let iter = get_iterator_mut(&iter_value).expect("islice() should return an iterator");

        let mut values = Vec::new();
        while let Some(value) = iter.next() {
            values.push(value.as_int().expect("islice(count()) should yield ints"));
        }

        assert_eq!(values, vec![3, 5, 7]);
    }

    #[test]
    fn test_builtin_islice_supports_none_stop() {
        let tuple = TupleObject::from_slice(&[
            Value::int_unchecked(10),
            Value::int_unchecked(11),
            Value::int_unchecked(12),
            Value::int_unchecked(13),
        ]);
        let tuple_ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();

        let iter_value = builtin_islice(&[
            Value::object_ptr(tuple_ptr),
            Value::int_unchecked(2),
            Value::none(),
        ])
        .expect("islice() should succeed");
        let iter = get_iterator_mut(&iter_value).expect("islice() should return an iterator");

        assert_eq!(iter.next().unwrap().as_int(), Some(12));
        assert_eq!(iter.next().unwrap().as_int(), Some(13));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_builtin_permutations_matches_cpython_order() {
        let iter_value = builtin_permutations(&[Value::string(intern("abc"))])
            .expect("permutations() should succeed");
        let iter = get_iterator_mut(&iter_value).expect("permutations() should return an iterator");

        let mut tuples = Vec::new();
        while let Some(value) = iter.next() {
            let tuple_ptr = value
                .as_object_ptr()
                .expect("permutations entries should be tuples");
            let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
            tuples.push((
                tuple.get(0).unwrap(),
                tuple.get(1).unwrap(),
                tuple.get(2).unwrap(),
            ));
        }

        assert_eq!(tuples.len(), 6);
        assert_eq!(tuples[0].0, Value::string(intern("a")));
        assert_eq!(tuples[0].1, Value::string(intern("b")));
        assert_eq!(tuples[0].2, Value::string(intern("c")));
        assert_eq!(tuples[1].0, Value::string(intern("a")));
        assert_eq!(tuples[1].1, Value::string(intern("c")));
        assert_eq!(tuples[1].2, Value::string(intern("b")));
        assert_eq!(tuples[5].0, Value::string(intern("c")));
        assert_eq!(tuples[5].1, Value::string(intern("b")));
        assert_eq!(tuples[5].2, Value::string(intern("a")));
    }

    #[test]
    fn test_builtin_permutations_rejects_negative_r() {
        let error = builtin_permutations(&[Value::string(intern("ab")), Value::int_unchecked(-1)])
            .expect_err("negative r should fail");
        assert!(matches!(error, BuiltinError::ValueError(_)));
    }
}
