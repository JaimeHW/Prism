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
use crate::ops::iteration::ensure_iterator_value;
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
static REPEAT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("itertools.repeat"), builtin_repeat));
itertools_stub!(
    ACCUMULATE_FUNCTION,
    builtin_accumulate,
    "itertools.accumulate",
    "accumulate"
);
static CHAIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("itertools.chain"), builtin_chain));
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

fn builtin_repeat(args: &[Value]) -> Result<Value, BuiltinError> {
    match args.len() {
        0 => {
            return Err(BuiltinError::TypeError(
                "repeat expected at least 1 argument, got 0".to_string(),
            ));
        }
        1 | 2 => {}
        given => {
            return Err(BuiltinError::TypeError(format!(
                "repeat expected at most 2 arguments, got {given}"
            )));
        }
    }

    let remaining = match args.get(1).copied() {
        Some(value) => {
            let raw = value_to_i64(value).ok_or_else(|| {
                BuiltinError::TypeError("repeat count must be an integer".to_string())
            })?;
            Some(usize::try_from(raw.max(0)).unwrap_or(usize::MAX))
        }
        None => None,
    };

    Ok(iterator_value(IteratorObject::repeat(args[0], remaining)))
}

fn builtin_chain(vm: &mut crate::VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let mut iterators = Vec::with_capacity(args.len());
    for iterable in args {
        let iterator_value = ensure_iterator_value(vm, *iterable)
            .map_err(crate::builtins::runtime_error_to_builtin_error)?;
        iterators.push(IteratorObject::from_existing_iterator(iterator_value));
    }

    Ok(iterator_value(IteratorObject::chain(iterators)))
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
mod mod_tests;
