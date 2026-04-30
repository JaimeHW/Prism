//! Native public `random` module facade.
//!
//! CPython's `random.py` is broad, but a small set of module-level helpers is
//! used heavily by regression tests and bootstrap code. Prism implements those
//! helpers natively on top of a lock-free SplitMix64 stream so common paths
//! avoid source import overhead and list shuffles mutate in place.

use super::{_random, Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::list::{ListObject, object_ptr_as_list_mut, value_as_list_ref};
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::value_as_tuple_ref;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

const MODULE_DOC: &str = "Native pseudo-random number helpers.";
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

static GLOBAL_STATE: AtomicU64 = AtomicU64::new(0xD1B5_4A32_D192_ED03);
static RANDOM_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new("random.random".into(), random_builtin));
static SEED_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new("random.seed".into(), seed_builtin));
static GETRANDBITS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new("random.getrandbits".into(), getrandbits_builtin));
static RANDRANGE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new("random.randrange".into(), randrange_builtin));
static RANDINT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new("random.randint".into(), randint_builtin));
static CHOICE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new("random.choice".into(), choice_builtin));
static CHOICES_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw("random.choices".into(), choices_builtin));
static SHUFFLE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new("random.shuffle".into(), shuffle_builtin));
static SAMPLE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new("random.sample".into(), sample_builtin));

const EXPORTS: &[&str] = &[
    "Random",
    "random",
    "seed",
    "getrandbits",
    "randrange",
    "randint",
    "choice",
    "choices",
    "shuffle",
    "sample",
];

/// Native `random` module descriptor.
#[derive(Debug, Clone)]
pub struct RandomPublicModule {
    attrs: Vec<std::sync::Arc<str>>,
    all: Value,
}

impl RandomPublicModule {
    /// Create a native public `random` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS
                .iter()
                .copied()
                .chain(["__all__", "__doc__"])
                .map(std::sync::Arc::from)
                .collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for RandomPublicModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for RandomPublicModule {
    fn name(&self) -> &str {
        "random"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "Random" => Ok(_random::random_class_value()),
            "random" => Ok(builtin_value(&RANDOM_FUNCTION)),
            "seed" => Ok(builtin_value(&SEED_FUNCTION)),
            "getrandbits" => Ok(builtin_value(&GETRANDBITS_FUNCTION)),
            "randrange" => Ok(builtin_value(&RANDRANGE_FUNCTION)),
            "randint" => Ok(builtin_value(&RANDINT_FUNCTION)),
            "choice" => Ok(builtin_value(&CHOICE_FUNCTION)),
            "choices" => Ok(builtin_value(&CHOICES_FUNCTION)),
            "shuffle" => Ok(builtin_value(&SHUFFLE_FUNCTION)),
            "sample" => Ok(builtin_value(&SAMPLE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'random' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<std::sync::Arc<str>> {
        self.attrs.clone()
    }
}

fn random_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("random", args, 0)?;
    Ok(Value::float(next_f64()))
}

fn seed_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "seed() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    let seed = args.first().copied().unwrap_or_else(Value::none);
    let state = seed_to_state(seed)?;
    GLOBAL_STATE.store(state, Ordering::Relaxed);
    Ok(Value::none())
}

fn getrandbits_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("getrandbits", args, 1)?;
    let bits = usize_from_value(args[0], "number of bits")?;
    if bits == 0 {
        return Ok(Value::int_unchecked(0));
    }

    let byte_len = bits.div_ceil(8);
    let mut bytes = vec![0_u8; byte_len];
    for chunk in bytes.chunks_mut(8) {
        let random = next_u64().to_le_bytes();
        chunk.copy_from_slice(&random[..chunk.len()]);
    }

    let excess_bits = (byte_len * 8) - bits;
    if excess_bits > 0 {
        *bytes
            .last_mut()
            .expect("positive bit count should allocate bytes") &= 0xFF_u8 >> excess_bits;
    }

    Ok(bigint_to_value(BigInt::from_bytes_le(Sign::Plus, &bytes)))
}

fn randrange_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "randrange expected 1 to 3 arguments, got {}",
            args.len()
        )));
    }

    let (start, stop, step) = match args {
        [stop] => (BigInt::from(0), int_value(*stop, "stop")?, BigInt::from(1)),
        [start, stop] => (
            int_value(*start, "start")?,
            int_value(*stop, "stop")?,
            BigInt::from(1),
        ),
        [start, stop, step] => (
            int_value(*start, "start")?,
            int_value(*stop, "stop")?,
            int_value(*step, "step")?,
        ),
        _ => unreachable!(),
    };

    let width = range_len(&start, &stop, &step)?;
    let offset = random_below_bigint(&width);
    Ok(bigint_to_value(start + step * offset))
}

fn randint_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("randint", args, 2)?;
    let start = int_value(args[0], "start")?;
    let stop = int_value(args[1], "stop")? + BigInt::from(1);
    let width = range_len(&start, &stop, &BigInt::from(1))?;
    Ok(bigint_to_value(start + random_below_bigint(&width)))
}

fn choice_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("choice", args, 1)?;
    let values = population_values(args[0])?;
    if values.is_empty() {
        return Err(BuiltinError::IndexError(
            "Cannot choose from an empty sequence".to_string(),
        ));
    }
    Ok(values[random_below(values.len())])
}

fn choices_builtin(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (population, weights, cum_weights, k) = bind_choices_args(args, keywords)?;
    let values = population_values(population)?;
    let count = choice_count_from_value(k)?;
    if count == 0 {
        return Ok(crate::alloc_managed_value(ListObject::from_iter(Vec::<
            Value,
        >::new(
        ))));
    }
    if values.is_empty() {
        return Err(BuiltinError::IndexError(
            "Cannot choose from an empty sequence".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(count);
    if weights.is_none() && cum_weights.is_none() {
        for _ in 0..count {
            result.push(values[random_below(values.len())]);
        }
        return Ok(crate::alloc_managed_value(ListObject::from_iter(result)));
    }

    if weights.is_some() && cum_weights.is_some() {
        return Err(BuiltinError::TypeError(
            "Cannot specify both weights and cumulative weights".to_string(),
        ));
    }

    let cumulative = match (weights, cum_weights) {
        (Some(weights), None) => cumulative_weights_from_weights(weights, values.len())?,
        (None, Some(cum_weights)) => cumulative_weights(cum_weights, values.len())?,
        _ => unreachable!("uniform choices returned above"),
    };
    let total = *cumulative
        .last()
        .expect("non-empty population must have at least one cumulative weight");
    if total <= 0.0 {
        return Err(BuiltinError::ValueError(
            "Total of weights must be greater than zero".to_string(),
        ));
    }
    if !total.is_finite() {
        return Err(BuiltinError::ValueError(
            "Total of weights must be finite".to_string(),
        ));
    }

    for _ in 0..count {
        let needle = next_f64() * total;
        let index = cumulative
            .partition_point(|&weight| weight <= needle)
            .min(values.len() - 1);
        result.push(values[index]);
    }
    Ok(crate::alloc_managed_value(ListObject::from_iter(result)))
}

fn shuffle_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("shuffle", args, 1)?;
    let ptr = args[0]
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("shuffle() argument must be a list".to_string()))?;
    let list = object_ptr_as_list_mut(ptr as *mut ())
        .ok_or_else(|| BuiltinError::TypeError("shuffle() argument must be a list".to_string()))?;

    for i in (1..list.len()).rev() {
        let j = random_below(i + 1);
        list.swap_indices(i, j);
    }
    Ok(Value::none())
}

fn sample_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("sample", args, 2)?;
    let mut values = population_values(args[0])?;
    let k = usize_from_value(args[1], "sample size")?;
    if k > values.len() {
        return Err(BuiltinError::ValueError(
            "Sample larger than population or is negative".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(k);
    for i in 0..k {
        let j = i + random_below(values.len() - i);
        values.swap(i, j);
        result.push(values[i]);
    }
    Ok(crate::alloc_managed_value(ListObject::from_iter(result)))
}

fn bind_choices_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<(Value, Option<Value>, Option<Value>, Option<Value>), BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "choices() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut population = args.first().copied();
    let mut weights = args.get(1).copied();
    let mut cum_weights = None;
    let mut k = None;

    for (name, value) in keywords {
        match *name {
            "population" => {
                if population.is_some() {
                    return Err(BuiltinError::TypeError(
                        "choices() got multiple values for argument 'population'".to_string(),
                    ));
                }
                population = Some(*value);
            }
            "weights" => {
                if weights.is_some() {
                    return Err(BuiltinError::TypeError(
                        "choices() got multiple values for argument 'weights'".to_string(),
                    ));
                }
                weights = Some(*value);
            }
            "cum_weights" => {
                if cum_weights.is_some() {
                    return Err(BuiltinError::TypeError(
                        "choices() got multiple values for argument 'cum_weights'".to_string(),
                    ));
                }
                cum_weights = Some(*value);
            }
            "k" => {
                if k.is_some() {
                    return Err(BuiltinError::TypeError(
                        "choices() got multiple values for argument 'k'".to_string(),
                    ));
                }
                k = Some(*value);
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "choices() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }

    let population = population.ok_or_else(|| {
        BuiltinError::TypeError("choices() missing required argument 'population'".to_string())
    })?;
    Ok((population, weights, cum_weights, k))
}

fn population_values(value: Value) -> Result<Vec<Value>, BuiltinError> {
    if let Some(list) = value_as_list_ref(value) {
        return Ok(list.as_slice().to_vec());
    }
    if let Some(tuple) = value_as_tuple_ref(value) {
        return Ok(tuple.as_slice().to_vec());
    }
    if let Some(string) = value_as_string_ref(value) {
        return Ok(string.as_str().chars().map(char_value).collect::<Vec<_>>());
    }
    Err(BuiltinError::TypeError(
        "population must be a sequence".to_string(),
    ))
}

fn cumulative_weights_from_weights(
    value: Value,
    expected: usize,
) -> Result<Vec<f64>, BuiltinError> {
    let weights = numeric_sequence_values(value, "weights")?;
    if weights.len() != expected {
        return Err(BuiltinError::ValueError(
            "The number of weights does not match the population".to_string(),
        ));
    }

    let mut total = 0.0;
    let mut cumulative = Vec::with_capacity(weights.len());
    for weight in weights {
        total += numeric_value_to_f64(weight, "weights")?;
        cumulative.push(total);
    }
    Ok(cumulative)
}

fn cumulative_weights(value: Value, expected: usize) -> Result<Vec<f64>, BuiltinError> {
    let weights = numeric_sequence_values(value, "cum_weights")?;
    if weights.len() != expected {
        return Err(BuiltinError::ValueError(
            "The number of weights does not match the population".to_string(),
        ));
    }

    weights
        .into_iter()
        .map(|value| numeric_value_to_f64(value, "cum_weights"))
        .collect()
}

fn numeric_sequence_values(value: Value, name: &'static str) -> Result<Vec<Value>, BuiltinError> {
    if let Some(list) = value_as_list_ref(value) {
        return Ok(list.as_slice().to_vec());
    }
    if let Some(tuple) = value_as_tuple_ref(value) {
        return Ok(tuple.as_slice().to_vec());
    }
    Err(BuiltinError::TypeError(format!(
        "{name} must be a sequence"
    )))
}

fn range_len(start: &BigInt, stop: &BigInt, step: &BigInt) -> Result<BigInt, BuiltinError> {
    if step.sign() == Sign::NoSign {
        return Err(BuiltinError::ValueError(
            "step argument must not be zero".to_string(),
        ));
    }

    let one = BigInt::from(1);
    let len = if step.sign() == Sign::Plus {
        if start >= stop {
            BigInt::from(0)
        } else {
            ((stop - start) + (step - &one)) / step
        }
    } else if start <= stop {
        BigInt::from(0)
    } else {
        ((start - stop) + ((-step) - &one)) / (-step)
    };

    if len.sign() == Sign::NoSign {
        return Err(BuiltinError::ValueError(
            "empty range for randrange()".to_string(),
        ));
    }
    Ok(len)
}

fn int_value(value: Value, name: &'static str) -> Result<BigInt, BuiltinError> {
    value_to_bigint(value)
        .ok_or_else(|| BuiltinError::TypeError(format!("non-integer {name} for randrange()")))
}

fn usize_from_value(value: Value, name: &'static str) -> Result<usize, BuiltinError> {
    let integer = value_to_bigint(value)
        .ok_or_else(|| BuiltinError::TypeError(format!("{name} must be an integer")))?;
    if integer.sign() == Sign::Minus {
        return Err(BuiltinError::ValueError(format!(
            "{name} must be non-negative"
        )));
    }
    integer
        .to_usize()
        .ok_or_else(|| BuiltinError::OverflowError(format!("{name} is too large")))
}

fn seed_to_state(value: Value) -> Result<u64, BuiltinError> {
    if value.is_none() {
        return Ok(crate::stdlib::secure_random::secure_random_u64()?);
    }
    if let Some(integer) = value_to_bigint(value) {
        return Ok(integer.to_u64().unwrap_or_else(|| {
            let bytes = integer.to_signed_bytes_le();
            hash_bytes(&bytes)
        }));
    }
    if let Some(float) = value.as_float() {
        return Ok(float.to_bits());
    }
    if let Some(string) = value_as_string_ref(value) {
        return Ok(hash_bytes(string.as_str().as_bytes()));
    }
    Err(BuiltinError::TypeError(
        "seed must be None, int, float, str, bytes, or bytearray".to_string(),
    ))
}

fn choice_count_from_value(value: Option<Value>) -> Result<usize, BuiltinError> {
    let Some(value) = value else {
        return Ok(1);
    };
    let integer = value_to_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "'{}' object cannot be interpreted as an integer",
            value.type_name()
        ))
    })?;
    if integer.sign() == Sign::Minus {
        return Ok(0);
    }
    integer
        .to_usize()
        .ok_or_else(|| BuiltinError::OverflowError("k is too large".to_string()))
}

fn numeric_value_to_f64(value: Value, name: &'static str) -> Result<f64, BuiltinError> {
    if let Some(float) = value.as_float() {
        return Ok(float);
    }
    value_to_bigint(value)
        .and_then(|integer| integer.to_f64())
        .ok_or_else(|| BuiltinError::TypeError(format!("{name} must be numeric")))
}

#[inline]
fn next_u64() -> u64 {
    mix64(GLOBAL_STATE.fetch_add(SPLITMIX64_GAMMA, Ordering::Relaxed))
}

#[inline]
fn next_f64() -> f64 {
    let bits = next_u64() >> 11;
    (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn random_below(upper: usize) -> usize {
    debug_assert!(upper > 0);
    random_below_u128(upper as u128) as usize
}

fn random_below_u128(upper: u128) -> u128 {
    debug_assert!(upper > 0);
    if upper <= u64::MAX as u128 {
        let upper = upper as u64;
        let zone = u64::MAX - (u64::MAX % upper);
        loop {
            let value = next_u64();
            if value < zone {
                return (value % upper) as u128;
            }
        }
    }

    let zone = u128::MAX - (u128::MAX % upper);
    loop {
        let high = next_u64() as u128;
        let low = next_u64() as u128;
        let value = (high << 64) | low;
        if value < zone {
            return value % upper;
        }
    }
}

fn random_below_bigint(upper: &BigInt) -> BigInt {
    debug_assert!(upper.sign() == Sign::Plus);

    if let Some(upper) = upper.to_u128() {
        return BigInt::from(random_below_u128(upper));
    }

    let bit_len = positive_bigint_bit_len(upper);
    let byte_len = bit_len.div_ceil(8);
    let excess_bits = (byte_len * 8) - bit_len;

    loop {
        let mut bytes = vec![0_u8; byte_len];
        for chunk in bytes.chunks_mut(8) {
            let random = next_u64().to_le_bytes();
            chunk.copy_from_slice(&random[..chunk.len()]);
        }
        if excess_bits > 0 {
            *bytes
                .last_mut()
                .expect("positive bit length should allocate bytes") &= 0xFF_u8 >> excess_bits;
        }

        let candidate = BigInt::from_bytes_le(Sign::Plus, &bytes);
        if &candidate < upper {
            return candidate;
        }
    }
}

fn positive_bigint_bit_len(value: &BigInt) -> usize {
    let (_, bytes) = value.to_bytes_le();
    let last = *bytes
        .last()
        .expect("positive BigInt should have at least one byte");
    ((bytes.len() - 1) * 8) + (u8::BITS as usize - last.leading_zeros() as usize)
}

#[inline]
fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xA076_1D64_78BD_642F_u64 ^ (bytes.len() as u64);
    for chunk in bytes.chunks(8) {
        let mut word = [0_u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        hash = mix64(hash ^ u64::from_le_bytes(word));
    }
    hash
}

fn char_value(ch: char) -> Value {
    let mut buffer = [0_u8; 4];
    Value::string(intern(ch.encode_utf8(&mut buffer)))
}

#[inline]
fn expect_arg_count(
    function: &'static str,
    args: &[Value],
    expected: usize,
) -> Result<(), BuiltinError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{function}() takes exactly {expected} argument{} ({} given)",
            if expected == 1 { "" } else { "s" },
            args.len()
        )))
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}
