//! Native `_random` compatibility layer.
//!
//! CPython's pure-Python `random.py` builds most of its API on top of the
//! `_random.Random` base class. Prism only needs a compact deterministic core
//! here: a subclassable type with `random()`, `seed()`, `getstate()`,
//! `setstate()`, and `getrandbits()` methods.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::stdlib::secure_random::secure_random_u64;
use num_bigint::{BigInt, Sign};
use num_traits::{ToPrimitive, Zero};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::string::StringObject;
use std::sync::{Arc, LazyLock};

const STATE_ATTR: &str = "__prism_random_state__";
const MODULE_DOC: &str = "Native compatibility implementation of the _random module.";
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

static RANDOM_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_random_class);
static RANDOM_INIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_random.Random.__init__"), random_init));
static RANDOM_RANDOM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_random.Random.random"), random_random));
static RANDOM_SEED_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_random.Random.seed"), random_seed));
static RANDOM_GETSTATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_random.Random.getstate"), random_getstate)
});
static RANDOM_SETSTATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_random.Random.setstate"), random_setstate)
});
static RANDOM_GETRANDBITS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_random.Random.getrandbits"), random_getrandbits)
});

/// Native `_random` module descriptor.
#[derive(Debug, Clone)]
pub struct RandomModule {
    attrs: Vec<Arc<str>>,
}

impl RandomModule {
    /// Create a new `_random` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("__doc__"), Arc::from("Random")],
        }
    }
}

impl Default for RandomModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for RandomModule {
    fn name(&self) -> &str {
        "_random"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "Random" => Ok(random_class_value()),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_random' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn random_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(random_class()) as *const ())
}

fn random_class() -> &'static Arc<PyClassObject> {
    &RANDOM_CLASS
}

fn build_random_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("Random"));

    class.set_attr(intern("__module__"), Value::string(intern("_random")));
    class.set_attr(intern("__qualname__"), Value::string(intern("Random")));
    class.set_attr(intern("__doc__"), Value::string(intern(MODULE_DOC)));
    class.set_attr(intern("__init__"), builtin_value(&RANDOM_INIT_METHOD));
    class.set_attr(intern("random"), builtin_value(&RANDOM_RANDOM_METHOD));
    class.set_attr(intern("seed"), builtin_value(&RANDOM_SEED_METHOD));
    class.set_attr(intern("getstate"), builtin_value(&RANDOM_GETSTATE_METHOD));
    class.set_attr(intern("setstate"), builtin_value(&RANDOM_SETSTATE_METHOD));
    class.set_attr(
        intern("getrandbits"),
        builtin_value(&RANDOM_GETRANDBITS_METHOD),
    );
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_INIT | ClassFlags::NATIVE_HEAPTYPE);

    let class = Arc::new(class);

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);

    class
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn random_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "Random.__init__() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let seed = args.get(1).copied().unwrap_or_else(Value::none);
    random_seed_impl(args[0], seed)?;
    Ok(Value::none())
}

fn random_seed(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "seed() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let seed = args.get(1).copied().unwrap_or_else(Value::none);
    random_seed_impl(args[0], seed)?;
    Ok(Value::none())
}

fn random_seed_impl(receiver: Value, seed: Value) -> Result<(), BuiltinError> {
    let state = seed_to_state(seed)?;
    store_state(receiver, state)
}

fn random_random(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "random() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let bits = next_random_u64(args[0])? >> 11;
    Ok(Value::float((bits as f64) * (1.0 / ((1_u64 << 53) as f64))))
}

fn random_getstate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "getstate() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    Ok(bigint_to_value(BigInt::from(ensure_state(args[0])?)))
}

fn random_setstate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "setstate() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let state = int_like_value_to_state(args[1], "setstate() state")?;
    store_state(args[0], state)?;
    Ok(Value::none())
}

fn random_getrandbits(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "getrandbits() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let bits = bit_count_from_value(args[1])?;
    if bits == 0 {
        return Ok(Value::int(0).expect("zero fits in Value::int"));
    }

    let byte_len = bits.div_ceil(8);
    let mut bytes = vec![0_u8; byte_len];
    let mut offset = 0;
    while offset < bytes.len() {
        let chunk = next_random_u64(args[0])?.to_le_bytes();
        let remaining = bytes.len() - offset;
        let copy_len = remaining.min(chunk.len());
        bytes[offset..offset + copy_len].copy_from_slice(&chunk[..copy_len]);
        offset += copy_len;
    }

    let excess_bits = (byte_len * 8) - bits;
    if excess_bits > 0 {
        let keep_mask = 0xFF_u8 >> excess_bits;
        let last = bytes
            .last_mut()
            .expect("bit_count > 0 should allocate at least one byte");
        *last &= keep_mask;
    }

    Ok(bigint_to_value(BigInt::from_bytes_le(Sign::Plus, &bytes)))
}

fn ensure_state(receiver: Value) -> Result<u64, BuiltinError> {
    let attr = intern(STATE_ATTR);
    if let Some(value) = shaped_object_ref(receiver)?.get_property_interned(&attr) {
        return int_like_value_to_state(value, STATE_ATTR);
    }

    let state = secure_random_u64()?;
    store_state(receiver, state)?;
    Ok(state)
}

fn next_random_u64(receiver: Value) -> Result<u64, BuiltinError> {
    let state = ensure_state(receiver)?.wrapping_add(SPLITMIX64_GAMMA);
    store_state(receiver, state)?;

    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    Ok(z ^ (z >> 31))
}

fn store_state(receiver: Value, state: u64) -> Result<(), BuiltinError> {
    shaped_object_mut(receiver)?.set_property(
        intern(STATE_ATTR),
        bigint_to_value(BigInt::from(state)),
        shape_registry(),
    );
    Ok(())
}

fn seed_to_state(seed: Value) -> Result<u64, BuiltinError> {
    if seed.is_none() {
        return secure_random_u64();
    }

    if let Some(bigint) = value_to_bigint(seed) {
        return Ok(hash_bigint_seed(&bigint));
    }

    if let Some(float) = seed.as_float() {
        return Ok(float.to_bits());
    }

    if let Some(text) = string_from_value(seed) {
        return Ok(hash_bytes_seed(text.as_bytes()));
    }

    if let Some(bytes) = bytes_from_value(seed) {
        return Ok(hash_bytes_seed(bytes));
    }

    Err(BuiltinError::TypeError(
        "seed must be None, int, float, str, bytes, or bytearray".to_string(),
    ))
}

fn bit_count_from_value(value: Value) -> Result<usize, BuiltinError> {
    let Some(bigint) = value_to_bigint(value) else {
        return Err(BuiltinError::TypeError(
            "number of bits must be an integer".to_string(),
        ));
    };

    if bigint.sign() == Sign::Minus {
        return Err(BuiltinError::ValueError(
            "number of bits must be non-negative".to_string(),
        ));
    }

    bigint
        .to_usize()
        .ok_or_else(|| BuiltinError::OverflowError("number of bits is too large".to_string()))
}

fn int_like_value_to_state(value: Value, context: &str) -> Result<u64, BuiltinError> {
    let Some(bigint) = value_to_bigint(value) else {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be an integer"
        )));
    };

    if bigint.sign() == Sign::Minus {
        return Err(BuiltinError::ValueError(format!(
            "{context} must be non-negative"
        )));
    }

    bigint
        .to_u64()
        .ok_or_else(|| BuiltinError::OverflowError(format!("{context} is too large")))
}

fn hash_bigint_seed(value: &BigInt) -> u64 {
    if value.is_zero() {
        return 0;
    }

    if let Some(u64_value) = value.to_u64() {
        return u64_value;
    }

    hash_bytes_seed(&value.to_signed_bytes_le())
}

fn hash_bytes_seed(bytes: &[u8]) -> u64 {
    let mut acc = 0xA076_1D64_78BD_642F_u64 ^ (bytes.len() as u64);
    for chunk in bytes.chunks(8) {
        let mut word = [0_u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        acc = mix_seed_word(acc ^ u64::from_le_bytes(word));
    }
    acc
}

fn mix_seed_word(word: u64) -> u64 {
    let mut z = word.wrapping_add(SPLITMIX64_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn shaped_object_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_random.Random methods require an instance".to_string())
    })?;
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn shaped_object_mut(value: Value) -> Result<&'static mut ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_random.Random methods require an instance".to_string())
    })?;
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn string_from_value(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        return interned_by_ptr(ptr as *const u8).map(|text| text.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    Some(
        unsafe { &*(ptr as *const StringObject) }
            .as_str()
            .to_string(),
    )
}

fn bytes_from_value(value: Value) -> Option<&'static [u8]> {
    let ptr = value.as_object_ptr()?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            Some(unsafe { &*(ptr as *const BytesObject) }.as_bytes())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
