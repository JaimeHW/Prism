//! Shared helpers for cryptographically secure random bytes.

use crate::builtins::BuiltinError;
use getrandom::getrandom;
use prism_core::Value;
use prism_runtime::types::bytes::BytesObject;

/// Parse a CPython-style `urandom(length)` call and return a bytes object.
pub(crate) fn urandom_value_from_args(
    args: &[Value],
    fn_name: &str,
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let Some(length) = args[0].as_int() else {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() argument must be an integer"
        )));
    };

    if length < 0 {
        return Err(BuiltinError::ValueError(
            "negative argument not allowed".to_string(),
        ));
    }

    let length = usize::try_from(length).map_err(|_| {
        BuiltinError::OverflowError(format!("{fn_name}() argument is too large to allocate"))
    })?;

    urandom_bytes_value(length)
}

/// Return `length` cryptographically secure random bytes as a Python `bytes`.
pub(crate) fn urandom_bytes_value(length: usize) -> Result<Value, BuiltinError> {
    let mut bytes = vec![0; length];
    fill_random_bytes(&mut bytes)?;
    Ok(leak_bytes_value(bytes))
}

/// Fill a caller-provided buffer with cryptographically secure random bytes.
pub(crate) fn fill_random_bytes(bytes: &mut [u8]) -> Result<(), BuiltinError> {
    getrandom(bytes).map_err(|err| BuiltinError::OSError(format!("os.urandom() failed: {err}")))
}

/// Generate a single cryptographically secure random `u64`.
pub(crate) fn secure_random_u64() -> Result<u64, BuiltinError> {
    let mut bytes = [0_u8; 8];
    fill_random_bytes(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn leak_bytes_value(bytes: Vec<u8>) -> Value {
    crate::alloc_managed_value(BytesObject::from_vec(bytes))
}
