//! Combinatorial functions.
//!
//! Includes binary GCD (no division), combinations, permutations, LCM.

use super::super::ModuleError;

// =============================================================================
// GCD and LCM
// =============================================================================

/// Return the greatest common divisor of a and b.
///
/// # Algorithm
/// Binary GCD (Stein's algorithm) - uses only shifts and subtracts.
/// O(log(min(a, b))) operations, no division.
pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    // Handle signs
    a = a.abs();
    b = b.abs();

    // Handle zeros
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }

    // Binary GCD: remove common factors of 2
    let shift = (a | b).trailing_zeros();

    // Remove all factors of 2 from a
    a >>= a.trailing_zeros();

    loop {
        // Remove all factors of 2 from b
        b >>= b.trailing_zeros();

        // Ensure a <= b
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }

        // Reduce b
        b -= a;

        if b == 0 {
            break;
        }
    }

    // Restore common factors of 2
    a << shift
}

/// Return the least common multiple of a and b.
///
/// # Algorithm
/// lcm(a, b) = |a * b| / gcd(a, b)
///
/// # Errors
/// - OverflowError if result overflows i64
pub fn lcm(a: i64, b: i64) -> Result<i64, ModuleError> {
    if a == 0 || b == 0 {
        return Ok(0);
    }

    let g = gcd(a, b);

    // Compute |a| * |b| / gcd in a way that avoids overflow
    let a_abs = a.abs();
    let b_abs = b.abs();

    // a_abs * (b_abs / g) - divide first to prevent overflow
    let result = a_abs / g;
    result
        .checked_mul(b_abs)
        .ok_or_else(|| ModuleError::MathRangeError("lcm() result too large".to_string()))
}

// =============================================================================
// Combinations and Permutations
// =============================================================================

/// Return the number of ways to choose k items from n items.
///
/// comb(n, k) = n! / (k! * (n-k)!)
///
/// # Algorithm
/// Multiplicative formula to avoid computing large factorials.
///
/// # Errors
/// - ValueError if n < 0 or k < 0 or k > n
pub fn comb(n: i64, k: i64) -> Result<i64, ModuleError> {
    if n < 0 || k < 0 {
        return Err(ModuleError::ValueError(
            "n and k must be nonnegative".to_string(),
        ));
    }

    if k > n {
        return Ok(0);
    }

    // Use symmetry: C(n, k) = C(n, n-k)
    let k = if k > n - k { n - k } else { k };

    if k == 0 {
        return Ok(1);
    }

    // Multiplicative formula: C(n, k) = n/1 * (n-1)/2 * ... * (n-k+1)/k
    let mut result: i64 = 1;

    for i in 0..k {
        // Multiply by (n - i), then divide by (i + 1)
        // Division is exact at each step
        result = result
            .checked_mul(n - i)
            .ok_or_else(|| ModuleError::MathRangeError("comb() result too large".to_string()))?;

        result /= i + 1;
    }

    Ok(result)
}

/// Return the number of ways to arrange k items from n items.
///
/// perm(n, k) = n! / (n-k)!
///
/// # Errors
/// - ValueError if n < 0 or k < 0 or k > n
pub fn perm(n: i64, k: i64) -> Result<i64, ModuleError> {
    if n < 0 || k < 0 {
        return Err(ModuleError::ValueError(
            "n and k must be nonnegative".to_string(),
        ));
    }

    if k > n {
        return Ok(0);
    }

    // perm(n, k) = n * (n-1) * ... * (n-k+1)
    let mut result: i64 = 1;

    for i in 0..k {
        result = result
            .checked_mul(n - i)
            .ok_or_else(|| ModuleError::MathRangeError("perm() result too large".to_string()))?;
    }

    Ok(result)
}
