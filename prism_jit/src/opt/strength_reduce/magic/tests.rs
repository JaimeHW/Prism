use super::*;

// =========================================================================
// Unsigned 64-bit Tests
// =========================================================================

#[test]
fn test_unsigned_magic_special_cases() {
    // Division by 0 not allowed
    assert!(UnsignedMagic::compute(0).is_none());

    // Division by 1 - identity
    assert!(UnsignedMagic::compute(1).is_none());
}

#[test]
fn test_unsigned_magic_power_of_two() {
    for shift in 1..63 {
        let d = 1u64 << shift;
        let magic = UnsignedMagic::compute(d).unwrap();
        assert!(magic.is_power_of_two());
        assert_eq!(magic.shift, shift);
        assert!(!magic.add);
    }
}

#[test]
fn test_unsigned_magic_small_divisors() {
    // Test divisors 2-20
    for d in 2u64..=20 {
        let magic = UnsignedMagic::compute(d);
        assert!(magic.is_some(), "Failed for divisor {}", d);

        let m = magic.unwrap();
        // Verify correctness for a sample of values
        for x in [0u64, 1, 10, 100, 1000, u64::MAX / 2, u64::MAX] {
            let expected = x / d;
            let actual = apply_unsigned_magic(x, &m);
            assert_eq!(
                actual, expected,
                "Failed for x={}, d={}: got {}, expected {}",
                x, d, actual, expected
            );
        }
    }
}

#[test]
fn test_unsigned_magic_large_divisors() {
    let test_divisors = [1000u64, 10000, 65537, 1_000_000, u64::MAX / 2];

    for d in test_divisors {
        let magic = UnsignedMagic::compute(d);
        assert!(magic.is_some(), "Failed for divisor {}", d);

        let m = magic.unwrap();
        for x in [0u64, 1, d - 1, d, d + 1, d * 2, u64::MAX] {
            let expected = x / d;
            let actual = apply_unsigned_magic(x, &m);
            assert_eq!(
                actual, expected,
                "Failed for x={}, d={}: got {}, expected {}",
                x, d, actual, expected
            );
        }
    }
}

#[test]
fn test_unsigned_magic_comprehensive() {
    // Test many divisors
    for d in 2u64..=1000 {
        let magic = UnsignedMagic::compute(d).unwrap();

        // Test boundary values
        let test_values = [0u64, 1, d - 1, d, d + 1, d * 2 - 1, d * 2, 1000 * d];

        for x in test_values {
            let expected = x / d;
            let actual = apply_unsigned_magic(x, &magic);
            assert_eq!(actual, expected, "Failed for x={}, d={}", x, d);
        }
    }
}

#[test]
fn test_unsigned_magic_cost() {
    // Power of 2 should be cheapest
    let pow2 = UnsignedMagic::compute(4).unwrap();
    assert_eq!(pow2.cost(), 1);

    // Divisor 3 requires add-shift sequence for 64-bit
    let with_add = UnsignedMagic::compute(3).unwrap();
    assert!(with_add.add);
    assert_eq!(with_add.cost(), 4);

    // Some divisors work without add-shift (e.g., divisors where
    // a power of 2 is close to a multiple of d)
    // Divisor 5 needs add-shift at 64-bit too
    let five = UnsignedMagic::compute(5).unwrap();
    assert_eq!(five.cost(), if five.add { 4 } else { 2 });
}

// =========================================================================
// Signed 64-bit Tests
// =========================================================================

#[test]
fn test_signed_magic_special_cases() {
    assert!(SignedMagic::compute(0).is_none());
    assert!(SignedMagic::compute(1).is_none());
    assert!(SignedMagic::compute(-1).is_none());
}

#[test]
fn test_signed_magic_power_of_two() {
    for shift in 1..62 {
        let d = 1i64 << shift;
        let magic = SignedMagic::compute(d).unwrap();
        assert!(magic.is_power_of_two());
        assert_eq!(magic.shift, shift);
    }
}

#[test]
fn test_signed_magic_positive_divisors() {
    for d in 2i64..=20 {
        let magic = SignedMagic::compute(d);
        assert!(magic.is_some(), "Failed for divisor {}", d);

        let m = magic.unwrap();
        for x in [0i64, 1, -1, 10, -10, 100, -100, i64::MAX / 2, i64::MIN / 2] {
            let expected = x / d;
            let actual = apply_signed_magic(x, &m, d);
            assert_eq!(
                actual, expected,
                "Failed for x={}, d={}: got {}, expected {}",
                x, d, actual, expected
            );
        }
    }
}

#[test]
fn test_signed_magic_negative_divisors() {
    for d in [-2i64, -3, -5, -7, -10, -100] {
        let magic = SignedMagic::compute(d);
        assert!(magic.is_some(), "Failed for divisor {}", d);

        let m = magic.unwrap();
        for x in [0i64, 1, -1, 10, -10, 100, -100] {
            let expected = x / d;
            let actual = apply_signed_magic(x, &m, d);
            assert_eq!(
                actual, expected,
                "Failed for x={}, d={}: got {}, expected {}",
                x, d, actual, expected
            );
        }
    }
}

#[test]
fn test_signed_magic_comprehensive() {
    for d in 2i64..=100 {
        let magic = SignedMagic::compute(d).unwrap();

        let test_values = [
            0i64,
            1,
            -1,
            d - 1,
            d,
            d + 1,
            -(d - 1),
            -d,
            -(d + 1),
            1000 * d,
            -1000 * d,
        ];

        for x in test_values {
            let expected = x / d;
            let actual = apply_signed_magic(x, &magic, d);
            assert_eq!(actual, expected, "Failed for x={}, d={}", x, d);
        }
    }
}

// =========================================================================
// Unsigned 32-bit Tests
// =========================================================================

#[test]
fn test_unsigned32_magic_special_cases() {
    assert!(UnsignedMagic32::compute(0).is_none());
    assert!(UnsignedMagic32::compute(1).is_none());
}

#[test]
fn test_unsigned32_magic_power_of_two() {
    for shift in 1..31 {
        let d = 1u32 << shift;
        let magic = UnsignedMagic32::compute(d).unwrap();
        assert!(magic.is_power_of_two());
        assert_eq!(magic.shift, shift);
    }
}

#[test]
fn test_unsigned32_magic_correctness() {
    for d in 2u32..=100 {
        let magic = UnsignedMagic32::compute(d).unwrap();

        for x in [0u32, 1, d - 1, d, d + 1, 1000 * d, u32::MAX] {
            let expected = x / d;
            let actual = apply_unsigned_magic32(x, &magic);
            assert_eq!(actual, expected, "Failed for x={}, d={}", x, d);
        }
    }
}

// =========================================================================
// Signed 32-bit Tests
// =========================================================================

#[test]
fn test_signed32_magic_special_cases() {
    assert!(SignedMagic32::compute(0).is_none());
    assert!(SignedMagic32::compute(1).is_none());
    assert!(SignedMagic32::compute(-1).is_none());
}

#[test]
fn test_signed32_magic_correctness() {
    for d in 2i32..=100 {
        let magic = SignedMagic32::compute(d).unwrap();

        for x in [0i32, 1, -1, d - 1, d, d + 1, -(d - 1), -d, i32::MAX / 2] {
            let expected = x / d;
            let actual = apply_signed_magic32(x, &magic, d);
            assert_eq!(actual, expected, "Failed for x={}, d={}", x, d);
        }
    }
}

// =========================================================================
// Helper Functions Tests
// =========================================================================

#[test]
fn test_is_power_of_two_u64() {
    assert!(!is_power_of_two_u64(0));
    assert!(is_power_of_two_u64(1));
    assert!(is_power_of_two_u64(2));
    assert!(!is_power_of_two_u64(3));
    assert!(is_power_of_two_u64(4));
    assert!(is_power_of_two_u64(1 << 63));
}

#[test]
fn test_is_power_of_two_i64() {
    assert!(!is_power_of_two_i64(0));
    assert!(is_power_of_two_i64(1));
    assert!(is_power_of_two_i64(2));
    assert!(!is_power_of_two_i64(-1));
    assert!(!is_power_of_two_i64(-2));
}

#[test]
fn test_log2_u64() {
    assert_eq!(log2_u64(1), 0);
    assert_eq!(log2_u64(2), 1);
    assert_eq!(log2_u64(4), 2);
    assert_eq!(log2_u64(1 << 63), 63);
}

#[test]
fn test_power_of_two_mask() {
    assert_eq!(power_of_two_mask(1), 1);
    assert_eq!(power_of_two_mask(2), 3);
    assert_eq!(power_of_two_mask(3), 7);
    assert_eq!(power_of_two_mask(8), 255);
}

// =========================================================================
// Test Helpers
// =========================================================================

/// Apply unsigned magic division (simulates generated code).
fn apply_unsigned_magic(x: u64, magic: &UnsignedMagic) -> u64 {
    if magic.is_power_of_two() {
        return x >> magic.shift;
    }

    // mulhu: multiply and get high 64 bits
    let product = (x as u128) * (magic.multiplier as u128);
    let high = (product >> 64) as u64;

    if magic.add {
        // add-shift sequence: ((x - high) >> 1) + high) >> (shift - 1)
        let t = ((x - high) >> 1) + high;
        t >> magic.shift
    } else {
        high >> magic.shift
    }
}

/// Apply signed magic division (simulates generated code).
fn apply_signed_magic(x: i64, magic: &SignedMagic, _d: i64) -> i64 {
    if magic.is_power_of_two() {
        // Signed power-of-2 division with rounding toward zero
        let shift = magic.shift;
        let sign_bit = x >> 63; // All 1s if negative, all 0s if positive
        let bias = (sign_bit as u64 >> (64 - shift)) as i64;
        let result = (x + bias) >> shift;
        // Negate if divisor was negative
        return if magic.negative_divisor {
            -result
        } else {
            result
        };
    }

    // mulhs: signed multiply high
    let product = (x as i128) * (magic.multiplier as i128);
    let mut q = (product >> 64) as i64;

    // Add x if we're using the add-shift sequence
    if magic.add {
        q = q.wrapping_add(x);
    }

    // Arithmetic shift right
    q >>= magic.shift;

    // Add 1 if result is negative (round toward zero)
    q += (q >> 63) as i64 & 1;

    // Negate if divisor was negative
    if magic.negative_divisor { -q } else { q }
}

/// Apply unsigned 32-bit magic division.
fn apply_unsigned_magic32(x: u32, magic: &UnsignedMagic32) -> u32 {
    if magic.is_power_of_two() {
        return x >> magic.shift;
    }

    let product = (x as u64) * (magic.multiplier as u64);
    let high = (product >> 32) as u32;

    if magic.add {
        let t = ((x - high) >> 1) + high;
        t >> magic.shift
    } else {
        high >> magic.shift
    }
}

/// Apply signed 32-bit magic division.
fn apply_signed_magic32(x: i32, magic: &SignedMagic32, _d: i32) -> i32 {
    if magic.is_power_of_two() {
        let shift = magic.shift;
        let sign_bit = x >> 31;
        let bias = (sign_bit as u32 >> (32 - shift)) as i32;
        let result = (x + bias) >> shift;
        return if magic.negative_divisor {
            -result
        } else {
            result
        };
    }

    let product = (x as i64) * (magic.multiplier as i64);
    let mut q = (product >> 32) as i32;

    if magic.add {
        q = q.wrapping_add(x);
    }

    q >>= magic.shift;
    q += (q >> 31) & 1;

    // Negate if divisor was negative
    if magic.negative_divisor { -q } else { q }
}
