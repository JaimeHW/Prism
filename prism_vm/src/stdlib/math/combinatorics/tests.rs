use super::*;

// =========================================================================
// gcd() Tests
// =========================================================================

#[test]
fn test_gcd_basic() {
    assert_eq!(gcd(12, 8), 4);
    assert_eq!(gcd(48, 18), 6);
    assert_eq!(gcd(54, 24), 6);
}

#[test]
fn test_gcd_coprime() {
    assert_eq!(gcd(13, 17), 1);
    assert_eq!(gcd(5, 7), 1);
}

#[test]
fn test_gcd_same() {
    assert_eq!(gcd(10, 10), 10);
}

#[test]
fn test_gcd_one() {
    assert_eq!(gcd(1, 100), 1);
    assert_eq!(gcd(100, 1), 1);
}

#[test]
fn test_gcd_zero() {
    assert_eq!(gcd(0, 5), 5);
    assert_eq!(gcd(5, 0), 5);
    assert_eq!(gcd(0, 0), 0);
}

#[test]
fn test_gcd_negative() {
    assert_eq!(gcd(-12, 8), 4);
    assert_eq!(gcd(12, -8), 4);
    assert_eq!(gcd(-12, -8), 4);
}

#[test]
fn test_gcd_large() {
    assert_eq!(gcd(1000000007, 1000000009), 1); // Two primes
    assert_eq!(gcd(1000000000, 1000000), 1000000);
}

#[test]
fn test_gcd_powers_of_two() {
    assert_eq!(gcd(1024, 256), 256);
    assert_eq!(gcd(2048, 512), 512);
}

// =========================================================================
// lcm() Tests
// =========================================================================

#[test]
fn test_lcm_basic() {
    assert_eq!(lcm(4, 6).unwrap(), 12);
    assert_eq!(lcm(3, 5).unwrap(), 15);
    assert_eq!(lcm(12, 18).unwrap(), 36);
}

#[test]
fn test_lcm_coprime() {
    assert_eq!(lcm(7, 11).unwrap(), 77);
}

#[test]
fn test_lcm_same() {
    assert_eq!(lcm(10, 10).unwrap(), 10);
}

#[test]
fn test_lcm_one() {
    assert_eq!(lcm(1, 100).unwrap(), 100);
    assert_eq!(lcm(100, 1).unwrap(), 100);
}

#[test]
fn test_lcm_zero() {
    assert_eq!(lcm(0, 5).unwrap(), 0);
    assert_eq!(lcm(5, 0).unwrap(), 0);
    assert_eq!(lcm(0, 0).unwrap(), 0);
}

#[test]
fn test_lcm_negative() {
    assert_eq!(lcm(-4, 6).unwrap(), 12);
    assert_eq!(lcm(4, -6).unwrap(), 12);
}

// =========================================================================
// comb() Tests
// =========================================================================

#[test]
fn test_comb_basic() {
    assert_eq!(comb(5, 2).unwrap(), 10);
    assert_eq!(comb(6, 3).unwrap(), 20);
    assert_eq!(comb(10, 5).unwrap(), 252);
}

#[test]
fn test_comb_edges() {
    assert_eq!(comb(5, 0).unwrap(), 1);
    assert_eq!(comb(5, 5).unwrap(), 1);
    assert_eq!(comb(0, 0).unwrap(), 1);
}

#[test]
fn test_comb_k_greater_than_n() {
    assert_eq!(comb(5, 6).unwrap(), 0);
    assert_eq!(comb(3, 10).unwrap(), 0);
}

#[test]
fn test_comb_symmetry() {
    // C(n, k) = C(n, n-k)
    for n in 0..=10 {
        for k in 0..=n {
            assert_eq!(comb(n, k).unwrap(), comb(n, n - k).unwrap());
        }
    }
}

#[test]
fn test_comb_pascals_triangle() {
    // C(n, k) = C(n-1, k-1) + C(n-1, k)
    for n in 2..=10 {
        for k in 1..n {
            let lhs = comb(n, k).unwrap();
            let rhs = comb(n - 1, k - 1).unwrap() + comb(n - 1, k).unwrap();
            assert_eq!(lhs, rhs);
        }
    }
}

#[test]
fn test_comb_negative() {
    assert!(comb(-1, 2).is_err());
    assert!(comb(5, -1).is_err());
}

#[test]
fn test_comb_large() {
    // C(20, 10) = 184756
    assert_eq!(comb(20, 10).unwrap(), 184756);
    // C(30, 15) = 155117520
    assert_eq!(comb(30, 15).unwrap(), 155117520);
}

// =========================================================================
// perm() Tests
// =========================================================================

#[test]
fn test_perm_basic() {
    assert_eq!(perm(5, 2).unwrap(), 20);
    assert_eq!(perm(6, 3).unwrap(), 120);
    assert_eq!(perm(10, 3).unwrap(), 720);
}

#[test]
fn test_perm_edges() {
    assert_eq!(perm(5, 0).unwrap(), 1);
    assert_eq!(perm(5, 5).unwrap(), 120); // 5!
    assert_eq!(perm(0, 0).unwrap(), 1);
}

#[test]
fn test_perm_k_greater_than_n() {
    assert_eq!(perm(5, 6).unwrap(), 0);
    assert_eq!(perm(3, 10).unwrap(), 0);
}

#[test]
fn test_perm_full() {
    // perm(n, n) = n!
    assert_eq!(perm(5, 5).unwrap(), 120);
    assert_eq!(perm(6, 6).unwrap(), 720);
    assert_eq!(perm(7, 7).unwrap(), 5040);
}

#[test]
fn test_perm_negative() {
    assert!(perm(-1, 2).is_err());
    assert!(perm(5, -1).is_err());
}

#[test]
fn test_perm_large() {
    // P(20, 10) = 670442572800
    assert_eq!(perm(20, 10).unwrap(), 670442572800);
}

// =========================================================================
// Identity Tests
// =========================================================================

#[test]
fn test_gcd_lcm_identity() {
    // gcd(a, b) * lcm(a, b) = |a * b|
    for a in [1, 2, 3, 5, 7, 12, 18, 100] {
        for b in [1, 2, 3, 5, 7, 12, 18, 100] {
            let g = gcd(a, b);
            let l = lcm(a, b).unwrap();
            assert_eq!(g * l, a * b);
        }
    }
}

#[test]
fn test_perm_comb_relationship() {
    // perm(n, k) = comb(n, k) * k!
    for n in 0..=10 {
        for k in 0..=n {
            let p = perm(n, k).unwrap();
            let c = comb(n, k).unwrap();
            let k_factorial = super::super::special::factorial(k).unwrap();
            assert_eq!(p, c * k_factorial);
        }
    }
}
