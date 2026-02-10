//! SIMD instruction encoding for AVX/AVX2/AVX-512.
//!
//! This module provides comprehensive SIMD instruction encoding including:
//! - VEX prefix encoding (AVX, AVX2)
//! - EVEX prefix encoding (AVX-512)
//! - YMM/ZMM register definitions
//! - Packed arithmetic, logical, memory, and shuffle operations
//!
//! # Encoding Reference
//!
//! VEX prefix format (2-byte and 3-byte):
//! ```text
//! 2-byte: C5 [R vvvv L pp]
//! 3-byte: C4 [R X B mmmmm] [W vvvv L pp]
//! ```
//!
//! EVEX prefix format (4-byte):
//! ```text
//! 62 [R X B R' 0 0 m m] [W vvvv 1 p p] [z L' L b V' a a a]
//! ```

use super::encoder::{EncodedInst, Mod, modrm, sib};
use super::registers::{MemOperand, Scale, Xmm};

// =============================================================================
// YMM Registers (256-bit)
// =============================================================================

/// x64 YMM register for 256-bit SIMD operations (AVX/AVX2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Ymm {
    Ymm0 = 0,
    Ymm1 = 1,
    Ymm2 = 2,
    Ymm3 = 3,
    Ymm4 = 4,
    Ymm5 = 5,
    Ymm6 = 6,
    Ymm7 = 7,
    Ymm8 = 8,
    Ymm9 = 9,
    Ymm10 = 10,
    Ymm11 = 11,
    Ymm12 = 12,
    Ymm13 = 13,
    Ymm14 = 14,
    Ymm15 = 15,
}

impl Ymm {
    /// All 16 YMM registers in encoding order.
    pub const ALL: [Ymm; 16] = [
        Ymm::Ymm0,
        Ymm::Ymm1,
        Ymm::Ymm2,
        Ymm::Ymm3,
        Ymm::Ymm4,
        Ymm::Ymm5,
        Ymm::Ymm6,
        Ymm::Ymm7,
        Ymm::Ymm8,
        Ymm::Ymm9,
        Ymm::Ymm10,
        Ymm::Ymm11,
        Ymm::Ymm12,
        Ymm::Ymm13,
        Ymm::Ymm14,
        Ymm::Ymm15,
    ];

    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    #[inline(always)]
    pub const fn low_bits(self) -> u8 {
        self.encoding() & 0x7
    }

    #[inline(always)]
    pub const fn high_bit(self) -> bool {
        self.encoding() >= 8
    }

    /// Convert from encoding value if valid.
    #[inline]
    pub const fn from_encoding(enc: u8) -> Option<Ymm> {
        match enc {
            0 => Some(Ymm::Ymm0),
            1 => Some(Ymm::Ymm1),
            2 => Some(Ymm::Ymm2),
            3 => Some(Ymm::Ymm3),
            4 => Some(Ymm::Ymm4),
            5 => Some(Ymm::Ymm5),
            6 => Some(Ymm::Ymm6),
            7 => Some(Ymm::Ymm7),
            8 => Some(Ymm::Ymm8),
            9 => Some(Ymm::Ymm9),
            10 => Some(Ymm::Ymm10),
            11 => Some(Ymm::Ymm11),
            12 => Some(Ymm::Ymm12),
            13 => Some(Ymm::Ymm13),
            14 => Some(Ymm::Ymm14),
            15 => Some(Ymm::Ymm15),
            _ => None,
        }
    }

    #[inline(always)]
    pub const fn from_xmm(xmm: Xmm) -> Self {
        // Safe because XMM and YMM share encoding
        match xmm.encoding() {
            0 => Ymm::Ymm0,
            1 => Ymm::Ymm1,
            2 => Ymm::Ymm2,
            3 => Ymm::Ymm3,
            4 => Ymm::Ymm4,
            5 => Ymm::Ymm5,
            6 => Ymm::Ymm6,
            7 => Ymm::Ymm7,
            8 => Ymm::Ymm8,
            9 => Ymm::Ymm9,
            10 => Ymm::Ymm10,
            11 => Ymm::Ymm11,
            12 => Ymm::Ymm12,
            13 => Ymm::Ymm13,
            14 => Ymm::Ymm14,
            _ => Ymm::Ymm15,
        }
    }

    #[inline(always)]
    pub const fn to_xmm(self) -> Xmm {
        match self.encoding() {
            0 => Xmm::Xmm0,
            1 => Xmm::Xmm1,
            2 => Xmm::Xmm2,
            3 => Xmm::Xmm3,
            4 => Xmm::Xmm4,
            5 => Xmm::Xmm5,
            6 => Xmm::Xmm6,
            7 => Xmm::Xmm7,
            8 => Xmm::Xmm8,
            9 => Xmm::Xmm9,
            10 => Xmm::Xmm10,
            11 => Xmm::Xmm11,
            12 => Xmm::Xmm12,
            13 => Xmm::Xmm13,
            14 => Xmm::Xmm14,
            _ => Xmm::Xmm15,
        }
    }
}

impl std::fmt::Display for Ymm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ymm{}", self.encoding())
    }
}

// =============================================================================
// ZMM Registers (512-bit) - AVX-512
// =============================================================================

/// x64 ZMM register for 512-bit SIMD operations (AVX-512).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Zmm {
    Zmm0 = 0,
    Zmm1 = 1,
    Zmm2 = 2,
    Zmm3 = 3,
    Zmm4 = 4,
    Zmm5 = 5,
    Zmm6 = 6,
    Zmm7 = 7,
    Zmm8 = 8,
    Zmm9 = 9,
    Zmm10 = 10,
    Zmm11 = 11,
    Zmm12 = 12,
    Zmm13 = 13,
    Zmm14 = 14,
    Zmm15 = 15,
    Zmm16 = 16,
    Zmm17 = 17,
    Zmm18 = 18,
    Zmm19 = 19,
    Zmm20 = 20,
    Zmm21 = 21,
    Zmm22 = 22,
    Zmm23 = 23,
    Zmm24 = 24,
    Zmm25 = 25,
    Zmm26 = 26,
    Zmm27 = 27,
    Zmm28 = 28,
    Zmm29 = 29,
    Zmm30 = 30,
    Zmm31 = 31,
}

impl Zmm {
    /// All 32 ZMM registers in encoding order.
    pub const ALL: [Zmm; 32] = [
        Zmm::Zmm0,
        Zmm::Zmm1,
        Zmm::Zmm2,
        Zmm::Zmm3,
        Zmm::Zmm4,
        Zmm::Zmm5,
        Zmm::Zmm6,
        Zmm::Zmm7,
        Zmm::Zmm8,
        Zmm::Zmm9,
        Zmm::Zmm10,
        Zmm::Zmm11,
        Zmm::Zmm12,
        Zmm::Zmm13,
        Zmm::Zmm14,
        Zmm::Zmm15,
        Zmm::Zmm16,
        Zmm::Zmm17,
        Zmm::Zmm18,
        Zmm::Zmm19,
        Zmm::Zmm20,
        Zmm::Zmm21,
        Zmm::Zmm22,
        Zmm::Zmm23,
        Zmm::Zmm24,
        Zmm::Zmm25,
        Zmm::Zmm26,
        Zmm::Zmm27,
        Zmm::Zmm28,
        Zmm::Zmm29,
        Zmm::Zmm30,
        Zmm::Zmm31,
    ];

    /// First 16 ZMM registers (available on all AVX-512 CPUs).
    pub const ALL16: [Zmm; 16] = [
        Zmm::Zmm0,
        Zmm::Zmm1,
        Zmm::Zmm2,
        Zmm::Zmm3,
        Zmm::Zmm4,
        Zmm::Zmm5,
        Zmm::Zmm6,
        Zmm::Zmm7,
        Zmm::Zmm8,
        Zmm::Zmm9,
        Zmm::Zmm10,
        Zmm::Zmm11,
        Zmm::Zmm12,
        Zmm::Zmm13,
        Zmm::Zmm14,
        Zmm::Zmm15,
    ];

    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    #[inline(always)]
    pub const fn low_bits(self) -> u8 {
        self.encoding() & 0x7
    }

    #[inline(always)]
    pub const fn high_bit(self) -> bool {
        (self.encoding() & 0x8) != 0
    }

    #[inline(always)]
    pub const fn ext_bit(self) -> bool {
        self.encoding() >= 16
    }

    /// Convert from encoding value if valid.
    #[inline]
    pub const fn from_encoding(enc: u8) -> Option<Zmm> {
        match enc {
            0 => Some(Zmm::Zmm0),
            1 => Some(Zmm::Zmm1),
            2 => Some(Zmm::Zmm2),
            3 => Some(Zmm::Zmm3),
            4 => Some(Zmm::Zmm4),
            5 => Some(Zmm::Zmm5),
            6 => Some(Zmm::Zmm6),
            7 => Some(Zmm::Zmm7),
            8 => Some(Zmm::Zmm8),
            9 => Some(Zmm::Zmm9),
            10 => Some(Zmm::Zmm10),
            11 => Some(Zmm::Zmm11),
            12 => Some(Zmm::Zmm12),
            13 => Some(Zmm::Zmm13),
            14 => Some(Zmm::Zmm14),
            15 => Some(Zmm::Zmm15),
            16 => Some(Zmm::Zmm16),
            17 => Some(Zmm::Zmm17),
            18 => Some(Zmm::Zmm18),
            19 => Some(Zmm::Zmm19),
            20 => Some(Zmm::Zmm20),
            21 => Some(Zmm::Zmm21),
            22 => Some(Zmm::Zmm22),
            23 => Some(Zmm::Zmm23),
            24 => Some(Zmm::Zmm24),
            25 => Some(Zmm::Zmm25),
            26 => Some(Zmm::Zmm26),
            27 => Some(Zmm::Zmm27),
            28 => Some(Zmm::Zmm28),
            29 => Some(Zmm::Zmm29),
            30 => Some(Zmm::Zmm30),
            31 => Some(Zmm::Zmm31),
            _ => None,
        }
    }

    /// Convert to the corresponding YMM register (first 16 only).
    #[inline(always)]
    pub const fn to_ymm(self) -> Option<Ymm> {
        if self.encoding() < 16 {
            Ymm::from_encoding(self.encoding())
        } else {
            None
        }
    }

    /// Convert from YMM register to ZMM.
    #[inline(always)]
    pub const fn from_ymm(ymm: Ymm) -> Self {
        match ymm.encoding() {
            0 => Zmm::Zmm0,
            1 => Zmm::Zmm1,
            2 => Zmm::Zmm2,
            3 => Zmm::Zmm3,
            4 => Zmm::Zmm4,
            5 => Zmm::Zmm5,
            6 => Zmm::Zmm6,
            7 => Zmm::Zmm7,
            8 => Zmm::Zmm8,
            9 => Zmm::Zmm9,
            10 => Zmm::Zmm10,
            11 => Zmm::Zmm11,
            12 => Zmm::Zmm12,
            13 => Zmm::Zmm13,
            14 => Zmm::Zmm14,
            _ => Zmm::Zmm15,
        }
    }
}

impl std::fmt::Display for Zmm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zmm{}", self.encoding())
    }
}

// =============================================================================
// YMM Register Set (256-bit AVX)
// =============================================================================

/// A set of YMM registers using a 16-bit bitfield for O(1) operations.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct YmmSet(u16);

impl YmmSet {
    /// Empty register set.
    pub const EMPTY: YmmSet = YmmSet(0);

    /// All 16 registers.
    pub const ALL: YmmSet = YmmSet(0xFFFF);

    /// Create a set containing a single register.
    #[inline(always)]
    pub const fn singleton(reg: Ymm) -> Self {
        YmmSet(1 << reg.encoding())
    }

    /// Create from a raw bitmask.
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        YmmSet(bits)
    }

    /// Get the raw bitmask.
    #[inline(always)]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Check if the set contains a register.
    #[inline(always)]
    pub const fn contains(self, reg: Ymm) -> bool {
        (self.0 & (1 << reg.encoding())) != 0
    }

    /// Check if the set is empty.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Add a register to the set.
    #[inline(always)]
    pub const fn insert(self, reg: Ymm) -> Self {
        YmmSet(self.0 | (1 << reg.encoding()))
    }

    /// Remove a register from the set.
    #[inline(always)]
    pub const fn remove(self, reg: Ymm) -> Self {
        YmmSet(self.0 & !(1 << reg.encoding()))
    }

    /// Union of two sets.
    #[inline(always)]
    pub const fn union(self, other: YmmSet) -> Self {
        YmmSet(self.0 | other.0)
    }

    /// Intersection of two sets.
    #[inline(always)]
    pub const fn intersection(self, other: YmmSet) -> Self {
        YmmSet(self.0 & other.0)
    }

    /// Difference (self - other).
    #[inline(always)]
    pub const fn difference(self, other: YmmSet) -> Self {
        YmmSet(self.0 & !other.0)
    }

    /// Count the number of registers in the set.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Get the first register (lowest encoding) in the set, if any.
    #[inline]
    pub const fn first(self) -> Option<Ymm> {
        if self.0 == 0 {
            None
        } else {
            Ymm::from_encoding(self.0.trailing_zeros() as u8)
        }
    }

    /// Iterate over registers in the set (ascending order).
    pub fn iter(self) -> impl Iterator<Item = Ymm> {
        (0..16).filter_map(move |i| {
            if (self.0 & (1 << i)) != 0 {
                Ymm::from_encoding(i)
            } else {
                None
            }
        })
    }
}

impl std::fmt::Debug for YmmSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "YmmSet{{")?;
        let mut first = true;
        for reg in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}", reg)?;
            first = false;
        }
        write!(f, "}}")
    }
}

// =============================================================================
// ZMM Register Set (512-bit AVX-512)
// =============================================================================

/// A set of ZMM registers using a 32-bit bitfield for O(1) operations.
/// Supports all 32 ZMM registers (ZMM0-ZMM31) available on AVX-512.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct ZmmSet(u32);

impl ZmmSet {
    /// Empty register set.
    pub const EMPTY: ZmmSet = ZmmSet(0);

    /// All 32 registers.
    pub const ALL: ZmmSet = ZmmSet(0xFFFF_FFFF);

    /// First 16 registers (available without AVX-512VL on some CPUs).
    pub const ALL16: ZmmSet = ZmmSet(0x0000_FFFF);

    /// Upper 16 registers (ZMM16-ZMM31).
    pub const UPPER16: ZmmSet = ZmmSet(0xFFFF_0000);

    /// Create a set containing a single register.
    #[inline(always)]
    pub const fn singleton(reg: Zmm) -> Self {
        ZmmSet(1 << reg.encoding())
    }

    /// Create from a raw bitmask.
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        ZmmSet(bits)
    }

    /// Get the raw bitmask.
    #[inline(always)]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Check if the set contains a register.
    #[inline(always)]
    pub const fn contains(self, reg: Zmm) -> bool {
        (self.0 & (1 << reg.encoding())) != 0
    }

    /// Check if the set is empty.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Add a register to the set.
    #[inline(always)]
    pub const fn insert(self, reg: Zmm) -> Self {
        ZmmSet(self.0 | (1 << reg.encoding()))
    }

    /// Remove a register from the set.
    #[inline(always)]
    pub const fn remove(self, reg: Zmm) -> Self {
        ZmmSet(self.0 & !(1 << reg.encoding()))
    }

    /// Union of two sets.
    #[inline(always)]
    pub const fn union(self, other: ZmmSet) -> Self {
        ZmmSet(self.0 | other.0)
    }

    /// Intersection of two sets.
    #[inline(always)]
    pub const fn intersection(self, other: ZmmSet) -> Self {
        ZmmSet(self.0 & other.0)
    }

    /// Difference (self - other).
    #[inline(always)]
    pub const fn difference(self, other: ZmmSet) -> Self {
        ZmmSet(self.0 & !other.0)
    }

    /// Count the number of registers in the set.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Get the first register (lowest encoding) in the set, if any.
    #[inline]
    pub const fn first(self) -> Option<Zmm> {
        if self.0 == 0 {
            None
        } else {
            Zmm::from_encoding(self.0.trailing_zeros() as u8)
        }
    }

    /// Iterate over registers in the set (ascending order).
    pub fn iter(self) -> impl Iterator<Item = Zmm> {
        (0..32).filter_map(move |i| {
            if (self.0 & (1 << i)) != 0 {
                Zmm::from_encoding(i)
            } else {
                None
            }
        })
    }
}

impl std::fmt::Debug for ZmmSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ZmmSet{{")?;
        let mut first = true;
        for reg in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}", reg)?;
            first = false;
        }
        write!(f, "}}")
    }
}

// =============================================================================
// Mask Registers (AVX-512)
// =============================================================================
/// AVX-512 mask register (k0-k7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KReg {
    K0 = 0,
    K1 = 1,
    K2 = 2,
    K3 = 3,
    K4 = 4,
    K5 = 5,
    K6 = 6,
    K7 = 7,
}

impl KReg {
    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }
}

// =============================================================================
// VEX Prefix
// =============================================================================

/// VEX prefix for AVX/AVX2 instructions.
#[derive(Debug, Clone, Copy)]
pub struct Vex {
    /// Use 256-bit vector length (YMM).
    pub l: bool,
    /// Operand prefix: 0=none, 1=66, 2=F3, 3=F2.
    pub pp: u8,
    /// Opcode map: 1=0F, 2=0F38, 3=0F3A.
    pub mm: u8,
    /// REX.W equivalent.
    pub w: bool,
    /// Inverted REX.R.
    pub r: bool,
    /// Inverted REX.X.
    pub x: bool,
    /// Inverted REX.B.
    pub b: bool,
    /// Source register (inverted, 4-bit).
    pub vvvv: u8,
}

impl Vex {
    /// Check if 2-byte VEX is sufficient.
    #[inline]
    pub const fn can_use_2byte(&self) -> bool {
        !self.w && self.mm == 1 && self.x && self.b
    }

    /// Encode as 2-byte VEX prefix.
    #[inline]
    pub fn encode_2byte(&self) -> [u8; 2] {
        debug_assert!(self.can_use_2byte());
        let byte1 = (!self.r as u8) << 7 | (self.vvvv ^ 0xF) << 3 | (self.l as u8) << 2 | self.pp;
        [0xC5, byte1]
    }

    /// Encode as 3-byte VEX prefix.
    #[inline]
    pub fn encode_3byte(&self) -> [u8; 3] {
        let byte1 = (!self.r as u8) << 7 | (!self.x as u8) << 6 | (!self.b as u8) << 5 | self.mm;
        let byte2 = (self.w as u8) << 7 | (self.vvvv ^ 0xF) << 3 | (self.l as u8) << 2 | self.pp;
        [0xC4, byte1, byte2]
    }
}

// =============================================================================
// VEX Encoded Instructions - Packed Double (PD)
// =============================================================================

/// VADDPD ymm, ymm, ymm - Add packed doubles.
#[inline]
pub fn encode_vaddpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x58, dst, src1, src2, true, 1) // pp=1 (66)
}

/// VSUBPD ymm, ymm, ymm - Subtract packed doubles.
#[inline]
pub fn encode_vsubpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x5C, dst, src1, src2, true, 1)
}

/// VMULPD ymm, ymm, ymm - Multiply packed doubles.
#[inline]
pub fn encode_vmulpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x59, dst, src1, src2, true, 1)
}

/// VDIVPD ymm, ymm, ymm - Divide packed doubles.
#[inline]
pub fn encode_vdivpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x5E, dst, src1, src2, true, 1)
}

/// VSQRTPD ymm, ymm - Square root packed doubles.
#[inline]
pub fn encode_vsqrtpd_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    encode_vex_rr(0x51, dst, src, true, 1)
}

/// VMAXPD ymm, ymm, ymm - Maximum packed doubles.
#[inline]
pub fn encode_vmaxpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x5F, dst, src1, src2, true, 1)
}

/// VMINPD ymm, ymm, ymm - Minimum packed doubles.
#[inline]
pub fn encode_vminpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x5D, dst, src1, src2, true, 1)
}

// =============================================================================
// VEX Encoded Instructions - Packed Single (PS)
// =============================================================================

/// VADDPS ymm, ymm, ymm - Add packed singles.
#[inline]
pub fn encode_vaddps_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x58, dst, src1, src2, true, 0) // pp=0 (none)
}

/// VSUBPS ymm, ymm, ymm - Subtract packed singles.
#[inline]
pub fn encode_vsubps_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x5C, dst, src1, src2, true, 0)
}

/// VMULPS ymm, ymm, ymm - Multiply packed singles.
#[inline]
pub fn encode_vmulps_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x59, dst, src1, src2, true, 0)
}

/// VDIVPS ymm, ymm, ymm - Divide packed singles.
#[inline]
pub fn encode_vdivps_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x5E, dst, src1, src2, true, 0)
}

// =============================================================================
// VEX Encoded Instructions - Integer (AVX2)
// =============================================================================

/// VPADDD ymm, ymm, ymm - Add packed 32-bit integers.
#[inline]
pub fn encode_vpaddd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr_0f38(0xFE, dst, src1, src2, true, 1)
}

/// VPSUBD ymm, ymm, ymm - Subtract packed 32-bit integers.
#[inline]
pub fn encode_vpsubd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr_0f38(0xFA, dst, src1, src2, true, 1)
}

/// VPADDQ ymm, ymm, ymm - Add packed 64-bit integers.
#[inline]
pub fn encode_vpaddq_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr_0f38(0xD4, dst, src1, src2, true, 1)
}

/// VPSUBQ ymm, ymm, ymm - Subtract packed 64-bit integers.
#[inline]
pub fn encode_vpsubq_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr_0f38(0xFB, dst, src1, src2, true, 1)
}

/// VPMULLD ymm, ymm, ymm - Multiply packed 32-bit integers (low 32 bits).
#[inline]
pub fn encode_vpmulld_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr_0f38(0x40, dst, src1, src2, true, 1)
}

// =============================================================================
// VEX Encoded Instructions - Logical
// =============================================================================

/// VANDPD ymm, ymm, ymm - Bitwise AND packed doubles.
#[inline]
pub fn encode_vandpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x54, dst, src1, src2, true, 1)
}

/// VORPD ymm, ymm, ymm - Bitwise OR packed doubles.
#[inline]
pub fn encode_vorpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x56, dst, src1, src2, true, 1)
}

/// VXORPD ymm, ymm, ymm - Bitwise XOR packed doubles.
#[inline]
pub fn encode_vxorpd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x57, dst, src1, src2, true, 1)
}

/// VPAND ymm, ymm, ymm - Bitwise AND packed integers.
#[inline]
pub fn encode_vpand_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0xDB, dst, src1, src2, true, 1)
}

/// VPOR ymm, ymm, ymm - Bitwise OR packed integers.
#[inline]
pub fn encode_vpor_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0xEB, dst, src1, src2, true, 1)
}

/// VPXOR ymm, ymm, ymm - Bitwise XOR packed integers.
#[inline]
pub fn encode_vpxor_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0xEF, dst, src1, src2, true, 1)
}

// =============================================================================
// VEX Encoded Instructions - YMM Register-to-Register Moves
// =============================================================================

/// VMOVAPD ymm, ymm - Move aligned packed double-precision (register-to-register).
///
/// This is the preferred instruction for YMM register-to-register moves
/// when alignment is not a concern (which it never is for register moves).
/// Uses opcode 0F 28 with VEX.256 prefix.
#[inline]
pub fn encode_vmovapd_ymm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    // VMOVAPD reg, reg uses opcode 0x28 (same as memory load)
    encode_vex_rr(0x28, dst, src, true, 1) // L=1 (256-bit), pp=1 (66)
}

/// VMOVAPS ymm, ymm - Move aligned packed single-precision (register-to-register).
///
/// Uses opcode 0F 28 with VEX.256 prefix and no operand prefix (pp=0).
#[inline]
pub fn encode_vmovaps_ymm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    encode_vex_rr(0x28, dst, src, true, 0) // L=1 (256-bit), pp=0 (none)
}

/// VMOVUPD ymm, ymm - Move unaligned packed double-precision (register-to-register).
///
/// For register operands, aligned vs unaligned makes no difference,
/// but this provides consistency with memory loading patterns.
#[inline]
pub fn encode_vmovupd_ymm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    encode_vex_rr(0x10, dst, src, true, 1) // L=1 (256-bit), pp=1 (66)
}

/// VMOVUPS ymm, ymm - Move unaligned packed single-precision (register-to-register).
#[inline]
pub fn encode_vmovups_ymm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    encode_vex_rr(0x10, dst, src, true, 0) // L=1 (256-bit), pp=0 (none)
}

/// VMOVDQA ymm, ymm - Move aligned double quadword (256-bit integer data).
///
/// Preferred for moving integer vector data in YMM registers.
/// Uses opcode 0F 6F with VEX.256 and 66 prefix.
#[inline]
pub fn encode_vmovdqa_ymm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    encode_vex_rr(0x6F, dst, src, true, 1) // L=1 (256-bit), pp=1 (66)
}

/// VMOVDQU ymm, ymm - Move unaligned double quadword (256-bit integer data).
#[inline]
pub fn encode_vmovdqu_ymm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    encode_vex_rr(0x6F, dst, src, true, 2) // L=1 (256-bit), pp=2 (F3)
}

// =============================================================================
// VEX Encoded Instructions - XMM Register-to-Register Moves (128-bit)
// =============================================================================

/// VMOVAPD xmm, xmm - Move aligned packed double-precision (128-bit register-to-register).
#[inline]
pub fn encode_vmovapd_xmm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    // Note: Uses Ymm type but L=0 makes it XMM operation
    encode_vex_rr(0x28, dst, src, false, 1) // L=0 (128-bit), pp=1 (66)
}

/// VMOVAPS xmm, xmm - Move aligned packed single-precision (128-bit register-to-register).
#[inline]
pub fn encode_vmovaps_xmm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    encode_vex_rr(0x28, dst, src, false, 0) // L=0 (128-bit), pp=0 (none)
}

/// VMOVDQA xmm, xmm - Move aligned double quadword (128-bit integer).
#[inline]
pub fn encode_vmovdqa_xmm_rr(dst: Ymm, src: Ymm) -> EncodedInst {
    encode_vex_rr(0x6F, dst, src, false, 1) // L=0 (128-bit), pp=1 (66)
}

// =============================================================================
// VEX Encoded Instructions - Memory
// =============================================================================

/// VMOVAPD ymm, [mem] - Move aligned packed doubles.
#[inline]
pub fn encode_vmovapd_rm(dst: Ymm, mem: &MemOperand) -> EncodedInst {
    encode_vex_rm(0x28, dst, mem, true, 1)
}

/// VMOVAPD [mem], ymm - Store aligned packed doubles.
#[inline]
pub fn encode_vmovapd_mr(mem: &MemOperand, src: Ymm) -> EncodedInst {
    encode_vex_mr(0x29, mem, src, true, 1)
}

/// VMOVUPD ymm, [mem] - Move unaligned packed doubles.
#[inline]
pub fn encode_vmovupd_rm(dst: Ymm, mem: &MemOperand) -> EncodedInst {
    encode_vex_rm(0x10, dst, mem, true, 1)
}

/// VMOVUPD [mem], ymm - Store unaligned packed doubles.
#[inline]
pub fn encode_vmovupd_mr(mem: &MemOperand, src: Ymm) -> EncodedInst {
    encode_vex_mr(0x11, mem, src, true, 1)
}

/// VMOVAPS ymm, [mem] - Move aligned packed singles from memory.
#[inline]
pub fn encode_vmovaps_rm(dst: Ymm, mem: &MemOperand) -> EncodedInst {
    encode_vex_rm(0x28, dst, mem, true, 0) // pp=0 for PS
}

/// VMOVAPS [mem], ymm - Store aligned packed singles to memory.
#[inline]
pub fn encode_vmovaps_mr(mem: &MemOperand, src: Ymm) -> EncodedInst {
    encode_vex_mr(0x29, mem, src, true, 0) // pp=0 for PS
}

/// VMOVUPS ymm, [mem] - Move unaligned packed singles from memory.
#[inline]
pub fn encode_vmovups_rm(dst: Ymm, mem: &MemOperand) -> EncodedInst {
    encode_vex_rm(0x10, dst, mem, true, 0) // pp=0 for PS
}

/// VMOVUPS [mem], ymm - Store unaligned packed singles to memory.
#[inline]
pub fn encode_vmovups_mr(mem: &MemOperand, src: Ymm) -> EncodedInst {
    encode_vex_mr(0x11, mem, src, true, 0) // pp=0 for PS
}

/// VMOVDQA ymm, [mem] - Move aligned double quadword from memory.
#[inline]
pub fn encode_vmovdqa_rm(dst: Ymm, mem: &MemOperand) -> EncodedInst {
    encode_vex_rm(0x6F, dst, mem, true, 1) // pp=1 (66)
}

/// VMOVDQA [mem], ymm - Store aligned double quadword to memory.
#[inline]
pub fn encode_vmovdqa_mr(mem: &MemOperand, src: Ymm) -> EncodedInst {
    encode_vex_mr(0x7F, mem, src, true, 1) // pp=1 (66), opcode 7F for store
}

/// VMOVDQU ymm, [mem] - Move unaligned double quadword from memory.
#[inline]
pub fn encode_vmovdqu_rm(dst: Ymm, mem: &MemOperand) -> EncodedInst {
    encode_vex_rm(0x6F, dst, mem, true, 2) // pp=2 (F3)
}

/// VMOVDQU [mem], ymm - Store unaligned double quadword to memory.
#[inline]
pub fn encode_vmovdqu_mr(mem: &MemOperand, src: Ymm) -> EncodedInst {
    encode_vex_mr(0x7F, mem, src, true, 2) // pp=2 (F3), opcode 7F for store
}

/// VBROADCASTSD ymm, [mem] - Broadcast scalar double to all lanes.
#[inline]
pub fn encode_vbroadcastsd_rm(dst: Ymm, mem: &MemOperand) -> EncodedInst {
    encode_vex_rm_0f38(0x19, dst, mem, true, 1)
}

// =============================================================================
// VEX Encoded Instructions - FMA (FMA3)
// =============================================================================

/// VFMADD132PD ymm, ymm, ymm - Fused multiply-add (dst = dst*src2 + src1).
#[inline]
pub fn encode_vfmadd132pd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr_0f38_w(0x98, dst, src1, src2, true, 1, true)
}

/// VFMADD213PD ymm, ymm, ymm - Fused multiply-add (dst = src1*dst + src2).
#[inline]
pub fn encode_vfmadd213pd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr_0f38_w(0xA8, dst, src1, src2, true, 1, true)
}

/// VFMADD231PD ymm, ymm, ymm - Fused multiply-add (dst = src1*src2 + dst).
#[inline]
pub fn encode_vfmadd231pd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr_0f38_w(0xB8, dst, src1, src2, true, 1, true)
}

// =============================================================================
// VEX Encoded Instructions - Shuffle/Permute
// =============================================================================

/// VPERMILPD ymm, ymm, imm8 - Permute double-precision in-lane.
#[inline]
pub fn encode_vpermilpd_rri(dst: Ymm, src: Ymm, imm: u8) -> EncodedInst {
    encode_vex_rri_0f3a(0x05, dst, src, imm, true, 1)
}

/// VPERM2F128 ymm, ymm, ymm, imm8 - Permute 128-bit lanes.
#[inline]
pub fn encode_vperm2f128_rrri(dst: Ymm, src1: Ymm, src2: Ymm, imm: u8) -> EncodedInst {
    encode_vex_rrri_0f3a(0x06, dst, src1, src2, imm, true, 1)
}

// =============================================================================
// VEX Encoded Instructions - 128-bit (XMM) Variants
// =============================================================================

/// VADDPD xmm, xmm, xmm - Add packed doubles (128-bit).
#[inline]
pub fn encode_vaddpd_xmm_rrr(dst: Xmm, src1: Xmm, src2: Xmm) -> EncodedInst {
    encode_vex_xmm_rrr(0x58, dst, src1, src2, 1)
}

/// VSUBPD xmm, xmm, xmm - Subtract packed doubles (128-bit).
#[inline]
pub fn encode_vsubpd_xmm_rrr(dst: Xmm, src1: Xmm, src2: Xmm) -> EncodedInst {
    encode_vex_xmm_rrr(0x5C, dst, src1, src2, 1)
}

/// VMULPD xmm, xmm, xmm - Multiply packed doubles (128-bit).
#[inline]
pub fn encode_vmulpd_xmm_rrr(dst: Xmm, src1: Xmm, src2: Xmm) -> EncodedInst {
    encode_vex_xmm_rrr(0x59, dst, src1, src2, 1)
}

/// VDIVPD xmm, xmm, xmm - Divide packed doubles (128-bit).
#[inline]
pub fn encode_vdivpd_xmm_rrr(dst: Xmm, src1: Xmm, src2: Xmm) -> EncodedInst {
    encode_vex_xmm_rrr(0x5E, dst, src1, src2, 1)
}

/// VADDPS xmm, xmm, xmm - Add packed singles (128-bit).
#[inline]
pub fn encode_vaddps_xmm_rrr(dst: Xmm, src1: Xmm, src2: Xmm) -> EncodedInst {
    encode_vex_xmm_rrr(0x58, dst, src1, src2, 0)
}

/// VSUBPS xmm, xmm, xmm - Subtract packed singles (128-bit).
#[inline]
pub fn encode_vsubps_xmm_rrr(dst: Xmm, src1: Xmm, src2: Xmm) -> EncodedInst {
    encode_vex_xmm_rrr(0x5C, dst, src1, src2, 0)
}

/// VMULPS xmm, xmm, xmm - Multiply packed singles (128-bit).
#[inline]
pub fn encode_vmulps_xmm_rrr(dst: Xmm, src1: Xmm, src2: Xmm) -> EncodedInst {
    encode_vex_xmm_rrr(0x59, dst, src1, src2, 0)
}

/// VDIVPS xmm, xmm, xmm - Divide packed singles (128-bit).
#[inline]
pub fn encode_vdivps_xmm_rrr(dst: Xmm, src1: Xmm, src2: Xmm) -> EncodedInst {
    encode_vex_xmm_rrr(0x5E, dst, src1, src2, 0)
}

// =============================================================================
// VEX Encoded Instructions - Comparisons
// =============================================================================

/// VCMPPD ymm, ymm, ymm, imm8 - Compare packed doubles with predicate.
#[inline]
pub fn encode_vcmppd_rrri(dst: Ymm, src1: Ymm, src2: Ymm, predicate: u8) -> EncodedInst {
    let mut enc = encode_vex_rrr(0xC2, dst, src1, src2, true, 1);
    enc.push(predicate);
    enc
}

/// VCMPPS ymm, ymm, ymm, imm8 - Compare packed singles with predicate.
#[inline]
pub fn encode_vcmpps_rrri(dst: Ymm, src1: Ymm, src2: Ymm, predicate: u8) -> EncodedInst {
    let mut enc = encode_vex_rrr(0xC2, dst, src1, src2, true, 0);
    enc.push(predicate);
    enc
}

/// VPCMPEQD ymm, ymm, ymm - Compare packed 32-bit integers for equality.
#[inline]
pub fn encode_vpcmpeqd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x76, dst, src1, src2, true, 1)
}

/// VPCMPGTD ymm, ymm, ymm - Compare packed 32-bit integers for greater-than.
#[inline]
pub fn encode_vpcmpgtd_rrr(dst: Ymm, src1: Ymm, src2: Ymm) -> EncodedInst {
    encode_vex_rrr(0x66, dst, src1, src2, true, 1)
}

// =============================================================================
// VEX Encoded Instructions - Conversions
// =============================================================================

/// VCVTPD2PS xmm, ymm - Convert packed doubles to singles.
#[inline]
pub fn encode_vcvtpd2ps_rr(dst: Xmm, src: Ymm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l: true,
        pp: 1,
        mm: 1,
        w: false,
        r: !dst.high_bit(),
        x: true,
        b: !src.high_bit(),
        vvvv: 0,
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(0x5A);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VCVTPS2PD ymm, xmm - Convert packed singles to doubles.
#[inline]
pub fn encode_vcvtps2pd_rr(dst: Ymm, src: Xmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l: true,
        pp: 0,
        mm: 1,
        w: false,
        r: !dst.high_bit(),
        x: true,
        b: !src.high_bit(),
        vvvv: 0,
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(0x5A);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VCVTDQ2PD ymm, xmm - Convert packed 32-bit integers to doubles.
#[inline]
pub fn encode_vcvtdq2pd_rr(dst: Ymm, src: Xmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l: true,
        pp: 2,
        mm: 1,
        w: false, // pp=2 is F3 prefix
        r: !dst.high_bit(),
        x: true,
        b: !src.high_bit(),
        vvvv: 0,
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(0xE6);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VCVTTPD2DQ xmm, ymm - Convert packed doubles to 32-bit integers with truncation.
#[inline]
pub fn encode_vcvttpd2dq_rr(dst: Xmm, src: Ymm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l: true,
        pp: 1,
        mm: 1,
        w: false,
        r: !dst.high_bit(),
        x: true,
        b: !src.high_bit(),
        vvvv: 0,
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(0xE6);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

// =============================================================================
// VEX Encoded Helper for XMM
// =============================================================================

/// Encode VEX 3-operand XMM instruction (128-bit).
fn encode_vex_xmm_rrr(opcode: u8, dst: Xmm, src1: Xmm, src2: Xmm, pp: u8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l: false, // 128-bit
        pp,
        mm: 1,
        w: false,
        r: !dst.high_bit(),
        x: true,
        b: !src2.high_bit(),
        vvvv: src1.encoding(),
    };
    if vex.can_use_2byte() {
        let bytes = vex.encode_2byte();
        enc.push(bytes[0]);
        enc.push(bytes[1]);
    } else {
        let bytes = vex.encode_3byte();
        enc.push(bytes[0]);
        enc.push(bytes[1]);
        enc.push(bytes[2]);
    }
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src2.low_bits()));
    enc
}

// =============================================================================
// VEX Encoding Helpers
// =============================================================================

/// Encode VEX 3-operand register-register instruction (0F map).
fn encode_vex_rrr(opcode: u8, dst: Ymm, src1: Ymm, src2: Ymm, l: bool, pp: u8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l,
        pp,
        mm: 1,
        w: false,
        r: !dst.high_bit(),
        x: true,
        b: !src2.high_bit(),
        vvvv: src1.encoding(),
    };
    if vex.can_use_2byte() {
        let bytes = vex.encode_2byte();
        enc.push(bytes[0]);
        enc.push(bytes[1]);
    } else {
        let bytes = vex.encode_3byte();
        enc.push(bytes[0]);
        enc.push(bytes[1]);
        enc.push(bytes[2]);
    }
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src2.low_bits()));
    enc
}

/// Encode VEX 2-operand register-register instruction.
fn encode_vex_rr(opcode: u8, dst: Ymm, src: Ymm, l: bool, pp: u8) -> EncodedInst {
    encode_vex_rrr(opcode, dst, Ymm::Ymm0, src, l, pp)
}

/// Encode VEX register-memory instruction.
fn encode_vex_rm(opcode: u8, dst: Ymm, mem: &MemOperand, l: bool, pp: u8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l,
        pp,
        mm: 1,
        w: false,
        r: !dst.high_bit(),
        x: !mem.index.map(|i| i.high_bit()).unwrap_or(false),
        b: !mem.base.map(|b| b.high_bit()).unwrap_or(false),
        vvvv: 0,
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(opcode);
    encode_modrm_sib_disp_simd(&mut enc, dst.low_bits(), mem);
    enc
}

/// Encode VEX memory-register instruction.
fn encode_vex_mr(opcode: u8, mem: &MemOperand, src: Ymm, l: bool, pp: u8) -> EncodedInst {
    encode_vex_rm(opcode, src, mem, l, pp)
}

/// Encode VEX 3-operand instruction with 0F38 map.
fn encode_vex_rrr_0f38(opcode: u8, dst: Ymm, src1: Ymm, src2: Ymm, l: bool, pp: u8) -> EncodedInst {
    encode_vex_rrr_0f38_w(opcode, dst, src1, src2, l, pp, false)
}

/// Encode VEX 3-operand instruction with 0F38 map and W bit.
fn encode_vex_rrr_0f38_w(
    opcode: u8,
    dst: Ymm,
    src1: Ymm,
    src2: Ymm,
    l: bool,
    pp: u8,
    w: bool,
) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l,
        pp,
        mm: 2,
        w, // mm=2 for 0F38
        r: !dst.high_bit(),
        x: true,
        b: !src2.high_bit(),
        vvvv: src1.encoding(),
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src2.low_bits()));
    enc
}

/// Encode VEX register-memory instruction with 0F38 map.
fn encode_vex_rm_0f38(opcode: u8, dst: Ymm, mem: &MemOperand, l: bool, pp: u8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l,
        pp,
        mm: 2,
        w: false,
        r: !dst.high_bit(),
        x: !mem.index.map(|i| i.high_bit()).unwrap_or(false),
        b: !mem.base.map(|b| b.high_bit()).unwrap_or(false),
        vvvv: 0,
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(opcode);
    encode_modrm_sib_disp_simd(&mut enc, dst.low_bits(), mem);
    enc
}

/// Encode VEX register-register-immediate with 0F3A map.
fn encode_vex_rri_0f3a(opcode: u8, dst: Ymm, src: Ymm, imm: u8, l: bool, pp: u8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l,
        pp,
        mm: 3,
        w: false, // mm=3 for 0F3A
        r: !dst.high_bit(),
        x: true,
        b: !src.high_bit(),
        vvvv: 0,
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc.push(imm);
    enc
}

/// Encode VEX 3-operand + immediate with 0F3A map.
fn encode_vex_rrri_0f3a(
    opcode: u8,
    dst: Ymm,
    src1: Ymm,
    src2: Ymm,
    imm: u8,
    l: bool,
    pp: u8,
) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let vex = Vex {
        l,
        pp,
        mm: 3,
        w: false,
        r: !dst.high_bit(),
        x: true,
        b: !src2.high_bit(),
        vvvv: src1.encoding(),
    };
    let bytes = vex.encode_3byte();
    enc.push(bytes[0]);
    enc.push(bytes[1]);
    enc.push(bytes[2]);
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src2.low_bits()));
    enc.push(imm);
    enc
}

/// Encode ModR/M, SIB, and displacement for SIMD memory operand.
fn encode_modrm_sib_disp_simd(enc: &mut EncodedInst, reg: u8, mem: &MemOperand) {
    match (mem.base, mem.index) {
        (None, None) => {
            enc.push(modrm(Mod::Indirect, reg, 0b101));
            enc.push_u32(mem.disp as u32);
        }
        (Some(base), None) => {
            let needs_sib = base.needs_sib_as_base();
            let needs_disp = base.needs_displacement() && mem.disp == 0;
            let mod_field = if mem.disp == 0 && !needs_disp {
                Mod::Indirect
            } else if mem.disp >= -128 && mem.disp <= 127 {
                Mod::IndirectDisp8
            } else {
                Mod::IndirectDisp32
            };
            if needs_sib {
                enc.push(modrm(mod_field, reg, 0b100));
                enc.push(sib(Scale::X1, 0b100, base.low_bits()));
            } else {
                enc.push(modrm(mod_field, reg, base.low_bits()));
            }
            match mod_field {
                Mod::IndirectDisp8 => enc.push(mem.disp as i8 as u8),
                Mod::IndirectDisp32 => enc.push_u32(mem.disp as u32),
                _ if needs_disp => enc.push(0),
                _ => {}
            }
        }
        (Some(base), Some(index)) => {
            let mod_field = if mem.disp == 0 && !base.needs_displacement() {
                Mod::Indirect
            } else if mem.disp >= -128 && mem.disp <= 127 {
                Mod::IndirectDisp8
            } else {
                Mod::IndirectDisp32
            };
            enc.push(modrm(mod_field, reg, 0b100));
            enc.push(sib(mem.scale, index.low_bits(), base.low_bits()));
            match mod_field {
                Mod::IndirectDisp8 => enc.push(mem.disp as i8 as u8),
                Mod::IndirectDisp32 => enc.push_u32(mem.disp as u32),
                Mod::Indirect if base.needs_displacement() => enc.push(0),
                _ => {}
            }
        }
        (None, Some(index)) => {
            enc.push(modrm(Mod::Indirect, reg, 0b100));
            enc.push(sib(mem.scale, index.low_bits(), 0b101));
            enc.push_u32(mem.disp as u32);
        }
    }
}

// Note: EncodedInst::push and push_u32 are pub(crate) in encoder.rs

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ymm_encoding() {
        assert_eq!(Ymm::Ymm0.encoding(), 0);
        assert_eq!(Ymm::Ymm8.encoding(), 8);
        assert!(!Ymm::Ymm7.high_bit());
        assert!(Ymm::Ymm8.high_bit());
    }

    #[test]
    fn test_vex_2byte() {
        let vex = Vex {
            l: true,
            pp: 1,
            mm: 1,
            w: false,
            r: true,
            x: true,
            b: true,
            vvvv: 0,
        };
        assert!(vex.can_use_2byte());
        let bytes = vex.encode_2byte();
        assert_eq!(bytes[0], 0xC5);
    }

    #[test]
    fn test_vex_3byte() {
        let vex = Vex {
            l: true,
            pp: 1,
            mm: 2,
            w: false,
            r: true,
            x: true,
            b: true,
            vvvv: 0,
        };
        assert!(!vex.can_use_2byte());
        let bytes = vex.encode_3byte();
        assert_eq!(bytes[0], 0xC4);
    }

    #[test]
    fn test_vaddpd() {
        let enc = encode_vaddpd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        assert!(enc.len() > 0);
        // Check for VEX prefix
        let slice = enc.as_slice();
        assert!(slice[0] == 0xC5 || slice[0] == 0xC4);
    }

    #[test]
    fn test_vmovapd_rm() {
        use super::super::registers::Gpr;
        let mem = MemOperand::base(Gpr::Rax);
        let enc = encode_vmovapd_rm(Ymm::Ymm0, &mem);
        assert!(enc.len() > 0);
    }

    #[test]
    fn test_ymm_xmm_conversion() {
        let ymm = Ymm::Ymm5;
        let xmm = ymm.to_xmm();
        assert_eq!(xmm, Xmm::Xmm5);
        assert_eq!(Ymm::from_xmm(xmm), ymm);
    }

    #[test]
    fn test_zmm_encoding() {
        assert_eq!(Zmm::Zmm0.encoding(), 0);
        assert_eq!(Zmm::Zmm16.encoding(), 16);
        assert!(Zmm::Zmm16.ext_bit());
        assert!(!Zmm::Zmm15.ext_bit());
    }

    #[test]
    fn test_vsubpd_encoding() {
        let enc = encode_vsubpd_rrr(Ymm::Ymm2, Ymm::Ymm3, Ymm::Ymm4);
        assert!(enc.len() > 0);
        let slice = enc.as_slice();
        // VEX prefix start
        assert!(slice[0] == 0xC5 || slice[0] == 0xC4);
    }

    #[test]
    fn test_vmulpd_encoding() {
        let enc = encode_vmulpd_rrr(Ymm::Ymm5, Ymm::Ymm6, Ymm::Ymm7);
        assert!(enc.len() >= 4); // VEX + opcode + modrm
    }

    #[test]
    fn test_vdivpd_encoding() {
        let enc = encode_vdivpd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        assert!(enc.len() >= 4);
    }

    #[test]
    fn test_packed_single_operations() {
        let add = encode_vaddps_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        let sub = encode_vsubps_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        let mul = encode_vmulps_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        let div = encode_vdivps_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);

        // All should have reasonable length
        assert!(add.len() >= 4);
        assert!(sub.len() >= 4);
        assert!(mul.len() >= 4);
        assert!(div.len() >= 4);
    }

    #[test]
    fn test_integer_operations() {
        let paddd = encode_vpaddd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        let psubd = encode_vpsubd_rrr(Ymm::Ymm3, Ymm::Ymm4, Ymm::Ymm5);
        let paddq = encode_vpaddq_rrr(Ymm::Ymm6, Ymm::Ymm7, Ymm::Ymm0);
        let psubq = encode_vpsubq_rrr(Ymm::Ymm1, Ymm::Ymm2, Ymm::Ymm3);

        assert!(paddd.len() >= 4);
        assert!(psubd.len() >= 4);
        assert!(paddq.len() >= 4);
        assert!(psubq.len() >= 4);
    }

    #[test]
    fn test_logical_operations() {
        let andpd = encode_vandpd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        let orpd = encode_vorpd_rrr(Ymm::Ymm3, Ymm::Ymm4, Ymm::Ymm5);
        let xorpd = encode_vxorpd_rrr(Ymm::Ymm6, Ymm::Ymm7, Ymm::Ymm0);

        assert!(andpd.len() >= 4);
        assert!(orpd.len() >= 4);
        assert!(xorpd.len() >= 4);
    }

    #[test]
    fn test_fma_operations() {
        let fma132 = encode_vfmadd132pd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        let fma213 = encode_vfmadd213pd_rrr(Ymm::Ymm3, Ymm::Ymm4, Ymm::Ymm5);
        let fma231 = encode_vfmadd231pd_rrr(Ymm::Ymm6, Ymm::Ymm7, Ymm::Ymm0);

        // FMA uses 3-byte VEX with W bit
        assert!(fma132.len() >= 5);
        assert!(fma213.len() >= 5);
        assert!(fma231.len() >= 5);

        // All should use C4 prefix (3-byte VEX for 0F38 map)
        assert_eq!(fma132.as_slice()[0], 0xC4);
        assert_eq!(fma213.as_slice()[0], 0xC4);
        assert_eq!(fma231.as_slice()[0], 0xC4);
    }

    #[test]
    fn test_comparison_operations() {
        // VCMPPD with EQ predicate (0)
        let cmp_eq = encode_vcmppd_rrri(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2, 0);
        // VCMPPD with LT predicate (1)
        let cmp_lt = encode_vcmppd_rrri(Ymm::Ymm3, Ymm::Ymm4, Ymm::Ymm5, 1);

        assert!(cmp_eq.len() >= 5); // VEX + opcode + modrm + imm8
        assert!(cmp_lt.len() >= 5);
    }

    #[test]
    fn test_conversion_operations() {
        let pd2ps = encode_vcvtpd2ps_rr(Xmm::Xmm0, Ymm::Ymm1);
        let ps2pd = encode_vcvtps2pd_rr(Ymm::Ymm2, Xmm::Xmm3);
        let dq2pd = encode_vcvtdq2pd_rr(Ymm::Ymm4, Xmm::Xmm5);
        let pd2dq = encode_vcvttpd2dq_rr(Xmm::Xmm6, Ymm::Ymm7);

        assert!(pd2ps.len() >= 5);
        assert!(ps2pd.len() >= 5);
        assert!(dq2pd.len() >= 5);
        assert!(pd2dq.len() >= 5);
    }

    #[test]
    fn test_xmm_128bit_operations() {
        let add_xmm = encode_vaddpd_xmm_rrr(Xmm::Xmm0, Xmm::Xmm1, Xmm::Xmm2);
        let mul_xmm = encode_vmulpd_xmm_rrr(Xmm::Xmm3, Xmm::Xmm4, Xmm::Xmm5);

        // 128-bit operations can use 2-byte VEX
        assert!(add_xmm.len() >= 4);
        assert!(mul_xmm.len() >= 4);

        // L bit should be 0 for 128-bit
        let slice = add_xmm.as_slice();
        if slice[0] == 0xC5 {
            // 2-byte VEX: byte 1 has L bit at position 2
            let l_bit = (slice[1] >> 2) & 1;
            assert_eq!(l_bit, 0); // 128-bit
        }
    }

    #[test]
    fn test_shuffle_operations() {
        let vpermilpd = encode_vpermilpd_rri(Ymm::Ymm0, Ymm::Ymm1, 0b1010);
        let vperm2f128 = encode_vperm2f128_rrri(Ymm::Ymm2, Ymm::Ymm3, Ymm::Ymm4, 0x31);

        assert!(vpermilpd.len() >= 5); // VEX + opcode + modrm + imm8
        assert!(vperm2f128.len() >= 6); // VEX + opcode + modrm + imm8
    }

    #[test]
    fn test_memory_operations() {
        use super::super::registers::Gpr;

        let mem_rax = MemOperand::base(Gpr::Rax);
        let mem_rbx_disp = MemOperand::base_disp(Gpr::Rbx, 64);

        let movapd_rm = encode_vmovapd_rm(Ymm::Ymm0, &mem_rax);
        let movupd_rm = encode_vmovupd_rm(Ymm::Ymm1, &mem_rbx_disp);
        let movapd_mr = encode_vmovapd_mr(&mem_rax, Ymm::Ymm2);
        let movupd_mr = encode_vmovupd_mr(&mem_rbx_disp, Ymm::Ymm3);
        let broadcast = encode_vbroadcastsd_rm(Ymm::Ymm4, &mem_rax);

        assert!(movapd_rm.len() > 0);
        assert!(movupd_rm.len() > 0);
        assert!(movapd_mr.len() > 0);
        assert!(movupd_mr.len() > 0);
        assert!(broadcast.len() > 0);
    }

    #[test]
    fn test_all_ymm_registers() {
        // Ensure high registers (Ymm8-Ymm15) work correctly
        let enc_low = encode_vaddpd_rrr(Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2);
        let enc_high = encode_vaddpd_rrr(Ymm::Ymm8, Ymm::Ymm9, Ymm::Ymm10);

        assert!(enc_low.len() > 0);
        assert!(enc_high.len() > 0);

        // High register encoding should use 3-byte VEX (C4 prefix)
        assert_eq!(enc_high.as_slice()[0], 0xC4);
    }

    #[test]
    fn test_vex_prefix_fields() {
        // Test all VEX field combinations
        let vex_basic = Vex {
            l: false,
            pp: 0,
            mm: 1,
            w: false,
            r: true,
            x: true,
            b: true,
            vvvv: 0,
        };
        assert!(vex_basic.can_use_2byte());

        let vex_256bit = Vex {
            l: true,
            pp: 1,
            mm: 1,
            w: false,
            r: true,
            x: true,
            b: true,
            vvvv: 5,
        };
        assert!(vex_256bit.can_use_2byte());

        let vex_0f38 = Vex {
            l: true,
            pp: 1,
            mm: 2,
            w: false,
            r: true,
            x: true,
            b: true,
            vvvv: 0,
        };
        assert!(!vex_0f38.can_use_2byte()); // mm != 1 forces 3-byte

        let vex_with_w = Vex {
            l: true,
            pp: 1,
            mm: 1,
            w: true,
            r: true,
            x: true,
            b: true,
            vvvv: 0,
        };
        assert!(!vex_with_w.can_use_2byte()); // w=1 forces 3-byte
    }

    #[test]
    fn test_kreg_encoding() {
        assert_eq!(KReg::K0.encoding(), 0);
        assert_eq!(KReg::K7.encoding(), 7);
    }

    // =========================================================================
    // YmmSet Tests
    // =========================================================================

    #[test]
    fn test_ymmset_empty_and_all() {
        assert!(YmmSet::EMPTY.is_empty());
        assert_eq!(YmmSet::EMPTY.count(), 0);
        assert!(!YmmSet::ALL.is_empty());
        assert_eq!(YmmSet::ALL.count(), 16);
    }

    #[test]
    fn test_ymmset_singleton() {
        let set = YmmSet::singleton(Ymm::Ymm5);
        assert!(set.contains(Ymm::Ymm5));
        assert!(!set.contains(Ymm::Ymm0));
        assert_eq!(set.count(), 1);
    }

    #[test]
    fn test_ymmset_insert_remove() {
        let mut set = YmmSet::EMPTY;
        set = set.insert(Ymm::Ymm0);
        set = set.insert(Ymm::Ymm15);

        assert!(set.contains(Ymm::Ymm0));
        assert!(set.contains(Ymm::Ymm15));
        assert!(!set.contains(Ymm::Ymm7));
        assert_eq!(set.count(), 2);

        set = set.remove(Ymm::Ymm0);
        assert!(!set.contains(Ymm::Ymm0));
        assert!(set.contains(Ymm::Ymm15));
        assert_eq!(set.count(), 1);
    }

    #[test]
    fn test_ymmset_union_intersection() {
        let a = YmmSet::singleton(Ymm::Ymm0)
            .insert(Ymm::Ymm1)
            .insert(Ymm::Ymm2);
        let b = YmmSet::singleton(Ymm::Ymm1)
            .insert(Ymm::Ymm2)
            .insert(Ymm::Ymm3);

        let union = a.union(b);
        assert_eq!(union.count(), 4);
        assert!(union.contains(Ymm::Ymm0));
        assert!(union.contains(Ymm::Ymm3));

        let intersection = a.intersection(b);
        assert_eq!(intersection.count(), 2);
        assert!(intersection.contains(Ymm::Ymm1));
        assert!(intersection.contains(Ymm::Ymm2));
        assert!(!intersection.contains(Ymm::Ymm0));
    }

    #[test]
    fn test_ymmset_difference() {
        let a = YmmSet::singleton(Ymm::Ymm0)
            .insert(Ymm::Ymm1)
            .insert(Ymm::Ymm2);
        let b = YmmSet::singleton(Ymm::Ymm1);

        let diff = a.difference(b);
        assert_eq!(diff.count(), 2);
        assert!(diff.contains(Ymm::Ymm0));
        assert!(diff.contains(Ymm::Ymm2));
        assert!(!diff.contains(Ymm::Ymm1));
    }

    #[test]
    fn test_ymmset_first() {
        assert_eq!(YmmSet::EMPTY.first(), None);

        let set = YmmSet::singleton(Ymm::Ymm5).insert(Ymm::Ymm10);
        assert_eq!(set.first(), Some(Ymm::Ymm5));

        let high_only = YmmSet::singleton(Ymm::Ymm15);
        assert_eq!(high_only.first(), Some(Ymm::Ymm15));
    }

    #[test]
    fn test_ymmset_iter() {
        let set = YmmSet::singleton(Ymm::Ymm3)
            .insert(Ymm::Ymm7)
            .insert(Ymm::Ymm11);

        let regs: Vec<Ymm> = set.iter().collect();
        assert_eq!(regs.len(), 3);
        assert_eq!(regs[0], Ymm::Ymm3);
        assert_eq!(regs[1], Ymm::Ymm7);
        assert_eq!(regs[2], Ymm::Ymm11);
    }

    #[test]
    fn test_ymmset_from_bits() {
        let set = YmmSet::from_bits(0b1010_0101);
        assert!(set.contains(Ymm::Ymm0));
        assert!(set.contains(Ymm::Ymm2));
        assert!(set.contains(Ymm::Ymm5));
        assert!(set.contains(Ymm::Ymm7));
        assert_eq!(set.count(), 4);
    }

    #[test]
    fn test_ymmset_all_registers() {
        for reg in YmmSet::ALL.iter() {
            assert!(YmmSet::ALL.contains(reg));
        }
        assert_eq!(YmmSet::ALL.iter().count(), 16);
    }

    // =========================================================================
    // ZmmSet Tests
    // =========================================================================

    #[test]
    fn test_zmmset_empty_and_all() {
        assert!(ZmmSet::EMPTY.is_empty());
        assert_eq!(ZmmSet::EMPTY.count(), 0);
        assert!(!ZmmSet::ALL.is_empty());
        assert_eq!(ZmmSet::ALL.count(), 32);
    }

    #[test]
    fn test_zmmset_all16_and_upper16() {
        assert_eq!(ZmmSet::ALL16.count(), 16);
        assert_eq!(ZmmSet::UPPER16.count(), 16);

        // Check they don't overlap
        let intersection = ZmmSet::ALL16.intersection(ZmmSet::UPPER16);
        assert!(intersection.is_empty());

        // Check they combine to ALL
        let union = ZmmSet::ALL16.union(ZmmSet::UPPER16);
        assert_eq!(union, ZmmSet::ALL);
    }

    #[test]
    fn test_zmmset_singleton() {
        let set = ZmmSet::singleton(Zmm::Zmm20);
        assert!(set.contains(Zmm::Zmm20));
        assert!(!set.contains(Zmm::Zmm0));
        assert_eq!(set.count(), 1);
    }

    #[test]
    fn test_zmmset_insert_remove() {
        let mut set = ZmmSet::EMPTY;
        set = set.insert(Zmm::Zmm0);
        set = set.insert(Zmm::Zmm31);
        set = set.insert(Zmm::Zmm16);

        assert!(set.contains(Zmm::Zmm0));
        assert!(set.contains(Zmm::Zmm31));
        assert!(set.contains(Zmm::Zmm16));
        assert_eq!(set.count(), 3);

        set = set.remove(Zmm::Zmm16);
        assert!(!set.contains(Zmm::Zmm16));
        assert_eq!(set.count(), 2);
    }

    #[test]
    fn test_zmmset_union_intersection() {
        let a = ZmmSet::singleton(Zmm::Zmm0)
            .insert(Zmm::Zmm16)
            .insert(Zmm::Zmm20);
        let b = ZmmSet::singleton(Zmm::Zmm16)
            .insert(Zmm::Zmm20)
            .insert(Zmm::Zmm30);

        let union = a.union(b);
        assert_eq!(union.count(), 4);

        let intersection = a.intersection(b);
        assert_eq!(intersection.count(), 2);
        assert!(intersection.contains(Zmm::Zmm16));
        assert!(intersection.contains(Zmm::Zmm20));
    }

    #[test]
    fn test_zmmset_difference() {
        let a = ZmmSet::ALL16;
        let b = ZmmSet::singleton(Zmm::Zmm0).insert(Zmm::Zmm1);

        let diff = a.difference(b);
        assert_eq!(diff.count(), 14);
        assert!(!diff.contains(Zmm::Zmm0));
        assert!(!diff.contains(Zmm::Zmm1));
    }

    #[test]
    fn test_zmmset_first() {
        assert_eq!(ZmmSet::EMPTY.first(), None);

        let set = ZmmSet::singleton(Zmm::Zmm20).insert(Zmm::Zmm25);
        assert_eq!(set.first(), Some(Zmm::Zmm20));

        let low = ZmmSet::singleton(Zmm::Zmm0);
        assert_eq!(low.first(), Some(Zmm::Zmm0));
    }

    #[test]
    fn test_zmmset_iter() {
        let set = ZmmSet::singleton(Zmm::Zmm5)
            .insert(Zmm::Zmm15)
            .insert(Zmm::Zmm25);

        let regs: Vec<Zmm> = set.iter().collect();
        assert_eq!(regs.len(), 3);
        assert_eq!(regs[0], Zmm::Zmm5);
        assert_eq!(regs[1], Zmm::Zmm15);
        assert_eq!(regs[2], Zmm::Zmm25);
    }

    #[test]
    fn test_zmmset_from_bits() {
        let set = ZmmSet::from_bits(0xFFFF_0000); // Upper 16
        assert_eq!(set, ZmmSet::UPPER16);
        assert!(set.contains(Zmm::Zmm16));
        assert!(!set.contains(Zmm::Zmm0));
    }

    #[test]
    fn test_zmmset_all_registers() {
        for reg in ZmmSet::ALL.iter() {
            assert!(ZmmSet::ALL.contains(reg));
        }
        assert_eq!(ZmmSet::ALL.iter().count(), 32);
    }

    // =========================================================================
    // Ymm/Zmm Conversion Tests
    // =========================================================================

    #[test]
    fn test_ymm_from_encoding() {
        for i in 0_u8..16 {
            let ymm = Ymm::from_encoding(i);
            assert!(ymm.is_some());
            assert_eq!(ymm.unwrap().encoding(), i);
        }
        assert_eq!(Ymm::from_encoding(16), None);
        assert_eq!(Ymm::from_encoding(255), None);
    }

    #[test]
    fn test_zmm_from_encoding() {
        for i in 0_u8..32 {
            let zmm = Zmm::from_encoding(i);
            assert!(zmm.is_some());
            assert_eq!(zmm.unwrap().encoding(), i);
        }
        assert_eq!(Zmm::from_encoding(32), None);
        assert_eq!(Zmm::from_encoding(255), None);
    }

    #[test]
    fn test_ymm_xmm_conversion_roundtrip() {
        use crate::backend::x64::Xmm;
        for i in 0_u8..16 {
            if let Some(ymm) = Ymm::from_encoding(i) {
                let xmm = ymm.to_xmm();
                let back = Ymm::from_xmm(xmm);
                assert_eq!(back, ymm);
            }
        }
    }

    #[test]
    fn test_zmm_ymm_conversion_roundtrip() {
        for i in 0_u8..16 {
            if let Some(zmm) = Zmm::from_encoding(i) {
                if let Some(ymm) = zmm.to_ymm() {
                    let back = Zmm::from_ymm(ymm);
                    assert_eq!(back, zmm);
                }
            }
        }
    }

    #[test]
    fn test_zmm_upper_no_ymm() {
        // ZMM16-31 don't have YMM equivalents
        for i in 16_u8..32 {
            if let Some(zmm) = Zmm::from_encoding(i) {
                assert_eq!(zmm.to_ymm(), None);
            }
        }
    }

    #[test]
    fn test_ymm_all_constant() {
        assert_eq!(Ymm::ALL.len(), 16);
        for (i, ymm) in Ymm::ALL.iter().enumerate() {
            assert_eq!(ymm.encoding(), i as u8);
        }
    }

    #[test]
    fn test_zmm_all_constants() {
        assert_eq!(Zmm::ALL.len(), 32);
        assert_eq!(Zmm::ALL16.len(), 16);
        for (i, zmm) in Zmm::ALL.iter().enumerate() {
            assert_eq!(zmm.encoding(), i as u8);
        }
    }

    // =========================================================================
    // Set Identity/Property Tests
    // =========================================================================

    #[test]
    fn test_ymmset_identity_properties() {
        let set = YmmSet::singleton(Ymm::Ymm3).insert(Ymm::Ymm7);

        // Union with empty is identity
        assert_eq!(set.union(YmmSet::EMPTY), set);

        // Intersection with ALL is identity
        assert_eq!(set.intersection(YmmSet::ALL), set);

        // Difference with empty is identity
        assert_eq!(set.difference(YmmSet::EMPTY), set);

        // Difference with self is empty
        assert_eq!(set.difference(set), YmmSet::EMPTY);
    }

    #[test]
    fn test_zmmset_identity_properties() {
        let set = ZmmSet::singleton(Zmm::Zmm10).insert(Zmm::Zmm25);

        // Union with empty is identity
        assert_eq!(set.union(ZmmSet::EMPTY), set);

        // Intersection with ALL is identity
        assert_eq!(set.intersection(ZmmSet::ALL), set);

        // Difference with empty is identity
        assert_eq!(set.difference(ZmmSet::EMPTY), set);

        // Difference with self is empty
        assert_eq!(set.difference(set), ZmmSet::EMPTY);
    }

    #[test]
    fn test_ymmset_debug_format() {
        let set = YmmSet::singleton(Ymm::Ymm0).insert(Ymm::Ymm15);
        let debug_str = format!("{:?}", set);
        assert!(debug_str.contains("YmmSet"));
        assert!(debug_str.contains("ymm0"));
        assert!(debug_str.contains("ymm15"));
    }

    #[test]
    fn test_zmmset_debug_format() {
        let set = ZmmSet::singleton(Zmm::Zmm0).insert(Zmm::Zmm31);
        let debug_str = format!("{:?}", set);
        assert!(debug_str.contains("ZmmSet"));
        assert!(debug_str.contains("zmm0"));
        assert!(debug_str.contains("zmm31"));
    }

    // =========================================================================
    // YMM Register-to-Register Move Tests
    // =========================================================================

    #[test]
    fn test_vmovapd_ymm_rr_basic() {
        let enc = super::encode_vmovapd_ymm_rr(Ymm::Ymm0, Ymm::Ymm1);
        // VEX.256 + 66 prefix + opcode 0x28 + ModR/M
        assert!(enc.len() >= 3);
        // Verify it starts with VEX prefix
        assert!(enc.as_slice()[0] == 0xC5 || enc.as_slice()[0] == 0xC4);
    }

    #[test]
    fn test_vmovapd_ymm_rr_extended_registers() {
        // YMM8 and YMM15 require REX bits via VEX
        let enc = super::encode_vmovapd_ymm_rr(Ymm::Ymm8, Ymm::Ymm15);
        assert!(enc.len() >= 3);
        // With extended registers, 3-byte VEX is needed
        assert_eq!(enc.as_slice()[0], 0xC4);
    }

    #[test]
    fn test_vmovaps_ymm_rr_basic() {
        let enc = super::encode_vmovaps_ymm_rr(Ymm::Ymm2, Ymm::Ymm3);
        assert!(enc.len() >= 3);
        // Should use VEX without 66 prefix (pp=0)
        assert!(enc.as_slice()[0] == 0xC5 || enc.as_slice()[0] == 0xC4);
    }

    #[test]
    fn test_vmovupd_ymm_rr_basic() {
        let enc = super::encode_vmovupd_ymm_rr(Ymm::Ymm4, Ymm::Ymm5);
        assert!(enc.len() >= 3);
        // Uses opcode 0x10
    }

    #[test]
    fn test_vmovups_ymm_rr_basic() {
        let enc = super::encode_vmovups_ymm_rr(Ymm::Ymm6, Ymm::Ymm7);
        assert!(enc.len() >= 3);
    }

    #[test]
    fn test_vmovdqa_ymm_rr_basic() {
        let enc = super::encode_vmovdqa_ymm_rr(Ymm::Ymm0, Ymm::Ymm1);
        assert!(enc.len() >= 3);
        // Uses opcode 0x6F
    }

    #[test]
    fn test_vmovdqu_ymm_rr_basic() {
        let enc = super::encode_vmovdqu_ymm_rr(Ymm::Ymm2, Ymm::Ymm3);
        assert!(enc.len() >= 3);
        // Uses F3 prefix (pp=2)
    }

    // =========================================================================
    // XMM Register-to-Register Move Tests (via VEX encoding)
    // =========================================================================

    #[test]
    fn test_vmovapd_xmm_rr_basic() {
        // Uses L=0 for 128-bit
        let enc = super::encode_vmovapd_xmm_rr(Ymm::Ymm0, Ymm::Ymm1);
        assert!(enc.len() >= 3);
    }

    #[test]
    fn test_vmovaps_xmm_rr_basic() {
        let enc = super::encode_vmovaps_xmm_rr(Ymm::Ymm4, Ymm::Ymm5);
        assert!(enc.len() >= 3);
    }

    #[test]
    fn test_vmovdqa_xmm_rr_basic() {
        let enc = super::encode_vmovdqa_xmm_rr(Ymm::Ymm6, Ymm::Ymm7);
        assert!(enc.len() >= 3);
    }

    #[test]
    fn test_ymm_move_all_registers() {
        // Test move instruction encoding for all YMM register pairs
        for i in 0..16u8 {
            for j in 0..16u8 {
                let src = Ymm::from_encoding(i).unwrap();
                let dst = Ymm::from_encoding(j).unwrap();

                // All combinations should encode without panic
                let enc = super::encode_vmovapd_ymm_rr(dst, src);
                assert!(enc.len() >= 3, "Failed for YMM{} -> YMM{}", i, j);
            }
        }
    }

    #[test]
    fn test_ymm_move_self_encoding() {
        // Moving register to itself should still encode correctly
        for i in 0..16u8 {
            let reg = Ymm::from_encoding(i).unwrap();
            let enc = super::encode_vmovapd_ymm_rr(reg, reg);
            assert!(enc.len() >= 3);
        }
    }
}
