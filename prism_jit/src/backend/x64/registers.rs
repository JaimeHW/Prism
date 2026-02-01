//! x64 register definitions and calling conventions.
//!
//! This module provides:
//! - General-purpose register (GPR) definitions with proper encoding
//! - XMM/SSE register definitions for floating-point operations
//! - Calling convention abstractions for Windows x64 and System V ABIs
//!
//! # Performance Considerations
//! - All register types are `Copy` with `#[repr(u8)]` for zero-cost encoding
//! - Register sets use bitfields for O(1) membership testing
//! - Calling conventions are const-evaluated where possible

use std::fmt;

// =============================================================================
// General-Purpose Registers (GPR)
// =============================================================================

/// x64 general-purpose register with proper hardware encoding.
///
/// The encoding bits directly map to the x64 instruction format:
/// - Bits 0-2: Stored in ModR/M or opcode extension
/// - Bit 3: Stored in REX.B, REX.R, or REX.X prefix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Gpr {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Gpr {
    /// All 16 general-purpose registers in encoding order.
    pub const ALL: [Gpr; 16] = [
        Gpr::Rax,
        Gpr::Rcx,
        Gpr::Rdx,
        Gpr::Rbx,
        Gpr::Rsp,
        Gpr::Rbp,
        Gpr::Rsi,
        Gpr::Rdi,
        Gpr::R8,
        Gpr::R9,
        Gpr::R10,
        Gpr::R11,
        Gpr::R12,
        Gpr::R13,
        Gpr::R14,
        Gpr::R15,
    ];

    /// Get the hardware encoding (0-15).
    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Get bits 0-2 for ModR/M encoding.
    #[inline(always)]
    pub const fn low_bits(self) -> u8 {
        self.encoding() & 0x7
    }

    /// Get bit 3 for REX prefix.
    #[inline(always)]
    pub const fn high_bit(self) -> bool {
        self.encoding() >= 8
    }

    /// Check if this is an "extended" register (R8-R15).
    #[inline(always)]
    pub const fn is_extended(self) -> bool {
        self.high_bit()
    }

    /// Check if this register requires SIB byte when used as base.
    /// RSP and R12 have encoding 0b100 which conflicts with SIB escape.
    #[inline(always)]
    pub const fn needs_sib_as_base(self) -> bool {
        self.low_bits() == 4
    }

    /// Check if this register cannot be used with ModR/M displacement-only mode.
    /// RBP and R13 have encoding 0b101 which means [disp32] in mod=00.
    #[inline(always)]
    pub const fn needs_displacement(self) -> bool {
        self.low_bits() == 5
    }

    /// Convert from encoding value if valid.
    #[inline]
    pub const fn from_encoding(enc: u8) -> Option<Gpr> {
        match enc {
            0 => Some(Gpr::Rax),
            1 => Some(Gpr::Rcx),
            2 => Some(Gpr::Rdx),
            3 => Some(Gpr::Rbx),
            4 => Some(Gpr::Rsp),
            5 => Some(Gpr::Rbp),
            6 => Some(Gpr::Rsi),
            7 => Some(Gpr::Rdi),
            8 => Some(Gpr::R8),
            9 => Some(Gpr::R9),
            10 => Some(Gpr::R10),
            11 => Some(Gpr::R11),
            12 => Some(Gpr::R12),
            13 => Some(Gpr::R13),
            14 => Some(Gpr::R14),
            15 => Some(Gpr::R15),
            _ => None,
        }
    }

    /// Get the 32-bit register name (for display).
    pub const fn name_32(self) -> &'static str {
        match self {
            Gpr::Rax => "eax",
            Gpr::Rcx => "ecx",
            Gpr::Rdx => "edx",
            Gpr::Rbx => "ebx",
            Gpr::Rsp => "esp",
            Gpr::Rbp => "ebp",
            Gpr::Rsi => "esi",
            Gpr::Rdi => "edi",
            Gpr::R8 => "r8d",
            Gpr::R9 => "r9d",
            Gpr::R10 => "r10d",
            Gpr::R11 => "r11d",
            Gpr::R12 => "r12d",
            Gpr::R13 => "r13d",
            Gpr::R14 => "r14d",
            Gpr::R15 => "r15d",
        }
    }

    /// Get the 64-bit register name.
    pub const fn name_64(self) -> &'static str {
        match self {
            Gpr::Rax => "rax",
            Gpr::Rcx => "rcx",
            Gpr::Rdx => "rdx",
            Gpr::Rbx => "rbx",
            Gpr::Rsp => "rsp",
            Gpr::Rbp => "rbp",
            Gpr::Rsi => "rsi",
            Gpr::Rdi => "rdi",
            Gpr::R8 => "r8",
            Gpr::R9 => "r9",
            Gpr::R10 => "r10",
            Gpr::R11 => "r11",
            Gpr::R12 => "r12",
            Gpr::R13 => "r13",
            Gpr::R14 => "r14",
            Gpr::R15 => "r15",
        }
    }
}

impl fmt::Display for Gpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_64())
    }
}

// =============================================================================
// XMM Registers (SSE/AVX)
// =============================================================================

/// x64 XMM register for floating-point and SIMD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Xmm {
    Xmm0 = 0,
    Xmm1 = 1,
    Xmm2 = 2,
    Xmm3 = 3,
    Xmm4 = 4,
    Xmm5 = 5,
    Xmm6 = 6,
    Xmm7 = 7,
    Xmm8 = 8,
    Xmm9 = 9,
    Xmm10 = 10,
    Xmm11 = 11,
    Xmm12 = 12,
    Xmm13 = 13,
    Xmm14 = 14,
    Xmm15 = 15,
}

impl Xmm {
    /// All 16 XMM registers.
    pub const ALL: [Xmm; 16] = [
        Xmm::Xmm0,
        Xmm::Xmm1,
        Xmm::Xmm2,
        Xmm::Xmm3,
        Xmm::Xmm4,
        Xmm::Xmm5,
        Xmm::Xmm6,
        Xmm::Xmm7,
        Xmm::Xmm8,
        Xmm::Xmm9,
        Xmm::Xmm10,
        Xmm::Xmm11,
        Xmm::Xmm12,
        Xmm::Xmm13,
        Xmm::Xmm14,
        Xmm::Xmm15,
    ];

    /// Get the hardware encoding.
    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Get bits 0-2 for ModR/M encoding.
    #[inline(always)]
    pub const fn low_bits(self) -> u8 {
        self.encoding() & 0x7
    }

    /// Get bit 3 for REX prefix.
    #[inline(always)]
    pub const fn high_bit(self) -> bool {
        self.encoding() >= 8
    }

    /// Convert from encoding value if valid.
    #[inline]
    pub const fn from_encoding(enc: u8) -> Option<Xmm> {
        match enc {
            0 => Some(Xmm::Xmm0),
            1 => Some(Xmm::Xmm1),
            2 => Some(Xmm::Xmm2),
            3 => Some(Xmm::Xmm3),
            4 => Some(Xmm::Xmm4),
            5 => Some(Xmm::Xmm5),
            6 => Some(Xmm::Xmm6),
            7 => Some(Xmm::Xmm7),
            8 => Some(Xmm::Xmm8),
            9 => Some(Xmm::Xmm9),
            10 => Some(Xmm::Xmm10),
            11 => Some(Xmm::Xmm11),
            12 => Some(Xmm::Xmm12),
            13 => Some(Xmm::Xmm13),
            14 => Some(Xmm::Xmm14),
            15 => Some(Xmm::Xmm15),
            _ => None,
        }
    }
}

impl fmt::Display for Xmm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "xmm{}", self.encoding())
    }
}

// =============================================================================
// Register Sets (Bitfield)
// =============================================================================

/// A set of GPR registers using a 16-bit bitfield for O(1) operations.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct GprSet(u16);

impl GprSet {
    /// Empty register set.
    pub const EMPTY: GprSet = GprSet(0);

    /// All 16 registers.
    pub const ALL: GprSet = GprSet(0xFFFF);

    /// Create a set containing a single register.
    #[inline(always)]
    pub const fn singleton(reg: Gpr) -> Self {
        GprSet(1 << reg.encoding())
    }

    /// Create from a raw bitmask.
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        GprSet(bits)
    }

    /// Get the raw bitmask.
    #[inline(always)]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Check if the set contains a register.
    #[inline(always)]
    pub const fn contains(self, reg: Gpr) -> bool {
        (self.0 & (1 << reg.encoding())) != 0
    }

    /// Add a register to the set.
    #[inline(always)]
    pub const fn insert(self, reg: Gpr) -> Self {
        GprSet(self.0 | (1 << reg.encoding()))
    }

    /// Remove a register from the set.
    #[inline(always)]
    pub const fn remove(self, reg: Gpr) -> Self {
        GprSet(self.0 & !(1 << reg.encoding()))
    }

    /// Union of two sets.
    #[inline(always)]
    pub const fn union(self, other: GprSet) -> Self {
        GprSet(self.0 | other.0)
    }

    /// Intersection of two sets.
    #[inline(always)]
    pub const fn intersection(self, other: GprSet) -> Self {
        GprSet(self.0 & other.0)
    }

    /// Difference (self - other).
    #[inline(always)]
    pub const fn difference(self, other: GprSet) -> Self {
        GprSet(self.0 & !other.0)
    }

    /// Check if the set is empty.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Count the number of registers in the set.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Get the first register (lowest encoding) in the set, if any.
    #[inline]
    pub const fn first(self) -> Option<Gpr> {
        if self.0 == 0 {
            None
        } else {
            Gpr::from_encoding(self.0.trailing_zeros() as u8)
        }
    }

    /// Iterate over registers in the set (ascending order).
    pub fn iter(self) -> impl Iterator<Item = Gpr> {
        (0..16).filter_map(move |i| {
            if (self.0 & (1 << i)) != 0 {
                Gpr::from_encoding(i)
            } else {
                None
            }
        })
    }
}

impl fmt::Debug for GprSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GprSet{{")?;
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

/// A set of XMM registers using a 16-bit bitfield.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct XmmSet(u16);

impl XmmSet {
    /// Empty register set.
    pub const EMPTY: XmmSet = XmmSet(0);

    /// All 16 registers.
    pub const ALL: XmmSet = XmmSet(0xFFFF);

    /// Create a set containing a single register.
    #[inline(always)]
    pub const fn singleton(reg: Xmm) -> Self {
        XmmSet(1 << reg.encoding())
    }

    /// Check if the set contains a register.
    #[inline(always)]
    pub const fn contains(self, reg: Xmm) -> bool {
        (self.0 & (1 << reg.encoding())) != 0
    }

    /// Add a register to the set.
    #[inline(always)]
    pub const fn insert(self, reg: Xmm) -> Self {
        XmmSet(self.0 | (1 << reg.encoding()))
    }

    /// Remove a register from the set.
    #[inline(always)]
    pub const fn remove(self, reg: Xmm) -> Self {
        XmmSet(self.0 & !(1 << reg.encoding()))
    }

    /// Union of two sets.
    #[inline(always)]
    pub const fn union(self, other: XmmSet) -> Self {
        XmmSet(self.0 | other.0)
    }

    /// Difference (self - other).
    #[inline(always)]
    pub const fn difference(self, other: XmmSet) -> Self {
        XmmSet(self.0 & !other.0)
    }

    /// Count the number of registers in the set.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterate over registers in the set.
    pub fn iter(self) -> impl Iterator<Item = Xmm> {
        (0..16).filter_map(move |i| {
            if (self.0 & (1 << i)) != 0 {
                Xmm::from_encoding(i)
            } else {
                None
            }
        })
    }
}

// =============================================================================
// Calling Conventions
// =============================================================================

/// Calling convention definitions for x64.
///
/// Supports both Windows x64 and System V (Linux/macOS) ABIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallingConvention {
    /// Windows x64 ABI (Microsoft x64).
    WindowsX64,
    /// System V AMD64 ABI (Linux, macOS, BSD).
    SystemV,
}

impl CallingConvention {
    /// Detect the calling convention for the current platform.
    #[cfg(target_os = "windows")]
    pub const fn host() -> Self {
        CallingConvention::WindowsX64
    }

    #[cfg(not(target_os = "windows"))]
    pub const fn host() -> Self {
        CallingConvention::SystemV
    }

    /// Get integer argument registers in order.
    pub const fn int_arg_regs(self) -> &'static [Gpr] {
        match self {
            CallingConvention::WindowsX64 => &[Gpr::Rcx, Gpr::Rdx, Gpr::R8, Gpr::R9],
            CallingConvention::SystemV => {
                &[Gpr::Rdi, Gpr::Rsi, Gpr::Rdx, Gpr::Rcx, Gpr::R8, Gpr::R9]
            }
        }
    }

    /// Get floating-point argument registers in order.
    pub const fn float_arg_regs(self) -> &'static [Xmm] {
        match self {
            CallingConvention::WindowsX64 => &[Xmm::Xmm0, Xmm::Xmm1, Xmm::Xmm2, Xmm::Xmm3],
            CallingConvention::SystemV => &[
                Xmm::Xmm0,
                Xmm::Xmm1,
                Xmm::Xmm2,
                Xmm::Xmm3,
                Xmm::Xmm4,
                Xmm::Xmm5,
                Xmm::Xmm6,
                Xmm::Xmm7,
            ],
        }
    }

    /// Get the integer return register.
    pub const fn int_return_reg(self) -> Gpr {
        Gpr::Rax
    }

    /// Get the floating-point return register.
    pub const fn float_return_reg(self) -> Xmm {
        Xmm::Xmm0
    }

    /// Get volatile (caller-saved) GPRs.
    pub const fn volatile_gprs(self) -> GprSet {
        match self {
            CallingConvention::WindowsX64 => {
                // RAX, RCX, RDX, R8-R11
                GprSet::from_bits(0x0F07)
            }
            CallingConvention::SystemV => {
                // RAX, RCX, RDX, RSI, RDI, R8-R11
                GprSet::from_bits(0x0FC7)
            }
        }
    }

    /// Get non-volatile (callee-saved) GPRs.
    pub const fn callee_saved_gprs(self) -> GprSet {
        match self {
            CallingConvention::WindowsX64 => {
                // RBX, RBP, RDI, RSI, R12-R15
                GprSet::from_bits(0xF0F8)
            }
            CallingConvention::SystemV => {
                // RBX, RBP, R12-R15
                GprSet::from_bits(0xF028)
            }
        }
    }

    /// Get volatile (caller-saved) XMM registers.
    pub const fn volatile_xmms(self) -> XmmSet {
        match self {
            CallingConvention::WindowsX64 => {
                // XMM0-XMM5
                XmmSet::from_bits(0x003F)
            }
            CallingConvention::SystemV => {
                // All XMM0-XMM15 are volatile
                XmmSet::ALL
            }
        }
    }

    /// Get callee-saved XMM registers.
    pub const fn callee_saved_xmms(self) -> XmmSet {
        match self {
            CallingConvention::WindowsX64 => {
                // XMM6-XMM15
                XmmSet::from_bits(0xFFC0)
            }
            CallingConvention::SystemV => {
                // None
                XmmSet::EMPTY
            }
        }
    }

    /// Get the stack alignment requirement in bytes.
    pub const fn stack_alignment(self) -> usize {
        16 // Both ABIs require 16-byte alignment before CALL
    }

    /// Get the shadow space size (Windows only).
    pub const fn shadow_space(self) -> usize {
        match self {
            CallingConvention::WindowsX64 => 32, // 4 * 8 bytes
            CallingConvention::SystemV => 0,
        }
    }

    /// Get the red zone size (System V only).
    pub const fn red_zone(self) -> usize {
        match self {
            CallingConvention::WindowsX64 => 0,
            CallingConvention::SystemV => 128,
        }
    }
}

impl XmmSet {
    /// Create from a raw bitmask.
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        XmmSet(bits)
    }
}

// =============================================================================
// Allocatable Registers for JIT
// =============================================================================

/// Registers available for allocation by the JIT compiler.
///
/// Excludes RSP (stack pointer) and platform-specific reserved registers.
pub struct AllocatableRegs {
    /// GPRs available for general allocation.
    pub gprs: GprSet,
    /// XMM registers available for floating-point.
    pub xmms: XmmSet,
    /// Scratch register (always available, not allocated).
    pub scratch_gpr: Gpr,
    /// Scratch XMM register.
    pub scratch_xmm: Xmm,
}

impl AllocatableRegs {
    /// Get allocatable registers for the host calling convention.
    pub const fn for_host() -> Self {
        Self::for_convention(CallingConvention::host())
    }

    /// Get allocatable registers for a specific calling convention.
    pub const fn for_convention(_cc: CallingConvention) -> Self {
        // Exclude RSP (stack pointer) - never allocatable
        // Use R11 as scratch (volatile on both ABIs)
        // Use XMM15 as scratch on Windows, XMM15 on System V
        let all_gprs = GprSet::ALL.remove(Gpr::Rsp).remove(Gpr::R11);

        AllocatableRegs {
            gprs: all_gprs,
            xmms: XmmSet::ALL.remove(Xmm::Xmm15),
            scratch_gpr: Gpr::R11,
            scratch_xmm: Xmm::Xmm15,
        }
    }

    /// Get the number of allocatable GPRs.
    pub const fn gpr_count(&self) -> u32 {
        self.gprs.count()
    }

    /// Get the number of allocatable XMM registers.
    pub const fn xmm_count(&self) -> u32 {
        self.xmms.count()
    }
}

// =============================================================================
// Memory Operands
// =============================================================================

/// Scale factor for SIB addressing (1, 2, 4, or 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Scale {
    X1 = 0,
    X2 = 1,
    X4 = 2,
    X8 = 3,
}

impl Scale {
    /// Get the SIB scale encoding.
    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Get the actual scale value.
    #[inline(always)]
    pub const fn value(self) -> u8 {
        1 << (self as u8)
    }

    /// Create from a scale value if valid.
    pub const fn from_value(val: u8) -> Option<Scale> {
        match val {
            1 => Some(Scale::X1),
            2 => Some(Scale::X2),
            4 => Some(Scale::X4),
            8 => Some(Scale::X8),
            _ => None,
        }
    }
}

/// A memory operand for x64 addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemOperand {
    /// Base register (optional).
    pub base: Option<Gpr>,
    /// Index register (optional, cannot be RSP).
    pub index: Option<Gpr>,
    /// Scale factor for index.
    pub scale: Scale,
    /// Displacement (signed 32-bit).
    pub disp: i32,
}

impl MemOperand {
    /// Create a simple [base] addressing mode.
    #[inline]
    pub const fn base(reg: Gpr) -> Self {
        MemOperand {
            base: Some(reg),
            index: None,
            scale: Scale::X1,
            disp: 0,
        }
    }

    /// Create a [base + disp] addressing mode.
    #[inline]
    pub const fn base_disp(reg: Gpr, disp: i32) -> Self {
        MemOperand {
            base: Some(reg),
            index: None,
            scale: Scale::X1,
            disp,
        }
    }

    /// Create a [base + index*scale] addressing mode.
    #[inline]
    pub const fn base_index(base: Gpr, index: Gpr, scale: Scale) -> Self {
        MemOperand {
            base: Some(base),
            index: Some(index),
            scale,
            disp: 0,
        }
    }

    /// Create a [base + index*scale + disp] addressing mode.
    #[inline]
    pub const fn base_index_disp(base: Gpr, index: Gpr, scale: Scale, disp: i32) -> Self {
        MemOperand {
            base: Some(base),
            index: Some(index),
            scale,
            disp,
        }
    }

    /// Create a RIP-relative addressing mode [RIP + disp32].
    /// The displacement is relative to the end of the instruction.
    #[inline]
    pub const fn rip_relative(disp: i32) -> Self {
        MemOperand {
            base: None,
            index: None,
            scale: Scale::X1,
            disp,
        }
    }

    /// Check if this operand requires a SIB byte.
    #[inline]
    pub const fn needs_sib(&self) -> bool {
        self.index.is_some() || matches!(self.base, Some(b) if b.needs_sib_as_base())
    }

    /// Check if displacement fits in 8 bits.
    #[inline]
    pub const fn disp_fits_i8(&self) -> bool {
        self.disp >= -128 && self.disp <= 127
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpr_encoding() {
        assert_eq!(Gpr::Rax.encoding(), 0);
        assert_eq!(Gpr::Rcx.encoding(), 1);
        assert_eq!(Gpr::R8.encoding(), 8);
        assert_eq!(Gpr::R15.encoding(), 15);
    }

    #[test]
    fn test_gpr_low_high_bits() {
        assert_eq!(Gpr::Rax.low_bits(), 0);
        assert!(!Gpr::Rax.high_bit());

        assert_eq!(Gpr::R8.low_bits(), 0);
        assert!(Gpr::R8.high_bit());

        assert_eq!(Gpr::R15.low_bits(), 7);
        assert!(Gpr::R15.high_bit());
    }

    #[test]
    fn test_gpr_sib_requirements() {
        assert!(Gpr::Rsp.needs_sib_as_base());
        assert!(Gpr::R12.needs_sib_as_base());
        assert!(!Gpr::Rax.needs_sib_as_base());
    }

    #[test]
    fn test_gpr_displacement_requirements() {
        assert!(Gpr::Rbp.needs_displacement());
        assert!(Gpr::R13.needs_displacement());
        assert!(!Gpr::Rax.needs_displacement());
    }

    #[test]
    fn test_gpr_set_operations() {
        let set = GprSet::EMPTY
            .insert(Gpr::Rax)
            .insert(Gpr::Rcx)
            .insert(Gpr::R8);

        assert!(set.contains(Gpr::Rax));
        assert!(set.contains(Gpr::Rcx));
        assert!(set.contains(Gpr::R8));
        assert!(!set.contains(Gpr::Rdx));
        assert_eq!(set.count(), 3);

        let removed = set.remove(Gpr::Rcx);
        assert!(!removed.contains(Gpr::Rcx));
        assert_eq!(removed.count(), 2);
    }

    #[test]
    fn test_gpr_set_first() {
        let set = GprSet::EMPTY.insert(Gpr::Rdx).insert(Gpr::R8);
        assert_eq!(set.first(), Some(Gpr::Rdx));

        assert_eq!(GprSet::EMPTY.first(), None);
    }

    #[test]
    fn test_gpr_set_iter() {
        let set = GprSet::EMPTY
            .insert(Gpr::Rax)
            .insert(Gpr::Rdx)
            .insert(Gpr::R15);

        let regs: Vec<_> = set.iter().collect();
        assert_eq!(regs, vec![Gpr::Rax, Gpr::Rdx, Gpr::R15]);
    }

    #[test]
    fn test_calling_convention_host() {
        let cc = CallingConvention::host();

        // Verify we have a valid calling convention
        assert!(!cc.int_arg_regs().is_empty());
        assert!(!cc.float_arg_regs().is_empty());
        assert_eq!(cc.int_return_reg(), Gpr::Rax);
        assert_eq!(cc.float_return_reg(), Xmm::Xmm0);
    }

    #[test]
    fn test_windows_calling_convention() {
        let cc = CallingConvention::WindowsX64;

        assert_eq!(cc.int_arg_regs(), &[Gpr::Rcx, Gpr::Rdx, Gpr::R8, Gpr::R9]);
        assert_eq!(cc.shadow_space(), 32);
        assert_eq!(cc.red_zone(), 0);
    }

    #[test]
    fn test_sysv_calling_convention() {
        let cc = CallingConvention::SystemV;

        assert_eq!(
            cc.int_arg_regs(),
            &[Gpr::Rdi, Gpr::Rsi, Gpr::Rdx, Gpr::Rcx, Gpr::R8, Gpr::R9]
        );
        assert_eq!(cc.shadow_space(), 0);
        assert_eq!(cc.red_zone(), 128);
    }

    #[test]
    fn test_allocatable_regs() {
        let alloc = AllocatableRegs::for_host();

        // RSP should never be allocatable
        assert!(!alloc.gprs.contains(Gpr::Rsp));
        // Scratch register should not be allocatable
        assert!(!alloc.gprs.contains(alloc.scratch_gpr));

        // Should have at least 10 allocatable GPRs
        assert!(alloc.gpr_count() >= 10);
    }

    #[test]
    fn test_scale() {
        assert_eq!(Scale::X1.value(), 1);
        assert_eq!(Scale::X2.value(), 2);
        assert_eq!(Scale::X4.value(), 4);
        assert_eq!(Scale::X8.value(), 8);

        assert_eq!(Scale::from_value(4), Some(Scale::X4));
        assert_eq!(Scale::from_value(3), None);
    }

    #[test]
    fn test_mem_operand() {
        let mem = MemOperand::base_disp(Gpr::Rbp, -16);
        assert_eq!(mem.base, Some(Gpr::Rbp));
        assert_eq!(mem.disp, -16);
        assert!(mem.disp_fits_i8());

        let mem_large = MemOperand::base_disp(Gpr::Rax, 1000);
        assert!(!mem_large.disp_fits_i8());

        let mem_sib = MemOperand::base_index(Gpr::Rax, Gpr::Rcx, Scale::X8);
        assert!(mem_sib.needs_sib());
    }

    #[test]
    fn test_xmm_registers() {
        assert_eq!(Xmm::Xmm0.encoding(), 0);
        assert_eq!(Xmm::Xmm8.encoding(), 8);
        assert!(Xmm::Xmm8.high_bit());
        assert!(!Xmm::Xmm7.high_bit());
    }

    #[test]
    fn test_xmm_set() {
        let set = XmmSet::EMPTY.insert(Xmm::Xmm0).insert(Xmm::Xmm8);

        assert!(set.contains(Xmm::Xmm0));
        assert!(set.contains(Xmm::Xmm8));
        assert!(!set.contains(Xmm::Xmm1));
        assert_eq!(set.count(), 2);
    }
}
