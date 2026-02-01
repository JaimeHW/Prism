//! Complete x64 instruction encoder.
//!
//! This module provides low-level encoding for all x64 instructions needed
//! by the JIT compiler. It handles:
//! - REX prefix generation
//! - ModR/M and SIB byte encoding
//! - Displacement encoding (8-bit and 32-bit)
//! - RIP-relative addressing
//!
//! # Encoding Reference
//! x64 instruction format:
//! ```text
//! [Prefixes] [REX] [Opcode] [ModR/M] [SIB] [Disp] [Imm]
//! ```
//!
//! # Performance Notes
//! - All encoding functions are `#[inline]` for zero-cost abstraction
//! - Fixed-size arrays avoid heap allocation
//! - Const functions enable compile-time evaluation where possible

use super::registers::{Gpr, MemOperand, Scale, Xmm};

// =============================================================================
// REX Prefix
// =============================================================================

/// REX prefix byte.
///
/// Format: 0100WRXB
/// - W: 64-bit operand size
/// - R: Extension of ModR/M reg field
/// - X: Extension of SIB index field
/// - B: Extension of ModR/M r/m field or SIB base field
#[derive(Debug, Clone, Copy)]
pub struct Rex {
    pub w: bool, // 64-bit operand
    pub r: bool, // Extends reg in ModR/M
    pub x: bool, // Extends index in SIB
    pub b: bool, // Extends r/m in ModR/M or base in SIB
}

impl Rex {
    /// Create a REX prefix for register-register operations.
    #[inline]
    pub const fn rr(w: bool, reg: Gpr, rm: Gpr) -> Self {
        Rex {
            w,
            r: reg.high_bit(),
            x: false,
            b: rm.high_bit(),
        }
    }

    /// Create a REX prefix for XMM register-register operations.
    #[inline]
    pub const fn xmm_rr(reg: Xmm, rm: Xmm) -> Self {
        Rex {
            w: false,
            r: reg.high_bit(),
            x: false,
            b: rm.high_bit(),
        }
    }

    /// Create a REX prefix for register-memory operations.
    #[inline]
    pub const fn rm(w: bool, reg: Gpr, mem: &MemOperand) -> Self {
        Rex {
            w,
            r: reg.high_bit(),
            x: match mem.index {
                Some(idx) => idx.high_bit(),
                None => false,
            },
            b: match mem.base {
                Some(base) => base.high_bit(),
                None => false,
            },
        }
    }

    /// Check if this REX prefix is needed (non-default bits set).
    #[inline]
    pub const fn is_needed(&self) -> bool {
        self.w || self.r || self.x || self.b
    }

    /// Encode the REX prefix byte.
    #[inline]
    pub const fn encode(&self) -> u8 {
        0x40 | ((self.w as u8) << 3)
            | ((self.r as u8) << 2)
            | ((self.x as u8) << 1)
            | (self.b as u8)
    }
}

// =============================================================================
// ModR/M Byte
// =============================================================================

/// Mod field values for ModR/M.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Mod {
    /// [reg] or [disp32] or [SIB] depending on r/m
    Indirect = 0b00,
    /// [reg + disp8]
    IndirectDisp8 = 0b01,
    /// [reg + disp32]
    IndirectDisp32 = 0b10,
    /// reg (direct)
    Direct = 0b11,
}

/// Encode a ModR/M byte.
#[inline]
pub const fn modrm(mod_: Mod, reg: u8, rm: u8) -> u8 {
    ((mod_ as u8) << 6) | ((reg & 0x7) << 3) | (rm & 0x7)
}

// =============================================================================
// SIB Byte
// =============================================================================

/// Encode a SIB byte.
#[inline]
pub const fn sib(scale: Scale, index: u8, base: u8) -> u8 {
    ((scale as u8) << 6) | ((index & 0x7) << 3) | (base & 0x7)
}

// =============================================================================
// Instruction Encoder
// =============================================================================

/// Maximum encoded instruction length.
pub const MAX_INST_LEN: usize = 15;

/// Instruction encoding buffer.
#[derive(Debug, Clone, Copy)]
pub struct EncodedInst {
    bytes: [u8; MAX_INST_LEN],
    len: u8,
}

impl EncodedInst {
    /// Create an empty encoding buffer.
    #[inline]
    pub const fn new() -> Self {
        EncodedInst {
            bytes: [0; MAX_INST_LEN],
            len: 0,
        }
    }

    /// Get the encoded bytes.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Get the length of the encoded instruction.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Check if empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Push a byte.
    #[inline]
    fn push(&mut self, byte: u8) {
        debug_assert!((self.len as usize) < MAX_INST_LEN);
        self.bytes[self.len as usize] = byte;
        self.len += 1;
    }

    /// Push a u16 (little-endian).
    #[inline]
    fn push_u16(&mut self, val: u16) {
        let bytes = val.to_le_bytes();
        self.push(bytes[0]);
        self.push(bytes[1]);
    }

    /// Push a u32 (little-endian).
    #[inline]
    fn push_u32(&mut self, val: u32) {
        let bytes = val.to_le_bytes();
        self.push(bytes[0]);
        self.push(bytes[1]);
        self.push(bytes[2]);
        self.push(bytes[3]);
    }

    /// Push a u64 (little-endian).
    #[inline]
    fn push_u64(&mut self, val: u64) {
        let bytes = val.to_le_bytes();
        for b in bytes {
            self.push(b);
        }
    }

    /// Push displacement (8 or 32 bit based on value).
    #[inline]
    #[allow(dead_code)]
    fn push_disp(&mut self, disp: i32, use_8bit: bool) {
        if use_8bit {
            self.push(disp as i8 as u8);
        } else {
            self.push_u32(disp as u32);
        }
    }
}

impl Default for EncodedInst {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Encoder Functions - Register-Register
// =============================================================================

/// Encode a register-to-register instruction with form: OP r/m64, r64
#[inline]
pub fn encode_rr(opcode: u8, dst: Gpr, src: Gpr, w: bool) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex::rr(w, src, dst);
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, src.low_bits(), dst.low_bits()));
    enc
}

/// Encode a register-to-register instruction with two-byte opcode (0F prefix).
#[inline]
pub fn encode_rr_0f(opcode: u8, dst: Gpr, src: Gpr, w: bool) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex::rr(w, dst, src);
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(0x0F);
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

// =============================================================================
// Encoder Functions - Register-Immediate
// =============================================================================

/// Encode a register-immediate instruction: OP r/m64, imm8 (opcode /digit form)
#[inline]
pub fn encode_ri8(opcode: u8, digit: u8, dst: Gpr, imm: i8, w: bool) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, digit, dst.low_bits()));
    enc.push(imm as u8);
    enc
}

/// Encode a register-immediate instruction: OP r/m64, imm32 (opcode /digit form)
#[inline]
pub fn encode_ri32(opcode: u8, digit: u8, dst: Gpr, imm: i32, w: bool) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, digit, dst.low_bits()));
    enc.push_u32(imm as u32);
    enc
}

/// Encode MOV r64, imm64 (REX.W + B8 + rd).
#[inline]
pub fn encode_mov_ri64(dst: Gpr, imm: i64) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xB8 + dst.low_bits());
    enc.push_u64(imm as u64);
    enc
}

/// Encode MOV r32, imm32 (B8 + rd).
#[inline]
pub fn encode_mov_ri32(dst: Gpr, imm: u32) -> EncodedInst {
    let mut enc = EncodedInst::new();
    if dst.high_bit() {
        enc.push(
            Rex {
                w: false,
                r: false,
                x: false,
                b: true,
            }
            .encode(),
        );
    }
    enc.push(0xB8 + dst.low_bits());
    enc.push_u32(imm);
    enc
}

// =============================================================================
// Encoder Functions - Memory Operations
// =============================================================================

/// Encode a register-memory instruction: OP r64, [mem]
#[inline]
pub fn encode_rm(opcode: u8, reg: Gpr, mem: &MemOperand, w: bool) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex::rm(w, reg, mem);
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(opcode);
    encode_modrm_sib_disp(&mut enc, reg.low_bits(), mem);
    enc
}

/// Encode a memory-register instruction: OP [mem], r64
#[inline]
pub fn encode_mr(opcode: u8, mem: &MemOperand, reg: Gpr, w: bool) -> EncodedInst {
    encode_rm(opcode, reg, mem, w)
}

/// Encode ModR/M, optional SIB, and displacement for memory operand.
fn encode_modrm_sib_disp(enc: &mut EncodedInst, reg: u8, mem: &MemOperand) {
    match (mem.base, mem.index) {
        // RIP-relative: [RIP + disp32]
        (None, None) => {
            enc.push(modrm(Mod::Indirect, reg, 0b101));
            enc.push_u32(mem.disp as u32);
        }

        // [base] or [base + disp]
        (Some(base), None) => {
            let needs_sib = base.needs_sib_as_base();
            let needs_disp = base.needs_displacement() && mem.disp == 0;

            let (mod_field, _disp8) = if mem.disp == 0 && !needs_disp {
                (Mod::Indirect, false)
            } else if mem.disp_fits_i8() {
                (Mod::IndirectDisp8, true)
            } else {
                (Mod::IndirectDisp32, false)
            };

            if needs_sib {
                enc.push(modrm(mod_field, reg, 0b100));
                // SIB with no index: scale=00, index=RSP(100), base
                enc.push(sib(Scale::X1, 0b100, base.low_bits()));
            } else {
                enc.push(modrm(mod_field, reg, base.low_bits()));
            }

            match mod_field {
                Mod::IndirectDisp8 => enc.push(mem.disp as i8 as u8),
                Mod::IndirectDisp32 => enc.push_u32(mem.disp as u32),
                _ if needs_disp => enc.push(0), // RBP/R13 needs at least disp8=0
                _ => {}
            }
        }

        // [base + index*scale] or [base + index*scale + disp]
        (Some(base), Some(index)) => {
            debug_assert!(
                index.low_bits() != 0b100,
                "RSP cannot be used as index register"
            );

            let mod_field = if mem.disp == 0 && !base.needs_displacement() {
                Mod::Indirect
            } else if mem.disp_fits_i8() {
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

        // [index*scale + disp32] (no base)
        (None, Some(index)) => {
            debug_assert!(
                index.low_bits() != 0b100,
                "RSP cannot be used as index register"
            );
            enc.push(modrm(Mod::Indirect, reg, 0b100));
            // SIB with no base: base=RBP(101) with mod=00 means [disp32]
            enc.push(sib(mem.scale, index.low_bits(), 0b101));
            enc.push_u32(mem.disp as u32);
        }
    }
}

// =============================================================================
// Specific Instructions - Arithmetic
// =============================================================================

/// ADD r64, r64
#[inline]
pub fn encode_add_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x01, dst, src, true)
}

/// ADD r64, imm8 (sign-extended)
#[inline]
pub fn encode_add_ri8(dst: Gpr, imm: i8) -> EncodedInst {
    encode_ri8(0x83, 0, dst, imm, true)
}

/// ADD r64, imm32 (sign-extended)
#[inline]
pub fn encode_add_ri32(dst: Gpr, imm: i32) -> EncodedInst {
    if imm >= -128 && imm <= 127 {
        encode_add_ri8(dst, imm as i8)
    } else {
        encode_ri32(0x81, 0, dst, imm, true)
    }
}

/// ADD r64, [mem]
#[inline]
pub fn encode_add_rm(dst: Gpr, mem: &MemOperand) -> EncodedInst {
    encode_rm(0x03, dst, mem, true)
}

/// SUB r64, r64
#[inline]
pub fn encode_sub_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x29, dst, src, true)
}

/// SUB r64, imm8 (sign-extended)
#[inline]
pub fn encode_sub_ri8(dst: Gpr, imm: i8) -> EncodedInst {
    encode_ri8(0x83, 5, dst, imm, true)
}

/// SUB r64, imm32 (sign-extended)
#[inline]
pub fn encode_sub_ri32(dst: Gpr, imm: i32) -> EncodedInst {
    if imm >= -128 && imm <= 127 {
        encode_sub_ri8(dst, imm as i8)
    } else {
        encode_ri32(0x81, 5, dst, imm, true)
    }
}

/// IMUL r64, r64
#[inline]
pub fn encode_imul_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr_0f(0xAF, dst, src, true)
}

/// IMUL r64, r64, imm8
#[inline]
pub fn encode_imul_rri8(dst: Gpr, src: Gpr, imm: i8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex::rr(true, dst, src);
    enc.push(rex.encode());
    enc.push(0x6B);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc.push(imm as u8);
    enc
}

/// IMUL r64, r64, imm32
#[inline]
pub fn encode_imul_rri32(dst: Gpr, src: Gpr, imm: i32) -> EncodedInst {
    if imm >= -128 && imm <= 127 {
        encode_imul_rri8(dst, src, imm as i8)
    } else {
        let mut enc = EncodedInst::new();
        let rex = Rex::rr(true, dst, src);
        enc.push(rex.encode());
        enc.push(0x69);
        enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
        enc.push_u32(imm as u32);
        enc
    }
}

/// IDIV r64 (divide RDX:RAX by r64)
#[inline]
pub fn encode_idiv(src: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: src.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xF7);
    enc.push(modrm(Mod::Direct, 7, src.low_bits()));
    enc
}

/// DIV r64 (unsigned divide RDX:RAX by r64)
#[inline]
pub fn encode_div(src: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: src.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xF7);
    enc.push(modrm(Mod::Direct, 6, src.low_bits()));
    enc
}

/// NEG r64
#[inline]
pub fn encode_neg(dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xF7);
    enc.push(modrm(Mod::Direct, 3, dst.low_bits()));
    enc
}

/// INC r64
#[inline]
pub fn encode_inc(dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xFF);
    enc.push(modrm(Mod::Direct, 0, dst.low_bits()));
    enc
}

/// DEC r64
#[inline]
pub fn encode_dec(dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xFF);
    enc.push(modrm(Mod::Direct, 1, dst.low_bits()));
    enc
}

// =============================================================================
// Specific Instructions - Bitwise
// =============================================================================

/// AND r64, r64
#[inline]
pub fn encode_and_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x21, dst, src, true)
}

/// AND r64, imm32 (sign-extended)
#[inline]
pub fn encode_and_ri32(dst: Gpr, imm: i32) -> EncodedInst {
    if imm >= -128 && imm <= 127 {
        encode_ri8(0x83, 4, dst, imm as i8, true)
    } else {
        encode_ri32(0x81, 4, dst, imm, true)
    }
}

/// OR r64, r64
#[inline]
pub fn encode_or_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x09, dst, src, true)
}

/// OR r64, imm32
#[inline]
pub fn encode_or_ri32(dst: Gpr, imm: i32) -> EncodedInst {
    if imm >= -128 && imm <= 127 {
        encode_ri8(0x83, 1, dst, imm as i8, true)
    } else {
        encode_ri32(0x81, 1, dst, imm, true)
    }
}

/// XOR r64, r64
#[inline]
pub fn encode_xor_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x31, dst, src, true)
}

/// XOR r64, imm32
#[inline]
pub fn encode_xor_ri32(dst: Gpr, imm: i32) -> EncodedInst {
    if imm >= -128 && imm <= 127 {
        encode_ri8(0x83, 6, dst, imm as i8, true)
    } else {
        encode_ri32(0x81, 6, dst, imm, true)
    }
}

/// NOT r64
#[inline]
pub fn encode_not(dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xF7);
    enc.push(modrm(Mod::Direct, 2, dst.low_bits()));
    enc
}

/// SHL r64, imm8
#[inline]
pub fn encode_shl_ri(dst: Gpr, imm: u8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    if imm == 1 {
        enc.push(0xD1);
        enc.push(modrm(Mod::Direct, 4, dst.low_bits()));
    } else {
        enc.push(0xC1);
        enc.push(modrm(Mod::Direct, 4, dst.low_bits()));
        enc.push(imm);
    }
    enc
}

/// SHL r64, cl
#[inline]
pub fn encode_shl_cl(dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xD3);
    enc.push(modrm(Mod::Direct, 4, dst.low_bits()));
    enc
}

/// SHR r64, imm8
#[inline]
pub fn encode_shr_ri(dst: Gpr, imm: u8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    if imm == 1 {
        enc.push(0xD1);
        enc.push(modrm(Mod::Direct, 5, dst.low_bits()));
    } else {
        enc.push(0xC1);
        enc.push(modrm(Mod::Direct, 5, dst.low_bits()));
        enc.push(imm);
    }
    enc
}

/// SHR r64, cl
#[inline]
pub fn encode_shr_cl(dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xD3);
    enc.push(modrm(Mod::Direct, 5, dst.low_bits()));
    enc
}

/// SAR r64, imm8 (arithmetic shift right)
#[inline]
pub fn encode_sar_ri(dst: Gpr, imm: u8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    if imm == 1 {
        enc.push(0xD1);
        enc.push(modrm(Mod::Direct, 7, dst.low_bits()));
    } else {
        enc.push(0xC1);
        enc.push(modrm(Mod::Direct, 7, dst.low_bits()));
        enc.push(imm);
    }
    enc
}

/// SAR r64, cl
#[inline]
pub fn encode_sar_cl(dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0xD3);
    enc.push(modrm(Mod::Direct, 7, dst.low_bits()));
    enc
}

// =============================================================================
// Specific Instructions - Comparison
// =============================================================================

/// CMP r64, r64
#[inline]
pub fn encode_cmp_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x39, dst, src, true)
}

/// CMP r64, imm8 (sign-extended)
#[inline]
pub fn encode_cmp_ri8(dst: Gpr, imm: i8) -> EncodedInst {
    encode_ri8(0x83, 7, dst, imm, true)
}

/// CMP r64, imm32 (sign-extended)
#[inline]
pub fn encode_cmp_ri32(dst: Gpr, imm: i32) -> EncodedInst {
    if imm >= -128 && imm <= 127 {
        encode_cmp_ri8(dst, imm as i8)
    } else {
        encode_ri32(0x81, 7, dst, imm, true)
    }
}

/// CMP r64, [mem]
#[inline]
pub fn encode_cmp_rm(dst: Gpr, mem: &MemOperand) -> EncodedInst {
    encode_rm(0x3B, dst, mem, true)
}

/// TEST r64, r64
#[inline]
pub fn encode_test_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x85, dst, src, true)
}

/// TEST r64, imm32
#[inline]
pub fn encode_test_ri32(dst: Gpr, imm: i32) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let rex = Rex {
        w: true,
        r: false,
        x: false,
        b: dst.high_bit(),
    };
    enc.push(rex.encode());
    if dst == Gpr::Rax {
        enc.push(0xA9);
    } else {
        enc.push(0xF7);
        enc.push(modrm(Mod::Direct, 0, dst.low_bits()));
    }
    enc.push_u32(imm as u32);
    enc
}

// =============================================================================
// Specific Instructions - Data Movement
// =============================================================================

/// MOV r64, r64
#[inline]
pub fn encode_mov_rr(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x89, dst, src, true)
}

/// MOV r32, r32 (32-bit, zero-extends to 64-bit)
#[inline]
pub fn encode_mov_rr32(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x89, dst, src, false)
}

/// MOV r64, [mem]
#[inline]
pub fn encode_mov_rm(dst: Gpr, mem: &MemOperand) -> EncodedInst {
    encode_rm(0x8B, dst, mem, true)
}

/// MOV [mem], r64
#[inline]
pub fn encode_mov_mr(mem: &MemOperand, src: Gpr) -> EncodedInst {
    encode_rm(0x89, src, mem, true)
}

/// MOV r32, [mem] (32-bit load, zero-extends)
#[inline]
pub fn encode_mov_rm32(dst: Gpr, mem: &MemOperand) -> EncodedInst {
    encode_rm(0x8B, dst, mem, false)
}

/// MOV [mem], r32 (32-bit store)
#[inline]
pub fn encode_mov_mr32(mem: &MemOperand, src: Gpr) -> EncodedInst {
    encode_rm(0x89, src, mem, false)
}

/// LEA r64, [mem]
#[inline]
pub fn encode_lea(dst: Gpr, mem: &MemOperand) -> EncodedInst {
    encode_rm(0x8D, dst, mem, true)
}

/// MOVZX r64, r/m8 (zero-extend byte to 64-bit)
#[inline]
pub fn encode_movzx_rb(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr_0f(0xB6, dst, src, true)
}

/// MOVZX r64, r/m16 (zero-extend word to 64-bit)
#[inline]
pub fn encode_movzx_rw(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr_0f(0xB7, dst, src, true)
}

/// MOVSX r64, r/m8 (sign-extend byte to 64-bit)
#[inline]
pub fn encode_movsx_rb(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr_0f(0xBE, dst, src, true)
}

/// MOVSX r64, r/m16 (sign-extend word to 64-bit)
#[inline]
pub fn encode_movsx_rw(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr_0f(0xBF, dst, src, true)
}

/// MOVSXD r64, r/m32 (sign-extend dword to 64-bit)
#[inline]
pub fn encode_movsxd(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_rr(0x63, dst, src, true)
}

/// CDQ - Sign-extend EAX into EDX:EAX
#[inline]
pub fn encode_cdq() -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0x99);
    enc
}

/// CQO - Sign-extend RAX into RDX:RAX
#[inline]
pub fn encode_cqo() -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0x48); // REX.W
    enc.push(0x99);
    enc
}

// =============================================================================
// Specific Instructions - Stack
// =============================================================================

/// PUSH r64
#[inline]
pub fn encode_push(src: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    if src.high_bit() {
        enc.push(
            Rex {
                w: false,
                r: false,
                x: false,
                b: true,
            }
            .encode(),
        );
    }
    enc.push(0x50 + src.low_bits());
    enc
}

/// POP r64
#[inline]
pub fn encode_pop(dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    if dst.high_bit() {
        enc.push(
            Rex {
                w: false,
                r: false,
                x: false,
                b: true,
            }
            .encode(),
        );
    }
    enc.push(0x58 + dst.low_bits());
    enc
}

// =============================================================================
// Specific Instructions - Control Flow
// =============================================================================

/// JMP rel8
#[inline]
pub fn encode_jmp_rel8(offset: i8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xEB);
    enc.push(offset as u8);
    enc
}

/// JMP rel32
#[inline]
pub fn encode_jmp_rel32(offset: i32) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xE9);
    enc.push_u32(offset as u32);
    enc
}

/// JMP r64 (indirect)
#[inline]
pub fn encode_jmp_r(target: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    if target.high_bit() {
        enc.push(
            Rex {
                w: false,
                r: false,
                x: false,
                b: true,
            }
            .encode(),
        );
    }
    enc.push(0xFF);
    enc.push(modrm(Mod::Direct, 4, target.low_bits()));
    enc
}

/// Condition codes for Jcc instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Condition {
    Overflow = 0x0,     // JO
    NoOverflow = 0x1,   // JNO
    Below = 0x2,        // JB/JNAE/JC
    AboveEqual = 0x3,   // JAE/JNB/JNC
    Equal = 0x4,        // JE/JZ
    NotEqual = 0x5,     // JNE/JNZ
    BelowEqual = 0x6,   // JBE/JNA
    Above = 0x7,        // JA/JNBE
    Sign = 0x8,         // JS
    NoSign = 0x9,       // JNS
    Parity = 0xA,       // JP/JPE
    NoParity = 0xB,     // JNP/JPO
    Less = 0xC,         // JL/JNGE
    GreaterEqual = 0xD, // JGE/JNL
    LessEqual = 0xE,    // JLE/JNG
    Greater = 0xF,      // JG/JNLE
}

impl Condition {
    /// Get the inverted condition.
    #[inline]
    pub const fn invert(self) -> Condition {
        match self {
            Condition::Overflow => Condition::NoOverflow,
            Condition::NoOverflow => Condition::Overflow,
            Condition::Below => Condition::AboveEqual,
            Condition::AboveEqual => Condition::Below,
            Condition::Equal => Condition::NotEqual,
            Condition::NotEqual => Condition::Equal,
            Condition::BelowEqual => Condition::Above,
            Condition::Above => Condition::BelowEqual,
            Condition::Sign => Condition::NoSign,
            Condition::NoSign => Condition::Sign,
            Condition::Parity => Condition::NoParity,
            Condition::NoParity => Condition::Parity,
            Condition::Less => Condition::GreaterEqual,
            Condition::GreaterEqual => Condition::Less,
            Condition::LessEqual => Condition::Greater,
            Condition::Greater => Condition::LessEqual,
        }
    }
}

/// Jcc rel8 (conditional jump, 8-bit offset)
#[inline]
pub fn encode_jcc_rel8(cond: Condition, offset: i8) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0x70 + cond as u8);
    enc.push(offset as u8);
    enc
}

/// Jcc rel32 (conditional jump, 32-bit offset)
#[inline]
pub fn encode_jcc_rel32(cond: Condition, offset: i32) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0x0F);
    enc.push(0x80 + cond as u8);
    enc.push_u32(offset as u32);
    enc
}

/// SETcc r/m8
#[inline]
pub fn encode_setcc(cond: Condition, dst: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    // Need REX for R8-R15 or to access low byte of RSI/RDI/RBP/RSP
    if dst.high_bit() || matches!(dst, Gpr::Rsp | Gpr::Rbp | Gpr::Rsi | Gpr::Rdi) {
        enc.push(
            Rex {
                w: false,
                r: false,
                x: false,
                b: dst.high_bit(),
            }
            .encode(),
        );
    }
    enc.push(0x0F);
    enc.push(0x90 + cond as u8);
    enc.push(modrm(Mod::Direct, 0, dst.low_bits()));
    enc
}

/// CALL rel32
#[inline]
pub fn encode_call_rel32(offset: i32) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xE8);
    enc.push_u32(offset as u32);
    enc
}

/// CALL r64 (indirect)
#[inline]
pub fn encode_call_r(target: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    if target.high_bit() {
        enc.push(
            Rex {
                w: false,
                r: false,
                x: false,
                b: true,
            }
            .encode(),
        );
    }
    enc.push(0xFF);
    enc.push(modrm(Mod::Direct, 2, target.low_bits()));
    enc
}

/// RET
#[inline]
pub fn encode_ret() -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xC3);
    enc
}

/// RET imm16 (pop imm16 bytes from stack after return)
#[inline]
pub fn encode_ret_imm(imm: u16) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xC2);
    enc.push_u16(imm);
    enc
}

/// NOP
#[inline]
pub fn encode_nop() -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0x90);
    enc
}

/// INT3 (breakpoint)
#[inline]
pub fn encode_int3() -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xCC);
    enc
}

/// UD2 (undefined instruction, for unreachable code)
#[inline]
pub fn encode_ud2() -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0x0F);
    enc.push(0x0B);
    enc
}

// =============================================================================
// Specific Instructions - SSE Floating Point
// =============================================================================

/// MOVSD xmm, xmm (move scalar double)
#[inline]
pub fn encode_movsd_rr(dst: Xmm, src: Xmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xF2); // REPNE prefix
    let rex = Rex::xmm_rr(dst, src);
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(0x0F);
    enc.push(0x10);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// MOVSD xmm, [mem]
#[inline]
pub fn encode_movsd_rm(dst: Xmm, mem: &MemOperand) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xF2);
    let rex = Rex {
        w: false,
        r: dst.high_bit(),
        x: mem.index.map(|i| i.high_bit()).unwrap_or(false),
        b: mem.base.map(|b| b.high_bit()).unwrap_or(false),
    };
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(0x0F);
    enc.push(0x10);
    encode_modrm_sib_disp(&mut enc, dst.low_bits(), mem);
    enc
}

/// MOVSD [mem], xmm
#[inline]
pub fn encode_movsd_mr(mem: &MemOperand, src: Xmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xF2);
    let rex = Rex {
        w: false,
        r: src.high_bit(),
        x: mem.index.map(|i| i.high_bit()).unwrap_or(false),
        b: mem.base.map(|b| b.high_bit()).unwrap_or(false),
    };
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(0x0F);
    enc.push(0x11);
    encode_modrm_sib_disp(&mut enc, src.low_bits(), mem);
    enc
}

/// ADDSD xmm, xmm
#[inline]
pub fn encode_addsd(dst: Xmm, src: Xmm) -> EncodedInst {
    encode_sse_binop(0x58, dst, src)
}

/// SUBSD xmm, xmm
#[inline]
pub fn encode_subsd(dst: Xmm, src: Xmm) -> EncodedInst {
    encode_sse_binop(0x5C, dst, src)
}

/// MULSD xmm, xmm
#[inline]
pub fn encode_mulsd(dst: Xmm, src: Xmm) -> EncodedInst {
    encode_sse_binop(0x59, dst, src)
}

/// DIVSD xmm, xmm
#[inline]
pub fn encode_divsd(dst: Xmm, src: Xmm) -> EncodedInst {
    encode_sse_binop(0x5E, dst, src)
}

/// SQRTSD xmm, xmm
#[inline]
pub fn encode_sqrtsd(dst: Xmm, src: Xmm) -> EncodedInst {
    encode_sse_binop(0x51, dst, src)
}

/// Generic SSE scalar double binary operation.
#[inline]
fn encode_sse_binop(opcode: u8, dst: Xmm, src: Xmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xF2);
    let rex = Rex::xmm_rr(dst, src);
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(0x0F);
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// UCOMISD xmm, xmm (unordered compare)
#[inline]
pub fn encode_ucomisd(dst: Xmm, src: Xmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0x66);
    let rex = Rex::xmm_rr(dst, src);
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(0x0F);
    enc.push(0x2E);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// CVTSI2SD xmm, r64 (convert int64 to double)
#[inline]
pub fn encode_cvtsi2sd(dst: Xmm, src: Gpr) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xF2);
    let rex = Rex {
        w: true, // 64-bit integer
        r: dst.high_bit(),
        x: false,
        b: src.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0x0F);
    enc.push(0x2A);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// CVTTSD2SI r64, xmm (convert double to int64 with truncation)
#[inline]
pub fn encode_cvttsd2si(dst: Gpr, src: Xmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0xF2);
    let rex = Rex {
        w: true, // 64-bit integer
        r: dst.high_bit(),
        x: false,
        b: src.high_bit(),
    };
    enc.push(rex.encode());
    enc.push(0x0F);
    enc.push(0x2C);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// XORPD xmm, xmm (XOR packed doubles, useful for zeroing)
#[inline]
pub fn encode_xorpd(dst: Xmm, src: Xmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    enc.push(0x66);
    let rex = Rex::xmm_rr(dst, src);
    if rex.is_needed() {
        enc.push(rex.encode());
    }
    enc.push(0x0F);
    enc.push(0x57);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rex_encoding() {
        assert_eq!(
            Rex {
                w: true,
                r: false,
                x: false,
                b: false
            }
            .encode(),
            0x48
        );
        assert_eq!(
            Rex {
                w: true,
                r: true,
                x: false,
                b: true
            }
            .encode(),
            0x4D
        );
        assert_eq!(
            Rex {
                w: false,
                r: false,
                x: false,
                b: true
            }
            .encode(),
            0x41
        );
    }

    #[test]
    fn test_modrm_encoding() {
        assert_eq!(modrm(Mod::Direct, 3, 0), 0xD8);
        assert_eq!(modrm(Mod::Indirect, 0, 5), 0x05);
        assert_eq!(modrm(Mod::IndirectDisp8, 1, 2), 0x4A);
    }

    #[test]
    fn test_mov_rr() {
        let enc = encode_mov_rr(Gpr::Rax, Gpr::Rbx);
        assert_eq!(enc.as_slice(), &[0x48, 0x89, 0xD8]);

        let enc = encode_mov_rr(Gpr::R8, Gpr::R9);
        assert_eq!(enc.as_slice(), &[0x4D, 0x89, 0xC8]);
    }

    #[test]
    fn test_mov_ri64() {
        let enc = encode_mov_ri64(Gpr::Rax, 0x123456789ABCDEF0u64 as i64);
        assert_eq!(enc.len(), 10);
        assert_eq!(enc.as_slice()[0], 0x48); // REX.W
        assert_eq!(enc.as_slice()[1], 0xB8); // MOV RAX, imm64
    }

    #[test]
    fn test_add_rr() {
        let enc = encode_add_rr(Gpr::Rax, Gpr::Rcx);
        assert_eq!(enc.as_slice(), &[0x48, 0x01, 0xC8]);
    }

    #[test]
    fn test_add_ri8() {
        let enc = encode_add_ri8(Gpr::Rax, 5);
        assert_eq!(enc.as_slice(), &[0x48, 0x83, 0xC0, 0x05]);
    }

    #[test]
    fn test_sub_rr() {
        let enc = encode_sub_rr(Gpr::Rcx, Gpr::Rdx);
        assert_eq!(enc.as_slice(), &[0x48, 0x29, 0xD1]);
    }

    #[test]
    fn test_imul_rr() {
        let enc = encode_imul_rr(Gpr::Rax, Gpr::Rcx);
        assert_eq!(enc.as_slice(), &[0x48, 0x0F, 0xAF, 0xC1]);
    }

    #[test]
    fn test_cmp_rr() {
        let enc = encode_cmp_rr(Gpr::Rax, Gpr::Rbx);
        assert_eq!(enc.as_slice(), &[0x48, 0x39, 0xD8]);
    }

    #[test]
    fn test_jmp_rel8() {
        let enc = encode_jmp_rel8(5);
        assert_eq!(enc.as_slice(), &[0xEB, 0x05]);
    }

    #[test]
    fn test_jcc_rel32() {
        let enc = encode_jcc_rel32(Condition::Equal, 0x1000);
        assert_eq!(enc.len(), 6);
        assert_eq!(enc.as_slice()[0..2], [0x0F, 0x84]);
    }

    #[test]
    fn test_call_ret() {
        let enc = encode_call_rel32(0);
        assert_eq!(enc.as_slice(), &[0xE8, 0x00, 0x00, 0x00, 0x00]);

        let enc = encode_ret();
        assert_eq!(enc.as_slice(), &[0xC3]);
    }

    #[test]
    fn test_push_pop() {
        let enc = encode_push(Gpr::Rbx);
        assert_eq!(enc.as_slice(), &[0x53]);

        let enc = encode_push(Gpr::R12);
        assert_eq!(enc.as_slice(), &[0x41, 0x54]);

        let enc = encode_pop(Gpr::Rax);
        assert_eq!(enc.as_slice(), &[0x58]);
    }

    #[test]
    fn test_memory_simple() {
        let mem = MemOperand::base(Gpr::Rax);
        let enc = encode_mov_rm(Gpr::Rcx, &mem);
        assert_eq!(enc.as_slice(), &[0x48, 0x8B, 0x08]);
    }

    #[test]
    fn test_memory_disp8() {
        let mem = MemOperand::base_disp(Gpr::Rbp, -8);
        let enc = encode_mov_rm(Gpr::Rax, &mem);
        assert_eq!(enc.as_slice(), &[0x48, 0x8B, 0x45, 0xF8]);
    }

    #[test]
    fn test_memory_sib() {
        let mem = MemOperand::base_index(Gpr::Rax, Gpr::Rcx, Scale::X8);
        let enc = encode_mov_rm(Gpr::Rdx, &mem);
        assert_eq!(enc.as_slice(), &[0x48, 0x8B, 0x14, 0xC8]);
    }

    #[test]
    fn test_sse_movsd() {
        let enc = encode_movsd_rr(Xmm::Xmm0, Xmm::Xmm1);
        assert_eq!(enc.as_slice(), &[0xF2, 0x0F, 0x10, 0xC1]);
    }

    #[test]
    fn test_sse_addsd() {
        let enc = encode_addsd(Xmm::Xmm0, Xmm::Xmm1);
        assert_eq!(enc.as_slice(), &[0xF2, 0x0F, 0x58, 0xC1]);
    }

    #[test]
    fn test_cvt_instructions() {
        let enc = encode_cvtsi2sd(Xmm::Xmm0, Gpr::Rax);
        assert!(enc.len() > 0);
        assert_eq!(enc.as_slice()[0], 0xF2);

        let enc = encode_cvttsd2si(Gpr::Rax, Xmm::Xmm0);
        assert!(enc.len() > 0);
    }

    #[test]
    fn test_lea() {
        let mem = MemOperand::base_disp(Gpr::Rbp, 16);
        let enc = encode_lea(Gpr::Rax, &mem);
        assert_eq!(enc.as_slice(), &[0x48, 0x8D, 0x45, 0x10]);
    }

    #[test]
    fn test_condition_invert() {
        assert_eq!(Condition::Equal.invert(), Condition::NotEqual);
        assert_eq!(Condition::Less.invert(), Condition::GreaterEqual);
        assert_eq!(Condition::Above.invert(), Condition::BelowEqual);
    }

    #[test]
    fn test_nop_int3_ud2() {
        assert_eq!(encode_nop().as_slice(), &[0x90]);
        assert_eq!(encode_int3().as_slice(), &[0xCC]);
        assert_eq!(encode_ud2().as_slice(), &[0x0F, 0x0B]);
    }
}
