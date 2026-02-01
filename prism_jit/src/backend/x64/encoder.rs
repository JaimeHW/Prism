//! x64 instruction encoder.
use crate::regalloc::Reg;

/// Encode a REX prefix.
#[inline]
pub const fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8)
}

/// Encode ModR/M byte.
#[inline]
pub const fn modrm(mod_: u8, reg: u8, rm: u8) -> u8 {
    (mod_ << 6) | ((reg & 7) << 3) | (rm & 7)
}

/// Encode a register-to-register MOV (64-bit).
pub fn encode_mov_rr(dst: Reg, src: Reg) -> [u8; 3] {
    let rex = rex(true, (src as u8) >= 8, false, (dst as u8) >= 8);
    let modrm = modrm(0b11, src as u8, dst as u8);
    [rex, 0x89, modrm]
}

/// Encode ADD r64, r64.
pub fn encode_add_rr(dst: Reg, src: Reg) -> [u8; 3] {
    let rex = rex(true, (src as u8) >= 8, false, (dst as u8) >= 8);
    let modrm = modrm(0b11, src as u8, dst as u8);
    [rex, 0x01, modrm]
}

/// Encode RET.
pub const fn encode_ret() -> [u8; 1] {
    [0xC3]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rex_w() {
        assert_eq!(rex(true, false, false, false), 0x48);
    }

    #[test]
    fn test_rex_wrb() {
        assert_eq!(rex(true, true, false, true), 0x4D);
    }

    #[test]
    fn test_modrm_reg_reg() {
        // mov rax, rbx -> mod=11, reg=rbx(3), rm=rax(0)
        assert_eq!(modrm(0b11, 3, 0), 0xD8);
    }

    #[test]
    fn test_encode_mov_rax_rbx() {
        let bytes = encode_mov_rr(Reg::Rax, Reg::Rbx);
        assert_eq!(bytes, [0x48, 0x89, 0xD8]);
    }

    #[test]
    fn test_encode_mov_r8_r9() {
        let bytes = encode_mov_rr(Reg::R8, Reg::R9);
        assert_eq!(bytes, [0x4D, 0x89, 0xC8]);
    }

    #[test]
    fn test_encode_add_rax_rcx() {
        let bytes = encode_add_rr(Reg::Rax, Reg::Rcx);
        assert_eq!(bytes, [0x48, 0x01, 0xC8]);
    }

    #[test]
    fn test_encode_ret() {
        assert_eq!(encode_ret(), [0xC3]);
    }
}
