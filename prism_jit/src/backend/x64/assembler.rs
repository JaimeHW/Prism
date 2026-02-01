//! x64 assembler for code emission.
/// Code buffer for emitting x64 instructions.
pub struct Assembler {
    code: Vec<u8>,
}

impl Assembler {
    /// Create a new assembler.
    pub fn new() -> Self {
        Self { code: Vec::new() }
    }

    /// Emit raw bytes.
    pub fn emit(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    /// Get the assembled code.
    pub fn finish(self) -> Vec<u8> {
        self.code
    }
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}
