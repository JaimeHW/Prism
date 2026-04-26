use super::*;
use prism_code::{CodeObject, Instruction, Opcode, Register};
use prism_core::speculation::NoSpeculation;

fn make_code(instructions: Vec<Instruction>) -> CodeObject {
    let mut code = CodeObject::new("test", "test.py");
    code.instructions = instructions.into_boxed_slice();
    code
}

mod core_ops;
mod extended_ops;
