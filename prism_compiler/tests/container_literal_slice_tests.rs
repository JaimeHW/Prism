//! Container literal and slice compilation tests.

use prism_compiler::{Compiler, Opcode};
use prism_parser::parse;

fn compile_source(source: &str) -> prism_compiler::CodeObject {
    let module = parse(source).expect("parse failed");
    Compiler::compile_module(&module, "<test>").expect("compile failed")
}

fn has_opcode(code: &prism_compiler::CodeObject, opcode: Opcode) -> bool {
    code.instructions.iter().any(|inst| inst.opcode() == opcode as u8)
}

#[test]
fn test_set_literal_emits_build_set() {
    let code = compile_source("x = {1, 2, 3}");
    assert!(has_opcode(&code, Opcode::BuildSet));
}

#[test]
fn test_dict_literal_emits_build_dict() {
    let code = compile_source("x = {'a': 1, 'b': 2}");
    assert!(has_opcode(&code, Opcode::BuildDict));
}

#[test]
fn test_dict_literal_unpack_emits_build_dict_unpack() {
    let code = compile_source("x = {'a': 1, **y}");
    assert!(has_opcode(&code, Opcode::BuildDictUnpack));
}

#[test]
fn test_slice_without_step_emits_build_slice_only() {
    let code = compile_source("x = [0, 1, 2, 3]\ny = x[1:3]");
    let build_idx = code
        .instructions
        .iter()
        .position(|inst| inst.opcode() == Opcode::BuildSlice as u8)
        .expect("missing BuildSlice");
    assert_eq!(code.instructions[build_idx + 1].opcode(), Opcode::GetItem as u8);
}

#[test]
fn test_slice_with_step_emits_extension_marker() {
    let code = compile_source("x = [0, 1, 2, 3]\ny = x[::2]");
    let build_idx = code
        .instructions
        .iter()
        .position(|inst| inst.opcode() == Opcode::BuildSlice as u8)
        .expect("missing BuildSlice");

    let ext = code.instructions[build_idx + 1];
    assert_eq!(ext.opcode(), Opcode::CallKwEx as u8);
    assert_eq!(ext.src1().0, b'S');
    assert_eq!(ext.src2().0, b'L');
}
