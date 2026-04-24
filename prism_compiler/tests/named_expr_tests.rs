use prism_compiler::{CodeObject, Compiler, Opcode};
use prism_parser::parse;

fn compile(source: &str) -> CodeObject {
    let module = parse(source).expect("parse failed");
    Compiler::compile_module(&module, "<test>").expect("compile failed")
}

fn count_opcode(code: &CodeObject, opcode: Opcode) -> usize {
    code.instructions
        .iter()
        .filter(|instruction| instruction.opcode() == opcode as u8)
        .count()
}

#[test]
fn named_expression_in_while_condition_stores_module_binding_before_branch() {
    let code = compile(
        r#"
while line := read():
    sink(line)
"#,
    );

    assert!(
        count_opcode(&code, Opcode::StoreGlobal) >= 1,
        "module-level walrus binding should store the evaluated value"
    );
    assert!(
        count_opcode(&code, Opcode::JumpIfFalse) >= 1,
        "while condition should branch on the stored expression value"
    );
}

#[test]
fn named_expression_inside_function_uses_scope_aware_local_binding() {
    let code = compile(
        r#"
def consume(read):
    while line := read():
        sink(line)
"#,
    );

    let function_code = code
        .nested_code_objects
        .first()
        .expect("function body should be compiled as nested code");

    assert!(
        count_opcode(function_code, Opcode::StoreLocal) >= 1,
        "function-scope walrus binding should store into a local slot"
    );
    assert!(
        count_opcode(function_code, Opcode::LoadLocal) >= 1,
        "the walrus target should be readable by the loop body"
    );
}
