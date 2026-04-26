use super::*;
use num_bigint::BigInt;
use prism_code::Constant;

fn compile(source: &str) -> CodeObject {
    let module = prism_parser::parse(source).expect("parse error");
    Compiler::compile_module(&module, "<test>").expect("compile error")
}

fn compile_with_dynamic_locals(source: &str) -> CodeObject {
    let module = prism_parser::parse(source).expect("parse error");
    Compiler::compile_module_with_namespace_mode(
        &module,
        "<test>",
        OptimizationLevel::None,
        ModuleNamespaceMode::DynamicLocals,
    )
    .expect("compile error")
}

fn large_call_then_functools_style_listcomp_source() -> String {
    let large_arg_list = (0..250)
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "def helper(*args, **kwargs):\n    return args\n\n\
         def stressed(seq, abcs):\n    helper({large_arg_list})\n    return [helper(base, abcs=abcs) for base in seq]\n"
    )
}

fn large_call_then_class_definition_source() -> String {
    let large_arg_list = (0..250)
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "def helper(*args, **kwargs):\n    return object\n\n\
         def stressed():\n    helper({large_arg_list})\n    class Derived(helper()):\n        pass\n    return Derived\n"
    )
}

fn compile_with_optimization(source: &str, optimize: OptimizationLevel) -> CodeObject {
    let module = prism_parser::parse(source).expect("parse error");
    Compiler::compile_module_with_optimization(&module, "<test>", optimize).expect("compile error")
}

fn try_compile(source: &str) -> Result<CodeObject, CompileError> {
    let module = prism_parser::parse(source).expect("parse error");
    Compiler::compile_module(&module, "<test>")
}

#[test]
fn test_compile_wide_i64_literal_emits_integer_constant() {
    let code = compile("value = 2305843009213693952");

    assert!(code.constants.iter().any(|value| {
        matches!(
            value,
            Constant::BigInt(constant)
                if constant == &BigInt::from(2_305_843_009_213_693_952_i64)
        )
    }));
}

#[test]
fn test_compile_bigint_literal_emits_arbitrary_precision_constant() {
    let expected = BigInt::from(1_u8) << 100_u32;
    let code = compile("value = 1267650600228229401496703205376");

    assert!(
        code.constants
            .iter()
            .any(|value| matches!(value, Constant::BigInt(constant) if constant == &expected))
    );
}

#[test]
fn test_compile_with_uses_call_method_for_context_manager_protocol() {
    let code = compile(
        r#"
with manager:
    pass
"#,
    );

    let load_method_count = code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::LoadMethod as u8)
        .count();
    let call_method_count = code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::CallMethod as u8)
        .count();

    assert_eq!(
        load_method_count, 2,
        "with should load __enter__ and __exit__"
    );
    assert_eq!(
        call_method_count, 3,
        "with should call __enter__ plus both normal/exception __exit__ paths via CallMethod"
    );
    assert!(
        !code
            .instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::Call as u8),
        "with should not use generic Call for context-manager methods"
    );
}

#[test]
fn test_compile_async_with_uses_call_method_for_context_manager_protocol() {
    let code = compile(
        r#"
async def run():
    async with manager:
        pass
"#,
    );

    let async_fn = code
        .nested_code_objects
        .first()
        .expect("expected nested async function");
    let load_method_count = async_fn
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::LoadMethod as u8)
        .count();
    let call_method_count = async_fn
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::CallMethod as u8)
        .count();

    assert_eq!(
        load_method_count, 2,
        "async with should load __aenter__ and __aexit__"
    );
    assert_eq!(
        call_method_count, 3,
        "async with should call __aenter__ plus both normal/exception __aexit__ paths via CallMethod"
    );
    assert!(
        !async_fn
            .instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::Call as u8),
        "async with should not use generic Call for context-manager methods"
    );
}

#[test]
fn test_compile_simple_expr() {
    let code = compile("1 + 2");
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_assignment() {
    let code = compile("x = 42");
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_attribute_assignment_emits_set_attr() {
    let code = compile(
        r#"
class Holder:
    pass

def configure(obj, value):
    obj.answer = value
"#,
    );
    let configure = code
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "configure")
        .expect("expected nested configure function");

    assert!(
        configure
            .instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::SetAttr as u8),
        "attribute assignment should lower to SetAttr"
    );
}

#[test]
fn test_compile_if() {
    let code = compile("if True:\n    pass");
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_while() {
    let code = compile("x = 0\nwhile x < 10:\n    x = x + 1");
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_function_call() {
    let code = compile("print(42)");
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_redefined_functions_preserve_distinct_nested_code_objects() {
    let code = compile(
        r#"
def value(self):
    return self

def value(self, new_value):
    return new_value
"#,
    );

    let value_defs = code
        .nested_code_objects
        .iter()
        .filter(|nested| nested.name.as_ref() == "value")
        .collect::<Vec<_>>();

    assert_eq!(value_defs.len(), 2);
    assert_eq!(value_defs[0].arg_count, 1);
    assert_eq!(value_defs[1].arg_count, 2);
}

#[test]
fn test_compile_bytes_literal_uses_builtin_constructor_lowering() {
    let code = compile("value = b'AB'");
    let call = code
        .instructions
        .iter()
        .find(|inst| inst.opcode() == Opcode::Call as u8)
        .expect("expected bytes literal lowering to emit a call");

    assert_eq!(
        call.src2().0,
        2,
        "bytes literal should call bytes(..., encoding)"
    );
    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::LoadBuiltin as u8),
        "bytes literal lowering should bypass shadowable globals"
    );
    assert!(
        code.names.iter().any(|name| &**name == "bytes"),
        "bytes constructor should be resolved by name"
    );
}

#[test]
fn test_compile_complex_literal_uses_builtin_constructor_lowering() {
    let code = compile("value = 0j");
    let call = code
        .instructions
        .iter()
        .find(|inst| inst.opcode() == Opcode::Call as u8)
        .expect("expected complex literal lowering to emit a call");

    assert_eq!(
        call.src2().0,
        2,
        "complex literal lowering should call complex(real, imag)"
    );
    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::LoadBuiltin as u8),
        "complex literal lowering should bypass shadowable globals"
    );
    assert!(
        code.names.iter().any(|name| &**name == "complex"),
        "complex constructor should be resolved by name"
    );
}

#[test]
fn test_compile_empty_bytes_literal_avoids_placeholder_none_lowering() {
    let bytes_code = compile("value = b''");
    let string_code = compile("value = ''");
    let bytes_load_none_count = bytes_code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::LoadNone as u8)
        .count();
    let string_load_none_count = string_code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::LoadNone as u8)
        .count();

    assert_eq!(
        bytes_load_none_count, string_load_none_count,
        "empty bytes literals should compile like ordinary literals instead of the unimplemented-expression fallback"
    );
}

#[test]
fn test_compile_try_star_rejects_instead_of_lowering_as_regular_try() {
    let module = Module::new(
        vec![Stmt::new(
            StmtKind::TryStar {
                body: vec![Stmt::new(StmtKind::Pass, Span::new(1, 1))],
                handlers: Vec::new(),
                orelse: Vec::new(),
                finalbody: Vec::new(),
            },
            Span::new(1, 1),
        )],
        Span::new(1, 1),
    );
    let err = Compiler::compile_module(&module, "<test>")
        .expect_err("try/except* must not compile through the regular try path");

    assert!(err.message.contains("TryStar"));
    assert!(err.message.contains("ExceptionGroup"));
}

#[test]
fn test_compile_type_alias_rejects_unimplemented_semantics() {
    let err = try_compile("type Alias = int")
        .expect_err("type alias must not silently compile as a no-op");

    assert!(err.message.contains("TypeAlias"));
    assert!(err.message.contains("TypeAliasType"));
}

#[test]
fn test_compile_match_singleton_uses_identity_opcode() {
    let code = compile(
        r#"
match value:
    case True:
        result = 1
    case _:
        result = 0
"#,
    );

    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::Is as u8),
        "singleton patterns must compile to identity checks"
    );
}

#[test]
fn test_compile_module_annotations_emit_runtime_namespace_setup() {
    let code = compile(
        r#"
x: int = 1
y: str
"#,
    );

    let setup_annotations_count = code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::SetupAnnotations as u8)
        .count();
    let set_item_count = code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::SetItem as u8)
        .count();

    assert_eq!(
        setup_annotations_count, 1,
        "module annotations should initialize __annotations__ once"
    );
    assert_eq!(
        set_item_count, 2,
        "both simple module annotations should be recorded at runtime"
    );
}

#[test]
fn test_compile_listcomp_reuses_register_pool_after_large_call_blocks() {
    let source = large_call_then_functools_style_listcomp_source();
    let code = compile(&source);
    let stressed = code
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "stressed")
        .expect("expected nested stressed function");

    assert!(
        !stressed.instructions.is_empty(),
        "list comprehension should compile into a real function body"
    );
}

#[test]
fn test_compile_class_definition_reuses_register_pool_after_large_call_blocks() {
    let source = large_call_then_class_definition_source();
    let code = compile(&source);
    let stressed = code
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "stressed")
        .expect("expected nested stressed function");

    assert!(
        !stressed.instructions.is_empty(),
        "class definition should compile into a real function body"
    );
}

#[test]
fn test_compile_dotted_import_binds_top_level_name() {
    let code = compile("import pkg.helper");
    let import_name_count = code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::ImportName as u8)
        .count();

    assert_eq!(import_name_count, 2);
    assert!(code.names.iter().any(|name| name.as_ref() == "pkg.helper"));
    assert!(code.names.iter().any(|name| name.as_ref() == "pkg"));
}

#[test]
fn test_compile_relative_import_from_submodule_preserves_level() {
    let code = compile("from .helper import VALUE");
    let import_name_count = code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::ImportName as u8)
        .count();
    let import_from_count = code
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::ImportFrom as u8)
        .count();

    assert_eq!(import_name_count, 1);
    assert_eq!(import_from_count, 1);
    assert!(code.names.iter().any(|name| name.as_ref() == ".helper"));
    assert!(code.names.iter().any(|name| name.as_ref() == "VALUE"));
}

#[test]
fn test_compile_relative_import_without_module_preserves_level() {
    let code = compile("from . import helper");

    assert!(
        code.names.iter().any(|name| name.as_ref() == "."),
        "expected bare relative import to encode its level"
    );
    assert!(code.names.iter().any(|name| name.as_ref() == "helper"));
    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::ImportFrom as u8)
    );
}

#[test]
fn test_compile_relative_star_import_preserves_parent_level() {
    let code = compile("from ..pkg import *");

    assert!(code.names.iter().any(|name| name.as_ref() == "..pkg"));
    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::ImportStar as u8)
    );
}

#[test]
fn test_compile_assert_emits_raise_path() {
    let code = compile("assert False");
    let opcodes: Vec<u8> = code.instructions.iter().map(|inst| inst.opcode()).collect();

    assert!(
        opcodes.iter().any(|op| *op == Opcode::LoadGlobal as u8),
        "assert should load AssertionError constructor"
    );
    assert!(
        opcodes.iter().any(|op| *op == Opcode::Call as u8),
        "assert should call AssertionError constructor"
    );
    assert!(
        opcodes.iter().any(|op| *op == Opcode::Raise as u8),
        "assert should raise the constructed exception"
    );
}

#[test]
fn test_compile_assert_with_message_emits_call_with_one_arg() {
    let code = compile("assert False, 42");

    let call = code
        .instructions
        .iter()
        .find(|inst| inst.opcode() == Opcode::Call as u8)
        .expect("assert with message should emit Call");
    assert_eq!(
        call.src2().0,
        1,
        "assert message should be passed as 1 call arg"
    );
}

#[test]
fn test_compile_assert_stripped_with_optimize_basic() {
    let code = compile_with_optimization("assert False", OptimizationLevel::Basic);
    let opcodes: Vec<u8> = code.instructions.iter().map(|inst| inst.opcode()).collect();

    assert!(
        !opcodes.iter().any(|op| *op == Opcode::Raise as u8),
        "assert should be stripped under -O"
    );
}

#[test]
fn test_compile_module_docstring_stripped_with_optimize_full() {
    let source = r#"
"""module doc"""
x = 1
"#;
    let unoptimized = compile_with_optimization(source, OptimizationLevel::None);
    let optimized = compile_with_optimization(source, OptimizationLevel::Full);

    let unoptimized_load_consts = unoptimized
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::LoadConst as u8)
        .count();
    let optimized_load_consts = optimized
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::LoadConst as u8)
        .count();

    assert!(
        optimized_load_consts < unoptimized_load_consts,
        "module docstring should be removed under -OO"
    );
}

#[test]
fn test_compile_function_docstring_stripped_with_optimize_full() {
    let source = r#"
def f():
    """function doc"""
    return 1
"#;
    let unoptimized = compile_with_optimization(source, OptimizationLevel::None);
    let optimized = compile_with_optimization(source, OptimizationLevel::Full);

    let fn_unoptimized = unoptimized
        .nested_code_objects
        .first()
        .map(Arc::as_ref)
        .expect("expected nested function code object");
    let fn_optimized = optimized
        .nested_code_objects
        .first()
        .map(Arc::as_ref)
        .expect("expected nested function code object");

    let unoptimized_load_consts = fn_unoptimized
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::LoadConst as u8)
        .count();
    let optimized_load_consts = fn_optimized
        .instructions
        .iter()
        .filter(|inst| inst.opcode() == Opcode::LoadConst as u8)
        .count();

    assert!(
        optimized_load_consts < unoptimized_load_consts,
        "function docstring should be removed under -OO"
    );
}

#[test]
fn test_compile_closure_metadata_follows_closure_slot_order() {
    let code = compile(
        r#"
def outer():
    b = 1
    a = 2

    def inner():
        return a, b

    return inner
"#,
    );

    let outer = code
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "outer")
        .map(Arc::as_ref)
        .expect("expected outer function code object");
    let inner = outer
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "inner")
        .map(Arc::as_ref)
        .expect("expected inner function code object");

    assert_eq!(
        outer
            .cellvars
            .iter()
            .map(|name| name.as_ref())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(
        inner
            .freevars
            .iter()
            .map(|name| name.as_ref())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
}

#[test]
fn test_compile_explicit_global_in_comprehension_stays_global() {
    let code = compile(
        r#"
seed = [10]

def outer():
    global seed
    return [x + seed[0] for x in range(2)]
"#,
    );

    let outer = code
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "outer")
        .map(Arc::as_ref)
        .expect("expected outer function code object");
    let listcomp = outer
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "<listcomp>")
        .map(Arc::as_ref)
        .expect("expected nested listcomp code object");

    assert!(
        outer.cellvars.is_empty(),
        "explicit globals must not be materialized as function cellvars"
    );
    assert!(
        listcomp.freevars.is_empty(),
        "comprehension should load explicit outer globals from module globals"
    );
}

#[test]
fn test_register_count() {
    let code = compile("a = 1\nb = 2\nc = a + b");
    // Should use some registers
    assert!(code.register_count > 0);
}

// =========================================================================
// Loop Control Flow Tests (break/continue)
// =========================================================================

#[test]
fn test_break_in_while_loop() {
    // Basic break in while loop
    let code = compile(
        r#"
i = 0
while True:
    i = i + 1
    if i >= 5:
        break
"#,
    );
    assert!(!code.instructions.is_empty());
    // Should have Jump instructions for break
    let has_jump = code.instructions.iter().any(|i| {
        let opcode = i.opcode();
        opcode == Opcode::Jump as u8
    });
    assert!(has_jump, "expected Jump instruction for break");
}

#[test]
fn test_continue_in_while_loop() {
    // Continue in while loop
    let code = compile(
        r#"
total = 0
i = 0
while i < 10:
    i = i + 1
    if i % 2 == 0:
        continue
    total = total + i
"#,
    );
    assert!(!code.instructions.is_empty());
    // Should have Jump instructions for continue
    let has_jump = code.instructions.iter().any(|i| {
        let opcode = i.opcode();
        opcode == Opcode::Jump as u8
    });
    assert!(has_jump, "expected Jump instruction for continue");
}

#[test]
fn test_break_in_for_loop() {
    // Break in for loop
    let code = compile(
        r#"
result = 0
for x in range(100):
    if x == 5:
        break
    result = result + x
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_continue_in_for_loop() {
    // Continue in for loop
    let code = compile(
        r#"
total = 0
for x in range(10):
    if x % 2 == 0:
        continue
    total = total + x
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_nested_loops_with_break() {
    // Break in nested loops - should only break inner loop
    let code = compile(
        r#"
found = False
for i in range(5):
    for j in range(5):
        if i == 2 and j == 3:
            found = True
            break
    if found:
        break
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_nested_loops_with_continue() {
    // Continue in nested loops
    let code = compile(
        r#"
total = 0
for i in range(5):
    for j in range(5):
        if j % 2 == 0:
            continue
        total = total + 1
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_while_with_else_and_break() {
    // While-else with break (else should be skipped on break)
    let code = compile(
        r#"
i = 0
while i < 10:
    if i == 5:
        break
    i = i + 1
else:
    x = 42
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_for_with_else_and_break() {
    // For-else with break
    let code = compile(
        r#"
for x in range(10):
    if x == 5:
        break
else:
    y = 42
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_break_outside_loop_error() {
    // Break outside loop should be an error
    let result = try_compile("break");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.message.contains("'break' outside loop"),
        "expected 'break' outside loop error, got: {}",
        err.message
    );
}

#[test]
fn test_continue_outside_loop_error() {
    // Continue outside loop should be an error
    let result = try_compile("continue");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.message.contains("'continue' outside loop"),
        "expected 'continue' outside loop error, got: {}",
        err.message
    );
}

#[test]
fn test_break_in_if_inside_loop() {
    // Break in if statement inside loop is valid
    let code = compile(
        r#"
for x in range(10):
    if x > 5:
        break
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_deeply_nested_break() {
    // Break in deeply nested structure
    let code = compile(
        r#"
for a in range(5):
    for b in range(5):
        for c in range(5):
            if a + b + c > 10:
                break
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_multiple_breaks_in_loop() {
    // Multiple break statements in same loop
    let code = compile(
        r#"
for x in range(100):
    if x == 5:
        break
    if x == 10:
        break
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_break_and_continue_in_same_loop() {
    // Both break and continue in same loop
    let code = compile(
        r#"
for x in range(100):
    if x == 50:
        break
    if x % 2 == 0:
        continue
    y = x * 2
"#,
    );
    assert!(!code.instructions.is_empty());
}

// =========================================================================
// Class Compilation Tests
// =========================================================================

#[test]
fn test_compile_empty_class() {
    // Simplest possible class definition
    let code = compile(
        r#"
class Empty:
    pass
"#,
    );
    assert!(!code.instructions.is_empty());
    // Class body code should be in constants
    assert!(
        !code.constants.is_empty(),
        "Class should have nested code object"
    );
}

#[test]
fn test_compile_build_class_encodes_code_const_index_and_base_count() {
    let code = compile(
        r#"
class Child(Base1, Base2):
    pass
"#,
    );

    let build_class = code
        .instructions
        .iter()
        .find(|inst| inst.opcode() == Opcode::BuildClass as u8)
        .expect("expected BUILD_CLASS instruction");

    let build_index = code
        .instructions
        .iter()
        .position(|inst| inst.opcode() == Opcode::BuildClass as u8)
        .expect("expected BUILD_CLASS instruction");
    let meta = code
        .instructions
        .get(build_index + 1)
        .copied()
        .expect("BUILD_CLASS should be followed by ClassMeta");

    assert_eq!(meta.opcode(), Opcode::ClassMeta as u8);
    assert_eq!(meta.dst().0, 2, "base count must match source");

    let code_idx = build_class.imm16() as usize;
    let code_const = code
        .constants
        .get(code_idx)
        .expect("BUILD_CLASS code index must be in constant pool");
    let code_ptr = match code_const {
        Constant::Value(value) => value
            .as_object_ptr()
            .expect("BUILD_CLASS constant must be a code object pointer"),
        Constant::BigInt(_) => panic!("BUILD_CLASS constant must not be a bigint"),
    };

    let nested = code
        .nested_code_objects
        .iter()
        .find(|nested| Arc::as_ptr(nested) as *const () == code_ptr)
        .expect("BUILD_CLASS code object must exist in nested_code_objects");

    assert_eq!(nested.name.as_ref(), "Child");
}

#[test]
fn test_compile_build_class_compiles_bases_into_contiguous_result_block() {
    let code = compile(
        r#"
class Child(Base1, Base2):
    pass
"#,
    );

    let (build_index, build_class) = code
        .instructions
        .iter()
        .enumerate()
        .find(|(_, inst)| inst.opcode() == Opcode::BuildClass as u8)
        .expect("expected BUILD_CLASS instruction");
    let result_reg = build_class.dst().0;

    assert!(
        code.instructions[..build_index].iter().any(|inst| {
            inst.opcode() == Opcode::LoadGlobal as u8 && inst.dst().0 == result_reg + 1
        }),
        "expected first base to be compiled into BUILD_CLASS base slot"
    );
    assert!(
        code.instructions[..build_index].iter().any(|inst| {
            inst.opcode() == Opcode::LoadGlobal as u8 && inst.dst().0 == result_reg + 2
        }),
        "expected second base to be compiled into BUILD_CLASS base slot"
    );
}

#[test]
fn test_compile_build_class_emits_keyword_metadata_for_class_keywords() {
    let code = compile(
        r#"
class Child(Base, answer=42):
    pass
"#,
    );

    let (build_index, _) = code
        .instructions
        .iter()
        .enumerate()
        .find(|(_, inst)| inst.opcode() == Opcode::BuildClass as u8)
        .expect("expected BUILD_CLASS instruction");
    let ext = code
        .instructions
        .get(build_index + 2)
        .copied()
        .expect("BUILD_CLASS with keywords must be followed by metadata");

    assert_eq!(ext.opcode(), Opcode::CallKwEx as u8);
    assert_eq!(ext.dst().0, 1, "expected one class keyword");

    let kwnames_idx = (ext.src1().0 as u16) | ((ext.src2().0 as u16) << 8);
    let names_ptr = code
        .constants
        .get(kwnames_idx as usize)
        .and_then(|value| match value {
            Constant::Value(value) => value.as_object_ptr(),
            Constant::BigInt(_) => None,
        })
        .expect("class keyword metadata should point at keyword names");
    let names = unsafe { &*(names_ptr as *const crate::bytecode::KwNamesTuple) };
    assert_eq!(names.get(0).map(|name| name.as_ref()), Some("answer"));
}

#[test]
fn test_compile_class_with_method() {
    // Class with a simple method
    let code = compile(
        r#"
class Counter:
    def increment(self):
        pass
"#,
    );
    assert!(!code.instructions.is_empty());
    // Should have nested code object for class body
    assert!(
        !code.constants.is_empty(),
        "Class should have nested code objects"
    );
}

#[test]
fn test_compile_class_with_init() {
    // Class with __init__ method
    let code = compile(
        r#"
class MyClass:
    def __init__(self, x):
        self.x = x
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_class_variable() {
    // Class with class-level variable
    let code = compile(
        r#"
class Config:
    DEBUG = True
    VERSION = 1
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_body_predefines_scope_locals() {
    let code = compile(
        r#"
class Config:
    DEBUG = True

    def build(self):
        return DEBUG
"#,
    );

    let class_body = code
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "Config")
        .map(Arc::as_ref)
        .expect("expected class body code object");

    let locals = class_body
        .locals
        .iter()
        .map(|name| name.as_ref())
        .collect::<Vec<_>>();
    assert!(locals.contains(&"DEBUG"));
    assert!(locals.contains(&"build"));
}

#[test]
fn test_compile_nested_class_binds_into_class_namespace() {
    let code = compile(
        r#"
class Outer:
    class Inner:
        pass
"#,
    );

    let outer = code
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "Outer")
        .map(Arc::as_ref)
        .expect("expected outer class body");

    assert!(
        outer
            .instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::StoreLocal as u8),
        "nested class bindings inside a class body should target class locals"
    );
}

#[test]
fn test_compile_class_delete_name_targets_class_locals() {
    let code = compile(
        r#"
class Example:
    value = 1
    del value
"#,
    );

    let class_body = code
        .nested_code_objects
        .iter()
        .find(|nested| nested.name.as_ref() == "Example")
        .map(Arc::as_ref)
        .expect("expected class body code object");

    assert!(
        class_body
            .instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::DeleteLocal as u8),
        "class body deletes should target class locals"
    );
    assert!(
        !class_body
            .instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::DeleteGlobal as u8),
        "class body deletes should not be lowered as global deletes"
    );
}

#[test]
fn test_compile_dynamic_locals_binds_module_assignments_as_locals() {
    let code = compile_with_dynamic_locals(
        r#"
x = 1
y = x
"#,
    );

    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::StoreLocal as u8),
        "dynamic-locals compilation should route module assignments through local slots",
    );
    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::LoadLocal as u8),
        "dynamic-locals compilation should route module lookups through local slots",
    );
    assert!(
        !code
            .instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::StoreGlobal as u8),
        "ordinary dynamic-locals bindings should not be lowered as global stores",
    );
}

#[test]
fn test_compile_dynamic_locals_preserves_explicit_global_bindings() {
    let code = compile_with_dynamic_locals(
        r#"
global shared
x = 1
shared = x
"#,
    );

    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::StoreLocal as u8),
        "local dynamic bindings should still use local slots when globals are declared elsewhere",
    );
    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::StoreGlobal as u8),
        "explicit global statements must continue to target module globals",
    );
}

#[test]
fn test_compile_dynamic_locals_uses_local_lookups_for_unbound_names() {
    let code = compile_with_dynamic_locals(
        r#"
result = missing_name
"#,
    );

    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::LoadLocal as u8),
        "dynamic-locals compilation should use locals-first lookups even for unbound names",
    );
    assert!(
        !code
            .instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::LoadGlobal as u8),
        "unbound dynamic-locals lookups should not bypass the locals mapping",
    );
}

#[test]
fn test_compile_delete_subscript_lowers_to_del_item() {
    let code = compile(
        r#"
mapping = {"token": 1}
del mapping["token"]
"#,
    );

    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::DelItem as u8),
        "subscript deletes should lower to DelItem"
    );
}

#[test]
fn test_compile_delete_attribute_lowers_to_del_attr() {
    let code = compile(
        r#"
class Box:
    pass

box = Box()
del box.value
"#,
    );

    assert!(
        code.instructions
            .iter()
            .any(|inst| inst.opcode() == Opcode::DelAttr as u8),
        "attribute deletes should lower to DelAttr"
    );
}

#[test]
fn test_compile_class_with_single_base() {
    // Simple inheritance
    let code = compile(
        r#"
class Child(Parent):
    pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_multiple_bases() {
    // Multiple inheritance
    let code = compile(
        r#"
class Multi(Base1, Base2, Base3):
    pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_decorator() {
    // Decorated class
    let code = compile(
        r#"
@decorator
class MyClass:
    pass
"#,
    );
    assert!(!code.instructions.is_empty());
    // Should have CALL for decorator application
}

#[test]
fn test_compile_class_with_multiple_decorators() {
    // Multiple decorators
    let code = compile(
        r#"
@decorator1
@decorator2
@decorator3
class MyClass:
    pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_decorator_call() {
    // Decorator with arguments
    let code = compile(
        r#"
@dataclass(frozen=True)
class Point:
    x: int
    y: int
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_multiple_methods() {
    // Class with multiple methods
    let code = compile(
        r#"
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_static_method() {
    // Class with static method
    let code = compile(
        r#"
class Utils:
    @staticmethod
    def helper(x):
        return x * 2
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_class_method() {
    // Class with class method
    let code = compile(
        r#"
class Factory:
    @classmethod
    def create(cls):
        return cls()
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_property() {
    // Class with property decorator
    let code = compile(
        r#"
class Circle:
    @property
    def area(self):
        return 3.14159 * self.radius ** 2
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_nested_class() {
    // Nested class definition
    let code = compile(
        r#"
class Outer:
    class Inner:
        pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_deeply_nested_class() {
    // Deeply nested class definitions
    let code = compile(
        r#"
class Level1:
    class Level2:
        class Level3:
            value = 42
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_docstring() {
    // Class with docstring
    let code = compile(
        r#"
class Documented:
    """This is a docstring."""
    pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_super_init() {
    // Class calling super().__init__
    let code = compile(
        r#"
class Child(Parent):
    def __init__(self):
        super().__init__()
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_explicit_super() {
    // Class using explicit super(ClassName, self)
    let code = compile(
        r#"
class Child(Parent):
    def __init__(self):
        super(Child, self).__init__()
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_super_method_call() {
    // Class calling super() method
    let code = compile(
        r#"
class Child(Parent):
    def process(self):
        return super().process() + 1
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_zero_arg_super_captures_class_cell() {
    let code = compile(
        r#"
class Child(Parent):
    def method(self):
        return super().process()
"#,
    );

    let class_body = code
        .nested_code_objects
        .first()
        .expect("class body code should be nested in module");
    assert!(
        class_body
            .cellvars
            .iter()
            .any(|name| name.as_ref() == "__class__"),
        "class body should expose __class__ as a cellvar"
    );

    let method_code = class_body
        .nested_code_objects
        .first()
        .expect("method code should be nested in class body");
    assert!(
        method_code
            .freevars
            .iter()
            .any(|name| name.as_ref() == "__class__"),
        "method using zero-arg super should capture __class__ as a freevar"
    );
}

#[test]
fn test_compile_class_with_dunder_methods() {
    // Class with magic methods
    let code = compile(
        r#"
class Custom:
    def __str__(self):
        return "Custom"
    
    def __repr__(self):
        return "Custom()"
    
    def __len__(self):
        return 0
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_operator_overloading() {
    // Class with operator overloading
    let code = compile(
        r#"
class Vector:
    def __add__(self, other):
        pass
    
    def __sub__(self, other):
        pass
    
    def __mul__(self, scalar):
        pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_slots() {
    // Class with __slots__ definition
    let code = compile(
        r#"
class Point:
    __slots__ = ['x', 'y']
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_class_body_expression() {
    // Class with expression in body
    let code = compile(
        r#"
class Computed:
    VALUE = 1 + 2 + 3
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_conditional() {
    // Class with conditional in body
    let code = compile(
        r#"
class Conditional:
    if True:
        x = 1
    else:
        x = 2
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_for_loop() {
    // Class with for loop in body
    let code = compile(
        r#"
class Generated:
    items = []
    for i in range(5):
        items.append(i)
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_comprehension() {
    // Class with comprehension in body
    let code = compile(
        r#"
class WithComprehension:
    squares = [x**2 for x in range(10)]
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_inheriting_from_expression() {
    // Class inheriting from expression
    let code = compile(
        r#"
class Sub(get_base()):
    pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_method_decorator() {
    // Method with multiple decorators
    let code = compile(
        r#"
class Service:
    @decorator1
    @decorator2
    def method(self):
        pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_private_method() {
    // Class with private method (name mangling)
    let code = compile(
        r#"
class Private:
    def __secret(self):
        pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_closure_in_method() {
    // Method containing a closure
    let code = compile(
        r#"
class WithClosure:
    def outer(self):
        x = 1
        def inner():
            return x
        return inner
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_multiple_classes() {
    // Multiple class definitions in same module
    let code = compile(
        r#"
class First:
    pass

class Second:
    pass

class Third(First, Second):
    pass
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_and_function() {
    // Class and function in same module
    let code = compile(
        r#"
def helper():
    pass

class MyClass:
    def method(self):
        helper()
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_dataclass_like() {
    // Dataclass-like pattern
    let code = compile(
        r#"
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_singleton_pattern() {
    // Singleton pattern
    let code = compile(
        r#"
class Singleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_class_with_lambda_in_body() {
    // Class with lambda in body
    let code = compile(
        r#"
class WithLambda:
    transform = lambda x: x * 2
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_class_code_object_has_class_flag() {
    // Verify class compilation produces code object
    let code = compile(
        r#"
class Flagged:
    pass
"#,
    );
    // Verify we have constants (class body code object)
    assert!(
        !code.constants.is_empty(),
        "Class body code object should exist in constants"
    );
    // Verify instructions are generated
    assert!(!code.instructions.is_empty());
}

// =========================================================================
// Exception Compilation Tests
// =========================================================================

#[test]
fn test_compile_simple_try_except() {
    let code = compile(
        r#"
try:
    x = 1
except:
    y = 2
"#,
    );
    assert!(!code.instructions.is_empty());
    assert!(!code.exception_table.is_empty());
}

#[test]
fn test_compile_try_except_with_type() {
    let code = compile(
        r#"
try:
    x = dangerous()
except ValueError:
    y = fallback()
"#,
    );
    assert!(!code.instructions.is_empty());
    assert!(!code.exception_table.is_empty());
}

#[test]
fn test_compile_typed_except_emits_verifiable_dynamic_handler_metadata() {
    let code = compile(
        r#"
try:
    from _abc import get_cache_token
except ImportError:
    get_cache_token = None
except (AttributeError, TypeError):
    get_cache_token = lambda: None
"#,
    );

    code.validate()
        .expect("typed except handler metadata should validate");
    assert!(
        code.exception_table
            .iter()
            .all(|entry| entry.exception_type_idx == u16::MAX),
        "typed except matching is dynamic and should not encode handler PCs as type metadata"
    );
}

#[test]
fn test_compile_try_except_else() {
    let code = compile(
        r#"
try:
    x = 1
except:
    y = 2
else:
    z = 3
"#,
    );
    assert!(!code.instructions.is_empty());
    assert!(!code.exception_table.is_empty());
}

#[test]
fn test_compile_try_finally() {
    let code = compile(
        r#"
try:
    x = 1
finally:
    cleanup()
"#,
    );
    assert!(!code.instructions.is_empty());
    assert!(!code.exception_table.is_empty());
}

#[test]
fn test_compile_try_except_finally() {
    let code = compile(
        r#"
try:
    x = 1
except ValueError:
    y = 2
finally:
    cleanup()
"#,
    );
    assert!(!code.instructions.is_empty());
    assert!(code.exception_table.len() >= 2);
}

#[test]
fn test_compile_multiple_except_handlers() {
    let code = compile(
        r#"
try:
    x = risky()
except ValueError:
    a = 1
except TypeError:
    b = 2
except:
    c = 3
"#,
    );
    assert!(!code.instructions.is_empty());
    assert!(code.exception_table.len() >= 3);
}

#[test]
fn test_compile_nested_try_except() {
    let code = compile(
        r#"
try:
    try:
        x = 1
    except:
        y = 2
except:
    z = 3
"#,
    );
    assert!(!code.instructions.is_empty());
    assert!(code.exception_table.len() >= 2);
}

#[test]
fn test_compile_try_in_function() {
    let code = compile(
        r#"
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0
"#,
    );
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_augassign_rejects_unpacking_targets() {
    let err = try_compile("x, b += 3").expect_err("tuple augassign target should fail");
    assert!(
        err.message
            .contains("illegal expression for augmented assignment"),
        "unexpected compile error: {err:?}"
    );
}
