use super::*;
use prism_core::Span;
use prism_parser::ast::ExprKind;
use std::sync::Arc;

fn test_expr(kind: ExprKind) -> Expr {
    Expr::new(kind, Span::dummy())
}

#[test]
fn test_subject_cache_new() {
    let root = Register::new(0);
    let cache = SubjectCache::new(root);
    assert!(cache.cache.contains_key(&AccessPath::Root));
}

#[test]
fn test_subject_cache_root_lookup() {
    let root = Register::new(0);
    let cache = SubjectCache::new(root);
    assert_eq!(cache.cache.get(&AccessPath::Root), Some(&root));
}

#[test]
fn test_emit_match_class_instruction() {
    let mut builder = FunctionBuilder::new("test");
    let dst = builder.alloc_register();
    let subject = builder.alloc_register();
    let cls = builder.alloc_register();

    emit_match_class(&mut builder, dst, subject, cls);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::MatchClass as u8);
}

#[test]
fn test_emit_match_mapping_instruction() {
    let mut builder = FunctionBuilder::new("test");
    let dst = builder.alloc_register();
    let subject = builder.alloc_register();

    emit_match_mapping(&mut builder, dst, subject);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::MatchMapping as u8);
}

#[test]
fn test_emit_match_sequence_instruction() {
    let mut builder = FunctionBuilder::new("test");
    let dst = builder.alloc_register();
    let subject = builder.alloc_register();

    emit_match_sequence(&mut builder, dst, subject);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::MatchSequence as u8);
}

#[test]
fn test_emit_match_keys_instruction() {
    let mut builder = FunctionBuilder::new("test");
    let dst = builder.alloc_register();
    let mapping = builder.alloc_register();
    let keys = builder.alloc_register();

    emit_match_keys(&mut builder, dst, mapping, keys);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::MatchKeys as u8);
}

#[test]
fn test_emit_copy_dict_without_keys_instruction() {
    let mut builder = FunctionBuilder::new("test");
    let dst = builder.alloc_register();
    let mapping = builder.alloc_register();
    let keys = builder.alloc_register();

    emit_copy_dict_without_keys(&mut builder, dst, mapping, keys);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(
        code.instructions[0].opcode(),
        Opcode::CopyDictWithoutKeys as u8
    );
}

#[test]
fn test_emit_get_match_args_instruction() {
    let mut builder = FunctionBuilder::new("test");
    let dst = builder.alloc_register();
    let subject = builder.alloc_register();

    emit_get_match_args(&mut builder, dst, subject);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::GetMatchArgs as u8);
}

#[test]
fn test_load_literal_int() {
    let mut builder = FunctionBuilder::new("test");
    let reg = load_literal_value(&mut builder, &LiteralValue::Int(42)).unwrap();

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::LoadConst as u8);
    assert_eq!(code.instructions[0].dst(), reg);
}

#[test]
fn test_load_literal_float() {
    let mut builder = FunctionBuilder::new("test");
    let reg = load_literal_value(&mut builder, &LiteralValue::Float(3.125)).unwrap();

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::LoadConst as u8);
    assert_eq!(code.instructions[0].dst(), reg);
}

#[test]
fn test_load_literal_string() {
    let mut builder = FunctionBuilder::new("test");
    let reg = load_literal_value(&mut builder, &LiteralValue::String(Arc::from("hello"))).unwrap();

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::LoadConst as u8);
    assert_eq!(code.instructions[0].dst(), reg);
}

#[test]
fn test_load_literal_bytes_rejects_placeholder_lowering() {
    let mut builder = FunctionBuilder::new("test");
    let err = load_literal_value(&mut builder, &LiteralValue::Bytes(Arc::from(&b"key"[..])))
        .expect_err("bytes literals must not be lowered as placeholder strings");

    assert!(err.message.contains("unsupported pattern bytes literal"));
}

#[test]
fn test_guarded_leaf_rejects_placeholder_lowering() {
    let mut builder = FunctionBuilder::new("test");
    let end_label = builder.create_label();
    let fail_label = builder.create_label();
    let subject = builder.alloc_register();
    let tree = DecisionTree::Leaf {
        bindings: Vec::new(),
        guard: Some(test_expr(ExprKind::Bool(true))),
        action: 0,
        fallback: Some(Box::new(DecisionTree::Fail)),
    };

    let err = emit_tree(&mut builder, &tree, subject, fail_label, end_label)
        .expect_err("guarded leaves need real guard expression lowering");

    assert!(err.message.contains("unsupported guarded match leaf"));
}

#[test]
fn test_class_constructor_rejects_placeholder_lowering() {
    let mut builder = FunctionBuilder::new("test");
    let result = builder.alloc_register();
    let value = builder.alloc_register();
    let ctor = Constructor::Class {
        cls: Box::new(test_expr(ExprKind::Name("C".to_string()))),
    };

    let err = emit_constructor_test(&mut builder, result, value, &ctor)
        .expect_err("class constructors need real class expression lowering");

    assert!(
        err.message
            .contains("unsupported class pattern constructor")
    );
}

#[test]
fn test_type_check_rejects_placeholder_lowering() {
    let mut builder = FunctionBuilder::new("test");
    let end_label = builder.create_label();
    let fail_label = builder.create_label();
    let subject = builder.alloc_register();
    let tree = DecisionTree::TypeCheck {
        access: AccessPath::Root,
        cls: Box::new(test_expr(ExprKind::Name("C".to_string()))),
        success: Box::new(DecisionTree::Fail),
        failure: Box::new(DecisionTree::Fail),
    };

    let err = emit_tree(&mut builder, &tree, subject, fail_label, end_label)
        .expect_err("type checks need real class expression lowering");

    assert!(err.message.contains("unsupported class pattern type check"));
}

#[test]
fn test_decision_tree_fail_emits_jump() {
    let mut builder = FunctionBuilder::new("test");
    let end_label = builder.create_label();
    let fail_label = builder.create_label();
    let subject = builder.alloc_register();

    emit_tree(
        &mut builder,
        &DecisionTree::Fail,
        subject,
        fail_label,
        end_label,
    )
    .unwrap();

    // Bind labels to avoid "unbound label" error
    builder.bind_label(fail_label);
    builder.bind_label(end_label);

    let code = builder.finish();
    // DecisionTree::Fail emits a jump to fail_label
    assert!(code.instructions.len() >= 1);
    assert_eq!(code.instructions[0].opcode(), Opcode::Jump as u8);
}

#[test]
fn test_subject_cache_preserves_root() {
    let root = Register::new(5);
    let cache = SubjectCache::new(root);

    assert_eq!(cache.root, root);
    assert_eq!(cache.cache.get(&AccessPath::Root), Some(&root));
}

#[test]
fn test_subject_cache_allocated_empty_initially() {
    let root = Register::new(0);
    let cache = SubjectCache::new(root);

    assert!(cache.allocated.is_empty());
}

#[test]
fn test_match_codegen_context_creation() {
    let mut builder = FunctionBuilder::new("test");
    let subject = builder.alloc_register();
    let mut cache = SubjectCache::new(subject);
    let end_label = builder.create_label();
    let fail_label = builder.create_label();

    let codegen = MatchCodegen::new(&mut builder, &mut cache, end_label, fail_label);

    assert_eq!(codegen.end_label, end_label);
    assert_eq!(codegen.fail_label, fail_label);
}
