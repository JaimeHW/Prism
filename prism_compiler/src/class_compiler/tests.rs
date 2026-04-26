use super::*;

#[test]
fn test_class_compiler_new() {
    let compiler = ClassCompiler::new("TestClass", "module.TestClass", "test.py");
    assert_eq!(&*compiler.name, "TestClass");
    assert_eq!(&*compiler.qualname, "module.TestClass");
}

#[test]
fn test_class_compiler_finish_has_class_flag() {
    let mut compiler = ClassCompiler::new("Test", "Test", "test.py");
    let code = compiler.finish();
    assert!(code.flags.contains(CodeFlags::CLASS));
}

#[test]
fn test_detect_zero_arg_super_simple() {
    // Parse and check detection
    let source = "def method(self):\n    super().__init__()";
    let module = prism_parser::parse(source).expect("parse failed");

    if let StmtKind::FunctionDef { body, .. } = &module.body[0].kind {
        assert!(ClassCompiler::uses_zero_arg_super(body));
    } else {
        panic!("Expected function def");
    }
}

#[test]
fn test_detect_zero_arg_super_with_args() {
    // super(Class, self) should NOT be detected as zero-arg
    let source = "def method(self):\n    super(Foo, self).__init__()";
    let module = prism_parser::parse(source).expect("parse failed");

    if let StmtKind::FunctionDef { body, .. } = &module.body[0].kind {
        assert!(!ClassCompiler::uses_zero_arg_super(body));
    } else {
        panic!("Expected function def");
    }
}

#[test]
fn test_detect_zero_arg_super_nested() {
    // super() in nested function should be detected
    let source = "def method(self):\n    def inner():\n        super()\n    inner()";
    let module = prism_parser::parse(source).expect("parse failed");

    if let StmtKind::FunctionDef { body, .. } = &module.body[0].kind {
        // The inner function contains super(), but we're checking the outer body
        // This should find it when recursing
        assert!(ClassCompiler::uses_zero_arg_super(body));
    }
}

#[test]
fn test_detect_no_super() {
    let source = "def method(self):\n    print('hello')";
    let module = prism_parser::parse(source).expect("parse failed");

    if let StmtKind::FunctionDef { body, .. } = &module.body[0].kind {
        assert!(!ClassCompiler::uses_zero_arg_super(body));
    }
}

#[test]
fn test_class_info_creation() {
    let info = ClassInfo {
        name: "MyClass".into(),
        bases: vec![0, 1],
        keywords: vec![("metaclass".into(), 2)],
        body_code_idx: 0,
        decorators: vec![],
    };
    assert_eq!(&*info.name, "MyClass");
    assert_eq!(info.bases.len(), 2);
    assert_eq!(info.keywords.len(), 1);
}
