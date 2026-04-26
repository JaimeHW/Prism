use super::*;

fn analyze(source: &str) -> SymbolTable {
    let module = prism_parser::parse(source).expect("parse error");
    ScopeAnalyzer::new().analyze(&module, "<test>")
}

#[test]
fn test_simple_assignment() {
    let table = analyze("x = 1\nprint(x)");
    let x = table.root.lookup("x").unwrap();
    assert!(x.flags.contains(SymbolFlags::DEF));
    assert!(x.flags.contains(SymbolFlags::USE));
}

#[test]
fn test_function_scope() {
    let table = analyze("def foo(a, b):\n    x = a + b\n    return x");
    let foo = table.root.lookup("foo").unwrap();
    assert!(foo.flags.contains(SymbolFlags::DEF));

    assert_eq!(table.root.children.len(), 1);
    let func_scope = &table.root.children[0];
    assert_eq!(&*func_scope.name, "foo");

    let a = func_scope.lookup("a").unwrap();
    assert!(a.flags.contains(SymbolFlags::PARAM));

    let x = func_scope.lookup("x").unwrap();
    assert!(x.is_local());
}

#[test]
fn test_global_declaration() {
    let table = analyze("x = 1\ndef foo():\n    global x\n    x = 2");

    let func_scope = &table.root.children[0];
    let x = func_scope.lookup("x").unwrap();
    assert!(x.flags.contains(SymbolFlags::GLOBAL_EXPLICIT));
}
