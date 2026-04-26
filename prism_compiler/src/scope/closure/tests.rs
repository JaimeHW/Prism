use super::*;
use crate::scope::{ScopeAnalyzer, SymbolTable};

/// Helper to analyze source and run closure analysis.
fn analyze(source: &str) -> SymbolTable {
    let module = prism_parser::parse(source).expect("parse error");
    let mut table = ScopeAnalyzer::new().analyze(&module, "<test>");

    // Run closure analysis
    let mut analyzer = ClosureAnalyzer::new();
    analyzer.analyze(&mut table.root);

    table
}

// -------------------------------------------------------------------------
// Basic Cell Detection
// -------------------------------------------------------------------------

#[test]
fn test_simple_closure_cell() {
    let source = r#"
def outer():
    x = 1
    def inner():
        return x
    return inner
"#;
    let table = analyze(source);

    // outer scope should have x as CELL
    let outer_scope = &table.root.children[0];
    let x_outer = outer_scope.lookup("x").unwrap();
    assert!(
        x_outer.flags.contains(SymbolFlags::CELL),
        "x should be a cell in outer"
    );
    assert!(
        x_outer.closure_slot.is_some(),
        "x should have closure slot in outer"
    );

    // inner scope should have x as FREE
    let inner_scope = &outer_scope.children[0];
    let x_inner = inner_scope.lookup("x").unwrap();
    assert!(
        x_inner.flags.contains(SymbolFlags::FREE),
        "x should be free in inner"
    );
    assert!(
        x_inner.closure_slot.is_some(),
        "x should have closure slot in inner"
    );
}

#[test]
fn test_multiple_freevars() {
    let source = r#"
def outer():
    a = 1
    b = 2
    c = 3
    def inner():
        return a + b + c
    return inner
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];
    let inner_scope = &outer_scope.children[0];

    // All three should be cells in outer
    assert!(
        outer_scope
            .lookup("a")
            .unwrap()
            .flags
            .contains(SymbolFlags::CELL)
    );
    assert!(
        outer_scope
            .lookup("b")
            .unwrap()
            .flags
            .contains(SymbolFlags::CELL)
    );
    assert!(
        outer_scope
            .lookup("c")
            .unwrap()
            .flags
            .contains(SymbolFlags::CELL)
    );

    // All three should be free in inner
    assert!(
        inner_scope
            .lookup("a")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );
    assert!(
        inner_scope
            .lookup("b")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );
    assert!(
        inner_scope
            .lookup("c")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );
}

#[test]
fn test_nested_closures() {
    let source = r#"
def outer():
    x = 1
    def middle():
        def inner():
            return x
        return inner
    return middle
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];
    let middle_scope = &outer_scope.children[0];
    let inner_scope = &middle_scope.children[0];

    // x is CELL in outer
    assert!(
        outer_scope
            .lookup("x")
            .unwrap()
            .flags
            .contains(SymbolFlags::CELL)
    );

    // x is FREE in middle (passed through)
    assert!(
        middle_scope
            .lookup("x")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );

    // x is FREE in inner
    assert!(
        inner_scope
            .lookup("x")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );
}

// -------------------------------------------------------------------------
// Nonlocal Declarations
// -------------------------------------------------------------------------

#[test]
fn test_nonlocal_creates_cell() {
    let source = r#"
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 2
    inner()
    return x
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];
    let inner_scope = &outer_scope.children[0];

    // x should be CELL in outer (because inner uses nonlocal)
    let x_outer = outer_scope.lookup("x").unwrap();
    assert!(
        x_outer.flags.contains(SymbolFlags::CELL),
        "x should be CELL in outer"
    );

    // x should be NONLOCAL and FREE in inner
    let x_inner = inner_scope.lookup("x").unwrap();
    assert!(
        x_inner.flags.contains(SymbolFlags::NONLOCAL),
        "x should be NONLOCAL in inner"
    );
    assert!(
        x_inner.flags.contains(SymbolFlags::FREE),
        "x should be FREE in inner"
    );
}

// -------------------------------------------------------------------------
// Closure Slot Assignment
// -------------------------------------------------------------------------

#[test]
fn test_closure_slot_assignment() {
    let source = r#"
def outer():
    a = 1
    b = 2
    def inner():
        return a + b
    return inner
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];

    // Cell variables should have consecutive slots
    let a_slot = outer_scope.lookup("a").unwrap().closure_slot.unwrap();
    let b_slot = outer_scope.lookup("b").unwrap().closure_slot.unwrap();

    // Slots should be 0 and 1 (order depends on sorting)
    assert!(a_slot <= 1);
    assert!(b_slot <= 1);
    assert_ne!(a_slot, b_slot);
}

// -------------------------------------------------------------------------
// Class Scope Quirks
// -------------------------------------------------------------------------

#[test]
fn test_class_scope_no_closure() {
    // Variables defined in class body should NOT be captured by methods
    // They need to be accessed via self, not closure
    let source = r#"
def outer():
    x = 1
    class MyClass:
        y = 2  # This is a class attribute, not captured
        def method(self):
            return x  # This captures x from outer, not y from class
    return MyClass
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];

    // x should be CELL (captured by method)
    assert!(
        outer_scope
            .lookup("x")
            .unwrap()
            .flags
            .contains(SymbolFlags::CELL)
    );

    // The class scope shouldn't have y as a cell
    let class_scope = &outer_scope.children[0];
    let y = class_scope.lookup("y").unwrap();
    assert!(
        !y.flags.contains(SymbolFlags::CELL),
        "class attributes aren't cells"
    );
}

// -------------------------------------------------------------------------
// No Capture Needed
// -------------------------------------------------------------------------

#[test]
fn test_local_var_not_captured() {
    let source = r#"
def outer():
    x = 1
    def inner():
        y = 2
        return y
    return inner
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];

    // x should NOT be a cell (not referenced in inner)
    let x = outer_scope.lookup("x").unwrap();
    assert!(
        !x.flags.contains(SymbolFlags::CELL),
        "x should not be a cell"
    );
    assert!(x.closure_slot.is_none());
}

#[test]
fn test_global_not_captured() {
    let source = r#"
x = 1
def outer():
    def inner():
        return x
    return inner
"#;
    let table = analyze(source);

    // Module level x is global, not a cell
    let x = table.root.lookup("x").unwrap();
    assert!(!x.flags.contains(SymbolFlags::CELL));

    // inner should treat x as global, not a free/cell capture.
    let outer_scope = &table.root.children[0];
    let inner_scope = &outer_scope.children[0];
    let x_inner = inner_scope.lookup("x").unwrap();
    assert!(x_inner.flags.contains(SymbolFlags::GLOBAL_IMPLICIT));
    assert!(!x_inner.flags.contains(SymbolFlags::FREE));
}

// -------------------------------------------------------------------------
// Lambda Closures
// -------------------------------------------------------------------------

#[test]
fn test_lambda_closure() {
    let source = r#"
def outer():
    x = 1
    return lambda: x
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];

    // x should be CELL
    assert!(
        outer_scope
            .lookup("x")
            .unwrap()
            .flags
            .contains(SymbolFlags::CELL)
    );

    // Lambda should have x as FREE
    let lambda_scope = &outer_scope.children[0];
    assert!(
        lambda_scope
            .lookup("x")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );
}

// -------------------------------------------------------------------------
// Comprehension Closures
// -------------------------------------------------------------------------

#[test]
fn test_comprehension_closure() {
    let source = r#"
def outer():
    x = 1
    return [x for _ in range(1)]
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];

    // x should be CELL (captured by comprehension scope)
    assert!(
        outer_scope
            .lookup("x")
            .unwrap()
            .flags
            .contains(SymbolFlags::CELL)
    );
}

#[test]
fn test_explicit_global_used_by_comprehension_is_not_cellvar() {
    let source = r#"
seed = [10]

def outer():
    global seed
    return [x + seed[0] for x in range(2)]
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];
    let seed = outer_scope.lookup("seed").unwrap();
    assert!(seed.flags.contains(SymbolFlags::GLOBAL_EXPLICIT));
    assert!(
        !seed.flags.contains(SymbolFlags::CELL),
        "explicit global must not become a cellvar"
    );

    let comp_scope = &outer_scope.children[0];
    let seed = comp_scope.lookup("seed").unwrap();
    assert!(seed.flags.contains(SymbolFlags::GLOBAL_IMPLICIT));
    assert!(
        !seed.flags.contains(SymbolFlags::FREE),
        "comprehension should load explicit outer globals as globals"
    );
}

// -------------------------------------------------------------------------
// Statistics
// -------------------------------------------------------------------------

#[test]
fn test_closure_stats() {
    let source = r#"
def outer():
    a = 1
    b = 2
    def inner1():
        return a
    def inner2():
        return b
    return inner1, inner2
"#;
    let module = prism_parser::parse(source).expect("parse error");
    let mut table = ScopeAnalyzer::new().analyze(&module, "<test>");

    let mut analyzer = ClosureAnalyzer::new();
    let stats = analyzer.analyze(&mut table.root);

    assert_eq!(stats.scopes_analyzed, 4); // module, outer, inner1, inner2
    assert_eq!(stats.scopes_with_freevars, 2); // inner1, inner2
    assert_eq!(stats.cell_count, 2); // a, b
    assert_eq!(stats.freevar_count, 2); // a in inner1, b in inner2
}

// Test sibling scopes with tuple return - verifies parser handles comma-separated returns
#[test]
fn test_sibling_scopes_no_closure_analyzer() {
    let source = r#"
def outer():
    a = 1
    b = 2
    def inner1():
        return a
    def inner2():
        return b
    return inner1, inner2
"#;
    // Only parse and run scope analyzer - no closure analyzer
    let module = prism_parser::parse(source).expect("parse error");
    let table = ScopeAnalyzer::new().analyze(&module, "<test>");

    // Verify basic structure
    assert_eq!(table.root.children.len(), 1); // outer
    let outer_scope = &table.root.children[0];
    assert_eq!(outer_scope.children.len(), 2); // inner1, inner2
}

// -------------------------------------------------------------------------
// Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_parameter_as_cell() {
    let source = r#"
def outer(x):
    def inner():
        return x
    return inner
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];

    // x is a parameter but also captured - should be CELL
    let x = outer_scope.lookup("x").unwrap();
    assert!(x.flags.contains(SymbolFlags::PARAM));
    assert!(x.flags.contains(SymbolFlags::CELL));
}

#[test]
fn test_shadowed_variable() {
    let source = r#"
def outer():
    x = 1
    def inner():
        x = 2  # Shadows, doesn't capture
        return x
    return inner
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];
    let inner_scope = &outer_scope.children[0];

    // outer's x should NOT be a cell (inner shadows it)
    let x_outer = outer_scope.lookup("x").unwrap();
    assert!(!x_outer.flags.contains(SymbolFlags::CELL));

    // inner's x is local, not free
    let x_inner = inner_scope.lookup("x").unwrap();
    assert!(x_inner.is_local());
    assert!(!x_inner.flags.contains(SymbolFlags::FREE));
}

#[test]
fn test_deeply_nested_closure() {
    let source = r#"
def level1():
    a = 1
    def level2():
        def level3():
            def level4():
                return a
            return level4
        return level3
    return level2
"#;
    let table = analyze(source);

    let level1 = &table.root.children[0];
    let level2 = &level1.children[0];
    let level3 = &level2.children[0];
    let level4 = &level3.children[0];

    // a is CELL in level1
    assert!(
        level1
            .lookup("a")
            .unwrap()
            .flags
            .contains(SymbolFlags::CELL)
    );

    // a is FREE in levels 2, 3, 4
    assert!(
        level2
            .lookup("a")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );
    assert!(
        level3
            .lookup("a")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );
    assert!(
        level4
            .lookup("a")
            .unwrap()
            .flags
            .contains(SymbolFlags::FREE)
    );
}

#[test]
fn test_closure_with_default_arg() {
    let source = r#"
def outer():
    x = 1
    def inner(y=x):  # x used in default, not as freevar
        return y
    return inner
"#;
    let table = analyze(source);

    let outer_scope = &table.root.children[0];

    // x is used in the default argument, which is evaluated in outer scope
    // So x should NOT be a cell (it's not captured by inner's body)
    let x = outer_scope.lookup("x").unwrap();
    // Note: This depends on exact semantics - defaults are in enclosing scope
    assert!(!x.flags.contains(SymbolFlags::CELL));
}

// -------------------------------------------------------------------------
// Helper Function Tests
// -------------------------------------------------------------------------

#[test]
fn test_scope_provides_closures() {
    assert!(scope_provides_closures(ScopeKind::Function));
    assert!(scope_provides_closures(ScopeKind::Lambda));
    assert!(scope_provides_closures(ScopeKind::Comprehension));
    assert!(!scope_provides_closures(ScopeKind::Class));
    assert!(!scope_provides_closures(ScopeKind::Module));
}

#[test]
fn test_scope_can_have_freevars() {
    assert!(scope_can_have_freevars(ScopeKind::Function));
    assert!(scope_can_have_freevars(ScopeKind::Lambda));
    assert!(scope_can_have_freevars(ScopeKind::Comprehension));
    assert!(scope_can_have_freevars(ScopeKind::Class));
    assert!(!scope_can_have_freevars(ScopeKind::Module));
}
