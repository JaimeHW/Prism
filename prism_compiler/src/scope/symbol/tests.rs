use super::*;

#[test]
fn test_symbol_flags() {
    let flags = SymbolFlags::DEF | SymbolFlags::USE;
    assert!(flags.contains(SymbolFlags::DEF));
    assert!(flags.contains(SymbolFlags::USE));
    assert!(!flags.contains(SymbolFlags::PARAM));
}

#[test]
fn test_scope_define() {
    let mut scope = Scope::new(ScopeKind::Function, "test");
    scope.define("x", SymbolFlags::PARAM);
    scope.use_symbol("y");

    let x = scope.lookup("x").unwrap();
    assert!(x.flags.contains(SymbolFlags::DEF));
    assert!(x.flags.contains(SymbolFlags::PARAM));

    let y = scope.lookup("y").unwrap();
    assert!(y.flags.contains(SymbolFlags::USE));
    assert!(!y.flags.contains(SymbolFlags::DEF));
}

#[test]
fn test_symbol_classification() {
    let mut sym = Symbol::new("x");
    sym.flags |= SymbolFlags::DEF;
    assert!(sym.is_local());
    assert!(!sym.is_global());

    sym.flags |= SymbolFlags::GLOBAL_EXPLICIT;
    assert!(!sym.is_local());
    assert!(sym.is_global());
}

#[test]
fn test_ordered_cellvars_follow_closure_slots() {
    let mut scope = Scope::new(ScopeKind::Function, "test");

    let mut z = Symbol::new("z");
    z.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    z.closure_slot = Some(2);
    scope.symbols.insert(Arc::from("z"), z);

    let mut a = Symbol::new("a");
    a.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    a.closure_slot = Some(0);
    scope.symbols.insert(Arc::from("a"), a);

    let mut m = Symbol::new("m");
    m.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    m.closure_slot = Some(1);
    scope.symbols.insert(Arc::from("m"), m);

    let ordered = scope
        .ordered_cellvars()
        .into_iter()
        .map(|symbol| symbol.name.as_ref().to_owned())
        .collect::<Vec<_>>();

    assert_eq!(ordered, vec!["a", "m", "z"]);
}

#[test]
fn test_ordered_freevars_follow_closure_slots() {
    let mut scope = Scope::new(ScopeKind::Function, "test");

    let mut second = Symbol::new("second");
    second.flags |= SymbolFlags::FREE | SymbolFlags::USE;
    second.closure_slot = Some(3);
    scope.symbols.insert(Arc::from("second"), second);

    let mut first = Symbol::new("first");
    first.flags |= SymbolFlags::FREE | SymbolFlags::USE;
    first.closure_slot = Some(2);
    scope.symbols.insert(Arc::from("first"), first);

    let ordered = scope
        .ordered_freevars()
        .into_iter()
        .map(|symbol| symbol.name.as_ref().to_owned())
        .collect::<Vec<_>>();

    assert_eq!(ordered, vec!["first", "second"]);
}
