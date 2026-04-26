use super::*;
use crate::scope::{ScopeKind, SymbolFlags};

// =========================================================================
// ScopeRef Tests
// =========================================================================

#[test]
fn test_scope_ref_creation() {
    let mut scope = Scope::new(ScopeKind::Function, "test_func");
    scope.define("x", SymbolFlags::DEF);

    let scope_ref = ScopeRef::new(&scope);
    assert_eq!(scope_ref.cellvar_count(), 0);
    assert!(scope_ref.lookup("x").is_some());
    assert!(scope_ref.lookup("y").is_none());
}

#[test]
fn test_scope_ref_with_cellvars() {
    let mut scope = Scope::new(ScopeKind::Function, "outer");

    // Define a cell variable
    let sym = scope
        .symbols
        .entry(Arc::from("captured"))
        .or_insert_with(|| Symbol::new("captured"));
    sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    sym.closure_slot = Some(0);

    let scope_ref = ScopeRef::new(&scope);
    assert_eq!(scope_ref.cellvar_count(), 1);
}

// =========================================================================
// Variable Resolution Tests
// =========================================================================

#[test]
fn test_resolve_local_variable() {
    let mut scope = Scope::new(ScopeKind::Function, "func");

    // Define local variable
    let sym = scope
        .symbols
        .entry(Arc::from("x"))
        .or_insert_with(|| Symbol::new("x"));
    sym.flags |= SymbolFlags::DEF;
    sym.local_slot = Some(0);

    let scope_ref = ScopeRef::new(&scope);
    assert_eq!(scope_ref.resolve("x"), VarLocation::Local(0));
}

#[test]
fn test_resolve_cell_variable() {
    let mut scope = Scope::new(ScopeKind::Function, "outer");

    // Define cell variable (captured by inner scope)
    let sym = scope
        .symbols
        .entry(Arc::from("captured"))
        .or_insert_with(|| Symbol::new("captured"));
    sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    sym.closure_slot = Some(0);
    sym.local_slot = Some(0); // Also has local slot

    let scope_ref = ScopeRef::new(&scope);
    // Should prefer closure slot since it's a cell
    assert_eq!(scope_ref.resolve("captured"), VarLocation::Closure(0));
}

#[test]
fn test_resolve_free_variable() {
    let mut scope = Scope::new(ScopeKind::Function, "inner");

    // Define free variable (captured from outer scope)
    let sym = scope
        .symbols
        .entry(Arc::from("outer_var"))
        .or_insert_with(|| Symbol::new("outer_var"));
    sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
    sym.closure_slot = Some(0);

    let scope_ref = ScopeRef::new(&scope);
    assert_eq!(scope_ref.resolve("outer_var"), VarLocation::Closure(0));
}

#[test]
fn test_resolve_global_variable() {
    let mut scope = Scope::new(ScopeKind::Function, "func");

    // Define global variable
    let sym = scope
        .symbols
        .entry(Arc::from("global_var"))
        .or_insert_with(|| Symbol::new("global_var"));
    sym.flags |= SymbolFlags::USE | SymbolFlags::GLOBAL_EXPLICIT;

    let scope_ref = ScopeRef::new(&scope);
    assert_eq!(scope_ref.resolve("global_var"), VarLocation::Global);
}

#[test]
fn test_resolve_undefined_variable() {
    let scope = Scope::new(ScopeKind::Function, "func");
    let scope_ref = ScopeRef::new(&scope);

    // Undefined variable defaults to global
    assert_eq!(scope_ref.resolve("undefined"), VarLocation::Global);
}

// =========================================================================
// FunctionContext Tests
// =========================================================================

#[test]
fn test_function_context_no_closure() {
    let mut scope = Scope::new(ScopeKind::Function, "simple");
    scope.define("x", SymbolFlags::DEF);

    let ctx = FunctionContext::new(&scope);
    assert!(!ctx.has_closure());
    assert_eq!(ctx.cellvars().count(), 0);
    assert_eq!(ctx.freevars().count(), 0);
}

#[test]
fn test_function_context_with_closure() {
    let mut scope = Scope::new(ScopeKind::Function, "outer");

    // Add cell variable
    let sym = scope
        .symbols
        .entry(Arc::from("captured"))
        .or_insert_with(|| Symbol::new("captured"));
    sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;

    let ctx = FunctionContext::new(&scope);
    assert!(ctx.has_closure());
    assert_eq!(ctx.cellvars().count(), 1);
}

#[test]
fn test_function_context_resolve() {
    let mut scope = Scope::new(ScopeKind::Function, "func");

    // Local
    let sym = scope
        .symbols
        .entry(Arc::from("local"))
        .or_insert_with(|| Symbol::new("local"));
    sym.flags |= SymbolFlags::DEF;
    sym.local_slot = Some(0);

    // Cell
    let sym = scope
        .symbols
        .entry(Arc::from("cell"))
        .or_insert_with(|| Symbol::new("cell"));
    sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    sym.closure_slot = Some(0);

    let ctx = FunctionContext::new(&scope);
    assert_eq!(ctx.resolve_variable("local"), VarLocation::Local(0));
    assert_eq!(ctx.resolve_variable("cell"), VarLocation::Closure(0));
    assert_eq!(ctx.resolve_variable("unknown"), VarLocation::Global);
}

// =========================================================================
// ClosureInfo Tests
// =========================================================================

#[test]
fn test_closure_info_empty() {
    let scope = Scope::new(ScopeKind::Function, "simple");
    let ctx = FunctionContext::new(&scope);
    let info = ClosureInfo::from_context(&ctx);

    assert!(info.is_empty());
    assert_eq!(info.slot_count(), 0);
}

#[test]
fn test_closure_info_with_cells() {
    let mut scope = Scope::new(ScopeKind::Function, "outer");

    // Add two cell variables
    for (i, name) in ["a", "b"].iter().enumerate() {
        let sym = scope
            .symbols
            .entry(Arc::from(*name))
            .or_insert_with(|| Symbol::new(*name));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
        sym.closure_slot = Some(i as u16);
    }

    let ctx = FunctionContext::new(&scope);
    let info = ClosureInfo::from_context(&ctx);

    assert!(!info.is_empty());
    assert_eq!(info.cellvar_names.len(), 2);
    assert_eq!(info.freevar_names.len(), 0);
    assert_eq!(info.slot_count(), 2);
}

#[test]
fn test_closure_info_with_freevars() {
    let mut scope = Scope::new(ScopeKind::Function, "inner");

    // Add free variable
    let sym = scope
        .symbols
        .entry(Arc::from("outer_x"))
        .or_insert_with(|| Symbol::new("outer_x"));
    sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
    sym.closure_slot = Some(0);

    let ctx = FunctionContext::new(&scope);
    let info = ClosureInfo::from_context(&ctx);

    assert!(!info.is_empty());
    assert_eq!(info.cellvar_names.len(), 0);
    assert_eq!(info.freevar_names.len(), 1);
    assert_eq!(info.slot_count(), 1);
}

#[test]
fn test_closure_info_mixed() {
    let mut scope = Scope::new(ScopeKind::Function, "middle");

    // Cell (captured by grandchild)
    let sym = scope
        .symbols
        .entry(Arc::from("mine"))
        .or_insert_with(|| Symbol::new("mine"));
    sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    sym.closure_slot = Some(0);

    // Free (from parent)
    let sym = scope
        .symbols
        .entry(Arc::from("parents"))
        .or_insert_with(|| Symbol::new("parents"));
    sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
    sym.closure_slot = Some(1);

    let ctx = FunctionContext::new(&scope);
    let info = ClosureInfo::from_context(&ctx);

    assert!(!info.is_empty());
    assert_eq!(info.cellvar_names.len(), 1);
    assert_eq!(info.freevar_names.len(), 1);
    assert_eq!(info.slot_count(), 2);
}

#[test]
fn test_closure_info_orders_names_by_closure_slot() {
    let mut scope = Scope::new(ScopeKind::Function, "ordered");

    let late = scope
        .symbols
        .entry(Arc::from("late"))
        .or_insert_with(|| Symbol::new("late"));
    late.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    late.closure_slot = Some(1);

    let early = scope
        .symbols
        .entry(Arc::from("early"))
        .or_insert_with(|| Symbol::new("early"));
    early.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    early.closure_slot = Some(0);

    let captured = scope
        .symbols
        .entry(Arc::from("captured"))
        .or_insert_with(|| Symbol::new("captured"));
    captured.flags |= SymbolFlags::USE | SymbolFlags::FREE;
    captured.closure_slot = Some(2);

    let ctx = FunctionContext::new(&scope);
    let info = ClosureInfo::from_context(&ctx);

    assert_eq!(
        info.cellvar_names,
        vec![Arc::<str>::from("early"), Arc::<str>::from("late")]
    );
    assert_eq!(info.freevar_names, vec![Arc::<str>::from("captured")]);
}

// =========================================================================
// Variable Emitter Tests
// =========================================================================

#[test]
fn test_emit_load_local() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    builder.emit_load_var(r0, VarLocation::Local(5), None);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::LoadLocal as u8);
    assert_eq!(code.instructions[0].imm16(), 5);
}

#[test]
fn test_emit_load_closure() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    builder.emit_load_var(r0, VarLocation::Closure(3), None);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::LoadClosure as u8);
    assert_eq!(code.instructions[0].imm16(), 3);
}

#[test]
fn test_emit_load_global() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    builder.emit_load_var(r0, VarLocation::Global, Some("my_global"));

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::LoadGlobal as u8);
}

#[test]
fn test_emit_store_local() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    builder.emit_store_var(VarLocation::Local(2), r0, None);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::StoreLocal as u8);
    assert_eq!(code.instructions[0].imm16(), 2);
}

#[test]
fn test_emit_store_closure() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    builder.emit_store_var(VarLocation::Closure(1), r0, None);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::StoreClosure as u8);
    assert_eq!(code.instructions[0].imm16(), 1);
}

#[test]
fn test_emit_store_global() {
    let mut builder = FunctionBuilder::new("test");
    let r0 = builder.alloc_register();

    builder.emit_store_var(VarLocation::Global, r0, Some("glob"));

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::StoreGlobal as u8);
}

#[test]
fn test_emit_delete_local() {
    let mut builder = FunctionBuilder::new("test");

    builder.emit_delete_var(VarLocation::Local(4), None);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::DeleteLocal as u8);
    assert_eq!(code.instructions[0].imm16(), 4);
}

#[test]
fn test_emit_delete_closure() {
    let mut builder = FunctionBuilder::new("test");

    builder.emit_delete_var(VarLocation::Closure(2), None);

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::DeleteClosure as u8);
    assert_eq!(code.instructions[0].imm16(), 2);
}

#[test]
fn test_emit_delete_global() {
    let mut builder = FunctionBuilder::new("test");

    builder.emit_delete_var(VarLocation::Global, Some("to_delete"));

    let code = builder.finish();
    assert_eq!(code.instructions.len(), 1);

    use crate::bytecode::Opcode;
    assert_eq!(code.instructions[0].opcode(), Opcode::DeleteGlobal as u8);
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_complete_closure_pattern() {
    // Simulate outer function with captured variable
    let mut outer_scope = Scope::new(ScopeKind::Function, "make_counter");

    // Parameter 'start' - local and cell (captured by inner)
    let sym = outer_scope
        .symbols
        .entry(Arc::from("count"))
        .or_insert_with(|| Symbol::new("count"));
    sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    sym.local_slot = Some(0);
    sym.closure_slot = Some(0);

    let outer_ctx = FunctionContext::new(&outer_scope);
    assert!(outer_ctx.has_closure());

    // count should be accessed via closure since it's a cell
    assert_eq!(outer_ctx.resolve_variable("count"), VarLocation::Closure(0));

    // Simulate inner function with free variable
    let mut inner_scope = Scope::new(ScopeKind::Function, "increment");

    let sym = inner_scope
        .symbols
        .entry(Arc::from("count"))
        .or_insert_with(|| Symbol::new("count"));
    sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
    sym.closure_slot = Some(0);

    let inner_ctx = FunctionContext::new(&inner_scope);
    assert!(inner_ctx.has_closure());
    assert_eq!(inner_ctx.resolve_variable("count"), VarLocation::Closure(0));
}

#[test]
fn test_multiple_scope_levels() {
    // Three-level closure:
    // grandparent (defines x) -> parent (passes through) -> child (uses x)

    // Grandparent scope
    let mut gp_scope = Scope::new(ScopeKind::Function, "grandparent");
    let sym = gp_scope
        .symbols
        .entry(Arc::from("x"))
        .or_insert_with(|| Symbol::new("x"));
    sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    sym.local_slot = Some(0);
    sym.closure_slot = Some(0);

    let gp_ctx = FunctionContext::new(&gp_scope);
    assert!(gp_ctx.has_closure());
    assert_eq!(gp_ctx.resolve_variable("x"), VarLocation::Closure(0));

    // Parent scope - x is both FREE (from grandparent) and CELL (for child)
    let mut p_scope = Scope::new(ScopeKind::Function, "parent");
    let sym = p_scope
        .symbols
        .entry(Arc::from("x"))
        .or_insert_with(|| Symbol::new("x"));
    sym.flags |= SymbolFlags::FREE | SymbolFlags::CELL;
    sym.closure_slot = Some(0);

    let p_ctx = FunctionContext::new(&p_scope);
    assert!(p_ctx.has_closure());
    assert_eq!(p_ctx.resolve_variable("x"), VarLocation::Closure(0));

    // Child scope - x is just FREE
    let mut c_scope = Scope::new(ScopeKind::Function, "child");
    let sym = c_scope
        .symbols
        .entry(Arc::from("x"))
        .or_insert_with(|| Symbol::new("x"));
    sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
    sym.closure_slot = Some(0);

    let c_ctx = FunctionContext::new(&c_scope);
    assert!(c_ctx.has_closure());
    assert_eq!(c_ctx.resolve_variable("x"), VarLocation::Closure(0));
}

#[test]
fn test_mixed_variable_types() {
    // Function with all variable types
    let mut scope = Scope::new(ScopeKind::Function, "mixed");

    // Pure local
    let sym = scope
        .symbols
        .entry(Arc::from("local"))
        .or_insert_with(|| Symbol::new("local"));
    sym.flags |= SymbolFlags::DEF;
    sym.local_slot = Some(0);

    // Cell variable
    let sym = scope
        .symbols
        .entry(Arc::from("cell"))
        .or_insert_with(|| Symbol::new("cell"));
    sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
    sym.closure_slot = Some(0);

    // Free variable
    let sym = scope
        .symbols
        .entry(Arc::from("free"))
        .or_insert_with(|| Symbol::new("free"));
    sym.flags |= SymbolFlags::FREE;
    sym.closure_slot = Some(1);

    // Explicit global
    let sym = scope
        .symbols
        .entry(Arc::from("glob"))
        .or_insert_with(|| Symbol::new("glob"));
    sym.flags |= SymbolFlags::GLOBAL_EXPLICIT;

    let ctx = FunctionContext::new(&scope);

    assert_eq!(ctx.resolve_variable("local"), VarLocation::Local(0));
    assert_eq!(ctx.resolve_variable("cell"), VarLocation::Closure(0));
    assert_eq!(ctx.resolve_variable("free"), VarLocation::Closure(1));
    assert_eq!(ctx.resolve_variable("glob"), VarLocation::Global);
    assert_eq!(ctx.resolve_variable("unknown"), VarLocation::Global);
}
