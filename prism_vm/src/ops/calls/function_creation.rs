use super::*;

// =============================================================================
// Function Creation
// =============================================================================

/// MakeFunction: create function from code object
/// dst = function, imm16 = code constant index
///
/// Creates a FunctionObject from a code constant and stores it in dst.
#[inline(always)]
pub fn make_function(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let code_idx = inst.imm16();
    let dst = inst.dst().0;

    // Get code object from constant pool (release frame borrow immediately)
    let code_clone = {
        let frame = vm.current_frame();
        match load_code_constant(frame, code_idx, "function") {
            Ok(code) => code,
            Err(err) => return ControlFlow::Error(err),
        }
    };

    // Create FunctionObject
    let func = new_function_object(vm, code_clone, None);

    let func_value = alloc_value_in_current_heap_or_box(func);
    vm.current_frame_mut().set_reg(dst, func_value);
    ControlFlow::Continue
}

/// MakeClosure: create closure with captured variables
///
/// Creates a FunctionObject with captured freevars from the enclosing scope.
#[inline(always)]
pub fn make_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let code_idx = inst.imm16();
    let dst = inst.dst().0;

    // Get code object from constant pool (release frame borrow immediately)
    let code_clone = {
        let frame = vm.current_frame();
        match load_code_constant(frame, code_idx, "closure") {
            Ok(code) => code,
            Err(err) => return ControlFlow::Error(err),
        }
    };

    let captured_closure = match capture_function_freevars(vm.current_frame(), &code_clone) {
        Ok(env) => env,
        Err(err) => return ControlFlow::Error(err),
    };

    // Create FunctionObject
    let func = new_function_object(vm, code_clone, captured_closure);

    let func_value = alloc_value_in_current_heap_or_box(func);
    vm.current_frame_mut().set_reg(dst, func_value);
    ControlFlow::Continue
}

#[inline]
pub(super) fn materialize_function_invocation_closure(
    func: &FunctionObject,
    code: &Arc<CodeObject>,
) -> Result<Option<Arc<ClosureEnv>>, RuntimeError> {
    let captured_freevars = func.closure().map(Arc::clone);
    materialize_invocation_closure(code, captured_freevars)
}

#[inline]
fn materialize_invocation_closure(
    code: &Arc<CodeObject>,
    captured_freevars: Option<Arc<ClosureEnv>>,
) -> Result<Option<Arc<ClosureEnv>>, RuntimeError> {
    let captured_freevar_count = captured_freevars.as_ref().map_or(0, |env| env.len());
    if captured_freevar_count != code.freevars.len() {
        return Err(RuntimeError::internal(format!(
            "closure environment mismatch in {}: expected {} freevars, found {}",
            code.qualname,
            code.freevars.len(),
            captured_freevar_count
        )));
    }

    if code.cellvars.is_empty() {
        return Ok(captured_freevars.filter(|env| !env.is_empty()));
    }

    let mut cells = Vec::with_capacity(code.cellvars.len() + captured_freevar_count);
    for _ in code.cellvars.iter() {
        cells.push(Arc::new(Cell::unbound()));
    }

    if let Some(captured_freevars) = captured_freevars.as_ref() {
        for idx in 0..captured_freevars.len() {
            cells.push(Arc::clone(captured_freevars.get_cell(idx)));
        }
    }

    Ok(Some(Arc::new(ClosureEnv::new(cells))))
}

#[inline]
fn capture_function_freevars(
    frame: &crate::frame::Frame,
    code: &Arc<CodeObject>,
) -> Result<Option<Arc<ClosureEnv>>, RuntimeError> {
    if code.freevars.is_empty() {
        return Ok(None);
    }

    let mut cells = Vec::with_capacity(code.freevars.len());
    for freevar in code.freevars.iter() {
        if let Some(cell) = capture_parent_cell(frame, freevar.as_ref()) {
            cells.push(cell);
        } else {
            return Err(RuntimeError::name_error(Arc::clone(freevar)));
        }
    }

    Ok(Some(Arc::new(ClosureEnv::new(cells))))
}

/// Capture a full frame closure environment for nested non-function scopes.
pub(crate) fn capture_closure_environment(
    frame: &crate::frame::Frame,
    code: &Arc<CodeObject>,
) -> Result<Arc<ClosureEnv>, RuntimeError> {
    let mut cells = Vec::with_capacity(code.cellvars.len() + code.freevars.len());

    // Child cellvars start as unbound cells and become bound when assigned.
    for _ in code.cellvars.iter() {
        cells.push(Arc::new(Cell::unbound()));
    }

    // Freevars capture existing cells from the parent frame when available.
    // If the parent variable is still a plain local, promote the current value
    // into a new captured cell.
    for freevar in code.freevars.iter() {
        if let Some(cell) = capture_parent_cell(frame, freevar.as_ref()) {
            cells.push(cell);
        } else {
            return Err(RuntimeError::name_error(Arc::clone(freevar)));
        }
    }

    Ok(Arc::new(ClosureEnv::new(cells)))
}

/// Resolve a free variable from the parent frame to a capture cell.
fn capture_parent_cell(frame: &crate::frame::Frame, name: &str) -> Option<Arc<Cell>> {
    if let Some(slot) = find_parent_closure_slot(&frame.code, name) {
        if let Some(env) = &frame.closure {
            if slot < env.len() {
                let cell = Arc::clone(env.get_cell(slot));
                // Parameter-backed cellvars may start unbound and are initialized in
                // registers first; sync once on first capture.
                if cell.get().is_none() {
                    if let Some(local_slot) = find_local_slot(&frame.code, name) {
                        if local_slot < parameter_local_count(&frame.code)
                            && local_slot <= u8::MAX as usize
                        {
                            cell.set(frame.get_reg(local_slot as u8));
                        }
                    }
                }
                return Some(cell);
            }
        }
    }

    find_local_slot(&frame.code, name).map(|slot| {
        let value = frame.get_reg(slot as u8);
        Arc::new(Cell::new(value))
    })
}

/// Find a local slot index for `name` in a code object.
fn find_local_slot(code: &CodeObject, name: &str) -> Option<usize> {
    code.locals.iter().position(|local| local.as_ref() == name)
}

/// Find closure slot index for `name` in the parent frame.
///
/// Closure slot layout is `[cellvars..., freevars...]`.
fn find_parent_closure_slot(code: &CodeObject, name: &str) -> Option<usize> {
    if let Some(idx) = code.cellvars.iter().position(|n| n.as_ref() == name) {
        return Some(idx);
    }
    code.freevars
        .iter()
        .position(|n| n.as_ref() == name)
        .map(|idx| code.cellvars.len() + idx)
}

/// Count initialized parameter slots in local layout.
#[inline]
fn parameter_local_count(code: &CodeObject) -> usize {
    let mut count = code.arg_count as usize + code.kwonlyarg_count as usize;
    if code.flags.contains(CodeFlags::VARARGS) {
        count += 1;
    }
    if code.flags.contains(CodeFlags::VARKEYWORDS) {
        count += 1;
    }
    count
}

/// Initialize closure cellvars from already-populated local slots.
///
/// Parameters are first written into local registers during call setup, while
/// cellvars are accessed through the closure environment. This bridge keeps
/// parameter-backed cells bound before closure capture/load occurs.
pub(crate) fn initialize_closure_cellvars_from_locals(
    frame: &mut crate::frame::Frame,
    initialized_locals: usize,
) {
    if initialized_locals == 0 || frame.code.cellvars.is_empty() || frame.code.locals.is_empty() {
        return;
    }
    let Some(env) = frame.closure.as_ref() else {
        return;
    };

    for (cell_idx, cell_name) in frame.code.cellvars.iter().enumerate() {
        let Some(local_slot) = frame
            .code
            .locals
            .iter()
            .position(|name| name.as_ref() == cell_name.as_ref())
        else {
            continue;
        };
        if local_slot >= initialized_locals {
            continue;
        }
        let value = frame.get_local(local_slot as u16);
        env.set(cell_idx, value);
    }
}
