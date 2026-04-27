use super::*;

/// Kind of comprehension being compiled.
#[derive(Debug, Clone, Copy)]
pub(super) enum ComprehensionKind {
    List,
    Set,
}

impl Compiler {
    // =========================================================================
    // Comprehension Expression Compilation
    // =========================================================================

    /// Compile a list comprehension.
    ///
    /// List comprehensions create a nested scope (as a hidden function) to prevent
    /// loop variables from leaking into the enclosing scope. This matches Python 3
    /// semantics.
    ///
    /// # Bytecode Strategy
    /// 1. Create a hidden function containing the comprehension logic
    /// 2. Inside: create empty list, iterate with FOR_ITER, append elements
    /// 3. Call the hidden function with the first iterator
    /// 4. Result is the completed list
    ///
    /// # Performance Optimizations
    /// - Uses LIST_APPEND opcode for O(1) amortized append
    /// - Inlines filter conditions to avoid function call overhead
    /// - Pre-allocates result register for minimal register pressure
    pub(super) fn compile_listcomp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        self.compile_sequence_comprehension(
            "<listcomp>",
            "<comprehension>",
            elt,
            generators,
            dst,
            ComprehensionKind::List,
            definition_line,
        )
    }

    /// Compile a set comprehension.
    pub(super) fn compile_setcomp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        self.compile_sequence_comprehension(
            "<setcomp>",
            "<comprehension>",
            elt,
            generators,
            dst,
            ComprehensionKind::Set,
            definition_line,
        )
    }

    /// Compile a dict comprehension.
    pub(super) fn compile_dictcomp(
        &mut self,
        key: &Expr,
        value: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        let comp_scope_idx = self.next_comprehension_scope("<dictcomp>");
        let (cellvars, freevars, locals) = self.comprehension_scope_layout(comp_scope_idx);
        let captures_freevars = !freevars.is_empty();

        let mut comp_builder = FunctionBuilder::new("<dictcomp>");
        comp_builder.set_filename(&*self.filename);
        comp_builder.set_first_lineno(definition_line);
        comp_builder.set_arg_count(1);
        comp_builder.define_local(".0");
        for name in locals {
            comp_builder.define_local(name);
        }
        Self::configure_closure_layout(&mut comp_builder, cellvars, freevars);

        let parent_builder = std::mem::replace(&mut self.builder, comp_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);
        if let Some(scope_idx) = comp_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        let iter_reg = self.builder.alloc_register();
        self.builder
            .emit_load_local(iter_reg, crate::bytecode::LocalSlot::new(0));

        let dict_reg = self.builder.alloc_register();
        self.builder
            .emit(Instruction::op_d(Opcode::BuildDict, dict_reg));
        self.compile_dict_comprehension_generators(key, value, generators, dict_reg, 0, iter_reg)?;
        self.builder.emit_return(dict_reg);
        self.builder.free_register(dict_reg);
        self.builder.free_register(iter_reg);

        if comp_scope_idx.is_some() {
            self.exit_child_scope();
        }

        self.finally_stack = parent_finally_stack;
        let comp_builder = std::mem::replace(&mut self.builder, parent_builder);
        let comp_code = comp_builder.finish();
        self.emit_comprehension_call(
            comp_code,
            captures_freevars,
            &generators[0].iter,
            generators[0].is_async,
            dst,
        )
    }

    pub(super) fn compile_sequence_comprehension(
        &mut self,
        code_name: &'static str,
        scope_name: &'static str,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        kind: ComprehensionKind,
        definition_line: u32,
    ) -> CompileResult<Register> {
        let comp_scope_idx = self.next_comprehension_scope(scope_name);
        let (cellvars, freevars, locals) = self.comprehension_scope_layout(comp_scope_idx);
        let captures_freevars = !freevars.is_empty();

        let mut comp_builder = FunctionBuilder::new(code_name);
        comp_builder.set_filename(&*self.filename);
        comp_builder.set_first_lineno(definition_line);
        comp_builder.set_arg_count(1);
        comp_builder.define_local(".0");
        for name in locals {
            comp_builder.define_local(name);
        }
        Self::configure_closure_layout(&mut comp_builder, cellvars, freevars);

        let parent_builder = std::mem::replace(&mut self.builder, comp_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);
        if let Some(scope_idx) = comp_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        let iter_reg = self.builder.alloc_register();
        self.builder
            .emit_load_local(iter_reg, crate::bytecode::LocalSlot::new(0));

        let result_reg = self.builder.alloc_register();
        match kind {
            ComprehensionKind::List => self.builder.emit_build_list(result_reg, result_reg, 0),
            ComprehensionKind::Set => self
                .builder
                .emit(Instruction::op_d(Opcode::BuildSet, result_reg)),
        }
        self.compile_comprehension_generators(elt, generators, result_reg, kind, 0, iter_reg)?;
        self.builder.emit_return(result_reg);
        self.builder.free_register(result_reg);
        self.builder.free_register(iter_reg);

        if comp_scope_idx.is_some() {
            self.exit_child_scope();
        }

        self.finally_stack = parent_finally_stack;
        let comp_builder = std::mem::replace(&mut self.builder, parent_builder);
        let comp_code = comp_builder.finish();
        self.emit_comprehension_call(
            comp_code,
            captures_freevars,
            &generators[0].iter,
            generators[0].is_async,
            dst,
        )
    }

    pub(super) fn comprehension_scope_layout(
        &self,
        scope_idx: Option<usize>,
    ) -> (Vec<Arc<str>>, Vec<Arc<str>>, Vec<Arc<str>>) {
        if let Some(scope_idx) = scope_idx {
            let scope = &self.current_scope().children[scope_idx];
            let cellvars = Self::ordered_cellvar_names(scope);
            let freevars = Self::ordered_freevar_names(scope);
            let mut locals = scope
                .locals()
                .map(|sym| Arc::from(sym.name.as_ref()))
                .collect::<Vec<Arc<str>>>();
            locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
            (cellvars, freevars, locals)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        }
    }

    pub(super) fn configure_closure_layout(
        builder: &mut FunctionBuilder,
        cellvars: Vec<Arc<str>>,
        freevars: Vec<Arc<str>>,
    ) {
        let has_cellvars = !cellvars.is_empty();
        let captures_freevars = !freevars.is_empty();

        for name in cellvars {
            builder.add_cellvar(name);
        }
        for name in freevars {
            builder.add_freevar(name);
        }

        if has_cellvars {
            builder.add_flags(CodeFlags::HAS_CELLVARS);
        }
        if captures_freevars {
            builder.add_flags(CodeFlags::HAS_FREEVARS);
        }
    }

    pub(super) fn emit_comprehension_call(
        &mut self,
        comp_code: CodeObject,
        captures_freevars: bool,
        first_iter_expr: &Expr,
        first_iter_is_async: bool,
        dst: Register,
    ) -> CompileResult<Register> {
        let code_idx = self.builder.add_code_object(Arc::new(comp_code));
        let mut func_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_di(
            if captures_freevars {
                Opcode::MakeClosure
            } else {
                Opcode::MakeFunction
            },
            func_reg,
            code_idx,
        ));

        let call_block = self.builder.alloc_register_block(2);
        if func_reg.0 >= call_block.0 && func_reg.0 < call_block.0 + 2 {
            let safe_reg = self.builder.alloc_register();
            self.builder.emit_move(safe_reg, func_reg);
            self.builder.free_register(func_reg);
            func_reg = safe_reg;
        }

        let first_iter = self.compile_expr(first_iter_expr)?;
        let arg_reg = Register::new(call_block.0 + 1);
        if first_iter_is_async {
            if !self.in_async_context {
                return Err(CompileError {
                    message: "asynchronous comprehension outside of an async function".to_string(),
                    line: 0,
                    column: 0,
                });
            }
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAIter, arg_reg, first_iter));
        } else {
            self.builder.emit_get_iter(arg_reg, first_iter);
        }
        self.builder.free_register(first_iter);

        self.builder.emit_call(call_block, func_reg, 1);
        if call_block != dst {
            self.builder.emit_move(dst, call_block);
        }
        self.builder.free_register(func_reg);
        self.builder.free_register_block(call_block, 2);

        Ok(dst)
    }

    /// Compile a generator expression.
    ///
    /// Generator expressions are lazy - they create a generator function that
    /// yields values on demand. This is more memory efficient for large sequences.
    pub(super) fn compile_genexp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        let gen_scope_idx = self.next_comprehension_scope("<comprehension>");
        let (gen_cellvars, gen_freevars, gen_locals) = if let Some(scope_idx) = gen_scope_idx {
            let scope = &self.current_scope().children[scope_idx];
            let cellvars = Self::ordered_cellvar_names(scope);
            let freevars = Self::ordered_freevar_names(scope);
            let mut locals = scope
                .locals()
                .map(|sym| Arc::from(sym.name.as_ref()))
                .collect::<Vec<Arc<str>>>();
            locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
            (cellvars, freevars, locals)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // Create a generator function that yields each element
        let mut gen_builder = FunctionBuilder::new("<genexpr>");
        gen_builder.set_filename(&*self.filename);
        gen_builder.set_first_lineno(definition_line);
        gen_builder.add_flags(CodeFlags::GENERATOR);

        // First iterator is passed as argument
        gen_builder.set_arg_count(1);
        gen_builder.define_local(".0"); // Hidden argument for first iterator
        for name in gen_locals {
            gen_builder.define_local(name);
        }

        let has_cellvars = !gen_cellvars.is_empty();
        let captures_freevars = !gen_freevars.is_empty();
        for name in gen_cellvars {
            gen_builder.add_cellvar(name);
        }
        for name in gen_freevars {
            gen_builder.add_freevar(name);
        }

        if has_cellvars {
            gen_builder.add_flags(CodeFlags::HAS_CELLVARS);
        }
        if captures_freevars {
            gen_builder.add_flags(CodeFlags::HAS_FREEVARS);
        }

        // Swap builders
        let parent_builder = std::mem::replace(&mut self.builder, gen_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);
        if let Some(scope_idx) = gen_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        // Get the first iterator from argument
        let iter_reg = self.builder.alloc_register();
        self.builder
            .emit_load_local(iter_reg, crate::bytecode::LocalSlot::new(0));

        // Compile generator loops (yields instead of appending)
        self.compile_genexp_generators(elt, generators, 0, iter_reg)?;
        self.builder.free_register(iter_reg);

        // Return None at end
        self.builder.emit_return_none();

        if gen_scope_idx.is_some() {
            self.exit_child_scope();
        }

        self.finally_stack = parent_finally_stack;
        // Swap back
        let gen_builder = std::mem::replace(&mut self.builder, parent_builder);
        let gen_code = gen_builder.finish();

        // Store code object and create function
        let code_idx = self.builder.add_code_object(Arc::new(gen_code));
        let mut func_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_di(
            if captures_freevars {
                Opcode::MakeClosure
            } else {
                Opcode::MakeFunction
            },
            func_reg,
            code_idx,
        ));

        // Reserve a fresh contiguous block [result, arg0] so the iterator
        // argument cannot overwrite the callable register before the call.
        let call_block = self.builder.alloc_register_block(2);
        if func_reg.0 >= call_block.0 && func_reg.0 < call_block.0 + 2 {
            let safe_reg = self.builder.alloc_register();
            self.builder.emit_move(safe_reg, func_reg);
            self.builder.free_register(func_reg);
            func_reg = safe_reg;
        }

        // Compile first iterator and pass it in the dedicated arg slot.
        let first_iter = self.compile_expr(&generators[0].iter)?;
        let arg_reg = Register::new(call_block.0 + 1);
        if generators[0].is_async {
            if !self.in_async_context {
                return Err(CompileError {
                    message: "asynchronous comprehension outside of an async function".to_string(),
                    line: 0,
                    column: 0,
                });
            }
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAIter, arg_reg, first_iter));
        } else {
            self.builder.emit_get_iter(arg_reg, first_iter);
        }
        self.builder.free_register(first_iter);

        self.builder.emit_call(call_block, func_reg, 1);
        if call_block != dst {
            self.builder.emit_move(dst, call_block);
        }
        self.builder.free_register(func_reg);
        self.builder.free_register_block(call_block, 2);

        Ok(dst)
    }

    /// Helper to compile comprehension generators (list/set).
    pub(super) fn compile_comprehension_generators(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        result_reg: Register,
        kind: ComprehensionKind,
        depth: usize,
        first_iter_reg: Register,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: compute element and add to collection
            let elem_reg = self.compile_expr(elt)?;
            match kind {
                ComprehensionKind::List => {
                    // ListAppend: src1.append(src2) - list in src1, element in src2
                    self.builder.emit(Instruction::op_dss(
                        Opcode::ListAppend,
                        Register(0), // dst unused for ListAppend
                        result_reg,  // src1 = list
                        elem_reg,    // src2 = element
                    ));
                }
                ComprehensionKind::Set => {
                    // SetAdd: src1.add(src2) - set in src1, element in src2
                    self.builder.emit(Instruction::op_dss(
                        Opcode::SetAdd,
                        Register(0), // dst unused for SetAdd
                        result_reg,  // src1 = set
                        elem_reg,    // src2 = element
                    ));
                }
            }
            self.builder.free_register(elem_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        let loop_regs = self.builder.alloc_register_block(2);
        let iter_reg = loop_regs;
        let item_reg = Register::new(loop_regs.0 + 1);

        if depth == 0 {
            self.builder.emit_move(iter_reg, first_iter_reg);
        } else {
            // Compile iterator
            let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;

            // Get iterator (sync or async)
            if comp_gen.is_async {
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "asynchronous comprehension outside of an async function"
                            .to_string(),
                        line: 0,
                        column: 0,
                    });
                }
                self.builder.emit(Instruction::op_ds(
                    Opcode::GetAIter,
                    iter_reg,
                    iter_expr_reg,
                ));
            } else {
                self.builder.emit_get_iter(iter_reg, iter_expr_reg);
            }
            self.builder.free_register(iter_expr_reg);
        }

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        if comp_gen.is_async {
            self.builder
                .emit(Instruction::op_ds(Opcode::GetANext, item_reg, iter_reg));
            // await the result
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAwaitable, item_reg, item_reg));
            self.emit_yield_from(item_reg, item_reg);
        } else {
            self.builder.emit_for_iter(item_reg, loop_end);
        }

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators or emit element
        self.compile_comprehension_generators(
            elt,
            rest,
            result_reg,
            kind,
            depth + 1,
            Register::new(0),
        )?;

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        self.builder.free_register(iter_reg);
        self.builder.free_register(item_reg);

        Ok(())
    }

    /// Helper to compile dict comprehension generators.
    pub(super) fn compile_dict_comprehension_generators(
        &mut self,
        key: &Expr,
        value: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        result_reg: Register,
        depth: usize,
        first_iter_reg: Register,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: compute key-value and add to dict
            let key_reg = self.compile_expr(key)?;
            let val_reg = self.compile_expr(value)?;
            self.builder.emit_set_item(result_reg, key_reg, val_reg);
            self.builder.free_register(key_reg);
            self.builder.free_register(val_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        let loop_regs = self.builder.alloc_register_block(2);
        let iter_reg = loop_regs;
        let item_reg = Register::new(loop_regs.0 + 1);

        if depth == 0 {
            self.builder.emit_move(iter_reg, first_iter_reg);
        } else {
            // Compile iterator
            let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;

            if comp_gen.is_async {
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "asynchronous comprehension outside of an async function"
                            .to_string(),
                        line: 0,
                        column: 0,
                    });
                }
                self.builder.emit(Instruction::op_ds(
                    Opcode::GetAIter,
                    iter_reg,
                    iter_expr_reg,
                ));
            } else {
                self.builder.emit_get_iter(iter_reg, iter_expr_reg);
            }
            self.builder.free_register(iter_expr_reg);
        }

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        if comp_gen.is_async {
            self.builder
                .emit(Instruction::op_ds(Opcode::GetANext, item_reg, iter_reg));
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAwaitable, item_reg, item_reg));
            self.emit_yield_from(item_reg, item_reg);
        } else {
            self.builder.emit_for_iter(item_reg, loop_end);
        }

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators
        self.compile_dict_comprehension_generators(
            key,
            value,
            rest,
            result_reg,
            depth + 1,
            Register::new(0),
        )?;

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        self.builder.free_register(iter_reg);
        self.builder.free_register(item_reg);

        Ok(())
    }

    /// Helper to compile generator expression generators.
    pub(super) fn compile_genexp_generators(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        depth: usize,
        iter_reg: Register,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: yield element
            let elem_reg = self.compile_expr(elt)?;
            let yield_result = self.builder.alloc_register();
            self.builder
                .emit(Instruction::op_ds(Opcode::Yield, yield_result, elem_reg));
            self.builder.free_register(yield_result);
            self.builder.free_register(elem_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        // For depth > 0, compile iterator; depth 0 uses passed-in iter_reg
        let loop_regs = self.builder.alloc_register_block(2);
        let actual_iter = loop_regs;
        let item_reg = Register::new(loop_regs.0 + 1);
        if depth == 0 {
            self.builder.emit_move(actual_iter, iter_reg);
        } else {
            let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;
            self.builder.emit_get_iter(actual_iter, iter_expr_reg);
            self.builder.free_register(iter_expr_reg);
        }

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        self.builder.emit_for_iter(item_reg, loop_end);

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators.
        // Deeper recursion compiles each nested iterator expression in the
        // current loop scope, so only the root generator consumes `iter_reg`.
        self.compile_genexp_generators(elt, rest, depth + 1, Register::new(0))?;

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        self.builder.free_register(item_reg);
        self.builder.free_register(actual_iter);

        Ok(())
    }
}
