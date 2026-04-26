use super::*;

impl Compiler {
    // =========================================================================
    // Function Definition Compilation
    // =========================================================================

    /// Compile positional default expressions into a tuple register.
    pub(super) fn compile_positional_defaults_tuple(
        &mut self,
        defaults: &[Expr],
    ) -> CompileResult<Option<Register>> {
        if defaults.is_empty() {
            return Ok(None);
        }
        if defaults.len() > u8::MAX as usize {
            return Err(CompileError {
                message: "too many positional defaults".to_string(),
                line: self.line_for_span(defaults[0].span),
                column: 0,
            });
        }

        let count = defaults.len() as u8;
        let first = self.builder.alloc_register_block(count);
        for (i, expr) in defaults.iter().enumerate() {
            let dst = Register::new(first.0 + i as u8);
            let tmp = self.compile_expr(expr)?;
            if tmp != dst {
                self.builder.emit_move(dst, tmp);
            }
            self.builder.free_register(tmp);
        }

        let tuple_reg = self.builder.alloc_register();
        self.builder.emit_build_tuple(tuple_reg, first, count);
        self.builder.free_register_block(first, count);
        Ok(Some(tuple_reg))
    }

    /// Compile keyword-only defaults into a dict register (name -> default value).
    pub(super) fn compile_kw_defaults_dict(
        &mut self,
        kwonlyargs: &[prism_parser::ast::Arg],
        kw_defaults: &[Option<Expr>],
    ) -> CompileResult<Option<Register>> {
        if kwonlyargs.len() != kw_defaults.len() {
            return Err(CompileError {
                message: "internal error: kwonly args/defaults length mismatch".to_string(),
                line: 0,
                column: 0,
            });
        }

        let mut entries: Vec<(&str, &Expr)> = Vec::new();
        for (arg, default_expr) in kwonlyargs.iter().zip(kw_defaults.iter()) {
            if let Some(expr) = default_expr {
                entries.push((arg.arg.as_str(), expr));
            }
        }

        if entries.is_empty() {
            return Ok(None);
        }
        if entries.len() > (u8::MAX as usize / 2) {
            return Err(CompileError {
                message: "too many keyword-only defaults".to_string(),
                line: self.line_for_span(entries[0].1.span),
                column: 0,
            });
        }

        let pair_count = entries.len() as u8;
        let pair_regs = pair_count
            .checked_mul(2)
            .expect("pair_count bounded by u8::MAX/2");
        let first_pair = self.builder.alloc_register_block(pair_regs);

        for (i, (name, value_expr)) in entries.iter().enumerate() {
            let key_reg = Register::new(first_pair.0 + (i as u8 * 2));
            let value_reg = Register::new(key_reg.0 + 1);

            let key_idx = self.builder.add_string(*name);
            self.builder.emit_load_const(key_reg, key_idx);

            let value_tmp = self.compile_expr(value_expr)?;
            if value_tmp != value_reg {
                self.builder.emit_move(value_reg, value_tmp);
            }
            self.builder.free_register(value_tmp);
        }

        let dict_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::new(
            Opcode::BuildDict,
            dict_reg.0,
            first_pair.0,
            pair_count,
        ));
        self.builder.free_register_block(first_pair, pair_regs);
        Ok(Some(dict_reg))
    }

    /// Compile a function definition (FunctionDef or AsyncFunctionDef).
    ///
    /// This creates a nested CodeObject for the function body and emits
    /// MakeFunction or MakeClosure opcode to create the function object.
    ///
    /// # Arguments
    ///
    /// * `name` - Function name
    /// * `args` - Function arguments specification
    /// * `body` - Function body statements
    /// * `decorator_list` - Decorators to apply
    /// * `is_async` - Whether this is an async function
    pub(super) fn compile_function_def(
        &mut self,
        name: &str,
        args: &prism_parser::ast::Arguments,
        body: &[Stmt],
        decorator_list: &[Expr],
        is_async: bool,
        definition_line: u32,
    ) -> CompileResult<()> {
        // Find the scope for this function from the symbol table
        // We need to look it up by name in the current scope's children
        let func_scope_idx = self.find_child_scope(ScopeKind::Function, name);
        let (func_cellvars, func_freevars, func_locals, scope_has_yield) =
            if let Some(scope_idx) = func_scope_idx {
                let scope = &self.current_scope().children[scope_idx];
                let cellvars = Self::ordered_cellvar_names(scope);
                let freevars = Self::ordered_freevar_names(scope);
                let mut locals = scope
                    .locals()
                    .map(|sym| Arc::from(sym.name.as_ref()))
                    .collect::<Vec<Arc<str>>>();
                locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
                (cellvars, freevars, locals, scope.has_yield)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), false)
            };

        // Create a new FunctionBuilder for the function body
        let mut func_builder = FunctionBuilder::new(name);
        func_builder.set_filename(&*self.filename);
        func_builder.set_first_lineno(definition_line);

        // Set function flags
        if is_async {
            func_builder.add_flags(CodeFlags::COROUTINE);
        }

        // Count parameters
        let posonly_count = args.posonlyargs.len() as u16;
        let kwonly_count = args.kwonlyargs.len() as u16;
        let total_positional = args.posonlyargs.len() + args.args.len();

        // Set parameter counts on the builder
        func_builder.set_arg_count(total_positional as u16);
        func_builder.set_kwonlyarg_count(kwonly_count);
        func_builder.set_posonlyarg_count(posonly_count);

        // Handle varargs and kwargs
        if args.vararg.is_some() {
            func_builder.add_flags(CodeFlags::VARARGS);
        }
        if args.kwarg.is_some() {
            func_builder.add_flags(CodeFlags::VARKEYWORDS);
        }

        // Register parameters as locals (they occupy the first slots)
        // Python parameter order: posonly, regular args, vararg, kwonly, kwarg

        // Position-only parameters
        for arg in &args.posonlyargs {
            func_builder.define_local(arg.arg.as_str());
        }

        // Regular positional parameters
        for arg in &args.args {
            func_builder.define_local(arg.arg.as_str());
        }

        // *args
        if let Some(ref vararg) = args.vararg {
            func_builder.define_local(vararg.arg.as_str());
        }

        // Keyword-only parameters
        for arg in &args.kwonlyargs {
            func_builder.define_local(arg.arg.as_str());
        }

        // **kwargs
        if let Some(ref kwarg) = args.kwarg {
            func_builder.define_local(kwarg.arg.as_str());
        }

        // Register non-cell locals from scope analysis so nested captures can
        // reliably resolve by name from `code.locals`.
        for name in func_locals {
            func_builder.define_local(name);
        }

        // Register cell and free variables from scope analysis.
        // Cellvars are materialized per invocation, while freevars require
        // definition-time capture from the enclosing scope.
        let has_cellvars = !func_cellvars.is_empty();
        let captures_freevars = !func_freevars.is_empty();
        // Cell variables: locals captured by inner functions
        for name in func_cellvars {
            func_builder.add_cellvar(name);
        }

        // Free variables: captured from outer scopes
        for name in func_freevars {
            func_builder.add_freevar(name);
        }

        if has_cellvars {
            func_builder.add_flags(CodeFlags::HAS_CELLVARS);
        }
        if captures_freevars {
            func_builder.add_flags(CodeFlags::HAS_FREEVARS);
        }

        // Set generator flag from scope analysis
        if scope_has_yield {
            if is_async {
                // async def with yield = async generator
                func_builder.add_flags(CodeFlags::ASYNC_GENERATOR);
            } else {
                // regular generator
                func_builder.add_flags(CodeFlags::GENERATOR);
            }
        }

        if has_cellvars || captures_freevars {
            func_builder.add_flags(CodeFlags::NESTED);
        }

        // Swap builders to compile function body
        let parent_builder = std::mem::replace(&mut self.builder, func_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);

        // Save and update context for function body compilation
        let parent_async_context = self.in_async_context;
        let parent_function_context = self.in_function_context;
        self.in_async_context = is_async;
        self.in_function_context = true;
        if let Some(scope_idx) = func_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        // Compile function body
        for (index, stmt) in body.iter().enumerate() {
            if self.should_strip_docstring_stmt(index, stmt) {
                continue;
            }
            self.compile_stmt(stmt)?;
        }

        if func_scope_idx.is_some() {
            self.exit_child_scope();
        }

        // Ensure function returns None if no explicit return
        self.builder.emit_return_none();

        // Restore contexts
        self.in_async_context = parent_async_context;
        self.in_function_context = parent_function_context;
        self.finally_stack = parent_finally_stack;

        // Swap back and get finished function code
        let func_builder = std::mem::replace(&mut self.builder, parent_builder);
        let func_code = func_builder.finish();

        // Store the nested CodeObject as a constant
        let code_const_idx = self.builder.add_code_object(Arc::new(func_code));

        // Compile decorators in reverse order (they'll wrap the function)
        // Decorators are compiled first, then applied after function creation
        let decorator_regs: Vec<Register> = decorator_list
            .iter()
            .map(|d| self.compile_expr(d))
            .collect::<Result<_, _>>()?;

        let positional_defaults_reg = self.compile_positional_defaults_tuple(&args.defaults)?;
        let kw_defaults_reg = self.compile_kw_defaults_dict(&args.kwonlyargs, &args.kw_defaults)?;

        // Emit function/closure creation
        let func_reg = self.builder.alloc_register();

        if captures_freevars {
            // MakeClosure is only needed when freevars must be captured from
            // the enclosing scope. Cellvars are created fresh for each call.
            self.builder.emit(Instruction::op_di(
                Opcode::MakeClosure,
                func_reg,
                code_const_idx,
            ));
        } else {
            // MakeFunction: simple function without captures
            self.builder.emit(Instruction::op_di(
                Opcode::MakeFunction,
                func_reg,
                code_const_idx,
            ));
        }

        if positional_defaults_reg.is_some() || kw_defaults_reg.is_some() {
            let none_reg = if positional_defaults_reg.is_none() || kw_defaults_reg.is_none() {
                let reg = self.builder.alloc_register();
                self.builder.emit_load_none(reg);
                Some(reg)
            } else {
                None
            };
            let positional_reg = positional_defaults_reg
                .or(none_reg)
                .expect("positional defaults register must exist");
            let kw_reg = kw_defaults_reg
                .or(none_reg)
                .expect("keyword defaults register must exist");
            self.builder
                .emit_set_function_defaults(func_reg, positional_reg, kw_reg);

            if let Some(reg) = positional_defaults_reg {
                self.builder.free_register(reg);
            }
            if let Some(reg) = kw_defaults_reg {
                self.builder.free_register(reg);
            }
            if let Some(reg) = none_reg {
                self.builder.free_register(reg);
            }
        }

        // Apply decorators in reverse order
        // @decorator1
        // @decorator2
        // def func(): ...
        // is equivalent to: func = decorator1(decorator2(func))
        for decorator_reg in decorator_regs.into_iter().rev() {
            // Decorator calls need a dedicated contiguous block [result, arg0]
            // so the argument register cannot alias any live temporary reused
            // by the allocator between decorated definitions.
            let call_block = self.builder.alloc_register_block(2);
            self.builder
                .emit_move(Register::new(call_block.0 + 1), func_reg);
            self.builder.emit(Instruction::op_dss(
                Opcode::Call,
                call_block,
                decorator_reg,
                Register::new(1), // 1 argument
            ));
            self.builder.emit_move(func_reg, call_block);
            self.builder.free_register_block(call_block, 2);
            self.builder.free_register(decorator_reg);
        }

        // Store function using lexical scope resolution.
        let location = self.resolve_variable(name);
        self.builder.emit_store_var(location, func_reg, Some(name));
        self.builder.free_register(func_reg);

        Ok(())
    }

    /// Find a child scope by kind and name in the current scope.
    ///
    /// Uses a per-scope cursor so repeated nested definitions with the same
    /// name (e.g. redefinitions) resolve deterministically in source order.
    pub(super) fn find_child_scope(&mut self, kind: ScopeKind, name: &str) -> Option<usize> {
        let depth = self.scope_path.len();
        let start = *self.scope_child_offsets.get(depth).unwrap_or(&0);
        let child_count = self.current_scope().children.len();

        let mut found = None;
        for idx in start..child_count {
            let child = &self.current_scope().children[idx];
            if child.kind == kind && child.name.as_ref() == name {
                found = Some(idx);
                break;
            }
        }

        if found.is_none() {
            for idx in 0..start.min(child_count) {
                let child = &self.current_scope().children[idx];
                if child.kind == kind && child.name.as_ref() == name {
                    found = Some(idx);
                    break;
                }
            }
        }

        if let Some(idx) = found {
            if let Some(offset) = self.scope_child_offsets.get_mut(depth) {
                *offset = idx + 1;
            }
            Some(idx)
        } else {
            None
        }
    }

    /// Advance to the next comprehension child scope in source order.
    ///
    /// Comprehensions participate in the same child-scope cursor as functions,
    /// lambdas, and classes. Even comprehensions that are currently compiled
    /// inline must still consume their analyzed scope entry so later sibling
    /// scopes remain aligned with the AST traversal order.
    pub(super) fn next_comprehension_scope(&mut self, name: &str) -> Option<usize> {
        self.find_child_scope(ScopeKind::Comprehension, name)
    }

    // =========================================================================
    // Lambda Expression Compilation
    // =========================================================================

    /// Compile a lambda expression.
    ///
    /// Lambda expressions create nested code objects like functions, but with:
    /// - Single expression body (not statements)
    /// - Implicit return of expression result
    /// - Anonymous name (`<lambda>`)
    /// - Inherits async context from enclosing scope
    ///
    /// # Performance Optimizations
    /// - Uses register-based evaluation for body expression
    /// - Direct return without intermediate storage
    /// - Closure handling only when capturing variables
    pub(super) fn compile_lambda(
        &mut self,
        args: &prism_parser::ast::Arguments,
        body: &Expr,
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        // Find lambda scope from symbol table (lambdas are named "<lambda>" in scope analysis)
        let lambda_scope_idx = self.find_child_scope(ScopeKind::Lambda, "<lambda>");
        let (lambda_cellvars, lambda_freevars, lambda_locals, lambda_has_yield) =
            if let Some(scope_idx) = lambda_scope_idx {
                let scope = &self.current_scope().children[scope_idx];
                let cellvars = Self::ordered_cellvar_names(scope);
                let freevars = Self::ordered_freevar_names(scope);
                let mut locals = scope
                    .locals()
                    .map(|sym| Arc::from(sym.name.as_ref()))
                    .collect::<Vec<Arc<str>>>();
                locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
                (cellvars, freevars, locals, scope.has_yield)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), false)
            };

        // Create a new FunctionBuilder for the lambda body
        let mut lambda_builder = FunctionBuilder::new("<lambda>");
        lambda_builder.set_filename(&*self.filename);
        lambda_builder.set_first_lineno(definition_line);

        // Calculate argument counts
        let posonly_count = args.posonlyargs.len() as u16;
        let regular_args = args.args.len() as u16;
        let total_positional = posonly_count + regular_args;
        let kwonly_count = args.kwonlyargs.len() as u16;

        // Set parameter counts
        lambda_builder.set_arg_count(total_positional);
        lambda_builder.set_kwonlyarg_count(kwonly_count);
        lambda_builder.set_posonlyarg_count(posonly_count);

        // Handle varargs and kwargs
        if args.vararg.is_some() {
            lambda_builder.add_flags(CodeFlags::VARARGS);
        }
        if args.kwarg.is_some() {
            lambda_builder.add_flags(CodeFlags::VARKEYWORDS);
        }
        if lambda_has_yield {
            lambda_builder.add_flags(CodeFlags::GENERATOR);
        }

        // Register parameters as locals
        for arg in &args.posonlyargs {
            lambda_builder.define_local(arg.arg.as_str());
        }
        for arg in &args.args {
            lambda_builder.define_local(arg.arg.as_str());
        }
        if let Some(ref vararg) = args.vararg {
            lambda_builder.define_local(vararg.arg.as_str());
        }
        for arg in &args.kwonlyargs {
            lambda_builder.define_local(arg.arg.as_str());
        }
        if let Some(ref kwarg) = args.kwarg {
            lambda_builder.define_local(kwarg.arg.as_str());
        }

        for name in lambda_locals {
            lambda_builder.define_local(name);
        }

        // Register cell and free variables from scope analysis.
        // Cellvars are invocation-local; only freevars require MakeClosure.
        let has_cellvars = !lambda_cellvars.is_empty();
        let captures_freevars = !lambda_freevars.is_empty();
        for name in lambda_cellvars {
            lambda_builder.add_cellvar(name);
        }
        for name in lambda_freevars {
            lambda_builder.add_freevar(name);
        }

        if has_cellvars {
            lambda_builder.add_flags(CodeFlags::HAS_CELLVARS);
        }
        if captures_freevars {
            lambda_builder.add_flags(CodeFlags::HAS_FREEVARS);
        }

        // Swap builders to compile lambda body
        let parent_builder = std::mem::replace(&mut self.builder, lambda_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);
        let parent_async_context = self.in_async_context;
        let parent_function_context = self.in_function_context;
        // Lambda inherits async context from enclosing scope but sets function context
        self.in_function_context = true;
        if let Some(scope_idx) = lambda_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        // Compile the expression body
        let result_reg = self.compile_expr(body)?;

        if lambda_scope_idx.is_some() {
            self.exit_child_scope();
        }

        // Emit implicit return of the expression result
        self.builder.emit_return(result_reg);

        // Restore parent contexts
        self.in_async_context = parent_async_context;
        self.in_function_context = parent_function_context;
        self.finally_stack = parent_finally_stack;

        // Swap back and get finished lambda code
        let lambda_builder = std::mem::replace(&mut self.builder, parent_builder);
        let lambda_code = lambda_builder.finish();

        // Store the nested CodeObject as a constant
        let code_const_idx = self.builder.add_code_object(Arc::new(lambda_code));

        let positional_defaults_reg = self.compile_positional_defaults_tuple(&args.defaults)?;
        let kw_defaults_reg = self.compile_kw_defaults_dict(&args.kwonlyargs, &args.kw_defaults)?;

        // Emit function/closure creation
        if captures_freevars {
            self.builder
                .emit(Instruction::op_di(Opcode::MakeClosure, dst, code_const_idx));
        } else {
            self.builder.emit(Instruction::op_di(
                Opcode::MakeFunction,
                dst,
                code_const_idx,
            ));
        }

        if positional_defaults_reg.is_some() || kw_defaults_reg.is_some() {
            let none_reg = if positional_defaults_reg.is_none() || kw_defaults_reg.is_none() {
                let reg = self.builder.alloc_register();
                self.builder.emit_load_none(reg);
                Some(reg)
            } else {
                None
            };
            let positional_reg = positional_defaults_reg
                .or(none_reg)
                .expect("positional defaults register must exist");
            let kw_reg = kw_defaults_reg
                .or(none_reg)
                .expect("keyword defaults register must exist");
            self.builder
                .emit_set_function_defaults(dst, positional_reg, kw_reg);

            if let Some(reg) = positional_defaults_reg {
                self.builder.free_register(reg);
            }
            if let Some(reg) = kw_defaults_reg {
                self.builder.free_register(reg);
            }
            if let Some(reg) = none_reg {
                self.builder.free_register(reg);
            }
        }

        Ok(dst)
    }
}
