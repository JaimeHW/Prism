use super::*;

impl Compiler {
    // =========================================================================
    // Exception Handling Compilation
    // =========================================================================

    /// Compile a try/except/finally statement with zero-cost exception handling.
    ///
    /// This generates exception table entries for the VM's table-driven unwinder.
    /// No runtime opcodes are executed on try block entry/exit - the exception
    /// table is consulted only when an exception is raised.
    ///
    /// # Layout
    /// ```text
    /// try_start:
    ///     <try body>              # Protected by exception entry
    ///     JUMP end_label          # Skip handlers on normal exit
    /// handler_0:                  # except Type1 as e:
    ///     <check exception type>
    ///     <handler body>
    ///     JUMP end_label
    /// handler_1:                  # except Type2:
    ///     <handler body>
    ///     JUMP end_label
    /// finally:                    # finally:
    ///     <finally body>
    /// end_label:
    /// ```
    pub(super) fn compile_try(
        &mut self,
        body: &[Stmt],
        handlers: &[ExceptHandler],
        orelse: &[Stmt],
        finalbody: &[Stmt],
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        // =================================================================
        // Analysis Phase - Determine handler structure
        // =================================================================

        // Check if there's a bare except clause (catches all exceptions)
        let has_bare_except = handlers.iter().any(|h| h.typ.is_none());

        // Check if there are any typed handlers that need matching
        let has_typed_handlers = handlers.iter().any(|h| h.typ.is_some());

        // =================================================================
        // Label Creation Phase
        // =================================================================

        let end_label = self.builder.create_label();

        let orelse_label = if !orelse.is_empty() {
            Some(self.builder.create_label())
        } else {
            None
        };

        let finally_label = if !finalbody.is_empty() {
            Some(self.builder.create_label())
        } else {
            None
        };

        // Only create reraise label if we have typed handlers AND no bare except
        // (if there's a bare except, it will catch everything, so no reraise needed)
        let reraise_label = if has_typed_handlers && !has_bare_except {
            Some(self.builder.create_label())
        } else {
            None
        };

        // Create handler labels (one per except clause)
        let handler_labels: Vec<_> = handlers
            .iter()
            .map(|_| self.builder.create_label())
            .collect();

        let has_finally = !finalbody.is_empty();
        if has_finally {
            let return_label = self.builder.create_label();
            let return_value_reg = self.builder.alloc_register();
            self.finally_stack.push(FinallyContext {
                return_label,
                return_value_reg,
                return_used: false,
                jump_continuations: SmallVec::new(),
            });
        }

        // =================================================================
        // Try Body Compilation
        // =================================================================

        let try_start_pc = self.builder.current_pc();
        let stack_depth = self.builder.current_stack_depth();

        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        let try_end_pc = self.builder.current_pc();

        // Jump to else/finally/end on normal completion (no exception)
        if let Some(else_label) = orelse_label {
            self.builder.emit_jump(else_label);
        } else if let Some(fin_label) = finally_label {
            self.builder.emit_jump(fin_label);
        } else if !handlers.is_empty() {
            self.builder.emit_jump(end_label);
        }

        // =================================================================
        // Exception Handler Compilation
        // =================================================================

        for (i, handler) in handlers.iter().enumerate() {
            self.builder.bind_label(handler_labels[i]);

            let handler_start_pc = self.builder.current_pc();
            let handler_abort_label = self.builder.create_label();

            // Compile handler match logic
            if let Some(type_expr) = &handler.typ {
                // -----------------------------------------------------------
                // Typed handler: except SomeException as e:
                // -----------------------------------------------------------

                // Compile the exception type expression to get the type class
                let type_reg = self.compile_expr(type_expr)?;

                // Load the current exception into a register for later binding
                let exc_reg = self.builder.alloc_register();
                self.builder
                    .emit(Instruction::op_d(Opcode::LoadException, exc_reg));

                // Check if exception matches type using dynamic matching
                // Note: ExceptionMatch reads src1 as the type, gets exception from VM state
                let match_reg = self.builder.alloc_register();
                self.builder.emit(Instruction::op_ds(
                    Opcode::ExceptionMatch,
                    match_reg,
                    type_reg,
                ));

                // Determine where to jump if no match
                let no_match_target = if i + 1 < handlers.len() {
                    // Try next handler
                    handler_labels[i + 1]
                } else if let Some(reraise_lbl) = reraise_label {
                    // No more handlers, reraise the exception
                    reraise_lbl
                } else if let Some(fin_label) = finally_label {
                    // No reraise needed, go to finally (bare except will catch)
                    fin_label
                } else {
                    // Should not happen if has_bare_except is true
                    end_label
                };

                self.builder.emit_jump_if_false(match_reg, no_match_target);

                self.builder.free_register(match_reg);
                self.builder.free_register(type_reg);

                // If handler has a name binding (except E as e:), store the exception
                if let Some(name) = &handler.name {
                    let location = self.resolve_variable(name);
                    self.builder
                        .emit_store_var(location, exc_reg, Some(name.as_ref()));
                }

                self.builder.free_register(exc_reg);
            } else {
                // -----------------------------------------------------------
                // Bare except: catches all exceptions
                // -----------------------------------------------------------

                if let Some(name) = &handler.name {
                    let exc_reg = self.builder.alloc_register();
                    self.builder
                        .emit(Instruction::op_d(Opcode::LoadException, exc_reg));
                    let location = self.resolve_variable(name);
                    self.builder
                        .emit_store_var(location, exc_reg, Some(name.as_ref()));
                    self.builder.free_register(exc_reg);
                }
            };

            // =============================================================
            // Handler Body Execution
            // =============================================================

            self.builder.emit(Instruction::op(Opcode::EnterExcept));
            let handler_body_start_pc = self.builder.current_pc();

            // Compile handler body
            for stmt in &handler.body {
                self.compile_stmt(stmt)?;
            }
            let handler_body_end_pc = self.builder.current_pc();

            // Successful handler completion restores any outer handler context
            // and fully clears the exception when this was the outermost handler.
            self.builder.emit(Instruction::op(Opcode::ExitExcept));

            // Jump to finally or end after successful handler execution
            if let Some(fin_label) = finally_label {
                self.builder.emit_jump(fin_label);
            } else {
                self.builder.emit_jump(end_label);
            }

            self.builder.bind_label(handler_abort_label);
            let handler_abort_pc = self.builder.current_pc();
            self.builder.emit(Instruction::op(Opcode::AbortExcept));
            self.builder.emit(Instruction::op(Opcode::Reraise));

            // Add exception entry for this handler
            self.builder.add_exception_entry(ExceptionEntry {
                start_pc: try_start_pc,
                end_pc: try_end_pc,
                handler_pc: handler_start_pc,
                finally_pc: u32::MAX,
                depth: stack_depth as u16,
                exception_type_idx: u16::MAX,
            });

            if handler_body_start_pc < handler_body_end_pc {
                self.builder.add_exception_entry(ExceptionEntry {
                    start_pc: handler_body_start_pc,
                    end_pc: handler_body_end_pc,
                    handler_pc: handler_abort_pc,
                    finally_pc: u32::MAX,
                    depth: stack_depth as u16,
                    exception_type_idx: u16::MAX,
                });
            }
        }

        // =================================================================
        // Else Block Compilation (runs only if no exception occurred)
        // =================================================================

        if let Some(else_label) = orelse_label {
            self.builder.bind_label(else_label);
            for stmt in orelse {
                self.compile_stmt(stmt)?;
            }
            if let Some(fin_label) = finally_label {
                self.builder.emit_jump(fin_label);
            } else {
                self.builder.emit_jump(end_label);
            }
        }

        // =================================================================
        // Reraise Path (only if typed handlers exist without bare except)
        // =================================================================

        if let Some(reraise_lbl) = reraise_label {
            self.builder.bind_label(reraise_lbl);

            if let Some(fin_label) = finally_label {
                // Execute finally before reraising
                self.builder.emit_jump(fin_label);
            } else {
                // No finally, reraise immediately
                self.builder.emit(Instruction::op(Opcode::Reraise));
            }
        }

        // =================================================================
        // Finally Block Compilation
        // =================================================================

        let cleanup_context = if has_finally {
            Some(
                self.finally_stack
                    .pop()
                    .expect("try/finally should have an active finally context"),
            )
        } else {
            None
        };

        if let Some(fin_label) = finally_label {
            self.builder.bind_label(fin_label);
            let finally_start_pc = self.builder.current_pc();

            // Push exception info to preserve state during finally execution
            self.builder.emit(Instruction::op(Opcode::PushExcInfo));

            // Compile finally body
            for stmt in finalbody {
                self.compile_stmt(stmt)?;
            }

            // Pop exception info
            self.builder.emit(Instruction::op(Opcode::PopExcInfo));

            // EndFinally will reraise if there's a pending exception
            self.builder.emit(Instruction::op(Opcode::EndFinally));
            self.builder.emit_jump(end_label);

            // Add finally exception entry
            self.builder.add_exception_entry(ExceptionEntry {
                start_pc: try_start_pc,
                end_pc: try_end_pc,
                handler_pc: finally_start_pc,
                finally_pc: finally_start_pc,
                depth: stack_depth as u16,
                exception_type_idx: u16::MAX,
            });
        }

        if let Some(cleanup_context) = cleanup_context {
            if cleanup_context.return_used {
                self.builder.bind_label(cleanup_context.return_label);
                self.compile_finally_cleanup_body(finalbody)?;
                self.emit_return_value(cleanup_context.return_value_reg);
            } else {
                self.builder.free_register(cleanup_context.return_value_reg);
            }

            for continuation in cleanup_context.jump_continuations {
                self.builder.bind_label(continuation.cleanup_label);
                self.compile_finally_cleanup_body(finalbody)?;
                self.emit_jump_through_finally_until(
                    continuation.target_label,
                    continuation.preserve_finally_depth,
                );
            }
        }

        // =================================================================
        // End Label - Normal exit point
        // =================================================================

        self.builder.bind_label(end_label);

        Ok(())
    }

    // =========================================================================
    // With Statement (Context Manager) Compilation
    // =========================================================================

    /// Compile a with statement.
    ///
    /// The with statement implements the context manager protocol:
    ///
    /// ```python
    /// with expr as var:
    ///     body
    /// ```
    ///
    /// Is equivalent to:
    /// ```python
    /// mgr = expr
    /// value = mgr.__enter__()
    /// try:
    ///     var = value  # if as clause present
    ///     body
    /// except:
    ///     if not mgr.__exit__(*sys.exc_info()):
    ///         raise
    /// else:
    ///     mgr.__exit__(None, None, None)
    /// ```
    ///
    /// For multiple context managers, they are nested from left to right:
    /// ```python
    /// with a as x, b as y:
    ///     body
    /// # is equivalent to:
    /// with a as x:
    ///     with b as y:
    ///         body
    /// ```
    pub(super) fn compile_with(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
    ) -> CompileResult<()> {
        // Compile nested context managers recursively
        self.compile_with_items(items, body, 0)
    }

    /// Compile with statement items recursively for nested context managers.
    pub(super) fn compile_with_items(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
        depth: usize,
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        if depth >= items.len() {
            // All context managers set up, compile the body
            for stmt in body {
                self.compile_stmt(stmt)?;
            }
            return Ok(());
        }

        let item = &items[depth];

        // Step 1: Evaluate context expression -> mgr
        let mgr_reg = self.compile_expr(&item.context_expr)?;

        // Step 2: Look up __enter__ and __exit__ methods
        let enter_name_idx = self.builder.add_name("__enter__");
        let exit_name_idx = self.builder.add_name("__exit__");

        // Step 3: Load __exit__ method (need to store for cleanup)
        // We store both the manager and __exit__ bound method for cleanup
        let exit_method_reg = self.builder.alloc_register_block(5);
        self.builder
            .emit_load_method(exit_method_reg, mgr_reg, exit_name_idx);

        // Step 4: Load __enter__ method
        let enter_method_reg = self.builder.alloc_register_block(2);
        self.builder
            .emit_load_method(enter_method_reg, mgr_reg, enter_name_idx);

        // Step 5: Call __enter__() -> value
        let value_reg = self.builder.alloc_register();
        self.builder
            .emit_call_method(value_reg, enter_method_reg, 0);
        self.builder.free_register_block(enter_method_reg, 2);

        // Step 6: If there's an as-clause, bind the value
        if let Some(optional_vars) = &item.optional_vars {
            self.compile_store(optional_vars, value_reg)?;
        }
        self.builder.free_register(value_reg);

        // Step 7: Set up exception handling for the body
        // Record try block start position
        let try_start_pc = self.builder.current_pc();
        let cleanup_label = self.builder.create_label();
        let end_label = self.builder.create_label();
        let return_label = self.builder.create_label();
        let return_value_reg = self.builder.alloc_register();
        self.finally_stack.push(FinallyContext {
            return_label,
            return_value_reg,
            return_used: false,
            jump_continuations: SmallVec::new(),
        });

        // Step 8: Compile nested items and body
        self.compile_with_items(items, body, depth + 1)?;

        // Step 9: Record try block end position (normal exit path)
        let try_end_pc = self.builder.current_pc();
        let cleanup_context = self
            .finally_stack
            .pop()
            .expect("with statement should have an active cleanup context");

        // Step 10: Normal exit - call __exit__(None, None, None)
        self.emit_context_exit_none(exit_method_reg);

        // Jump to end (skip exception path)
        self.builder.emit_jump(end_label);

        // Step 11: Exception cleanup path
        self.builder.bind_label(cleanup_label);
        let cleanup_start_pc = self.builder.current_pc();

        // Push exception info for cleanup
        self.builder.emit(Instruction::op(Opcode::PushExcInfo));

        // Load exception info registers
        let exc_type_reg = self.builder.alloc_register();
        let exc_val_reg = self.builder.alloc_register();
        let exc_tb_reg = self.builder.alloc_register();

        // Load the active exception instance and compute its concrete class for
        // __exit__(exc_type, exc, tb) compatibility with unittest/assertRaises.
        self.builder
            .emit(Instruction::op_d(Opcode::LoadException, exc_val_reg));
        self.emit_exception_type_attr(exc_type_reg, exc_val_reg);
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, exc_tb_reg));

        // Call __exit__(type, value, tb)
        let suppress_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_method_reg.0 + 2),
            exc_type_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_method_reg.0 + 3),
            exc_val_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_method_reg.0 + 4),
            exc_tb_reg,
        ));
        self.builder
            .emit_call_method(suppress_reg, exit_method_reg, 3);

        // Pop exception info
        self.builder.emit(Instruction::op(Opcode::PopExcInfo));

        // A truthy __exit__ suppresses the active exception entirely.
        let suppress_label = self.builder.create_label();
        self.builder.emit_jump_if_true(suppress_reg, suppress_label);

        self.builder.free_register(exc_type_reg);
        self.builder.free_register(exc_val_reg);
        self.builder.free_register(exc_tb_reg);
        self.builder.free_register(suppress_reg);

        // Reraise the exception
        self.builder.emit(Instruction::op(Opcode::Reraise));

        self.builder.bind_label(suppress_label);
        self.builder.emit(Instruction::op(Opcode::ClearException));
        self.builder.emit_jump(end_label);

        if cleanup_context.return_used {
            self.builder.bind_label(cleanup_context.return_label);
            self.emit_context_exit_none(exit_method_reg);
            self.emit_return_value(cleanup_context.return_value_reg);
        } else {
            self.builder.free_register(cleanup_context.return_value_reg);
        }

        for continuation in cleanup_context.jump_continuations {
            self.builder.bind_label(continuation.cleanup_label);
            self.emit_context_exit_none(exit_method_reg);
            self.emit_jump_through_finally_until(
                continuation.target_label,
                continuation.preserve_finally_depth,
            );
        }

        // Step 12: End label
        self.builder.bind_label(end_label);

        // Free the stored method and manager registers
        self.builder.free_register_block(exit_method_reg, 5);
        self.builder.free_register(mgr_reg);

        // Step 13: Add exception table entry for cleanup
        self.builder.add_exception_entry(ExceptionEntry {
            start_pc: try_start_pc,
            end_pc: try_end_pc,
            handler_pc: cleanup_start_pc,
            finally_pc: u32::MAX, // No separate finally, cleanup handles both
            depth: depth as u16,
            exception_type_idx: u16::MAX, // Catches all exceptions
        });

        Ok(())
    }

    // =========================================================================
    // Async With Statement Compilation
    // =========================================================================

    /// Compile async with statement with awaited __aenter__/__aexit__.
    ///
    /// `async with ctx as var:` compiles to roughly:
    ///   mgr = ctx
    ///   aexit = mgr.__aexit__
    ///   aenter = mgr.__aenter__
    ///   val = await aenter()
    ///   var = val
    ///   try:
    ///       <body>
    ///   except:
    ///       if not await aexit(type, val, tb): raise
    ///   else:
    ///       await aexit(None, None, None)
    pub(super) fn compile_async_with(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
    ) -> CompileResult<()> {
        self.compile_async_with_items(items, body, 0)
    }

    /// Compile async with statement items recursively for nested async context managers.
    pub(super) fn compile_async_with_items(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
        depth: usize,
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        if depth >= items.len() {
            // All async context managers set up, compile the body
            for stmt in body {
                self.compile_stmt(stmt)?;
            }
            return Ok(());
        }

        let item = &items[depth];

        // Step 1: Evaluate context expression -> mgr
        let mgr_reg = self.compile_expr(&item.context_expr)?;

        // Step 2: Look up __aenter__ and __aexit__ methods
        let aenter_name_idx = self.builder.add_name("__aenter__");
        let aexit_name_idx = self.builder.add_name("__aexit__");

        // Step 3: Load __aexit__ method (need to store for cleanup)
        let aexit_method_reg = self.builder.alloc_register_block(5);
        self.builder
            .emit_load_method(aexit_method_reg, mgr_reg, aexit_name_idx);

        // Step 4: Load __aenter__ method
        let aenter_method_reg = self.builder.alloc_register_block(2);
        self.builder
            .emit_load_method(aenter_method_reg, mgr_reg, aenter_name_idx);

        // Step 5: Call __aenter__() and AWAIT the result
        let aenter_awaitable_reg = self.builder.alloc_register();
        self.builder
            .emit_call_method(aenter_awaitable_reg, aenter_method_reg, 0);
        self.builder.free_register_block(aenter_method_reg, 2);

        // Await the __aenter__ result: GetAwaitable + YieldFrom
        let value_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::GetAwaitable,
            value_reg,
            aenter_awaitable_reg,
        ));
        self.builder.free_register(aenter_awaitable_reg);
        self.emit_yield_from(value_reg, value_reg);

        // Step 6: If there's an as-clause, bind the value
        if let Some(optional_vars) = &item.optional_vars {
            self.compile_store(optional_vars, value_reg)?;
        }
        self.builder.free_register(value_reg);

        // Step 7: Set up exception handling for the body
        let try_start_pc = self.builder.current_pc();
        let cleanup_label = self.builder.create_label();
        let end_label = self.builder.create_label();
        let return_label = self.builder.create_label();
        let return_value_reg = self.builder.alloc_register();
        self.finally_stack.push(FinallyContext {
            return_label,
            return_value_reg,
            return_used: false,
            jump_continuations: SmallVec::new(),
        });

        // Step 8: Compile nested items and body
        self.compile_async_with_items(items, body, depth + 1)?;

        // Step 9: Normal exit path
        let try_end_pc = self.builder.current_pc();
        let cleanup_context = self
            .finally_stack
            .pop()
            .expect("async with statement should have an active cleanup context");

        self.emit_async_context_exit_none(aexit_method_reg);

        // Jump to end (skip exception path)
        self.builder.emit_jump(end_label);

        // Step 10: Exception cleanup path
        self.builder.bind_label(cleanup_label);
        let cleanup_start_pc = self.builder.current_pc();

        // Push exception info for cleanup
        self.builder.emit(Instruction::op(Opcode::PushExcInfo));

        // Load exception info registers
        let exc_type_reg = self.builder.alloc_register();
        let exc_val_reg = self.builder.alloc_register();
        let exc_tb_reg = self.builder.alloc_register();

        self.builder
            .emit(Instruction::op_d(Opcode::LoadException, exc_val_reg));
        self.emit_exception_type_attr(exc_type_reg, exc_val_reg);
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, exc_tb_reg));

        // Call __aexit__(type, value, tb)
        let suppress_awaitable_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_method_reg.0 + 2),
            exc_type_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_method_reg.0 + 3),
            exc_val_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_method_reg.0 + 4),
            exc_tb_reg,
        ));
        self.builder
            .emit_call_method(suppress_awaitable_reg, aexit_method_reg, 3);

        // Await the __aexit__ result for exception case
        let suppress_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::GetAwaitable,
            suppress_reg,
            suppress_awaitable_reg,
        ));
        self.builder.free_register(suppress_awaitable_reg);
        self.emit_yield_from(suppress_reg, suppress_reg);

        // Pop exception info
        self.builder.emit(Instruction::op(Opcode::PopExcInfo));

        // A truthy __aexit__ suppresses the active exception entirely.
        let suppress_label = self.builder.create_label();
        self.builder.emit_jump_if_true(suppress_reg, suppress_label);

        self.builder.free_register(exc_type_reg);
        self.builder.free_register(exc_val_reg);
        self.builder.free_register(exc_tb_reg);
        self.builder.free_register(suppress_reg);

        // Reraise the exception
        self.builder.emit(Instruction::op(Opcode::Reraise));

        self.builder.bind_label(suppress_label);
        self.builder.emit(Instruction::op(Opcode::ClearException));
        self.builder.emit_jump(end_label);

        if cleanup_context.return_used {
            self.builder.bind_label(cleanup_context.return_label);
            self.emit_async_context_exit_none(aexit_method_reg);
            self.emit_return_value(cleanup_context.return_value_reg);
        } else {
            self.builder.free_register(cleanup_context.return_value_reg);
        }

        for continuation in cleanup_context.jump_continuations {
            self.builder.bind_label(continuation.cleanup_label);
            self.emit_async_context_exit_none(aexit_method_reg);
            self.emit_jump_through_finally_until(
                continuation.target_label,
                continuation.preserve_finally_depth,
            );
        }

        // Step 11: End label
        self.builder.bind_label(end_label);

        // Free the stored method and manager registers
        self.builder.free_register_block(aexit_method_reg, 5);
        self.builder.free_register(mgr_reg);

        // Step 12: Add exception table entry for cleanup
        self.builder.add_exception_entry(ExceptionEntry {
            start_pc: try_start_pc,
            end_pc: try_end_pc,
            handler_pc: cleanup_start_pc,
            finally_pc: u32::MAX,
            depth: depth as u16,
            exception_type_idx: u16::MAX,
        });

        Ok(())
    }

    // =========================================================================
    // Match Statement (Pattern Matching) Compilation
    // =========================================================================

    /// Compile a match statement using Maranget's decision tree algorithm.
    ///
    /// This implements Python 3.10+ structural pattern matching (PEP 634).
    /// The algorithm:
    /// 1. Evaluate the subject expression once
    /// 2. Build a pattern matrix from all cases
    /// 3. Generate a decision tree for optimal pattern testing
    /// 4. Emit bytecode that traverses the decision tree
    pub(super) fn compile_match(
        &mut self,
        subject: &Expr,
        cases: &[prism_parser::ast::MatchCase],
    ) -> CompileResult<()> {
        // Step 1: Compile subject expression and store in register
        let subject_reg = self.compile_expr(subject)?;

        // Step 2: Create labels for each case and the end
        let end_label = self.builder.create_label();
        let case_labels: Vec<Label> = cases.iter().map(|_| self.builder.create_label()).collect();

        // Step 3: Compile pattern tests and bindings for each case
        // We compile cases in order, with fallthrough to next case on failure
        for (i, case) in cases.iter().enumerate() {
            let next_label = if i + 1 < cases.len() {
                case_labels[i + 1]
            } else {
                end_label
            };

            // Compile pattern match
            self.compile_pattern_match(&case.pattern, subject_reg, next_label)?;

            // Compile guard if present
            if let Some(guard) = &case.guard {
                let guard_reg = self.compile_expr(guard)?;
                self.builder.emit_jump_if_false(guard_reg, next_label);
                self.builder.free_register(guard_reg);
            }

            // Compile case body if pattern (and guard) matched
            for stmt in &case.body {
                self.compile_stmt(stmt)?;
            }

            // Jump to end after executing matched case
            self.builder.emit_jump(end_label);

            // Bind next case label for fallthrough
            if i + 1 < cases.len() {
                self.builder.bind_label(case_labels[i + 1]);
            }
        }

        // End label
        self.builder.bind_label(end_label);
        self.builder.free_register(subject_reg);

        Ok(())
    }

    /// Compile a pattern match test.
    ///
    /// On success, any bindings are stored to locals and execution continues.
    /// On failure, jumps to fail_label.
    pub(super) fn compile_pattern_match(
        &mut self,
        pattern: &prism_parser::ast::Pattern,
        subject_reg: Register,
        fail_label: Label,
    ) -> CompileResult<()> {
        use prism_parser::ast::PatternKind;

        match &pattern.kind {
            PatternKind::MatchValue(expr) => {
                // Value pattern: subject == expr
                let value_reg = self.compile_expr(expr)?;
                let result_reg = self.builder.alloc_register();
                self.builder.emit_eq(result_reg, subject_reg, value_reg);
                self.builder.emit_jump_if_false(result_reg, fail_label);
                self.builder.free_register(result_reg);
                self.builder.free_register(value_reg);
            }

            PatternKind::MatchSingleton(singleton) => {
                // Singleton pattern: subject is True/False/None
                use prism_parser::ast::Singleton;
                let cmp_reg = self.builder.alloc_register();
                match singleton {
                    Singleton::True => self.builder.emit_load_true(cmp_reg),
                    Singleton::False => self.builder.emit_load_false(cmp_reg),
                    Singleton::None => self.builder.emit_load_none(cmp_reg),
                }
                let result_reg = self.builder.alloc_register();
                self.builder.emit(Instruction::op_dss(
                    Opcode::Is,
                    result_reg,
                    subject_reg,
                    cmp_reg,
                ));
                self.builder.emit_jump_if_false(result_reg, fail_label);
                self.builder.free_register(result_reg);
                self.builder.free_register(cmp_reg);
            }

            PatternKind::MatchSequence(patterns) => {
                if patterns
                    .iter()
                    .any(|sub_pattern| matches!(sub_pattern.kind, PatternKind::MatchStar(_)))
                {
                    return Err(self.unsupported_pattern_error(
                        pattern,
                        "sequence star patterns require variable-length slicing and rest binding semantics",
                    ));
                }

                // Sequence pattern: [a, b, c]
                // First check if subject is a sequence type using MatchSequence opcode
                let is_seq_reg = self.builder.alloc_register();
                crate::match_compiler::emit_match_sequence(
                    &mut self.builder,
                    is_seq_reg,
                    subject_reg,
                );
                self.builder.emit_jump_if_false(is_seq_reg, fail_label);
                self.builder.free_register(is_seq_reg);

                // Check length
                let len_reg = self.builder.alloc_register();
                let len_name = self.builder.add_name(Arc::from("__len__"));
                let method_reg = self.builder.alloc_register();
                self.builder
                    .emit_get_attr(method_reg, subject_reg, len_name);
                self.builder.emit_call(len_reg, method_reg, 0);
                self.builder.free_register(method_reg);

                // Check length
                let expected_len = self.builder.add_int(patterns.len() as i64);
                let expected_reg = self.builder.alloc_register();
                self.builder.emit_load_const(expected_reg, expected_len);
                let cmp_reg = self.builder.alloc_register();
                self.builder.emit_eq(cmp_reg, len_reg, expected_reg);
                self.builder.emit_jump_if_false(cmp_reg, fail_label);
                self.builder.free_register(cmp_reg);
                self.builder.free_register(expected_reg);
                self.builder.free_register(len_reg);

                // Match each element
                for (idx, sub_pattern) in patterns.iter().enumerate() {
                    let idx_const = self.builder.add_int(idx as i64);
                    let idx_reg = self.builder.alloc_register();
                    self.builder.emit_load_const(idx_reg, idx_const);
                    let elem_reg = self.builder.alloc_register();
                    self.builder.emit_get_item(elem_reg, subject_reg, idx_reg);
                    self.compile_pattern_match(sub_pattern, elem_reg, fail_label)?;
                    self.builder.free_register(elem_reg);
                    self.builder.free_register(idx_reg);
                }
            }

            PatternKind::MatchMapping { .. } => {
                return Err(self.unsupported_pattern_error(
                    pattern,
                    "mapping patterns require missing-key failure, duplicate-key validation, rest-copy allocation, and full mapping protocol semantics",
                ));
            }

            PatternKind::MatchClass {
                cls,
                patterns,
                kwd_attrs,
                kwd_patterns,
            } => {
                self.compile_class_pattern_match(
                    pattern,
                    cls,
                    patterns,
                    kwd_attrs,
                    kwd_patterns,
                    subject_reg,
                    fail_label,
                )?;
            }

            PatternKind::MatchStar(_name) => {
                return Err(self.unsupported_pattern_error(
                    pattern,
                    "star patterns are only valid inside fully implemented sequence patterns",
                ));
            }

            PatternKind::MatchAs { pattern, name } => {
                // As pattern: pattern as name, or just name (wildcard)
                // First match the inner pattern if any
                if let Some(inner) = pattern {
                    self.compile_pattern_match(inner, subject_reg, fail_label)?;
                }

                // Bind the name if present (None means wildcard _)
                if let Some(bound_name) = name {
                    // Use resolve_variable to properly handle scope
                    match self.resolve_variable(bound_name) {
                        VarLocation::Local(slot) => {
                            self.builder
                                .emit_store_local(LocalSlot::new(slot), subject_reg);
                        }
                        VarLocation::Global => {
                            let name_idx = self.builder.add_name(Arc::from(bound_name.as_ref()));
                            self.builder.emit_store_global(name_idx, subject_reg);
                        }
                        VarLocation::Closure(slot) => {
                            self.builder.emit_store_closure(slot, subject_reg);
                        }
                    }
                }
                // If name is None, it's just a wildcard _ which always matches
            }

            PatternKind::MatchOr(alternatives) => {
                // Or pattern: pattern1 | pattern2 | ...
                // Match succeeds if any alternative matches
                let success_label = self.builder.create_label();

                for (i, alt) in alternatives.iter().enumerate() {
                    let is_last = i + 1 == alternatives.len();

                    if is_last {
                        // Last alternative - fail to outer fail_label
                        self.compile_pattern_match(alt, subject_reg, fail_label)?;
                    } else {
                        // Not last - create temp fail label
                        let temp_fail = self.builder.create_label();
                        self.compile_pattern_match(alt, subject_reg, temp_fail)?;
                        // Match succeeded - jump to success
                        self.builder.emit_jump(success_label);
                        self.builder.bind_label(temp_fail);
                    }
                }

                self.builder.bind_label(success_label);
            }
        }

        Ok(())
    }

    pub(super) fn compile_class_pattern_match(
        &mut self,
        pattern: &prism_parser::ast::Pattern,
        cls: &Expr,
        patterns: &[prism_parser::ast::Pattern],
        kwd_attrs: &[String],
        kwd_patterns: &[prism_parser::ast::Pattern],
        subject_reg: Register,
        fail_label: Label,
    ) -> CompileResult<()> {
        if kwd_attrs.len() != kwd_patterns.len() {
            return Err(self.unsupported_pattern_error(
                pattern,
                "class pattern keyword attribute and pattern counts differ",
            ));
        }

        let mut seen_attrs = std::collections::HashSet::new();
        for attr in kwd_attrs {
            if !seen_attrs.insert(attr.as_str()) {
                return Err(CompileError {
                    message: format!("duplicate attribute name in class pattern: {attr}"),
                    line: self.line_for_span(pattern.span),
                    column: 0,
                });
            }
        }

        if patterns.len() > u8::MAX as usize {
            return Err(CompileError {
                message: "class pattern has too many positional subpatterns".to_string(),
                line: self.line_for_span(pattern.span),
                column: 0,
            });
        }

        let class_reg = self.compile_expr(cls)?;
        let matches_class_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_dss(
            Opcode::MatchClass,
            matches_class_reg,
            subject_reg,
            class_reg,
        ));
        self.builder
            .emit_jump_if_false(matches_class_reg, fail_label);
        self.builder.free_register(matches_class_reg);

        if !patterns.is_empty() {
            let match_args_reg = self.builder.alloc_register();
            self.builder.emit(Instruction::op_ds(
                Opcode::GetMatchArgs,
                match_args_reg,
                class_reg,
            ));

            for (index, sub_pattern) in patterns.iter().enumerate() {
                let index_const = self.builder.add_int(index as i64);
                let index_reg = self.builder.alloc_register();
                self.builder.emit_load_const(index_reg, index_const);

                let attr_name_reg = self.builder.alloc_register();
                self.builder
                    .emit_get_item(attr_name_reg, match_args_reg, index_reg);
                self.builder.free_register(index_reg);

                let attr_value_reg = self.builder.alloc_register();
                self.emit_named_call_from_regs(
                    "getattr",
                    &[subject_reg, attr_name_reg],
                    attr_value_reg,
                )?;
                self.builder.free_register(attr_name_reg);

                self.compile_pattern_match(sub_pattern, attr_value_reg, fail_label)?;
                self.builder.free_register(attr_value_reg);
            }

            self.builder.free_register(match_args_reg);
        }

        for (attr, sub_pattern) in kwd_attrs.iter().zip(kwd_patterns.iter()) {
            let attr_reg = self.builder.alloc_register();
            let attr_name_idx = self.builder.add_name(Arc::<str>::from(attr.as_str()));
            self.builder
                .emit_get_attr(attr_reg, subject_reg, attr_name_idx);
            self.compile_pattern_match(sub_pattern, attr_reg, fail_label)?;
            self.builder.free_register(attr_reg);
        }

        self.builder.free_register(class_reg);
        Ok(())
    }

}
