use super::*;

impl Compiler {
    // =========================================================================
    // Dynamic Call Compilation (with *args/**kwargs unpacking)
    // =========================================================================

    /// Compile a function call that contains *args or **kwargs unpacking.
    ///
    /// This builds a tuple for positional arguments and a dict for keyword arguments,
    /// then uses CallEx to invoke the function with the unpacked args.
    ///
    /// # Algorithm
    /// 1. Compile function expression
    /// 2. Build positional args into a tuple (merging any *iterables)
    /// 3. Build keyword args into a dict (merging any **mappings)
    /// 4. Emit CallEx(dst, func, args_tuple, kwargs_dict)
    pub(super) fn compile_dynamic_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        keywords: &[prism_parser::ast::Keyword],
        dst: Register,
        line: u32,
    ) -> CompileResult<Register> {
        // Step 1: Compile function
        let func_reg = self.compile_expr(func)?;

        // Step 2: Build positional args tuple
        // Separate regular args from starred args (for unpack flags)
        let args_tuple_reg = if args.is_empty() {
            // Empty tuple - use BuildTuple with 0 count
            let tuple_reg = self.builder.alloc_register();
            self.builder.emit_build_tuple(tuple_reg, tuple_reg, 0);
            tuple_reg
        } else {
            if args.len() > 24 {
                return Err(CompileError {
                    message: "call-site *args unpack supports at most 24 positional entries"
                        .to_string(),
                    line,
                    column: 0,
                });
            }
            // Compile each arg and track which are starred
            let base_reg = self.builder.alloc_register_block(args.len() as u8);
            let mut unpack_flags: u32 = 0;

            for (i, arg) in args.iter().enumerate() {
                let arg_reg = Register::new(base_reg.0 + i as u8);

                match &arg.kind {
                    ExprKind::Starred(inner) => {
                        // This is a *iterable - compile the inner expression
                        let temp = self.compile_expr(inner)?;
                        if temp != arg_reg {
                            self.builder.emit_move(arg_reg, temp);
                        }
                        self.builder.free_register(temp);
                        // Mark this position for unpacking
                        unpack_flags |= 1 << i;
                    }
                    _ => {
                        // Regular arg - compile directly
                        let temp = self.compile_expr(arg)?;
                        if temp != arg_reg {
                            self.builder.emit_move(arg_reg, temp);
                        }
                        self.builder.free_register(temp);
                    }
                }
            }

            // Build tuple with unpacking (merges starred iterables)
            let tuple_reg = self.builder.alloc_register();
            self.builder.emit_build_tuple_unpack(
                tuple_reg,
                base_reg,
                args.len() as u8,
                unpack_flags,
            );

            // Free arg register block
            self.builder.free_register_block(base_reg, args.len() as u8);

            tuple_reg
        };

        // Step 3: Build keyword args dict (if any)
        let kwargs_dict_reg = if keywords.is_empty() {
            None
        } else {
            if keywords.len() > 24 {
                return Err(CompileError {
                    message: "call-site **kwargs unpack supports at most 24 keyword entries"
                        .to_string(),
                    line,
                    column: 0,
                });
            }

            // Represent every entry as a mapping in `base_reg+i`, then merge.
            let base_reg = self.builder.alloc_register_block(keywords.len() as u8);
            let mut unpack_flags: u32 = 0;

            for (i, kw) in keywords.iter().enumerate() {
                let entry_reg = Register::new(base_reg.0 + i as u8);

                if kw.arg.is_none() {
                    // **mapping entry
                    let temp = self.compile_expr(&kw.value)?;
                    if temp != entry_reg {
                        self.builder.emit_move(entry_reg, temp);
                    }
                    self.builder.free_register(temp);
                    unpack_flags |= 1 << i; // merge this mapping
                } else {
                    // Static keyword: build singleton dict {"name": value}
                    let key_name = kw.arg.as_ref().unwrap();
                    let key_idx = self.builder.add_string(key_name);
                    let pair_base = self.builder.alloc_register_block(2);
                    let key_reg = pair_base;
                    let val_reg = Register::new(pair_base.0 + 1);
                    self.builder.emit_load_const(key_reg, key_idx);

                    let temp = self.compile_expr(&kw.value)?;
                    if temp != val_reg {
                        self.builder.emit_move(val_reg, temp);
                    }
                    self.builder.free_register(temp);

                    self.builder.emit(Instruction::new(
                        Opcode::BuildDict,
                        entry_reg.0,
                        pair_base.0,
                        1,
                    ));
                    self.builder.free_register_block(pair_base, 2);
                    unpack_flags |= 1 << i; // merge singleton mapping
                }
            }

            // Build dict with potential unpacking
            let dict_reg = self.builder.alloc_register();
            self.builder.emit_build_dict_unpack(
                dict_reg,
                base_reg,
                keywords.len() as u8,
                unpack_flags,
            );

            // Free mapping entry block
            self.builder
                .free_register_block(base_reg, keywords.len() as u8);

            Some(dict_reg)
        };

        // Step 4: Emit CallEx
        self.builder
            .emit_call_ex(dst, func_reg, args_tuple_reg, kwargs_dict_reg);

        // Cleanup
        self.builder.free_register(func_reg);
        self.builder.free_register(args_tuple_reg);
        if let Some(kr) = kwargs_dict_reg {
            self.builder.free_register(kr);
        }

        Ok(dst)
    }

}
