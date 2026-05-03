use super::*;

impl VirtualMachine {
    #[inline]
    fn generator_eval_breaker_tick(&mut self, caller_depth: usize) -> VmResult<()> {
        if self.execution_budget.has_limit() {
            self.execution_budget.consume_step()?;
            crate::threading_runtime::checkpoint();
            return self.poll_generator_eval_breaker(caller_depth);
        }

        let remaining = self.eval_breaker_countdown;
        if remaining > 1 {
            self.eval_breaker_countdown = remaining - 1;
            return Ok(());
        }

        self.eval_breaker_countdown = super::EVAL_BREAKER_INTERVAL_OPCODES;
        self.poll_generator_eval_breaker_slow_path(caller_depth)
    }

    #[inline(never)]
    fn poll_generator_eval_breaker_slow_path(&mut self, caller_depth: usize) -> VmResult<()> {
        crate::threading_runtime::checkpoint_now();
        self.poll_generator_eval_breaker(caller_depth)
    }

    fn poll_generator_eval_breaker(&mut self, caller_depth: usize) -> VmResult<()> {
        if let Some(exception) =
            crate::stdlib::_thread::take_pending_async_exception_for_current_thread()
        {
            let (exception, type_id) = self.normalize_async_exception(exception)?;
            self.set_active_exception_with_type(exception, type_id);
            if !self.propagate_exception_within_generator_frames(type_id, caller_depth) {
                return Err(self.uncaught_exception_error(type_id));
            }
        }

        let Some(signum) =
            crate::stdlib::_thread::take_pending_main_interrupt(self.thread_interrupt_target)
        else {
            return Ok(());
        };

        if let Err(err) = crate::stdlib::_thread::deliver_interrupt_signal(self, signum) {
            if err.is_control_transferred() {
                return Ok(());
            }

            let type_id = self.materialize_active_exception_from_runtime_error(&err);
            if !self.propagate_exception_within_generator_frames(type_id, caller_depth) {
                return Err(err);
            }
        }

        Ok(())
    }

    /// Resume a generator object for exactly one send()/next() step.
    pub(crate) fn resume_generator_for_send(
        &mut self,
        generator: &mut GeneratorObject,
        send_value: Value,
    ) -> VmResult<GeneratorResumeOutcome> {
        self.resume_generator(generator, GeneratorResumeMode::Send(send_value))
    }

    /// Resume a suspended generator by raising an exception at its yield point.
    pub(crate) fn resume_generator_for_throw(
        &mut self,
        generator: &mut GeneratorObject,
        exception: Value,
        type_id: u16,
    ) -> VmResult<GeneratorResumeOutcome> {
        self.resume_generator(generator, GeneratorResumeMode::Throw { exception, type_id })
    }

    /// Drive a native async-generator helper operation (`asend`, `athrow`,
    /// `aclose`) through its coroutine-compatible send protocol.
    pub(crate) fn resume_async_generator_operation(
        &mut self,
        operation: &mut AsyncGeneratorOperationObject,
        send_value: Value,
    ) -> VmResult<AsyncGeneratorOperationOutcome> {
        start_async_generator_operation(operation, send_value)?;

        let result = match operation.kind() {
            AsyncGeneratorOperationKind::ASend => self.resume_async_generator_asend(operation),
            AsyncGeneratorOperationKind::AThrow => self.resume_async_generator_athrow(operation),
            AsyncGeneratorOperationKind::AClose => self.resume_async_generator_close(operation),
        };
        operation.finish();
        result
    }

    /// Drive an exception thrown into a native async-generator helper operation.
    pub(crate) fn throw_async_generator_operation(
        &mut self,
        operation: &mut AsyncGeneratorOperationObject,
        exception: Value,
        type_id: u16,
    ) -> VmResult<AsyncGeneratorOperationOutcome> {
        operation
            .try_start()
            .map_err(async_generator_operation_start_error)?;

        let result = self.with_operation_generator_mut(operation, |vm, generator| {
            match vm.resume_generator_for_throw(generator, exception, type_id) {
                Ok(GeneratorResumeOutcome::Yielded(value)) => {
                    Ok(AsyncGeneratorOperationOutcome::Completed(value))
                }
                Ok(GeneratorResumeOutcome::Returned(_)) => Err(stop_async_iteration_error()),
                Err(err) => Err(err),
            }
        });
        operation.finish();
        result
    }

    fn resume_async_generator_asend(
        &mut self,
        operation: &AsyncGeneratorOperationObject,
    ) -> VmResult<AsyncGeneratorOperationOutcome> {
        self.with_operation_generator_mut(operation, |vm, generator| {
            match vm.resume_generator_for_send(generator, operation.send_value()) {
                Ok(GeneratorResumeOutcome::Yielded(value)) => {
                    Ok(AsyncGeneratorOperationOutcome::Completed(value))
                }
                Ok(GeneratorResumeOutcome::Returned(_)) => Err(stop_async_iteration_error()),
                Err(err) => Err(err),
            }
        })
    }

    fn resume_async_generator_athrow(
        &mut self,
        operation: &AsyncGeneratorOperationObject,
    ) -> VmResult<AsyncGeneratorOperationOutcome> {
        self.with_operation_generator_mut(operation, |vm, generator| {
            match vm.resume_generator_for_throw(
                generator,
                operation.exception(),
                operation.exception_type_id(),
            ) {
                Ok(GeneratorResumeOutcome::Yielded(value)) => {
                    Ok(AsyncGeneratorOperationOutcome::Completed(value))
                }
                Ok(GeneratorResumeOutcome::Returned(_)) => Err(stop_async_iteration_error()),
                Err(err) => Err(err),
            }
        })
    }

    fn resume_async_generator_close(
        &mut self,
        operation: &AsyncGeneratorOperationObject,
    ) -> VmResult<AsyncGeneratorOperationOutcome> {
        self.with_operation_generator_mut(operation, |vm, generator| match generator.state() {
            RuntimeGeneratorState::Created => {
                let _ = generator.try_start();
                generator.exhaust();
                Ok(AsyncGeneratorOperationOutcome::Completed(Value::none()))
            }
            RuntimeGeneratorState::Exhausted => {
                Ok(AsyncGeneratorOperationOutcome::Completed(Value::none()))
            }
            RuntimeGeneratorState::Running => Err(RuntimeError::value_error(
                "async generator already executing",
            )),
            RuntimeGeneratorState::Suspended => match vm.resume_generator_for_throw(
                generator,
                operation.exception(),
                operation.exception_type_id(),
            ) {
                Ok(GeneratorResumeOutcome::Yielded(_)) => {
                    generator.exhaust();
                    Err(async_generator_ignored_generator_exit_error())
                }
                Ok(GeneratorResumeOutcome::Returned(_)) => {
                    Ok(AsyncGeneratorOperationOutcome::Completed(Value::none()))
                }
                Err(err)
                    if runtime_error_is_exception_type(&err, ExceptionTypeId::GeneratorExit)
                        || runtime_error_is_exception_type(
                            &err,
                            ExceptionTypeId::StopAsyncIteration,
                        ) =>
                {
                    Ok(AsyncGeneratorOperationOutcome::Completed(Value::none()))
                }
                Err(err) => Err(err),
            },
        })
    }

    fn with_operation_generator_mut<T>(
        &mut self,
        operation: &AsyncGeneratorOperationObject,
        f: impl FnOnce(&mut Self, &mut GeneratorObject) -> VmResult<T>,
    ) -> VmResult<T> {
        let generator =
            GeneratorObject::from_value_mut(operation.generator()).ok_or_else(|| {
                RuntimeError::internal("async generator operation lost its generator")
            })?;

        if !generator.is_async() {
            return Err(RuntimeError::type_error(
                "async generator operation target is not an async generator",
            ));
        }

        f(self, generator)
    }

    fn resume_generator(
        &mut self,
        generator: &mut GeneratorObject,
        mode: GeneratorResumeMode,
    ) -> VmResult<GeneratorResumeOutcome> {
        if self.frames.is_empty() {
            return Err(RuntimeError::internal(
                "cannot resume generator without an active caller frame",
            ));
        }

        let prev_state = match mode {
            GeneratorResumeMode::Send(send_value) => {
                let prev_state = match generator.try_start() {
                    Some(state) => state,
                    None if generator.is_running() => {
                        return Err(RuntimeError::value_error("generator already executing"));
                    }
                    None => {
                        return Err(RuntimeError::stop_iteration());
                    }
                };

                if prev_state == RuntimeGeneratorState::Created && !send_value.is_none() {
                    return Err(RuntimeError::type_error(
                        "can't send non-None value to a just-started generator",
                    ));
                }

                prev_state
            }
            GeneratorResumeMode::Throw { exception, type_id } => match generator.state() {
                RuntimeGeneratorState::Created => {
                    let _ = generator.try_start();
                    generator.exhaust();
                    return Err(RuntimeError::raised_exception(
                        type_id,
                        exception,
                        Self::raised_exception_message(exception),
                    ));
                }
                RuntimeGeneratorState::Suspended => generator.try_start().ok_or_else(|| {
                    RuntimeError::internal("suspended generator refused to start")
                })?,
                RuntimeGeneratorState::Running => {
                    return Err(RuntimeError::value_error("generator already executing"));
                }
                RuntimeGeneratorState::Exhausted => {
                    return Err(RuntimeError::raised_exception(
                        type_id,
                        exception,
                        Self::raised_exception_message(exception),
                    ));
                }
            },
        };

        if prev_state == RuntimeGeneratorState::Created
            && let Err(err) = self.initialize_async_generator_hooks(generator)
        {
            generator.reset_to_created();
            return Err(err);
        }

        let caller_idx = self.current_frame_idx;
        let caller_depth = self.frames.len();
        let caller_scratch_255 = self.frames[caller_idx].get_reg(255);
        let caller_exception_context = self.capture_exception_context();
        let mut generator_exception_context = if prev_state == RuntimeGeneratorState::Suspended {
            generator
                .take_exception_context()
                .unwrap_or_else(ExceptionContextSnapshot::empty)
        } else {
            generator.clear_exception_context();
            ExceptionContextSnapshot::empty()
        };

        let generator_module = self.module_from_globals_ptr(generator.module_ptr());
        let mut frame = self.acquire_frame(
            Arc::clone(generator.code()),
            Some(caller_idx as u32),
            255,
            generator.closure().cloned(),
            generator_module,
        );
        frame.ip = if prev_state == RuntimeGeneratorState::Suspended {
            generator.ip()
        } else {
            0
        };

        // Restore captured live state (or seeded locals for first start).
        let restored_liveness = generator.liveness();
        generator.restore(&mut frame.registers);
        for reg in restored_liveness.iter() {
            frame.mark_reg_written(reg);
        }
        if prev_state == RuntimeGeneratorState::Created {
            initialize_closure_cellvars_from_locals(&mut frame, restored_liveness.count() as usize);
        }

        if prev_state == RuntimeGeneratorState::Suspended
            && let GeneratorResumeMode::Send(send_value) = mode
        {
            let resume_reg = u8::try_from(generator.resume_index())
                .map_err(|_| RuntimeError::internal("generator resume register out of range"))?;
            frame.set_reg(resume_reg, send_value);
        }

        self.frames.push(frame);
        self.set_current_frame_idx(self.frames.len() - 1);
        let generator_frame_idx = self.current_frame_idx;
        generator_exception_context.remap_all_frame_ids(generator_frame_idx as u32);
        self.restore_exception_context(generator_exception_context);

        let mut outcome: Option<GeneratorResumeOutcome> = None;
        let mut failure: Option<RuntimeError> = None;

        if let GeneratorResumeMode::Throw { exception, type_id } = mode {
            self.set_active_exception_with_type(exception, type_id);
            self.set_exception_state(ExceptionState::Propagating);
            if !self.propagate_exception_within_generator_frames(type_id, caller_depth) {
                failure = Some(RuntimeError::raised_exception(
                    type_id,
                    exception,
                    Self::raised_exception_message(exception),
                ));
            }
        }

        if failure.is_none() {
            'exec: loop {
                if let Err(err) = self.generator_eval_breaker_tick(caller_depth) {
                    failure = Some(err);
                    break 'exec;
                }

                let inst = {
                    let frame = &mut self.frames[self.current_frame_idx];

                    if frame.ip as usize >= frame.code.instructions.len() {
                        if self.current_frame_idx == generator_frame_idx {
                            generator.exhaust();
                            outcome = Some(GeneratorResumeOutcome::Returned(Value::none()));
                            break 'exec;
                        }

                        match self.pop_frame(Value::none()) {
                            Ok(None) => {}
                            Ok(Some(_)) => {
                                failure = Some(RuntimeError::internal(
                                    "generator resume unwound to empty frame stack",
                                ));
                                break 'exec;
                            }
                            Err(e) => {
                                failure = Some(e);
                                break 'exec;
                            }
                        }
                        continue;
                    }

                    frame.fetch()
                };

                let control = get_handler(inst.opcode())(self, inst);
                match control {
                    ControlFlow::Continue => {}
                    ControlFlow::Jump(offset) => {
                        let frame = &mut self.frames[self.current_frame_idx];
                        let new_ip = (frame.ip as i32) + (offset as i32);
                        frame.ip = new_ip.max(0) as u32;
                    }
                    ControlFlow::Call { code, return_reg } => {
                        if let Err(e) = self.push_frame_with_module(
                            code,
                            return_reg,
                            self.current_module_cloned(),
                        ) {
                            failure = Some(e);
                            break 'exec;
                        }
                    }
                    ControlFlow::Return(value) => {
                        if self.current_frame_idx == generator_frame_idx {
                            generator.exhaust();
                            outcome = Some(GeneratorResumeOutcome::Returned(value));
                            break 'exec;
                        }

                        match self.pop_frame(value) {
                            Ok(None) => {}
                            Ok(Some(_)) => {
                                failure = Some(RuntimeError::internal(
                                    "generator return unwound to empty frame stack",
                                ));
                                break 'exec;
                            }
                            Err(e) => {
                                failure = Some(e);
                                break 'exec;
                            }
                        }
                    }
                    ControlFlow::Yield {
                        value,
                        resume_point,
                    } => {
                        if self.current_frame_idx != generator_frame_idx {
                            failure = Some(RuntimeError::internal(
                                "nested frame yielded during generator resume",
                            ));
                            break 'exec;
                        }

                        let frame = &self.frames[self.current_frame_idx];
                        generator.suspend(
                            frame.ip,
                            resume_point,
                            &frame.registers,
                            LivenessMap::ALL,
                        );
                        generator.set_exception_context(self.capture_exception_context());
                        outcome = Some(GeneratorResumeOutcome::Yielded(value));
                        break 'exec;
                    }
                    ControlFlow::Resume { send_value } => {
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.set_reg(0, send_value);
                    }
                    ControlFlow::Error(err) => {
                        if err.is_control_transferred() {
                            continue;
                        }

                        let type_id = self.materialize_active_exception_from_runtime_error(&err);
                        if !self.propagate_exception_within_generator_frames(type_id, caller_depth)
                        {
                            failure = Some(err);
                            break 'exec;
                        }
                    }
                    ControlFlow::Exception { type_id, .. } => {
                        if !self.propagate_exception_within_generator_frames(type_id, caller_depth)
                        {
                            failure = Some(self.uncaught_exception_error(type_id));
                            break 'exec;
                        }
                    }
                    ControlFlow::Reraise => {
                        let type_id = if let Some(tid) = self.active_exception_type_id {
                            tid
                        } else if let Some(exc_info) = self.exc_info_stack.peek() {
                            exc_info.type_id()
                        } else {
                            failure =
                                Some(RuntimeError::type_error("No active exception to re-raise"));
                            break 'exec;
                        };

                        if type_id == 0 {
                            failure = Some(RuntimeError::internal(
                                "Reraise without active exception type",
                            ));
                            break 'exec;
                        }

                        if !self.propagate_exception_within_generator_frames(type_id, caller_depth)
                        {
                            failure = Some(self.uncaught_reraised_exception_error(type_id));
                            break 'exec;
                        }
                    }
                    ControlFlow::EnterHandler { handler_pc, .. } => {
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.ip = handler_pc;
                    }
                    ControlFlow::EnterFinally { finally_pc, .. } => {
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.ip = finally_pc;
                    }
                    ControlFlow::ExitHandler => {
                        self.pop_exception_handler();
                    }
                }
            }
        }

        // Always restore caller-visible frame stack state.
        while self.frames.len() > caller_depth {
            self.pop_top_frame_for_unwind();
        }
        self.set_current_frame_idx(caller_idx);
        self.frames[caller_idx].set_reg(255, caller_scratch_255);
        self.restore_exception_context(caller_exception_context);

        if let Some(err) = failure {
            generator.exhaust();
            generator.clear_exception_context();
            return Err(err);
        }

        match outcome {
            Some(result @ GeneratorResumeOutcome::Yielded(_)) => Ok(result),
            Some(result @ GeneratorResumeOutcome::Returned(_)) => {
                generator.clear_exception_context();
                Ok(result)
            }
            None => Err(RuntimeError::internal(
                "generator resume exited without outcome",
            )),
        }
    }

    fn initialize_async_generator_hooks(
        &mut self,
        generator: &mut GeneratorObject,
    ) -> VmResult<()> {
        if !generator.is_async() || generator.asyncgen_hooks_initialized() {
            return Ok(());
        }

        generator.mark_asyncgen_hooks_initialized();
        let generator_value = Value::object_ptr(generator as *mut GeneratorObject as *const ());
        if let Some(finalizer) = crate::stdlib::sys::asyncgen_finalizer_hook() {
            generator.set_asyncgen_finalizer(finalizer);
            self.register_finalizer_candidate(generator_value);
        }
        if let Some(firstiter) = crate::stdlib::sys::asyncgen_firstiter_hook() {
            crate::ops::calls::invoke_callable_value(self, firstiter, &[generator_value])?;
        }

        Ok(())
    }
}

#[inline]
fn start_async_generator_operation(
    operation: &mut AsyncGeneratorOperationObject,
    send_value: Value,
) -> VmResult<()> {
    operation
        .try_start()
        .map_err(async_generator_operation_start_error)?;

    if !send_value.is_none() {
        operation.finish();
        return Err(RuntimeError::type_error(
            "can't send non-None value to a just-started async generator operation",
        ));
    }

    Ok(())
}

#[inline]
fn async_generator_operation_start_error(error: AsyncGeneratorOperationStartError) -> RuntimeError {
    match error {
        AsyncGeneratorOperationStartError::AlreadyRunning => {
            RuntimeError::value_error("async generator operation already executing")
        }
        AsyncGeneratorOperationStartError::Closed => RuntimeError::exception(
            ExceptionTypeId::RuntimeError.as_u8() as u16,
            "cannot reuse already awaited async generator operation",
        ),
    }
}

#[inline]
fn async_generator_ignored_generator_exit_error() -> RuntimeError {
    RuntimeError::exception(
        ExceptionTypeId::RuntimeError.as_u8() as u16,
        "async generator ignored GeneratorExit",
    )
}

#[inline]
fn stop_async_iteration_error() -> RuntimeError {
    let exception = crate::builtins::create_exception(ExceptionTypeId::StopAsyncIteration, None);
    RuntimeError::raised_exception(
        ExceptionTypeId::StopAsyncIteration.as_u8() as u16,
        exception,
        "",
    )
}

#[inline]
fn runtime_error_is_exception_type(error: &RuntimeError, exception_type: ExceptionTypeId) -> bool {
    let expected = exception_type.as_u8() as u16;
    match error.kind() {
        RuntimeErrorKind::GeneratorExit => exception_type == ExceptionTypeId::GeneratorExit,
        RuntimeErrorKind::StopIteration => exception_type == ExceptionTypeId::StopIteration,
        RuntimeErrorKind::Exception { type_id, .. } => *type_id == expected,
        _ => false,
    }
}
