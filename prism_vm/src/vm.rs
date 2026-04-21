//! Virtual machine implementation.
//!
//! The VirtualMachine is the main execution engine for Prism bytecode.
//! It manages frames, globals, builtins, and the dispatch loop.

use crate::allocator::GcAllocator;
use crate::builtins::BuiltinRegistry;
use crate::dispatch::{ControlFlow, get_handler};
use crate::error::{RuntimeError, VmResult};
use crate::exception::{ExcInfoStack, ExceptionState, HandlerStack};
use crate::frame::{ClosureEnv, Frame, FramePool, MAX_RECURSION_DEPTH};
use crate::gc_integration::ManagedHeap;
use crate::globals::GlobalScope;
use crate::ic_manager::ICManager;
use crate::import::{ImportError, ImportResolver, ModuleObject, resolve_relative_import};
use crate::inline_cache::InlineCacheStore;
use crate::jit_context::{JitConfig, JitContext};
use crate::jit_executor::ExecutionResult;
use crate::ops::calls::{capture_closure_environment, initialize_closure_cellvars_from_locals};
use crate::profiler::{CodeId, Profiler, TierUpDecision};
use crate::speculative::SpeculationCache;
use crate::stdlib::generators::{
    GeneratorObject, GeneratorState as RuntimeGeneratorState, LivenessMap,
};
use prism_code::CodeObject;
use prism_compiler::OptimizationLevel;
use prism_core::intern::intern;
use prism_core::{PrismResult, Value};
use prism_parser::parse as parse_module_source;
use prism_runtime::allocation_context::RuntimeHeapBinding;
use prism_runtime::object::class::ClassDict;
use std::collections::HashMap;
use std::sync::Arc;

fn standard_runtime_builtins_and_import_resolver(
    sys_args: Option<Vec<String>>,
) -> (BuiltinRegistry, ImportResolver) {
    let builtins = BuiltinRegistry::with_standard_builtins();
    let import_resolver = match sys_args {
        Some(args) => ImportResolver::with_sys_args_and_builtins(args, builtins.clone()),
        None => ImportResolver::new_with_builtins(builtins.clone()),
    };

    (builtins, import_resolver)
}

/// Result of driving a generator frame for a single send()/next() step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum GeneratorResumeOutcome {
    /// Generator yielded a value and suspended.
    Yielded(Value),
    /// Generator returned and is exhausted.
    Returned(Value),
}

#[derive(Clone, Copy, Debug)]
enum GeneratorResumeMode {
    Send(Value),
    Throw { exception: Value, type_id: u16 },
}

#[derive(Clone)]
pub(crate) struct NamespaceExecutionResult {
    pub namespace: ClassDict,
    pub closure: Option<Arc<ClosureEnv>>,
}

/// Active `except` handler context that must survive nested handlers.
#[derive(Debug, Clone, Copy)]
struct ActiveExceptHandler {
    frame_id: u32,
    value: Value,
    type_id: u16,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum NestedTargetFrameOutcome {
    Returned(Value),
    ControlTransferred,
}

/// Snapshot of caller-visible exception bookkeeping around a synchronous
/// Rust-to-Python callback boundary.
///
/// Successful nested callback execution must not leak handler-local exception
/// state, active exceptions, or auxiliary exception metadata back into the
/// caller. Restoring this snapshot after a successful direct invocation keeps
/// callback semantics aligned with ordinary Python calls.
#[derive(Clone)]
pub(crate) struct ExceptionContextSnapshot {
    exc_state: ExceptionState,
    handler_stack: HandlerStack,
    active_exception: Option<Value>,
    active_exception_type_id: Option<u16>,
    exc_info_stack: ExcInfoStack,
    active_except_handlers: Vec<ActiveExceptHandler>,
}

/// Deterministic interpreter execution budget.
#[derive(Debug, Clone, Copy, Default)]
struct ExecutionBudget {
    step_limit: Option<u64>,
    steps_executed: u64,
}

impl ExecutionBudget {
    #[inline]
    fn set_step_limit(&mut self, limit: Option<u64>) {
        self.step_limit = limit.filter(|value| *value > 0);
        self.steps_executed = 0;
    }

    #[inline]
    fn reset_counter(&mut self) {
        self.steps_executed = 0;
    }

    #[inline]
    fn step_limit(&self) -> Option<u64> {
        self.step_limit
    }

    #[inline]
    fn steps_executed(&self) -> u64 {
        self.steps_executed
    }

    #[inline]
    fn consume_step(&mut self) -> VmResult<()> {
        if let Some(limit) = self.step_limit {
            if self.steps_executed >= limit {
                return Err(RuntimeError::execution_limit_exceeded(limit));
            }
            self.steps_executed += 1;
        }

        Ok(())
    }
}

/// The Prism virtual machine.
///
/// Executes register-based bytecode with:
/// - Frame stack for function calls
/// - Global scope for module-level names
/// - Builtin registry for Python builtins
/// - Inline caching for attribute access
/// - Profiling for JIT tier-up decisions
/// - Optional JIT compilation and execution
/// - GC-managed heap for object allocation
pub struct VirtualMachine {
    /// Frame stack (limited by MAX_RECURSION_DEPTH).
    pub frames: Vec<Frame>,
    /// Reusable frame storage for ordinary calls and JIT entry trampolines.
    frame_pool: FramePool,
    /// Current frame index (frames.len() - 1).
    current_frame_idx: usize,
    /// Global scope.
    pub globals: GlobalScope,
    /// Builtin functions and values.
    pub builtins: BuiltinRegistry,
    /// Inline cache storage.
    pub inline_caches: InlineCacheStore,
    /// Execution profiler.
    pub profiler: Profiler,
    /// IC Manager for centralized type profiling.
    pub ic_manager: ICManager,
    /// Speculation cache for O(1) fast-path lookup.
    pub speculation_cache: SpeculationCache,
    /// JIT context (None when JIT is disabled).
    jit: Option<JitContext>,
    /// Temporary storage for JIT return value when root frame executes via JIT.
    jit_return_value: Option<Value>,
    /// Captured closure environments keyed by function object pointer.
    ///
    /// Function objects currently do not carry VM-native closure environments,
    /// so MakeClosure registers captured cells here and call dispatch looks them up.
    function_closures: HashMap<*const (), Arc<crate::frame::ClosureEnv>>,

    // =========================================================================
    // GC Integration
    // =========================================================================
    /// Thread-local binding that exposes this heap to runtime helpers.
    ///
    /// This must drop before `heap` so runtime helper teardown cannot observe a
    /// stale heap binding.
    _runtime_heap_binding: RuntimeHeapBinding,
    /// GC-managed heap for object allocation.
    /// Stored behind a box so the underlying `GcHeap` address remains stable even
    /// if the `VirtualMachine` itself moves.
    heap: Box<ManagedHeap>,

    // =========================================================================
    // Exception Handling State
    // =========================================================================
    /// Exception state machine for tracking exception propagation phases.
    exc_state: ExceptionState,
    /// Runtime handler stack for active try/except/finally blocks.
    handler_stack: HandlerStack,
    /// Currently active exception (if any) being propagated.
    active_exception: Option<Value>,
    /// Type ID of the active exception for fast matching.
    active_exception_type_id: Option<u16>,
    /// Exception info stack for CPython 3.11+ semantics.
    exc_info_stack: ExcInfoStack,
    /// Stack of entered except handlers for nested bare-raise semantics.
    active_except_handlers: Vec<ActiveExceptHandler>,
    /// Import resolver for module imports.
    pub import_resolver: ImportResolver,
    /// CPython-style import tracing verbosity (`-v`, `-vv`, ...).
    import_verbosity: u32,
    /// Optimization level used when compiling imported source modules.
    compiler_optimization: OptimizationLevel,
    /// Deterministic interpreter execution budget.
    execution_budget: ExecutionBudget,
    /// Most recent runtime error reported by a native AOT helper call.
    last_aot_error: Option<RuntimeError>,
}

impl VirtualMachine {
    fn current_module_ref(&self) -> Option<&Arc<ModuleObject>> {
        self.frames
            .get(self.current_frame_idx)
            .and_then(|frame| frame.module.as_ref())
    }

    pub fn current_module(&self) -> Option<&ModuleObject> {
        self.current_module_ref().map(Arc::as_ref)
    }

    pub fn current_module_cloned(&self) -> Option<Arc<ModuleObject>> {
        self.current_module_ref().cloned()
    }

    pub fn module_from_globals_ptr(&self, ptr: *const ()) -> Option<Arc<ModuleObject>> {
        if ptr.is_null() {
            None
        } else {
            self.import_resolver.module_from_ptr(ptr)
        }
    }

    pub(crate) fn module_scope_value_for_module(
        &self,
        module: &ModuleObject,
        name: &str,
    ) -> Option<Value> {
        if let Some(value) = module.get_attr(name) {
            return Some(value);
        }

        if module.name() == "__main__" {
            return self.globals.get(name);
        }

        None
    }

    pub fn module_scope_value(&self, name: &Arc<str>) -> Option<Value> {
        if let Some(module) = self.current_module() {
            self.module_scope_value_for_module(module, name)
        } else {
            self.globals.get_arc(name)
        }
    }

    pub(crate) fn set_module_scope_value_for_module(
        &mut self,
        module: &Arc<ModuleObject>,
        name: &str,
        value: Value,
    ) {
        module.set_attr(name, value);
        if module.name() == "__main__" {
            self.globals.set(Arc::from(name), value);
        }
    }

    pub fn set_module_scope_value(&mut self, name: Arc<str>, value: Value) {
        if let Some(module) = self.current_module_cloned() {
            self.set_module_scope_value_for_module(&module, &name, value);
        } else {
            self.globals.set(name, value);
        }
    }

    pub fn delete_module_scope_value(&mut self, name: &str) -> Option<Value> {
        if let Some(module) = self.current_module_cloned() {
            let removed = module.get_attr(name);
            let existed = module.del_attr(name);
            if module.name() == "__main__" {
                let _ = self.globals.delete(name);
            }
            if existed { removed } else { None }
        } else {
            self.globals.delete(name)
        }
    }

    pub fn import_star_into_current_scope(&mut self, module: &ModuleObject) {
        if let Some(target_module) = self.current_module_cloned() {
            let is_main = target_module.name() == "__main__";
            for (name, value) in module.public_attrs() {
                target_module.set_attr(&name, value);
                if is_main {
                    self.globals.set(Arc::from(name.as_ref()), value);
                }
            }
        } else {
            for (name, value) in module.public_attrs() {
                self.globals.set(Arc::from(name.as_ref()), value);
            }
        }
    }

    pub fn bind_module(&mut self, module: Arc<ModuleObject>) {
        let name = Arc::<str>::from(module.name());
        self.import_resolver
            .insert_module(&name, Arc::clone(&module));
        if module.name() == "__main__" {
            self.globals = GlobalScope::new();
            for (attr_name, value) in module.all_attrs() {
                self.globals.set(Arc::from(attr_name.as_ref()), value);
            }
        }
    }

    pub fn execute_in_module(
        &mut self,
        code: Arc<CodeObject>,
        module: Arc<ModuleObject>,
    ) -> PrismResult<Value> {
        if self.frames.is_empty() {
            self.execution_budget.reset_counter();
        }
        self.bind_module(Arc::clone(&module));

        self.push_frame_with_module(code, 0, Some(module))?;

        if self.frames.is_empty() {
            return Ok(self.jit_return_value.take().unwrap_or_else(Value::none));
        }

        self.run_loop()
    }

    #[inline]
    pub fn set_compiler_optimization(&mut self, level: OptimizationLevel) {
        self.compiler_optimization = level;
    }

    pub(crate) fn record_aot_error(&mut self, err: RuntimeError) {
        self.last_aot_error = Some(err);
    }

    pub(crate) fn clear_last_aot_error(&mut self) {
        self.last_aot_error = None;
    }

    pub fn take_last_aot_error(&mut self) -> Option<RuntimeError> {
        self.last_aot_error.take()
    }

    #[inline]
    pub fn set_execution_step_limit(&mut self, limit: Option<u64>) {
        self.execution_budget.set_step_limit(limit);
    }

    #[inline]
    pub fn set_import_verbosity(&mut self, verbosity: u32) {
        self.import_verbosity = verbosity;
    }

    #[inline]
    pub fn import_verbosity(&self) -> u32 {
        self.import_verbosity
    }

    #[inline]
    pub fn execution_step_limit(&self) -> Option<u64> {
        self.execution_budget.step_limit()
    }

    #[inline]
    pub fn executed_steps(&self) -> u64 {
        self.execution_budget.steps_executed()
    }

    pub(crate) fn import_error_to_runtime(err: ImportError) -> RuntimeError {
        let rendered = Arc::from(err.to_string());
        match err {
            ImportError::ModuleNotFound { module } => RuntimeError::module_not_found(module),
            ImportError::CircularImport { module }
            | ImportError::LoadError { module, .. }
            | ImportError::ExecutionError { module, .. }
            | ImportError::ImportFromError { module, .. } => {
                RuntimeError::import_error(module, rendered)
            }
        }
    }

    fn resolve_import_name_with_context(
        &self,
        raw_name: &str,
        current_module: Option<&ModuleObject>,
    ) -> VmResult<String> {
        let level = raw_name.chars().take_while(|ch| *ch == '.').count() as u32;
        if level == 0 {
            return Ok(raw_name.to_string());
        }

        let current_module = current_module.ok_or_else(|| {
            RuntimeError::import_error(raw_name, "relative import outside of module execution")
        })?;
        let package = current_module.package_name().unwrap_or("");
        resolve_relative_import(&raw_name[level as usize..], level, package)
            .map_err(Self::import_error_to_runtime)
    }

    fn resolve_import_name(&self, raw_name: &str) -> VmResult<String> {
        self.resolve_import_name_with_context(raw_name, self.current_module())
    }

    fn compile_source_module(
        &self,
        module_name: &str,
        source: &str,
        filename: &str,
    ) -> VmResult<Arc<CodeObject>> {
        let parsed = parse_module_source(source)
            .map_err(|err| RuntimeError::import_error(module_name, Arc::from(err.to_string())))?;

        prism_compiler::Compiler::compile_module_with_optimization(
            &parsed,
            filename,
            self.compiler_optimization,
        )
        .map(Arc::new)
        .map_err(|err| RuntimeError::import_error(module_name, Arc::from(err.to_string())))
    }

    fn run_nested_module_until_depth(&mut self, stop_depth: usize) -> VmResult<()> {
        loop {
            if self.frames.len() <= stop_depth {
                return Ok(());
            }

            self.execution_budget.consume_step()?;

            let inst = {
                let frame = &mut self.frames[self.current_frame_idx];

                if frame.ip as usize >= frame.code.instructions.len() {
                    if self.frames.len() == stop_depth + 1 {
                        let emptied_stack = self.pop_frame_discarding_return()?;
                        if emptied_stack || self.frames.len() <= stop_depth {
                            return Ok(());
                        }
                        continue;
                    }

                    match self.pop_frame(Value::none())? {
                        Some(_) => return Ok(()),
                        None => {
                            if self.frames.len() <= stop_depth {
                                return Ok(());
                            }
                            continue;
                        }
                    }
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
                    self.push_frame(code, return_reg)?;
                }
                ControlFlow::Return(value) => {
                    if self.frames.len() == stop_depth + 1 {
                        let emptied_stack = self.pop_frame_discarding_return()?;
                        if emptied_stack || self.frames.len() <= stop_depth {
                            return Ok(());
                        }
                    } else {
                        match self.pop_frame(value)? {
                            Some(_) => return Ok(()),
                            None => {
                                if self.frames.len() <= stop_depth {
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
                ControlFlow::Exception { type_id, .. } => {
                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        self.frames[self.current_frame_idx].ip = handler_entry;
                    } else {
                        loop {
                            if self.frames.len() <= stop_depth {
                                return Err(self.uncaught_exception_error(type_id));
                            }

                            self.pop_top_frame_for_unwind();
                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                self.frames[self.current_frame_idx].ip = handler_entry;
                                break;
                            }
                        }
                    }
                }
                ControlFlow::Reraise => {
                    let type_id = if let Some(tid) = self.active_exception_type_id {
                        tid
                    } else if let Some(exc_info) = self.exc_info_stack.peek() {
                        exc_info.type_id()
                    } else {
                        return Err(RuntimeError::type_error("No active exception to re-raise"));
                    };

                    if type_id == 0 {
                        return Err(RuntimeError::internal(
                            "Reraise without active exception type",
                        ));
                    }

                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        self.frames[self.current_frame_idx].ip = handler_entry;
                    } else {
                        loop {
                            if self.frames.len() <= stop_depth {
                                return Err(self.uncaught_reraised_exception_error(type_id));
                            }

                            self.pop_top_frame_for_unwind();
                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                self.frames[self.current_frame_idx].ip = handler_entry;
                                break;
                            }
                        }
                    }
                }
                ControlFlow::EnterHandler { handler_pc, .. } => {
                    self.frames[self.current_frame_idx].ip = handler_pc;
                }
                ControlFlow::EnterFinally { finally_pc, .. } => {
                    self.frames[self.current_frame_idx].ip = finally_pc;
                }
                ControlFlow::ExitHandler => {
                    self.pop_exception_handler();
                }
                ControlFlow::Yield { .. } => {
                    return Err(RuntimeError::internal(
                        "yield is not valid while executing module top-level code",
                    ));
                }
                ControlFlow::Resume { send_value } => {
                    self.frames[self.current_frame_idx].set_reg(0, send_value);
                }
                ControlFlow::Error(err) => {
                    let type_id = self.materialize_active_exception_from_runtime_error(&err);
                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        self.frames[self.current_frame_idx].ip = handler_entry;
                    } else {
                        loop {
                            if self.frames.len() <= stop_depth {
                                return Err(err);
                            }

                            self.pop_top_frame_for_unwind();
                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                self.frames[self.current_frame_idx].ip = handler_entry;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    fn collect_frame_locals_namespace(frame: &Frame) -> ClassDict {
        if frame.locals_mapping().is_some() {
            return ClassDict::new();
        }

        let namespace = ClassDict::new();
        for (slot, name) in frame.code.locals.iter().enumerate() {
            let slot = slot as u8;
            if frame.reg_is_written(slot) {
                namespace.set(intern(name.as_ref()), frame.get_reg(slot));
            }
        }
        namespace
    }

    #[inline]
    fn route_nested_exception(&mut self, stop_depth: usize, type_id: u16) -> VmResult<()> {
        if let Some(handler_entry) = self.find_exception_handler(type_id) {
            self.frames[self.current_frame_idx].ip = handler_entry;
            return Ok(());
        }

        loop {
            if self.frames.len() <= stop_depth {
                return Err(self.uncaught_exception_error(type_id));
            }

            self.pop_top_frame_for_unwind();
            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                self.frames[self.current_frame_idx].ip = handler_entry;
                return Ok(());
            }
        }
    }

    #[inline]
    fn route_nested_reraise(&mut self, stop_depth: usize, type_id: u16) -> VmResult<()> {
        if let Some(handler_entry) = self.find_exception_handler(type_id) {
            self.frames[self.current_frame_idx].ip = handler_entry;
            return Ok(());
        }

        loop {
            if self.frames.len() <= stop_depth {
                return Err(self.uncaught_reraised_exception_error(type_id));
            }

            self.pop_top_frame_for_unwind();
            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                self.frames[self.current_frame_idx].ip = handler_entry;
                return Ok(());
            }
        }
    }

    #[inline]
    fn route_nested_runtime_error(&mut self, stop_depth: usize, err: RuntimeError) -> VmResult<()> {
        let type_id = self.materialize_active_exception_from_runtime_error(&err);
        if let Some(handler_entry) = self.find_exception_handler(type_id) {
            self.frames[self.current_frame_idx].ip = handler_entry;
            return Ok(());
        }

        loop {
            if self.frames.len() <= stop_depth {
                return Err(err);
            }

            self.pop_top_frame_for_unwind();
            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                self.frames[self.current_frame_idx].ip = handler_entry;
                return Ok(());
            }
        }
    }

    #[inline]
    fn apply_nested_handler_control_flow(&mut self, control: ControlFlow) -> VmResult<()> {
        match control {
            ControlFlow::EnterHandler { handler_pc, .. } => {
                self.frames[self.current_frame_idx].ip = handler_pc;
                Ok(())
            }
            ControlFlow::EnterFinally { finally_pc, .. } => {
                self.frames[self.current_frame_idx].ip = finally_pc;
                Ok(())
            }
            ControlFlow::ExitHandler => {
                self.pop_exception_handler();
                Ok(())
            }
            _ => Err(RuntimeError::internal(
                "apply_nested_handler_control_flow called with non-handler control flow",
            )),
        }
    }

    pub(crate) fn execute_until_stack_depth_restored(
        &mut self,
        stop_depth: usize,
        return_reg: u8,
    ) -> VmResult<Value> {
        if stop_depth == 0 {
            return Err(RuntimeError::internal(
                "nested execution requires a caller frame",
            ));
        }

        loop {
            if self.frames.len() == stop_depth {
                return Ok(self.frames[self.current_frame_idx].get_reg(return_reg));
            }

            if self.frames.len() < stop_depth {
                return Err(RuntimeError::internal(
                    "nested execution unwound below caller frame",
                ));
            }

            let inst = {
                let current_frame_idx = self.current_frame_idx;
                let frame = &mut self.frames[current_frame_idx];

                if frame.ip as usize >= frame.code.instructions.len() {
                    match self.pop_frame(Value::none())? {
                        Some(_) => {
                            return Err(RuntimeError::internal(
                                "nested execution unwound to empty frame stack",
                            ));
                        }
                        None => {
                            if self.frames.len() == stop_depth {
                                return Ok(self.frames[self.current_frame_idx].get_reg(return_reg));
                            }
                            continue;
                        }
                    }
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
                    self.push_frame(code, return_reg)?;
                }
                ControlFlow::Return(value) => match self.pop_frame(value)? {
                    Some(_) => {
                        return Err(RuntimeError::internal(
                            "nested execution unwound to empty frame stack",
                        ));
                    }
                    None => {
                        if self.frames.len() == stop_depth {
                            return Ok(self.frames[self.current_frame_idx].get_reg(return_reg));
                        }
                    }
                },
                ControlFlow::Exception { type_id, .. } => {
                    self.route_nested_exception(stop_depth, type_id)?;
                }
                ControlFlow::Reraise => {
                    let type_id = if let Some(tid) = self.get_active_exception_type_id() {
                        tid
                    } else if let Some(exc_info) = self.exc_info_stack().peek() {
                        exc_info.type_id()
                    } else {
                        return Err(RuntimeError::type_error("No active exception to re-raise"));
                    };

                    if type_id == 0 {
                        return Err(RuntimeError::internal(
                            "Reraise without active exception type",
                        ));
                    }

                    self.route_nested_reraise(stop_depth, type_id)?;
                }
                ControlFlow::Error(err) => {
                    self.route_nested_runtime_error(stop_depth, err)?;
                }
                ControlFlow::Yield { .. } | ControlFlow::Resume { .. } => {
                    return Err(RuntimeError::type_error(
                        "nested execution cannot suspend or yield",
                    ));
                }
                ControlFlow::EnterHandler { .. }
                | ControlFlow::EnterFinally { .. }
                | ControlFlow::ExitHandler => {
                    self.apply_nested_handler_control_flow(control)?;
                }
            }
        }
    }

    pub(crate) fn execute_until_target_frame_returns(
        &mut self,
        stop_depth: usize,
        target_frame_id: u32,
    ) -> VmResult<Value> {
        match self.execute_until_target_frame_returns_with_outcome(stop_depth, target_frame_id)? {
            NestedTargetFrameOutcome::Returned(value) => Ok(value),
            NestedTargetFrameOutcome::ControlTransferred => Err(RuntimeError::internal(
                "nested execution target frame returned without a result",
            )),
        }
    }

    pub(crate) fn execute_until_target_frame_returns_with_outcome(
        &mut self,
        stop_depth: usize,
        target_frame_id: u32,
    ) -> VmResult<NestedTargetFrameOutcome> {
        if stop_depth == 0 {
            return Err(RuntimeError::internal(
                "nested execution requires a caller frame",
            ));
        }

        loop {
            if self.frames.len() < stop_depth {
                return Err(RuntimeError::internal(
                    "nested execution unwound below caller frame",
                ));
            }

            if self.frames.len() == stop_depth {
                // The target frame unwound into the caller, typically because the caller
                // handled an exception raised by the nested call.
                return Ok(NestedTargetFrameOutcome::ControlTransferred);
            }

            let inst = {
                let current_frame_idx = self.current_frame_idx;
                let frame = &mut self.frames[current_frame_idx];

                if frame.ip as usize >= frame.code.instructions.len() {
                    if current_frame_idx as u32 == target_frame_id {
                        match self.pop_frame(Value::none())? {
                            Some(_) => {
                                return Err(RuntimeError::internal(
                                    "nested execution unwound to empty frame stack",
                                ));
                            }
                            None => return Ok(NestedTargetFrameOutcome::Returned(Value::none())),
                        }
                    }

                    match self.pop_frame(Value::none())? {
                        Some(_) => {
                            return Err(RuntimeError::internal(
                                "nested execution unwound to empty frame stack",
                            ));
                        }
                        None => continue,
                    }
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
                    self.push_frame(code, return_reg)?;
                }
                ControlFlow::Return(value) => {
                    if self.current_frame_id() == target_frame_id {
                        match self.pop_frame(value)? {
                            Some(_) => {
                                return Err(RuntimeError::internal(
                                    "nested execution unwound to empty frame stack",
                                ));
                            }
                            None => return Ok(NestedTargetFrameOutcome::Returned(value)),
                        }
                    }

                    match self.pop_frame(value)? {
                        Some(_) => {
                            return Err(RuntimeError::internal(
                                "nested execution unwound to empty frame stack",
                            ));
                        }
                        None => {}
                    }
                }
                ControlFlow::Exception { type_id, .. } => {
                    self.route_nested_exception(stop_depth, type_id)?;
                }
                ControlFlow::Reraise => {
                    let type_id = if let Some(tid) = self.get_active_exception_type_id() {
                        tid
                    } else if let Some(exc_info) = self.exc_info_stack().peek() {
                        exc_info.type_id()
                    } else {
                        return Err(RuntimeError::type_error("No active exception to re-raise"));
                    };

                    if type_id == 0 {
                        return Err(RuntimeError::internal(
                            "Reraise without active exception type",
                        ));
                    }

                    self.route_nested_reraise(stop_depth, type_id)?;
                }
                ControlFlow::Error(err) => {
                    self.route_nested_runtime_error(stop_depth, err)?;
                }
                ControlFlow::Yield { .. } | ControlFlow::Resume { .. } => {
                    return Err(RuntimeError::type_error(
                        "nested execution cannot suspend or yield",
                    ));
                }
                ControlFlow::EnterHandler { .. }
                | ControlFlow::EnterFinally { .. }
                | ControlFlow::ExitHandler => {
                    self.apply_nested_handler_control_flow(control)?;
                }
            }
        }
    }

    pub(crate) fn execute_code_collect_locals_namespace(
        &mut self,
        code: Arc<CodeObject>,
    ) -> VmResult<NamespaceExecutionResult> {
        self.execute_code_collect_locals_namespace_with_context(
            code,
            self.current_module_cloned(),
            None,
        )
    }

    pub(crate) fn execute_code_collect_locals_namespace_with_mapping(
        &mut self,
        code: Arc<CodeObject>,
        locals_mapping: Option<Value>,
    ) -> VmResult<NamespaceExecutionResult> {
        self.execute_code_collect_locals_namespace_with_context(
            code,
            self.current_module_cloned(),
            locals_mapping,
        )
    }

    pub(crate) fn execute_code_collect_locals_namespace_in_module(
        &mut self,
        code: Arc<CodeObject>,
        module: Arc<ModuleObject>,
        locals_mapping: Option<Value>,
    ) -> VmResult<NamespaceExecutionResult> {
        self.bind_module(Arc::clone(&module));
        self.execute_code_collect_locals_namespace_with_context(code, Some(module), locals_mapping)
    }

    fn execute_code_collect_locals_namespace_with_context(
        &mut self,
        code: Arc<CodeObject>,
        module: Option<Arc<ModuleObject>>,
        locals_mapping: Option<Value>,
    ) -> VmResult<NamespaceExecutionResult> {
        if self.frames.is_empty() {
            self.execution_budget.reset_counter();
        }

        let stop_depth = self.frames.len();
        let closure = if code.cellvars.is_empty() && code.freevars.is_empty() {
            None
        } else {
            Some(capture_closure_environment(self.current_frame(), &code)?)
        };
        self.push_frame_with_closure_and_module(code, 0, closure.clone(), module)?;
        if locals_mapping.is_some() {
            self.current_frame_mut().set_locals_mapping(locals_mapping);
        }
        let target_frame_id = self.current_frame_id();

        loop {
            let inst = {
                let current_frame_idx = self.current_frame_idx;
                let frame = &mut self.frames[current_frame_idx];

                if frame.ip as usize >= frame.code.instructions.len() {
                    if current_frame_idx as u32 == target_frame_id {
                        let namespace = Self::collect_frame_locals_namespace(frame);
                        let _ = self.pop_frame_discarding_return()?;
                        return Ok(NamespaceExecutionResult { namespace, closure });
                    }

                    match self.pop_frame(Value::none())? {
                        Some(_) => {
                            return Err(RuntimeError::internal(
                                "namespace execution unwound to empty frame stack",
                            ));
                        }
                        None => continue,
                    }
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
                    self.push_frame(code, return_reg)?;
                }
                ControlFlow::Return(value) => {
                    if self.current_frame_id() == target_frame_id {
                        let namespace = Self::collect_frame_locals_namespace(self.current_frame());
                        let _ = self.pop_frame_discarding_return()?;
                        return Ok(NamespaceExecutionResult { namespace, closure });
                    }

                    match self.pop_frame(value)? {
                        Some(_) => {
                            return Err(RuntimeError::internal(
                                "namespace execution unwound to empty frame stack",
                            ));
                        }
                        None => {}
                    }
                }
                ControlFlow::Exception { type_id, .. } => {
                    self.route_nested_exception(stop_depth, type_id)?;
                }
                ControlFlow::Reraise => {
                    let type_id = if let Some(tid) = self.get_active_exception_type_id() {
                        tid
                    } else if let Some(exc_info) = self.exc_info_stack().peek() {
                        exc_info.type_id()
                    } else {
                        return Err(RuntimeError::type_error("No active exception to re-raise"));
                    };

                    if type_id == 0 {
                        return Err(RuntimeError::internal(
                            "Reraise without active exception type",
                        ));
                    }

                    self.route_nested_reraise(stop_depth, type_id)?;
                }
                ControlFlow::Error(err) => {
                    self.route_nested_runtime_error(stop_depth, err)?;
                }
                ControlFlow::Yield { .. } | ControlFlow::Resume { .. } => {
                    return Err(RuntimeError::type_error(
                        "class body execution cannot suspend or yield",
                    ));
                }
                ControlFlow::EnterHandler { .. }
                | ControlFlow::EnterFinally { .. }
                | ControlFlow::ExitHandler => {
                    self.apply_nested_handler_control_flow(control)?;
                }
            }
        }
    }

    fn load_source_module(&mut self, name: &str) -> VmResult<Arc<ModuleObject>> {
        if let Some(module) = self.import_resolver.get_cached(name) {
            return Ok(module);
        }

        let Some(location) = self.import_resolver.resolve_source_location(name) else {
            return Err(Self::import_error_to_runtime(ImportError::ModuleNotFound {
                module: Arc::from(name),
            }));
        };

        if self.import_verbosity > 0 {
            eprintln!("import {} # from {}", name, location.path.display());
        }

        let source = std::fs::read_to_string(&location.path).map_err(|err| {
            RuntimeError::import_error(
                name,
                Arc::<str>::from(format!(
                    "failed to read '{}': {}",
                    location.path.display(),
                    err
                )),
            )
        })?;
        let filename: Arc<str> = Arc::from(location.path.to_string_lossy().into_owned());
        let code = self.compile_source_module(name, &source, filename.as_ref());

        let code = match code {
            Ok(code) => code,
            Err(err) => return Err(err),
        };

        self.execute_loaded_module(
            name,
            code,
            filename,
            module_package_name(name, location.is_package),
        )
    }

    fn load_frozen_module(&mut self, name: &str) -> VmResult<Arc<ModuleObject>> {
        if let Some(module) = self.import_resolver.get_cached(name) {
            return Ok(module);
        }

        let Some(frozen) = self.import_resolver.get_frozen_module(name) else {
            return Err(Self::import_error_to_runtime(ImportError::ModuleNotFound {
                module: Arc::from(name),
            }));
        };

        if self.import_verbosity > 0 {
            eprintln!("import {} # frozen", name);
        }

        self.execute_loaded_module(
            name,
            Arc::clone(&frozen.code),
            Arc::clone(&frozen.filename),
            Arc::clone(&frozen.package_name),
        )
    }

    fn execute_loaded_module(
        &mut self,
        name: &str,
        code: Arc<CodeObject>,
        filename: Arc<str>,
        package_name: Arc<str>,
    ) -> VmResult<Arc<ModuleObject>> {
        let module = Arc::new(ModuleObject::with_metadata(
            Arc::from(name),
            None,
            Some(Arc::clone(&filename)),
            Some(package_name),
        ));

        self.import_resolver
            .insert_module(name, Arc::clone(&module));
        let caller_depth = self.frames.len();

        if let Err(err) = self.push_frame_internal(code, 0, None, Some(Arc::clone(&module)), false)
        {
            self.import_resolver.remove_module(name);
            return Err(err);
        }

        if self.frames.len() > caller_depth
            && let Err(err) = self.run_nested_module_until_depth(caller_depth)
        {
            self.import_resolver.remove_module(name);
            return Err(err);
        }

        Ok(module)
    }

    fn load_non_stdlib_module(&mut self, name: &str) -> VmResult<Arc<ModuleObject>> {
        if self.import_resolver.has_frozen_module(name) {
            self.load_frozen_module(name)
        } else {
            self.load_source_module(name)
        }
    }

    pub(crate) fn import_module_with_context(
        &mut self,
        raw_name: &str,
        current_module: Option<&Arc<ModuleObject>>,
    ) -> VmResult<Arc<ModuleObject>> {
        let absolute_name =
            self.resolve_import_name_with_context(raw_name, current_module.map(Arc::as_ref))?;

        if let Some(module) = self.import_resolver.get_cached(&absolute_name) {
            return Ok(module);
        }

        if !absolute_name.contains('.') {
            if self
                .import_resolver
                .should_load_from_source_first(&absolute_name)
            {
                return self.load_non_stdlib_module(&absolute_name);
            }

            return match self.import_resolver.import_module(&absolute_name) {
                Ok(module) => Ok(module),
                Err(ImportError::ModuleNotFound { .. }) => {
                    self.load_non_stdlib_module(&absolute_name)
                }
                Err(err) => Err(Self::import_error_to_runtime(err)),
            };
        }

        let mut segments = absolute_name.split('.');
        let top_level = segments
            .next()
            .expect("dotted import missing top-level segment");
        let mut current = self.import_module_with_context(top_level, current_module)?;
        let mut prefix = top_level.to_string();

        for segment in segments {
            prefix.push('.');
            prefix.push_str(segment);

            if let Some(module) = self.import_resolver.get_cached(&prefix) {
                current.set_attr(
                    segment,
                    Value::object_ptr(Arc::as_ptr(&module) as *const ()),
                );
                current = module;
                continue;
            }

            if self.import_resolver.should_load_from_source_first(&prefix) {
                let next = self.load_non_stdlib_module(&prefix)?;
                current.set_attr(segment, Value::object_ptr(Arc::as_ptr(&next) as *const ()));
                current = next;
                continue;
            }

            let next = match self.import_resolver.import_module(&prefix) {
                Ok(module) => module,
                Err(ImportError::ModuleNotFound { .. }) => {
                    if let Some(value) = current.get_attr(segment)
                        && let Some(module_ptr) = value.as_object_ptr()
                        && let Some(module) = self.import_resolver.module_from_ptr(module_ptr)
                    {
                        self.import_resolver
                            .insert_module(&prefix, Arc::clone(&module));
                        module
                    } else {
                        self.load_non_stdlib_module(&prefix)?
                    }
                }
                Err(err) => return Err(Self::import_error_to_runtime(err)),
            };

            current.set_attr(segment, Value::object_ptr(Arc::as_ptr(&next) as *const ()));
            current = next;
        }

        Ok(current)
    }

    pub fn import_module_named(&mut self, raw_name: &str) -> VmResult<Arc<ModuleObject>> {
        let current_module = self.current_module_cloned();
        self.import_module_with_context(raw_name, current_module.as_ref())
    }

    pub(crate) fn import_from_with_context(
        &mut self,
        module_spec: &str,
        name: &str,
        current_module: Option<&Arc<ModuleObject>>,
    ) -> VmResult<Value> {
        let module = self.import_module_with_context(module_spec, current_module)?;
        self.import_resolver
            .import_from(&module, name)
            .map_err(Self::import_error_to_runtime)
    }

    /// Create a new virtual machine (interpreter only, no JIT).
    pub fn new() -> Self {
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
        let heap = Box::new(ManagedHeap::with_defaults());
        let runtime_heap_binding = RuntimeHeapBinding::register(heap.heap());
        Self {
            frames: Vec::with_capacity(64),
            frame_pool: FramePool::new(),
            current_frame_idx: 0,
            globals: GlobalScope::new(),
            builtins,
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
            ic_manager: ICManager::new(),
            speculation_cache: SpeculationCache::new(),
            jit: None,
            jit_return_value: None,
            function_closures: HashMap::new(),
            heap,
            _runtime_heap_binding: runtime_heap_binding,
            exc_state: ExceptionState::default(),
            handler_stack: HandlerStack::new(),
            active_exception: None,
            active_exception_type_id: None,
            exc_info_stack: ExcInfoStack::new(),
            active_except_handlers: Vec::new(),
            import_resolver,
            import_verbosity: 0,
            compiler_optimization: OptimizationLevel::None,
            execution_budget: ExecutionBudget::default(),
            last_aot_error: None,
        }
    }

    /// Create a new virtual machine with JIT compilation enabled.
    pub fn with_jit() -> Self {
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
        let heap = Box::new(ManagedHeap::with_defaults());
        let runtime_heap_binding = RuntimeHeapBinding::register(heap.heap());
        Self {
            frames: Vec::with_capacity(64),
            frame_pool: FramePool::new(),
            current_frame_idx: 0,
            globals: GlobalScope::new(),
            builtins,
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
            ic_manager: ICManager::new(),
            speculation_cache: SpeculationCache::new(),
            jit: Some(JitContext::with_defaults()),
            jit_return_value: None,
            function_closures: HashMap::new(),
            heap,
            _runtime_heap_binding: runtime_heap_binding,
            exc_state: ExceptionState::default(),
            handler_stack: HandlerStack::new(),
            active_exception: None,
            active_exception_type_id: None,
            exc_info_stack: ExcInfoStack::new(),
            active_except_handlers: Vec::new(),
            import_resolver,
            import_verbosity: 0,
            compiler_optimization: OptimizationLevel::None,
            execution_budget: ExecutionBudget::default(),
            last_aot_error: None,
        }
    }

    /// Create a virtual machine with custom JIT configuration.
    pub fn with_jit_config(config: JitConfig) -> Self {
        let jit = if config.enabled {
            Some(JitContext::new(config))
        } else {
            None
        };
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
        let heap = Box::new(ManagedHeap::with_defaults());
        let runtime_heap_binding = RuntimeHeapBinding::register(heap.heap());
        Self {
            frames: Vec::with_capacity(64),
            frame_pool: FramePool::new(),
            current_frame_idx: 0,
            globals: GlobalScope::new(),
            builtins,
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
            ic_manager: ICManager::new(),
            speculation_cache: SpeculationCache::new(),
            jit,
            jit_return_value: None,
            function_closures: HashMap::new(),
            heap,
            _runtime_heap_binding: runtime_heap_binding,
            exc_state: ExceptionState::default(),
            handler_stack: HandlerStack::new(),
            active_exception: None,
            active_exception_type_id: None,
            exc_info_stack: ExcInfoStack::new(),
            active_except_handlers: Vec::new(),
            import_resolver,
            import_verbosity: 0,
            compiler_optimization: OptimizationLevel::None,
            execution_budget: ExecutionBudget::default(),
            last_aot_error: None,
        }
    }

    /// Create with pre-populated globals.
    pub fn with_globals(globals: GlobalScope) -> Self {
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
        let heap = Box::new(ManagedHeap::with_defaults());
        let runtime_heap_binding = RuntimeHeapBinding::register(heap.heap());
        Self {
            frames: Vec::with_capacity(64),
            frame_pool: FramePool::new(),
            current_frame_idx: 0,
            globals,
            builtins,
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
            ic_manager: ICManager::new(),
            speculation_cache: SpeculationCache::new(),
            jit: None,
            jit_return_value: None,
            function_closures: HashMap::new(),
            heap,
            _runtime_heap_binding: runtime_heap_binding,
            exc_state: ExceptionState::default(),
            handler_stack: HandlerStack::new(),
            active_exception: None,
            active_exception_type_id: None,
            exc_info_stack: ExcInfoStack::new(),
            active_except_handlers: Vec::new(),
            import_resolver,
            import_verbosity: 0,
            compiler_optimization: OptimizationLevel::None,
            execution_budget: ExecutionBudget::default(),
            last_aot_error: None,
        }
    }

    /// Switch active frame.
    #[inline(always)]
    fn set_current_frame_idx(&mut self, idx: usize) {
        self.current_frame_idx = idx;
    }

    #[inline]
    fn restore_outer_except_handler(&mut self) {
        if let Some(previous) = self.active_except_handlers.last().copied() {
            self.active_exception = Some(previous.value);
            self.active_exception_type_id = Some(previous.type_id);
            self.exc_state = ExceptionState::Handling;
        } else {
            self.active_exception = None;
            self.active_exception_type_id = None;
            self.exc_state = ExceptionState::Normal;
        }
    }

    #[inline]
    fn discard_except_handlers_for_frame(&mut self, frame_id: u32) {
        while self
            .active_except_handlers
            .last()
            .is_some_and(|entry| entry.frame_id == frame_id)
        {
            self.active_except_handlers.pop();
        }
        if self.exc_state == ExceptionState::Handling {
            self.restore_outer_except_handler();
        }
    }

    #[inline]
    pub(crate) fn capture_exception_context(&self) -> ExceptionContextSnapshot {
        ExceptionContextSnapshot {
            exc_state: self.exc_state,
            handler_stack: self.handler_stack.clone(),
            active_exception: self.active_exception,
            active_exception_type_id: self.active_exception_type_id,
            exc_info_stack: self.exc_info_stack.clone(),
            active_except_handlers: self.active_except_handlers.clone(),
        }
    }

    #[inline]
    pub(crate) fn restore_exception_context(&mut self, snapshot: ExceptionContextSnapshot) {
        self.exc_state = snapshot.exc_state;
        self.handler_stack = snapshot.handler_stack;
        self.active_exception = snapshot.active_exception;
        self.active_exception_type_id = snapshot.active_exception_type_id;
        self.exc_info_stack = snapshot.exc_info_stack;
        self.active_except_handlers = snapshot.active_except_handlers;
    }

    // =========================================================================
    // Execution
    // =========================================================================

    /// Execute a code object and return the result.
    pub fn execute(&mut self, code: Arc<CodeObject>) -> PrismResult<Value> {
        self.execute_in_module(code, Arc::new(ModuleObject::new("__main__")))
    }

    /// Main dispatch loop.
    #[inline(never)] // Prevent inlining for better branch prediction
    fn run_loop(&mut self) -> PrismResult<Value> {
        loop {
            self.execution_budget.consume_step()?;

            // Fetch instruction
            let inst = {
                let frame = &mut self.frames[self.current_frame_idx];

                // Check if we've reached the end of the code
                if frame.ip as usize >= frame.code.instructions.len() {
                    // Implicit return None at end of function
                    match self.pop_frame(Value::none())? {
                        Some(value) => return Ok(value),
                        None => continue,
                    }
                }

                frame.fetch()
            };

            // Dispatch to handler
            let handler = get_handler(inst.opcode());
            let control = handler(self, inst);

            // Handle control flow
            match control {
                ControlFlow::Continue => {}

                ControlFlow::Jump(offset) => {
                    // Apply relative jump
                    // Note: offset is computed by compiler relative to instruction after jump,
                    // and fetch() already advanced ip, so just add offset directly
                    let frame = &mut self.frames[self.current_frame_idx];
                    let new_ip = (frame.ip as i32) + (offset as i32);
                    frame.ip = new_ip.max(0) as u32;
                }

                ControlFlow::Call { code, return_reg } => {
                    self.push_frame(code, return_reg)?;
                }

                ControlFlow::Return(value) => {
                    match self.pop_frame(value)? {
                        Some(result) => return Ok(result),
                        None => {} // Continue with caller frame
                    }
                }

                // =========================================================
                // Exception Handling
                // =========================================================
                ControlFlow::Exception {
                    type_id,
                    handler_pc: hint_pc,
                } => {
                    // Store the active exception for handlers to access
                    // (handler_pc in the ControlFlow is a hint from raise instruction encoding)
                    let _ = hint_pc; // Compiler hint, actual PC comes from exception table

                    // Look up handler in current frame's exception table
                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        // Found a matching handler - jump to it
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.ip = handler_entry;
                    } else {
                        // No handler in current frame - unwind to caller
                        loop {
                            // Pop current frame
                            if self.frames.len() <= 1 {
                                // No more frames - return as uncaught exception
                                return Err(self.uncaught_exception_error(type_id).into());
                            }

                            self.pop_top_frame_for_unwind();

                            // Try to find handler in caller
                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                let frame = &mut self.frames[self.current_frame_idx];
                                frame.ip = handler_entry;
                                break;
                            }
                        }
                    }
                }

                ControlFlow::Reraise => {
                    // Re-raise the current active exception
                    // First check active_exception_type_id (for except handlers)
                    // then fall back to exc_info_stack (for finally blocks)
                    let type_id = if let Some(tid) = self.active_exception_type_id {
                        tid
                    } else if let Some(exc_info) = self.exc_info_stack.peek() {
                        exc_info.type_id()
                    } else {
                        return Err(
                            RuntimeError::type_error("No active exception to re-raise").into()
                        );
                    };

                    if type_id == 0 {
                        return Err(RuntimeError::internal(
                            "Reraise without active exception type",
                        )
                        .into());
                    }

                    // Look for handler (same as Exception flow)
                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.ip = handler_entry;
                    } else {
                        // Unwind and propagate
                        loop {
                            if self.frames.len() <= 1 {
                                return Err(self.uncaught_reraised_exception_error(type_id).into());
                            }

                            self.pop_top_frame_for_unwind();

                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                let frame = &mut self.frames[self.current_frame_idx];
                                frame.ip = handler_entry;
                                break;
                            }
                        }
                    }
                }

                ControlFlow::EnterHandler {
                    handler_pc,
                    stack_depth: _,
                } => {
                    // Jump to handler code
                    let frame = &mut self.frames[self.current_frame_idx];
                    frame.ip = handler_pc;
                }

                ControlFlow::EnterFinally {
                    finally_pc,
                    stack_depth: _,
                    reraise: _,
                } => {
                    // Jump to finally block
                    let frame = &mut self.frames[self.current_frame_idx];
                    frame.ip = finally_pc;
                }

                ControlFlow::ExitHandler => {
                    // Handler completed, resume normal execution
                    self.pop_exception_handler();
                }

                // =========================================================
                // Generator Protocol
                // =========================================================
                ControlFlow::Yield {
                    value,
                    resume_point,
                } => {
                    // Generator suspension: capture frame state and return value
                    //
                    // When a generator yields:
                    // 1. Store the resume point (which yield we're at)
                    // 2. Store the current IP for resumption
                    // 3. Return the yielded value to the caller
                    //
                    // The generator's frame state (registers) is already captured
                    // by the GeneratorObject via FrameStorage when the generator
                    // was created. The VM only needs to track the resume_point.

                    // Store resume information in the frame for later
                    let frame = &mut self.frames[self.current_frame_idx];
                    frame.set_yield_point(resume_point);

                    // Pop the generator frame and return the yielded value
                    // The caller (either user code or the iterator protocol)
                    // will receive this value
                    return Ok(value);
                }

                ControlFlow::Resume { send_value } => {
                    // Generator resumption: restore frame state and continue
                    //
                    // When a generator is resumed via next()/send():
                    // 1. The send_value becomes the result of the yield expression
                    // 2. Execution continues from the stored resume point
                    //
                    // For now, place the sent value in register 0 (result register)
                    // and continue execution from where we left off.

                    let frame = &mut self.frames[self.current_frame_idx];

                    // The sent value becomes the result of the yield expression
                    // Register 0 is the conventional result register for yield
                    frame.set_reg(0, send_value);

                    // Continue normal execution from current IP
                    // (The IP was preserved when we yielded)
                }

                ControlFlow::Error(err) => {
                    let type_id = self.materialize_active_exception_from_runtime_error(&err);

                    // Try to find a handler in current frame
                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.ip = handler_entry;
                    } else {
                        // Unwind stack looking for handler
                        loop {
                            if self.frames.len() <= 1 {
                                // No handlers found - propagate the error
                                return Err(err.into());
                            }

                            self.pop_top_frame_for_unwind();

                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                let frame = &mut self.frames[self.current_frame_idx];
                                frame.ip = handler_entry;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Map a runtime error kind to a concrete exception type ID.
    #[inline(always)]
    fn runtime_error_exception_type_id(kind: &crate::error::RuntimeErrorKind) -> u16 {
        use crate::error::RuntimeErrorKind;
        use crate::stdlib::exceptions::types::ExceptionTypeId;

        match kind {
            RuntimeErrorKind::TypeError { .. }
            | RuntimeErrorKind::UnsupportedOperandTypes { .. }
            | RuntimeErrorKind::NotCallable { .. }
            | RuntimeErrorKind::NotIterable { .. }
            | RuntimeErrorKind::NotSubscriptable { .. } => {
                ExceptionTypeId::TypeError.as_u8() as u16
            }
            RuntimeErrorKind::NameError { .. } => ExceptionTypeId::NameError.as_u8() as u16,
            RuntimeErrorKind::AttributeError { .. } => {
                ExceptionTypeId::AttributeError.as_u8() as u16
            }
            RuntimeErrorKind::UnboundLocalError { .. } => {
                ExceptionTypeId::UnboundLocalError.as_u8() as u16
            }
            RuntimeErrorKind::IndexError { .. } => ExceptionTypeId::IndexError.as_u8() as u16,
            RuntimeErrorKind::KeyError { .. } => ExceptionTypeId::KeyError.as_u8() as u16,
            RuntimeErrorKind::ValueError { .. } => ExceptionTypeId::ValueError.as_u8() as u16,
            RuntimeErrorKind::ZeroDivisionError { .. } => {
                ExceptionTypeId::ZeroDivisionError.as_u8() as u16
            }
            RuntimeErrorKind::OverflowError { .. } => ExceptionTypeId::OverflowError.as_u8() as u16,
            RuntimeErrorKind::StopIteration => ExceptionTypeId::StopIteration.as_u8() as u16,
            RuntimeErrorKind::GeneratorExit => ExceptionTypeId::GeneratorExit.as_u8() as u16,
            RuntimeErrorKind::AssertionError { .. } => {
                ExceptionTypeId::AssertionError.as_u8() as u16
            }
            RuntimeErrorKind::RecursionError { .. } => {
                ExceptionTypeId::RecursionError.as_u8() as u16
            }
            RuntimeErrorKind::ExecutionLimitExceeded { .. } => {
                ExceptionTypeId::RuntimeError.as_u8() as u16
            }
            RuntimeErrorKind::ImportError { missing, .. } => {
                if *missing {
                    ExceptionTypeId::ModuleNotFoundError.as_u8() as u16
                } else {
                    ExceptionTypeId::ImportError.as_u8() as u16
                }
            }
            RuntimeErrorKind::InvalidOpcode { .. } => ExceptionTypeId::SystemError.as_u8() as u16,
            RuntimeErrorKind::InternalError { .. } => ExceptionTypeId::RuntimeError.as_u8() as u16,
            RuntimeErrorKind::Exception { type_id, .. } => *type_id,
        }
    }

    /// Materialize and register active exception state from a runtime error.
    #[inline]
    fn materialize_active_exception_from_runtime_error(&mut self, err: &RuntimeError) -> u16 {
        use crate::stdlib::exceptions::types::ExceptionTypeId;

        let type_id = Self::runtime_error_exception_type_id(&err.kind);
        if let Some(value) = err.raised_value {
            self.set_active_exception_with_type(value, type_id);
            return type_id;
        }
        let exc_type_id_enum =
            ExceptionTypeId::from_u8(type_id as u8).unwrap_or(ExceptionTypeId::RuntimeError);
        let exc_value = match &err.kind {
            crate::error::RuntimeErrorKind::ImportError {
                message,
                name,
                path,
                ..
            } => crate::builtins::create_exception_with_import_details(
                exc_type_id_enum,
                Some(Arc::clone(message)),
                name.clone(),
                path.clone(),
            ),
            _ => {
                let error_message = err.to_string();
                crate::builtins::create_exception(
                    exc_type_id_enum,
                    Some(Arc::from(error_message.as_str())),
                )
            }
        };
        self.set_active_exception_with_type(exc_value, type_id);
        type_id
    }

    #[inline]
    fn raised_exception_message(value: Value) -> Arc<str> {
        unsafe { crate::builtins::ExceptionValue::from_value(value) }
            .map(|exception| {
                let message = exception.display_text();
                if message.is_empty() {
                    Arc::<str>::from(exception.repr_text())
                } else {
                    Arc::<str>::from(message)
                }
            })
            .unwrap_or_else(|| Arc::<str>::from("Uncaught exception"))
    }

    /// Propagate an active exception through generator-owned frames.
    ///
    /// Returns true when a handler was found and execution can continue.
    /// Returns false when propagation reaches the non-generator caller boundary.
    #[inline]
    fn propagate_exception_within_generator_frames(
        &mut self,
        type_id: u16,
        caller_depth: usize,
    ) -> bool {
        if let Some(handler_entry) = self.find_exception_handler(type_id) {
            let frame = &mut self.frames[self.current_frame_idx];
            frame.ip = handler_entry;
            return true;
        }

        while self.frames.len() > caller_depth {
            self.pop_top_frame_for_unwind();

            if self.frames.len() <= caller_depth {
                return false;
            }

            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                let frame = &mut self.frames[self.current_frame_idx];
                frame.ip = handler_entry;
                return true;
            }
        }

        false
    }

    // =========================================================================
    // Frame Management
    // =========================================================================

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

    #[inline]
    fn acquire_frame(
        &mut self,
        code: Arc<CodeObject>,
        return_frame: Option<u32>,
        return_reg: u8,
        closure: Option<Arc<ClosureEnv>>,
        module: Option<Arc<ModuleObject>>,
    ) -> Frame {
        self.frame_pool
            .acquire(code, return_frame, return_reg, closure, module)
    }

    #[inline]
    fn recycle_frame(&mut self, frame: Frame) {
        self.frame_pool.release(frame);
    }

    fn recycle_all_frames(&mut self) {
        while let Some(frame) = self.frames.pop() {
            self.recycle_frame(frame);
        }
    }

    #[cfg(test)]
    fn pooled_frame_count(&self) -> usize {
        self.frame_pool.len()
    }

    fn resume_generator(
        &mut self,
        generator: &mut GeneratorObject,
        mode: GeneratorResumeMode,
    ) -> VmResult<GeneratorResumeOutcome> {
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

        if self.frames.is_empty() {
            return Err(RuntimeError::internal(
                "cannot resume generator without an active caller frame",
            ));
        }

        let caller_idx = self.current_frame_idx;
        let caller_depth = self.frames.len();
        let caller_scratch_255 = self.frames[caller_idx].get_reg(255);
        let caller_exception_context = self.capture_exception_context();

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
        generator.restore(&mut frame.registers);
        if prev_state == RuntimeGeneratorState::Created {
            initialize_closure_cellvars_from_locals(
                &mut frame,
                generator.liveness().count() as usize,
            );
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
                if let Err(err) = self.execution_budget.consume_step() {
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
                        outcome = Some(GeneratorResumeOutcome::Yielded(value));
                        break 'exec;
                    }
                    ControlFlow::Resume { send_value } => {
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.set_reg(0, send_value);
                    }
                    ControlFlow::Error(err) => {
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

        if let Some(err) = failure {
            generator.exhaust();
            return Err(err);
        }

        self.restore_exception_context(caller_exception_context);

        match outcome {
            Some(result) => Ok(result),
            None => Err(RuntimeError::internal(
                "generator resume exited without outcome",
            )),
        }
    }

    /// Push a new frame for calling a function.
    ///
    /// This method implements a JIT-first dispatch strategy:
    /// 1. Profile the call and handle tier-up decisions
    /// 2. If compiled code exists, execute it directly
    /// 3. On JIT return, propagate value to caller
    /// 4. On deopt, create frame and resume interpreter
    /// 5. On miss, fall through to interpreter
    pub fn push_frame(&mut self, code: Arc<CodeObject>, return_reg: u8) -> VmResult<()> {
        self.push_frame_internal(code, return_reg, None, self.current_module_cloned(), true)
    }

    pub fn push_frame_with_module(
        &mut self,
        code: Arc<CodeObject>,
        return_reg: u8,
        module: Option<Arc<ModuleObject>>,
    ) -> VmResult<()> {
        self.push_frame_internal(code, return_reg, None, module, true)
    }

    /// Push a new frame with an optional captured closure environment.
    ///
    /// This path intentionally bypasses JIT dispatch because call opcodes must
    /// bind arguments into frame registers before execution starts.
    pub fn push_frame_with_closure(
        &mut self,
        code: Arc<CodeObject>,
        return_reg: u8,
        closure: Option<Arc<crate::frame::ClosureEnv>>,
    ) -> VmResult<()> {
        self.push_frame_with_closure_and_module(
            code,
            return_reg,
            closure,
            self.current_module_cloned(),
        )
    }

    pub fn push_frame_with_closure_and_module(
        &mut self,
        code: Arc<CodeObject>,
        return_reg: u8,
        closure: Option<Arc<crate::frame::ClosureEnv>>,
        module: Option<Arc<ModuleObject>>,
    ) -> VmResult<()> {
        self.push_frame_internal(code, return_reg, closure, module, false)
    }

    fn push_frame_internal(
        &mut self,
        code: Arc<CodeObject>,
        return_reg: u8,
        closure: Option<Arc<crate::frame::ClosureEnv>>,
        module: Option<Arc<ModuleObject>>,
        allow_jit: bool,
    ) -> VmResult<()> {
        // Check recursion limit
        if self.frames.len() >= MAX_RECURSION_DEPTH {
            return Err(RuntimeError::recursion_error(self.frames.len()));
        }

        // Record call for profiling
        let code_id = CodeId::from_ptr(Arc::as_ptr(&code) as *const ());
        self.profiler.record_call(code_id);

        // Handle JIT: check for compiled code, handle tier-up, and try execution
        if allow_jit && closure.is_none() && self.execution_budget.step_limit().is_none() {
            enum JitDispatchOutcome {
                Returned(Value),
                Deopt(Frame),
                Exception(RuntimeError),
                Continue,
            }

            let return_frame_idx = if self.frames.is_empty() {
                None
            } else {
                Some(self.current_frame_idx as u32)
            };

            let jit_outcome = {
                let profiler = &self.profiler;
                let (frame_pool, jit_slot) = (&mut self.frame_pool, &mut self.jit);

                if let Some(jit) = jit_slot.as_mut() {
                    let tier_decision = jit.check_tier_up(profiler, code_id);

                    if tier_decision != TierUpDecision::None {
                        jit.handle_tier_up(&code, tier_decision);
                    }

                    let code_ptr_id = Arc::as_ptr(&code) as u64;

                    if jit.lookup(code_ptr_id).is_some() {
                        let mut jit_frame = frame_pool.acquire(
                            Arc::clone(&code),
                            return_frame_idx,
                            return_reg,
                            None,
                            module.clone(),
                        );

                        Some(match jit.try_execute(code_ptr_id, &mut jit_frame) {
                            Some(ExecutionResult::Return(value)) => {
                                frame_pool.release(jit_frame);
                                JitDispatchOutcome::Returned(value)
                            }
                            Some(ExecutionResult::Deopt { bc_offset, reason }) => {
                                jit.handle_deopt(code_ptr_id, reason);
                                jit_frame.ip = bc_offset;
                                JitDispatchOutcome::Deopt(jit_frame)
                            }
                            Some(ExecutionResult::Exception(err)) => {
                                frame_pool.release(jit_frame);
                                JitDispatchOutcome::Exception(err)
                            }
                            Some(ExecutionResult::TailCall { .. }) => {
                                // TODO: Implement tail call optimization.
                                frame_pool.release(jit_frame);
                                jit.record_miss();
                                JitDispatchOutcome::Continue
                            }
                            None => {
                                frame_pool.release(jit_frame);
                                jit.record_miss();
                                JitDispatchOutcome::Continue
                            }
                        })
                    } else {
                        jit.record_miss();
                        Some(JitDispatchOutcome::Continue)
                    }
                } else {
                    None
                }
            };

            if let Some(outcome) = jit_outcome {
                match outcome {
                    JitDispatchOutcome::Returned(value) => {
                        if self.frames.is_empty() {
                            self.jit_return_value = Some(value);
                        } else {
                            self.frames[self.current_frame_idx].set_reg(return_reg, value);
                        }
                        return Ok(());
                    }
                    JitDispatchOutcome::Deopt(frame) => {
                        self.frames.push(frame);
                        self.set_current_frame_idx(self.frames.len() - 1);
                        return Ok(());
                    }
                    JitDispatchOutcome::Exception(err) => return Err(err),
                    JitDispatchOutcome::Continue => {}
                }
            }
        }

        // Fall through to interpreter - push frame normally
        let return_frame = if self.frames.is_empty() {
            None
        } else {
            Some(self.current_frame_idx as u32)
        };

        let frame = self.acquire_frame(code, return_frame, return_reg, closure, module);
        self.frames.push(frame);
        self.set_current_frame_idx(self.frames.len() - 1);

        Ok(())
    }

    /// Pop the top frame during exception/generator unwinding.
    ///
    /// Keeps handler stack entries for the popped frame in sync.
    #[inline]
    fn pop_top_frame_for_unwind(&mut self) {
        let top_idx = self.frames.len() - 1;
        self.handler_stack.pop_frame_handlers(top_idx as u32);
        self.discard_except_handlers_for_frame(top_idx as u32);
        if let Some(frame) = self.frames.pop() {
            self.recycle_frame(frame);
        }
        self.set_current_frame_idx(self.frames.len().saturating_sub(1));
    }

    /// Pop the current frame without writing its return value into the caller.
    ///
    /// Internal execution paths such as imported-module initialization and
    /// class-body namespace collection do not semantically return a value to
    /// their caller. Discarding the frame result here prevents clobbering live
    /// caller registers while still restoring the correct caller frame.
    #[inline]
    fn pop_frame_discarding_return(&mut self) -> VmResult<bool> {
        let top_idx = self.frames.len() - 1;
        self.handler_stack.pop_frame_handlers(top_idx as u32);
        self.discard_except_handlers_for_frame(top_idx as u32);
        let frame = self.frames.pop().expect("no frame to pop");
        let return_frame = frame.return_frame;
        self.recycle_frame(frame);

        if self.frames.is_empty() {
            self.set_current_frame_idx(0);
            Ok(true)
        } else {
            let return_frame_idx = return_frame.unwrap_or(0) as usize;
            self.set_current_frame_idx(return_frame_idx);
            Ok(false)
        }
    }

    /// Pop the current frame and return to caller.
    /// Returns Some(value) if this was the last frame, None otherwise.
    pub fn pop_frame(&mut self, return_value: Value) -> VmResult<Option<Value>> {
        let top_idx = self.frames.len() - 1;
        self.handler_stack.pop_frame_handlers(top_idx as u32);
        self.discard_except_handlers_for_frame(top_idx as u32);
        let frame = self.frames.pop().expect("no frame to pop");
        let return_frame_idx = frame.return_frame.unwrap_or(0) as usize;
        let return_reg = frame.return_reg;
        self.recycle_frame(frame);

        if self.frames.is_empty() {
            // This was the last frame - return final value
            self.set_current_frame_idx(0);
            Ok(Some(return_value))
        } else {
            // Store return value in caller's register
            self.set_current_frame_idx(return_frame_idx);
            self.frames[return_frame_idx].set_reg(return_reg, return_value);

            Ok(None)
        }
    }

    /// Get reference to current frame.
    #[inline(always)]
    pub fn current_frame(&self) -> &Frame {
        &self.frames[self.current_frame_idx]
    }

    /// Get mutable reference to current frame.
    #[inline(always)]
    pub fn current_frame_mut(&mut self) -> &mut Frame {
        &mut self.frames[self.current_frame_idx]
    }

    // =========================================================================
    // State Access
    // =========================================================================

    /// Get the current call depth.
    #[inline]
    pub fn call_depth(&self) -> usize {
        self.frames.len()
    }

    /// Check if VM is idle (no frames).
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.frames.is_empty()
    }

    /// Reset VM state for reuse.
    pub fn reset(&mut self) {
        self.recycle_all_frames();
        self.set_current_frame_idx(0);
        self.function_closures.clear();
        self.globals = GlobalScope::new();
        self.inline_caches = InlineCacheStore::default();
        self.exc_state = ExceptionState::default();
        self.handler_stack.clear();
        self.active_exception = None;
        self.active_exception_type_id = None;
        self.exc_info_stack.clear();
        self.active_except_handlers.clear();
    }

    /// Clear only the frame stack (keep globals).
    pub fn clear_frames(&mut self) {
        self.recycle_all_frames();
        self.set_current_frame_idx(0);
        self.function_closures.clear();
        self.handler_stack.clear();
        self.active_exception = None;
        self.active_exception_type_id = None;
        self.exc_info_stack.clear();
        self.active_except_handlers.clear();
        self.exc_state = ExceptionState::Normal;
    }

    /// Register captured closure environment for a function object.
    #[inline]
    pub fn register_function_closure(
        &mut self,
        func_ptr: *const (),
        closure: Arc<crate::frame::ClosureEnv>,
    ) {
        self.function_closures.insert(func_ptr, closure);
    }

    /// Look up captured closure environment for a function object.
    #[inline]
    pub fn lookup_function_closure(
        &self,
        func_ptr: *const (),
    ) -> Option<Arc<crate::frame::ClosureEnv>> {
        self.function_closures.get(&func_ptr).cloned()
    }

    // =========================================================================
    // GC Integration
    // =========================================================================

    /// Get read-only access to the managed heap.
    ///
    /// Use this to query heap statistics, check collection thresholds,
    /// or read heap configuration.
    #[inline]
    pub fn heap(&self) -> &ManagedHeap {
        self.heap.as_ref()
    }

    /// Get mutable access to the managed heap.
    ///
    /// Required for:
    /// - Triggering garbage collection
    /// - Updating root sets
    /// - Modifying heap configuration
    #[inline]
    pub fn heap_mut(&mut self) -> &mut ManagedHeap {
        self.heap.as_mut()
    }

    /// Get a typed allocator for GC-managed object allocation.
    ///
    /// This provides a zero-cost typed interface for allocating objects
    /// on the GC heap. The allocator borrows the underlying GcHeap,
    /// ensuring type-safe allocation with proper Trace bounds.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let list = vm.allocator().alloc(ListObject::from_slice(&values))?;
    /// let value = Value::object_ptr(list as *const ());
    /// ```
    ///
    /// # Performance
    ///
    /// This method is `#[inline]` and creates a zero-cost wrapper.
    /// The allocator itself is stack-allocated and holds only a reference.
    #[inline]
    pub fn allocator(&self) -> GcAllocator<'_> {
        GcAllocator::new(self.heap.heap())
    }

    // =========================================================================
    // Exception Handling
    // =========================================================================

    /// Get the current frame's ID for exception handler tracking.
    #[inline]
    pub fn current_frame_id(&self) -> u32 {
        self.current_frame_idx as u32
    }

    /// Set the active exception being propagated.
    /// Uses generic Exception type (4) - prefer set_active_exception_with_type for proper matching.
    #[inline]
    pub fn set_active_exception(&mut self, exc: Value) {
        self.active_exception = Some(exc);
        self.active_exception_type_id = Some(4); // Generic Exception type
        self.exc_state = ExceptionState::Propagating;
    }

    /// Set the active exception with a specific type ID.
    /// This enables proper exception type matching in except handlers.
    #[inline]
    pub fn set_active_exception_with_type(&mut self, exc: Value, type_id: u16) {
        self.active_exception = Some(exc);
        self.active_exception_type_id = Some(type_id);
        self.exc_state = ExceptionState::Propagating;
    }

    /// Get the active exception if any.
    #[inline]
    pub fn get_active_exception(&self) -> Option<&Value> {
        self.active_exception.as_ref()
    }

    /// Check if there's an active exception.
    #[inline]
    pub fn has_active_exception(&self) -> bool {
        self.active_exception.is_some()
    }

    /// Clear the active exception.
    #[inline]
    pub fn clear_active_exception(&mut self) {
        self.active_exception = None;
        self.active_exception_type_id = None;
    }

    /// Get the type ID of the active exception.
    ///
    /// Returns the exception type ID for fast matching, or None if no
    /// active exception exists.
    #[inline]
    pub fn get_active_exception_type_id(&self) -> Option<u16> {
        self.active_exception_type_id
    }

    #[inline]
    fn active_exception_message(&self) -> Option<String> {
        let active = *self.get_active_exception()?;
        let exception = unsafe { crate::builtins::ExceptionValue::from_value(active) }?;
        let message = exception.display_text();
        if !message.is_empty() {
            return Some(message);
        }

        Some(exception.repr_text())
    }

    #[inline]
    fn uncaught_exception_error(&self, type_id: u16) -> RuntimeError {
        let message = self
            .active_exception_message()
            .unwrap_or_else(|| format!("Uncaught exception (type_id={type_id})"));
        if let Some(value) = self.active_exception {
            return RuntimeError::raised_exception(type_id, value, message);
        }
        RuntimeError::exception(type_id, message)
    }

    #[inline]
    fn uncaught_reraised_exception_error(&self, type_id: u16) -> RuntimeError {
        let message = self
            .active_exception_message()
            .unwrap_or_else(|| "Uncaught re-raised exception".to_string());
        if let Some(value) = self.active_exception {
            return RuntimeError::raised_exception(type_id, value, message);
        }
        RuntimeError::exception(type_id, message)
    }

    /// Enter an `except` handler and preserve its exception for nested handlers.
    #[inline]
    pub fn enter_except_handler(&mut self) -> bool {
        let Some(value) = self.active_exception else {
            return false;
        };
        let Some(type_id) = self.active_exception_type_id else {
            return false;
        };

        self.active_except_handlers.push(ActiveExceptHandler {
            frame_id: self.current_frame_id(),
            value,
            type_id,
        });
        self.exc_state = ExceptionState::Handling;
        true
    }

    /// Exit the current `except` handler normally.
    #[inline]
    pub fn exit_except_handler(&mut self) -> bool {
        if self.active_except_handlers.pop().is_none() {
            return false;
        }

        self.restore_outer_except_handler();
        true
    }

    /// Abort the current `except` handler while preserving the new propagating exception.
    #[inline]
    pub fn abort_except_handler(&mut self) -> bool {
        if self.active_except_handlers.pop().is_none() {
            return false;
        }

        self.exc_state = if self.active_exception_type_id.is_some() {
            ExceptionState::Propagating
        } else {
            ExceptionState::Normal
        };
        true
    }

    /// Push an exception handler onto the handler stack.
    ///
    /// Returns false if the stack is full.
    #[inline]
    pub fn push_exception_handler(&mut self, frame: crate::exception::HandlerFrame) -> bool {
        self.handler_stack.push(frame)
    }

    /// Pop an exception handler from the handler stack.
    #[inline]
    pub fn pop_exception_handler(&mut self) -> Option<crate::exception::HandlerFrame> {
        self.handler_stack.pop()
    }

    /// Check if we should reraise after a finally block.
    ///
    /// We should reraise if there's still an active exception after the finally
    /// body executes. Note: We can't rely on exc_state == Finally because
    /// PopExcInfo may have changed it during the finally execution.
    #[inline]
    pub fn should_reraise_after_finally(&self) -> bool {
        self.exc_state == ExceptionState::Propagating
            && self.active_exception_type_id.is_some()
            && self.active_exception_type_id != Some(0)
    }

    /// Clear the reraise flag after handling.
    #[inline]
    pub fn clear_reraise_flag(&mut self) {
        // Transition state - exception will be preserved for reraise
    }

    /// Clear exception state (after successful handling).
    #[inline]
    pub fn clear_exception_state(&mut self) {
        self.exc_state = ExceptionState::Normal;
    }

    /// Get the current exception state.
    #[inline]
    pub fn exception_state(&self) -> ExceptionState {
        self.exc_state
    }

    /// Set the exception state directly.
    #[inline]
    pub fn set_exception_state(&mut self, state: ExceptionState) {
        self.exc_state = state;
    }

    /// Cache a handler lookup result for fast path.
    #[inline]
    pub fn cache_handler(&mut self, pc: u32, handler_idx: u16) {
        self.frames[self.current_frame_idx]
            .handler_cache
            .record(pc, handler_idx);
    }

    /// Look up a cached handler for a PC.
    #[inline]
    pub fn lookup_cached_handler(&mut self, pc: u32) -> Option<u16> {
        self.frames[self.current_frame_idx]
            .handler_cache
            .try_get(pc)
    }

    /// Get the handler stack depth.
    #[inline]
    pub fn handler_stack_depth(&self) -> usize {
        self.handler_stack.len()
    }

    /// Find an exception handler for the given exception type in the current frame.
    ///
    /// Searches the frame's exception table for a handler that:
    /// 1. Covers the current PC (start_pc <= pc < end_pc)
    /// 2. Matches the exception type (or is a catch-all with type_idx = 0xFFFF)
    ///
    /// Returns the handler PC if a matching handler is found.
    ///
    /// # Performance
    ///
    /// Exception tables are typically small (<10 entries).
    /// We prefer the most specific covering range so nested handlers work even
    /// when entries are emitted in source order rather than sorted order.
    #[inline]
    pub fn find_exception_handler(&mut self, _type_id: u16) -> Option<u32> {
        if self.frames.is_empty() {
            return None;
        }

        let pc = {
            let frame = &self.frames[self.current_frame_idx];
            frame.ip.saturating_sub(1) // PC is post-increment, so -1 for current instruction
        };

        // Fast path: cached handler for this PC.
        if let Some(cached_idx) = self.lookup_cached_handler(pc) {
            if let Some(entry) = self.frames[self.current_frame_idx]
                .code
                .exception_table
                .get(cached_idx as usize)
                && pc >= entry.start_pc
                && pc < entry.end_pc
            {
                return Some(entry.handler_pc);
            }
            self.frames[self.current_frame_idx]
                .handler_cache
                .invalidate();
        }

        // Slow path: linear scan for the most specific covering region.
        let matched = {
            let frame = &self.frames[self.current_frame_idx];
            let mut matched: Option<(u16, u32, u32, u32)> = None;
            for (idx, entry) in frame.code.exception_table.iter().enumerate() {
                if pc < entry.start_pc || pc >= entry.end_pc {
                    continue;
                }

                let span = entry.end_pc.saturating_sub(entry.start_pc);
                let replace = match matched {
                    None => true,
                    Some((_, _, best_span, best_start)) => {
                        span < best_span || (span == best_span && entry.start_pc > best_start)
                    }
                };

                if replace {
                    matched = Some((idx as u16, entry.handler_pc, span, entry.start_pc));
                }
            }
            matched.map(|(idx, handler_pc, _, _)| (idx, handler_pc))
        };

        if let Some((handler_idx, handler_pc)) = matched {
            self.cache_handler(pc, handler_idx);
            return Some(handler_pc);
        }

        self.frames[self.current_frame_idx]
            .handler_cache
            .record_miss(pc);
        None
    }

    // =========================================================================
    // Exception Info Stack (CPython 3.11+ semantics)
    // =========================================================================

    /// Get a reference to the exception info stack.
    #[inline]
    pub fn exc_info_stack(&self) -> &ExcInfoStack {
        &self.exc_info_stack
    }

    /// Get a mutable reference to the exception info stack.
    #[inline]
    pub fn exc_info_stack_mut(&mut self) -> &mut ExcInfoStack {
        &mut self.exc_info_stack
    }

    /// Push current exception info onto the stack.
    /// Returns false if stack is full.
    #[inline]
    pub fn push_exc_info(&mut self) -> bool {
        use crate::exception::{EntryFlags, ExcInfoEntry};

        let type_id = self.get_active_exception_type_id().unwrap_or(0);
        let value = self.active_exception.clone();
        let mut entry = ExcInfoEntry::new(type_id, value);
        if self.exc_state == ExceptionState::Handling {
            entry.flags_mut().set(EntryFlags::HANDLING);
        }
        self.exc_info_stack.push(entry)
    }

    /// Pop exception info from the stack and restore it as active.
    #[inline]
    pub fn pop_exc_info(&mut self) -> bool {
        use crate::exception::EntryFlags;

        if let Some(entry) = self.exc_info_stack.pop() {
            if entry.is_active() {
                self.active_exception = entry.value_cloned();
                self.active_exception_type_id = Some(entry.type_id());
                self.exc_state = if entry.flags().has(EntryFlags::HANDLING) {
                    ExceptionState::Handling
                } else {
                    ExceptionState::Propagating
                };
            } else {
                self.active_exception = None;
                self.active_exception_type_id = None;
                self.exc_state = ExceptionState::Normal;
            }
            true
        } else {
            false
        }
    }

    /// Check if there's exception info on the stack.
    #[inline]
    pub fn has_exc_info(&self) -> bool {
        !self.exc_info_stack.is_empty() || self.active_exception.is_some()
    }

    /// Get current exception info as (type_id, value, traceback_id).
    #[inline]
    pub fn current_exc_info(&self) -> (Option<u16>, Option<Value>, Option<u32>) {
        self.exc_info_stack.current_exc_info()
    }
}

impl Default for VirtualMachine {
    fn default() -> Self {
        Self::new()
    }
}

fn module_package_name(name: &str, is_package: bool) -> Arc<str> {
    if is_package {
        Arc::from(name)
    } else if let Some((package, _)) = name.rsplit_once('.') {
        Arc::from(package)
    } else {
        Arc::from("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::builtin_getattr;
    use crate::exception::HandlerFrame;
    use crate::import::FrozenModuleSource;
    use prism_code::{CodeFlags, CodeObject, ExceptionEntry};
    use prism_compiler::{Compiler, OptimizationLevel};
    use prism_core::intern::intern;
    use prism_parser::parse;
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::type_obj::TypeId;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestTempDir {
        path: PathBuf,
    }

    impl TestTempDir {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time went backwards")
                .as_nanos();

            let mut path = std::env::temp_dir();
            path.push(format!(
                "prism_vm_tests_{}_{}_{}",
                std::process::id(),
                nanos,
                unique
            ));
            std::fs::create_dir_all(&path).expect("failed to create temp dir");
            Self { path }
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_file(path: &std::path::Path, content: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create parent dir");
        }
        std::fs::write(path, content).expect("failed to write file");
    }

    fn empty_code(name: &str) -> Arc<CodeObject> {
        Arc::new(CodeObject {
            name: Arc::from(name),
            register_count: 1,
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            instructions: Box::new([]),
            constants: Box::new([]),
            names: Box::new([]),
            locals: Box::new([]),
            freevars: Box::new([]),
            cellvars: Box::new([]),
            line_table: Box::new([]),
            exception_table: Box::new([]),
            filename: Arc::from("<test>"),
            qualname: Arc::from(name),
            flags: CodeFlags::NONE,
            first_lineno: 1,
            nested_code_objects: Box::new([]),
        })
    }

    fn code_with_exception_entries(
        name: &str,
        exception_table: Vec<ExceptionEntry>,
    ) -> Arc<CodeObject> {
        Arc::new(CodeObject {
            name: Arc::from(name),
            register_count: 1,
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            instructions: Box::new([]),
            constants: Box::new([]),
            names: Box::new([]),
            locals: Box::new([]),
            freevars: Box::new([]),
            cellvars: Box::new([]),
            line_table: Box::new([]),
            exception_table: exception_table.into_boxed_slice(),
            filename: Arc::from("<test>"),
            qualname: Arc::from(name),
            flags: CodeFlags::NONE,
            first_lineno: 1,
            nested_code_objects: Box::new([]),
        })
    }

    fn catch_all_entry(start_pc: u32, end_pc: u32, handler_pc: u32) -> ExceptionEntry {
        ExceptionEntry {
            start_pc,
            end_pc,
            handler_pc,
            finally_pc: u32::MAX,
            depth: 0,
            exception_type_idx: u16::MAX,
        }
    }

    fn compile_module(source: &str, filename: &str) -> Arc<CodeObject> {
        let parsed = parse(source).expect("source should parse");
        Arc::new(
            Compiler::compile_module_with_optimization(&parsed, filename, OptimizationLevel::Basic)
                .expect("source should compile"),
        )
    }

    #[test]
    fn test_vm_creation() {
        let vm = VirtualMachine::new();
        assert!(vm.is_idle());
        assert_eq!(vm.call_depth(), 0);
    }

    #[test]
    fn test_vm_import_verbosity_configuration_round_trips() {
        let mut vm = VirtualMachine::new();
        assert_eq!(vm.import_verbosity(), 0);
        vm.set_import_verbosity(2);
        assert_eq!(vm.import_verbosity(), 2);
    }

    #[test]
    fn test_vm_with_globals() {
        let mut globals = GlobalScope::new();
        globals.set("x".into(), Value::int(42).unwrap());

        let vm = VirtualMachine::with_globals(globals);
        assert_eq!(vm.globals.get("x").unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_imported_source_module_can_read_sys_prefix_family() {
        let temp = TestTempDir::new();
        write_file(
            &temp.path.join("probe.py"),
            "import sys\nVALUE = (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix)\n",
        );

        let mut vm = VirtualMachine::new();
        vm.import_resolver
            .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

        let module = vm
            .import_module_named("probe")
            .expect("probe module should import successfully");
        let value = module
            .get_attr("VALUE")
            .expect("probe module should publish VALUE");
        let tuple_ptr = value
            .as_object_ptr()
            .expect("VALUE should be stored as tuple object");
        let tuple = unsafe { &*(tuple_ptr as *const prism_runtime::types::tuple::TupleObject) };

        assert_eq!(tuple.len(), 4, "prefix family should expose four entries");
        for entry in tuple.iter() {
            let string_ptr = entry
                .as_string_object_ptr()
                .expect("prefix entry should be an interned string")
                as *const u8;
            let resolved = prism_core::intern::interned_by_ptr(string_ptr)
                .expect("prefix entry should resolve");
            assert!(
                !resolved.as_ref().is_empty(),
                "prefix entry should not be empty"
            );
        }
    }

    #[test]
    fn test_imported_source_module_can_use_builtin_warning_categories() {
        let temp = TestTempDir::new();
        write_file(
            &temp.path.join("warning_probe.py"),
            concat!(
                "VALUE = (\n",
                "    BytesWarning.__name__,\n",
                "    issubclass(BytesWarning, Warning),\n",
                "    issubclass(ResourceWarning, Warning),\n",
                "    issubclass(ImportWarning, Warning),\n",
                ")\n",
            ),
        );

        let mut vm = VirtualMachine::new();
        vm.import_resolver
            .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

        let module = vm
            .import_module_named("warning_probe")
            .expect("warning probe should import successfully");
        let value = module
            .get_attr("VALUE")
            .expect("warning probe should publish VALUE");
        let tuple_ptr = value
            .as_object_ptr()
            .expect("VALUE should be stored as tuple object");
        let tuple = unsafe { &*(tuple_ptr as *const prism_runtime::types::tuple::TupleObject) };

        assert_eq!(tuple.len(), 4, "warning probe should expose four entries");

        let name_ptr = tuple.as_slice()[0]
            .as_string_object_ptr()
            .expect("first tuple entry should be an interned string")
            as *const u8;
        let resolved = prism_core::intern::interned_by_ptr(name_ptr)
            .expect("warning category name should resolve");
        assert_eq!(resolved.as_ref(), "BytesWarning");

        for entry in &tuple.as_slice()[1..] {
            assert_eq!(
                entry.as_bool(),
                Some(true),
                "warning category relationship should be true",
            );
        }
    }

    #[test]
    fn test_imported_source_module_supports_metaclass_prepare_dict_subclasses() {
        let temp = TestTempDir::new();
        write_file(
            &temp.path.join("prepare_probe.py"),
            concat!(
                "class Namespace(dict):\n",
                "    def __setitem__(self, key, value):\n",
                "        dict.__setitem__(self, key, value)\n",
                "\n",
                "class Meta(type):\n",
                "    @classmethod\n",
                "    def __prepare__(mcls, name, bases):\n",
                "        return Namespace()\n",
                "\n",
                "class Target(metaclass=Meta):\n",
                "    answer = 42\n",
                "    label = 'ready'\n",
                "\n",
                "RESULT = (Target.answer, Target.label)\n",
            ),
        );

        let mut vm = VirtualMachine::new();
        vm.import_resolver
            .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

        let module = vm
            .import_module_named("prepare_probe")
            .expect("prepare probe should import successfully");
        let value = module
            .get_attr("RESULT")
            .expect("prepare probe should publish RESULT");
        let tuple_ptr = value
            .as_object_ptr()
            .expect("RESULT should be stored as tuple object");
        let tuple = unsafe { &*(tuple_ptr as *const prism_runtime::types::tuple::TupleObject) };

        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.as_slice()[0].as_int(), Some(42));
        let label_ptr = tuple.as_slice()[1]
            .as_string_object_ptr()
            .expect("label should be an interned string") as *const u8;
        let label =
            prism_core::intern::interned_by_ptr(label_ptr).expect("label string should resolve");
        assert_eq!(label.as_ref(), "ready");
    }

    #[test]
    fn test_imported_source_module_inherits_metaclass_prepare_dict_subclasses() {
        let temp = TestTempDir::new();
        write_file(
            &temp.path.join("prepare_inherit_probe.py"),
            concat!(
                "class Namespace(dict):\n",
                "    def __setitem__(self, key, value):\n",
                "        dict.__setitem__(self, key, value)\n",
                "\n",
                "class Meta(type):\n",
                "    @classmethod\n",
                "    def __prepare__(mcls, name, bases):\n",
                "        return Namespace()\n",
                "\n",
                "class Base(metaclass=Meta):\n",
                "    base = 'ok'\n",
                "\n",
                "class Derived(Base):\n",
                "    answer = 42\n",
                "\n",
                "RESULT = (Base.base, Derived.answer)\n",
            ),
        );

        let mut vm = VirtualMachine::new();
        vm.import_resolver
            .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

        let module = vm
            .import_module_named("prepare_inherit_probe")
            .expect("prepare inherit probe should import successfully");
        let value = module
            .get_attr("RESULT")
            .expect("prepare inherit probe should publish RESULT");
        let tuple_ptr = value
            .as_object_ptr()
            .expect("RESULT should be stored as tuple object");
        let tuple = unsafe { &*(tuple_ptr as *const prism_runtime::types::tuple::TupleObject) };

        assert_eq!(tuple.len(), 2);
        let base_ptr = tuple.as_slice()[0]
            .as_string_object_ptr()
            .expect("base marker should be an interned string") as *const u8;
        let base = prism_core::intern::interned_by_ptr(base_ptr)
            .expect("base marker string should resolve");
        assert_eq!(base.as_ref(), "ok");
        assert_eq!(tuple.as_slice()[1].as_int(), Some(42));
    }

    #[test]
    fn test_builtin_type_new_with_vm_preserves_class_result_after_handled_set_name_exception() {
        let mut vm = VirtualMachine::new();
        let module = Arc::new(ModuleObject::new("__main__"));

        vm.execute_in_module(
            compile_module(
                concat!(
                    "class Descriptor:\n",
                    "    def __set_name__(self, owner, name):\n",
                    "        try:\n",
                    "            {}['missing']\n",
                    "        except KeyError:\n",
                    "            pass\n",
                    "        owner.marker = name\n",
                    "\n",
                    "class Meta(type):\n",
                    "    pass\n",
                    "\n",
                    "DESCRIPTOR = Descriptor()\n",
                ),
                "<type-new-vm-probe>",
            ),
            Arc::clone(&module),
        )
        .expect("probe module should execute");

        let metaclass = module.get_attr("Meta").expect("Meta should be exported");
        let descriptor = module
            .get_attr("DESCRIPTOR")
            .expect("descriptor instance should be exported");

        let namespace_ptr = Box::into_raw(Box::new(prism_runtime::types::dict::DictObject::new()));
        unsafe {
            (*namespace_ptr).set(Value::string(intern("field")), descriptor);
        }
        let namespace_value = Value::object_ptr(namespace_ptr as *const ());

        let bases_ptr = Box::into_raw(Box::new(prism_runtime::types::tuple::TupleObject::empty()));
        let bases_value = Value::object_ptr(bases_ptr as *const ());

        vm.push_frame_with_module(
            empty_code("type_new_vm_probe"),
            0,
            Some(Arc::clone(&module)),
        )
        .expect("caller frame push should succeed");

        let result = crate::builtins::builtin_type_new_with_vm(
            &mut vm,
            &[
                metaclass,
                Value::string(intern("Example")),
                bases_value,
                namespace_value,
            ],
        )
        .expect("type.__new__ vm builtin should succeed");

        assert_ne!(
            result, namespace_value,
            "type.__new__ should not leak the namespace mapping as its result"
        );

        let result_ptr = result
            .as_object_ptr()
            .expect("type.__new__ should return an object-backed class");
        assert_eq!(
            crate::ops::objects::extract_type_id(result_ptr),
            TypeId::TYPE,
            "type.__new__ should return a class object, not a transient callback value",
        );

        let class = unsafe { &*(result_ptr as *const PyClassObject) };
        assert_eq!(
            class.metaclass(),
            metaclass,
            "type.__new__ should preserve the explicit heap metaclass",
        );

        let marker = class
            .get_attr(&intern("marker"))
            .expect("descriptor callback should publish marker attribute");
        let marker_ptr = marker
            .as_string_object_ptr()
            .expect("marker should be stored as an interned string")
            as *const u8;
        let marker_text =
            prism_core::intern::interned_by_ptr(marker_ptr).expect("marker should resolve");
        assert_eq!(
            marker_text.as_ref(),
            "field",
            "__set_name__ should still run against the created class",
        );

        vm.clear_frames();
        unsafe {
            drop(Box::from_raw(namespace_ptr));
            drop(Box::from_raw(bases_ptr));
        }
    }

    #[test]
    fn test_imported_source_module_supports_enum_style_metaclass_isinstance_checks() {
        let temp = TestTempDir::new();
        write_file(
            &temp.path.join("enum_meta_probe.py"),
            concat!(
                "class EnumType(type):\n",
                "    @classmethod\n",
                "    def __prepare__(metacls, cls, bases, **kwds):\n",
                "        if bases and not isinstance(bases[-1], EnumType):\n",
                "            raise TypeError('bad enum base')\n",
                "        return {}\n",
                "\n",
                "class Enum(metaclass=EnumType):\n",
                "    pass\n",
                "\n",
                "class ReprEnum(Enum):\n",
                "    pass\n",
                "\n",
                "RESULT = (isinstance(Enum, EnumType), isinstance(ReprEnum, EnumType))\n",
            ),
        );

        let mut vm = VirtualMachine::new();
        vm.import_resolver
            .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

        let module = vm
            .import_module_named("enum_meta_probe")
            .expect("enum meta probe should import successfully");
        let value = module
            .get_attr("RESULT")
            .expect("enum meta probe should publish RESULT");
        let tuple_ptr = value
            .as_object_ptr()
            .expect("RESULT should be stored as tuple object");
        let tuple = unsafe { &*(tuple_ptr as *const prism_runtime::types::tuple::TupleObject) };

        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.as_slice()[0].as_bool(), Some(true));
        assert_eq!(tuple.as_slice()[1].as_bool(), Some(true));
    }

    #[test]
    fn test_imported_source_module_can_use_str_replace() {
        let temp = TestTempDir::new();
        write_file(
            &temp.path.join("replace_probe.py"),
            concat!(
                "VALUE = (\n",
                "    'banana'.replace('na', 'NA'),\n",
                "    'banana'.replace('na', 'NA', 1),\n",
                "    'abc'.replace('', '-', 3),\n",
                ")\n",
            ),
        );

        let mut vm = VirtualMachine::new();
        vm.import_resolver
            .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

        let module = vm
            .import_module_named("replace_probe")
            .expect("replace probe should import successfully");
        let value = module
            .get_attr("VALUE")
            .expect("replace probe should publish VALUE");
        let tuple_ptr = value
            .as_object_ptr()
            .expect("replace probe value should be a tuple");
        let tuple = unsafe { &*(tuple_ptr as *const prism_runtime::types::tuple::TupleObject) };
        let string_entry = |index: usize| {
            let ptr = tuple
                .get(i64::try_from(index).expect("tuple index should fit into i64"))
                .and_then(|entry| entry.as_string_object_ptr())
                .expect("tuple entry should be an interned string")
                as *const u8;
            prism_core::intern::interned_by_ptr(ptr)
                .expect("tuple entry should resolve")
                .as_ref()
                .to_string()
        };

        assert_eq!(tuple.len(), 3);
        assert_eq!(string_entry(0), "baNANA");
        assert_eq!(string_entry(1), "baNAna");
        assert_eq!(string_entry(2), "-a-b-c");
    }

    #[test]
    fn test_builtins_available() {
        let vm = VirtualMachine::new();
        assert!(vm.builtins.get("None").is_some());
        assert!(vm.builtins.get("True").is_some());
        assert!(vm.builtins.get("False").is_some());
    }

    #[test]
    fn test_execution_step_limit_interrupts_infinite_loop() {
        let mut vm = VirtualMachine::new();
        vm.set_execution_step_limit(Some(128));

        let err = vm
            .execute_in_module(
                compile_module("while True:\n    pass\n", "<step-limit>"),
                Arc::new(ModuleObject::new("__main__")),
            )
            .expect_err("infinite loop should hit execution limit");

        assert!(
            err.to_string()
                .contains("execution step limit exceeded (128)")
        );
        assert_eq!(vm.executed_steps(), 128);
    }

    #[test]
    fn test_execution_step_limit_resets_between_top_level_runs() {
        let mut vm = VirtualMachine::new();
        vm.set_execution_step_limit(Some(64));

        let module = Arc::new(ModuleObject::new("__main__"));
        vm.execute_in_module(
            compile_module("value = 1\n", "<first-run>"),
            Arc::clone(&module),
        )
        .expect("first run should succeed");
        let first_steps = vm.executed_steps();
        assert!(first_steps > 0);

        vm.execute_in_module(
            compile_module("value = 1\n", "<second-run>"),
            Arc::new(ModuleObject::new("__main__")),
        )
        .expect("second run should succeed");

        assert_eq!(vm.executed_steps(), first_steps);
    }

    #[test]
    fn test_execution_step_limit_disables_jit_fast_path_for_bounded_runs() {
        let mut vm = VirtualMachine::with_jit();
        vm.set_execution_step_limit(Some(128));

        let err = vm
            .execute_in_module(
                compile_module("while True:\n    pass\n", "<bounded-jit>"),
                Arc::new(ModuleObject::new("__main__")),
            )
            .expect_err("bounded run should not bypass the step limit via JIT");

        assert!(
            err.to_string()
                .contains("execution step limit exceeded (128)")
        );
    }

    #[test]
    fn test_imported_builtins_module_shares_runtime_builtin_objects() {
        let mut vm = VirtualMachine::new();
        let builtins_module = vm
            .import_module_named("builtins")
            .expect("builtins module should import");

        let imported_open = builtins_module
            .get_attr("open")
            .and_then(|value| value.as_object_ptr())
            .expect("builtins.open should be callable");
        let runtime_open = vm
            .builtins
            .get("open")
            .and_then(|value| value.as_object_ptr())
            .expect("open should exist in the builtin registry");

        assert_eq!(imported_open, runtime_open);
    }

    #[test]
    fn test_execute_in_module_sets_function_identity_metadata_for_getattr() {
        let mut vm = VirtualMachine::new();
        let module = Arc::new(ModuleObject::new("pkg.mod"));

        vm.execute_in_module(
            compile_module("def f():\n    return 1\n", "<function-metadata>"),
            Arc::clone(&module),
        )
        .expect("module execution should succeed");

        let func = module.get_attr("f").expect("function should be exported");
        assert_eq!(
            builtin_getattr(&[func, Value::string(intern("__name__"))]).unwrap(),
            Value::string(intern("f"))
        );
        assert_eq!(
            builtin_getattr(&[func, Value::string(intern("__qualname__"))]).unwrap(),
            Value::string(intern("f"))
        );
        assert_eq!(
            builtin_getattr(&[func, Value::string(intern("__module__"))]).unwrap(),
            Value::string(intern("pkg.mod"))
        );
        assert!(
            builtin_getattr(&[func, Value::string(intern("__doc__"))])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_execute_in_module_sets_nested_function_name_and_module_for_getattr() {
        let mut vm = VirtualMachine::new();
        let module = Arc::new(ModuleObject::new("__main__"));

        vm.execute_in_module(
            compile_module(
                "def outer():\n    def inner():\n        return 1\n    return inner\ninner = outer()\n",
                "<nested-function-metadata>",
            ),
            Arc::clone(&module),
        )
        .expect("nested function module should execute");

        let inner = module
            .get_attr("inner")
            .expect("outer() should publish the nested function");
        assert_eq!(
            builtin_getattr(&[inner, Value::string(intern("__name__"))]).unwrap(),
            Value::string(intern("inner"))
        );
        assert_eq!(
            builtin_getattr(&[inner, Value::string(intern("__module__"))]).unwrap(),
            Value::string(intern("__main__"))
        );
    }

    #[test]
    fn test_import_error_to_runtime_preserves_module_not_found_metadata() {
        let err = VirtualMachine::import_error_to_runtime(ImportError::ModuleNotFound {
            module: Arc::from("pkg.missing"),
        });
        let mut vm = VirtualMachine::new();
        let type_id = vm.materialize_active_exception_from_runtime_error(&err);
        assert_eq!(
            type_id,
            crate::stdlib::exceptions::ExceptionTypeId::ModuleNotFoundError.as_u8() as u16
        );

        let active = *vm
            .get_active_exception()
            .expect("materialized import error should be active");
        let exc = unsafe {
            crate::builtins::ExceptionValue::from_value(active)
                .expect("active import error should be an ExceptionValue")
        };
        assert_eq!(exc.import_name(), Some("pkg.missing"));
        assert!(exc.import_path().is_none());
    }

    #[test]
    fn test_uncaught_exception_error_preserves_active_exception_message() {
        let mut vm = VirtualMachine::new();
        let exc = crate::builtins::create_exception(
            crate::stdlib::exceptions::ExceptionTypeId::TypeError,
            Some(Arc::from("real uncaught message")),
        );
        vm.set_active_exception_with_type(
            exc,
            crate::stdlib::exceptions::ExceptionTypeId::TypeError.as_u8() as u16,
        );

        let err = vm.uncaught_exception_error(
            crate::stdlib::exceptions::ExceptionTypeId::TypeError.as_u8() as u16,
        );

        match err.kind {
            crate::error::RuntimeErrorKind::Exception { type_id, message } => {
                assert_eq!(
                    type_id,
                    crate::stdlib::exceptions::ExceptionTypeId::TypeError.as_u8() as u16
                );
                assert_eq!(&*message, "real uncaught message");
            }
            other => panic!("expected exception runtime error, got {other:?}"),
        }
    }

    #[test]
    fn test_uncaught_exception_error_falls_back_without_active_exception() {
        let vm = VirtualMachine::new();
        let err = vm.uncaught_exception_error(
            crate::stdlib::exceptions::ExceptionTypeId::TypeError.as_u8() as u16,
        );

        match err.kind {
            crate::error::RuntimeErrorKind::Exception { type_id, message } => {
                assert_eq!(
                    type_id,
                    crate::stdlib::exceptions::ExceptionTypeId::TypeError.as_u8() as u16
                );
                assert_eq!(&*message, "Uncaught exception (type_id=52)");
            }
            other => panic!("expected exception runtime error, got {other:?}"),
        }
    }

    #[test]
    fn test_pop_exc_info_restores_exception_type_and_value() {
        let mut vm = VirtualMachine::new();
        vm.set_active_exception_with_type(Value::int(123).unwrap(), 24);
        assert!(vm.push_exc_info());

        vm.set_active_exception_with_type(Value::int(999).unwrap(), 5);
        assert!(vm.pop_exc_info());

        assert_eq!(vm.get_active_exception_type_id(), Some(24));
        assert_eq!(vm.get_active_exception().and_then(Value::as_int), Some(123));
        assert_eq!(vm.exception_state(), ExceptionState::Propagating);
    }

    #[test]
    fn test_pop_exc_info_restores_empty_state() {
        let mut vm = VirtualMachine::new();
        assert!(vm.push_exc_info());

        vm.set_active_exception_with_type(Value::int(1).unwrap(), 24);
        assert!(vm.pop_exc_info());

        assert!(vm.get_active_exception().is_none());
        assert_eq!(vm.get_active_exception_type_id(), None);
        assert_eq!(vm.exception_state(), ExceptionState::Normal);
    }

    #[test]
    fn test_pop_exc_info_empty_stack_noop() {
        let mut vm = VirtualMachine::new();
        vm.set_active_exception_with_type(Value::int(7).unwrap(), 24);

        assert!(!vm.pop_exc_info());
        assert_eq!(vm.get_active_exception_type_id(), Some(24));
        assert_eq!(vm.get_active_exception().and_then(Value::as_int), Some(7));
    }

    #[test]
    fn test_pop_frame_cleans_handlers_for_popped_frame() {
        let mut vm = VirtualMachine::new();
        let code = empty_code("f");

        vm.push_frame(Arc::clone(&code), 0).unwrap();
        assert!(vm.push_exception_handler(HandlerFrame::new(10, 0, 0)));

        vm.push_frame(code, 0).unwrap();
        assert!(vm.push_exception_handler(HandlerFrame::new(20, 0, 1)));
        assert_eq!(vm.handler_stack_depth(), 2);

        let popped = vm.pop_frame(Value::none()).unwrap();
        assert!(popped.is_none());
        assert_eq!(vm.call_depth(), 1);
        assert_eq!(vm.handler_stack_depth(), 1);

        let remaining = vm.pop_exception_handler().expect("missing root handler");
        assert_eq!(remaining.frame_id, 0);
        assert_eq!(remaining.handler_idx, 10);
    }

    #[test]
    fn test_pop_frame_cleans_handlers_for_last_frame() {
        let mut vm = VirtualMachine::new();
        let code = empty_code("root");
        vm.push_frame(code, 0).unwrap();
        assert!(vm.push_exception_handler(HandlerFrame::new(1, 0, 0)));

        let popped = vm.pop_frame(Value::none()).unwrap();
        assert!(popped.is_some());
        assert_eq!(vm.call_depth(), 0);
        assert_eq!(vm.handler_stack_depth(), 0);
    }

    #[test]
    fn test_propagate_exception_unwinds_and_cleans_generator_handlers() {
        let mut vm = VirtualMachine::new();
        let code = empty_code("g");

        vm.push_frame(Arc::clone(&code), 0).unwrap();
        assert!(vm.push_exception_handler(HandlerFrame::new(1, 0, 0)));
        vm.push_frame(code, 0).unwrap();
        assert!(vm.push_exception_handler(HandlerFrame::new(2, 0, 1)));

        let handled = vm.propagate_exception_within_generator_frames(24, 1);
        assert!(!handled);
        assert_eq!(vm.call_depth(), 1);
        assert_eq!(vm.handler_stack_depth(), 1);
        assert_eq!(vm.current_frame_id(), 0);
    }

    #[test]
    fn test_reset_clears_exception_and_handler_state() {
        let mut vm = VirtualMachine::new();
        let code = empty_code("r");
        vm.push_frame(code, 0).unwrap();
        assert!(vm.push_exception_handler(HandlerFrame::new(3, 0, 0)));
        vm.set_active_exception_with_type(Value::int(1).unwrap(), 24);
        assert!(vm.push_exc_info());

        vm.reset();
        assert_eq!(vm.call_depth(), 0);
        assert_eq!(vm.handler_stack_depth(), 0);
        assert_eq!(vm.get_active_exception_type_id(), None);
        assert!(!vm.has_exc_info());
        assert_eq!(vm.exception_state(), ExceptionState::Normal);
    }

    #[test]
    fn test_clear_frames_keeps_globals_but_clears_exception_and_handler_state() {
        let mut vm = VirtualMachine::new();
        vm.globals.set("x".into(), Value::int(42).unwrap());
        let code = empty_code("c");
        vm.push_frame(code, 0).unwrap();
        assert!(vm.push_exception_handler(HandlerFrame::new(4, 0, 0)));
        vm.set_active_exception_with_type(Value::int(9).unwrap(), 24);
        assert!(vm.push_exc_info());

        vm.clear_frames();
        assert_eq!(vm.call_depth(), 0);
        assert_eq!(vm.handler_stack_depth(), 0);
        assert_eq!(vm.get_active_exception_type_id(), None);
        assert!(!vm.has_exc_info());
        assert_eq!(vm.exception_state(), ExceptionState::Normal);
        assert_eq!(vm.globals.get("x").and_then(|v| v.as_int()), Some(42));
    }

    #[test]
    fn test_reset_recycles_all_frames_into_pool() {
        let mut vm = VirtualMachine::new();
        let code = empty_code("reset-pool");
        vm.push_frame(Arc::clone(&code), 0).unwrap();
        vm.push_frame(code, 0).unwrap();

        vm.reset();

        assert_eq!(vm.call_depth(), 0);
        assert_eq!(vm.pooled_frame_count(), 2);
    }

    #[test]
    fn test_clear_frames_reuses_clean_pooled_frame() {
        let mut vm = VirtualMachine::new();
        let code = empty_code("pool-clean");
        vm.push_frame(Arc::clone(&code), 0).unwrap();
        vm.current_frame_mut().set_reg(0, Value::int(123).unwrap());

        vm.clear_frames();
        assert_eq!(vm.pooled_frame_count(), 1);

        vm.push_frame(code, 0).unwrap();
        assert_eq!(vm.pooled_frame_count(), 0);
        assert!(vm.current_frame().get_reg(0).is_none());
        assert!(!vm.current_frame().reg_is_written(0));
    }

    #[test]
    fn test_exit_except_handler_restores_outer_context() {
        let mut vm = VirtualMachine::new();
        vm.push_frame(empty_code("outer"), 0).unwrap();

        vm.set_active_exception_with_type(Value::int(10).unwrap(), 24);
        assert!(vm.enter_except_handler());
        assert_eq!(vm.exception_state(), ExceptionState::Handling);

        vm.set_active_exception_with_type(Value::int(20).unwrap(), 25);
        assert!(vm.enter_except_handler());
        assert_eq!(vm.get_active_exception_type_id(), Some(25));

        assert!(vm.exit_except_handler());
        assert_eq!(vm.get_active_exception_type_id(), Some(24));
        assert_eq!(vm.get_active_exception().and_then(Value::as_int), Some(10));
        assert_eq!(vm.exception_state(), ExceptionState::Handling);

        assert!(vm.exit_except_handler());
        assert!(vm.get_active_exception().is_none());
        assert_eq!(vm.get_active_exception_type_id(), None);
        assert_eq!(vm.exception_state(), ExceptionState::Normal);
    }

    #[test]
    fn test_abort_except_handler_preserves_escaping_exception() {
        let mut vm = VirtualMachine::new();
        vm.push_frame(empty_code("outer"), 0).unwrap();

        vm.set_active_exception_with_type(Value::int(10).unwrap(), 24);
        assert!(vm.enter_except_handler());

        vm.set_active_exception_with_type(Value::int(99).unwrap(), 33);
        assert!(vm.abort_except_handler());

        assert_eq!(vm.get_active_exception_type_id(), Some(33));
        assert_eq!(vm.get_active_exception().and_then(Value::as_int), Some(99));
        assert_eq!(vm.exception_state(), ExceptionState::Propagating);
    }

    #[test]
    fn test_find_exception_handler_populates_cache_and_hits_fast_path() {
        let mut vm = VirtualMachine::new();
        let code = code_with_exception_entries(
            "eh",
            vec![ExceptionEntry {
                start_pc: 0,
                end_pc: 10,
                handler_pc: 77,
                finally_pc: u32::MAX,
                depth: 0,
                exception_type_idx: u16::MAX,
            }],
        );

        vm.push_frame(code, 0).unwrap();
        vm.current_frame_mut().ip = 5;

        assert_eq!(vm.find_exception_handler(24), Some(77));
        assert!(vm.current_frame().handler_cache.is_valid());
        assert_eq!(vm.current_frame().handler_cache.cached_handler(), Some(0));

        assert_eq!(vm.find_exception_handler(24), Some(77));
        assert!(vm.current_frame().handler_cache.hit_count() >= 1);
    }

    #[test]
    fn test_find_exception_handler_returns_none_with_empty_frame_stack() {
        let mut vm = VirtualMachine::new();

        assert_eq!(vm.find_exception_handler(24), None);
    }

    #[test]
    fn test_find_exception_handler_records_cache_miss() {
        let mut vm = VirtualMachine::new();
        let code = code_with_exception_entries("eh_miss", vec![catch_all_entry(10, 20, 99)]);

        vm.push_frame(code, 0).unwrap();
        vm.current_frame_mut().ip = 3;

        assert_eq!(vm.find_exception_handler(24), None);
        assert!(
            vm.current_frame().handler_cache.is_empty()
                || vm.current_frame().handler_cache.cached_handler().is_none()
        );
        assert_eq!(vm.current_frame().handler_cache.cached_pc(), Some(2));
    }

    #[test]
    fn test_find_exception_handler_prefers_most_specific_range_when_unsorted() {
        let mut vm = VirtualMachine::new();
        let code = code_with_exception_entries(
            "eh_nested",
            vec![
                catch_all_entry(0, 20, 100),
                catch_all_entry(5, 10, 200),
                catch_all_entry(0, 12, 150),
            ],
        );

        vm.push_frame(code, 0).unwrap();
        vm.current_frame_mut().ip = 8;

        assert_eq!(vm.find_exception_handler(24), Some(200));
        assert_eq!(vm.current_frame().handler_cache.cached_handler(), Some(1));
    }

    #[test]
    fn test_handler_cache_isolation_on_push_frame_switch() {
        let mut vm = VirtualMachine::new();
        let frame_a = code_with_exception_entries(
            "frame_a",
            vec![catch_all_entry(0, 4, 10), catch_all_entry(0, 10, 11)],
        );
        let frame_b = code_with_exception_entries(
            "frame_b",
            vec![catch_all_entry(0, 10, 21), catch_all_entry(0, 10, 22)],
        );

        vm.push_frame(frame_a, 0).unwrap();
        vm.current_frame_mut().ip = 5;
        assert_eq!(vm.find_exception_handler(24), Some(11));
        assert_eq!(vm.current_frame().handler_cache.cached_handler(), Some(1));

        vm.push_frame(frame_b, 0).unwrap();
        assert!(vm.current_frame().handler_cache.is_empty());
        vm.current_frame_mut().ip = 5;
        assert_eq!(vm.find_exception_handler(24), Some(21));
    }

    #[test]
    fn test_handler_cache_persists_per_frame_across_pop_switch() {
        let mut vm = VirtualMachine::new();
        let caller = code_with_exception_entries(
            "caller",
            vec![catch_all_entry(0, 10, 31), catch_all_entry(0, 10, 32)],
        );
        let callee = code_with_exception_entries(
            "callee",
            vec![catch_all_entry(0, 4, 40), catch_all_entry(0, 10, 41)],
        );

        vm.push_frame(caller, 0).unwrap();
        vm.current_frame_mut().ip = 5;
        assert_eq!(vm.find_exception_handler(24), Some(31));

        vm.push_frame(callee, 0).unwrap();
        vm.current_frame_mut().ip = 5;
        assert_eq!(vm.find_exception_handler(24), Some(41));
        assert_eq!(vm.current_frame().handler_cache.cached_handler(), Some(1));

        let popped = vm.pop_frame(Value::none()).unwrap();
        assert!(popped.is_none());
        assert_eq!(vm.current_frame().handler_cache.cached_handler(), Some(0));
        vm.current_frame_mut().ip = 5;
        assert_eq!(vm.find_exception_handler(24), Some(31));
        assert!(vm.current_frame().handler_cache.hit_count() >= 1);
    }

    #[test]
    fn test_import_module_named_executes_frozen_module() {
        let mut vm = VirtualMachine::new();
        vm.import_resolver.insert_frozen_module(
            "helper",
            FrozenModuleSource::new(
                compile_module("VALUE = 123\n", "<frozen:helper>"),
                "<frozen:helper>",
                "",
                false,
            ),
        );

        let helper = vm
            .import_module_named("helper")
            .expect("frozen helper should import");
        assert_eq!(
            helper.get_attr("VALUE").and_then(|value| value.as_int()),
            Some(123)
        );
        assert!(vm.import_resolver.get_cached("helper").is_some());
    }

    #[test]
    fn test_execute_in_module_supports_relative_imports_from_frozen_package() {
        let mut vm = VirtualMachine::new();
        vm.import_resolver.insert_frozen_module(
            "pkg",
            FrozenModuleSource::new(
                compile_module("PACKAGE = True\n", "<frozen:pkg.__init__>"),
                "<frozen:pkg.__init__>",
                "pkg",
                true,
            ),
        );
        vm.import_resolver.insert_frozen_module(
            "pkg.helper",
            FrozenModuleSource::new(
                compile_module("VALUE = 7\n", "<frozen:pkg.helper>"),
                "<frozen:pkg.helper>",
                "pkg",
                false,
            ),
        );

        let main_code = compile_module(
            "from .helper import VALUE\nRESULT = VALUE + 1\n",
            "<frozen:pkg.__main__>",
        );
        let main_module = Arc::new(ModuleObject::with_metadata(
            "__main__",
            None,
            Some(Arc::from("<frozen:pkg.__main__>")),
            Some(Arc::from("pkg")),
        ));

        vm.execute_in_module(main_code, Arc::clone(&main_module))
            .expect("frozen package entry should execute");

        assert_eq!(
            main_module
                .get_attr("RESULT")
                .and_then(|value| value.as_int()),
            Some(8)
        );
        assert!(vm.import_resolver.get_cached("pkg").is_some());
        assert!(vm.import_resolver.get_cached("pkg.helper").is_some());
    }
}
