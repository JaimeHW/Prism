//! Virtual machine implementation.
//!
//! The VirtualMachine is the main execution engine for Prism bytecode.
//! It manages frames, globals, builtins, and the dispatch loop.

use crate::allocator::GcAllocator;
use crate::builtins::{BuiltinRegistry, set_exception_traceback_for_value};
use crate::dispatch::{ControlFlow, get_handler};
use crate::error::{RuntimeError, RuntimeErrorKind, TracebackEntry, VmResult};
use crate::exception::{ExcInfoStack, ExceptionState, HandlerStack};
use crate::frame::{ClosureEnv, Frame, FramePool, MAX_RECURSION_DEPTH};
use crate::gc_integration::ManagedHeap;
use crate::globals::GlobalScope;
use crate::ic_manager::ICManager;
use crate::import::{
    FrozenModuleSource, ImportError, ImportLoadPlan, ImportResolver, ModuleExportError,
    ModuleObject, resolve_relative_import,
};
use crate::inline_cache::InlineCacheStore;
use crate::jit_context::{JitConfig, JitContext};
use crate::jit_executor::ExecutionResult;
use crate::ops::calls::{
    capture_closure_environment, initialize_closure_cellvars_from_locals, invoke_callable_value,
};
use crate::ops::objects::list_storage_ref_from_ptr;
use crate::profiler::{CodeId, Profiler, TierUpDecision};
use crate::speculative::SpeculationCache;
use crate::stdlib::_codecs::{SharedCodecRegistry, new_shared_codec_registry};
use crate::stdlib::_thread::{SharedThreadGroup, new_thread_group};
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::stdlib::generators::{
    GeneratorObject, GeneratorState as RuntimeGeneratorState, LivenessMap,
};
use prism_code::{CodeObject, Instruction, LineTableEntry, Opcode};
use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_core::intern::intern;
use prism_core::{PrismResult, Value};
use prism_runtime::allocation_context::{RuntimeHeapBinding, alloc_value_in_current_heap_or_box};
use prism_runtime::object::class::ClassDict;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::unregister_global_class;
use prism_runtime::object::views::{FrameViewObject, TracebackViewObject};
use prism_runtime::types::dict::DictObject;

mod generator;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub(crate) type SharedManagedHeap = Arc<Mutex<ManagedHeap>>;
pub(crate) type SharedClassPublications = Arc<PublishedClassScope>;

#[derive(Debug, Default)]
pub(crate) struct PublishedClassScope {
    class_ids: Mutex<HashSet<ClassId>>,
}

impl PublishedClassScope {
    #[inline]
    fn record(&self, class_id: ClassId) {
        self.class_ids
            .lock()
            .expect("published class scope lock poisoned")
            .insert(class_id);
    }
}

impl Drop for PublishedClassScope {
    fn drop(&mut self) {
        let class_ids = std::mem::take(
            self.class_ids
                .get_mut()
                .expect("published class scope lock poisoned"),
        );
        if class_ids.is_empty() {
            return;
        }

        crate::ops::method_dispatch::method_cache().clear();
        crate::stdlib::_abc::clear_abc_state_for_class_ids(class_ids.iter().copied());
        for class_id in class_ids {
            unregister_global_class(class_id);
        }
    }
}

fn new_shared_managed_heap() -> SharedManagedHeap {
    Arc::new(Mutex::new(ManagedHeap::with_defaults()))
}

fn new_shared_class_publications() -> SharedClassPublications {
    Arc::new(PublishedClassScope::default())
}

fn bind_runtime_heap(heap: &SharedManagedHeap) -> RuntimeHeapBinding {
    let guard = heap.lock().expect("managed heap lock poisoned");
    RuntimeHeapBinding::register(guard.heap())
}

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

#[inline]
fn module_value(module: &Arc<ModuleObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(module) as *const ())
}

fn module_list_attr_is_populated(module: &Arc<ModuleObject>, attr: &str) -> bool {
    module
        .get_attr(attr)
        .and_then(|value| value.as_object_ptr())
        .and_then(list_storage_ref_from_ptr)
        .is_some_and(|list| !list.is_empty())
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryHandlerPolicy {
    IncludeBoundary,
    ExcludeBoundary,
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
    /// Code objects that have passed bytecode validation for this VM.
    ///
    /// The cache holds an `Arc` to each validated object so pointer identities
    /// cannot be reused for different bytecode during the VM lifetime. This
    /// keeps validation off the hot call path after the first entry while still
    /// preserving the safety contract required by unchecked opcode handlers.
    validated_code_objects: HashMap<usize, Arc<CodeObject>>,
    /// Per-interpreter target used by worker threads to route `_thread.interrupt_main()`.
    thread_interrupt_target: u64,
    /// Native Python threads owned by this interpreter.
    thread_group: SharedThreadGroup,
    /// Whether this VM is responsible for joining its thread group during teardown.
    join_threads_on_drop: bool,
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
    /// Interpreter-local codec registry shared by native Python threads.
    codec_registry: SharedCodecRegistry,
    /// Heap-type publications whose class dictionaries may reference this VM heap.
    ///
    /// Process-global type/class registries are fast, but Python-defined class
    /// dictionaries can hold raw `Value` pointers into a VM heap. This scope is
    /// shared by native Python worker VMs and unregisters those heap classes
    /// before the owning heap can disappear.
    class_publications: SharedClassPublications,

    // =========================================================================
    // GC Integration
    // =========================================================================
    /// Thread-local binding that exposes this heap to runtime helpers.
    ///
    /// This must drop before `heap` so runtime helper teardown cannot observe a
    /// stale heap binding.
    _runtime_heap_binding: RuntimeHeapBinding,
    /// GC-managed heap for object allocation.
    ///
    /// Python threads spawned from a VM share this heap so objects allocated by a
    /// worker can safely escape into shared Python containers and module globals.
    /// The mutex protects collection/root bookkeeping. Python execution is not
    /// serialized by a global interpreter lock; shared runtime structures own
    /// their own synchronization.
    ///
    /// Keep this as the final field. Many VM fields contain raw `Value` pointers
    /// into the heap, and Rust drops fields in declaration order.
    heap: SharedManagedHeap,
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
        } else if let Some(module) = self.import_resolver.module_from_ptr(ptr) {
            Some(module)
        } else {
            let module = unsafe { Arc::from_raw(ptr as *const ModuleObject) };
            let cloned = Arc::clone(&module);
            std::mem::forget(module);
            self.import_resolver
                .insert_module(cloned.name(), Arc::clone(&cloned));
            Some(cloned)
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

    pub fn import_star_into_current_scope(
        &mut self,
        module: &ModuleObject,
    ) -> Result<(), RuntimeError> {
        let attrs = module
            .public_attrs()
            .map_err(module_export_error_to_runtime_error)?;

        if let Some(target_module) = self.current_module_cloned() {
            let is_main = target_module.name() == "__main__";
            for (name, value) in attrs {
                target_module.set_attr(&name, value);
                if is_main {
                    self.globals.set(Arc::from(name.as_ref()), value);
                }
            }
        } else {
            for (name, value) in attrs {
                self.globals.set(Arc::from(name.as_ref()), value);
            }
        }
        Ok(())
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
        self.execute_in_module_runtime(code, module)
            .map_err(Into::into)
    }

    pub fn execute_in_module_runtime(
        &mut self,
        code: Arc<CodeObject>,
        module: Arc<ModuleObject>,
    ) -> VmResult<Value> {
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

    pub fn run_shutdown_hooks(&mut self) -> Vec<RuntimeError> {
        self.execution_budget.reset_counter();

        let baseline_depth = self.frames.len();
        if baseline_depth == 0 {
            let code = Arc::new(CodeObject::new("<shutdown>", "<shutdown>"));
            if let Err(err) = self.push_frame_with_module(code, 0, self.current_module_cloned()) {
                return vec![err];
            }
        }

        let exception_context = self.capture_exception_context();
        self.clear_exception_context_for_shutdown();

        let mut errors = Vec::new();
        self.run_threading_shutdown(&mut errors);
        if let Err(err) = crate::stdlib::atexit::run_exitfuncs(self) {
            errors.push(RuntimeError::from(err));
        }

        while self.frames.len() > baseline_depth {
            self.pop_top_frame_for_unwind();
        }

        self.restore_exception_context(exception_context);

        errors
    }

    fn clear_exception_context_for_shutdown(&mut self) {
        self.exc_state = ExceptionState::Normal;
        self.handler_stack.clear();
        self.active_exception = None;
        self.active_exception_type_id = None;
        self.exc_info_stack.clear();
        self.active_except_handlers.clear();
    }

    fn run_threading_shutdown(&mut self, errors: &mut Vec<RuntimeError>) {
        let Some(threading) = self.import_resolver.get_cached("threading") else {
            return;
        };
        let Some(shutdown) = threading.get_attr("_shutdown") else {
            return;
        };

        if let Err(err) = invoke_callable_value(self, shutdown, &[]) {
            errors.push(err);
        }
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
            | ImportError::StarImportError { module, .. }
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
        compile_source_code(source, filename, self.compiler_optimization)
            .map_err(|err| RuntimeError::import_error(module_name, Arc::from(err.to_string())))
    }

    fn run_nested_module_until_depth(&mut self, stop_depth: usize) -> VmResult<()> {
        let _execution_region = crate::threading_runtime::enter_execution_region();
        loop {
            if self.frames.len() <= stop_depth {
                return Ok(());
            }

            self.execution_budget.consume_step()?;
            crate::threading_runtime::checkpoint();
            self.poll_pending_main_interrupt_to_depth(stop_depth)?;

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
                    self.propagate_exception_to_depth(stop_depth, type_id)?;
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

                    self.propagate_reraise_to_depth(stop_depth, type_id)?;
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
                    if err.is_control_transferred() {
                        continue;
                    }
                    self.propagate_runtime_error_to_depth(stop_depth, err)?;
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
        self.propagate_exception_to_depth_with_policy(
            stop_depth,
            type_id,
            BoundaryHandlerPolicy::ExcludeBoundary,
        )
    }

    #[inline]
    fn route_nested_reraise(&mut self, stop_depth: usize, type_id: u16) -> VmResult<()> {
        self.propagate_reraise_to_depth_with_policy(
            stop_depth,
            type_id,
            BoundaryHandlerPolicy::ExcludeBoundary,
        )
    }

    #[inline]
    fn route_nested_runtime_error(&mut self, stop_depth: usize, err: RuntimeError) -> VmResult<()> {
        if err.is_control_transferred() {
            return Ok(());
        }
        self.propagate_runtime_error_to_depth_with_policy(
            stop_depth,
            err,
            BoundaryHandlerPolicy::ExcludeBoundary,
        )
    }

    #[inline]
    fn poll_pending_main_interrupt_to_depth(&mut self, min_depth: usize) -> VmResult<()> {
        self.poll_pending_async_exception_to_depth(min_depth)?;

        let Some(signum) =
            crate::stdlib::_thread::take_pending_main_interrupt(self.thread_interrupt_target)
        else {
            return Ok(());
        };

        if let Err(err) = crate::stdlib::_thread::deliver_interrupt_signal(self, signum) {
            self.propagate_runtime_error_to_depth(min_depth, err)?;
        }
        Ok(())
    }

    #[inline]
    fn poll_pending_async_exception_to_depth(&mut self, min_depth: usize) -> VmResult<()> {
        let Some(exception) =
            crate::stdlib::_thread::take_pending_async_exception_for_current_thread()
        else {
            return Ok(());
        };

        let (exception, type_id) = self.normalize_async_exception(exception)?;
        self.set_active_exception_with_type(exception, type_id);
        self.propagate_exception_to_depth(min_depth, type_id)
    }

    fn normalize_async_exception(&mut self, exception: Value) -> VmResult<(Value, u16)> {
        use crate::ops::exception::helpers::{
            extract_type_id_from_value, is_exception_class_value, is_exception_instance_value,
        };

        let normalized = if is_exception_instance_value(&exception) {
            exception
        } else if is_exception_class_value(&exception) {
            invoke_callable_value(self, exception, &[])?
        } else {
            return Err(RuntimeError::type_error(
                "async exception must derive from BaseException",
            ));
        };

        if !is_exception_instance_value(&normalized) {
            return Err(RuntimeError::type_error(
                "async exception construction must yield an exception instance",
            ));
        }

        Ok((normalized, extract_type_id_from_value(&normalized)))
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

    pub(crate) fn execute_until_target_frame_returns(
        &mut self,
        stop_depth: usize,
        target_frame_id: u32,
    ) -> VmResult<Value> {
        match self.execute_until_target_frame_returns_with_outcome(stop_depth, target_frame_id)? {
            NestedTargetFrameOutcome::Returned(value) => Ok(value),
            NestedTargetFrameOutcome::ControlTransferred => {
                Err(RuntimeError::control_transferred())
            }
        }
    }

    pub(crate) fn execute_until_target_frame_returns_with_outcome(
        &mut self,
        stop_depth: usize,
        target_frame_id: u32,
    ) -> VmResult<NestedTargetFrameOutcome> {
        let _execution_region = crate::threading_runtime::enter_execution_region();
        if stop_depth == 0 {
            return Err(RuntimeError::internal(
                "nested execution requires a caller frame",
            ));
        }

        loop {
            crate::threading_runtime::checkpoint();
            self.poll_pending_main_interrupt_to_depth(stop_depth)?;

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
        let _execution_region = crate::threading_runtime::enter_execution_region();
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
            crate::threading_runtime::checkpoint();
            self.poll_pending_main_interrupt_to_depth(stop_depth)?;

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

    fn load_source_module_from_location(
        &mut self,
        name: &str,
        location: crate::import::SourceModuleLocation,
    ) -> VmResult<Arc<ModuleObject>> {
        if let Some(module) = self.import_resolver.get_cached(name) {
            return Ok(module);
        }

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

    fn load_frozen_module_source(
        &mut self,
        name: &str,
        frozen: Arc<FrozenModuleSource>,
    ) -> VmResult<Arc<ModuleObject>> {
        if let Some(module) = self.import_resolver.get_cached(name) {
            return Ok(module);
        }

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

        if name == "importlib" {
            self.install_importlib_importers(&module)?;
        }

        Ok(module)
    }

    fn install_importlib_importers(&mut self, importlib: &Arc<ModuleObject>) -> VmResult<()> {
        let sys_module = self.import_module_with_context("sys", None)?;
        if module_list_attr_is_populated(&sys_module, "meta_path") {
            return Ok(());
        }

        let imp_module = self.import_module_with_context("_imp", None)?;
        let Some(bootstrap) = self.module_attr_as_imported_module(importlib, "_bootstrap") else {
            return Ok(());
        };
        let Some(bootstrap_external) =
            self.module_attr_as_imported_module(importlib, "_bootstrap_external")
        else {
            return Ok(());
        };

        let install_core = bootstrap
            .get_attr("_install")
            .ok_or_else(|| RuntimeError::attribute_error("module", "_install"))?;
        invoke_callable_value(
            self,
            install_core,
            &[module_value(&sys_module), module_value(&imp_module)],
        )?;

        let install_external = bootstrap_external
            .get_attr("_install")
            .ok_or_else(|| RuntimeError::attribute_error("module", "_install"))?;
        invoke_callable_value(self, install_external, &[module_value(&bootstrap)])?;

        Ok(())
    }

    fn module_attr_as_imported_module(
        &self,
        module: &Arc<ModuleObject>,
        attr: &str,
    ) -> Option<Arc<ModuleObject>> {
        let value = module.get_attr(attr)?;
        let ptr = value.as_object_ptr()?;
        self.import_resolver.module_from_ptr(ptr)
    }

    fn load_module_from_plan(
        &mut self,
        name: &str,
        plan: ImportLoadPlan,
    ) -> VmResult<Arc<ModuleObject>> {
        match plan {
            ImportLoadPlan::Cached(module) => Ok(module),
            ImportLoadPlan::Frozen(frozen) => self.load_frozen_module_source(name, frozen),
            ImportLoadPlan::Source(location) => {
                self.load_source_module_from_location(name, location)
            }
            ImportLoadPlan::Native => self
                .import_resolver
                .import_module(name)
                .map_err(Self::import_error_to_runtime),
            ImportLoadPlan::Missing => {
                Err(Self::import_error_to_runtime(ImportError::ModuleNotFound {
                    module: Arc::from(name),
                }))
            }
        }
    }

    pub(crate) fn import_module_with_context(
        &mut self,
        raw_name: &str,
        current_module: Option<&Arc<ModuleObject>>,
    ) -> VmResult<Arc<ModuleObject>> {
        let absolute_name =
            self.resolve_import_name_with_context(raw_name, current_module.map(Arc::as_ref))?;

        if !absolute_name.contains('.') {
            let plan = self.import_resolver.resolve_load_plan(&absolute_name);
            return self.load_module_from_plan(&absolute_name, plan);
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

            let plan = self.import_resolver.resolve_load_plan(&prefix);
            let next = match plan {
                ImportLoadPlan::Cached(module) => module,
                ImportLoadPlan::Missing => {
                    if let Some(value) = current.get_attr(segment)
                        && let Some(module_ptr) = value.as_object_ptr()
                        && let Some(module) = self.import_resolver.module_from_ptr(module_ptr)
                    {
                        self.import_resolver
                            .insert_module(&prefix, Arc::clone(&module));
                        module
                    } else {
                        self.load_module_from_plan(&prefix, ImportLoadPlan::Missing)?
                    }
                }
                plan => self.load_module_from_plan(&prefix, plan)?,
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

    pub(crate) fn import_from_module(
        &mut self,
        module: &Arc<ModuleObject>,
        name: &str,
    ) -> VmResult<Value> {
        let current_module = self.current_module_cloned();
        self.import_from_module_with_context(module, name, current_module.as_ref())
    }

    pub(crate) fn import_from_with_context(
        &mut self,
        module_spec: &str,
        name: &str,
        current_module: Option<&Arc<ModuleObject>>,
    ) -> VmResult<Value> {
        let module = self.import_module_with_context(module_spec, current_module)?;
        self.import_from_module_with_context(&module, name, current_module)
    }

    fn import_from_module_with_context(
        &mut self,
        module: &Arc<ModuleObject>,
        name: &str,
        current_module: Option<&Arc<ModuleObject>>,
    ) -> VmResult<Value> {
        if let Ok(value) = self.import_resolver.import_from(module, name) {
            return Ok(value);
        }

        let submodule_name = format!("{}.{}", module.name(), name);
        match self.import_module_with_context(&submodule_name, current_module) {
            Ok(submodule) => {
                let value = Value::object_ptr(Arc::as_ptr(&submodule) as *const ());
                module.set_attr(name, value);
                Ok(value)
            }
            Err(err) if Self::missing_submodule_matches(&err, &submodule_name) => Err(
                Self::import_error_to_runtime(ImportError::ImportFromError {
                    module: Arc::from(module.name()),
                    name: Arc::from(name),
                }),
            ),
            Err(err) => Err(err),
        }
    }

    fn missing_submodule_matches(err: &RuntimeError, expected: &str) -> bool {
        matches!(
            &err.kind,
            RuntimeErrorKind::ImportError {
                name: Some(name),
                missing: true,
                ..
            } if name.as_ref() == expected
        )
    }

    /// Create a new virtual machine (interpreter only, no JIT).
    pub fn new() -> Self {
        let thread_interrupt_target = crate::stdlib::_thread::new_main_interrupt_target();
        let thread_group = new_thread_group();
        let heap = new_shared_managed_heap();
        let class_publications = new_shared_class_publications();
        let codec_registry = new_shared_codec_registry();
        let runtime_heap_binding = bind_runtime_heap(&heap);
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
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
            validated_code_objects: HashMap::new(),
            thread_interrupt_target,
            thread_group,
            join_threads_on_drop: true,
            class_publications,
            codec_registry,
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

    /// Create an interpreter VM that allocates into an existing Python heap.
    ///
    /// Native Python threads execute with their own VM stacks and execution
    /// caches, but they must share the spawning VM's heap because Python
    /// objects can freely escape across thread boundaries.
    pub(crate) fn with_shared_heap(heap: SharedManagedHeap, thread_interrupt_target: u64) -> Self {
        let runtime_heap_binding = bind_runtime_heap(&heap);
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
        drop(runtime_heap_binding);
        Self::with_shared_heap_and_import_resolver(
            heap,
            thread_interrupt_target,
            builtins,
            import_resolver,
            new_shared_class_publications(),
            new_shared_codec_registry(),
            new_thread_group(),
            true,
        )
    }

    /// Create an interpreter VM for a native Python thread.
    ///
    /// The thread gets an independent frame stack and inline cache state while
    /// sharing interpreter-wide runtime state such as `sys.modules`, `sys.path`,
    /// and the import in-progress table with the spawning interpreter.
    pub(crate) fn with_shared_heap_and_import_resolver(
        heap: SharedManagedHeap,
        thread_interrupt_target: u64,
        builtins: BuiltinRegistry,
        import_resolver: ImportResolver,
        class_publications: SharedClassPublications,
        codec_registry: SharedCodecRegistry,
        thread_group: SharedThreadGroup,
        join_threads_on_drop: bool,
    ) -> Self {
        let runtime_heap_binding = bind_runtime_heap(&heap);
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
            validated_code_objects: HashMap::new(),
            thread_interrupt_target,
            thread_group,
            join_threads_on_drop,
            class_publications,
            codec_registry,
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

    /// Return the shared heap handle used by native Python threads.
    pub(crate) fn shared_heap(&self) -> SharedManagedHeap {
        Arc::clone(&self.heap)
    }

    pub(crate) fn class_publications(&self) -> SharedClassPublications {
        Arc::clone(&self.class_publications)
    }

    pub(crate) fn thread_group(&self) -> SharedThreadGroup {
        Arc::clone(&self.thread_group)
    }

    pub(crate) fn join_owned_threads(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        self.thread_group.join_all_finished_before(deadline)
    }

    pub(crate) fn shared_codec_registry(&self) -> SharedCodecRegistry {
        Arc::clone(&self.codec_registry)
    }

    #[inline]
    pub(crate) fn codec_registry(&self) -> &SharedCodecRegistry {
        &self.codec_registry
    }

    #[inline]
    pub(crate) fn record_published_class(&self, class_id: ClassId) {
        self.class_publications.record(class_id);
    }

    pub(crate) fn thread_interrupt_target(&self) -> u64 {
        self.thread_interrupt_target
    }

    /// Create a new virtual machine with JIT compilation enabled.
    pub fn with_jit() -> Self {
        let thread_interrupt_target = crate::stdlib::_thread::new_main_interrupt_target();
        let thread_group = new_thread_group();
        let heap = new_shared_managed_heap();
        let class_publications = new_shared_class_publications();
        let codec_registry = new_shared_codec_registry();
        let runtime_heap_binding = bind_runtime_heap(&heap);
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
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
            validated_code_objects: HashMap::new(),
            thread_interrupt_target,
            thread_group,
            join_threads_on_drop: true,
            class_publications,
            codec_registry,
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
        let thread_interrupt_target = crate::stdlib::_thread::new_main_interrupt_target();
        let thread_group = new_thread_group();
        let class_publications = new_shared_class_publications();
        let codec_registry = new_shared_codec_registry();
        let jit = if config.enabled {
            Some(JitContext::new(config))
        } else {
            None
        };
        let heap = new_shared_managed_heap();
        let runtime_heap_binding = bind_runtime_heap(&heap);
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
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
            validated_code_objects: HashMap::new(),
            thread_interrupt_target,
            thread_group,
            join_threads_on_drop: true,
            class_publications,
            codec_registry,
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
        let thread_interrupt_target = crate::stdlib::_thread::new_main_interrupt_target();
        let thread_group = new_thread_group();
        let heap = new_shared_managed_heap();
        let class_publications = new_shared_class_publications();
        let codec_registry = new_shared_codec_registry();
        let runtime_heap_binding = bind_runtime_heap(&heap);
        let (builtins, import_resolver) = standard_runtime_builtins_and_import_resolver(None);
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
            validated_code_objects: HashMap::new(),
            thread_interrupt_target,
            thread_group,
            join_threads_on_drop: true,
            class_publications,
            codec_registry,
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
        self.execute_runtime(code).map_err(Into::into)
    }

    /// Execute a code object while preserving the original `RuntimeError`.
    pub fn execute_runtime(&mut self, code: Arc<CodeObject>) -> VmResult<Value> {
        self.execute_in_module_runtime(code, Arc::new(ModuleObject::new("__main__")))
    }

    /// Main dispatch loop.
    #[inline(never)] // Prevent inlining for better branch prediction
    fn run_loop(&mut self) -> VmResult<Value> {
        let _execution_region = crate::threading_runtime::enter_execution_region();
        loop {
            self.execution_budget.consume_step()?;
            crate::threading_runtime::checkpoint();
            self.poll_pending_main_interrupt_to_depth(1)?;

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

                    self.propagate_exception_to_depth(1, type_id)?;
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
                        return Err(RuntimeError::type_error("No active exception to re-raise"));
                    };

                    if type_id == 0 {
                        return Err(RuntimeError::internal(
                            "Reraise without active exception type",
                        ));
                    }

                    self.propagate_reraise_to_depth(1, type_id)?;
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
                    if err.is_control_transferred() {
                        continue;
                    }
                    self.propagate_runtime_error_to_depth(1, err)?;
                }
            }
        }
    }

    #[inline]
    fn current_traceback_entry(&self) -> Option<TracebackEntry> {
        let frame = self.frames.get(self.current_frame_idx)?;
        let pc = frame.ip.saturating_sub(1);
        let line = Self::traceback_line_for_pc(&frame.code, pc);
        Some(TracebackEntry {
            func_name: Arc::clone(&frame.code.name),
            filename: Arc::clone(&frame.code.filename),
            line,
        })
    }

    fn traceback_line_for_pc(code: &CodeObject, pc: u32) -> u32 {
        code.line_for_pc(pc)
            .or_else(|| {
                code.line_table
                    .iter()
                    .rev()
                    .find(|entry| entry.start_pc <= pc)
                    .map(|entry| entry.line)
            })
            .unwrap_or(code.first_lineno)
    }

    #[inline]
    fn prepend_traceback_entry(traceback: &mut Vec<TracebackEntry>, entry: TracebackEntry) {
        let duplicate = traceback.first().is_some_and(|existing| {
            existing.line == entry.line
                && existing.func_name.as_ref() == entry.func_name.as_ref()
                && existing.filename.as_ref() == entry.filename.as_ref()
        });
        if !duplicate {
            traceback.insert(0, entry);
        }
    }

    #[inline]
    fn prepend_current_traceback_entry(&self, traceback: &mut Vec<TracebackEntry>) {
        if let Some(entry) = self.current_traceback_entry() {
            Self::prepend_traceback_entry(traceback, entry);
        }
    }

    #[inline]
    fn prepend_current_traceback_to_error(&self, err: &mut RuntimeError) {
        self.prepend_current_traceback_entry(&mut err.traceback);
    }

    fn alloc_traceback_view<T>(&self, object: T) -> Value
    where
        T: prism_runtime::Trace + 'static,
    {
        alloc_value_in_current_heap_or_box(object)
    }

    fn synthetic_traceback_code(entry: &TracebackEntry) -> Arc<CodeObject> {
        let line = entry.line.max(1);
        let mut code = CodeObject::new(Arc::clone(&entry.func_name), Arc::clone(&entry.filename));
        code.first_lineno = line;
        code.instructions = vec![Instruction::op(Opcode::Nop)].into_boxed_slice();
        code.line_table = vec![LineTableEntry {
            start_pc: 0,
            end_pc: 1,
            line,
        }]
        .into_boxed_slice();
        Arc::new(code)
    }

    fn python_traceback_from_entries(&self, entries: &[TracebackEntry]) -> VmResult<Value> {
        if entries.is_empty() {
            return Ok(Value::none());
        }

        let mut frames = Vec::with_capacity(entries.len());
        let mut back = None;
        for entry in entries {
            let globals = self.alloc_traceback_view(DictObject::new());
            let locals = self.alloc_traceback_view(DictObject::new());
            let frame = self.alloc_traceback_view(FrameViewObject::new(
                Some(Self::synthetic_traceback_code(entry)),
                globals,
                locals,
                entry.line,
                0,
                back,
            ));
            frames.push(frame);
            back = Some(frame);
        }

        let mut next = None;
        for (entry, frame) in entries.iter().zip(frames.iter()).rev() {
            let traceback =
                self.alloc_traceback_view(TracebackViewObject::new(*frame, next, entry.line, 0));
            next = Some(traceback);
        }

        Ok(next.unwrap_or_else(Value::none))
    }

    fn attach_active_python_traceback(&self, entries: &[TracebackEntry]) -> VmResult<()> {
        let Some(exception) = self.active_exception else {
            return Ok(());
        };
        let traceback = self.python_traceback_from_entries(entries)?;
        set_exception_traceback_for_value(exception, traceback)
            .map_err(RuntimeError::type_error)?;
        Ok(())
    }

    #[inline]
    fn propagate_exception_to_depth(&mut self, min_depth: usize, type_id: u16) -> VmResult<()> {
        self.propagate_exception_to_depth_with_policy(
            min_depth,
            type_id,
            BoundaryHandlerPolicy::IncludeBoundary,
        )
    }

    #[inline]
    fn propagate_exception_to_depth_with_policy(
        &mut self,
        min_depth: usize,
        type_id: u16,
        boundary_policy: BoundaryHandlerPolicy,
    ) -> VmResult<()> {
        let mut traceback = Vec::with_capacity(
            self.frames
                .len()
                .saturating_sub(min_depth)
                .saturating_add(1),
        );
        self.prepend_current_traceback_entry(&mut traceback);

        if let Some(handler_entry) = self.find_exception_handler(type_id) {
            self.attach_active_python_traceback(&traceback)?;
            self.frames[self.current_frame_idx].ip = handler_entry;
            return Ok(());
        }

        loop {
            if self.frames.len() <= min_depth {
                let mut err = self.uncaught_exception_error(type_id);
                err.traceback = traceback;
                self.attach_active_python_traceback(&err.traceback)?;
                return Err(err);
            }

            self.pop_top_frame_for_unwind();
            if boundary_policy == BoundaryHandlerPolicy::ExcludeBoundary
                && self.frames.len() <= min_depth
            {
                let mut err = self.uncaught_exception_error(type_id);
                err.traceback = traceback;
                self.attach_active_python_traceback(&err.traceback)?;
                return Err(err);
            }

            self.prepend_current_traceback_entry(&mut traceback);
            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                self.attach_active_python_traceback(&traceback)?;
                self.frames[self.current_frame_idx].ip = handler_entry;
                return Ok(());
            }
        }
    }

    #[inline]
    fn propagate_reraise_to_depth(&mut self, min_depth: usize, type_id: u16) -> VmResult<()> {
        self.propagate_reraise_to_depth_with_policy(
            min_depth,
            type_id,
            BoundaryHandlerPolicy::IncludeBoundary,
        )
    }

    #[inline]
    fn propagate_reraise_to_depth_with_policy(
        &mut self,
        min_depth: usize,
        type_id: u16,
        boundary_policy: BoundaryHandlerPolicy,
    ) -> VmResult<()> {
        let mut traceback = Vec::with_capacity(
            self.frames
                .len()
                .saturating_sub(min_depth)
                .saturating_add(1),
        );
        self.prepend_current_traceback_entry(&mut traceback);

        if let Some(handler_entry) = self.find_exception_handler(type_id) {
            self.attach_active_python_traceback(&traceback)?;
            self.frames[self.current_frame_idx].ip = handler_entry;
            return Ok(());
        }

        loop {
            if self.frames.len() <= min_depth {
                let mut err = self.uncaught_reraised_exception_error(type_id);
                err.traceback = traceback;
                self.attach_active_python_traceback(&err.traceback)?;
                return Err(err);
            }

            self.pop_top_frame_for_unwind();
            if boundary_policy == BoundaryHandlerPolicy::ExcludeBoundary
                && self.frames.len() <= min_depth
            {
                let mut err = self.uncaught_reraised_exception_error(type_id);
                err.traceback = traceback;
                self.attach_active_python_traceback(&err.traceback)?;
                return Err(err);
            }

            self.prepend_current_traceback_entry(&mut traceback);
            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                self.attach_active_python_traceback(&traceback)?;
                self.frames[self.current_frame_idx].ip = handler_entry;
                return Ok(());
            }
        }
    }

    #[inline]
    fn propagate_runtime_error_to_depth(
        &mut self,
        min_depth: usize,
        err: RuntimeError,
    ) -> VmResult<()> {
        self.propagate_runtime_error_to_depth_with_policy(
            min_depth,
            err,
            BoundaryHandlerPolicy::IncludeBoundary,
        )
    }

    #[inline]
    fn propagate_runtime_error_to_depth_with_policy(
        &mut self,
        min_depth: usize,
        mut err: RuntimeError,
        boundary_policy: BoundaryHandlerPolicy,
    ) -> VmResult<()> {
        if err.is_control_transferred() {
            return Ok(());
        }

        let type_id = self.materialize_active_exception_from_runtime_error(&err);
        self.prepend_current_traceback_to_error(&mut err);
        if let Some(handler_entry) = self.find_exception_handler(type_id) {
            self.attach_active_python_traceback(&err.traceback)?;
            self.frames[self.current_frame_idx].ip = handler_entry;
            return Ok(());
        }

        loop {
            if self.frames.len() <= min_depth {
                self.attach_active_python_traceback(&err.traceback)?;
                return Err(err);
            }

            self.pop_top_frame_for_unwind();
            if boundary_policy == BoundaryHandlerPolicy::ExcludeBoundary
                && self.frames.len() <= min_depth
            {
                self.attach_active_python_traceback(&err.traceback)?;
                return Err(err);
            }

            self.prepend_current_traceback_to_error(&mut err);
            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                self.attach_active_python_traceback(&err.traceback)?;
                self.frames[self.current_frame_idx].ip = handler_entry;
                return Ok(());
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
            RuntimeErrorKind::ControlTransferred => ExceptionTypeId::RuntimeError.as_u8() as u16,
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
                crate::builtins::create_exception(exc_type_id_enum, err.python_exception_message())
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
    /// Call opcodes bind arguments after this frame is created. Once the frame is
    /// populated, the call path should invoke
    /// [`Self::dispatch_prepared_current_frame_via_jit`] before falling back to
    /// the interpreter loop.
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
        self.validate_code_object_for_execution(&code)?;

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

            let global_scope = &self.globals as *const GlobalScope as *const u64;
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

                        Some(
                            match jit.try_execute_with_global_scope(
                                code_ptr_id,
                                &mut jit_frame,
                                global_scope,
                            ) {
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
                            },
                        )
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

    fn validate_code_object_for_execution(&mut self, code: &Arc<CodeObject>) -> VmResult<()> {
        let key = Arc::as_ptr(code) as usize;
        if self.validated_code_objects.contains_key(&key) {
            return Ok(());
        }

        code.validate().map_err(|err| {
            RuntimeError::internal(format!(
                "invalid bytecode in {} ({}): {}",
                code.qualname, code.filename, err
            ))
        })?;

        self.validated_code_objects.insert(key, Arc::clone(code));
        Ok(())
    }

    /// Try to execute the current, already-initialized call frame through JIT.
    ///
    /// Ordinary Python calls need their argument registers populated before JIT
    /// entry. This method is the hot post-binding dispatch hook: it performs the
    /// same tier-up and compiled-code lookup as root-frame dispatch, but executes
    /// against the real callee frame instead of allocating a temporary empty
    /// frame. On return, the callee frame is popped exactly like the interpreter
    /// return path; on deopt, the frame is left in place at the requested bytecode
    /// offset.
    #[inline]
    pub(crate) fn dispatch_prepared_current_frame_via_jit(&mut self) -> VmResult<bool> {
        if self.frames.is_empty() || self.execution_budget.step_limit().is_some() {
            return Ok(false);
        }

        let frame_idx = self.current_frame_idx;
        let code = {
            let frame = &self.frames[frame_idx];
            if frame.ip != 0 || frame.closure.is_some() {
                return Ok(false);
            }
            Arc::clone(&frame.code)
        };

        let code_id = CodeId::from_ptr(Arc::as_ptr(&code) as *const ());
        let code_ptr_id = Arc::as_ptr(&code) as u64;
        let global_scope = &self.globals as *const GlobalScope as *const u64;
        let execution_result = {
            let Some(jit) = self.jit.as_mut() else {
                return Ok(false);
            };

            let tier_decision = jit.check_tier_up(&self.profiler, code_id);
            if tier_decision != TierUpDecision::None {
                jit.handle_tier_up(&code, tier_decision);
            }

            if jit.lookup(code_ptr_id).is_none() {
                jit.record_miss();
                return Ok(false);
            }

            jit.try_execute_with_global_scope(
                code_ptr_id,
                &mut self.frames[frame_idx],
                global_scope,
            )
        };

        match execution_result {
            Some(ExecutionResult::Return(value)) => {
                if let Some(root_value) = self.pop_frame(value)? {
                    self.jit_return_value = Some(root_value);
                }
                Ok(true)
            }
            Some(ExecutionResult::Deopt { bc_offset, reason }) => {
                if let Some(jit) = self.jit.as_mut() {
                    jit.handle_deopt(code_ptr_id, reason);
                }
                self.frames[frame_idx].ip = bc_offset;
                Ok(false)
            }
            Some(ExecutionResult::Exception(err)) => Err(err),
            Some(ExecutionResult::TailCall { .. }) | None => {
                if let Some(jit) = self.jit.as_mut() {
                    jit.record_miss();
                }
                Ok(false)
            }
        }
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

    /// Resolve an active frame index by caller depth from the current frame.
    #[inline]
    pub(crate) fn frame_index_at_depth(&self, depth: usize) -> Option<usize> {
        let mut frame_index = self.current_frame_idx;
        let mut remaining = depth;

        while remaining > 0 {
            let frame = self.frames.get(frame_index)?;
            frame_index = frame.return_frame? as usize;
            remaining -= 1;
        }

        self.frames.get(frame_index).map(|_| frame_index)
    }

    /// Get an active frame by caller depth from the current frame.
    #[inline]
    pub fn frame_at_depth(&self, depth: usize) -> Option<&Frame> {
        self.frame_index_at_depth(depth)
            .and_then(|frame_index| self.frames.get(frame_index))
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
        self.handler_stack.clear();
        self.active_exception = None;
        self.active_exception_type_id = None;
        self.exc_info_stack.clear();
        self.active_except_handlers.clear();
        self.exc_state = ExceptionState::Normal;
    }

    // =========================================================================
    // GC Integration
    // =========================================================================

    /// Get read-only access to the managed heap.
    ///
    /// Use this to query heap statistics, check collection thresholds,
    /// or read heap configuration.
    #[inline]
    pub fn heap(&self) -> MutexGuard<'_, ManagedHeap> {
        self.heap.lock().expect("managed heap lock poisoned")
    }

    /// Get mutable access to the managed heap.
    ///
    /// Required for:
    /// - Triggering garbage collection
    /// - Updating root sets
    /// - Modifying heap configuration
    #[inline]
    pub fn heap_mut(&self) -> MutexGuard<'_, ManagedHeap> {
        self.heap.lock().expect("managed heap lock poisoned")
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
        let heap_ptr = {
            let heap = self.heap.lock().expect("managed heap lock poisoned");
            heap.heap() as *const _
        };

        // The shared heap is owned by `self.heap` and its allocation never moves
        // while the VM holds the Arc. GcHeap allocation is internally atomic; GC
        // collection remains synchronized through `heap_mut()`.
        GcAllocator::new(unsafe { &*heap_ptr })
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
        if self.has_active_exception() {
            self.active_exception.as_ref()
        } else {
            None
        }
    }

    /// Check if there's an active exception.
    #[inline]
    pub fn has_active_exception(&self) -> bool {
        self.exc_state.has_exception()
            && self.active_exception.is_some()
            && self
                .active_exception_type_id
                .is_some_and(|type_id| type_id != 0)
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

    /// Finish a finally cleanup that did not need to reraise.
    #[inline]
    pub fn finish_finally_without_reraise(&mut self) {
        match self.exc_state {
            ExceptionState::Handling | ExceptionState::Finally => {}
            ExceptionState::Normal | ExceptionState::Propagating | ExceptionState::Unhandled => {
                self.clear_active_exception();
                self.exc_state = ExceptionState::Normal;
            }
        }
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
    /// Searches the frame's exception table for a handler chain that covers the
    /// current PC (`start_pc <= pc < end_pc`). Python `except` expressions are
    /// evaluated dynamically in the emitted handler bytecode, so this lookup
    /// deliberately routes to the most specific protected range and leaves type
    /// matching to `ExceptionMatch`.
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

        let has_active_exception = self.has_active_exception();
        let type_id = if has_active_exception {
            self.get_active_exception_type_id().unwrap_or(0)
        } else {
            0
        };
        let value = if has_active_exception {
            self.active_exception
        } else {
            None
        };
        let mut entry = ExcInfoEntry::new(type_id, value);
        if self.exc_state == ExceptionState::Handling {
            entry.flags_mut().set(EntryFlags::HANDLING);
        } else if self.exc_state == ExceptionState::Finally {
            entry.flags_mut().set(EntryFlags::FINALLY);
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
                } else if entry.flags().has(EntryFlags::FINALLY) {
                    ExceptionState::Finally
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
        !self.exc_info_stack.is_empty() || self.has_active_exception()
    }

    /// Get current exception info as (type_id, value, traceback_id).
    #[inline]
    pub fn current_exc_info(&self) -> (Option<u16>, Option<Value>, Option<u32>) {
        self.exc_info_stack.current_exc_info()
    }
}

fn module_export_error_to_runtime_error(err: ModuleExportError) -> RuntimeError {
    match err {
        ModuleExportError::InvalidAll { message, .. } => RuntimeError::type_error(message),
        ModuleExportError::NonStringAllItem { .. } => RuntimeError::type_error(err.to_string()),
        ModuleExportError::MissingAllAttribute { module, name } => RuntimeError::exception(
            ExceptionTypeId::AttributeError.as_u8() as u16,
            format!("module '{}' has no attribute '{}'", module, name),
        ),
    }
}

impl Default for VirtualMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VirtualMachine {
    fn drop(&mut self) {
        if !self.join_threads_on_drop {
            return;
        }

        crate::threading_runtime::blocking_operation(|| {
            let _ = self.join_owned_threads(Duration::from_secs(5));
        });
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
mod tests;
