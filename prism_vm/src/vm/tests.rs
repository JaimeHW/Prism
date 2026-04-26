use super::*;
use crate::builtins::builtin_getattr;
use crate::exception::HandlerFrame;
use crate::import::FrozenModuleSource;
use prism_code::{CodeFlags, CodeObject, ExceptionEntry, Register};
use prism_compiler::OptimizationLevel;
use prism_core::intern::intern;
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

#[test]
fn test_published_class_scope_unregisters_vm_owned_heap_classes() {
    use crate::ops::method_dispatch::{CachedMethod, method_cache};
    use prism_runtime::object::type_builtins::{
        SubclassBitmap, global_class, global_class_version, register_global_class,
    };

    method_cache().clear();
    let scope = new_shared_class_publications();
    let class = Arc::new(PyClassObject::new_simple(intern("ScopedHeapType")));
    let class_id = class.class_id();
    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(class.class_type_id());

    register_global_class(Arc::clone(&class), bitmap);
    scope.record(class_id);
    assert!(global_class(class_id).is_some());

    let method_name = intern("__scoped_method_cache_probe__");
    let method_name_ptr = method_name.as_ptr() as u64;
    let version = global_class_version(class_id).expect("published class version");
    method_cache().insert(
        class.class_type_id(),
        method_name_ptr,
        version,
        CachedMethod::simple(Value::int_unchecked(17)),
    );
    assert!(
        method_cache()
            .get(class.class_type_id(), method_name_ptr, version)
            .is_some()
    );

    drop(scope);
    assert!(global_class(class_id).is_none());
    assert!(
        method_cache()
            .get(class.class_type_id(), method_name_ptr, version)
            .is_none()
    );
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
    let instruction_count = exception_table.iter().fold(1_u32, |count, entry| {
        let finally_limit = if entry.finally_pc == u32::MAX {
            0
        } else {
            entry.finally_pc.saturating_add(1)
        };
        count
            .max(entry.end_pc)
            .max(entry.handler_pc.saturating_add(1))
            .max(finally_limit)
    }) as usize;
    Arc::new(CodeObject {
        name: Arc::from(name),
        register_count: 1,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: vec![Instruction::op(Opcode::Nop); instruction_count].into_boxed_slice(),
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
    compile_source_code(source, filename, OptimizationLevel::Basic).expect("source should compile")
}

fn compile_source_module_for_test(source: &str, filename: &str) -> Arc<CodeObject> {
    prism_compiler::compile_source_code(source, filename, OptimizationLevel::Basic)
        .expect("source should compile")
}

#[test]
fn test_vm_creation() {
    let vm = VirtualMachine::new();
    assert!(vm.is_idle());
    assert_eq!(vm.call_depth(), 0);
}

#[test]
fn test_execute_rejects_invalid_bytecode_before_dispatch() {
    let mut code = CodeObject::new("bad_bytecode", "<test>");
    code.register_count = 1;
    code.instructions =
        vec![Instruction::op_di(Opcode::LoadConst, Register::new(0), 0)].into_boxed_slice();

    let mut vm = VirtualMachine::new();
    let err = vm
        .execute_runtime(Arc::new(code))
        .expect_err("invalid bytecode should be rejected before dispatch");

    match err.kind() {
        RuntimeErrorKind::InternalError { message } => {
            assert!(message.contains("invalid bytecode in bad_bytecode (<test>)"));
            assert!(message.contains("constant index 0 out of bounds"));
        }
        other => panic!("expected internal bytecode validation error, got {other:?}"),
    }
    assert!(vm.is_idle());
}

#[test]
fn test_push_frame_caches_bytecode_validation_by_code_identity() {
    let code = empty_code("cached");
    let mut vm = VirtualMachine::new();

    vm.push_frame(Arc::clone(&code), 0)
        .expect("valid bytecode should push");
    assert_eq!(vm.validated_code_objects.len(), 1);
    vm.pop_frame(Value::none())
        .expect("valid frame should pop cleanly");

    vm.push_frame(Arc::clone(&code), 0)
        .expect("validated bytecode should push again");
    assert_eq!(
        vm.validated_code_objects.len(),
        1,
        "validation cache should reuse the original code-object identity"
    );
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
        let resolved =
            prism_core::intern::interned_by_ptr(string_ptr).expect("prefix entry should resolve");
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
        .expect("first tuple entry should be an interned string") as *const u8;
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
    let base =
        prism_core::intern::interned_by_ptr(base_ptr).expect("base marker string should resolve");
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
        .expect("marker should be stored as an interned string") as *const u8;
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
            .expect("tuple entry should be an interned string") as *const u8;
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
fn test_function_closure_cells_are_owned_by_function_objects_across_vm_instances() {
    let source = concat!(
        "def outer():\n",
        "    x = 41\n",
        "    def inner():\n",
        "        return x\n",
        "    return inner\n",
        "fn = outer()\n",
        "assert fn() == 41\n",
        "assert fn.__closure__[0].cell_contents == 41\n",
    );

    for _ in 0..8 {
        let mut vm = VirtualMachine::new();
        vm.execute_in_module(
            compile_module(source, "<closure-owner>"),
            Arc::new(ModuleObject::new("__main__")),
        )
        .expect("function-owned closure cells should not depend on VM pointer registries");
    }
}

#[test]
fn test_prepared_user_function_calls_tier_up_and_execute_jit() {
    let mut vm = VirtualMachine::with_jit_config(JitConfig::for_testing());
    let module = Arc::new(ModuleObject::new("__main__"));

    vm.execute_in_module_runtime(
        compile_source_module_for_test(
            concat!(
                "def answer():\n",
                "    return 7\n",
                "result = 0\n",
                "for _ in range(12):\n",
                "    result = answer()\n",
            ),
            "prepared_jit_call.py",
        ),
        Arc::clone(&module),
    )
    .expect("prepared user-function calls should execute correctly under JIT");

    assert_eq!(
        module.get_attr("result").and_then(|value| value.as_int()),
        Some(7)
    );

    let stats = vm.jit.as_ref().expect("JIT should be enabled").stats();
    assert!(
        stats.compilations_triggered > 0,
        "ordinary user-function calls should trigger JIT tier-up"
    );
    assert!(
        stats.cache_hits > 0,
        "ordinary user-function calls should execute compiled code after tier-up"
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
fn test_execute_in_module_runtime_reports_name_error_line() {
    let mut vm = VirtualMachine::new();
    let err = vm
        .execute_in_module_runtime(
            compile_module("x = 1\ny = missing_name\n", "traceback_name_error.py"),
            Arc::new(ModuleObject::new("__main__")),
        )
        .expect_err("execution should raise a NameError");

    assert_eq!(err.traceback.len(), 1);
    assert_eq!(
        err.traceback[0].filename.as_ref(),
        "traceback_name_error.py"
    );
    assert_eq!(err.traceback[0].line, 2);
    assert!(err.to_string().contains("missing_name"));
}

#[test]
fn test_execute_in_module_restores_generator_seeded_locals_as_assigned() {
    let mut vm = VirtualMachine::new();
    let module = Arc::new(ModuleObject::new("__main__"));

    vm.execute_in_module_runtime(
        compile_module(
            "def first(seq):\n    return next(x for x in seq)\nresult = first([7])\n",
            "generator_seeded_locals.py",
        ),
        Arc::clone(&module),
    )
    .expect("generator hidden iterator argument should remain assigned");

    assert_eq!(
        module.get_attr("result").and_then(|value| value.as_int()),
        Some(7)
    );
}

#[test]
fn test_execute_in_module_runtime_preserves_nested_traceback_frames() {
    let mut vm = VirtualMachine::new();
    let err = vm
        .execute_in_module_runtime(
            compile_module(
                "def inner():\n    raise ValueError('boom')\ndef outer():\n    inner()\nouter()\n",
                "traceback_stack.py",
            ),
            Arc::new(ModuleObject::new("__main__")),
        )
        .expect_err("execution should raise a ValueError");

    let observed: Vec<_> = err
        .traceback
        .iter()
        .map(|entry| (entry.func_name.as_ref().to_string(), entry.line))
        .collect();
    assert_eq!(
        observed,
        vec![
            ("<module>".to_string(), 5),
            ("outer".to_string(), 4),
            ("inner".to_string(), 2),
        ]
    );
}

#[test]
fn test_python_traceback_allocation_survives_full_nursery() {
    let vm = VirtualMachine::new();
    let mut exhausted = false;

    for _ in 0..200_000 {
        if vm.allocator().alloc(DictObject::new()).is_none() {
            exhausted = true;
            break;
        }
    }
    assert!(exhausted, "test setup should fill the nursery");

    let traceback = vm
        .python_traceback_from_entries(&[TracebackEntry {
            func_name: Arc::from("boom"),
            filename: Arc::from("traceback_exhaustion.py"),
            line: 7,
        }])
        .expect("traceback allocation should not turn nursery exhaustion into OOM");

    assert!(traceback.as_object_ptr().is_some());
}

#[test]
fn test_handled_runtime_error_attaches_python_traceback_to_exc_info() {
    let mut vm = VirtualMachine::new();
    let module = Arc::new(ModuleObject::new("__main__"));
    vm.execute_in_module_runtime(
        compile_source_module_for_test(
            "import sys\ntry:\n    1 / 0\nexcept ZeroDivisionError:\n    tb = sys.exc_info()[2]\n",
            "handled_traceback.py",
        ),
        Arc::clone(&module),
    )
    .expect("handled exception should not escape");

    let traceback = module
        .get_attr("tb")
        .expect("except handler should store traceback");
    let traceback_ptr = traceback
        .as_object_ptr()
        .expect("traceback should be an object");
    assert_eq!(
        crate::ops::objects::extract_type_id(traceback_ptr),
        TypeId::TRACEBACK
    );

    let line = crate::ops::objects::get_attribute_value(&mut vm, traceback, &intern("tb_lineno"))
        .expect("tb_lineno should be readable");
    assert_eq!(line.as_int(), Some(3));

    let frame = crate::ops::objects::get_attribute_value(&mut vm, traceback, &intern("tb_frame"))
        .expect("tb_frame should be readable");
    let code = crate::ops::objects::get_attribute_value(&mut vm, frame, &intern("f_code"))
        .expect("f_code should be readable");
    let name = crate::ops::objects::get_attribute_value(&mut vm, code, &intern("co_name"))
        .expect("co_name should be readable");
    assert_eq!(name, Value::string(intern("<module>")));
}

#[test]
fn test_traceback_for_caller_handler_includes_call_site_and_inner_raise() {
    let mut vm = VirtualMachine::new();
    let module = Arc::new(ModuleObject::new("__main__"));
    vm.execute_in_module_runtime(
        compile_source_module_for_test(
            "import sys\ndef inner():\n    raise ValueError('boom')\ndef outer():\n    try:\n        inner()\n    except ValueError:\n        return sys.exc_info()[2]\ntb = outer()\n",
            "handled_nested_traceback.py",
        ),
        Arc::clone(&module),
    )
    .expect("handled nested exception should not escape");

    let outer_tb = module
        .get_attr("tb")
        .expect("except handler should store traceback");
    let outer_line =
        crate::ops::objects::get_attribute_value(&mut vm, outer_tb, &intern("tb_lineno"))
            .expect("outer tb_lineno should be readable");
    assert_eq!(outer_line.as_int(), Some(6));

    let inner_tb = crate::ops::objects::get_attribute_value(&mut vm, outer_tb, &intern("tb_next"))
        .expect("tb_next should be readable");
    let inner_line =
        crate::ops::objects::get_attribute_value(&mut vm, inner_tb, &intern("tb_lineno"))
            .expect("inner tb_lineno should be readable");
    assert_eq!(inner_line.as_int(), Some(3));
}

#[test]
fn test_builtin_setattr_property_exception_transfers_to_python_handler() {
    let mut vm = VirtualMachine::new();
    let module = Arc::new(ModuleObject::new("__main__"));

    vm.execute_in_module_runtime(
        compile_source_module_for_test(
            concat!(
                "class Managed:\n",
                "    @property\n",
                "    def value(self):\n",
                "        return 1\n",
                "    @value.setter\n",
                "    def value(self, new_value):\n",
                "        raise RuntimeError('managed setter failed')\n",
                "\n",
                "obj = Managed()\n",
                "try:\n",
                "    setattr(obj, 'value', 3)\n",
                "except RuntimeError:\n",
                "    RESULT = 1\n",
                "else:\n",
                "    RESULT = 0\n",
            ),
            "handled_setattr_property.py",
        ),
        Arc::clone(&module),
    )
    .expect("handled property setter exception should not escape");

    assert_eq!(
        module.get_attr("RESULT").and_then(|value| value.as_int()),
        Some(1)
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
fn test_import_from_with_context_loads_source_backed_submodule() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg/__init__.py"), "");
    write_file(&temp.path.join("pkg/child.py"), "VALUE = 123\n");

    let mut vm = VirtualMachine::new();
    vm.import_resolver
        .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

    let child_value = vm
        .import_from_with_context("pkg", "child", None)
        .expect("from pkg import child should succeed");
    let child_ptr = child_value
        .as_object_ptr()
        .expect("child import should return a module object");
    let child = vm
        .import_resolver
        .module_from_ptr(child_ptr)
        .expect("returned object should map to a cached module");
    assert_eq!(child.name(), "pkg.child");
    assert_eq!(
        child.get_attr("VALUE").and_then(|value| value.as_int()),
        Some(123)
    );

    let package = vm
        .import_module_named("pkg")
        .expect("pkg should remain importable");
    assert_eq!(package.get_attr("child"), Some(child_value));
}

#[test]
fn test_star_import_uses_child_module_all_without_leaking_public_names() {
    let temp = TestTempDir::new();
    write_file(
        &temp.path.join("pkg/__init__.py"),
        "from .child import *\nRESULT = VALUE\n",
    );
    write_file(
        &temp.path.join("pkg/child.py"),
        "__all__ = ('VALUE',)\nVALUE = 42\nsubprocess = 'should not leak'\n",
    );

    let mut vm = VirtualMachine::new();
    vm.import_resolver
        .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

    let module = vm
        .import_module_named("pkg")
        .expect("package star import should honor child __all__");

    assert_eq!(
        module.get_attr("RESULT").and_then(|value| value.as_int()),
        Some(42)
    );
    assert!(
        module.get_attr("subprocess").is_none(),
        "star import should not import public names omitted from __all__"
    );
}

#[test]
fn test_import_from_with_context_preserves_nested_submodule_failure() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg/__init__.py"), "");
    write_file(
        &temp.path.join("pkg/child.py"),
        "import missing_dependency\nVALUE = 123\n",
    );

    let mut vm = VirtualMachine::new();
    vm.import_resolver
        .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));

    let err = vm
        .import_from_with_context("pkg", "child", None)
        .expect_err("nested import failure should propagate");
    match err.kind() {
        RuntimeErrorKind::ImportError {
            name,
            missing,
            module,
            ..
        } => {
            assert!(*missing);
            assert_eq!(name.as_deref(), Some("missing_dependency"));
            assert_eq!(module.as_ref(), "missing_dependency");
        }
        other => panic!("expected missing nested import error, got {other:?}"),
    }
    assert!(
        err.traceback
            .iter()
            .any(|entry| { entry.filename.as_ref().ends_with("child.py") && entry.line == 1 }),
        "nested import failure should retain the child module traceback"
    );
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

    match err.kind() {
        crate::error::RuntimeErrorKind::Exception { type_id, message } => {
            assert_eq!(
                *type_id,
                crate::stdlib::exceptions::ExceptionTypeId::TypeError.as_u8() as u16
            );
            assert_eq!(message.as_ref(), "real uncaught message");
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

    match err.kind() {
        crate::error::RuntimeErrorKind::Exception { type_id, message } => {
            assert_eq!(
                *type_id,
                crate::stdlib::exceptions::ExceptionTypeId::TypeError.as_u8() as u16
            );
            assert_eq!(message.as_ref(), "Uncaught exception (type_id=52)");
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
fn test_push_exc_info_ignores_stale_normal_exception_value() {
    let mut vm = VirtualMachine::new();
    vm.set_active_exception_with_type(Value::int(11).unwrap(), 24);
    vm.clear_exception_state();

    assert!(!vm.has_active_exception());
    assert!(vm.get_active_exception().is_none());
    assert!(!vm.has_exc_info());

    assert!(vm.push_exc_info());
    vm.set_active_exception_with_type(Value::int(22).unwrap(), 5);
    assert!(vm.pop_exc_info());

    assert!(vm.get_active_exception().is_none());
    assert_eq!(vm.get_active_exception_type_id(), None);
    assert_eq!(vm.exception_state(), ExceptionState::Normal);
}

#[test]
fn test_pop_exc_info_restores_handling_state() {
    let mut vm = VirtualMachine::new();
    vm.push_frame(empty_code("handler"), 0).unwrap();

    vm.set_active_exception_with_type(Value::int(31).unwrap(), 24);
    assert!(vm.enter_except_handler());
    assert!(vm.push_exc_info());

    vm.set_active_exception_with_type(Value::int(32).unwrap(), 5);
    assert!(vm.pop_exc_info());

    assert_eq!(vm.exception_state(), ExceptionState::Handling);
    assert_eq!(vm.get_active_exception_type_id(), Some(24));
    assert_eq!(vm.get_active_exception().and_then(Value::as_int), Some(31));
}

#[test]
fn test_pop_exc_info_restores_finally_state() {
    let mut vm = VirtualMachine::new();

    vm.set_active_exception_with_type(Value::int(41).unwrap(), 24);
    vm.set_exception_state(ExceptionState::Finally);
    assert!(vm.push_exc_info());

    vm.set_active_exception_with_type(Value::int(42).unwrap(), 5);
    assert!(vm.pop_exc_info());

    assert_eq!(vm.exception_state(), ExceptionState::Finally);
    assert_eq!(vm.get_active_exception_type_id(), Some(24));
    assert_eq!(vm.get_active_exception().and_then(Value::as_int), Some(41));
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
