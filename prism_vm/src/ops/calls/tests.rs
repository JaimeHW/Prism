use super::BoundArgs;
use super::DIRECT_CALL_RETURN_REG;
use super::call;
use super::invoke_callable_value;
use super::invoke_callable_value_with_keywords;
use super::resolve_instantiation_slot;
use super::restore_direct_call_caller_state;
use super::value_supports_call_protocol;
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::dispatch::ControlFlow;
use prism_code::{CodeObject, Constant, Instruction, Opcode, Register};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassDict, ClassFlags, PyClassObject};
use prism_runtime::object::descriptor::{BoundMethod, StaticMethodDescriptor};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, builtin_class_mro, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn register_test_class(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(TypeId::from_raw(class_id.0));
    }

    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

struct TestTempDir {
    path: PathBuf,
}

impl TestTempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut path = std::env::temp_dir();
        path.push(format!(
            "prism_ops_calls_tests_{}_{}_{}",
            std::process::id(),
            nanos,
            unique
        ));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        Self { path }
    }
}

impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("failed to create parent dir");
    }
    fs::write(path, content).expect("failed to write test file");
}

fn bound_method_probe(args: &[Value]) -> Result<Value, BuiltinError> {
    assert_eq!(args.len(), 2);
    assert_eq!(args[0].as_int(), Some(7));
    assert_eq!(args[1].as_int(), Some(11));
    Ok(args[1])
}

fn builtin_init_probe(args: &[Value]) -> Result<Value, BuiltinError> {
    assert_eq!(args.len(), 2);
    assert!(args[0].as_object_ptr().is_some());
    assert_eq!(args[1].as_int(), Some(17));
    Ok(Value::none())
}

fn exhaust_nursery(vm: &VirtualMachine) {
    for _ in 0..200_000 {
        if vm.allocator().alloc(DictObject::new()).is_none() {
            return;
        }
    }
    panic!("test setup should fill the nursery");
}

#[test]
fn test_bound_args_inline_tracks_edges() {
    let mut bound = BoundArgs::new(64);
    bound.set_bound(0);
    bound.set_bound(63);

    assert!(bound.is_bound(0));
    assert!(bound.is_bound(63));
    assert!(!bound.is_bound(1));
    assert!(!bound.is_bound(62));
}

#[test]
fn test_bound_args_heap_handles_large_signatures() {
    let mut bound = BoundArgs::new(80);
    bound.set_bound(0);
    bound.set_bound(64);
    bound.set_bound(79);

    assert!(bound.is_bound(0));
    assert!(bound.is_bound(64));
    assert!(bound.is_bound(79));
    assert!(!bound.is_bound(1));
    assert!(!bound.is_bound(78));
}

#[test]
fn test_bound_args_heap_out_of_range_is_safe() {
    let mut bound = BoundArgs::new(65);
    bound.set_bound(70);
    assert!(!bound.is_bound(70));
}

#[test]
fn test_bound_variadics_allocate_after_full_nursery() {
    let mut vm = VirtualMachine::new();
    exhaust_nursery(&vm);

    let mut bound = crate::ops::kw_binding::BoundArguments {
        parameters: Vec::new(),
        varargs: Some(Box::new(TupleObject::from_slice(&[Value::int(1).unwrap()]))),
        varkw: Some(Box::new(DictObject::new())),
    };

    let (varargs, varkw) = super::allocate_bound_variadics(&mut vm, &mut bound)
        .expect("bound variadics should use stable fallback storage");
    assert!(varargs.and_then(|value| value.as_object_ptr()).is_some());
    assert!(varkw.and_then(|value| value.as_object_ptr()).is_some());
}

#[test]
fn test_make_function_allocates_after_full_nursery() {
    let child_code = Arc::new(CodeObject::new("child", "<test>"));
    let mut root_code = CodeObject::new("root", "<test>");
    root_code.constants = vec![Constant::Value(Value::object_ptr(
        Arc::as_ptr(&child_code) as *const ()
    ))]
    .into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(root_code), 0)
        .expect("frame push should succeed");
    exhaust_nursery(&vm);

    let inst = Instruction::op_di(Opcode::MakeFunction, Register::new(0), 0);
    assert!(matches!(
        super::make_function(&mut vm, inst),
        ControlFlow::Continue
    ));
    assert!(vm.current_frame().get_reg(0).as_object_ptr().is_some());
}

#[test]
fn test_restore_direct_call_caller_state_only_restores_scratch_register() {
    let mut vm = VirtualMachine::new();
    vm.push_frame(
        Arc::new(CodeObject::new("direct_call_restore", "<test>")),
        0,
    )
    .expect("frame push should succeed");
    vm.current_frame_mut().set_reg(7, Value::int(1).unwrap());

    let saved_register = vm.current_frame().snapshot_register(DIRECT_CALL_RETURN_REG);
    let saved_exception_context = vm.capture_exception_context();

    vm.current_frame_mut()
        .set_reg(DIRECT_CALL_RETURN_REG, Value::int(99).unwrap());
    vm.current_frame_mut().set_reg(7, Value::int(42).unwrap());

    let stop_depth = vm.call_depth();
    restore_direct_call_caller_state(&mut vm, stop_depth, saved_register, saved_exception_context);

    assert_eq!(vm.current_frame().get_reg(7).as_int(), Some(42));
    assert_eq!(
        vm.current_frame().get_reg(DIRECT_CALL_RETURN_REG),
        Value::none()
    );
    assert!(vm.current_frame().reg_is_written(7));
    assert!(!vm.current_frame().reg_is_written(DIRECT_CALL_RETURN_REG));
}

#[test]
fn test_restore_direct_call_caller_state_preserves_written_scratch_register_state() {
    let mut vm = VirtualMachine::new();
    vm.push_frame(
        Arc::new(CodeObject::new("direct_call_written", "<test>")),
        0,
    )
    .expect("frame push should succeed");
    vm.current_frame_mut()
        .set_reg(DIRECT_CALL_RETURN_REG, Value::int(5).unwrap());
    vm.current_frame_mut().set_reg(8, Value::int(11).unwrap());

    let saved_register = vm.current_frame().snapshot_register(DIRECT_CALL_RETURN_REG);
    let saved_exception_context = vm.capture_exception_context();

    vm.current_frame_mut()
        .set_reg(DIRECT_CALL_RETURN_REG, Value::int(77).unwrap());
    vm.current_frame_mut().set_reg(8, Value::int(22).unwrap());

    let stop_depth = vm.call_depth();
    restore_direct_call_caller_state(&mut vm, stop_depth, saved_register, saved_exception_context);

    assert_eq!(
        vm.current_frame().get_reg(DIRECT_CALL_RETURN_REG).as_int(),
        Some(5)
    );
    assert_eq!(vm.current_frame().get_reg(8).as_int(), Some(22));
    assert!(vm.current_frame().reg_is_written(DIRECT_CALL_RETURN_REG));
    assert!(vm.current_frame().reg_is_written(8));
}

#[test]
fn test_value_supports_call_protocol_for_instance_property_dunder_call() {
    let registry = shape_registry();
    let mut object = ShapedObject::with_empty_shape(registry.empty_shape());
    let func = Box::new(FunctionObject::new(
        Arc::new(CodeObject::new("__call__", "<test>")),
        Arc::from("__call__"),
        None,
        None,
    ));
    let func_ptr = Box::into_raw(func);
    object.set_property(
        intern("__call__"),
        Value::object_ptr(func_ptr as *const ()),
        registry,
    );
    let object_ptr = Box::into_raw(Box::new(object));
    let object_value = Value::object_ptr(object_ptr as *const ());

    assert!(value_supports_call_protocol(object_value));

    unsafe {
        drop(Box::from_raw(object_ptr));
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_value_supports_call_protocol_for_heap_type_dunder_call() {
    let func = Box::new(FunctionObject::new(
        Arc::new(CodeObject::new("__call__", "<test>")),
        Arc::from("__call__"),
        None,
        None,
    ));
    let func_ptr = Box::into_raw(func);

    let mut class = PyClassObject::new_simple(intern("CallableType"));
    class.set_attr(intern("__call__"), Value::object_ptr(func_ptr as *const ()));
    let class = register_test_class(class);

    let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    assert!(value_supports_call_protocol(instance_value));

    unsafe {
        drop(Box::from_raw(instance_ptr));
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_call_executes_bound_method_values() {
    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(CodeObject::new("call_bound_method", "<test>")), 0)
        .expect("frame push should succeed");

    let builtin_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
        Arc::from("test.bound_method_probe"),
        bound_method_probe,
    )));
    let method_ptr = Box::into_raw(Box::new(BoundMethod::new(
        Value::object_ptr(builtin_ptr as *const ()),
        Value::int(7).unwrap(),
    )));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(method_ptr as *const ()));
    vm.current_frame_mut().set_reg(3, Value::int(11).unwrap());

    let inst = Instruction::op_dss(
        Opcode::Call,
        Register::new(2),
        Register::new(1),
        Register::new(1),
    );

    assert!(matches!(call(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(11));

    unsafe {
        drop(Box::from_raw(method_ptr));
        drop(Box::from_raw(builtin_ptr));
    }
}

fn staticmethod_identity(args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(args.first().copied().unwrap_or_else(Value::none))
}

#[test]
fn test_staticmethod_values_are_callable() {
    let builtin_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
        Arc::from("test.staticmethod_identity"),
        staticmethod_identity,
    )));
    let staticmethod_ptr = Box::into_raw(Box::new(StaticMethodDescriptor::new(Value::object_ptr(
        builtin_ptr as *const (),
    ))));
    let staticmethod_value = Value::object_ptr(staticmethod_ptr as *const ());
    let mut vm = VirtualMachine::new();

    assert!(value_supports_call_protocol(staticmethod_value));
    assert_eq!(
        invoke_callable_value(&mut vm, staticmethod_value, &[Value::int(7).unwrap()])
            .expect("staticmethod should be directly callable")
            .as_int(),
        Some(7)
    );

    vm.push_frame(Arc::new(CodeObject::new("call_staticmethod", "<test>")), 0)
        .expect("frame push should succeed");
    vm.current_frame_mut().set_reg(1, staticmethod_value);
    vm.current_frame_mut().set_reg(3, Value::int(11).unwrap());

    let inst = Instruction::op_dss(
        Opcode::Call,
        Register::new(2),
        Register::new(1),
        Register::new(1),
    );
    assert!(matches!(call(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(11));

    unsafe {
        drop(Box::from_raw(staticmethod_ptr));
        drop(Box::from_raw(builtin_ptr));
    }
}

#[test]
fn test_invoke_callable_value_executes_reflected_wrapper_descriptor() {
    let mut vm = VirtualMachine::new();
    let descriptor =
        crate::builtins::builtin_type_attribute_value_static(TypeId::OBJECT, &intern("__init__"))
            .expect("lookup should succeed")
            .expect("object.__init__ descriptor should exist");
    let instance = crate::builtins::builtin_object(&[]).expect("object() should succeed");

    let result = invoke_callable_value(&mut vm, descriptor, &[instance])
        .expect("wrapper descriptor should resolve to a callable");
    assert!(result.is_none());
}

#[test]
fn test_resolve_instantiation_slot_for_metaclass_falls_back_to_type_init() {
    let mut meta = PyClassObject::new(intern("Meta"), &[ClassId(TypeId::TYPE.raw())], |id| {
        Some(
            builtin_class_mro(class_id_to_type_id(id))
                .into_iter()
                .collect(),
        )
    })
    .expect("metaclass should build");
    meta.add_flags(ClassFlags::METACLASS);

    assert!(resolve_instantiation_slot(&meta, "__new__").is_some());
    assert!(resolve_instantiation_slot(&meta, "__init__").is_some());
}

#[test]
fn test_invoke_callable_value_instantiates_heap_metaclass_with_type_init() {
    let mut vm = VirtualMachine::new();

    let meta = prism_runtime::object::type_builtins::type_new_with_metaclass(
        intern("Meta"),
        &[ClassId(TypeId::TYPE.raw())],
        &ClassDict::new(),
        crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE),
        prism_runtime::object::type_builtins::global_class_registry(),
    )
    .expect("type_new_with_metaclass should create a metaclass");
    register_global_class(meta.class.clone(), meta.bitmap);

    let bases = TupleObject::empty();
    let class_value = super::invoke_callable_value(
        &mut vm,
        Value::object_ptr(Arc::as_ptr(&meta.class) as *const ()),
        &[
            Value::string(intern("Generated")),
            Value::object_ptr(Box::into_raw(Box::new(bases)) as *const ()),
            Value::object_ptr(Box::into_raw(Box::new(DictObject::new())) as *const ()),
        ],
    )
    .expect("heap metaclass call should succeed");

    let class_ptr = class_value
        .as_object_ptr()
        .expect("metaclass call should return a class object");
    assert_eq!(super::extract_type_id(class_ptr), TypeId::TYPE);
}

#[test]
fn test_invoke_callable_value_instantiates_heap_class_with_inherited_object_new() {
    let mut vm = VirtualMachine::new();

    let init_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
        Arc::from("tests.PositionalInit.__init__"),
        builtin_init_probe,
    )));

    let mut class = PyClassObject::new_simple(intern("PositionalInit"));
    class.set_attr(intern("__init__"), Value::object_ptr(init_ptr as *const ()));
    let class = register_test_class(class);

    let instance = super::invoke_callable_value(
        &mut vm,
        Value::object_ptr(Arc::as_ptr(&class) as *const ()),
        &[Value::int(17).unwrap()],
    )
    .expect("heap class call should succeed");

    let instance_ptr = instance
        .as_object_ptr()
        .expect("heap class call should return an instance");
    assert_eq!(super::extract_type_id(instance_ptr), class.class_type_id());

    unsafe {
        drop(Box::from_raw(init_ptr));
    }
}

#[test]
fn test_invoke_callable_value_instantiates_int_subclass_with_native_new() {
    let mut vm = VirtualMachine::new();
    let class = PyClassObject::new(intern("IntSubclass"), &[ClassId(TypeId::INT.raw())], |id| {
        Some(builtin_class_mro(class_id_to_type_id(id)).into())
    })
    .expect("int subclass mro should be valid");
    let class = register_test_class(class);

    let instance = super::invoke_callable_value(
        &mut vm,
        Value::object_ptr(Arc::as_ptr(&class) as *const ()),
        &[Value::int(42).unwrap()],
    )
    .expect("int subclass call should succeed without object.__init__ seeing constructor args");

    let instance_ptr = instance
        .as_object_ptr()
        .expect("int subclass call should return an instance");
    assert_eq!(super::extract_type_id(instance_ptr), class.class_type_id());
    let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };
    assert_eq!(
        shaped
            .int_backing()
            .expect("int backing should exist")
            .to_string(),
        "42"
    );

    unsafe {
        drop(Box::from_raw(instance_ptr as *mut ShapedObject));
    }
}

#[test]
fn test_invoke_callable_value_instantiates_bytes_subclass_with_native_new() {
    let mut vm = VirtualMachine::new();
    let class = PyClassObject::new(
        intern("BytesSubclass"),
        &[ClassId(TypeId::BYTES.raw())],
        |id| Some(builtin_class_mro(class_id_to_type_id(id)).into()),
    )
    .expect("bytes subclass mro should be valid");
    let class = register_test_class(class);
    let source =
        Value::object_ptr(Box::into_raw(Box::new(BytesObject::from_slice(b"auth"))) as *const ());

    let instance = super::invoke_callable_value(
        &mut vm,
        Value::object_ptr(Arc::as_ptr(&class) as *const ()),
        &[source],
    )
    .expect("bytes subclass call should succeed");

    let instance_ptr = instance
        .as_object_ptr()
        .expect("bytes subclass call should return an instance");
    assert_eq!(super::extract_type_id(instance_ptr), class.class_type_id());
    let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };
    assert_eq!(
        shaped
            .bytes_backing()
            .expect("bytes backing should exist")
            .as_bytes(),
        b"auth"
    );

    unsafe {
        drop(Box::from_raw(
            source.as_object_ptr().unwrap() as *mut BytesObject
        ));
        drop(Box::from_raw(instance_ptr as *mut ShapedObject));
    }
}

#[test]
fn test_instantiate_user_defined_dict_subclass_allocates_native_dict_backing() {
    let mut vm = VirtualMachine::new();
    let class = PyClassObject::new(
        intern("DictSubclass"),
        &[ClassId(TypeId::DICT.raw())],
        |id| {
            Some(
                builtin_class_mro(class_id_to_type_id(id))
                    .into_iter()
                    .collect(),
            )
        },
    )
    .expect("dict subclass should build");
    let class = register_test_class(class);

    let instance = super::instantiate_user_defined_class_from_values(&mut vm, class.as_ref(), &[])
        .expect("dict subclass instantiation should succeed");
    let instance_ptr = instance
        .as_object_ptr()
        .expect("instantiation should return a heap instance");
    let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };

    assert!(shaped.has_dict_backing());
}

#[test]
fn test_instantiate_user_defined_list_subclass_allocates_native_list_backing() {
    let mut vm = VirtualMachine::new();
    let class = PyClassObject::new(
        intern("ListSubclass"),
        &[ClassId(TypeId::LIST.raw())],
        |id| {
            Some(
                builtin_class_mro(class_id_to_type_id(id))
                    .into_iter()
                    .collect(),
            )
        },
    )
    .expect("list subclass should build");
    let class = register_test_class(class);

    let instance = super::instantiate_user_defined_class_from_values(&mut vm, class.as_ref(), &[])
        .expect("list subclass instantiation should succeed");
    let instance_ptr = instance
        .as_object_ptr()
        .expect("instantiation should return a heap instance");
    let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };

    assert!(shaped.has_list_backing());
}

#[test]
fn test_invoke_callable_value_with_keywords_supports_sorted_reverse() {
    let mut vm = VirtualMachine::new();
    let sorted_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
        Arc::from("sorted"),
        crate::builtins::builtin_sorted,
    )));
    let iterable_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(3).unwrap(),
        Value::int(2).unwrap(),
    ])));

    let result = invoke_callable_value_with_keywords(
        &mut vm,
        Value::object_ptr(sorted_ptr as *const ()),
        &[Value::object_ptr(iterable_ptr as *const ())],
        &[("reverse", Value::bool(true))],
    )
    .expect("sorted(reverse=...) should succeed");

    let result_ptr = result
        .as_object_ptr()
        .expect("sorted should return a list object");
    let result_list = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(
        result_list.as_slice(),
        &[
            Value::int(3).unwrap(),
            Value::int(2).unwrap(),
            Value::int(1).unwrap()
        ]
    );

    unsafe {
        drop(Box::from_raw(iterable_ptr));
        drop(Box::from_raw(sorted_ptr));
        drop(Box::from_raw(result_ptr as *mut ListObject));
    }
}

#[test]
fn test_invoke_callable_value_with_keywords_forwards_staticmethod_keywords() {
    let mut vm = VirtualMachine::new();
    let sorted_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
        Arc::from("sorted"),
        crate::builtins::builtin_sorted,
    )));
    let staticmethod_ptr = Box::into_raw(Box::new(StaticMethodDescriptor::new(Value::object_ptr(
        sorted_ptr as *const (),
    ))));
    let iterable_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(3).unwrap(),
        Value::int(2).unwrap(),
    ])));

    let result = invoke_callable_value_with_keywords(
        &mut vm,
        Value::object_ptr(staticmethod_ptr as *const ()),
        &[Value::object_ptr(iterable_ptr as *const ())],
        &[("reverse", Value::bool(true))],
    )
    .expect("staticmethod should forward keyword calls");

    let result_ptr = result
        .as_object_ptr()
        .expect("sorted should return a list object");
    let result_list = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(
        result_list.as_slice(),
        &[
            Value::int(3).unwrap(),
            Value::int(2).unwrap(),
            Value::int(1).unwrap()
        ]
    );

    unsafe {
        drop(Box::from_raw(iterable_ptr));
        drop(Box::from_raw(staticmethod_ptr));
        drop(Box::from_raw(sorted_ptr));
        drop(Box::from_raw(result_ptr as *mut ListObject));
    }
}

#[test]
fn test_invoke_callable_value_with_keywords_rejects_unexpected_builtin_keyword() {
    let mut vm = VirtualMachine::new();
    let sorted_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
        Arc::from("sorted"),
        crate::builtins::builtin_sorted,
    )));
    let iterable_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])));

    let err = invoke_callable_value_with_keywords(
        &mut vm,
        Value::object_ptr(sorted_ptr as *const ()),
        &[Value::object_ptr(iterable_ptr as *const ())],
        &[("bogus", Value::bool(true))],
    )
    .expect_err("unexpected builtin keyword should fail");
    assert!(
        err.to_string()
            .contains("sorted() got an unexpected keyword argument 'bogus'"),
        "unexpected error: {err}"
    );

    unsafe {
        drop(Box::from_raw(iterable_ptr));
        drop(Box::from_raw(sorted_ptr));
    }
}

#[test]
fn test_invoke_callable_value_with_keywords_supports_import_fromlist() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg").join("__init__.py"), "");
    write_file(&temp.path.join("pkg").join("child.py"), "VALUE = 1\n");

    let mut vm = VirtualMachine::new();
    vm.import_resolver
        .add_search_path(Arc::from(temp.path.to_string_lossy().into_owned()));
    let import_builtin = vm
        .builtins
        .get("__import__")
        .expect("__import__ builtin should be registered");
    let fromlist_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[Value::string(intern(
        "VALUE",
    ))])));

    let value = invoke_callable_value_with_keywords(
        &mut vm,
        import_builtin,
        &[Value::string(intern("pkg.child"))],
        &[("fromlist", Value::object_ptr(fromlist_ptr as *const ()))],
    )
    .expect("__import__(..., fromlist=...) should succeed");

    let module_ptr = value
        .as_object_ptr()
        .expect("__import__ should return a module object");
    let module = unsafe { &*(module_ptr as *const crate::import::ModuleObject) };
    assert_eq!(module.name(), "pkg.child");
    assert_eq!(
        module.get_attr("VALUE").and_then(|value| value.as_int()),
        Some(1)
    );

    unsafe {
        drop(Box::from_raw(fromlist_ptr));
    }
}

#[test]
fn test_invoke_callable_value_with_keywords_rejects_duplicate_import_fromlist() {
    let mut vm = VirtualMachine::new();
    let import_builtin = vm
        .builtins
        .get("__import__")
        .expect("__import__ builtin should be registered");

    let err = invoke_callable_value_with_keywords(
        &mut vm,
        import_builtin,
        &[
            Value::string(intern("pkg.child")),
            Value::none(),
            Value::none(),
            Value::none(),
        ],
        &[("fromlist", Value::none())],
    )
    .expect_err("duplicate __import__ fromlist keyword should fail");
    assert!(
        err.to_string()
            .contains("__import__() got multiple values for argument 'fromlist'"),
        "unexpected error: {err}"
    );
}
