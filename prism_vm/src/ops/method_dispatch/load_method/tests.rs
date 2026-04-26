use super::*;
use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::import::ModuleObject;
use crate::ops::method_dispatch::call_method::call_method;
use prism_code::CodeObject;
use prism_code::{Instruction, Opcode, Register};
use prism_core::intern::intern;
use prism_runtime::object::class::ClassFlags;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, builtin_class_mro, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::views::CodeObjectView;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::function::FunctionObject;
use std::sync::Arc;

fn register_test_class(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(TypeId::from_raw(class_id.0));
    }

    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

fn make_test_function_value(name: &str) -> (*mut FunctionObject, Value) {
    let mut code = CodeObject::new(name, "<test>");
    code.register_count = 8;
    let func = Box::new(FunctionObject::new(
        Arc::new(code),
        Arc::from(name),
        None,
        None,
    ));
    let ptr = Box::into_raw(func);
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn builtin_arg_count(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
    Ok(Value::int(args.len() as i64).unwrap())
}

fn make_test_builtin_value(name: &str) -> (*mut BuiltinFunctionObject, Value) {
    let builtin = Box::new(BuiltinFunctionObject::new(
        Arc::from(name),
        builtin_arg_count,
    ));
    let ptr = Box::into_raw(builtin);
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn make_test_bound_builtin_value(
    name: &str,
    bound_self: Value,
) -> (*mut BuiltinFunctionObject, Value) {
    let builtin = Box::new(BuiltinFunctionObject::new_bound(
        Arc::from(name),
        builtin_arg_count,
        bound_self,
    ));
    let ptr = Box::into_raw(builtin);
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn register_dict_subclass(name: &str) -> Arc<PyClassObject> {
    let class = PyClassObject::new(intern(name), &[ClassId(TypeId::DICT.raw())], |id| {
        (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
            builtin_class_mro(class_id_to_type_id(id))
                .into_iter()
                .collect()
        })
    })
    .expect("dict subclass should build");
    register_test_class(class)
}

fn vm_with_names(names: &[&str]) -> VirtualMachine {
    let mut code = CodeObject::new("test_load_method", "<test>");
    code.names = names
        .iter()
        .map(|name| Arc::<str>::from(*name))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0).expect("frame push failed");
    vm
}

fn names_with_extended_method() -> Vec<Arc<str>> {
    (0..=0x0123)
        .map(|index| {
            if index == 0x0123 {
                Arc::from("extended")
            } else {
                Arc::from(format!("unused_{index}"))
            }
        })
        .collect()
}

#[test]
fn test_get_primitive_type_id() {
    assert_eq!(get_primitive_type_id(Value::none()), TypeId::NONE);
    assert_eq!(get_primitive_type_id(Value::bool(true)), TypeId::BOOL);
    assert_eq!(get_primitive_type_id(Value::int_unchecked(42)), TypeId::INT);
    assert_eq!(
        get_primitive_type_id(Value::string(prism_core::intern::intern("Path"))),
        TypeId::STR
    );
}

#[test]
fn test_resolve_list_method_known() {
    let result = resolve_list_method("append");
    assert!(result.is_ok());
}

#[test]
fn test_resolve_list_method_unknown() {
    // Unknown methods should return plain attribute error
    let result = resolve_list_method("foobar");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("foobar"));
}

#[test]
fn test_resolve_deque_method_known() {
    let result = resolve_deque_method("append");
    assert!(result.is_ok());
}

#[test]
fn test_resolve_dict_method_known() {
    let result = resolve_dict_method("keys");
    assert!(result.is_ok());
}

#[test]
fn test_resolve_bytes_method_known() {
    let result = resolve_bytes_method("decode");
    assert!(result.is_ok());
}

#[test]
fn test_resolve_str_method_known() {
    let result = resolve_str_method("upper");
    assert!(result.is_ok());
}

#[test]
fn test_resolve_primitive_method_inherits_object_new_for_none() {
    let vm = vm_with_names(&[]);

    let cached = resolve_primitive_method(Value::none(), TypeId::NONE, "__new__")
        .expect("None should inherit object.__new__ for method calls");
    let method_ptr = cached
        .method
        .as_object_ptr()
        .expect("object.__new__ should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert!(cached.is_descriptor);
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "object.__new__");
    assert!(builtin.bound_self().is_none());
}

#[test]
fn test_resolve_int_method_known() {
    let result = resolve_int_method("to_bytes");
    assert!(result.is_ok());
}

#[test]
fn test_resolve_generator_close_method_known() {
    let result = resolve_generator_method("close");
    assert!(result.is_ok());
}

#[test]
fn test_resolve_generator_throw_method_known() {
    let result = resolve_generator_method("throw");
    assert!(result.is_ok());
}

#[test]
fn test_load_method_reads_module_attributes_without_binding_self() {
    let mut vm = vm_with_names(&["iskeyword"]);
    let module = Arc::new(ModuleObject::new("keyword"));
    let builtin_len = vm.builtins.get("len").expect("len builtin should exist");
    module.set_attr("iskeyword", builtin_len);
    vm.import_resolver.insert_module("keyword", module.clone());
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(Arc::as_ptr(&module) as *const ()));

    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2), builtin_len);
    assert!(vm.current_frame().get_reg(3).is_none());
}

#[test]
fn test_load_method_consumes_extended_name_index() {
    let primary = Instruction::new(Opcode::LoadMethod, 2, 1, u8::MAX);
    let extension = Instruction::op_di(Opcode::AttrName, Register::new(0), 0x0123);
    let mut code = CodeObject::new("test_load_method", "<test>");
    code.names = names_with_extended_method().into_boxed_slice();
    code.instructions = vec![primary, extension].into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0).expect("frame push failed");
    vm.current_frame_mut().ip = 1;

    let module = Arc::new(ModuleObject::new("method_ext"));
    let builtin_len = vm.builtins.get("len").expect("len builtin should exist");
    module.set_attr("extended", builtin_len);
    vm.import_resolver
        .insert_module("method_ext", module.clone());
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(Arc::as_ptr(&module) as *const ()));

    assert!(matches!(
        load_method(&mut vm, primary),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(2), builtin_len);
    assert!(vm.current_frame().get_reg(3).is_none());
    assert_eq!(vm.current_frame().ip, 2);
}

#[test]
fn test_load_method_resolves_str_maketrans_without_implicit_self() {
    let mut vm = vm_with_names(&["maketrans"]);
    vm.current_frame_mut()
        .set_reg(1, Value::string(prism_core::intern::intern("seed")));

    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("str.maketrans should resolve to a builtin function");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "str.maketrans");
    assert!(builtin.bound_self().is_none());
    assert!(vm.current_frame().get_reg(3).is_none());
}

#[test]
fn test_load_method_resolves_bytes_decode_with_implicit_self() {
    let mut vm = vm_with_names(&["decode"]);
    let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"abc")));
    let bytes_value = Value::object_ptr(bytes_ptr as *const ());
    vm.current_frame_mut().set_reg(1, bytes_value);

    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("bytes.decode should resolve to a builtin function");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "bytes.decode");
    assert_eq!(vm.current_frame().get_reg(3), bytes_value);

    unsafe {
        drop(Box::from_raw(bytes_ptr));
    }
}

#[test]
fn test_load_method_resolves_none_new_without_implicit_self() {
    let mut vm = vm_with_names(&["__new__"]);
    vm.current_frame_mut().set_reg(1, Value::none());

    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("None.__new__ should resolve to a builtin function");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "object.__new__");
    assert!(builtin.bound_self().is_none());
    assert!(vm.current_frame().get_reg(3).is_none());
}

#[test]
fn test_load_method_resolves_code_positions_without_implicit_self() {
    let mut vm = vm_with_names(&["co_positions"]);
    let code_view = CodeObjectView::new(Arc::new(CodeObject::new("trace_target", "<test>")));
    let code_value = Value::object_ptr(Box::into_raw(Box::new(code_view)) as *const ());
    vm.current_frame_mut().set_reg(1, code_value);

    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("code.co_positions should resolve to a builtin function");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "code.co_positions");
    assert_eq!(builtin.bound_self(), Some(code_value));
    assert!(vm.current_frame().get_reg(3).is_none());
}

#[test]
fn test_resolve_user_defined_method_inherits_from_parent() {
    let (func_ptr, func_value) = make_test_function_value("method");

    let mut parent = PyClassObject::new_simple(intern("Parent"));
    parent.set_attr(intern("method"), func_value);
    let parent = register_test_class(parent);

    let child = PyClassObject::new(intern("Child"), &[parent.class_id()], |id| {
        (id == parent.class_id()).then(|| parent.mro().iter().copied().collect())
    })
    .expect("child class should build");
    let child = register_test_class(child);
    let instance = ShapedObject::new(child.class_type_id(), child.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let cached = resolve_user_defined_method(instance_value, child.class_type_id(), "method")
        .expect("expected inherited method lookup");
    assert_eq!(cached.method, func_value);
    assert!(!cached.is_descriptor);

    unsafe {
        drop(Box::from_raw(instance_ptr));
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_resolve_user_defined_non_callable_uses_unbound_marker() {
    let class = PyClassObject::new_simple(intern("AttrHolder"));
    class.set_attr(intern("value"), Value::int_unchecked(42));
    let class = register_test_class(class);
    let instance = ShapedObject::new(class.class_type_id(), shape_registry().empty_shape());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let cached = resolve_user_defined_method(instance_value, class.class_type_id(), "value")
        .expect("expected value");
    assert_eq!(cached.method, Value::int_unchecked(42));
    assert!(cached.is_descriptor);

    unsafe {
        drop(Box::from_raw(instance_ptr));
    }
}

#[test]
fn test_resolve_user_defined_method_inherits_builtin_dict_setdefault_for_heap_class() {
    let class = register_dict_subclass("DictSubclassSetDefault");
    let instance =
        ShapedObject::new_dict_backed(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let cached = resolve_user_defined_method(instance_value, class.class_type_id(), "setdefault")
        .expect("heap dict subclass should inherit dict.setdefault");
    let method_ptr = cached
        .method
        .as_object_ptr()
        .expect("dict.setdefault should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "dict.setdefault");
    assert!(!cached.is_descriptor);

    unsafe {
        drop(Box::from_raw(instance_ptr));
    }
}

#[test]
fn test_resolve_user_defined_method_inherits_builtin_object_init_for_heap_class() {
    let class = register_test_class(PyClassObject::new_simple(intern("InitCarrier")));
    let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let cached = resolve_user_defined_method(instance_value, class.class_type_id(), "__init__")
        .expect("heap instances should inherit object.__init__");
    let method_ptr = cached
        .method
        .as_object_ptr()
        .expect("object.__init__ should resolve to a bound builtin");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "object.__init__");
    assert!(cached.is_descriptor);

    unsafe {
        drop(Box::from_raw(instance_ptr));
    }
}

#[test]
fn test_load_method_resolves_builtin_dict_setdefault_for_heap_class() {
    let class = register_dict_subclass("DictSubclassLoadMethod");
    let instance =
        ShapedObject::new_dict_backed(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));

    let mut vm = vm_with_names(&["setdefault"]);
    let instance_value = Value::object_ptr(instance_ptr as *const ());
    vm.current_frame_mut().set_reg(1, instance_value);

    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("dict.setdefault should resolve to a builtin function");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "dict.setdefault");
    assert_eq!(vm.current_frame().get_reg(3), instance_value);

    unsafe {
        drop(Box::from_raw(instance_ptr));
    }
}

#[test]
fn test_load_method_resolves_inherited_object_init_for_heap_class() {
    let class = register_test_class(PyClassObject::new_simple(intern("InitLoadCarrier")));
    let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));

    let mut vm = vm_with_names(&["__init__"]);
    let instance_value = Value::object_ptr(instance_ptr as *const ());
    vm.current_frame_mut().set_reg(1, instance_value);

    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("object.__init__ should resolve to a bound builtin");
    let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(builtin.name(), "object.__init__");
    assert!(vm.current_frame().get_reg(3).is_none());

    unsafe {
        drop(Box::from_raw(instance_ptr));
    }
}

#[test]
fn test_resolve_type_object_method_inherits_builtin_object_ne_for_heap_class() {
    let class = register_test_class(PyClassObject::new_simple(intern("HeapType")));
    let class_value = Value::object_ptr(Arc::as_ptr(&class) as *const ());

    let cached = resolve_type_object_method(class_value, "__ne__")
        .expect("heap class should inherit object.__ne__");
    let method_ptr = cached
        .method
        .as_object_ptr()
        .expect("object.__ne__ should be heap allocated");
    let header = unsafe { &*(method_ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    assert!(cached.is_descriptor);
}

#[test]
fn test_load_method_refreshes_cached_heap_method_after_class_version_change() {
    method_cache().clear();

    let method_name = intern("method");
    let (first_ptr, first_value) = make_test_function_value("method_v1");
    let class = PyClassObject::new_simple(intern("VersionedCarrier"));
    class.set_attr(method_name.clone(), first_value);
    let class = register_test_class(class);

    let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let mut vm = vm_with_names(&["method"]);
    let method_name_ptr = vm.current_frame().code.names[0].as_ptr() as u64;
    vm.current_frame_mut().set_reg(1, instance_value);
    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );

    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2), first_value);
    assert_eq!(vm.current_frame().get_reg(3), instance_value);
    let first_cache_version = method_cache_version(class.class_type_id());
    let cached = method_cache()
        .get(class.class_type_id(), method_name_ptr, first_cache_version)
        .expect("heap method should be cached at the current class version");
    assert_eq!(cached.method, first_value);

    let (second_ptr, second_value) = make_test_function_value("method_v2");
    class.set_attr(method_name, second_value);

    vm.current_frame_mut().set_reg(2, Value::none());
    vm.current_frame_mut().set_reg(3, Value::none());
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2), second_value);
    assert_eq!(vm.current_frame().get_reg(3), instance_value);
    let second_cache_version = method_cache_version(class.class_type_id());
    assert_ne!(second_cache_version, first_cache_version);
    assert!(
        method_cache()
            .get(class.class_type_id(), method_name_ptr, first_cache_version)
            .is_none(),
        "stale cache entries must not be returned after class version changes"
    );
    let cached = method_cache()
        .get(class.class_type_id(), method_name_ptr, second_cache_version)
        .expect("heap method should be refreshed at the new class version");
    assert_eq!(cached.method, second_value);

    unsafe {
        drop(Box::from_raw(instance_ptr));
        drop(Box::from_raw(first_ptr));
        drop(Box::from_raw(second_ptr));
    }

    method_cache().clear();
}

#[test]
fn test_load_method_does_not_share_type_object_cache_between_heap_classes() {
    method_cache().clear();

    let method_name = intern("method");
    let (first_ptr, first_value) = make_test_function_value("type_method_v1");
    let (second_ptr, second_value) = make_test_function_value("type_method_v2");

    let mut first = PyClassObject::new_simple(intern("TypeLoadOne"));
    first.set_attr(method_name.clone(), first_value);
    let first = register_test_class(first);

    let mut second = PyClassObject::new_simple(intern("TypeLoadTwo"));
    second.set_attr(method_name.clone(), second_value);
    let second = register_test_class(second);

    let mut vm = vm_with_names(&["method"]);
    let method_name_ptr = vm.current_frame().code.names[0].as_ptr() as u64;
    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(Arc::as_ptr(&first) as *const ()));
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2), first_value);
    assert!(vm.current_frame().get_reg(3).is_none());

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(Arc::as_ptr(&second) as *const ()));
    vm.current_frame_mut().set_reg(2, Value::none());
    vm.current_frame_mut().set_reg(3, Value::none());
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2), second_value);
    assert!(vm.current_frame().get_reg(3).is_none());
    assert!(
        method_cache()
            .get(
                TypeId::TYPE,
                method_name_ptr,
                method_cache_version(TypeId::TYPE)
            )
            .is_none(),
        "type-object method loads must bypass the instance method cache"
    );

    unsafe {
        drop(Box::from_raw(first_ptr));
        drop(Box::from_raw(second_ptr));
    }

    method_cache().clear();
}

#[test]
fn test_resolve_special_method_on_plain_object_uses_builtin_object_methods() {
    let registry = shape_registry();
    let object = ShapedObject::with_empty_shape(registry.empty_shape());
    let object_ptr = Box::into_raw(Box::new(object));
    let object_value = Value::object_ptr(object_ptr as *const ());

    let bound = resolve_special_method(object_value, "__ne__")
        .expect("plain object special lookup should inherit object.__ne__");
    let method_ptr = bound
        .callable
        .as_object_ptr()
        .expect("object.__ne__ should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };

    assert_eq!(builtin.name(), "object.__ne__");
    assert_eq!(bound.implicit_self, Some(object_value));

    unsafe {
        drop(Box::from_raw(object_ptr));
    }
}

#[test]
fn test_resolve_special_method_bypasses_instance_attributes_for_heap_types() {
    let (class_func_ptr, class_func_value) = make_test_function_value("class_len");
    let (instance_func_ptr, instance_func_value) = make_test_function_value("instance_len");

    let mut class = PyClassObject::new_simple(intern("SpecialLookupCarrier"));
    class.set_attr(intern("__len__"), class_func_value);
    let class = register_test_class(class);

    let mut instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    instance.set_property(intern("__len__"), instance_func_value, shape_registry());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let bound = resolve_special_method(instance_value, "__len__")
        .expect("special lookup should resolve through the type");
    assert_eq!(bound.callable, class_func_value);
    assert_eq!(bound.implicit_self, Some(instance_value));

    unsafe {
        drop(Box::from_raw(instance_ptr));
        drop(Box::from_raw(class_func_ptr));
        drop(Box::from_raw(instance_func_ptr));
    }
}

#[test]
fn test_resolve_object_instance_method_returns_callable_instance_property_unbound() {
    let (func_ptr, func_value) = make_test_function_value("instance_method");
    let registry = shape_registry();
    let mut object = ShapedObject::with_empty_shape(registry.empty_shape());
    object.set_property(intern("instance_method"), func_value, registry);
    let object_ptr = Box::into_raw(Box::new(object));
    let object_value = Value::object_ptr(object_ptr as *const ());

    let cached = resolve_object_instance_method(object_value, "instance_method")
        .expect("instance callable should resolve");
    assert_eq!(cached.method, func_value);
    assert!(cached.is_descriptor);

    unsafe {
        drop(Box::from_raw(object_ptr));
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_load_method_prefers_user_defined_instance_callable_without_binding_self() {
    let (func_ptr, func_value) = make_test_function_value("instance_callable");

    let class = register_test_class(PyClassObject::new_simple(intern("CallableHolder")));
    let mut instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    instance.set_property(intern("instance_callable"), func_value, shape_registry());
    let instance_ptr = Box::into_raw(Box::new(instance));

    let mut vm = vm_with_names(&["instance_callable"]);
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(instance_ptr as *const ()));

    let inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2), func_value);
    assert!(vm.current_frame().get_reg(3).is_none());

    unsafe {
        drop(Box::from_raw(instance_ptr));
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_load_method_leaves_plain_builtin_class_attributes_unbound() {
    let (builtin_ptr, builtin_value) = make_test_builtin_value("count_args");

    let mut class = PyClassObject::new_simple(intern("PlainBuiltinLoadMethod"));
    class.set_attr(intern("helper"), builtin_value);
    let class = register_test_class(class);

    let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let mut vm = vm_with_names(&["helper"]);
    vm.current_frame_mut().set_reg(1, instance_value);

    let load_inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(
        load_method(&mut vm, load_inst),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(2), builtin_value);
    assert!(vm.current_frame().get_reg(3).is_none());

    vm.current_frame_mut().set_reg(4, Value::int(7).unwrap());
    let call_inst = Instruction::op_dss(
        Opcode::CallMethod,
        Register::new(5),
        Register::new(2),
        Register::new(1),
    );
    assert!(matches!(
        call_method(&mut vm, call_inst),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(5).as_int(), Some(1));

    unsafe {
        drop(Box::from_raw(instance_ptr));
        drop(Box::from_raw(builtin_ptr));
    }
}

#[test]
fn test_load_method_preserves_prebound_builtin_class_attributes() {
    let (builtin_ptr, builtin_value) =
        make_test_bound_builtin_value("count_args", Value::int(99).unwrap());

    let mut class = PyClassObject::new_simple(intern("BoundBuiltinLoadMethod"));
    class.set_attr(intern("helper"), builtin_value);
    let class = register_test_class(class);

    let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let mut vm = vm_with_names(&["helper"]);
    vm.current_frame_mut().set_reg(1, instance_value);

    let load_inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(
        load_method(&mut vm, load_inst),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(2), builtin_value);
    assert!(vm.current_frame().get_reg(3).is_none());

    let call_inst = Instruction::op_dss(
        Opcode::CallMethod,
        Register::new(5),
        Register::new(2),
        Register::new(0),
    );
    assert!(matches!(
        call_method(&mut vm, call_inst),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(5).as_int(), Some(1));

    unsafe {
        drop(Box::from_raw(instance_ptr));
        drop(Box::from_raw(builtin_ptr));
    }
}

#[test]
fn test_load_method_binds_builtin_class_attributes_for_native_heap_classes() {
    let (builtin_ptr, builtin_value) = make_test_builtin_value("count_args");

    let mut class = PyClassObject::new_simple(intern("NativeBuiltinLoadMethod"));
    class.add_flags(ClassFlags::NATIVE_HEAPTYPE);
    class.set_attr(intern("helper"), builtin_value);
    let class = register_test_class(class);

    let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());

    let mut vm = vm_with_names(&["helper"]);
    vm.current_frame_mut().set_reg(1, instance_value);

    let load_inst = Instruction::op_dss(
        Opcode::LoadMethod,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(
        load_method(&mut vm, load_inst),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(2), builtin_value);
    assert_eq!(vm.current_frame().get_reg(3), instance_value);

    let call_inst = Instruction::op_dss(
        Opcode::CallMethod,
        Register::new(5),
        Register::new(2),
        Register::new(0),
    );
    assert!(matches!(
        call_method(&mut vm, call_inst),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(5).as_int(), Some(1));

    unsafe {
        drop(Box::from_raw(instance_ptr));
        drop(Box::from_raw(builtin_ptr));
    }
}
