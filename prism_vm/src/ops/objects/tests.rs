use super::*;
use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::frame::ClosureEnv;
use crate::import::ModuleObject;
use prism_code::{CodeFlags, CodeObject, Instruction, LineTableEntry, Opcode, Register};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::{
    BoundMethod, ClassMethodDescriptor, PropertyDescriptor, StaticMethodDescriptor,
};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, builtin_class_mro, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{CodeObjectView, FrameViewObject, TracebackViewObject};
use prism_runtime::types::Cell;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
use std::sync::Arc;

fn vm_with_frame() -> VirtualMachine {
    let mut vm = VirtualMachine::new();
    let code = Arc::new(CodeObject::new("test_len", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");
    vm
}

fn exhaust_nursery(vm: &VirtualMachine) {
    while vm.allocator().alloc(DictObject::new()).is_some() {}
}

fn boxed_value<T>(obj: T) -> (Value, *mut T) {
    let ptr = Box::into_raw(Box::new(obj));
    (Value::object_ptr(ptr as *const ()), ptr)
}

unsafe fn drop_boxed<T>(ptr: *mut T) {
    drop(unsafe { Box::from_raw(ptr) });
}

fn vm_with_name_arcs(names: Vec<Arc<str>>) -> VirtualMachine {
    let mut code = CodeObject::new("test_attrs", "<test>");
    code.names = names.into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0).expect("frame push failed");
    vm
}

fn vm_with_names(names: &[&str]) -> VirtualMachine {
    vm_with_name_arcs(names.iter().map(|name| Arc::<str>::from(*name)).collect())
}

fn names_with_extended_attr() -> Vec<Arc<str>> {
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

fn class_value(class: PyClassObject) -> (Value, *const PyClassObject) {
    let class = Arc::new(class);
    let ptr = Arc::into_raw(class);
    (Value::object_ptr(ptr as *const ()), ptr)
}

unsafe fn drop_class(ptr: *const PyClassObject) {
    drop(unsafe { Arc::from_raw(ptr) });
}

fn register_test_class(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(prism_runtime::object::type_obj::TypeId::from_raw(
            class_id.0,
        ));
    }

    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

fn make_test_function_value(name: &str) -> (*mut FunctionObject, Value) {
    make_test_function_value_with_closure(name, None)
}

fn make_test_function_value_with_closure(
    name: &str,
    closure: Option<Arc<ClosureEnv>>,
) -> (*mut FunctionObject, Value) {
    let mut code = CodeObject::new(name, "<test>");
    code.register_count = 8;
    let func = Box::new(FunctionObject::new(
        Arc::new(code),
        Arc::from(name),
        None,
        closure,
    ));
    let ptr = Box::into_raw(func);
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn value_as_str(value: Value) -> String {
    prism_runtime::types::string::value_as_string_ref(value)
        .expect("value should be a Python string")
        .as_str()
        .to_string()
}

fn instance_value(class: &Arc<PyClassObject>) -> (*mut ShapedObject, Value) {
    let ptr = Box::into_raw(Box::new(ShapedObject::new(
        class.class_type_id(),
        class.instance_shape().clone(),
    )));
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn dict_backed_instance_value(class: &Arc<PyClassObject>) -> (*mut ShapedObject, Value) {
    let ptr = Box::into_raw(Box::new(ShapedObject::new_dict_backed(
        class.class_type_id(),
        class.instance_shape().clone(),
    )));
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn register_dict_subclass(name: &str) -> Arc<PyClassObject> {
    let class = PyClassObject::new(intern(name), &[ClassId(TypeId::DICT.raw())], |id| {
        if id.0 < TypeId::FIRST_USER_TYPE {
            Some(
                builtin_class_mro(class_id_to_type_id(id))
                    .into_iter()
                    .collect(),
            )
        } else {
            None
        }
    })
    .expect("dict subclass should build");
    register_test_class(class)
}

fn property_storage_getter(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
    if args.len() != 1 {
        return Err(crate::builtins::BuiltinError::TypeError(format!(
            "getter expected 1 argument, got {}",
            args.len()
        )));
    }
    let ptr = args[0].as_object_ptr().ok_or_else(|| {
        crate::builtins::BuiltinError::TypeError("getter requires object receiver".to_string())
    })?;
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    shaped.get_property("_value").ok_or_else(|| {
        crate::builtins::BuiltinError::AttributeError("_value missing".to_string())
    })
}

fn metaclass_property_getter(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
    if args.len() != 1 {
        return Err(crate::builtins::BuiltinError::TypeError(format!(
            "getter expected 1 argument, got {}",
            args.len()
        )));
    }
    let ptr = args[0].as_object_ptr().ok_or_else(|| {
        crate::builtins::BuiltinError::TypeError("getter requires class receiver".to_string())
    })?;
    if extract_type_id(ptr) != TypeId::TYPE {
        return Err(crate::builtins::BuiltinError::TypeError(format!(
            "getter requires class receiver, got {}",
            extract_type_id(ptr).name()
        )));
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    Ok(Value::string(class.name().clone()))
}

fn property_storage_setter(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
    if args.len() != 2 {
        return Err(crate::builtins::BuiltinError::TypeError(format!(
            "setter expected 2 arguments, got {}",
            args.len()
        )));
    }
    let ptr = args[0].as_object_ptr().ok_or_else(|| {
        crate::builtins::BuiltinError::TypeError("setter requires object receiver".to_string())
    })?;
    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
    shaped.set_property(intern("_value"), args[1], shape_registry());
    Ok(Value::none())
}

fn property_storage_deleter(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
    if args.len() != 1 {
        return Err(crate::builtins::BuiltinError::TypeError(format!(
            "deleter expected 1 argument, got {}",
            args.len()
        )));
    }
    let ptr = args[0].as_object_ptr().ok_or_else(|| {
        crate::builtins::BuiltinError::TypeError("deleter requires object receiver".to_string())
    })?;
    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
    shaped.delete_property("_value");
    Ok(Value::none())
}

fn builtin_function_value(
    name: &str,
    func: fn(&[Value]) -> Result<Value, crate::builtins::BuiltinError>,
) -> (*mut BuiltinFunctionObject, Value) {
    let builtin = Box::new(BuiltinFunctionObject::new(Arc::from(name), func));
    let ptr = Box::into_raw(builtin);
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn builtin_first_arg(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
    Ok(args.first().copied().unwrap_or_else(Value::none))
}

fn builtin_arg_count(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
    Ok(Value::int(args.len() as i64).expect("argument count should fit in Value::int"))
}

#[test]
fn test_extract_type_id() {
    // Create a list and verify TypeId extraction
    let list = Box::new(ListObject::new());
    let ptr = Box::into_raw(list) as *const ();

    let type_id = extract_type_id(ptr);
    assert_eq!(type_id, TypeId::LIST);

    // Clean up
    unsafe {
        drop(Box::from_raw(ptr as *mut ListObject));
    }
}

#[test]
fn test_type_id_layout() {
    // Verify ObjectHeader layout is correct for JIT compatibility
    assert_eq!(std::mem::offset_of!(ObjectHeader, type_id), 0);
    assert_eq!(std::mem::size_of::<TypeId>(), 4);
    assert_eq!(std::mem::size_of::<ObjectHeader>(), 16);
}

#[test]
fn test_len_opcode_tagged_string() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut()
        .set_reg(1, Value::string(intern("hello")));

    let inst = Instruction::op_ds(Opcode::Len, Register::new(2), Register::new(1));
    assert!(matches!(len(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(5));
}

#[test]
fn test_len_opcode_set_object() {
    let mut vm = vm_with_frame();
    let set = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let (set_value, ptr) = boxed_value(set);
    vm.current_frame_mut().set_reg(1, set_value);

    let inst = Instruction::op_ds(Opcode::Len, Register::new(2), Register::new(1));
    assert!(matches!(len(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(3));

    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_opcode_string_object() {
    let mut vm = vm_with_frame();
    let (string_value, ptr) = boxed_value(StringObject::new("runtime"));
    vm.current_frame_mut().set_reg(1, string_value);

    let inst = Instruction::op_ds(Opcode::Len, Register::new(2), Register::new(1));
    assert!(matches!(len(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(7));

    unsafe { drop_boxed(ptr) };
}

#[test]
fn test_len_opcode_type_error_for_int() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(1, Value::int(42).unwrap());

    let inst = Instruction::op_ds(Opcode::Len, Register::new(2), Register::new(1));
    let flow = len(&mut vm, inst);
    assert!(matches!(flow, ControlFlow::Error(_)));
}

#[test]
fn test_get_attribute_value_reads_standalone_module_attributes() {
    let mut vm = vm_with_names(&[]);
    let (module_value, module_ptr) = boxed_value(ModuleObject::new("standalone"));
    let module = unsafe { &*module_ptr };
    module.set_attr("answer", Value::int(42).unwrap());

    let name = get_attribute_value(&mut vm, module_value, &intern("__name__"))
        .expect("standalone module __name__ should resolve");
    assert_eq!(value_as_str(name), "standalone");

    let answer = get_attribute_value(&mut vm, module_value, &intern("answer"))
        .expect("standalone module dynamic attributes should resolve");
    assert_eq!(answer.as_int(), Some(42));

    let dict_value = get_attribute_value(&mut vm, module_value, &intern("__dict__"))
        .expect("standalone module __dict__ should resolve");
    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("__dict__ should be a dict");
    let dict = unsafe { &*(dict_ptr as *const DictObject) };
    assert!(dict.get(Value::string(intern("__name__"))).is_some());
    assert_eq!(
        dict.get(Value::string(intern("answer")))
            .and_then(|value| value.as_int()),
        Some(42)
    );

    unsafe {
        drop_boxed(module_ptr);
    }
}

#[test]
fn test_set_and_delete_attribute_value_update_standalone_modules() {
    let mut vm = vm_with_names(&[]);
    let (module_value, module_ptr) = boxed_value(ModuleObject::new("standalone"));

    set_attribute_value(
        &mut vm,
        module_value,
        &intern("token"),
        Value::string(intern("ready")),
    )
    .expect("setattr should update standalone modules");
    let token = get_attribute_value(&mut vm, module_value, &intern("token"))
        .expect("new module attribute should be readable");
    assert_eq!(value_as_str(token), "ready");

    delete_attribute_value(&mut vm, module_value, &intern("token"))
        .expect("delattr should remove standalone module attributes");
    let err = get_attribute_value(&mut vm, module_value, &intern("token"))
        .expect_err("deleted module attribute should be missing");
    assert!(matches!(
        err.kind(),
        RuntimeErrorKind::AttributeError { .. }
    ));

    unsafe {
        drop_boxed(module_ptr);
    }
}

#[test]
fn test_get_attr_reads_class_attributes() {
    let mut vm = vm_with_names(&["field"]);
    let class = PyClassObject::new_simple(intern("Example"));
    class.set_attr(intern("field"), Value::int(99).unwrap());
    let (class_value, class_ptr) = class_value(class);
    vm.current_frame_mut().set_reg(1, class_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(99));

    unsafe { drop_class(class_ptr) };
}

#[test]
fn test_get_attr_consumes_extended_name_index() {
    let primary = Instruction::new(Opcode::GetAttr, 2, 1, EXTENDED_ATTR_NAME_SENTINEL);
    let extension = Instruction::op_di(Opcode::AttrName, Register::new(0), 0x0123);
    let mut code = CodeObject::new("test_attrs", "<test>");
    code.names = names_with_extended_attr().into_boxed_slice();
    code.instructions = vec![primary, extension].into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0).expect("frame push failed");
    vm.current_frame_mut().ip = 1;

    let class = PyClassObject::new_simple(intern("Example"));
    class.set_attr(intern("extended"), Value::int(123).unwrap());
    let (class_value, class_ptr) = class_value(class);
    vm.current_frame_mut().set_reg(1, class_value);

    assert!(matches!(get_attr(&mut vm, primary), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(123));
    assert_eq!(vm.current_frame().ip, 2);

    unsafe { drop_class(class_ptr) };
}

#[test]
fn test_get_attr_rejects_missing_extended_name_metadata() {
    let primary = Instruction::new(Opcode::GetAttr, 2, 1, EXTENDED_ATTR_NAME_SENTINEL);
    let mut vm = vm_with_names(&["field"]);

    assert!(matches!(
        get_attr(&mut vm, primary),
        ControlFlow::Error(RuntimeError { .. })
    ));
}

#[test]
fn test_get_attr_reads_inherited_class_attributes() {
    let mut parent = PyClassObject::new_simple(intern("Parent"));
    parent.set_attr(intern("field"), Value::int(123).unwrap());
    let parent = register_test_class(parent);

    let child = PyClassObject::new(intern("Child"), &[parent.class_id()], |id| {
        (id == parent.class_id()).then(|| parent.mro().iter().copied().collect())
    })
    .expect("child class should build");
    let child = register_test_class(child);

    let mut vm = vm_with_names(&["field"]);
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(Arc::as_ptr(&child) as *const ()));

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(123));
}

#[test]
fn test_get_attr_invokes_metaclass_data_descriptor_before_class_attr() {
    let (getter_ptr, getter_value) =
        builtin_function_value("metaclass_property.getter", metaclass_property_getter);
    let property_ptr = Box::into_raw(Box::new(PropertyDescriptor::new_getter(getter_value)));
    let property_value = Value::object_ptr(property_ptr as *const ());

    let metaclass = PyClassObject::new_simple(intern("MetaWithProperty"));
    metaclass.set_attr(intern("managed"), property_value);
    let metaclass = register_test_class(metaclass);

    let mut class = PyClassObject::new_simple(intern("ManagedByMeta"));
    class.set_attr(intern("managed"), Value::int(99).unwrap());
    class.set_metaclass(Value::object_ptr(Arc::as_ptr(&metaclass) as *const ()));
    let class = register_test_class(class);
    let class_value = Value::object_ptr(Arc::as_ptr(&class) as *const ());

    let mut vm = vm_with_names(&["managed"]);
    vm.current_frame_mut().set_reg(1, class_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(value_as_str(vm.current_frame().get_reg(2)), "ManagedByMeta");

    unsafe {
        drop(Box::from_raw(property_ptr));
        drop(Box::from_raw(getter_ptr));
    }
}

#[test]
fn test_get_attr_binds_inherited_instance_methods() {
    let (func_ptr, func_value) = make_test_function_value("method");

    let mut parent = PyClassObject::new_simple(intern("Parent"));
    parent.set_attr(intern("method"), func_value);
    let parent = register_test_class(parent);

    let child = PyClassObject::new(intern("Child"), &[parent.class_id()], |id| {
        (id == parent.class_id()).then(|| parent.mro().iter().copied().collect())
    })
    .expect("child class should build");
    let child = register_test_class(child);

    let (instance_ptr, instance_value) = instance_value(&child);

    let mut vm = vm_with_names(&["method"]);
    vm.current_frame_mut().set_reg(1, instance_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

    let bound_value = vm.current_frame().get_reg(2);
    let bound_ptr = bound_value
        .as_object_ptr()
        .expect("bound method should be heap allocated");
    assert_eq!(
        extract_type_id(bound_ptr),
        prism_runtime::object::type_obj::TypeId::METHOD
    );

    let bound = unsafe { &*(bound_ptr as *const BoundMethod) };
    assert_eq!(bound.function(), func_value);
    assert_eq!(bound.instance(), instance_value);

    unsafe {
        drop_boxed(instance_ptr);
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_get_attr_binds_builtin_list_methods_as_builtin_functions() {
    let mut vm = vm_with_names(&["append"]);
    let (list_value, list_ptr) = boxed_value(ListObject::new());
    vm.current_frame_mut().set_reg(1, list_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("builtin method should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "list.append");
    let result = builtin
        .call(&[Value::int(9).unwrap()])
        .expect("append should call");
    assert!(result.is_none());

    let list = unsafe { &*list_ptr.cast::<ListObject>() };
    assert_eq!(list.len(), 1);
    assert_eq!(list.as_slice()[0].as_int(), Some(9));
}

#[test]
fn test_get_attr_binds_builtin_frozenset_contains_method() {
    let mut vm = vm_with_names(&["__contains__"]);
    let mut frozenset =
        SetObject::from_slice(&[Value::int(3).unwrap(), Value::int(5).unwrap()]);
    frozenset.header.type_id = TypeId::FROZENSET;
    let (frozenset_value, frozenset_ptr) = boxed_value(frozenset);
    vm.current_frame_mut().set_reg(1, frozenset_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("builtin method should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "frozenset.__contains__");
    let present = builtin
        .call(&[Value::int(5).unwrap()])
        .expect("membership call should succeed");
    let missing = builtin
        .call(&[Value::int(9).unwrap()])
        .expect("membership call should succeed");
    assert_eq!(present.as_bool(), Some(true));
    assert_eq!(missing.as_bool(), Some(false));

    unsafe {
        drop_boxed(frozenset_ptr);
    }
}

#[test]
fn test_get_attribute_value_exposes_function_get_descriptor() {
    let mut vm = vm_with_names(&[]);
    let (func_ptr, func_value) = make_test_function_value("descriptor");

    assert_eq!(
        get_attribute_value(&mut vm, func_value, &intern("__class__"))
            .expect("function __class__ should resolve"),
        crate::builtins::builtin_type_object_for_type_id(TypeId::FUNCTION)
    );
    assert!(
        function_attr_exists(unsafe { &*func_ptr }, &intern("__class__")),
        "hasattr(function, '__class__') should match direct lookup"
    );

    let method = get_attribute_value(&mut vm, func_value, &intern("__get__"))
        .expect("function objects should expose __get__");
    let method_ptr = method
        .as_object_ptr()
        .expect("function.__get__ should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "function.__get__");
    assert_eq!(builtin.bound_self(), Some(func_value));

    assert_eq!(
        builtin
            .call(&[Value::none()])
            .expect("__get__(None) should return the underlying function"),
        func_value
    );

    let bound_method = builtin
        .call(&[Value::int(7).unwrap()])
        .expect("__get__(instance) should create a bound method");
    let bound_ptr = bound_method
        .as_object_ptr()
        .expect("bound method should be heap allocated");
    assert_eq!(extract_type_id(bound_ptr), TypeId::METHOD);

    let bound = unsafe { &*(bound_ptr as *const BoundMethod) };
    assert_eq!(bound.function(), func_value);
    assert_eq!(bound.instance(), Value::int(7).unwrap());

    unsafe {
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_get_attribute_value_exposes_bound_method_metadata() {
    let mut vm = vm_with_names(&[]);
    let (func_ptr, func_value) = make_test_function_value("metadata_method");
    let instance = Value::int(7).unwrap();
    let method_value = bind_instance_attribute(func_value, instance);

    assert_eq!(
        get_attribute_value(&mut vm, method_value, &intern("__self__"))
            .expect("__self__ should resolve"),
        instance
    );
    assert_eq!(
        get_attribute_value(&mut vm, method_value, &intern("__func__"))
            .expect("__func__ should resolve"),
        func_value
    );
    assert_eq!(
        get_attribute_value(&mut vm, method_value, &intern("__name__"))
            .expect("__name__ should resolve"),
        Value::string(intern("metadata_method"))
    );
    assert_eq!(
        get_attribute_value(&mut vm, method_value, &intern("__class__"))
            .expect("bound method __class__ should resolve"),
        crate::builtins::builtin_type_object_for_type_id(TypeId::METHOD)
    );
    assert!(
        get_attribute_value(&mut vm, method_value, &intern("__doc__"))
            .expect("__doc__ should resolve")
            .is_none()
    );

    unsafe {
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_get_attribute_value_prefers_callable_descriptor_doc_metadata() {
    let mut vm = vm_with_names(&[]);
    let (func_ptr, func_value) = make_test_function_value("documented_callable");
    let callable_doc = Value::string(intern("callable docs"));
    unsafe { &*func_ptr }.set_attr(intern("__doc__"), callable_doc);

    assert_eq!(
        get_attribute_value(&mut vm, func_value, &intern("__doc__"))
            .expect("function __doc__ should resolve"),
        callable_doc
    );

    let method_value = bind_instance_attribute(func_value, Value::int(7).unwrap());
    assert_eq!(
        get_attribute_value(&mut vm, method_value, &intern("__doc__"))
            .expect("bound method __doc__ should resolve"),
        callable_doc
    );

    let classmethod_ptr = Box::into_raw(Box::new(ClassMethodDescriptor::new(func_value)));
    let classmethod_value = Value::object_ptr(classmethod_ptr as *const ());
    assert_eq!(
        get_attribute_value(&mut vm, classmethod_value, &intern("__doc__"))
            .expect("classmethod __doc__ should resolve"),
        callable_doc
    );

    let staticmethod_ptr = Box::into_raw(Box::new(StaticMethodDescriptor::new(func_value)));
    let staticmethod_value = Value::object_ptr(staticmethod_ptr as *const ());
    assert_eq!(
        get_attribute_value(&mut vm, staticmethod_value, &intern("__doc__"))
            .expect("staticmethod __doc__ should resolve"),
        callable_doc
    );

    let property_doc = Value::string(intern("property docs"));
    let property_ptr = Box::into_raw(Box::new(PropertyDescriptor::new_full(
        None,
        None,
        None,
        Some(property_doc),
    )));
    let property_value = Value::object_ptr(property_ptr as *const ());
    assert_eq!(
        get_attribute_value(&mut vm, property_value, &intern("__doc__"))
            .expect("property __doc__ should resolve"),
        property_doc
    );

    unsafe {
        drop(Box::from_raw(property_ptr));
        drop(Box::from_raw(staticmethod_ptr));
        drop(Box::from_raw(classmethod_ptr));
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_get_attr_binds_builtin_dict_items_method_for_dict_subclass() {
    let mut vm = vm_with_names(&["items"]);
    let class = register_dict_subclass("DictSubclassAttr");
    let (instance_ptr, instance_value) = dict_backed_instance_value(&class);
    unsafe { &mut *instance_ptr }
        .dict_backing_mut()
        .expect("dict backing should exist")
        .set(Value::string(intern("alpha")), Value::int(1).unwrap());
    vm.current_frame_mut().set_reg(1, instance_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

    let method_value = vm.current_frame().get_reg(2);
    let method_ptr = method_value
        .as_object_ptr()
        .expect("builtin method should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "dict.items");
    let result = builtin.call(&[]).expect("dict.items should call");
    let result_ptr = result.as_object_ptr().expect("items should return a view");
    assert_eq!(extract_type_id(result_ptr), TypeId::DICT_ITEMS);

    unsafe {
        drop_boxed(instance_ptr);
    }
}

#[test]
fn test_get_attribute_value_binds_builtin_dict_setitem_for_dict_subclass() {
    let mut vm = vm_with_names(&[]);
    let class = register_dict_subclass("DictSubclassSetitem");
    let (instance_ptr, instance_value) = dict_backed_instance_value(&class);

    let method = get_attribute_value(&mut vm, instance_value, &intern("__setitem__"))
        .expect("dict subclass should inherit dict.__setitem__");
    let method_ptr = method
        .as_object_ptr()
        .expect("bound builtin method should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "dict.__setitem__");
    builtin
        .call(&[Value::string(intern("beta")), Value::int(7).unwrap()])
        .expect("bound dict.__setitem__ should accept dict subclasses");

    let backing = unsafe { &*instance_ptr }
        .dict_backing()
        .expect("dict backing should exist");
    assert_eq!(
        backing.get(Value::string(intern("beta"))),
        Some(Value::int(7).unwrap())
    );

    unsafe {
        drop_boxed(instance_ptr);
    }
}

#[test]
fn test_user_defined_class_leaves_plain_builtin_attributes_unbound() {
    let mut vm = vm_with_names(&[]);
    let (builtin_ptr, builtin_value) = builtin_function_value("count_args", builtin_arg_count);

    let mut class = PyClassObject::new_simple(intern("PlainBuiltinAttr"));
    class.set_attr(intern("helper"), builtin_value);
    let class = register_test_class(class);
    let (instance_ptr, instance_value) = instance_value(&class);

    let helper = get_attribute_value(&mut vm, instance_value, &intern("helper"))
        .expect("plain builtin class attribute should resolve");
    let helper_ptr = helper
        .as_object_ptr()
        .expect("builtin should be heap allocated");
    assert_eq!(extract_type_id(helper_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(helper, builtin_value);

    let builtin = unsafe { &*(helper_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "count_args");
    assert_eq!(builtin.bound_self(), None);
    let result = builtin
        .call(&[Value::int(7).unwrap()])
        .expect("plain builtin should stay unbound on Python heap classes");
    assert_eq!(result.as_int(), Some(1));

    unsafe {
        drop_boxed(instance_ptr);
        drop_boxed(builtin_ptr);
    }
}

#[test]
fn test_class_objects_leave_plain_builtin_attributes_unbound() {
    let mut vm = vm_with_names(&[]);
    let (builtin_ptr, builtin_value) = builtin_function_value("count_args", builtin_arg_count);

    let mut class = PyClassObject::new_simple(intern("PlainBuiltinAttr"));
    class.set_attr(intern("helper"), builtin_value);
    let class = register_test_class(class);

    let helper = get_attribute_value(
        &mut vm,
        Value::object_ptr(Arc::as_ptr(&class) as *const ()),
        &intern("helper"),
    )
    .expect("plain builtin class attribute should resolve on the class object");
    let helper_ptr = helper
        .as_object_ptr()
        .expect("builtin should be heap allocated");
    assert_eq!(extract_type_id(helper_ptr), TypeId::BUILTIN_FUNCTION);
    assert_eq!(helper, builtin_value);

    let builtin = unsafe { &*(helper_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "count_args");
    assert_eq!(builtin.bound_self(), None);
    let result = builtin
        .call(&[Value::int(7).unwrap()])
        .expect("class lookup should preserve the stored builtin callable");
    assert_eq!(result.as_int(), Some(1));

    unsafe {
        drop_boxed(builtin_ptr);
    }
}

#[test]
fn test_native_heaptype_binds_builtin_attributes_on_instances() {
    let mut vm = vm_with_names(&[]);
    let (builtin_ptr, builtin_value) = builtin_function_value("count_args", builtin_arg_count);

    let mut class = PyClassObject::new_simple(intern("NativeBuiltinAttr"));
    class.add_flags(prism_runtime::object::class::ClassFlags::NATIVE_HEAPTYPE);
    class.set_attr(intern("helper"), builtin_value);
    let class = register_test_class(class);
    let (instance_ptr, instance_value) = instance_value(&class);

    let helper = get_attribute_value(&mut vm, instance_value, &intern("helper"))
        .expect("native heap builtin class attribute should resolve");
    let helper_ptr = helper
        .as_object_ptr()
        .expect("bound builtin should be heap allocated");
    assert_eq!(extract_type_id(helper_ptr), TypeId::BUILTIN_FUNCTION);
    assert_ne!(helper, builtin_value);

    let bound = unsafe { &*(helper_ptr as *const BuiltinFunctionObject) };
    assert_eq!(bound.name(), "count_args");
    assert_eq!(bound.bound_self(), Some(instance_value));
    let result = bound
        .call(&[Value::int(7).unwrap()])
        .expect("native heap builtin should receive an implicit instance");
    assert_eq!(result.as_int(), Some(2));

    let builtin = unsafe { &*(builtin_ptr as *const BuiltinFunctionObject) };
    let result = builtin
        .call(&[Value::int(7).unwrap()])
        .expect("class attribute storage should remain unbound");
    assert_eq!(result.as_int(), Some(1));

    unsafe {
        drop_boxed(instance_ptr);
        drop_boxed(builtin_ptr);
    }
}

#[test]
fn test_user_defined_class_preserves_prebound_builtin_attributes_on_instances() {
    let mut vm = vm_with_names(&[]);
    let builtin = Box::new(BuiltinFunctionObject::new_bound(
        Arc::from("first_arg"),
        builtin_first_arg,
        Value::int(99).unwrap(),
    ));
    let builtin_ptr = Box::into_raw(builtin);
    let builtin_value = Value::object_ptr(builtin_ptr as *const ());

    let mut class = PyClassObject::new_simple(intern("BoundBuiltinAttr"));
    class.set_attr(intern("helper"), builtin_value);
    let class = register_test_class(class);
    let (instance_ptr, instance_value) = instance_value(&class);

    let helper = get_attribute_value(&mut vm, instance_value, &intern("helper"))
        .expect("pre-bound builtin class attribute should resolve");
    assert_eq!(helper, builtin_value);

    let builtin = unsafe { &*(builtin_ptr as *const BuiltinFunctionObject) };
    let result = builtin
        .call(&[])
        .expect("pre-bound builtin should retain its original receiver");
    assert_eq!(result.as_int(), Some(99));

    unsafe {
        drop_boxed(instance_ptr);
        drop_boxed(builtin_ptr);
    }
}

#[test]
fn test_exception_proxy_classes_bind_builtin_attributes_on_instances() {
    let mut vm = vm_with_names(&[]);
    let exception_base = crate::builtins::exception_proxy_class(
        crate::stdlib::exceptions::ExceptionTypeId::Exception,
    );
    let exception_base_id = exception_base.class_id();
    let class = PyClassObject::new(intern("ProxyExceptionChild"), &[exception_base_id], |id| {
        if id == exception_base_id {
            Some(exception_base.mro().iter().copied().collect())
        } else if id.0 < TypeId::FIRST_USER_TYPE {
            Some(
                builtin_class_mro(class_id_to_type_id(id))
                    .into_iter()
                    .collect(),
            )
        } else {
            None
        }
    })
    .expect("exception proxy subclass should build");
    let class = register_test_class(class);
    let (instance_ptr, instance_value) = instance_value(&class);

    let method = get_attribute_value(&mut vm, instance_value, &intern("with_traceback"))
        .expect("exception proxy builtin should resolve");
    let method_ptr = method
        .as_object_ptr()
        .expect("bound builtin method should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "BaseException.with_traceback");
    assert_eq!(builtin.bound_self(), Some(instance_value));

    unsafe {
        drop_boxed(instance_ptr);
    }
}

#[test]
fn test_builtin_exception_type_exposes_unbound_base_exception_str_method() {
    let mut vm = vm_with_names(&[]);
    let base_exception = crate::builtins::get_exception_type("BaseException")
        .expect("BaseException type should exist");
    let base_exception_value = Value::object_ptr(
        base_exception as *const crate::builtins::ExceptionTypeObject as *const (),
    );

    let method = get_attribute_value(&mut vm, base_exception_value, &intern("__str__"))
        .expect("BaseException.__str__ should resolve");
    let method_ptr = method
        .as_object_ptr()
        .expect("BaseException.__str__ should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "BaseException.__str__");
    assert_eq!(builtin.bound_self(), None);
}

#[test]
fn test_get_attribute_value_binds_python_setitem_on_dict_subclass() {
    let mut vm = vm_with_names(&[]);
    let (func_ptr, func_value) = make_test_function_value("__setitem__");

    let mut class = PyClassObject::new(
        intern("PreparedNamespace"),
        &[ClassId(TypeId::DICT.raw())],
        |id| {
            (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                builtin_class_mro(class_id_to_type_id(id))
                    .into_iter()
                    .collect()
            })
        },
    )
    .expect("dict subclass should build");
    class.set_attr(intern("__setitem__"), func_value);
    let class = register_test_class(class);

    let (instance_ptr, instance_value) = dict_backed_instance_value(&class);

    let method = get_attribute_value(&mut vm, instance_value, &intern("__setitem__"))
        .expect("dict subclass should expose class-defined __setitem__");
    let method_ptr = method
        .as_object_ptr()
        .expect("bound method should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::METHOD);

    let bound = unsafe { &*(method_ptr as *const BoundMethod) };
    assert_eq!(bound.function(), func_value);
    assert_eq!(bound.instance(), instance_value);

    unsafe {
        drop_boxed(instance_ptr);
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_user_defined_instance_dict_is_live_and_authoritative() {
    let mut vm = vm_with_names(&[]);
    let class = register_test_class(PyClassObject::new_simple(intern("InstanceDictLive")));
    let (instance_ptr, instance) = instance_value(&class);

    set_attribute_value(&mut vm, instance, &intern("alpha"), Value::int(1).unwrap())
        .expect("setting an ordinary instance attribute should succeed");
    assert_eq!(
        get_attribute_value(&mut vm, instance, &intern("alpha"))
            .expect("shape-backed attribute should resolve")
            .as_int(),
        Some(1)
    );

    let dict_value = get_attribute_value(&mut vm, instance, &intern("__dict__"))
        .expect("user instances should expose __dict__");
    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("__dict__ should be a dict");
    assert_eq!(extract_type_id(dict_ptr), TypeId::DICT);
    {
        let dict = dict_storage_mut_from_ptr(dict_ptr).expect("__dict__ should be mutable");
        assert_eq!(
            dict.get(Value::string(intern("alpha")))
                .expect("materialized dict should contain shaped attributes")
                .as_int(),
            Some(1)
        );
        dict.set(Value::string(intern("alpha")), Value::int(3).unwrap());
        dict.set(Value::string(intern("beta")), Value::int(2).unwrap());
    }

    assert_eq!(
        get_attribute_value(&mut vm, instance, &intern("alpha"))
            .expect("__dict__ writes should override stale shaped slots")
            .as_int(),
        Some(3)
    );
    assert_eq!(
        get_attribute_value(&mut vm, instance, &intern("beta"))
            .expect("__dict__ writes should be visible as attributes")
            .as_int(),
        Some(2)
    );

    set_attribute_value(&mut vm, instance, &intern("gamma"), Value::int(4).unwrap())
        .expect("setattr should update a materialized __dict__");
    let dict = dict_storage_ref_from_ptr(dict_ptr).expect("__dict__ should stay valid");
    assert_eq!(
        dict.get(Value::string(intern("gamma")))
            .expect("setattr should mirror into __dict__")
            .as_int(),
        Some(4)
    );

    delete_attribute_value(&mut vm, instance, &intern("alpha"))
        .expect("delattr should remove attributes from materialized __dict__");
    let err = get_attribute_value(&mut vm, instance, &intern("alpha"))
        .expect_err("deleted instance attribute should be missing");
    assert!(matches!(
        err.kind(),
        RuntimeErrorKind::AttributeError { .. }
    ));

    unsafe {
        drop_boxed(instance_ptr);
    }
}

#[test]
fn test_user_defined_instance_dict_assignment_aliases_external_dict() {
    let mut vm = vm_with_names(&[]);
    let class = register_test_class(PyClassObject::new_simple(intern("InstanceDictAlias")));
    let (instance_ptr, instance) = instance_value(&class);
    let (external_value, external_ptr) = boxed_value(DictObject::new());

    set_attribute_value(&mut vm, instance, &intern("__dict__"), external_value)
        .expect("__dict__ assignment should accept dictionaries");
    unsafe { &mut *external_ptr }
        .set(Value::string(intern("external")), Value::int(5).unwrap());
    assert_eq!(
        get_attribute_value(&mut vm, instance, &intern("external"))
            .expect("external __dict__ mutations should drive attributes")
            .as_int(),
        Some(5)
    );

    set_attribute_value(
        &mut vm,
        instance,
        &intern("mirrored"),
        Value::int(6).unwrap(),
    )
    .expect("setattr should mirror into an assigned __dict__");
    assert_eq!(
        unsafe { &*external_ptr }
            .get(Value::string(intern("mirrored")))
            .expect("assigned dict should receive mirrored writes")
            .as_int(),
        Some(6)
    );

    delete_attribute_value(&mut vm, instance, &intern("__dict__"))
        .expect("deleting __dict__ should reset the instance dictionary");
    let err = get_attribute_value(&mut vm, instance, &intern("external"))
        .expect_err("reset instance dictionary should drop external attribute");
    assert!(matches!(
        err.kind(),
        RuntimeErrorKind::AttributeError { .. }
    ));
    let reset_dict = get_attribute_value(&mut vm, instance, &intern("__dict__"))
        .expect("__dict__ should rematerialize after deletion");
    assert_ne!(reset_dict, external_value);
    assert_eq!(
        dict_storage_ref_from_value(reset_dict)
            .expect("reset __dict__ should be a dictionary")
            .len(),
        0
    );

    unsafe {
        drop_boxed(instance_ptr);
        drop_boxed(external_ptr);
    }
}

#[test]
fn test_get_attribute_value_resolves_object_new_for_none_primitive() {
    let mut vm = vm_with_names(&[]);

    let doc = get_attribute_value(&mut vm, Value::none(), &intern("__doc__"))
        .expect("None.__doc__ should resolve");
    assert_eq!(value_as_str(doc), "The type of the None singleton.");

    let method = get_attribute_value(&mut vm, Value::none(), &intern("__new__"))
        .expect("None should inherit object.__new__");
    let method_ptr = method
        .as_object_ptr()
        .expect("bound builtin method should be heap allocated");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "object.__new__");
}

#[test]
fn test_get_attribute_value_reuses_object_new_for_none_primitive() {
    let mut vm = vm_with_names(&[]);

    let first = get_attribute_value(&mut vm, Value::none(), &intern("__new__"))
        .expect("first None.__new__ lookup should succeed");
    let second = get_attribute_value(&mut vm, Value::none(), &intern("__new__"))
        .expect("second None.__new__ lookup should succeed");

    assert_eq!(first, second, "None.__new__ should remain stable");
    assert_eq!(
        first.as_object_ptr(),
        second.as_object_ptr(),
        "None.__new__ should reuse the shared builtin callable",
    );
}

#[test]
fn test_get_attribute_value_reuses_str_maketrans_for_primitive_instance() {
    let mut vm = vm_with_names(&[]);
    let owner = Value::string(intern("seed"));

    let first = get_attribute_value(&mut vm, owner, &intern("maketrans"))
        .expect("first str.maketrans lookup should succeed");
    let second = get_attribute_value(&mut vm, owner, &intern("maketrans"))
        .expect("second str.maketrans lookup should succeed");

    let first_ptr = first
        .as_object_ptr()
        .expect("str.maketrans should be heap allocated");
    let second_ptr = second
        .as_object_ptr()
        .expect("str.maketrans should be heap allocated");
    assert_eq!(first_ptr, second_ptr, "str.maketrans should stay unbound");

    let builtin = unsafe { &*(first_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "str.maketrans");
    assert!(builtin.bound_self().is_none());
}

#[test]
fn test_set_and_get_item_use_dict_backing_for_heap_dict_subclass() {
    let mut vm = vm_with_frame();
    let class = register_dict_subclass("DictSubclassItems");
    let (instance_ptr, instance_value) = dict_backed_instance_value(&class);
    let key = Value::string(intern("key"));
    let value = Value::int(42).unwrap();

    vm.current_frame_mut().set_reg(1, instance_value);
    vm.current_frame_mut().set_reg(2, value);
    vm.current_frame_mut().set_reg(3, key);

    let set_inst = Instruction::op_dss(
        Opcode::SetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(set_item(&mut vm, set_inst), ControlFlow::Continue));

    let get_inst = Instruction::op_dss(
        Opcode::GetItem,
        Register::new(4),
        Register::new(1),
        Register::new(3),
    );
    assert!(matches!(get_item(&mut vm, get_inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(4), value);

    let backing = unsafe { &*instance_ptr }
        .dict_backing()
        .expect("dict backing should exist");
    assert_eq!(backing.get(key), Some(value));

    unsafe {
        drop_boxed(instance_ptr);
    }
}

#[test]
fn test_get_attr_reads_imported_module_attributes_from_registry() {
    let mut vm = vm_with_names(&["iskeyword"]);
    let module = Arc::new(ModuleObject::new("keyword"));
    module.set_attr("iskeyword", Value::bool(true));
    vm.import_resolver.insert_module("keyword", module.clone());
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(Arc::as_ptr(&module) as *const ()));

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_bool(), Some(true));
}

#[test]
fn test_get_attr_exposes_live_module_dict() {
    let mut vm = vm_with_names(&[]);
    let module = Arc::new(ModuleObject::new("errno"));
    module.set_attr("EINVAL", Value::int(22).unwrap());
    vm.import_resolver
        .insert_module("errno", Arc::clone(&module));

    let dict_value = get_attribute_value(
        &mut vm,
        Value::object_ptr(Arc::as_ptr(&module) as *const ()),
        &intern("__dict__"),
    )
    .expect("module __dict__ lookup should succeed");
    let dict = dict_storage_ref_from_ptr(
        dict_value
            .as_object_ptr()
            .expect("module __dict__ should be a dict object"),
    )
    .expect("module __dict__ should expose dict storage");
    assert_eq!(
        dict.get(Value::string(intern("EINVAL")))
            .and_then(|value| value.as_int()),
        Some(22)
    );

    let dict = dict_storage_mut_from_ptr(
        dict_value
            .as_object_ptr()
            .expect("module __dict__ should remain a dict object"),
    )
    .expect("module __dict__ should be mutable");
    dict.set(Value::string(intern("EPERM")), Value::int(1).unwrap());
    assert_eq!(
        module.get_attr("EPERM").and_then(|value| value.as_int()),
        Some(1)
    );
}

#[test]
fn test_get_attr_exposes_import_exception_metadata() {
    let mut vm = vm_with_names(&[]);
    let exc = crate::builtins::create_exception_with_import_details(
        crate::stdlib::exceptions::ExceptionTypeId::ModuleNotFoundError,
        Some(Arc::from("No module named 'pkg.missing'")),
        Some(Arc::from("pkg.missing")),
        None,
    );

    let args_value = get_attribute_value(&mut vm, exc, &intern("args"))
        .expect("exception args should be readable");
    let args_ptr = args_value.as_object_ptr().expect("args should be a tuple");
    let args = unsafe { &*(args_ptr as *const TupleObject) };
    assert_eq!(args.len(), 1);
    assert_eq!(
        args.get(0),
        Some(Value::string(intern("No module named 'pkg.missing'")))
    );

    let class_value = get_attribute_value(&mut vm, exc, &intern("__class__"))
        .expect("__class__ should be readable");
    let class_name = get_attribute_value(&mut vm, class_value, &intern("__name__"))
        .expect("__class__.__name__ should be readable");
    assert_eq!(class_name, Value::string(intern("ModuleNotFoundError")));

    let name_value =
        get_attribute_value(&mut vm, exc, &intern("name")).expect("name should be readable");
    assert_eq!(name_value, Value::string(intern("pkg.missing")));

    let path_value =
        get_attribute_value(&mut vm, exc, &intern("path")).expect("path should be readable");
    assert!(path_value.is_none());
}

#[test]
fn test_native_exception_class_attribute_preserves_concrete_type() {
    let mut vm = vm_with_names(&[]);
    let exc = crate::builtins::create_exception(
        crate::stdlib::exceptions::ExceptionTypeId::TypeError,
        Some(Arc::from("boom")),
    );

    let class_value = get_attribute_value(&mut vm, exc, &intern("__class__"))
        .expect("__class__ should be readable");
    let class_name = get_attribute_value(&mut vm, class_value, &intern("__name__"))
        .expect("__class__.__name__ should be readable");

    assert_eq!(class_name, Value::string(intern("TypeError")));
}

#[test]
fn test_get_attr_exposes_os_error_metadata_from_args() {
    let mut vm = vm_with_names(&[]);
    let exc = crate::builtins::create_exception_with_args(
        crate::stdlib::exceptions::ExceptionTypeId::FileNotFoundError,
        None,
        vec![
            Value::int(2).unwrap(),
            Value::string(intern("No such file or directory")),
            Value::string(intern("missing.txt")),
            Value::none(),
            Value::string(intern("target.txt")),
        ]
        .into_boxed_slice(),
    );

    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("errno")).expect("errno should work"),
        Value::int(2).unwrap()
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("strerror")).expect("strerror should work"),
        Value::string(intern("No such file or directory"))
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("filename")).expect("filename should work"),
        Value::string(intern("missing.txt"))
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("filename2")).expect("filename2 should work"),
        Value::string(intern("target.txt"))
    );
    assert!(
        get_attribute_value(&mut vm, exc, &intern("winerror"))
            .expect("winerror should work")
            .is_none()
    );
}

#[test]
fn test_get_attr_exposes_os_error_strerror_from_message_without_args() {
    let mut vm = vm_with_names(&[]);
    let exc = crate::builtins::create_exception(
        crate::stdlib::exceptions::ExceptionTypeId::OSError,
        Some(Arc::from("operation failed")),
    );

    assert!(
        get_attribute_value(&mut vm, exc, &intern("errno"))
            .expect("errno should work")
            .is_none()
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("strerror")).expect("strerror should work"),
        Value::string(intern("operation failed"))
    );
    assert!(
        get_attribute_value(&mut vm, exc, &intern("filename"))
            .expect("filename should work")
            .is_none()
    );
}

#[test]
fn test_get_attr_exposes_syntax_exception_metadata() {
    let mut vm = vm_with_names(&[]);
    let exc = crate::builtins::create_exception_with_syntax_details(
        crate::stdlib::exceptions::ExceptionTypeId::SyntaxError,
        Some(Arc::from("invalid syntax")),
        crate::builtins::SyntaxErrorDetails::new(
            Some(Arc::from("sample.py")),
            Some(4),
            Some(6),
            Some(Arc::from("value =\n")),
            Some(4),
            Some(7),
        ),
    );

    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("filename")).expect("filename should work"),
        Value::string(intern("sample.py"))
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("lineno")).expect("lineno should work"),
        Value::int(4).unwrap()
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("offset")).expect("offset should work"),
        Value::int(6).unwrap()
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("text")).expect("text should work"),
        Value::string(intern("value =\n"))
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("end_lineno"))
            .expect("end_lineno should work"),
        Value::int(4).unwrap()
    );
    assert_eq!(
        get_attribute_value(&mut vm, exc, &intern("end_offset"))
            .expect("end_offset should work"),
        Value::int(7).unwrap()
    );
    assert!(
        get_attribute_value(&mut vm, exc, &intern("print_file_and_line"))
            .expect("print_file_and_line should work")
            .is_none()
    );
}

#[test]
fn test_get_attr_reads_function_code_view() {
    let (func_ptr, func_value) = make_test_function_value("callable");
    let mut vm = vm_with_names(&["__code__"]);
    vm.current_frame_mut().set_reg(1, func_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

    let code_ptr = vm.current_frame().get_reg(2).as_object_ptr().unwrap();
    assert_eq!(extract_type_id(code_ptr), TypeId::CODE);

    unsafe {
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_get_attr_exposes_code_object_metadata_used_by_warnings() {
    let mut code = CodeObject::new("warnsite", "warning_probe.py");
    code.qualname = Arc::from("WarningProbe.warnsite");
    code.first_lineno = 27;
    code.arg_count = 2;
    code.posonlyarg_count = 1;
    code.kwonlyarg_count = 1;
    code.flags = CodeFlags::MODULE;
    code.locals = vec![Arc::from("alpha"), Arc::from("beta")].into_boxed_slice();
    code.names = vec![Arc::from("__warningregistry__")].into_boxed_slice();
    code.freevars = vec![Arc::from("captured")].into_boxed_slice();
    code.cellvars = vec![Arc::from("cell")].into_boxed_slice();
    code.constants = vec![
        Constant::Value(Value::none()),
        Constant::Value(Value::int(7).unwrap()),
    ]
    .into_boxed_slice();

    let (code_value, code_ptr) = boxed_value(CodeObjectView::new(Arc::new(code)));
    let mut vm = vm_with_frame();

    assert_eq!(
        get_attribute_value(&mut vm, code_value, &intern("__class__"))
            .expect("code __class__ should be readable"),
        crate::builtins::builtin_type_object_for_type_id(TypeId::CODE)
    );
    assert_eq!(
        get_attribute_value(&mut vm, code_value, &intern("co_filename"))
            .expect("co_filename should be readable"),
        Value::string(intern("warning_probe.py"))
    );
    assert_eq!(
        get_attribute_value(&mut vm, code_value, &intern("co_name"))
            .expect("co_name should be readable"),
        Value::string(intern("warnsite"))
    );
    assert_eq!(
        get_attribute_value(&mut vm, code_value, &intern("co_qualname"))
            .expect("co_qualname should be readable"),
        Value::string(intern("WarningProbe.warnsite"))
    );
    assert_eq!(
        get_attribute_value(&mut vm, code_value, &intern("co_firstlineno"))
            .expect("co_firstlineno should be readable")
            .as_int(),
        Some(27)
    );

    let varnames = get_attribute_value(&mut vm, code_value, &intern("co_varnames"))
        .expect("co_varnames should be readable");
    let varnames_ptr = varnames
        .as_object_ptr()
        .expect("co_varnames should be a tuple");
    let varnames = unsafe { &*(varnames_ptr as *const TupleObject) };
    assert_eq!(
        varnames.as_slice(),
        &[
            Value::string(intern("alpha")),
            Value::string(intern("beta"))
        ]
    );

    let consts = get_attribute_value(&mut vm, code_value, &intern("co_consts"))
        .expect("co_consts should be readable");
    let consts_ptr = consts.as_object_ptr().expect("co_consts should be a tuple");
    let consts = unsafe { &*(consts_ptr as *const TupleObject) };
    assert_eq!(consts.as_slice(), &[Value::none(), Value::int(7).unwrap()]);

    unsafe {
        drop_boxed(code_ptr);
    }
}

#[test]
fn test_get_attr_exposes_code_positions_iterator_used_by_traceback() {
    let mut code = CodeObject::new("warnsite", "warning_probe.py");
    code.instructions = vec![
        Instruction::op(Opcode::Nop),
        Instruction::op(Opcode::Nop),
        Instruction::op(Opcode::Nop),
    ]
    .into_boxed_slice();
    code.line_table = vec![
        LineTableEntry {
            start_pc: 0,
            end_pc: 1,
            line: 27,
        },
        LineTableEntry {
            start_pc: 1,
            end_pc: 3,
            line: 31,
        },
    ]
    .into_boxed_slice();

    let (code_value, code_ptr) = boxed_value(CodeObjectView::new(Arc::new(code)));
    let mut vm = vm_with_frame();

    let method = get_attribute_value(&mut vm, code_value, &intern("co_positions"))
        .expect("co_positions should be readable");
    let method_ptr = method
        .as_object_ptr()
        .expect("co_positions should bind a builtin");
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "code.co_positions");

    let iter_value = builtin
        .call(&[])
        .expect("co_positions() should return an iterator");
    let iter = crate::builtins::get_iterator_mut(&iter_value)
        .expect("co_positions() result should be iterable");

    let first = unsafe {
        &*(iter
            .next()
            .expect("first position should exist")
            .as_object_ptr()
            .expect("position entries should be tuples") as *const TupleObject)
    };
    assert_eq!(
        first.as_slice(),
        &[
            Value::int(27).unwrap(),
            Value::int(27).unwrap(),
            Value::none(),
            Value::none(),
        ]
    );

    let second = unsafe {
        &*(iter
            .next()
            .expect("second position should exist")
            .as_object_ptr()
            .expect("position entries should be tuples") as *const TupleObject)
    };
    assert_eq!(
        second.as_slice(),
        &[
            Value::int(31).unwrap(),
            Value::int(31).unwrap(),
            Value::none(),
            Value::none(),
        ]
    );

    let third = unsafe {
        &*(iter
            .next()
            .expect("third position should exist")
            .as_object_ptr()
            .expect("position entries should be tuples") as *const TupleObject)
    };
    assert_eq!(
        third.as_slice(),
        &[
            Value::int(31).unwrap(),
            Value::int(31).unwrap(),
            Value::none(),
            Value::none(),
        ]
    );
    assert!(iter.next().is_none(), "iterator should be exhausted");

    unsafe {
        drop_boxed(code_ptr);
    }
}

#[test]
fn test_get_attr_exposes_frame_globals_and_locals_snapshots() {
    let mut globals = DictObject::new();
    globals.set(
        Value::string(intern("__name__")),
        Value::string(intern("warning_probe")),
    );
    let (globals_value, globals_ptr) = boxed_value(globals);

    let mut locals = DictObject::new();
    locals.set(Value::string(intern("flag")), Value::bool(true));
    let (locals_value, locals_ptr) = boxed_value(locals);

    let (frame_value, frame_ptr) = boxed_value(FrameViewObject::new(
        None,
        globals_value,
        locals_value,
        19,
        5,
        None,
    ));
    let mut vm = vm_with_frame();

    assert_eq!(
        get_attribute_value(&mut vm, frame_value, &intern("f_globals"))
            .expect("f_globals should be readable"),
        globals_value
    );
    assert_eq!(
        get_attribute_value(&mut vm, frame_value, &intern("f_locals"))
            .expect("f_locals should be readable"),
        locals_value
    );
    assert_eq!(
        get_attribute_value(&mut vm, frame_value, &intern("f_lineno"))
            .expect("f_lineno should be readable")
            .as_int(),
        Some(19)
    );
    assert_eq!(
        get_attribute_value(&mut vm, frame_value, &intern("f_lasti"))
            .expect("f_lasti should be readable")
            .as_int(),
        Some(5)
    );
    assert!(
        get_attribute_value(&mut vm, frame_value, &intern("f_back"))
            .expect("f_back should be readable")
            .is_none()
    );

    unsafe {
        drop_boxed(frame_ptr);
        drop_boxed(globals_ptr);
        drop_boxed(locals_ptr);
    }
}

#[test]
fn test_frame_code_view_allocates_after_full_nursery() {
    let code = Arc::new(CodeObject::new("frame_code", "frame_probe.py"));
    let (frame_value, frame_ptr) = boxed_value(FrameViewObject::new(
        Some(Arc::clone(&code)),
        Value::none(),
        Value::none(),
        27,
        3,
        None,
    ));
    let mut vm = vm_with_frame();

    exhaust_nursery(&vm);

    let code_value = get_attribute_value(&mut vm, frame_value, &intern("f_code"))
        .expect("f_code should allocate even after nursery exhaustion");
    let code_ptr = code_value
        .as_object_ptr()
        .expect("f_code should be an object");
    assert_eq!(extract_type_id(code_ptr), TypeId::CODE);
    let code_view = unsafe { &*(code_ptr as *const CodeObjectView) };
    assert_eq!(code_view.code().name.as_ref(), "frame_code");

    unsafe {
        drop_boxed(frame_ptr);
    }
}

#[test]
fn test_set_attr_allows_traceback_next_truncation() {
    let (next_value, next_ptr) =
        boxed_value(TracebackViewObject::new(Value::none(), None, 20, 0));
    let (traceback_value, traceback_ptr) = boxed_value(TracebackViewObject::new(
        Value::none(),
        Some(next_value),
        10,
        0,
    ));
    let mut vm = vm_with_frame();

    assert_eq!(
        get_attribute_value(&mut vm, traceback_value, &intern("tb_next"))
            .expect("tb_next should be readable"),
        next_value
    );

    set_attribute_value(&mut vm, traceback_value, &intern("tb_next"), Value::none())
        .expect("tb_next should accept None for traceback truncation");

    assert!(
        get_attribute_value(&mut vm, traceback_value, &intern("tb_next"))
            .expect("tb_next should remain readable")
            .is_none()
    );

    unsafe {
        drop_boxed(traceback_ptr);
        drop_boxed(next_ptr);
    }
}

#[test]
fn test_set_attr_validates_traceback_next() {
    let (traceback_value, traceback_ptr) =
        boxed_value(TracebackViewObject::new(Value::none(), None, 10, 0));
    let mut vm = vm_with_frame();

    let non_traceback_err = set_attribute_value(
        &mut vm,
        traceback_value,
        &intern("tb_next"),
        Value::int(1).unwrap(),
    )
    .expect_err("tb_next should reject non-traceback values");
    assert!(matches!(
        non_traceback_err.kind,
        RuntimeErrorKind::TypeError { .. }
    ));

    let loop_err = set_attribute_value(
        &mut vm,
        traceback_value,
        &intern("tb_next"),
        traceback_value,
    )
    .expect_err("tb_next should reject loops");
    assert!(matches!(loop_err.kind, RuntimeErrorKind::ValueError { .. }));

    unsafe {
        drop_boxed(traceback_ptr);
    }
}

#[test]
fn test_get_attr_reads_closure_and_cell_contents() {
    let closure = Arc::new(ClosureEnv::new(vec![Arc::new(Cell::new(
        Value::int(41).unwrap(),
    ))]));
    let (func_ptr, func_value) = make_test_function_value_with_closure("inner", Some(closure));
    let mut vm = vm_with_names(&["__closure__", "cell_contents"]);
    vm.current_frame_mut().set_reg(1, func_value);

    let closure_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(
        get_attr(&mut vm, closure_inst),
        ControlFlow::Continue
    ));

    let closure_value = vm.current_frame().get_reg(2);
    let closure_ptr = closure_value.as_object_ptr().unwrap();
    let tuple = unsafe { &*(closure_ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 1);

    let cell_value = tuple.get(0).unwrap();
    let cell_ptr = cell_value.as_object_ptr().unwrap();
    assert_eq!(extract_type_id(cell_ptr), TypeId::CELL_VIEW);
    vm.current_frame_mut().set_reg(3, cell_value);

    let cell_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(4),
        Register::new(3),
        Register::new(1),
    );
    assert!(matches!(
        get_attr(&mut vm, cell_inst),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(4).as_int(), Some(41));

    unsafe {
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_get_attr_exposes_builtin_type_reflection_objects() {
    let mut vm = vm_with_names(&["__dict__", "__init__", "join", "__code__", "__globals__"]);
    vm.current_frame_mut().set_reg(
        1,
        crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE),
    );
    vm.current_frame_mut().set_reg(
        2,
        crate::builtins::builtin_type_object_for_type_id(TypeId::OBJECT),
    );
    vm.current_frame_mut().set_reg(
        3,
        crate::builtins::builtin_type_object_for_type_id(TypeId::STR),
    );
    vm.current_frame_mut().set_reg(
        4,
        crate::builtins::builtin_type_object_for_type_id(TypeId::FUNCTION),
    );

    let type_dict_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(10),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(
        get_attr(&mut vm, type_dict_inst),
        ControlFlow::Continue
    ));
    assert_eq!(
        extract_type_id(vm.current_frame().get_reg(10).as_object_ptr().unwrap()),
        TypeId::MAPPING_PROXY
    );

    let object_init_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(11),
        Register::new(2),
        Register::new(1),
    );
    assert!(matches!(
        get_attr(&mut vm, object_init_inst),
        ControlFlow::Continue
    ));
    assert_eq!(
        extract_type_id(vm.current_frame().get_reg(11).as_object_ptr().unwrap()),
        TypeId::WRAPPER_DESCRIPTOR
    );

    let str_join_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(12),
        Register::new(3),
        Register::new(2),
    );
    assert!(matches!(
        get_attr(&mut vm, str_join_inst),
        ControlFlow::Continue
    ));
    assert_eq!(
        extract_type_id(vm.current_frame().get_reg(12).as_object_ptr().unwrap()),
        TypeId::METHOD_DESCRIPTOR
    );

    let function_code_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(13),
        Register::new(4),
        Register::new(3),
    );
    assert!(matches!(
        get_attr(&mut vm, function_code_inst),
        ControlFlow::Continue
    ));
    assert_eq!(
        extract_type_id(vm.current_frame().get_reg(13).as_object_ptr().unwrap()),
        TypeId::GETSET_DESCRIPTOR
    );

    let function_globals_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(14),
        Register::new(4),
        Register::new(4),
    );
    assert!(matches!(
        get_attr(&mut vm, function_globals_inst),
        ControlFlow::Continue
    ));
    assert_eq!(
        extract_type_id(vm.current_frame().get_reg(14).as_object_ptr().unwrap()),
        TypeId::MEMBER_DESCRIPTOR
    );
}

#[test]
fn test_get_attr_exposes___class___for_user_defined_instances() {
    let class = register_test_class(PyClassObject::new_simple(intern("HeapCarrier")));
    let (instance_ptr, instance_value) = instance_value(&class);
    let mut vm = vm_with_names(&[]);

    let class_value = get_attribute_value(&mut vm, instance_value, &intern("__class__"))
        .expect("__class__ should be readable on heap instances");
    assert_eq!(
        class_value.as_object_ptr(),
        Some(Arc::as_ptr(&class) as *const ()),
    );

    unsafe {
        drop_boxed(instance_ptr);
    }
}

#[test]
fn test_lookup_user_class_attr_tracks_registered_parent_mutations() {
    let shared = intern("shared");

    let parent = register_test_class(PyClassObject::new_simple(intern("LookupParent")));
    let child = PyClassObject::new(intern("LookupChild"), &[parent.class_id()], |id| {
        (id == parent.class_id()).then(|| parent.mro().iter().copied().collect())
    })
    .expect("child class should build");
    let child = register_test_class(child);

    assert!(lookup_user_class_attr(child.as_ref(), &shared).is_none());

    parent.set_attr(shared.clone(), Value::int_unchecked(10));
    assert_eq!(
        lookup_user_class_attr(child.as_ref(), &shared),
        Some(Value::int_unchecked(10))
    );

    child.set_attr(shared.clone(), Value::int_unchecked(20));
    assert_eq!(
        lookup_user_class_attr(child.as_ref(), &shared),
        Some(Value::int_unchecked(20))
    );

    assert_eq!(child.del_attr(&shared), Some(Value::int_unchecked(20)));
    assert_eq!(
        lookup_user_class_attr(child.as_ref(), &shared),
        Some(Value::int_unchecked(10))
    );
}

#[test]
fn test_get_attr_exposes_object_method_wrapper_view() {
    let mut vm = vm_with_names(&["__str__"]);
    let object = crate::builtins::builtin_object(&[]).expect("object() should succeed");
    vm.current_frame_mut().set_reg(1, object);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(
        extract_type_id(vm.current_frame().get_reg(2).as_object_ptr().unwrap()),
        TypeId::METHOD_WRAPPER
    );
}

#[test]
fn test_get_attr_exposes_function_annotations_dict_and_descriptor_getters() {
    let (func_ptr, func_value) = make_test_function_value("callable");
    let classmethod_ptr = Box::into_raw(Box::new(ClassMethodDescriptor::new(func_value)));
    let staticmethod_ptr = Box::into_raw(Box::new(StaticMethodDescriptor::new(func_value)));
    let classmethod_value = Value::object_ptr(classmethod_ptr as *const ());
    let staticmethod_value = Value::object_ptr(staticmethod_ptr as *const ());
    let mut vm = vm_with_frame();

    let annotations = get_attribute_value(&mut vm, func_value, &intern("__annotations__"))
        .expect("function annotations should materialize");
    let annotations_ptr = annotations
        .as_object_ptr()
        .expect("__annotations__ should be a dict");
    assert_eq!(extract_type_id(annotations_ptr), TypeId::DICT);
    assert_eq!(
        get_attribute_value(&mut vm, func_value, &intern("__annotations__"))
            .expect("function annotations should be stable"),
        annotations
    );

    let function_get = get_attribute_value(&mut vm, func_value, &intern("__get__"))
        .expect("functions should expose __get__");
    let function_get_ptr = function_get
        .as_object_ptr()
        .expect("function.__get__ should be heap allocated");
    let function_get_builtin = unsafe { &*(function_get_ptr as *const BuiltinFunctionObject) };
    assert_eq!(function_get_builtin.name(), "function.__get__");
    assert_eq!(function_get_builtin.bound_self(), Some(func_value));

    let classmethod_get = get_attribute_value(&mut vm, classmethod_value, &intern("__get__"))
        .expect("classmethod should expose __get__");
    let classmethod_get_ptr = classmethod_get
        .as_object_ptr()
        .expect("classmethod.__get__ should be heap allocated");
    let classmethod_get_builtin =
        unsafe { &*(classmethod_get_ptr as *const BuiltinFunctionObject) };
    assert_eq!(classmethod_get_builtin.name(), "classmethod.__get__");
    assert_eq!(
        classmethod_get_builtin.bound_self(),
        Some(classmethod_value)
    );

    let staticmethod_get = get_attribute_value(&mut vm, staticmethod_value, &intern("__get__"))
        .expect("staticmethod should expose __get__");
    let staticmethod_get_ptr = staticmethod_get
        .as_object_ptr()
        .expect("staticmethod.__get__ should be heap allocated");
    let staticmethod_get_builtin =
        unsafe { &*(staticmethod_get_ptr as *const BuiltinFunctionObject) };
    assert_eq!(staticmethod_get_builtin.name(), "staticmethod.__get__");
    assert_eq!(
        staticmethod_get_builtin.bound_self(),
        Some(staticmethod_value)
    );

    unsafe {
        drop(Box::from_raw(staticmethod_ptr));
        drop(Box::from_raw(classmethod_ptr));
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_get_attr_binds_property_setter_builtin_method() {
    let property = Box::new(PropertyDescriptor::new_getter(Value::int(1).unwrap()));
    let property_ptr = Box::into_raw(property);
    let property_value = Value::object_ptr(property_ptr as *const ());

    let mut vm = vm_with_names(&["setter"]);
    vm.current_frame_mut().set_reg(1, property_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

    let method_ptr = vm.current_frame().get_reg(2).as_object_ptr().unwrap();
    assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "property.setter");

    let copied = builtin
        .call(&[Value::int(2).unwrap()])
        .expect("bound property setter should call");
    let copied_ptr = copied.as_object_ptr().unwrap();
    let copied_desc = unsafe { &*(copied_ptr as *const PropertyDescriptor) };
    assert_eq!(copied_desc.getter(), Some(Value::int(1).unwrap()));
    assert_eq!(copied_desc.setter(), Some(Value::int(2).unwrap()));

    unsafe {
        drop(Box::from_raw(copied_ptr as *mut PropertyDescriptor));
        drop(Box::from_raw(property_ptr));
    }
}

#[test]
fn test_get_attr_invokes_property_data_descriptor_before_instance_attr() {
    let (getter_ptr, getter_value) =
        builtin_function_value("property_test.getter", property_storage_getter);
    let property = Box::new(PropertyDescriptor::new_getter(getter_value));
    let property_ptr = Box::into_raw(property);
    let property_value = Value::object_ptr(property_ptr as *const ());

    let mut class = PyClassObject::new_simple(intern("Managed"));
    class.set_attr(intern("managed"), property_value);
    let class = register_test_class(class);
    let (instance_ptr, instance_value) = instance_value(&class);
    unsafe {
        (*instance_ptr).set_property(
            intern("_value"),
            Value::int(41).unwrap(),
            shape_registry(),
        );
        (*instance_ptr).set_property(
            intern("managed"),
            Value::int(99).unwrap(),
            shape_registry(),
        );
    }

    let mut vm = vm_with_names(&["managed"]);
    vm.current_frame_mut().set_reg(1, instance_value);

    let inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(2),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(41));

    unsafe {
        drop_boxed(instance_ptr);
        drop(Box::from_raw(property_ptr));
        drop(Box::from_raw(getter_ptr));
    }
}

#[test]
fn test_set_attr_invokes_property_setter_on_user_instance() {
    let (getter_ptr, getter_value) =
        builtin_function_value("property_test.getter", property_storage_getter);
    let (setter_ptr, setter_value) =
        builtin_function_value("property_test.setter", property_storage_setter);
    let property = Box::new(PropertyDescriptor::new_full(
        Some(getter_value),
        Some(setter_value),
        None,
        None,
    ));
    let property_ptr = Box::into_raw(property);
    let property_value = Value::object_ptr(property_ptr as *const ());

    let mut class = PyClassObject::new_simple(intern("Managed"));
    class.set_attr(intern("managed"), property_value);
    let class = register_test_class(class);
    let (instance_ptr, instance_value) = instance_value(&class);

    let mut vm = vm_with_names(&["managed"]);
    vm.current_frame_mut().set_reg(1, instance_value);
    vm.current_frame_mut().set_reg(2, Value::int(73).unwrap());

    let inst = Instruction::op_dss(
        Opcode::SetAttr,
        Register::new(1),
        Register::new(0),
        Register::new(2),
    );
    assert!(matches!(set_attr(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(
        unsafe { &*instance_ptr }
            .get_property("_value")
            .and_then(|value| value.as_int()),
        Some(73)
    );
    assert!(unsafe { &*instance_ptr }.get_property("managed").is_none());

    unsafe {
        drop_boxed(instance_ptr);
        drop(Box::from_raw(property_ptr));
        drop(Box::from_raw(setter_ptr));
        drop(Box::from_raw(getter_ptr));
    }
}

#[test]
fn test_del_attr_invokes_property_deleter_on_user_instance() {
    let (getter_ptr, getter_value) =
        builtin_function_value("property_test.getter", property_storage_getter);
    let (deleter_ptr, deleter_value) =
        builtin_function_value("property_test.deleter", property_storage_deleter);
    let property = Box::new(PropertyDescriptor::new_full(
        Some(getter_value),
        None,
        Some(deleter_value),
        None,
    ));
    let property_ptr = Box::into_raw(property);
    let property_value = Value::object_ptr(property_ptr as *const ());

    let mut class = PyClassObject::new_simple(intern("Managed"));
    class.set_attr(intern("managed"), property_value);
    let class = register_test_class(class);
    let (instance_ptr, instance_value) = instance_value(&class);
    unsafe {
        (*instance_ptr).set_property(
            intern("_value"),
            Value::int(11).unwrap(),
            shape_registry(),
        );
    }

    let mut vm = vm_with_names(&["managed"]);
    vm.current_frame_mut().set_reg(1, instance_value);

    let inst = Instruction::op_dss(
        Opcode::DelAttr,
        Register::new(0),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(del_attr(&mut vm, inst), ControlFlow::Continue));
    assert!(unsafe { &*instance_ptr }.get_property("_value").is_none());

    unsafe {
        drop_boxed(instance_ptr);
        drop(Box::from_raw(property_ptr));
        drop(Box::from_raw(deleter_ptr));
        drop(Box::from_raw(getter_ptr));
    }
}

#[test]
fn test_set_and_del_attr_operate_on_class_objects() {
    let mut vm = vm_with_names(&["field"]);
    let (class_value, class_ptr) = class_value(PyClassObject::new_simple(intern("Example")));
    vm.current_frame_mut().set_reg(1, class_value);
    vm.current_frame_mut().set_reg(2, Value::int(7).unwrap());

    let set_inst = Instruction::op_dss(
        Opcode::SetAttr,
        Register::new(1),
        Register::new(0),
        Register::new(2),
    );
    assert!(matches!(set_attr(&mut vm, set_inst), ControlFlow::Continue));

    let get_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(3),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, get_inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(3).as_int(), Some(7));

    let del_inst = Instruction::op_dss(
        Opcode::DelAttr,
        Register::new(0),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(del_attr(&mut vm, del_inst), ControlFlow::Continue));
    assert!(matches!(get_attr(&mut vm, get_inst), ControlFlow::Error(_)));

    unsafe { drop_class(class_ptr) };
}

#[test]
fn test_set_and_del_attr_consume_extended_name_index() {
    let set_inst = Instruction::new(Opcode::SetAttr, 1, EXTENDED_ATTR_NAME_SENTINEL, 2);
    let del_inst = Instruction::new(Opcode::DelAttr, 0, 1, EXTENDED_ATTR_NAME_SENTINEL);
    let extension = Instruction::op_di(Opcode::AttrName, Register::new(0), 0x0123);

    let mut code = CodeObject::new("test_attrs", "<test>");
    code.names = names_with_extended_attr().into_boxed_slice();
    code.instructions = vec![set_inst, extension, del_inst, extension].into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0).expect("frame push failed");
    let (class_value, class_ptr) = class_value(PyClassObject::new_simple(intern("Example")));
    vm.current_frame_mut().set_reg(1, class_value);
    vm.current_frame_mut().set_reg(2, Value::int(7).unwrap());

    vm.current_frame_mut().ip = 1;
    assert!(matches!(set_attr(&mut vm, set_inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().ip, 2);
    assert_eq!(
        get_attribute_value(&mut vm, class_value, &intern("extended"))
            .expect("extended attribute should resolve")
            .as_int(),
        Some(7)
    );

    vm.current_frame_mut().ip = 3;
    assert!(matches!(del_attr(&mut vm, del_inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().ip, 4);
    let err = get_attribute_value(&mut vm, class_value, &intern("extended"))
        .expect_err("deleted class attribute should be missing");
    assert!(matches!(
        err.kind(),
        RuntimeErrorKind::AttributeError { .. }
    ));

    unsafe { drop_class(class_ptr) };
}

#[test]
fn test_set_and_del_attr_operate_on_imported_modules() {
    let mut vm = vm_with_names(&["field"]);
    let module = Arc::new(ModuleObject::new("modprobe"));
    vm.import_resolver.insert_module("modprobe", module.clone());
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(Arc::as_ptr(&module) as *const ()));
    vm.current_frame_mut().set_reg(2, Value::int(7).unwrap());

    let set_inst = Instruction::op_dss(
        Opcode::SetAttr,
        Register::new(1),
        Register::new(0),
        Register::new(2),
    );
    assert!(matches!(set_attr(&mut vm, set_inst), ControlFlow::Continue));
    assert_eq!(
        module.get_attr("field").and_then(|value| value.as_int()),
        Some(7)
    );

    let get_inst = Instruction::op_dss(
        Opcode::GetAttr,
        Register::new(3),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(get_attr(&mut vm, get_inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(3).as_int(), Some(7));

    let del_inst = Instruction::op_dss(
        Opcode::DelAttr,
        Register::new(0),
        Register::new(1),
        Register::new(0),
    );
    assert!(matches!(del_attr(&mut vm, del_inst), ControlFlow::Continue));
    assert!(module.get_attr("field").is_none());
    assert!(matches!(get_attr(&mut vm, get_inst), ControlFlow::Error(_)));
}
