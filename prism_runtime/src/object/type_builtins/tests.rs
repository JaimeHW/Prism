use super::*;

// =========================================================================
// SubclassBitmap Tests
// =========================================================================

#[test]
fn test_bitmap_new_is_empty() {
    let bitmap = SubclassBitmap::new();
    assert!(bitmap.is_empty());
    assert_eq!(bitmap.count_bits(), 0);
}

#[test]
fn test_bitmap_for_type() {
    let bitmap = SubclassBitmap::for_type(TypeId::INT);
    assert!(!bitmap.is_empty());
    assert!(bitmap.is_subclass_of(TypeId::INT));
    assert!(!bitmap.is_subclass_of(TypeId::FLOAT));
}

#[test]
fn test_bitmap_set_bit_inline() {
    let mut bitmap = SubclassBitmap::new();

    bitmap.set_bit(TypeId::INT);
    bitmap.set_bit(TypeId::FLOAT);
    bitmap.set_bit(TypeId::STR);

    assert!(bitmap.is_subclass_of(TypeId::INT));
    assert!(bitmap.is_subclass_of(TypeId::FLOAT));
    assert!(bitmap.is_subclass_of(TypeId::STR));
    assert!(!bitmap.is_subclass_of(TypeId::LIST));
    assert_eq!(bitmap.count_bits(), 3);
}

#[test]
fn test_bitmap_set_bit_overflow() {
    let mut bitmap = SubclassBitmap::new();

    // Type ID 256 is first user-defined type (in overflow region)
    let user_type = TypeId::from_raw(256);
    bitmap.set_bit(user_type);

    assert!(bitmap.is_subclass_of(user_type));
    assert!(bitmap.overflow.is_some());
}

#[test]
fn test_bitmap_grows_for_high_heap_type_ids() {
    let mut bitmap = SubclassBitmap::new();
    let parent = TypeId::from_raw(1_500);
    let child = TypeId::from_raw(4_096);

    bitmap.set_bit(parent);
    bitmap.set_bit(child);

    assert!(bitmap.is_subclass_of(parent));
    assert!(bitmap.is_subclass_of(child));
    assert!(!bitmap.is_subclass_of(TypeId::from_raw(4_095)));
}

#[test]
fn test_bitmap_merge() {
    let mut parent1 = SubclassBitmap::new();
    parent1.set_bit(TypeId::INT);
    parent1.set_bit(TypeId::OBJECT);

    let mut parent2 = SubclassBitmap::new();
    parent2.set_bit(TypeId::STR);
    parent2.set_bit(TypeId::OBJECT);

    let mut child = SubclassBitmap::new();
    child.set_bit(TypeId::from_raw(300)); // Child's own bit
    child.merge(&parent1);
    child.merge(&parent2);

    // Child should have all parent bits
    assert!(child.is_subclass_of(TypeId::INT));
    assert!(child.is_subclass_of(TypeId::STR));
    assert!(child.is_subclass_of(TypeId::OBJECT));
    assert!(child.is_subclass_of(TypeId::from_raw(300)));
}

#[test]
fn test_bitmap_merge_preserves_high_heap_type_ids() {
    let high_parent_type = TypeId::from_raw(2_048);
    let high_child_type = TypeId::from_raw(4_096);

    let mut parent = SubclassBitmap::new();
    parent.set_bit(high_parent_type);
    parent.set_bit(TypeId::OBJECT);

    let mut child = SubclassBitmap::new();
    child.set_bit(high_child_type);
    child.merge(&parent);

    assert!(child.is_subclass_of(high_child_type));
    assert!(child.is_subclass_of(high_parent_type));
    assert!(child.is_subclass_of(TypeId::OBJECT));
}

#[test]
fn test_bitmap_from_parents() {
    let parent1 = builtin_type_bitmap(TypeId::INT);
    let parent2 = builtin_type_bitmap(TypeId::STR);

    let child_type = TypeId::from_raw(300);
    let child = SubclassBitmap::from_parents(child_type, [&parent1, &parent2].into_iter());

    assert!(child.is_subclass_of(child_type));
    assert!(child.is_subclass_of(TypeId::INT));
    assert!(child.is_subclass_of(TypeId::STR));
    assert!(child.is_subclass_of(TypeId::OBJECT));
}

#[test]
fn test_bitmap_is_subclass_of_any() {
    let bitmap = builtin_type_bitmap(TypeId::INT);

    assert!(bitmap.is_subclass_of_any(&[TypeId::INT, TypeId::FLOAT]));
    assert!(bitmap.is_subclass_of_any(&[TypeId::OBJECT]));
    assert!(!bitmap.is_subclass_of_any(&[TypeId::STR, TypeId::LIST]));
    assert!(!bitmap.is_subclass_of_any(&[]));
}

#[test]
fn test_bitmap_all_builtin_types() {
    // Test all built-in types fit in inline storage
    let builtins = [
        TypeId::NONE,
        TypeId::BOOL,
        TypeId::INT,
        TypeId::FLOAT,
        TypeId::STR,
        TypeId::BYTES,
        TypeId::BYTEARRAY,
        TypeId::LIST,
        TypeId::TUPLE,
        TypeId::DICT,
        TypeId::SET,
        TypeId::FROZENSET,
        TypeId::FUNCTION,
        TypeId::METHOD,
        TypeId::CLOSURE,
        TypeId::CODE,
        TypeId::MODULE,
        TypeId::TYPE,
        TypeId::OBJECT,
        TypeId::SLICE,
        TypeId::RANGE,
        TypeId::ITERATOR,
        TypeId::GENERATOR,
        TypeId::EXCEPTION,
        TypeId::BUILTIN_FUNCTION,
        TypeId::SUPER,
    ];

    let mut bitmap = SubclassBitmap::new();
    for &type_id in &builtins {
        bitmap.set_bit(type_id);
    }

    // Should all be in inline storage
    assert!(bitmap.overflow.is_none());

    for &type_id in &builtins {
        assert!(bitmap.is_subclass_of(type_id));
    }
}

#[test]
fn test_bitmap_many_user_types() {
    let mut bitmap = SubclassBitmap::new();

    // Add 100 user-defined types
    for i in 256..356 {
        bitmap.set_bit(TypeId::from_raw(i));
    }

    assert!(bitmap.overflow.is_some());

    for i in 256..356 {
        assert!(bitmap.is_subclass_of(TypeId::from_raw(i)));
    }
}

#[test]
fn test_bitmap_clone() {
    let mut original = SubclassBitmap::new();
    original.set_bit(TypeId::INT);
    original.set_bit(TypeId::from_raw(300));

    let cloned = original.clone();

    assert!(cloned.is_subclass_of(TypeId::INT));
    assert!(cloned.is_subclass_of(TypeId::from_raw(300)));
}

// =========================================================================
// TypeCheckIC Tests
// =========================================================================

#[test]
fn test_ic_new_is_empty() {
    let ic = TypeCheckIC::new();
    assert!(ic.is_empty());
    assert_eq!(ic.len(), 0);
}

#[test]
fn test_ic_insert_and_lookup() {
    let mut ic = TypeCheckIC::new();

    ic.insert(ClassId(100), true);
    ic.insert(ClassId(101), false);

    assert_eq!(ic.lookup(ClassId(100)), Some(true));
    assert_eq!(ic.lookup(ClassId(101)), Some(false));
    assert_eq!(ic.lookup(ClassId(102)), None);
    assert_eq!(ic.len(), 2);
}

#[test]
fn test_ic_full_replacement() {
    let mut ic = TypeCheckIC::new();

    // Fill cache
    ic.insert(ClassId(100), true);
    ic.insert(ClassId(101), true);
    ic.insert(ClassId(102), true);
    ic.insert(ClassId(103), true);

    assert_eq!(ic.len(), 4);

    // Insert fifth entry - should replace first
    ic.insert(ClassId(104), false);

    assert_eq!(ic.len(), 4);
    assert_eq!(ic.lookup(ClassId(104)), Some(false));
    // First entry was replaced
    assert_eq!(ic.lookup(ClassId(100)), None);
}

#[test]
fn test_ic_clear() {
    let mut ic = TypeCheckIC::new();

    ic.insert(ClassId(100), true);
    ic.insert(ClassId(101), true);

    ic.clear();

    assert!(ic.is_empty());
    assert_eq!(ic.lookup(ClassId(100)), None);
}

#[test]
fn test_ic_circular_replacement() {
    let mut ic = TypeCheckIC::new();

    // Fill and overflow multiple times
    for i in 0..12 {
        ic.insert(ClassId(i), i % 2 == 0);
    }

    // Only last 4 should be present
    assert_eq!(ic.len(), 4);
    assert!(ic.lookup(ClassId(8)).is_some());
    assert!(ic.lookup(ClassId(9)).is_some());
    assert!(ic.lookup(ClassId(10)).is_some());
    assert!(ic.lookup(ClassId(11)).is_some());
}

// =========================================================================
// Type Builtin Function Tests
// =========================================================================

#[test]
fn test_type_of_int() {
    let value = Value::int_unchecked(42);
    assert_eq!(type_of_value(value), TypeId::INT);
}

#[test]
fn test_type_of_float() {
    let value = Value::from(3.125f64);
    assert_eq!(type_of_value(value), TypeId::FLOAT);
}

#[test]
fn test_type_of_bool() {
    assert_eq!(type_of_value(Value::bool(true)), TypeId::BOOL);
    assert_eq!(type_of_value(Value::bool(false)), TypeId::BOOL);
}

#[test]
fn test_type_of_none() {
    assert_eq!(type_of_value(Value::none()), TypeId::NONE);
}

#[test]
fn test_class_id_to_type_id_maps_object_sentinel() {
    assert_eq!(class_id_to_type_id(ClassId::OBJECT), TypeId::OBJECT);
    assert_eq!(
        class_id_to_type_id(ClassId(TypeId::NONE.raw())),
        TypeId::NONE
    );
    assert_eq!(class_id_to_type_id(ClassId(TypeId::INT.raw())), TypeId::INT);
    assert_eq!(
        class_id_to_type_id(ClassId(TypeId::FIRST_USER_TYPE)),
        TypeId::from_raw(TypeId::FIRST_USER_TYPE)
    );
}

#[test]
fn test_builtin_none_mro_preserves_none_type() {
    assert_eq!(
        builtin_class_mro(TypeId::NONE),
        vec![ClassId(TypeId::NONE.raw()), ClassId::OBJECT]
    );
}

#[test]
fn test_isinstance_same_type() {
    let value = Value::int_unchecked(42);
    let int_class = ClassId(TypeId::INT.raw());

    let result = isinstance(value, int_class, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(result);
}

#[test]
fn test_isinstance_parent_type() {
    let value = Value::int_unchecked(42);
    let int_class = ClassId(TypeId::INT.raw());
    let object_class = ClassId(TypeId::OBJECT.raw());

    let result = isinstance(value, object_class, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(result);
}

#[test]
fn test_isinstance_parent_type_accepts_object_sentinel() {
    let value = Value::int_unchecked(42);
    let int_class = ClassId(TypeId::INT.raw());

    let result = isinstance(value, ClassId::OBJECT, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(result);
}

#[test]
fn test_isinstance_unrelated_type() {
    let value = Value::int_unchecked(42);
    let int_class = ClassId(TypeId::INT.raw());
    let str_class = ClassId(TypeId::STR.raw());

    let result = isinstance(value, str_class, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(!result);
}

#[test]
fn test_isinstance_multi() {
    let value = Value::int_unchecked(42);
    let int_class = ClassId(TypeId::INT.raw());

    let classes = vec![
        ClassId(TypeId::STR.raw()),
        ClassId(TypeId::LIST.raw()),
        ClassId(TypeId::INT.raw()),
    ];

    let result = isinstance_multi(value, &classes, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(result);
}

#[test]
fn test_issubclass_same() {
    let int_class = ClassId(TypeId::INT.raw());

    let result = issubclass(int_class, int_class, |_| None);
    assert!(result);
}

#[test]
fn test_issubclass_parent() {
    let int_class = ClassId(TypeId::INT.raw());
    let object_class = ClassId(TypeId::OBJECT.raw());

    let result = issubclass(int_class, object_class, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(result);
}

#[test]
fn test_issubclass_parent_accepts_object_sentinel() {
    let int_class = ClassId(TypeId::INT.raw());

    let result = issubclass(int_class, ClassId::OBJECT, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(result);
}

#[test]
fn test_issubclass_unrelated() {
    let int_class = ClassId(TypeId::INT.raw());
    let str_class = ClassId(TypeId::STR.raw());

    let result = issubclass(int_class, str_class, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(!result);
}

#[test]
fn test_issubclass_multi() {
    let int_class = ClassId(TypeId::INT.raw());
    let classes = vec![ClassId(TypeId::STR.raw()), ClassId(TypeId::OBJECT.raw())];

    let result = issubclass_multi(int_class, &classes, |id| {
        if id == int_class {
            Some(builtin_type_bitmap(TypeId::INT))
        } else {
            None
        }
    });

    assert!(result);
}

// =========================================================================
// Builtin Type Bitmap Tests
// =========================================================================

#[test]
fn test_builtin_type_bitmap() {
    let int_bitmap = builtin_type_bitmap(TypeId::INT);

    assert!(int_bitmap.is_subclass_of(TypeId::INT));
    assert!(int_bitmap.is_subclass_of(TypeId::OBJECT));
    assert!(!int_bitmap.is_subclass_of(TypeId::STR));
}

#[test]
fn test_object_bitmap_static() {
    assert!(OBJECT_BITMAP.is_subclass_of(TypeId::OBJECT));
    assert!(!OBJECT_BITMAP.is_subclass_of(TypeId::INT));
}

// =========================================================================
// Memory Layout Tests
// =========================================================================

#[test]
fn test_bitmap_size() {
    // SubclassBitmap should be compact
    let size = std::mem::size_of::<SubclassBitmap>();
    assert!(
        size <= 32,
        "SubclassBitmap size ({} bytes) should be <= 32",
        size
    );
}

#[test]
fn test_ic_size() {
    // TypeCheckIC should fit in a cache line
    let size = std::mem::size_of::<TypeCheckIC>();
    assert!(
        size <= 64,
        "TypeCheckIC size ({} bytes) should be <= 64",
        size
    );
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_bitmap_boundary_type_63() {
    // Type 63 is the last inline bit
    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(TypeId::from_raw(63));

    assert!(bitmap.is_subclass_of(TypeId::from_raw(63)));
    assert!(bitmap.overflow.is_none());
}

#[test]
fn test_bitmap_boundary_type_64() {
    // Type 64 is the first overflow bit
    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(TypeId::from_raw(64));

    assert!(bitmap.is_subclass_of(TypeId::from_raw(64)));
    assert!(bitmap.overflow.is_some());
}

#[test]
fn test_ic_lookup_none_class() {
    let mut ic = TypeCheckIC::new();
    ic.insert(ClassId::NONE, true);

    assert_eq!(ic.lookup(ClassId::NONE), Some(true));
}

#[test]
fn test_isinstance_empty_tuple() {
    let value = Value::int_unchecked(42);

    let result = isinstance_multi(value, &[], |_| None);
    assert!(!result);
}

// =========================================================================
// type_new Tests
// =========================================================================

use super::{
    ClassRegistry, SimpleClassRegistry, TypeCreationError, type_new, type_new_with_metaclass,
};
use crate::object::class::{ClassDict, ClassFlags, PyClassObject};
use crate::object::descriptor::{ClassMethodDescriptor, StaticMethodDescriptor};
use crate::object::registry::global_registry;
use crate::object::type_obj::TypeFlags;
use crate::types::function::FunctionObject;
use prism_code::CodeObject;
use prism_core::intern::intern;
use std::sync::Arc;

fn create_test_registry() -> SimpleClassRegistry {
    SimpleClassRegistry::new()
}

fn bitmap_for_class(class: &PyClassObject) -> SubclassBitmap {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    bitmap
}

fn test_function_value(name: &str) -> Value {
    let code = Arc::new(CodeObject::new(name, "<test>"));
    let function = FunctionObject::new(code, Arc::from(name), None, None);
    Value::object_ptr(Box::into_raw(Box::new(function)) as *const ())
}

#[test]
fn test_type_new_simple_class() {
    let registry = create_test_registry();
    let name = intern("MyClass");
    let namespace = ClassDict::new();

    let result = type_new(name, &[], &namespace, &registry);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.class.name().as_str(), "MyClass");
    assert!(result.class.bases().is_empty());
    assert!(
        result
            .flags
            .contains(crate::object::class::ClassFlags::INITIALIZED)
    );
}

#[test]
fn test_type_new_records_explicit_metaclass() {
    let registry = create_test_registry();
    let name = intern("MetaBoundClass");
    let namespace = ClassDict::new();
    let explicit_metaclass = Value::int_unchecked(123);

    let result =
        type_new_with_metaclass(name, &[], &namespace, explicit_metaclass, &registry).unwrap();

    assert_eq!(result.class.metaclass(), explicit_metaclass);
}

#[test]
fn test_type_new_marks_classes_derived_from_type_as_metaclasses() {
    let registry = create_test_registry();
    let name = intern("MetaClass");
    let namespace = ClassDict::new();

    let result = type_new(name, &[ClassId(TypeId::TYPE.raw())], &namespace, &registry).unwrap();

    assert!(result.class.flags().contains(ClassFlags::METACLASS));
}

#[test]
fn test_type_new_with_init() {
    let registry = create_test_registry();
    let name = intern("InitClass");
    let namespace = ClassDict::new();

    // Add __init__ to namespace
    let init_name = intern("__init__");
    namespace.set(init_name, Value::int_unchecked(0));

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    assert!(
        result
            .flags
            .contains(crate::object::class::ClassFlags::HAS_INIT)
    );
}

#[test]
fn test_type_new_with_new() {
    let registry = create_test_registry();
    let name = intern("NewClass");
    let namespace = ClassDict::new();

    // Add __new__ to namespace
    let new_name = intern("__new__");
    namespace.set(new_name, Value::int_unchecked(0));

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    assert!(
        result
            .flags
            .contains(crate::object::class::ClassFlags::HAS_NEW)
    );
}

#[test]
fn test_type_new_wraps_function_dunder_new_as_staticmethod() {
    let registry = create_test_registry();
    let name = intern("WrappedNewClass");
    let namespace = ClassDict::new();
    let function_value = test_function_value("__new__");
    namespace.set(intern("__new__"), function_value);

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    let stored = result
        .class
        .get_attr(&intern("__new__"))
        .expect("__new__ should be present on the class");
    let ptr = stored
        .as_object_ptr()
        .expect("normalized __new__ should be a descriptor object");
    let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };

    assert_eq!(
        unsafe { (*(ptr as *const crate::object::ObjectHeader)).type_id },
        TypeId::STATICMETHOD
    );
    assert_eq!(descriptor.function(), function_value);
}

#[test]
fn test_type_new_implicit_descriptor_uses_bound_heap() {
    let heap = prism_gc::heap::GcHeap::with_defaults();
    let _binding = crate::allocation_context::RuntimeHeapBinding::register(&heap);
    let registry = create_test_registry();
    let name = intern("HeapWrappedNewClass");
    let namespace = ClassDict::new();
    namespace.set(intern("__new__"), test_function_value("__new__"));

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    let stored = result
        .class
        .get_attr(&intern("__new__"))
        .expect("__new__ should be present on the class");
    let ptr = stored
        .as_object_ptr()
        .expect("normalized __new__ should be a descriptor object");

    assert!(heap.contains(ptr));
}

#[test]
fn test_type_new_wraps_function_dunder_init_subclass_as_classmethod() {
    let registry = create_test_registry();
    let name = intern("WrappedInitSubclassClass");
    let namespace = ClassDict::new();
    let function_value = test_function_value("__init_subclass__");
    namespace.set(intern("__init_subclass__"), function_value);

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    let stored = result
        .class
        .get_attr(&intern("__init_subclass__"))
        .expect("__init_subclass__ should be present on the class");
    let ptr = stored
        .as_object_ptr()
        .expect("normalized __init_subclass__ should be a descriptor object");
    let descriptor = unsafe { &*(ptr as *const ClassMethodDescriptor) };

    assert_eq!(
        unsafe { (*(ptr as *const crate::object::ObjectHeader)).type_id },
        TypeId::CLASSMETHOD
    );
    assert_eq!(descriptor.function(), function_value);
}

#[test]
fn test_type_new_wraps_function_dunder_class_getitem_as_classmethod() {
    let registry = create_test_registry();
    let name = intern("WrappedClassGetitemClass");
    let namespace = ClassDict::new();
    let function_value = test_function_value("__class_getitem__");
    namespace.set(intern("__class_getitem__"), function_value);

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    let stored = result
        .class
        .get_attr(&intern("__class_getitem__"))
        .expect("__class_getitem__ should be present on the class");
    let ptr = stored
        .as_object_ptr()
        .expect("normalized __class_getitem__ should be a descriptor object");
    let descriptor = unsafe { &*(ptr as *const ClassMethodDescriptor) };

    assert_eq!(
        unsafe { (*(ptr as *const crate::object::ObjectHeader)).type_id },
        TypeId::CLASSMETHOD
    );
    assert_eq!(descriptor.function(), function_value);
}

#[test]
fn test_type_new_preserves_explicit_staticmethod_dunder_new() {
    let registry = create_test_registry();
    let name = intern("ExplicitStaticNewClass");
    let namespace = ClassDict::new();
    let function_value = test_function_value("__new__");
    let descriptor = StaticMethodDescriptor::new(function_value);
    let descriptor_value = Value::object_ptr(Box::into_raw(Box::new(descriptor)) as *const ());
    namespace.set(intern("__new__"), descriptor_value);

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    let stored = result
        .class
        .get_attr(&intern("__new__"))
        .expect("__new__ should be present on the class");

    assert_eq!(stored, descriptor_value);
}

#[test]
fn test_type_new_preserves_explicit_classmethod_dunder_init_subclass() {
    let registry = create_test_registry();
    let name = intern("ExplicitClassMethodInitSubclassClass");
    let namespace = ClassDict::new();
    let function_value = test_function_value("__init_subclass__");
    let descriptor = ClassMethodDescriptor::new(function_value);
    let descriptor_value = Value::object_ptr(Box::into_raw(Box::new(descriptor)) as *const ());
    namespace.set(intern("__init_subclass__"), descriptor_value);

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    let stored = result
        .class
        .get_attr(&intern("__init_subclass__"))
        .expect("__init_subclass__ should be present on the class");

    assert_eq!(stored, descriptor_value);
}

#[test]
fn test_type_new_with_slots() {
    let registry = create_test_registry();
    let name = intern("SlotsClass");
    let namespace = ClassDict::new();

    // Add __slots__ to namespace
    let slots_name = intern("__slots__");
    namespace.set(slots_name, Value::int_unchecked(0));

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    assert!(
        result
            .flags
            .contains(crate::object::class::ClassFlags::HAS_SLOTS)
    );
}

#[test]
fn test_type_new_with_hash() {
    let registry = create_test_registry();
    let name = intern("HashClass");
    let namespace = ClassDict::new();

    // Add __hash__ to namespace
    let hash_name = intern("__hash__");
    namespace.set(hash_name, Value::int_unchecked(0));

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    assert!(
        result
            .flags
            .contains(crate::object::class::ClassFlags::HASHABLE)
    );
}

#[test]
fn test_type_new_with_eq() {
    let registry = create_test_registry();
    let name = intern("EqClass");
    let namespace = ClassDict::new();

    // Add __eq__ to namespace
    let eq_name = intern("__eq__");
    namespace.set(eq_name, Value::int_unchecked(0));

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    assert!(
        result
            .flags
            .contains(crate::object::class::ClassFlags::HAS_EQ)
    );
}

#[test]
fn test_type_new_with_del() {
    let registry = create_test_registry();
    let name = intern("DelClass");
    let namespace = ClassDict::new();

    // Add __del__ to namespace
    let del_name = intern("__del__");
    namespace.set(del_name, Value::int_unchecked(0));

    let result = type_new(name, &[], &namespace, &registry).unwrap();
    assert!(
        result
            .flags
            .contains(crate::object::class::ClassFlags::HAS_FINALIZER)
    );
}

#[test]
fn test_type_new_all_special_methods() {
    let registry = create_test_registry();
    let name = intern("AllSpecialClass");
    let namespace = ClassDict::new();

    // Add all special methods
    namespace.set(intern("__new__"), Value::int_unchecked(0));
    namespace.set(intern("__init__"), Value::int_unchecked(0));
    namespace.set(intern("__slots__"), Value::int_unchecked(0));
    namespace.set(intern("__hash__"), Value::int_unchecked(0));
    namespace.set(intern("__eq__"), Value::int_unchecked(0));
    namespace.set(intern("__del__"), Value::int_unchecked(0));

    let result = type_new(name, &[], &namespace, &registry).unwrap();

    use crate::object::class::ClassFlags;
    assert!(result.flags.contains(ClassFlags::HAS_NEW));
    assert!(result.flags.contains(ClassFlags::HAS_INIT));
    assert!(result.flags.contains(ClassFlags::HAS_SLOTS));
    assert!(result.flags.contains(ClassFlags::HASHABLE));
    assert!(result.flags.contains(ClassFlags::HAS_EQ));
    assert!(result.flags.contains(ClassFlags::HAS_FINALIZER));
}

#[test]
fn test_type_new_empty_name_error() {
    let registry = create_test_registry();
    let name = intern("");
    let namespace = ClassDict::new();

    let result = type_new(name, &[], &namespace, &registry);
    assert!(result.is_err());

    match result.unwrap_err() {
        TypeCreationError::InvalidName { name } => {
            assert_eq!(name, "");
        }
        _ => panic!("Expected InvalidName error"),
    }
}

#[test]
fn test_type_new_base_not_found() {
    let registry = create_test_registry();
    let name = intern("DerivedClass");
    let namespace = ClassDict::new();
    let fake_base = ClassId(99999);

    let result = type_new(name, &[fake_base], &namespace, &registry);
    assert!(result.is_err());

    match result.unwrap_err() {
        TypeCreationError::BaseNotFound { class_id } => {
            assert_eq!(class_id, fake_base);
        }
        _ => panic!("Expected BaseNotFound error"),
    }
}

#[test]
fn test_type_new_final_base_error() {
    let registry = create_test_registry();

    // Create a final base class
    let mut base = PyClassObject::new_simple(intern("FinalBase"));
    base.mark_final();
    let base_id = base.class_id();
    let base = Arc::new(base);

    // Register it with a bitmap
    let bitmap = SubclassBitmap::new();
    registry.register(base, bitmap);

    // Try to inherit from final class
    let name = intern("DerivedFromFinal");
    let namespace = ClassDict::new();

    let result = type_new(name, &[base_id], &namespace, &registry);
    assert!(result.is_err());

    match result.unwrap_err() {
        TypeCreationError::FinalBase { class_name } => {
            assert_eq!(class_name, "FinalBase");
        }
        _ => panic!("Expected FinalBase error"),
    }
}

#[test]
fn test_type_new_rejects_bool_as_base_type() {
    let registry = create_test_registry();
    let name = intern("DerivedFromBool");
    let namespace = ClassDict::new();

    let result = type_new(name, &[ClassId(TypeId::BOOL.raw())], &namespace, &registry);
    assert!(result.is_err());

    match result.unwrap_err() {
        TypeCreationError::UnacceptableBaseType { class_name } => {
            assert_eq!(class_name, "bool");
        }
        _ => panic!("Expected UnacceptableBaseType error"),
    }
}

#[test]
fn test_type_new_namespace_copied() {
    let registry = create_test_registry();
    let name = intern("AttrClass");
    let namespace = ClassDict::new();

    // Add some attributes
    let attr1 = intern("method1");
    let attr2 = intern("class_var");
    namespace.set(attr1.clone(), Value::int_unchecked(42));
    namespace.set(attr2.clone(), Value::int_unchecked(1));

    let result = type_new(name, &[], &namespace, &registry).unwrap();

    // Verify attributes were copied
    assert_eq!(
        result.class.get_attr(&attr1),
        Some(Value::int_unchecked(42))
    );
    assert_eq!(result.class.get_attr(&attr2), Some(Value::int_unchecked(1)));
}

#[test]
fn test_type_new_bitmap_has_object() {
    let registry = create_test_registry();
    let name = intern("BitmapClass");
    let namespace = ClassDict::new();

    let result = type_new(name, &[], &namespace, &registry).unwrap();

    // All classes should be subclass of object
    assert!(result.bitmap.is_subclass_of(TypeId::OBJECT));
}

#[test]
fn test_type_new_bitmap_has_self() {
    let registry = create_test_registry();
    let name = intern("SelfBitmapClass");
    let namespace = ClassDict::new();

    let result = type_new(name, &[], &namespace, &registry).unwrap();

    // Class should be in its own bitmap
    let self_type_id = TypeId::from_raw(result.class.class_id().0);
    assert!(result.bitmap.is_subclass_of(self_type_id));
}

#[test]
fn test_type_new_with_inheritance() {
    let registry = create_test_registry();

    // Create parent class
    let parent = PyClassObject::new_simple(intern("Parent"));
    let parent_id = parent.class_id();
    let parent = Arc::new(parent);

    // Create parent bitmap
    let mut parent_bitmap = SubclassBitmap::new();
    parent_bitmap.set_bit(TypeId::from_raw(parent_id.0));
    parent_bitmap.set_bit(TypeId::OBJECT);
    registry.register(parent, parent_bitmap.clone());

    // Create child class
    let name = intern("Child");
    let namespace = ClassDict::new();

    let result = type_new(name, &[parent_id], &namespace, &registry).unwrap();

    // Child should have parent in bitmap
    assert!(result.bitmap.is_subclass_of(TypeId::from_raw(parent_id.0)));
    assert!(result.bitmap.is_subclass_of(TypeId::OBJECT));
}

#[test]
fn test_type_creation_error_display() {
    let err = TypeCreationError::FinalBase {
        class_name: "MyFinal".to_string(),
    };
    assert_eq!(err.to_string(), "cannot subclass final class 'MyFinal'");

    let err = TypeCreationError::UnacceptableBaseType {
        class_name: "bool".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "type 'bool' is not an acceptable base type"
    );

    let err = TypeCreationError::MroError {
        message: "conflict".to_string(),
    };
    assert_eq!(err.to_string(), "MRO error: conflict");

    let err = TypeCreationError::BaseNotFound {
        class_id: ClassId(123),
    };
    assert!(err.to_string().contains("123"));

    let err = TypeCreationError::InvalidName {
        name: "".to_string(),
    };
    assert_eq!(err.to_string(), "invalid class name: ''");
}

#[test]
fn test_simple_registry_operations() {
    let registry = SimpleClassRegistry::new();

    // Create and register a class
    let class = Arc::new(PyClassObject::new_simple(intern("TestClass")));
    let class_id = class.class_id();
    let bitmap = SubclassBitmap::new();

    registry.register(class.clone(), bitmap.clone());

    // Test retrieval
    assert!(registry.get_class(class_id).is_some());
    assert!(registry.get_bitmap(class_id).is_some());

    // Test not found
    assert!(registry.get_class(ClassId(99999)).is_none());
    assert!(registry.get_bitmap(ClassId(99999)).is_none());
}

#[test]
fn test_register_global_class_publishes_heap_type_metadata() {
    let class = Arc::new(PyClassObject::new_simple(intern("PublishedHeapType")));
    let class_id = class.class_id();
    let type_id = class.class_type_id();

    register_global_class(Arc::clone(&class), bitmap_for_class(class.as_ref()));

    let published = global_registry()
        .get(type_id)
        .expect("heap type should be published into the dense type registry");
    assert_eq!(published.type_id(), type_id);
    assert_eq!(published.name.as_str(), "PublishedHeapType");
    assert!(published.flags.contains(TypeFlags::HEAPTYPE));
    assert_eq!(
        global_class(class_id)
            .as_ref()
            .map(|published_class| published_class.class_type_id()),
        Some(type_id)
    );
    assert!(
        global_class_bitmap(class_id)
            .expect("heap class bitmap should be published")
            .is_subclass_of(type_id)
    );

    unregister_global_class(class_id);
}

#[test]
fn test_registered_hierarchy_refreshes_published_layouts_and_versions() {
    let shared = intern("shared");

    let parent = Arc::new(PyClassObject::new_simple(intern("PublishedParent")));
    let parent_id = parent.class_id();
    register_global_class(Arc::clone(&parent), bitmap_for_class(parent.as_ref()));

    let child = Arc::new(
        PyClassObject::new(intern("PublishedChild"), &[parent_id], |id| {
            (id == parent_id).then(|| parent.mro().iter().copied().collect())
        })
        .expect("child class should build"),
    );
    let child_id = child.class_id();
    register_global_class(Arc::clone(&child), bitmap_for_class(child.as_ref()));

    let parent_version = global_class_version(parent_id).expect("parent version should exist");
    let child_version = global_class_version(child_id).expect("child version should exist");
    assert!(child.lookup_method_published(&shared).is_none());

    parent.set_attr(shared.clone(), Value::int_unchecked(10));

    let inherited = child
        .lookup_method_published(&shared)
        .expect("child layout should refresh inherited members after parent mutation");
    assert_eq!(inherited.value, Value::int_unchecked(10));
    assert_eq!(inherited.defining_class, parent_id);
    assert_eq!(inherited.mro_index, 1);
    assert!(
        global_class_version(parent_id).expect("parent version should update") > parent_version
    );
    let child_version_after_parent =
        global_class_version(child_id).expect("child version should update");
    assert!(child_version_after_parent > child_version);

    child.set_attr(shared.clone(), Value::int_unchecked(20));

    let overridden = child
        .lookup_method_published(&shared)
        .expect("child layout should prefer direct override");
    assert_eq!(overridden.value, Value::int_unchecked(20));
    assert_eq!(overridden.defining_class, child_id);
    assert_eq!(overridden.mro_index, 0);
    assert!(
        global_class_version(child_id).expect("child version should advance again")
            > child_version_after_parent
    );

    assert_eq!(
        child.del_attr(&shared),
        Some(Value::int_unchecked(20)),
        "deleting the override should fall back to the parent publication"
    );
    let fallback = child
        .lookup_method_published(&shared)
        .expect("published layout should fall back to the parent after delete");
    assert_eq!(fallback.value, Value::int_unchecked(10));
    assert_eq!(fallback.defining_class, parent_id);

    unregister_global_class(child_id);
    unregister_global_class(parent_id);
}
