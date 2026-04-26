use super::*;
use prism_gc::heap::GcHeap;
use prism_runtime::allocation_context::RuntimeHeapBinding;

#[test]
fn test_module_new() {
    let module = ModuleObject::new("test_module");
    assert_eq!(module.name(), "test_module");
    assert_eq!(module.header.type_id, TypeId::MODULE);
    assert!(module.has_attr("__name__"));
    assert!(module.has_attr("__doc__"));
    assert!(module.has_attr("__loader__"));
    assert!(module.has_attr("__package__"));
    assert!(module.has_attr("__spec__"));
    assert!(module.get_attr("__doc__").unwrap().is_none());
}

#[test]
fn test_module_get_set_attr() {
    let module = ModuleObject::new("test");
    module.set_attr("foo", Value::int(42).unwrap());
    assert!(module.has_attr("foo"));
    let val = module.get_attr("foo").unwrap();
    assert_eq!(val.as_int(), Some(42));
}

#[test]
fn test_module_dict_value_uses_bound_runtime_heap() {
    let heap = GcHeap::with_defaults();
    let _binding = RuntimeHeapBinding::register(&heap);
    let module = ModuleObject::new("test");
    module.set_attr("foo", Value::int(42).unwrap());

    let value = module.dict_value();
    let ptr = value
        .as_object_ptr()
        .expect("module __dict__ should be an object pointer");

    assert!(heap.contains(ptr));
}

#[test]
fn test_module_dict_is_live_namespace() {
    let module = ModuleObject::new("test");
    module.set_attr("existing", Value::int(1).unwrap());

    let dict_value = module.dict_value();
    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("module __dict__ should be a dict object");
    let dict = unsafe { &mut *(dict_ptr as *mut DictObject) };

    dict.set(Value::string(intern("injected")), Value::int(2).unwrap());
    assert_eq!(
        module.get_attr("injected").and_then(|v| v.as_int()),
        Some(2)
    );

    module.set_attr("published", Value::int(3).unwrap());
    assert_eq!(
        dict.get(Value::string(intern("published")))
            .and_then(|v| v.as_int()),
        Some(3)
    );

    assert!(module.del_attr("published"));
    assert!(dict.get(Value::string(intern("published"))).is_none());
}

#[test]
fn test_module_del_attr() {
    let module = ModuleObject::new("test");
    module.set_attr("bar", Value::int(100).unwrap());
    assert!(module.has_attr("bar"));
    assert!(module.del_attr("bar"));
    assert!(!module.has_attr("bar"));
}

#[test]
fn test_module_del_nonexistent() {
    let module = ModuleObject::new("test");
    assert!(!module.del_attr("nonexistent"));
}

#[test]
fn test_module_dir() {
    let module = ModuleObject::new("test");
    module.set_attr("alpha", Value::int(1).unwrap());
    module.set_attr("beta", Value::int(2).unwrap());
    let names = module.dir();
    // Should have __name__, alpha, beta
    assert!(names.len() >= 3);
}

#[test]
fn test_module_public_names() {
    let module = ModuleObject::new("test");
    module.set_attr("public", Value::int(1).unwrap());
    module.set_attr("_private", Value::int(2).unwrap());
    module.set_attr("__dunder__", Value::int(3).unwrap());

    let public = module.public_names().expect("public names should resolve");
    // Should contain "public" but not "_private" or "__dunder__"
    let public_strs: Vec<&str> = public.iter().map(|s| s.as_ref()).collect();
    assert!(public_strs.contains(&"public"));
    assert!(!public_strs.contains(&"_private"));
    assert!(!public_strs.contains(&"__dunder__"));
}

#[test]
fn test_module_with_metadata() {
    let module = ModuleObject::with_metadata(
        "mymodule",
        Some(Arc::from("Module documentation")),
        Some(Arc::from("/path/to/module.py")),
        Some(Arc::from("mypackage")),
    );

    assert_eq!(module.name(), "mymodule");
    assert!(module.has_attr("__name__"));
    assert!(module.has_attr("__doc__"));
    assert!(module.has_attr("__file__"));
    assert!(module.has_attr("__package__"));
}

#[test]
fn test_module_with_metadata_preserves_doc_binding_when_doc_is_absent() {
    let module = ModuleObject::with_metadata(
        "mymodule",
        None,
        Some(Arc::from("/path/to/module.py")),
        Some(Arc::from("mypackage")),
    );

    assert!(module.has_attr("__doc__"));
    assert!(module.get_attr("__doc__").unwrap().is_none());
}

#[test]
fn test_package_metadata_sets_path_from_init_file() {
    let (file, expected_dir) = if cfg!(windows) {
        (Arc::from(r"C:\work\pkg\__init__.py"), r"C:\work\pkg")
    } else {
        (Arc::from("/work/pkg/__init__.py"), "/work/pkg")
    };
    let module = ModuleObject::with_metadata("pkg", None, Some(file), Some(Arc::from("pkg")));

    let path_value = module.get_attr("__path__").expect("__path__ should exist");
    let path_ptr = path_value
        .as_object_ptr()
        .expect("__path__ should be a list object");
    let list = unsafe { &*(path_ptr as *const ListObject) };
    assert_eq!(list.len(), 1);

    let first = list.as_slice()[0];
    let first = prism_runtime::types::string::value_as_string_ref(first)
        .expect("package search path should contain strings");
    assert_eq!(first.as_str(), expected_dir);
}

#[test]
fn test_module_len_and_is_empty() {
    let module = ModuleObject::new("test");
    // Has at least __name__
    assert!(!module.is_empty());
    assert!(module.len() >= 1);
}

#[test]
fn test_module_all_attrs() {
    let module = ModuleObject::new("test");
    module.set_attr("x", Value::int(10).unwrap());
    module.set_attr("y", Value::int(20).unwrap());

    let attrs = module.all_attrs();
    assert!(attrs.len() >= 3); // __name__, x, y
}

#[test]
fn test_module_public_attrs() {
    let module = ModuleObject::new("test");
    module.set_attr("public_var", Value::int(1).unwrap());
    module.set_attr("_hidden", Value::int(2).unwrap());

    let public = module.public_attrs().expect("public attrs should resolve");
    let names: Vec<&str> = public.iter().map(|(k, _)| k.as_ref()).collect();
    assert!(names.contains(&"public_var"));
    assert!(!names.contains(&"_hidden"));
}

#[test]
fn test_module_public_attrs_honors_tuple_all_exactly() {
    let module = ModuleObject::new("test");
    module.set_attr("visible", Value::int(1).unwrap());
    module.set_attr("also_visible", Value::int(2).unwrap());
    module.set_attr("not_exported", Value::int(3).unwrap());
    module.set_attr("_explicit", Value::int(4).unwrap());
    module.set_attr("__all__", tuple_value(&["_explicit", "visible"]));

    let public = module
        .public_attrs()
        .expect("__all__ names should resolve to attributes");
    let names: Vec<&str> = public.iter().map(|(name, _)| name.as_ref()).collect();

    assert_eq!(names, vec!["_explicit", "visible"]);
    assert_eq!(public[0].1.as_int(), Some(4));
    assert_eq!(public[1].1.as_int(), Some(1));
}

#[test]
fn test_module_public_names_honors_list_all() {
    let module = ModuleObject::new("test");
    module.set_attr("first", Value::int(1).unwrap());
    module.set_attr("second", Value::int(2).unwrap());
    module.set_attr("__all__", list_value(&["second", "first"]));

    let names = module
        .public_names()
        .expect("__all__ names should be accepted from lists");
    let names: Vec<&str> = names.iter().map(|name| name.as_ref()).collect();

    assert_eq!(names, vec!["second", "first"]);
}

#[test]
fn test_module_public_attrs_reports_missing_all_member() {
    let module = ModuleObject::new("test");
    module.set_attr("__all__", tuple_value(&["missing"]));

    let err = module
        .public_attrs()
        .expect_err("missing __all__ member should be an error");

    assert_eq!(
        err,
        ModuleExportError::MissingAllAttribute {
            module: Arc::from("test"),
            name: Arc::from("missing"),
        }
    );
}

#[test]
fn test_module_concurrent_access() {
    use std::thread;

    let module = Arc::new(ModuleObject::new("concurrent"));

    // Spawn multiple threads to read/write
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let m = Arc::clone(&module);
            thread::spawn(move || {
                m.set_attr(&format!("attr_{}", i), Value::int(i).unwrap());
                m.get_attr(&format!("attr_{}", i))
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // All attributes should exist
    for i in 0..10 {
        assert!(module.has_attr(&format!("attr_{}", i)));
    }
}

fn tuple_value(names: &[&str]) -> Value {
    let items: Vec<Value> = names
        .iter()
        .map(|name| Value::string(intern(name)))
        .collect();
    Value::object_ptr(Box::into_raw(Box::new(TupleObject::from_vec(items))) as *const ())
}

fn list_value(names: &[&str]) -> Value {
    let items: Vec<Value> = names
        .iter()
        .map(|name| Value::string(intern(name)))
        .collect();
    Value::object_ptr(Box::into_raw(Box::new(ListObject::from_iter(items))) as *const ())
}
