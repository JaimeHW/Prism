use super::*;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::{ListObject, object_ptr_as_list_mut};
use std::path::PathBuf;
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
            "prism_import_resolver_tests_{}_{}_{}",
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
    std::fs::write(path, content).expect("failed to write test file");
}

#[test]
fn test_import_resolver_new() {
    let resolver = ImportResolver::new();
    assert!(resolver.cached_modules().is_empty());
}

#[test]
fn test_import_stdlib_math() {
    let resolver = ImportResolver::new();
    let result = resolver.import_module("math");
    assert!(result.is_ok());

    let module = result.unwrap();
    assert_eq!(module.name(), "math");
}

#[test]
fn test_import_stdlib_os() {
    let resolver = ImportResolver::new();
    let result = resolver.import_module("os");
    assert!(result.is_ok());

    let module = result.unwrap();
    assert_eq!(module.name(), "os");
}

#[test]
fn test_import_stdlib_builtins() {
    let resolver = ImportResolver::new();
    let module = resolver
        .import_module("builtins")
        .expect("builtins import should succeed");

    assert_eq!(module.name(), "builtins");
    assert!(module.get_attr("open").is_some());
    assert!(module.get_attr("len").is_some());
}

#[test]
fn test_import_stdlib_sys() {
    let resolver = ImportResolver::new();
    let result = resolver.import_module("sys");
    assert!(result.is_ok());

    let module = result.unwrap();
    assert_eq!(module.name(), "sys");
}

#[test]
fn test_import_nonexistent() {
    let resolver = ImportResolver::new();
    let result = resolver.import_module("nonexistent_module_12345");
    assert!(result.is_err());

    match result.unwrap_err() {
        ImportError::ModuleNotFound { module } => {
            assert_eq!(module.as_ref(), "nonexistent_module_12345");
        }
        _ => panic!("Expected ModuleNotFound error"),
    }
}

#[test]
fn test_import_caching() {
    let resolver = ImportResolver::new();

    // First import
    let math1 = resolver.import_module("math").unwrap();

    // Second import should return cached module
    let math2 = resolver.import_module("math").unwrap();

    // Should be the same Arc (pointer equality)
    assert!(Arc::ptr_eq(&math1, &math2));
}

#[test]
fn test_cloned_resolvers_share_interpreter_import_state() {
    let resolver = ImportResolver::new();
    let cloned = resolver.clone();

    let module = Arc::new(ModuleObject::new("shared_module"));
    resolver.insert_module("shared_module", Arc::clone(&module));

    let cached = cloned
        .get_cached("shared_module")
        .expect("cloned resolver should see cached module");
    assert!(Arc::ptr_eq(&cached, &module));

    cloned.add_search_path(Arc::from("thread-visible-path"));
    assert!(
        resolver
            .search_paths()
            .iter()
            .any(|path| path.as_ref() == "thread-visible-path")
    );

    let sys_from_parent = resolver.import_module("sys").unwrap();
    let sys_from_clone = cloned.import_module("sys").unwrap();
    assert!(Arc::ptr_eq(&sys_from_parent, &sys_from_clone));
}

#[test]
fn test_get_cached() {
    let resolver = ImportResolver::new();

    // Not cached yet
    assert!(resolver.get_cached("math").is_none());

    // Import it
    resolver.import_module("math").unwrap();

    // Now cached
    assert!(resolver.get_cached("math").is_some());
}

#[test]
fn test_insert_module() {
    let resolver = ImportResolver::new();
    let module = Arc::new(ModuleObject::new("custom_module"));

    resolver.insert_module("custom_module", Arc::clone(&module));

    let cached = resolver.get_cached("custom_module");
    assert!(cached.is_some());
    assert!(Arc::ptr_eq(&cached.unwrap(), &module));
}

#[test]
fn test_remove_module() {
    let resolver = ImportResolver::new();

    // Import math
    resolver.import_module("math").unwrap();
    assert!(resolver.get_cached("math").is_some());

    // Remove it
    let removed = resolver.remove_module("math");
    assert!(removed.is_some());
    assert!(resolver.get_cached("math").is_none());
}

#[test]
fn test_module_exists() {
    let resolver = ImportResolver::new();

    // Stdlib modules exist
    assert!(resolver.module_exists("math"));
    assert!(resolver.module_exists("os"));
    assert!(resolver.module_exists("sys"));

    // Unknown modules don't exist
    assert!(!resolver.module_exists("unknown_module_xyz"));
}

#[test]
fn test_frozen_module_registry() {
    let resolver = ImportResolver::new();
    let code = Arc::new(prism_code::CodeObject::new("<module>", "<frozen>"));

    resolver.insert_frozen_module(
        "pkg.helper",
        FrozenModuleSource::new(code.clone(), "<frozen>", "pkg", false),
    );

    let frozen = resolver
        .get_frozen_module("pkg.helper")
        .expect("frozen module should be available");
    assert_eq!(frozen.filename.as_ref(), "<frozen>");
    assert_eq!(frozen.package_name.as_ref(), "pkg");
    assert!(Arc::ptr_eq(&frozen.code, &code));
    assert!(resolver.module_exists("pkg.helper"));

    let removed = resolver
        .remove_frozen_module("pkg.helper")
        .expect("frozen module should be removable");
    assert!(Arc::ptr_eq(&removed.code, &code));
    assert!(resolver.get_frozen_module("pkg.helper").is_none());
}

#[test]
fn test_import_from() {
    let resolver = ImportResolver::new();
    let math = resolver.import_module("math").unwrap();

    // Import pi from math
    let result = resolver.import_from(&math, "pi");
    assert!(result.is_ok());

    let pi = result.unwrap();
    if let Some(f) = pi.as_float() {
        assert!((f - std::f64::consts::PI).abs() < 1e-10);
    } else {
        panic!("Expected float value for pi");
    }
}

#[test]
fn test_import_from_builtins() {
    let resolver = ImportResolver::new();
    let builtins = resolver
        .import_module("builtins")
        .expect("builtins import should succeed");

    let open = resolver
        .import_from(&builtins, "open")
        .expect("from builtins import open should succeed");
    assert!(open.as_object_ptr().is_some());
}

#[test]
fn test_import_from_nonexistent() {
    let resolver = ImportResolver::new();
    let math = resolver.import_module("math").unwrap();

    let result = resolver.import_from(&math, "nonexistent_attr");
    assert!(result.is_err());

    match result.unwrap_err() {
        ImportError::ImportFromError { module, name } => {
            assert_eq!(module.as_ref(), "math");
            assert_eq!(name.as_ref(), "nonexistent_attr");
        }
        _ => panic!("Expected ImportFromError"),
    }
}

#[test]
fn test_import_star() {
    let resolver = ImportResolver::new();
    let math = resolver.import_module("math").unwrap();

    let result = resolver.import_star(&math);
    assert!(result.is_ok());

    let names = result.unwrap();
    // Should have some public names
    assert!(!names.is_empty());

    // Check for expected names
    let name_strs: Vec<&str> = names.iter().map(|(k, _)| k.as_ref()).collect();
    assert!(name_strs.contains(&"pi") || name_strs.contains(&"e"));
}

#[test]
fn test_import_dotted_stdlib_submodule() {
    let resolver = ImportResolver::new();
    let module = resolver
        .import_dotted("os.path")
        .expect("os.path import should succeed");

    assert_eq!(module.name(), "os.path");

    let os = resolver
        .import_module("os")
        .expect("os import should succeed");
    let path_value = resolver
        .import_from(&os, "path")
        .expect("from os import path should resolve submodule");
    let path_ptr = path_value
        .as_object_ptr()
        .expect("os.path should be exposed as module object");
    let resolved = resolver
        .module_from_ptr(path_ptr)
        .expect("module pointer should resolve");
    assert_eq!(resolved.name(), "os.path");
}

#[test]
fn test_search_paths() {
    let resolver = ImportResolver::new();

    assert!(resolver.search_paths().is_empty());

    resolver.add_search_path(Arc::from("/usr/lib/python"));
    resolver.add_search_path(Arc::from("/home/user/lib"));

    let paths = resolver.search_paths();
    assert_eq!(paths.len(), 2);
}

#[test]
fn test_search_paths_include_runtime_sys_path_mutations_first() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("dynamic_path_module.py"), "VALUE = 1\n");

    let resolver = ImportResolver::with_paths(vec![Arc::from("configured")]);
    let sys = resolver
        .import_module("sys")
        .expect("sys import should succeed");
    let path_value = sys.get_attr("path").expect("sys.path should exist");
    let path_ptr = path_value
        .as_object_ptr()
        .expect("sys.path should be a list object") as *mut ();
    let path_list = object_ptr_as_list_mut(path_ptr).expect("sys.path should be mutable");
    let temp_path = temp.path.to_string_lossy().into_owned();
    path_list.insert(0, Value::string(intern(&temp_path)));

    let paths = resolver.search_paths();
    assert_eq!(paths.first().map(Arc::as_ref), Some(temp_path.as_str()));
    assert!(paths.iter().any(|path| path.as_ref() == "configured"));

    let resolved = resolver
        .resolve_source_location("dynamic_path_module")
        .expect("runtime sys.path mutation should be visible to imports");
    assert_eq!(resolved.path, temp.path.join("dynamic_path_module.py"));
    assert!(!resolved.is_package);
}

#[test]
fn test_with_paths() {
    let paths = vec![Arc::from("/path1"), Arc::from("/path2")];
    let resolver = ImportResolver::with_paths(paths);

    assert_eq!(resolver.search_paths().len(), 2);
}

#[test]
fn test_resolve_source_location_finds_dotted_module() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg").join("__init__.py"), "");
    write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 1\n");

    let resolver =
        ImportResolver::with_paths(vec![Arc::from(temp.path.to_string_lossy().into_owned())]);
    let resolved = resolver
        .resolve_source_location("pkg.helper")
        .expect("expected dotted module source");

    assert_eq!(resolved.path, temp.path.join("pkg").join("helper.py"));
    assert!(!resolved.is_package);
}

#[test]
fn test_source_first_policy_for_fallback_stdlib_module() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("re.py"), "VALUE = 1\n");
    write_file(&temp.path.join("os").join("__init__.py"), "");
    write_file(&temp.path.join("os").join("path.py"), "VALUE = 2\n");
    write_file(&temp.path.join("sys.py"), "VALUE = 3\n");

    let resolver =
        ImportResolver::with_paths(vec![Arc::from(temp.path.to_string_lossy().into_owned())]);

    assert!(resolver.should_load_from_source_first("re"));
    assert!(resolver.should_load_from_source_first("os"));
    assert!(resolver.should_load_from_source_first("os.path"));
    assert!(!resolver.should_load_from_source_first("sys"));
    assert!(!resolver.should_load_from_source_first("math"));
}

#[test]
fn test_resolve_load_plan_prefers_source_for_fallback_stdlib_when_available() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("re.py"), "VALUE = 1\n");

    let resolver =
        ImportResolver::with_paths(vec![Arc::from(temp.path.to_string_lossy().into_owned())]);

    match resolver.resolve_load_plan("re") {
        ImportLoadPlan::Source(location) => {
            assert_eq!(location.path, temp.path.join("re.py"));
            assert!(!location.is_package);
        }
        plan => panic!("expected source load plan for re.py, got {plan:?}"),
    }
}

#[test]
fn test_resolve_load_plan_uses_native_for_builtin_without_source_override() {
    let resolver = ImportResolver::new();

    match resolver.resolve_load_plan("re") {
        ImportLoadPlan::Native => {}
        plan => panic!("expected native load plan for re without source path, got {plan:?}"),
    }
}

#[test]
fn test_with_sys_args_populates_imported_sys_argv() {
    let resolver = ImportResolver::with_sys_args(vec!["prog.py".to_string(), "--fast".to_string()]);
    let sys = resolver
        .import_module("sys")
        .expect("sys import should succeed");
    let argv = sys.get_attr("argv").expect("sys.argv should be present");

    let argv_ptr = argv
        .as_object_ptr()
        .expect("sys.argv should be represented as list object");
    let list = unsafe { &*(argv_ptr as *const ListObject) };
    assert_eq!(list.len(), 2);

    let arg0 = list.get(0).expect("argv[0] should exist");
    let arg1 = list.get(1).expect("argv[1] should exist");

    let arg0_ptr = arg0
        .as_string_object_ptr()
        .expect("argv[0] should be string") as *const u8;
    let arg1_ptr = arg1
        .as_string_object_ptr()
        .expect("argv[1] should be string") as *const u8;

    assert_eq!(
        interned_by_ptr(arg0_ptr)
            .expect("argv[0] should resolve")
            .as_ref(),
        "prog.py"
    );
    assert_eq!(
        interned_by_ptr(arg1_ptr)
            .expect("argv[1] should resolve")
            .as_ref(),
        "--fast"
    );
}

#[test]
fn test_imported_sys_exposes_live_modules_dict() {
    let resolver = ImportResolver::new();
    let sys = resolver
        .import_module("sys")
        .expect("sys import should succeed");
    let modules = sys
        .get_attr("modules")
        .expect("sys.modules should be injected");
    let modules_ptr = modules
        .as_object_ptr()
        .expect("sys.modules should be represented as dict object");

    let math = resolver
        .import_module("math")
        .expect("math import should succeed");
    let dict = unsafe { &*(modules_ptr as *const DictObject) };
    let value = dict
        .get(Value::string(intern("math")))
        .expect("math should appear in sys.modules");

    assert_eq!(value.as_object_ptr(), Some(Arc::as_ptr(&math) as *const ()));
}

#[test]
fn test_public_sys_modules_aliases_are_visible_to_cache_and_imports() {
    let resolver = ImportResolver::new();
    let sys = resolver
        .import_module("sys")
        .expect("sys import should succeed");
    let alias_target = Arc::new(ModuleObject::new("ntpath"));
    resolver.insert_module("ntpath", Arc::clone(&alias_target));

    let modules = sys
        .get_attr("modules")
        .expect("sys.modules should be injected");
    let modules_ptr = modules
        .as_object_ptr()
        .expect("sys.modules should be represented as dict object");
    let dict = unsafe { &mut *(modules_ptr as *mut DictObject) };
    dict.set(
        Value::string(intern("os.path")),
        Value::object_ptr(Arc::as_ptr(&alias_target) as *const ()),
    );

    let cached = resolver
        .get_cached("os.path")
        .expect("sys.modules alias should resolve");
    assert!(Arc::ptr_eq(&cached, &alias_target));

    let imported = resolver
        .import_dotted("os.path")
        .expect("import_dotted should honor sys.modules alias");
    assert!(Arc::ptr_eq(&imported, &alias_target));
}

#[test]
fn test_module_from_ptr_resolves_cached_module() {
    let resolver = ImportResolver::new();
    let math = resolver
        .import_module("math")
        .expect("math import should succeed");
    let ptr = Arc::as_ptr(&math) as *const ();

    let resolved = resolver
        .module_from_ptr(ptr)
        .expect("module pointer should resolve");
    assert_eq!(resolved.name(), "math");
    assert!(Arc::ptr_eq(&math, &resolved));
}

#[test]
fn test_cached_modules() {
    let resolver = ImportResolver::new();

    resolver.import_module("math").unwrap();
    resolver.import_module("os").unwrap();

    let cached = resolver.cached_modules();
    assert_eq!(cached.len(), 2);
}

#[test]
fn test_concurrent_imports() {
    use std::thread;

    let resolver = Arc::new(ImportResolver::new());

    // Spawn 10 threads all trying to import "math" simultaneously
    // The first thread to win the race will load the module,
    // other threads will wait via Condvar and return the cached result
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let r = Arc::clone(&resolver);
            thread::spawn(move || r.import_module("math").unwrap())
        })
        .collect();

    let modules: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All should be the same cached module
    for i in 1..modules.len() {
        assert!(Arc::ptr_eq(&modules[0], &modules[i]));
    }
}
