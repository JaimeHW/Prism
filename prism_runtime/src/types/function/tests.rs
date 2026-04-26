use super::*;
use prism_core::intern::intern;

fn make_test_code() -> Arc<CodeObject> {
    let mut code = CodeObject::new("test", "test.py");
    code.arg_count = 2;
    code.register_count = 4;
    Arc::new(code)
}

#[test]
fn test_function_creation() {
    let code = make_test_code();
    let func = FunctionObject::new(code, "my_func".into(), None, None);
    assert_eq!(func.arg_count(), 2);
    assert_eq!(&*func.name, "my_func");
}

#[test]
fn test_closure_env() {
    let values: Box<[Value]> = vec![Value::int(42).unwrap(), Value::int(100).unwrap()].into();
    let env = ClosureEnv::from_values(values, None);
    assert_eq!(env.len(), 2);
    assert_eq!(env.get(0).as_int(), Some(42));
    assert_eq!(env.get(1).as_int(), Some(100));
}

#[test]
fn test_closure_chain() {
    let outer: Box<[Value]> = vec![Value::int(1).unwrap()].into();
    let outer_env = Arc::new(ClosureEnv::from_values(outer, None));

    let inner: Box<[Value]> = vec![Value::int(2).unwrap()].into();
    let inner_env = ClosureEnv::from_values(inner, Some(outer_env));

    assert_eq!(inner_env.get_chain(0, 0).unwrap().as_int(), Some(2));
    assert_eq!(inner_env.get_chain(1, 0).unwrap().as_int(), Some(1));
}

#[test]
fn test_closure_env_shares_cells_across_clones() {
    let cell = Arc::new(Cell::new(Value::int(41).unwrap()));
    let env = ClosureEnv::new(vec![Arc::clone(&cell)]);
    let cloned = env.clone();

    cloned.set(0, Value::int(42).unwrap());

    assert_eq!(env.get(0).as_int(), Some(42));
    assert_eq!(cell.get().unwrap().as_int(), Some(42));
}

#[test]
fn test_closure_env_overflow_supports_large_cell_counts() {
    let env = ClosureEnv::with_unbound_cells(300);

    assert_eq!(env.len(), 300);
    assert!(!env.is_inline());
    env.set(299, Value::int(299).unwrap());
    assert_eq!(env.get(299).as_int(), Some(299));
}

#[test]
fn test_function_attr_dict_materialization_preserves_existing_attrs() {
    let func = FunctionObject::new(make_test_code(), "dict_func".into(), None, None);
    func.set_attr(intern("copied"), Value::int(7).unwrap());

    let dict_ptr = func
        .ensure_attr_dict(|dict| Ok::<*mut DictObject, ()>(Box::into_raw(Box::new(dict))))
        .expect("dict allocation should succeed");
    let dict = unsafe { &*dict_ptr };

    assert_eq!(
        dict.get(Value::string(intern("copied"))).unwrap().as_int(),
        Some(7)
    );
    assert_eq!(func.get_attr(&intern("copied")).unwrap().as_int(), Some(7));
}

#[test]
fn test_function_attr_reads_follow_materialized_dict_mutations() {
    let func = FunctionObject::new(make_test_code(), "dict_func".into(), None, None);
    let dict_ptr = func
        .ensure_attr_dict(|dict| Ok::<*mut DictObject, ()>(Box::into_raw(Box::new(dict))))
        .expect("dict allocation should succeed");

    unsafe { &mut *dict_ptr }.set(Value::string(intern("dynamic")), Value::int(11).unwrap());

    assert_eq!(
        func.get_attr(&intern("dynamic")).unwrap().as_int(),
        Some(11)
    );
    assert!(func.has_attr(&intern("dynamic")));
}
