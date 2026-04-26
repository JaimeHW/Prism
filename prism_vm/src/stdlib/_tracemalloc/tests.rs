use super::*;

#[test]
fn test_tracemalloc_reports_inactive_state() {
    assert_eq!(
        tracemalloc_is_tracing(&[])
            .expect("is_tracing should succeed")
            .as_bool(),
        Some(false)
    );
    let traced = tracemalloc_get_traced_memory(&[]).expect("get_traced_memory should succeed");
    assert!(traced.as_object_ptr().is_some());
}

#[test]
fn test_tracemalloc_module_exports_cpython_bootstrap_surface() {
    let module = TraceMallocModule::new();
    for name in [
        "is_tracing",
        "start",
        "stop",
        "clear_traces",
        "get_traceback_limit",
        "get_traced_memory",
        "_get_object_traceback",
        "_get_traces",
    ] {
        assert!(module.get_attr(name).is_ok(), "missing {name}");
    }
}
