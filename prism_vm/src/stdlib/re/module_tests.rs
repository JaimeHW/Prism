
use super::*;

#[test]
fn test_re_module_exposes_callable_entries_and_types() {
    let module = ReModule::new();
    assert!(
        module
            .get_attr("compile")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(module.get_attr("search").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("Pattern")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(module.get_attr("Match").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("error").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("findall")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("finditer")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(module.get_attr("sub").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("subn").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("split").unwrap().as_object_ptr().is_some());
}
