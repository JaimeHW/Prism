
use super::*;
use prism_core::intern::intern;

fn str_val(s: &str) -> Value {
    Value::string(intern(s))
}

fn int(i: i64) -> Value {
    Value::int_unchecked(i)
}

// =========================================================================
// WrapperMetadata construction tests
// =========================================================================

#[test]
fn test_new_empty() {
    let meta = WrapperMetadata::new();
    assert!(!meta.has_any());
    assert_eq!(meta.count(), 0);
}

#[test]
fn test_from_wrapped() {
    let meta = WrapperMetadata::from_wrapped(
        Some(str_val("mymodule")),
        Some(str_val("myfunc")),
        Some(str_val("MyClass.myfunc")),
        Some(str_val("Does stuff")),
        int(42), // function reference placeholder
    );
    assert!(meta.has_any());
    assert_eq!(meta.count(), 5); // module, name, qualname, doc, wrapped
}

#[test]
fn test_full() {
    let meta = WrapperMetadata::full(
        str_val("mod"),
        str_val("func"),
        str_val("mod.func"),
        str_val("docstring"),
        int(0),
        None,
    );
    assert_eq!(meta.count(), 5); // annotations is None
}

// =========================================================================
// Attribute access tests
// =========================================================================

#[test]
fn test_get_attr_module() {
    let meta = WrapperMetadata::from_wrapped(Some(str_val("mymod")), None, None, None, int(0));
    assert!(meta.get_attr("__module__").is_some());
    assert!(meta.get_attr("__name__").is_none());
}

#[test]
fn test_get_attr_all_fields() {
    let meta = WrapperMetadata::full(
        str_val("mod"),
        str_val("fn"),
        str_val("mod.fn"),
        str_val("doc"),
        int(0),
        Some(int(99)),
    );
    assert!(meta.get_attr("__module__").is_some());
    assert!(meta.get_attr("__name__").is_some());
    assert!(meta.get_attr("__qualname__").is_some());
    assert!(meta.get_attr("__doc__").is_some());
    assert!(meta.get_attr("__wrapped__").is_some());
    assert!(meta.get_attr("__annotations__").is_some());
}

#[test]
fn test_get_attr_unknown() {
    let meta = WrapperMetadata::new();
    assert!(meta.get_attr("__unknown__").is_none());
}

#[test]
fn test_set_attr() {
    let mut meta = WrapperMetadata::new();
    assert!(meta.set_attr("__name__", str_val("test")));
    assert_eq!(meta.count(), 1);
    assert!(meta.get_attr("__name__").is_some());
}

#[test]
fn test_set_attr_unknown() {
    let mut meta = WrapperMetadata::new();
    assert!(!meta.set_attr("__unknown__", int(0)));
}

#[test]
fn test_set_attr_overwrites() {
    let mut meta = WrapperMetadata::new();
    meta.set_attr("__doc__", str_val("old"));
    meta.set_attr("__doc__", str_val("new"));
    // Should still count as 1
    assert_eq!(meta.count(), 1);
}

// =========================================================================
// dir tests
// =========================================================================

#[test]
fn test_dir_empty() {
    let meta = WrapperMetadata::new();
    assert!(meta.dir().is_empty());
}

#[test]
fn test_dir_partial() {
    let mut meta = WrapperMetadata::new();
    meta.set_attr("__name__", str_val("f"));
    meta.set_attr("__doc__", str_val("d"));

    let dir = meta.dir();
    assert_eq!(dir.len(), 2);
    assert!(dir.contains(&"__name__"));
    assert!(dir.contains(&"__doc__"));
}

#[test]
fn test_dir_full() {
    let meta = WrapperMetadata::full(
        str_val("m"),
        str_val("n"),
        str_val("q"),
        str_val("d"),
        int(0),
        Some(int(1)),
    );
    assert_eq!(meta.dir().len(), 6);
}

// =========================================================================
// update_wrapper tests
// =========================================================================

#[test]
fn test_update_wrapper_copies_all() {
    let source = WrapperMetadata::full(
        str_val("source_mod"),
        str_val("source_fn"),
        str_val("source_mod.source_fn"),
        str_val("source docs"),
        int(0),
        None,
    );
    let mut dest = WrapperMetadata::new();

    update_wrapper(&mut dest, &source, None);

    // WRAPPER_ASSIGNMENTS: module, name, qualname, annotations, doc
    assert!(dest.get_attr("__module__").is_some());
    assert!(dest.get_attr("__name__").is_some());
    assert!(dest.get_attr("__qualname__").is_some());
    assert!(dest.get_attr("__doc__").is_some());
    // __wrapped__ is NOT in WRAPPER_ASSIGNMENTS
    assert!(dest.get_attr("__wrapped__").is_none());
}

#[test]
fn test_update_wrapper_custom_assignments() {
    let source = WrapperMetadata::full(
        str_val("mod"),
        str_val("fn"),
        str_val("mod.fn"),
        str_val("doc"),
        int(0),
        None,
    );
    let mut dest = WrapperMetadata::new();

    // Only copy name and doc
    update_wrapper(&mut dest, &source, Some(&["__name__", "__doc__"]));

    assert!(dest.get_attr("__name__").is_some());
    assert!(dest.get_attr("__doc__").is_some());
    assert!(dest.get_attr("__module__").is_none());
}

#[test]
fn test_update_wrapper_skips_missing() {
    let mut source = WrapperMetadata::new();
    source.set_attr("__name__", str_val("fn"));
    // module, qualname, doc are NOT set

    let mut dest = WrapperMetadata::new();
    update_wrapper(&mut dest, &source, None);

    // Only __name__ should be copied
    assert!(dest.get_attr("__name__").is_some());
    assert!(dest.get_attr("__module__").is_none());
}

#[test]
fn test_update_wrapper_overwrites_dest() {
    let mut source = WrapperMetadata::new();
    source.set_attr("__name__", str_val("original"));

    let mut dest = WrapperMetadata::new();
    dest.set_attr("__name__", str_val("wrapper"));

    update_wrapper(&mut dest, &source, None);

    // Source should have overwritten dest
    let name = dest.get_attr("__name__").unwrap();
    assert!(name.is_string());
}

// =========================================================================
// Constants tests
// =========================================================================

#[test]
fn test_wrapper_assignments_contents() {
    assert!(WRAPPER_ASSIGNMENTS.contains(&"__module__"));
    assert!(WRAPPER_ASSIGNMENTS.contains(&"__name__"));
    assert!(WRAPPER_ASSIGNMENTS.contains(&"__qualname__"));
    assert!(WRAPPER_ASSIGNMENTS.contains(&"__annotations__"));
    assert!(WRAPPER_ASSIGNMENTS.contains(&"__doc__"));
    assert_eq!(WRAPPER_ASSIGNMENTS.len(), 5);
}

#[test]
fn test_wrapper_updates_contents() {
    assert!(WRAPPER_UPDATES.contains(&"__dict__"));
    assert_eq!(WRAPPER_UPDATES.len(), 1);
}

// =========================================================================
// Clone/Default tests
// =========================================================================

#[test]
fn test_metadata_clone() {
    let meta = WrapperMetadata::full(
        str_val("m"),
        str_val("n"),
        str_val("q"),
        str_val("d"),
        int(0),
        None,
    );
    let clone = meta.clone();
    assert_eq!(clone.count(), meta.count());
}
