use super::*;

fn intern(s: &str) -> InternedString {
    prism_core::intern::intern(s)
}

// -------------------------------------------------------------------------
// PropertyFlags Tests
// -------------------------------------------------------------------------

#[test]
fn test_property_flags_read_only() {
    let flags = PropertyFlags::read_only();
    assert!(!flags.contains(PropertyFlags::WRITABLE));
    assert!(flags.contains(PropertyFlags::ENUMERABLE));
    assert!(flags.contains(PropertyFlags::CONFIGURABLE));
}

#[test]
fn test_property_flags_hidden() {
    let flags = PropertyFlags::hidden();
    assert!(flags.contains(PropertyFlags::WRITABLE));
    assert!(!flags.contains(PropertyFlags::ENUMERABLE));
    assert!(flags.contains(PropertyFlags::CONFIGURABLE));
}

#[test]
fn test_property_flags_combinations() {
    let flags = PropertyFlags::WRITABLE | PropertyFlags::ENUMERABLE;
    assert!(flags.contains(PropertyFlags::WRITABLE));
    assert!(flags.contains(PropertyFlags::ENUMERABLE));
    assert!(!flags.contains(PropertyFlags::CONFIGURABLE));
}

// -------------------------------------------------------------------------
// PropertyDescriptor Tests
// -------------------------------------------------------------------------

#[test]
fn test_property_descriptor_new() {
    let desc = PropertyDescriptor::new(intern("x"), 0, PropertyFlags::default());
    assert_eq!(desc.name.as_str(), "x");
    assert_eq!(desc.slot_index, 0);
    assert!(desc.is_writable());
    assert!(desc.is_enumerable());
    assert!(desc.is_configurable());
    assert!(desc.is_data());
}

#[test]
fn test_property_descriptor_writable() {
    let desc = PropertyDescriptor::writable(intern("foo"), 3);
    assert_eq!(desc.name.as_str(), "foo");
    assert_eq!(desc.slot_index, 3);
    assert!(desc.is_writable());
}

#[test]
fn test_property_descriptor_read_only() {
    let desc = PropertyDescriptor::new(intern("const"), 0, PropertyFlags::read_only());
    assert!(!desc.is_writable());
    assert!(desc.is_enumerable());
}

// -------------------------------------------------------------------------
// ShapeId Tests
// -------------------------------------------------------------------------

#[test]
fn test_shape_id_empty() {
    assert!(ShapeId::EMPTY.is_empty());
    assert!(!ShapeId(1).is_empty());
}

#[test]
fn test_shape_id_raw() {
    assert_eq!(ShapeId(42).raw(), 42);
}

// -------------------------------------------------------------------------
// Shape Tests - Basic
// -------------------------------------------------------------------------

#[test]
fn test_empty_shape() {
    let empty = Shape::empty();
    assert!(empty.is_empty());
    assert_eq!(empty.id(), ShapeId::EMPTY);
    assert!(empty.parent().is_none());
    assert!(empty.property().is_none());
    assert_eq!(empty.property_count(), 0);
    assert_eq!(empty.inline_count(), 0);
    assert!(empty.is_fully_inline());
}

#[test]
fn test_shape_single_property() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape_x = registry.transition_default(&empty, intern("x"));

    assert!(!shape_x.is_empty());
    assert!(shape_x.parent().is_some());
    assert_eq!(shape_x.property_count(), 1);
    assert_eq!(shape_x.inline_count(), 1);

    let prop = shape_x.property().unwrap();
    assert_eq!(prop.name.as_str(), "x");
    assert_eq!(prop.slot_index, 0);
}

#[test]
fn test_shape_multiple_properties() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape_x = registry.transition_default(&empty, intern("x"));
    let shape_xy = registry.transition_default(&shape_x, intern("y"));
    let shape_xyz = registry.transition_default(&shape_xy, intern("z"));

    assert_eq!(shape_xyz.property_count(), 3);
    assert_eq!(shape_xyz.inline_count(), 3);

    let z_prop = shape_xyz.property().unwrap();
    assert_eq!(z_prop.name.as_str(), "z");
    assert_eq!(z_prop.slot_index, 2);
}

// -------------------------------------------------------------------------
// Shape Tests - Lookup
// -------------------------------------------------------------------------

#[test]
fn test_shape_lookup_found() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape = registry.transition_default(&empty, intern("x"));
    let shape = registry.transition_default(&shape, intern("y"));
    let shape = registry.transition_default(&shape, intern("z"));

    assert_eq!(shape.lookup("x"), Some(0));
    assert_eq!(shape.lookup("y"), Some(1));
    assert_eq!(shape.lookup("z"), Some(2));
}

#[test]
fn test_shape_lookup_not_found() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape = registry.transition_default(&empty, intern("x"));

    assert_eq!(shape.lookup("y"), None);
    assert_eq!(shape.lookup("not_exists"), None);
}

#[test]
fn test_shape_lookup_empty() {
    let empty = Shape::empty();
    assert_eq!(empty.lookup("anything"), None);
}

#[test]
fn test_shape_lookup_interned() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let name = intern("property");
    let shape = registry.transition_default(&empty, name.clone());

    assert_eq!(shape.lookup_interned(&name), Some(0));
    assert_eq!(shape.lookup_interned(&intern("other")), None);
}

#[test]
fn test_shape_get_descriptor() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let flags = PropertyFlags::hidden();
    let shape = registry.transition(&empty, intern("_private"), flags);

    let desc = shape.get_descriptor("_private").unwrap();
    assert_eq!(desc.name.as_str(), "_private");
    assert!(!desc.is_enumerable());
    assert!(desc.is_writable());
}

// -------------------------------------------------------------------------
// Shape Tests - Property Collection
// -------------------------------------------------------------------------

#[test]
fn test_property_names_order() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape = registry.transition_default(&empty, intern("first"));
    let shape = registry.transition_default(&shape, intern("second"));
    let shape = registry.transition_default(&shape, intern("third"));

    let names = shape.property_names();
    assert_eq!(names.len(), 3);
    assert_eq!(names[0].as_str(), "first");
    assert_eq!(names[1].as_str(), "second");
    assert_eq!(names[2].as_str(), "third");
}

#[test]
fn test_all_descriptors() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape = registry.transition_default(&empty, intern("a"));
    let shape = registry.transition_default(&shape, intern("b"));

    let descriptors = shape.all_descriptors();
    assert_eq!(descriptors.len(), 2);
    assert_eq!(descriptors[0].name.as_str(), "a");
    assert_eq!(descriptors[0].slot_index, 0);
    assert_eq!(descriptors[1].name.as_str(), "b");
    assert_eq!(descriptors[1].slot_index, 1);
}

// -------------------------------------------------------------------------
// Shape Tests - Transitions
// -------------------------------------------------------------------------

#[test]
fn test_transition_caching() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let name = intern("x");

    let shape1 = registry.transition_default(&empty, name.clone());
    let shape2 = registry.transition_default(&empty, name.clone());

    // Should return the same cached shape
    assert!(Arc::ptr_eq(&shape1, &shape2));
}

#[test]
fn test_transition_different_properties() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();

    let shape_x = registry.transition_default(&empty, intern("x"));
    let shape_y = registry.transition_default(&empty, intern("y"));

    // Different properties -> different shapes
    assert!(!Arc::ptr_eq(&shape_x, &shape_y));
    assert_ne!(shape_x.id(), shape_y.id());
}

#[test]
fn test_transition_branching() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape_x = registry.transition_default(&empty, intern("x"));

    // Two different paths from shape_x
    let shape_xy = registry.transition_default(&shape_x, intern("y"));
    let shape_xz = registry.transition_default(&shape_x, intern("z"));

    assert_ne!(shape_xy.id(), shape_xz.id());
    assert_eq!(shape_xy.lookup("y"), Some(1));
    assert_eq!(shape_xz.lookup("z"), Some(1));
}

#[test]
fn test_has_transition() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let name = intern("x");

    assert!(!empty.has_transition(&name));
    let _shape = registry.transition_default(&empty, name.clone());
    assert!(empty.has_transition(&name));
}

#[test]
fn test_get_transition() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let name = intern("x");

    assert!(empty.get_transition(&name).is_none());
    let shape = registry.transition_default(&empty, name.clone());
    let cached = empty.get_transition(&name).unwrap();
    assert!(Arc::ptr_eq(&shape, &cached));
}

// -------------------------------------------------------------------------
// Shape Tests - Inline Storage Limits
// -------------------------------------------------------------------------

#[test]
fn test_inline_storage_grows() {
    let registry = ShapeRegistry::new();
    let mut shape = registry.empty_shape();

    for i in 0..MAX_INLINE_SLOTS {
        shape = registry.transition_default(&shape, intern(&format!("prop{}", i)));
        assert_eq!(shape.inline_count() as usize, i + 1);
        assert!(shape.is_fully_inline());
    }
}

#[test]
fn test_inline_storage_limit() {
    let registry = ShapeRegistry::new();
    let mut shape = registry.empty_shape();

    // Fill inline storage
    for i in 0..MAX_INLINE_SLOTS {
        shape = registry.transition_default(&shape, intern(&format!("p{}", i)));
    }

    assert_eq!(shape.inline_count() as usize, MAX_INLINE_SLOTS);
    assert!(shape.is_fully_inline());

    // Add one more - should spill
    shape = registry.transition_default(&shape, intern("overflow"));
    assert_eq!(shape.property_count() as usize, MAX_INLINE_SLOTS + 1);
    // Inline count doesn't increase beyond max
    assert_eq!(shape.inline_count() as usize, MAX_INLINE_SLOTS);
    assert!(!shape.is_fully_inline());
}

// -------------------------------------------------------------------------
// ShapeRegistry Tests
// -------------------------------------------------------------------------

#[test]
fn test_registry_new() {
    let registry = ShapeRegistry::new();
    assert_eq!(registry.shape_count(), 1); // Empty shape counts
}

#[test]
fn test_registry_unique_ids() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();

    let ids: Vec<_> = (0..100)
        .map(|i| {
            let shape = registry.transition_default(&empty, intern(&format!("p{}", i)));
            shape.id()
        })
        .collect();

    // All IDs should be unique
    let mut unique_ids = ids.clone();
    unique_ids.sort_by_key(|id| id.raw());
    unique_ids.dedup();
    assert_eq!(ids.len(), unique_ids.len());
}

#[test]
fn test_registry_shape_count() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();

    let initial = registry.shape_count();
    let _s1 = registry.transition_default(&empty, intern("a"));
    assert_eq!(registry.shape_count(), initial + 1);

    let _s2 = registry.transition_default(&empty, intern("b"));
    assert_eq!(registry.shape_count(), initial + 2);

    // Cached transition doesn't increase count
    let _s1_again = registry.transition_default(&empty, intern("a"));
    assert_eq!(registry.shape_count(), initial + 2);
}

// -------------------------------------------------------------------------
// Global Registry Tests
// -------------------------------------------------------------------------

#[test]
fn test_global_registry_access() {
    init_shape_registry();
    let registry = shape_registry();
    let empty = registry.empty_shape();
    assert!(empty.is_empty());
}

// -------------------------------------------------------------------------
// Thread Safety Tests
// -------------------------------------------------------------------------

#[test]
fn test_shape_thread_safety() {
    use std::thread;

    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let empty_clone = Arc::clone(&empty);
            let _name = intern(&format!("thread_prop_{}", i));
            thread::spawn(move || {
                // Can't use registry across threads, but shapes should be thread-safe
                empty_clone.lookup(&format!("prop{}", i))
            })
        })
        .collect();

    for handle in handles {
        let _ = handle.join().unwrap();
    }
}

// -------------------------------------------------------------------------
// Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_empty_property_name() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape = registry.transition_default(&empty, intern(""));

    assert_eq!(shape.lookup(""), Some(0));
    assert_eq!(shape.property().unwrap().name.as_str(), "");
}

#[test]
fn test_unicode_property_names() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let shape = registry.transition_default(&empty, intern("名前"));
    let shape = registry.transition_default(&shape, intern("привет"));
    let shape = registry.transition_default(&shape, intern("🚀"));

    assert_eq!(shape.lookup("名前"), Some(0));
    assert_eq!(shape.lookup("привет"), Some(1));
    assert_eq!(shape.lookup("🚀"), Some(2));
}

#[test]
fn test_long_property_chain() {
    let registry = ShapeRegistry::new();
    let mut shape = registry.empty_shape();

    // Create a long chain
    for i in 0..50 {
        shape = registry.transition_default(&shape, intern(&format!("property_{}", i)));
    }

    // Should still be able to look up all properties
    for i in 0..50 {
        let name = format!("property_{}", i);
        assert!(shape.lookup(&name).is_some(), "Failed to find {}", name);
    }
}

#[test]
fn test_property_flags_preserved() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();

    let read_only = registry.transition(&empty, intern("ro"), PropertyFlags::read_only());
    let hidden = registry.transition(&empty, intern("hidden"), PropertyFlags::hidden());

    let ro_desc = read_only.get_descriptor("ro").unwrap();
    assert!(!ro_desc.is_writable());

    let hidden_desc = hidden.get_descriptor("hidden").unwrap();
    assert!(!hidden_desc.is_enumerable());
}

#[test]
fn test_transition_distinguishes_same_name_with_different_flags() {
    let registry = ShapeRegistry::new();
    let empty = registry.empty_shape();
    let name = intern("shared");

    let writable = registry.transition(&empty, name.clone(), PropertyFlags::default());
    let read_only = registry.transition(&empty, name.clone(), PropertyFlags::read_only());

    assert_ne!(writable.id(), read_only.id());
    assert!(empty.has_transition_with_flags(&name, PropertyFlags::default()));
    assert!(empty.has_transition_with_flags(&name, PropertyFlags::read_only()));
    assert_eq!(
        writable.get_descriptor("shared").unwrap().flags,
        PropertyFlags::default()
    );
    assert_eq!(
        read_only.get_descriptor("shared").unwrap().flags,
        PropertyFlags::read_only()
    );
}

#[test]
fn test_transition_is_canonical_under_contention() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let registry = Arc::new(ShapeRegistry::new());
    let empty = registry.empty_shape();
    let barrier = Arc::new(Barrier::new(16));

    let handles: Vec<_> = (0..16)
        .map(|_| {
            let registry = Arc::clone(&registry);
            let empty = Arc::clone(&empty);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();
                registry
                    .transition(&empty, intern("shared"), PropertyFlags::hidden())
                    .id()
            })
        })
        .collect();

    let ids: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("shape transition thread panicked"))
        .collect();

    assert!(ids.iter().all(|id| *id == ids[0]));
    assert_eq!(registry.shape_count(), 2);
}
