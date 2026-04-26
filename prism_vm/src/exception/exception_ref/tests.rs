use super::*;
use crate::stdlib::exceptions::ExceptionTypeId;

// ════════════════════════════════════════════════════════════════════════
// FlyweightExceptionRef Creation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_ref_from_shared_ref() {
    let exc = ExceptionObject::new(ExceptionTypeId::ValueError);
    let exc_ref = FlyweightExceptionRef::new(&exc);

    assert_eq!(exc_ref.as_ptr(), &exc as *const _);
}

#[test]
fn test_flyweight_ref_from_mut_ref() {
    let mut exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let exc_ref = FlyweightExceptionRef::from_mut(&mut exc);

    assert_eq!(exc_ref.as_ptr(), &exc as *const _);
}

#[test]
fn test_flyweight_ref_from_raw() {
    let exc = ExceptionObject::new(ExceptionTypeId::KeyError);
    let ptr = &exc as *const ExceptionObject;

    let exc_ref = unsafe { FlyweightExceptionRef::from_raw(ptr) };
    assert_eq!(exc_ref.as_ptr(), ptr);
}

// ════════════════════════════════════════════════════════════════════════
// FlyweightExceptionRef Access Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_ref_as_ref() {
    let exc = ExceptionObject::new(ExceptionTypeId::IndexError);
    let exc_ref = FlyweightExceptionRef::new(&exc);

    unsafe {
        let deref = exc_ref.as_ref();
        assert_eq!(deref.type_id(), ExceptionTypeId::IndexError);
    }
}

#[test]
fn test_flyweight_ref_as_mut() {
    let mut exc = ExceptionObject::new(ExceptionTypeId::AttributeError);
    let mut exc_ref = FlyweightExceptionRef::from_mut(&mut exc);

    unsafe {
        let deref = exc_ref.as_mut();
        // Verify we can access mutably
        assert_eq!(deref.type_id(), ExceptionTypeId::AttributeError);
    }
}

// ════════════════════════════════════════════════════════════════════════
// FlyweightExceptionRef Comparison Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_ref_ptr_eq_same() {
    let exc = ExceptionObject::new(ExceptionTypeId::StopIteration);
    let ref1 = FlyweightExceptionRef::new(&exc);
    let ref2 = FlyweightExceptionRef::new(&exc);

    assert!(ref1.ptr_eq(&ref2));
    assert_eq!(ref1, ref2);
}

#[test]
fn test_flyweight_ref_ptr_eq_different() {
    let exc1 = ExceptionObject::new(ExceptionTypeId::StopIteration);
    let exc2 = ExceptionObject::new(ExceptionTypeId::StopIteration);

    let ref1 = FlyweightExceptionRef::new(&exc1);
    let ref2 = FlyweightExceptionRef::new(&exc2);

    assert!(!ref1.ptr_eq(&ref2));
    assert_ne!(ref1, ref2);
}

// ════════════════════════════════════════════════════════════════════════
// FlyweightExceptionRef Size Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_ref_size() {
    // Must be pointer-sized for efficiency
    assert_eq!(std::mem::size_of::<FlyweightExceptionRef>(), 8);
}

#[test]
fn test_flyweight_ref_option_size() {
    // NonNull enables niche optimization
    assert_eq!(
        std::mem::size_of::<Option<FlyweightExceptionRef>>(),
        std::mem::size_of::<FlyweightExceptionRef>()
    );
}

#[test]
fn test_flyweight_ref_alignment() {
    // Must be pointer-aligned
    assert_eq!(std::mem::align_of::<FlyweightExceptionRef>(), 8);
}

// ════════════════════════════════════════════════════════════════════════
// FlyweightExceptionRef Hash Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_flyweight_ref_hash_consistency() {
    use std::collections::hash_map::DefaultHasher;

    let exc = ExceptionObject::new(ExceptionTypeId::ValueError);
    let ref1 = FlyweightExceptionRef::new(&exc);
    let ref2 = FlyweightExceptionRef::new(&exc);

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    ref1.hash(&mut hasher1);
    ref2.hash(&mut hasher2);

    assert_eq!(hasher1.finish(), hasher2.finish());
}

// ════════════════════════════════════════════════════════════════════════
// OwnedExceptionRef Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_owned_exception_ref_new() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let owned = OwnedExceptionRef::new(exc);

    assert_eq!(owned.type_id(), ExceptionTypeId::TypeError);
}

#[test]
fn test_owned_exception_ref_from_boxed() {
    let boxed = Box::new(ExceptionObject::new(ExceptionTypeId::KeyError));
    let owned = OwnedExceptionRef::from_boxed(boxed);

    assert_eq!(owned.type_id(), ExceptionTypeId::KeyError);
}

#[test]
fn test_owned_exception_ref_as_flyweight() {
    let exc = ExceptionObject::new(ExceptionTypeId::IndexError);
    let owned = OwnedExceptionRef::new(exc);
    let exc_ref = owned.as_flyweight();

    // The flyweight ref should point to the owned data
    assert_eq!(exc_ref.as_ptr(), owned.get() as *const _);
}

#[test]
fn test_owned_exception_ref_deref() {
    let exc = ExceptionObject::new(ExceptionTypeId::AttributeError);
    let owned = OwnedExceptionRef::new(exc);

    // Test Deref trait
    assert_eq!(owned.type_id(), ExceptionTypeId::AttributeError);
}

#[test]
fn test_owned_exception_ref_into_inner() {
    let exc = ExceptionObject::new(ExceptionTypeId::OSError);
    let owned = OwnedExceptionRef::new(exc);

    let inner = owned.into_inner();
    assert_eq!(inner.type_id(), ExceptionTypeId::OSError);
}

#[test]
fn test_owned_exception_ref_into_boxed() {
    let exc = ExceptionObject::new(ExceptionTypeId::EOFError);
    let owned = OwnedExceptionRef::new(exc);

    let boxed = owned.into_boxed();
    assert_eq!(boxed.type_id(), ExceptionTypeId::EOFError);
}

#[test]
fn test_owned_exception_ref_from_exception() {
    let exc = ExceptionObject::new(ExceptionTypeId::MemoryError);
    let owned: OwnedExceptionRef = exc.into();

    assert_eq!(owned.type_id(), ExceptionTypeId::MemoryError);
}

#[test]
fn test_owned_exception_ref_size() {
    // Should be just a Box (pointer-sized)
    assert_eq!(std::mem::size_of::<OwnedExceptionRef>(), 8);
}
