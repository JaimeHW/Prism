use super::*;
use std::collections::HashMap;

/// Helper to create MRO lookup function from a map
fn mro_lookup(mros: &HashMap<ClassId, Mro>) -> impl Fn(ClassId) -> Option<Mro> + '_ {
    move |id| mros.get(&id).cloned()
}

#[test]
fn test_single_inheritance_chain() {
    // class A: pass
    // class B(A): pass
    // class C(B): pass
    // MRO of C should be [C, B, A, object]

    let a = ClassId(10);
    let b = ClassId(11);
    let c = ClassId(12);

    let mut mros = HashMap::new();

    // A's MRO: [A, object]
    let a_mro = compute_c3_mro(a, &[], |_| None).unwrap();
    assert_eq!(a_mro.as_slice(), &[a, ClassId::OBJECT]);
    mros.insert(a, a_mro);

    // B's MRO: [B, A, object]
    let b_mro = compute_c3_mro(b, &[a], mro_lookup(&mros)).unwrap();
    assert_eq!(b_mro.as_slice(), &[b, a, ClassId::OBJECT]);
    mros.insert(b, b_mro);

    // C's MRO: [C, B, A, object]
    let c_mro = compute_c3_mro(c, &[b], mro_lookup(&mros)).unwrap();
    assert_eq!(c_mro.as_slice(), &[c, b, a, ClassId::OBJECT]);
}

#[test]
fn test_diamond_inheritance() {
    // class A: pass
    // class B(A): pass
    // class C(A): pass
    // class D(B, C): pass
    // MRO of D should be [D, B, C, A, object]

    let a = ClassId(10);
    let b = ClassId(11);
    let c = ClassId(12);
    let d = ClassId(13);

    let mut mros = HashMap::new();

    // A's MRO: [A, object]
    let a_mro = compute_c3_mro(a, &[], |_| None).unwrap();
    mros.insert(a, a_mro);

    // B's MRO: [B, A, object]
    let b_mro = compute_c3_mro(b, &[a], mro_lookup(&mros)).unwrap();
    mros.insert(b, b_mro);

    // C's MRO: [C, A, object]
    let c_mro = compute_c3_mro(c, &[a], mro_lookup(&mros)).unwrap();
    mros.insert(c, c_mro);

    // D's MRO: [D, B, C, A, object]
    let d_mro = compute_c3_mro(d, &[b, c], mro_lookup(&mros)).unwrap();
    assert_eq!(d_mro.as_slice(), &[d, b, c, a, ClassId::OBJECT]);
}

#[test]
fn test_complex_diamond() {
    // class O: pass  (object)
    // class A(O): pass
    // class B(O): pass
    // class C(O): pass
    // class D(A, B): pass
    // class E(B, C): pass
    // class F(D, E): pass
    // MRO of F: [F, D, A, E, B, C, O]

    let o = ClassId::OBJECT;
    let a = ClassId(10);
    let b = ClassId(11);
    let c = ClassId(12);
    let d = ClassId(13);
    let e = ClassId(14);
    let f = ClassId(15);

    let mut mros = HashMap::new();

    // O is object - already computed implicitly
    let o_mro: Mro = smallvec::smallvec![ClassId::OBJECT];
    mros.insert(o, o_mro);

    // A(O): [A, O]
    let a_mro = compute_c3_mro(a, &[], |_| None).unwrap();
    mros.insert(a, a_mro);

    // B(O): [B, O]
    let b_mro = compute_c3_mro(b, &[], |_| None).unwrap();
    mros.insert(b, b_mro);

    // C(O): [C, O]
    let c_mro = compute_c3_mro(c, &[], |_| None).unwrap();
    mros.insert(c, c_mro);

    // D(A, B): [D, A, B, O]
    let d_mro = compute_c3_mro(d, &[a, b], mro_lookup(&mros)).unwrap();
    mros.insert(d, d_mro.clone());
    assert_eq!(d_mro.as_slice(), &[d, a, b, ClassId::OBJECT]);

    // E(B, C): [E, B, C, O]
    let e_mro = compute_c3_mro(e, &[b, c], mro_lookup(&mros)).unwrap();
    mros.insert(e, e_mro.clone());
    assert_eq!(e_mro.as_slice(), &[e, b, c, ClassId::OBJECT]);

    // F(D, E): [F, D, A, E, B, C, O]
    let f_mro = compute_c3_mro(f, &[d, e], mro_lookup(&mros)).unwrap();
    assert_eq!(f_mro.as_slice(), &[f, d, a, e, b, c, ClassId::OBJECT]);
}

#[test]
fn test_duplicate_base_error() {
    let a = ClassId(10);

    // class B(A, A) - duplicate base
    let result = compute_c3_mro(ClassId(11), &[a, a], |_| None);
    assert!(matches!(result, Err(MroError::DuplicateBase { .. })));
}

#[test]
fn test_inconsistent_mro_error() {
    // class A: pass
    // class B(A): pass
    // class C(A, B): pass  <- Invalid! A comes before B but B inherits from A

    let a = ClassId(10);
    let b = ClassId(11);
    let c = ClassId(12);

    let mut mros = HashMap::new();

    // A's MRO: [A, object]
    let a_mro = compute_c3_mro(a, &[], |_| None).unwrap();
    mros.insert(a, a_mro);

    // B's MRO: [B, A, object]
    let b_mro = compute_c3_mro(b, &[a], mro_lookup(&mros)).unwrap();
    mros.insert(b, b_mro);

    // C(A, B) - should fail
    let result = compute_c3_mro(c, &[a, b], mro_lookup(&mros));
    assert!(matches!(result, Err(MroError::InconsistentMro { .. })));
}

#[test]
fn test_object_base() {
    // class A: pass (implicitly inherits from object)
    // MRO should be [A, object]

    let a = ClassId(10);
    let mro = compute_c3_mro(a, &[], |_| None).unwrap();
    assert_eq!(mro.as_slice(), &[a, ClassId::OBJECT]);
}

#[test]
fn test_object_class() {
    // object's MRO is just [object]
    let mro = compute_c3_mro(ClassId::OBJECT, &[], |_| None).unwrap();
    assert_eq!(mro.as_slice(), &[ClassId::OBJECT]);
}

#[test]
fn test_deeply_nested_single_inheritance() {
    // Test a chain of 10 classes
    // A -> B -> C -> D -> E -> F -> G -> H -> I -> J

    let classes: Vec<ClassId> = (300..310).map(ClassId).collect();
    let mut mros = HashMap::new();

    // First class inherits from nothing
    let first_mro = compute_c3_mro(classes[0], &[], |_| None).unwrap();
    mros.insert(classes[0], first_mro);

    // Each subsequent class inherits from the previous
    for i in 1..classes.len() {
        let mro = compute_c3_mro(classes[i], &[classes[i - 1]], mro_lookup(&mros)).unwrap();

        // Verify MRO length: class + all ancestors + object
        assert_eq!(mro.len(), i + 2);

        // Verify first element is the class itself
        assert_eq!(mro[0], classes[i]);

        // Verify last element is object
        assert_eq!(mro[mro.len() - 1], ClassId::OBJECT);

        mros.insert(classes[i], mro);
    }
}

#[test]
fn test_multiple_bases_no_overlap() {
    // class A: pass
    // class B: pass
    // class C(A, B): pass
    // MRO of C: [C, A, B, object]

    let a = ClassId(10);
    let b = ClassId(11);
    let c = ClassId(12);

    let mut mros = HashMap::new();

    let a_mro = compute_c3_mro(a, &[], |_| None).unwrap();
    mros.insert(a, a_mro);

    let b_mro = compute_c3_mro(b, &[], |_| None).unwrap();
    mros.insert(b, b_mro);

    let c_mro = compute_c3_mro(c, &[a, b], mro_lookup(&mros)).unwrap();
    assert_eq!(c_mro.as_slice(), &[c, a, b, ClassId::OBJECT]);
}

#[test]
fn test_mro_caching_efficiency() {
    // Verify that MRO computation doesn't allocate for small hierarchies
    // by checking SmallVec capacity

    let a = ClassId(10);
    let mro = compute_c3_mro(a, &[], |_| None).unwrap();

    // SmallVec with capacity 8 should not heap-allocate for 2 elements
    assert!(mro.len() <= 8);
    assert!(!mro.spilled());
}
