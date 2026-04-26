use super::*;

struct TestNode {
    value: i32,
}

#[test]
fn test_arena_alloc() {
    let mut arena: Arena<TestNode> = Arena::new();

    let id1 = arena.alloc(TestNode { value: 10 });
    let id2 = arena.alloc(TestNode { value: 20 });
    let id3 = arena.alloc(TestNode { value: 30 });

    assert_eq!(id1.index(), 0);
    assert_eq!(id2.index(), 1);
    assert_eq!(id3.index(), 2);

    assert_eq!(arena[id1].value, 10);
    assert_eq!(arena[id2].value, 20);
    assert_eq!(arena[id3].value, 30);

    arena[id2].value = 200;
    assert_eq!(arena[id2].value, 200);
}

#[test]
fn test_arena_iter() {
    let mut arena: Arena<TestNode> = Arena::new();

    arena.alloc(TestNode { value: 1 });
    arena.alloc(TestNode { value: 2 });
    arena.alloc(TestNode { value: 3 });

    let values: Vec<_> = arena.iter().map(|(_, n)| n.value).collect();
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_secondary_map() {
    let mut arena: Arena<TestNode> = Arena::new();
    let id1 = arena.alloc(TestNode { value: 10 });
    let id2 = arena.alloc(TestNode { value: 20 });

    let mut map: SecondaryMap<TestNode, String> = SecondaryMap::new();
    map.set(id1, "first".to_string());
    map.set(id2, "second".to_string());

    assert_eq!(map[id1], "first");
    assert_eq!(map[id2], "second");
}

#[test]
fn test_bit_set() {
    let mut set = BitSet::new();

    set.insert(0);
    set.insert(5);
    set.insert(63);
    set.insert(64);
    set.insert(100);

    assert!(set.contains(0));
    assert!(set.contains(5));
    assert!(set.contains(63));
    assert!(set.contains(64));
    assert!(set.contains(100));
    assert!(!set.contains(1));
    assert!(!set.contains(65));

    assert_eq!(set.count(), 5);

    let indices: Vec<_> = set.iter().collect();
    assert_eq!(indices, vec![0, 5, 63, 64, 100]);
}

#[test]
fn test_bit_set_union_intersect() {
    let mut set1 = BitSet::new();
    set1.insert(0);
    set1.insert(2);
    set1.insert(4);

    let mut set2 = BitSet::new();
    set2.insert(1);
    set2.insert(2);
    set2.insert(3);

    let mut union = set1.clone();
    union.union_with(&set2);
    assert_eq!(union.count(), 5); // 0,1,2,3,4

    let mut intersect = set1.clone();
    intersect.intersect_with(&set2);
    assert_eq!(intersect.count(), 1); // just 2
    assert!(intersect.contains(2));
}

#[test]
fn test_id_invalid() {
    let id: Id<TestNode> = Id::INVALID;
    assert!(!id.is_valid());

    let valid_id: Id<TestNode> = Id::new(0);
    assert!(valid_id.is_valid());
}
