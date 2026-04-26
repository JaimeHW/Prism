use super::*;

#[test]
fn test_worklist_new() {
    let wl = Worklist::new();
    assert!(wl.is_empty());
    assert_eq!(wl.len(), 0);
}

#[test]
fn test_worklist_with_capacity() {
    let wl = Worklist::with_capacity(100);
    assert!(wl.is_empty());
}

#[test]
fn test_worklist_push() {
    let mut wl = Worklist::new();
    let added = wl.push(NodeId::new(1));
    assert!(added);
    assert_eq!(wl.len(), 1);
    assert!(!wl.is_empty());
}

#[test]
fn test_worklist_push_dedup() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));
    let added = wl.push(NodeId::new(1));
    assert!(!added); // Duplicate
    assert_eq!(wl.len(), 1);
}

#[test]
fn test_worklist_push_all() {
    let mut wl = Worklist::new();
    wl.push_all([NodeId::new(1), NodeId::new(2), NodeId::new(3)]);
    assert_eq!(wl.len(), 3);
}

#[test]
fn test_worklist_push_all_dedup() {
    let mut wl = Worklist::new();
    wl.push_all([NodeId::new(1), NodeId::new(1), NodeId::new(2)]);
    assert_eq!(wl.len(), 2);
}

#[test]
fn test_worklist_pop() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));
    wl.push(NodeId::new(2));

    let first = wl.pop();
    assert_eq!(first, Some(NodeId::new(1)));
    assert_eq!(wl.len(), 1);

    let second = wl.pop();
    assert_eq!(second, Some(NodeId::new(2)));
    assert!(wl.is_empty());
}

#[test]
fn test_worklist_pop_empty() {
    let mut wl = Worklist::new();
    assert_eq!(wl.pop(), None);
}

#[test]
fn test_worklist_fifo_order() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));
    wl.push(NodeId::new(2));
    wl.push(NodeId::new(3));

    assert_eq!(wl.pop(), Some(NodeId::new(1)));
    assert_eq!(wl.pop(), Some(NodeId::new(2)));
    assert_eq!(wl.pop(), Some(NodeId::new(3)));
}

#[test]
fn test_worklist_contains() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));

    assert!(wl.contains(NodeId::new(1)));
    assert!(!wl.contains(NodeId::new(2)));
}

#[test]
fn test_worklist_contains_after_pop() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));
    wl.pop();

    assert!(!wl.contains(NodeId::new(1)));
}

#[test]
fn test_worklist_total_added() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));
    wl.push(NodeId::new(2));
    wl.push(NodeId::new(1)); // Dup - not added

    assert_eq!(wl.total_added(), 2);
}

#[test]
fn test_worklist_total_processed() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));
    wl.push(NodeId::new(2));
    wl.pop();

    assert_eq!(wl.total_processed(), 1);
}

#[test]
fn test_worklist_clear() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));
    wl.push(NodeId::new(2));
    wl.clear();

    assert!(wl.is_empty());
    assert!(!wl.contains(NodeId::new(1)));
}

#[test]
fn test_worklist_repush_after_pop() {
    let mut wl = Worklist::new();
    wl.push(NodeId::new(1));
    wl.pop();

    // Should be able to add again after pop
    let added = wl.push(NodeId::new(1));
    assert!(added);
    assert_eq!(wl.len(), 1);
}

#[test]
fn test_worklist_stress() {
    let mut wl = Worklist::new();

    // Add many nodes
    for i in 0..1000 {
        wl.push(NodeId::new(i));
    }
    assert_eq!(wl.len(), 1000);

    // Process all
    while wl.pop().is_some() {}

    assert!(wl.is_empty());
    assert_eq!(wl.total_added(), 1000);
    assert_eq!(wl.total_processed(), 1000);
}
