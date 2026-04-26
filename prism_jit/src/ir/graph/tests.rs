use super::*;

#[test]
fn test_graph_creation() {
    let g = Graph::new();
    assert!(g.len() >= 2); // start + end
    assert!(g.get(g.start).is_some());
    assert!(g.get(g.end).is_some());
}

#[test]
fn test_add_constant() {
    let mut g = Graph::new();

    let c1 = g.const_int(42);
    let c2 = g.const_int(10);

    assert_eq!(g.node(c1).as_int(), Some(42));
    assert_eq!(g.node(c2).as_int(), Some(10));
}

#[test]
fn test_add_arithmetic() {
    let mut g = Graph::new();

    let a = g.const_int(5);
    let b = g.const_int(3);
    let sum = g.int_add(a, b);

    assert_eq!(g.node(sum).inputs.len(), 2);
    assert_eq!(g.node(sum).inputs.get(0), Some(a));
    assert_eq!(g.node(sum).inputs.get(1), Some(b));
}

#[test]
fn test_use_chains() {
    let mut g = Graph::new();

    let c = g.const_int(5);
    let _add1 = g.int_add(c, c);
    let _add2 = g.int_add(c, c);

    // c should have 4 uses (twice in each add)
    assert_eq!(g.use_count(c), 4);
}

#[test]
fn test_replace_all_uses() {
    let mut g = Graph::new();

    let c1 = g.const_int(5);
    let c2 = g.const_int(10);
    let add = g.int_add(c1, c1);

    g.replace_all_uses(c1, c2);

    // add should now use c2
    assert_eq!(g.node(add).inputs.get(0), Some(c2));
    assert_eq!(g.node(add).inputs.get(1), Some(c2));
}

#[test]
fn test_phi_node() {
    let mut g = Graph::new();

    let region = g.region(&[g.start]);
    let v1 = g.const_int(1);
    let v2 = g.const_int(2);
    let phi = g.phi(region, &[v1, v2], ValueType::Int64);

    assert!(g.node(phi).is_phi());
    assert_eq!(g.node(phi).inputs.len(), 3); // region + 2 values
}

#[test]
fn test_graph_verify() {
    let g = Graph::new();
    assert!(g.verify().is_ok());
}
