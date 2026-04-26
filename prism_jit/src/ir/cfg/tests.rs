use super::*;

#[test]
fn test_cfg_build() {
    let g = Graph::new();
    let cfg = Cfg::build(&g);

    assert!(cfg.entry.is_valid());
    assert!(!cfg.is_empty());
}

#[test]
fn test_dominator_tree() {
    let g = Graph::new();
    let cfg = Cfg::build(&g);
    let dom = DominatorTree::build(&cfg);

    // Entry dominates everything
    if cfg.exit.is_valid() {
        assert!(dom.dominates(cfg.entry, cfg.exit));
    }
}

#[test]
fn test_dominator_self() {
    let g = Graph::new();
    let cfg = Cfg::build(&g);
    let dom = DominatorTree::build(&cfg);

    assert!(dom.dominates(cfg.entry, cfg.entry));
}
