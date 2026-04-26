# Contributing

Prism is performance-sensitive runtime infrastructure. Changes should be small
enough to review carefully and structured enough that correctness, performance,
and architecture remain visible.

## Commit Style

Use conventional commit prefixes that match the existing history:

- `fix(scope): ...`
- `feat(scope): ...`
- `test(scope): ...`
- `docs(scope): ...`
- `refactor(scope): ...`
- `perf(scope): ...`
- `chore(scope): ...`

Prefer granular commits. A parser fix, VM fix, test update, and documentation
update should usually be separate commits.

## Before Opening A PR

Run:

```bash
cargo check-all
cargo test-fast
./tools/check_project_layout.ps1
```

Formatting is configured in `rustfmt.toml`, but the repository still needs a
single mechanical formatting baseline before formatting can be required in CI.
Strict clippy is useful for cleanup work, but it is not yet part of the default
PR gate because the current VM baseline still contains legacy lint blockers.

For focused work, also run the smallest relevant test package first. For example:

```bash
cargo test -p prism_compiler
cargo test -p prism_vm exception
cargo bench -p prism_vm
```

## Architecture Expectations

- Keep crate boundaries aligned with `docs/ARCHITECTURE.md`.
- Keep hot paths allocation-conscious.
- Keep public APIs narrow.
- Prefer explicit data contracts over cross-layer shortcuts.
- Add tests for every behavior change.
- Add benchmarks or benchmark notes for performance-sensitive changes.

## Refactoring Expectations

Refactors should preserve behavior unless the commit explicitly says otherwise.
When splitting large files, prefer mechanical movement first, then semantic
cleanup in later commits. That keeps review diffs readable and preserves blame.

## Unsafe Code

Unsafe code must follow `docs/UNSAFE.md`. New unsafe blocks need specific
`SAFETY:` comments and tests that cover the relevant invariants.
