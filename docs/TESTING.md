# Testing Strategy

Prism needs layered tests because parser, compiler, VM, GC, JIT, stdlib, and AOT
fail in different ways.

## Test Tiers

1. Unit tests for pure data structures and local invariants.
2. Crate integration tests for public crate behavior.
3. VM integration tests for Python-visible semantics.
4. CPython compatibility tests for language and stdlib parity.
5. GC stress tests for lifetime, movement, roots, and barriers.
6. JIT tests for lowering, guards, deopt, OSR, and compiled execution.
7. AOT tests for deterministic planning and frozen-module loading.
8. Benchmarks for performance regressions.

## Local Checks

Run these before committing broad changes:

```bash
cargo check --workspace --all-targets
cargo test --workspace --lib --bins
./tools/check_project_layout.ps1
```

Aliases are available through `.cargo/config.toml`:

```bash
cargo check-all
cargo test-fast
```

`rustfmt.toml` is checked in so the project has a single formatting contract.
The current codebase still needs a dedicated mechanical formatting baseline
before `cargo fmt --all -- --check` can become a required CI gate.

Strict clippy is an audit lane until the existing VM baseline is cleaned up.
Run `cargo clippy --workspace --all-targets --no-deps` before large refactors
when you are prepared to address legacy lints such as mutable borrows derived
from immutable inputs.

## Test Placement

Use crate-local `tests/` for behavior visible through the crate API.

Use `src/*_tests.rs` only when tests need private implementation details.

Use shared test helpers when a fixture appears in three or more tests.

Avoid catch-all integration files. Once a test file exceeds 1,500 lines, split it
by semantic area.

## Regression Tests

Every bug fix should include a regression test unless one of these is true:

- The behavior is already covered by a stronger compatibility test.
- The fix is purely documentation or metadata.
- The only meaningful validation is a benchmark or external environment test.

When skipping a test, explain why in the commit message.

## Compatibility Tests

Compatibility tests should prefer Python source snippets that describe the
observable behavior. Keep implementation details out of expected results.

For CPython deltas, document whether the gap is:

- Not implemented yet.
- Platform-specific.
- Intentional Prism behavior.
- A known bug.

## Performance Tests

Performance changes should include either:

- A benchmark update.
- A new benchmark covering the optimized path.
- A note explaining why existing benchmarks already cover it.

Do not use wall-clock timing assertions in normal tests. Use benchmarks for
timing and tests for semantics.

## CI Contract

CI runs repository-layout checks, workspace checks, and fast tests on Linux and
Windows. Linux catches common Unix behavior; Windows is required because Prism
exposes Windows-specific stdlib modules and object-file flows.

CI is a floor, not a full release gate. Release candidates should also run
compatibility suites and representative benchmarks.

Use `cargo test --workspace --all-targets` before release candidates. Some
CPython compatibility tests require a local CPython checkout and can be much
slower than unit tests; keep those in the compatibility lane rather than the
fast CI lane.
