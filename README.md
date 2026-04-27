<p align="center">
  <h1 align="center">Prism</h1>
  <p align="center">
    A high-performance Python 3.12 runtime written from scratch in Rust.
    <br />
    <em>NaN-boxed values · Register-based VM · Tiered JIT · Generational GC</em>
  </p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#key-design-decisions">Design</a> ·
  <a href="#tooling">Tooling</a> ·
  <a href="#standard-library">Stdlib</a> ·
  <a href="#building">Building</a> ·
  <a href="#project-status">Status</a> ·
  <a href="#license">License</a>
</p>

---

Prism is a ground-up Python 3.12 implementation: parser, compiler, virtual machine, garbage collector, JIT compiler, standard library, and AOT tooling. The project is organized as an 11-crate Cargo workspace with explicit ownership boundaries for source compilation, runtime execution, and stdlib import metadata.

The codebase spans roughly 230k lines of Rust across the workspace.

## Quickstart

```bash
# Run a script
cargo run -p prism_cli --bin prism -- script.py

# Execute an inline expression
cargo run -p prism_cli --bin prism -- -c "print('hello from Prism')"

# Run a module
cargo run -p prism_cli --bin prism -- -m package.module

# Start the REPL
cargo run -p prism_cli --bin prism
```

## Architecture

```text
                  prism_cli
         prism / prismc / prism-test
                    |
        +-----------+-----------+
        |                       |
     prism_vm               prism_aot
  interpreter, imports      build planner
        |                       |
        |                  prism_stdlib
        |                 stdlib metadata
        |
  prism_runtime <---- prism_gc
   object model      collector
        |
    prism_jit
  tiered codegen

prism_parser -> prism_compiler -> prism_code
   lexer/AST     source frontend    bytecode format
                 scope analysis
                 code generation
```

### Crate Overview

| Crate | Purpose |
|---|---|
| **prism_core** | NaN-boxed value representation, string interning, error types, span tracking, small integer cache, speculation primitives, and AOT symbol definitions |
| **prism_code** | Shared bytecode instruction set, `CodeObject` format, and `FunctionBuilder` API used by compiler, VM, JIT, and AOT layers |
| **prism_parser** | Full Python 3.12 lexer, token representation, parser, and AST |
| **prism_compiler** | Source-to-bytecode frontend: parse/compile entrypoints, scope analysis with closure capture, class/function/exception/match-statement compilers, and register-based bytecode emission |
| **prism_stdlib** | Static stdlib inventory, import-policy metadata, and builtin module name lists shared by AOT planning and VM import machinery. Also ships Python source fallback modules |
| **prism_runtime** | Python object model: `ObjectHeader`, type objects, type slots, descriptors, metaclasses, MRO, hidden classes (shapes), object pooling, GC trace/dispatch integration, and write barriers |
| **prism_gc** | Generational garbage collector with nursery, survivor space, tenured space, large-object space, card-table write barriers, remembered sets, SATB buffers, TLABs, mark bitmaps, minor collection, major collection, and concurrent marking |
| **prism_jit** | Tiered JIT: baseline Tier 1 (template codegen with deoptimization), optimizing Tier 2 (Sea-of-Nodes IR, optimization pipeline, safepoint placement, OSR), graph-coloring and linear-scan register allocation, x64 and ARM64 backends, inline caches, GC stack maps, and code caching |
| **prism_vm** | Bytecode interpreter, dispatch table, inline caching, type feedback, profiling and tier-up, OSR triggers, JIT bridge, deoptimization/recovery, import system, threading runtime, generators, coroutines, exception handling, and native stdlib execution |
| **prism_aot** | Ahead-of-time build planner, frozen-module bundler, build manifest emitter, native module init lowering, linkable object-file generation, and AOT runtime loader |
| **prism_cli** | `prism` interpreter, `prismc` AOT driver, `prism-test` CPython compatibility harness, REPL, and diagnostic utilities |

### Ownership Boundaries

- `prism_compiler` owns source-to-bytecode frontend entrypoints. CLI execution, REPL evaluation, VM source imports, and AOT planning all compile source through the same API.
- `prism_stdlib` owns static stdlib capability metadata. Planning code no longer reaches into VM runtime construction details just to answer "is this a native stdlib module?"
- `prism_vm` owns execution, import realization, and native module behavior. It consumes compiler output and stdlib metadata instead of being the source of both.

---

## Key Design Decisions

### NaN-Boxed Values

Every Python value fits in a single 64-bit word. IEEE 754 quiet-NaN bit patterns encode type tags for `None`, booleans, integers, floats, object pointers, and interned strings. That keeps the float fast path cheap while still supporting inline small integers.

### Register-Based VM

Prism uses a register-based bytecode format rather than a pure stack VM. Each call frame has 256 registers inlined in the stack (2 KB L1 fit). The design favors straightforward lowering, fewer transient moves in hot paths, and a cleaner handoff to tiered JIT compilation. Opcode dispatch uses a static function pointer table for O(1) lookup.

### Tiered JIT Compilation

| Tier | Role |
|---|---|
| **Tier 0** | Bytecode interpreter with profiling counters, type feedback collection, and inline caching |
| **Tier 1** | Baseline JIT with fast template compilation, frame management, and deoptimization guards |
| **Tier 2** | Optimizing JIT with Sea-of-Nodes IR, profile-guided optimization, and advanced passes |

On-stack replacement (OSR) promotes hot loops from interpreter to JIT mid-execution. Deoptimization safely returns execution to the interpreter when speculative guards fail, with full state recovery via trampolines.

**Optimization passes** include: GVN, DCE, LICM, SCCP, PRE, copy propagation, dead store elimination, escape analysis, inlining, instruction combining, range check elimination, strength reduction, tail call optimization, loop unrolling, vectorization, branch probability estimation, and hot/cold code splitting.

**Backend targets**: x64 (primary, with EVEX/SIMD/AVX support) and ARM64 (with NEON support, gated behind `aarch64` target or `arm64` feature).

### Generational Garbage Collector

- Nursery collection uses bump-pointer allocation with TLAB (thread-local allocation buffer) support and copying evacuation.
- Survivor space holds objects that survived one minor GC before promotion.
- Tenured collection supports mark/sweep with mark bitmaps and compaction paths.
- Large objects are allocated out of line in a dedicated large-object space.
- Card-table write barriers and remembered sets track old-to-young references for generational correctness.
- SATB (snapshot-at-the-beginning) buffers support concurrent marking.
- Concurrent major collection runs marking alongside mutator threads.
- JIT-generated code participates through safepoints and precise stack maps.

### Hidden Classes

Objects share structural metadata through shape transitions, which lets inline caches resolve attribute offsets without a full dictionary lookup in the common case. Shapes are tracked in `prism_runtime` with a dedicated object pool and transition table.

### Generators and Coroutines

Full generator and coroutine support including suspend/resume, frame pooling, resume caching, and the async iteration protocol (`__aiter__`, `__anext__`, `get_awaitable`, `send`).

---

## Tooling

### `prism` — Interpreter

```bash
prism script.py            # Run a script
prism -c "expr"            # Execute an inline expression
prism -m module            # Run a module
prism                      # Start the REPL
echo "print(1)" | prism -  # Read from stdin
```

Supported flags: `-B`, `-d`, `-E`, `-i`, `-I`, `-O`, `-OO`, `-q`, `-s`, `-S`, `-u`, `-v`, `-V`/`--version`, `-h`/`--help`, `-W <arg>`, and `-X <opt>`.

JIT and runtime controls are available through `-X`:

| Option | Description |
|---|---|
| `-X jit=on` | Enable JIT compilation (default) |
| `-X jit=off` | Disable JIT compilation |
| `-X nojit` | Disable JIT compilation (alias) |
| `-X max-steps=N` | Limit execution to N bytecode steps |

### `prismc` — AOT Compiler

```bash
prismc build app.py                       # Build a script
prismc build -m pkg.tool -I vendor -OO    # Build a module with search paths
prismc build app.py --target x86_64-pc-windows-msvc  # Cross-compile target
```

Full usage:

```
prismc build [-m|--module] <entry> [-o <file>] [--emit-bundle <file>]
             [--emit-object <file>] [--target <triple>] [-I <path>] [-O|-OO]
```

Default outputs under `prism-build/`:

| File | Purpose |
|---|---|
| `build-plan.json` | Deterministic build manifest |
| `frozen-modules.prism` | Serialized bytecode bundle |
| `frozen-modules.obj` | Linkable object payload (Windows targets by default, or explicit `--emit-object`) |

### `prism-test` — CPython Compatibility Harness

```bash
# Run specific tests against vendored CPython 3.12 test corpus
prism-test test_grammar test_math

# Run with JSON reporting
prism-test --json-report results.json test_grammar

# Run against an external CPython checkout
prism-test --cpython-root /path/to/cpython-3.12 test_grammar

# List discovered tests without running them
prism-test --list-tests
```

The harness supports multiple runner modes (`import`, `suite`, `test-main`), per-test isolation via subprocess execution, configurable per-test and suite-level timeouts, fail-fast mode, verbose/quiet output, starting at a specific test, and JSON reporting. A vendored subset of CPython 3.12's test corpus is included under `tests/cpython_3.12/`.

---

## Standard Library

Prism ships native Rust implementations for core modules and can fall back to filesystem-backed CPython stdlib source when it improves compatibility.

Native builtin/import metadata is defined once in `prism_stdlib` and consumed by both the AOT planner and the VM import system. Resolution policy is per-module:

- **PreferNative**: Modules like `math`, `sys`, `time`, `io`, `gc`, `itertools`, `struct`, `array`, `marshal`, `binascii`, `signal`, `_thread`, `_codecs`, `_sre`, `_functools`, `weakref`, `typing`, `collections`, `operator`, `inspect`, `pickle`, and platform modules (`nt`, `msvcrt`, `winreg` on Windows; `posix` on Unix).
- **PreferSourceWhenAvailable**: Modules like `os`, `os.path`, `json`, `functools`, and `re`, which prefer a source-backed implementation when the filesystem is available.

Source fallback modules (e.g., `abc`, `contextlib`, `decimal`, `fractions`, `unittest`) are shipped in `prism_stdlib/python/`.

The import system also supports frozen source modules so the AOT pipeline can bundle code without runtime filesystem access.

---

## Building

### Prerequisites

- Rust 1.94+ (edition 2024)
- x86-64 target support (ARM64 supported with the `arm64` feature or `aarch64` target)
- Optional: a CPython 3.12 source checkout for `prism-test` (a vendored test subset is included)

### Build

```bash
cargo build --workspace
cargo build --workspace --release
```

### Test

```bash
cargo test --workspace
```

### Bench

```bash
cargo bench --workspace
```

Benchmarks are available in `prism_core`, `prism_vm`, and `prism_runtime`.

---

## Project Status

Prism is under active development. The runtime is architecturally broad — from parser through JIT — and it already runs meaningful end-to-end Python programs. Compatibility work continues across the standard library and CPython edge-case behavior.

### What Works Today

- Python 3.12 parsing and AST support across the language surface
- Source-to-bytecode compilation through shared frontend APIs, including class hierarchies, exception handling, pattern matching, closures, and generators
- Bytecode interpretation with 256-register frames, dispatch tables, inline caching, and type feedback
- Tier-up from interpreter into Tier 1 (baseline) and Tier 2 (optimizing) JIT with OSR and deoptimization
- Generational GC with nursery, survivor, tenured, and large-object spaces; concurrent major marking; and JIT safepoints
- Generator and coroutine execution with frame pooling and async iteration
- Hidden class (shape) transitions for fast attribute access
- Native stdlib coverage for 50+ modules plus source-backed fallback imports
- AOT planning with frozen-module bundling, manifest emission, and linkable object output
- CPython compatibility harness with vendored test corpus

### In Progress

- Broader stdlib coverage and behavior parity
- Standalone executable packaging from the AOT pipeline
- Expanded concurrent GC capabilities
- ARM64 JIT backend maturation

For the most current runtime behavior, inspect the implementation crates directly and use the `prism-test` harness against a CPython 3.12 checkout.

---

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.
