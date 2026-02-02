<p align="center">
  <h1 align="center">⬡ Prism</h1>
  <p align="center"><i>A high-performance Python runtime with JIT compilation</i></p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#building">Building</a> •
  <a href="#contributing">Contributing</a>
</p>

---

Prism is a from-scratch implementation of the Python 3.12 runtime, engineered for performance through a custom multi-tier JIT compiler. Written entirely in Rust, Prism combines a register-based bytecode interpreter with an optimizing compiler that generates native x64 machine code.

## Features

### Multi-Tier Execution Engine
- **Tier 0 Interpreter** — Register-based bytecode VM with static dispatch tables and arithmetic fast-paths
- **Tier 1 Template JIT** — Direct bytecode-to-machine-code translation with minimal compilation overhead
- **Tier 2 Optimizing JIT** — Sea-of-Nodes IR with aggressive optimizations and profile-guided compilation

### Advanced Optimizations
- **Inline Caching** — Monomorphic and polymorphic caches for property access and method dispatch
- **Type Speculation** — Profile-driven type guards with fast-path native arithmetic
- **On-Stack Replacement** — Mid-loop tier-up from interpreter to optimized code
- **Loop Optimizations** — LICM, Range Check Elimination, and induction variable analysis
- **Function Inlining** — Budget-based graph merging with escape analysis

### High-Performance Memory Management
- **Generational GC** — Immix-based heap with opportunistic evacuation and line-level marking
- **Thread-Local Allocation** — Zero-synchronization bump allocation via TLABs
- **Precise Stack Scanning** — Stackmap-driven root identification in JIT frames
- **Page-Protection Safepoints** — Minimal-overhead stop-the-world coordination

### V8-Style Object Model
- **Hidden Classes (Shapes)** — O(1) property access through inline slots and transition chains
- **NaN-Boxing** — Efficient 64-bit value representation for primitives and pointers
- **Small Integer Cache** — Pre-allocated integers from -5 to 256

### Python 3.12 Compatibility
- **Complete Parser** — Pratt parser with 16 precedence tiers for Python's complex grammar
- **Scope Analysis** — Deep binding analysis with Local/Global/Cell/Free variable resolution
- **Arbitrary Precision Integers** — Full `BigInt` support for Python integer semantics
- **Standard Library Foundations** — `math`, `sys`, and `os` modules with native performance

## Quick Start

Run a Python script:

```bash
prism script.py
```

Start the interactive REPL:

```bash
prism
```

## Architecture

Prism is organized as a modular Rust workspace:

```
prism/
├── prism_core      # Fundamental types: Value (NaN-boxing), Span, Error
├── prism_parser    # Python 3.12 grammar and AST construction
├── prism_compiler  # Scope analysis and register-based bytecode emission
├── prism_vm        # Execution engine, interpreter, and JIT orchestration
├── prism_jit       # Multi-tier JIT: IR, optimization passes, x64 codegen
├── prism_runtime   # Object system, shapes, and type implementations
├── prism_gc        # Generational Immix collector with TLABs
├── prism_builtins  # Builtin function implementations
└── prism_cli       # Command-line interface
```

### Execution Pipeline

```
                              ┌─────────────────────────────────────────┐
                              │              Tier 2 JIT                 │
                              │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
                              │  │   GVN   │─▶│  LICM   │─▶│   RCE   │  │
                              │  └─────────┘  └─────────┘  └─────────┘  │
                              │         │                       │       │
                              │         ▼                       ▼       │
                              │  ┌─────────────────────────────────┐    │
                              │  │      Register Allocation        │    │
Source ─▶ Parser ─▶ Compiler ─┼─▶│         (Linear Scan)           │────┼──▶ Native x64
   │                          │  └─────────────────────────────────┘    │
   │                          └─────────────────▲───────────────────────┘
   │                                            │ OSR (hot loops)
   │                          ┌─────────────────┴───────────────────────┐
   │                          │           Tier 1 Template JIT           │
   │                          │  Direct bytecode → machine code mapping │
   │                          └─────────────────▲───────────────────────┘
   │                                            │ tier-up (hot functions)
   │                          ┌─────────────────┴───────────────────────┐
   └─────────────────────────▶│          Tier 0 Interpreter             │
                              │   Static dispatch · Type profiling      │
                              └─────────────────────────────────────────┘
```

### JIT Tier Details

| Tier | Strategy | Trigger | Optimizations |
|:-----|:---------|:--------|:--------------|
| **0** | Interpreter | Default | Inline caches, type feedback collection |
| **1** | Template | ~100 calls | Direct translation, speculative guards |
| **2** | Optimizing | ~1000 calls or hot loop | GVN, DCE, LICM, RCE, Inlining, Escape Analysis |

### Object Model

Prism implements a V8-style hidden class system:

```
                    ┌──────────────────┐
                    │      Shape       │
                    │  (Hidden Class)  │
                    ├──────────────────┤
                    │ property: "x"    │────┐
                    │ slot: 0          │    │ transition
                    │ parent: ─────────┼────┘
                    └──────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │     Object A     │            │     Object B     │
    ├──────────────────┤            ├──────────────────┤
    │ header (16 bytes)│            │ header (16 bytes)│
    │ slot[0]: 42      │            │ slot[0]: 100     │
    │ slot[1]: ...     │            │ slot[1]: ...     │
    └──────────────────┘            └──────────────────┘
```

Objects with identical property insertion order share the same Shape, enabling O(1) property access through fixed inline slots.

## Building

### Prerequisites

- **Rust 1.85+** (2024 Edition)
- **x64 architecture** (ARM64 support planned)

### Build

```bash
# Debug build
cargo build --workspace

# Release build (recommended for benchmarking)
cargo build --workspace --release

# Run tests
cargo test --workspace
```

### Release Profile

The release profile is tuned for maximum performance:

```toml
[profile.release]
lto = "fat"           # Link-time optimization
codegen-units = 1     # Single codegen unit for better optimization
panic = "abort"       # Reduced binary size
strip = true          # Strip symbols
```

## Project Status

Prism is under active development. Current status:

| Component | Status | Tests |
|:----------|:-------|:------|
| Parser | ✅ Complete | 153 |
| Compiler | ✅ Complete | — |
| VM & Interpreter | ✅ Complete | 31 integration |
| Object System (Shapes) | ✅ Complete | 180+ |
| Garbage Collector | ✅ Complete | — |
| JIT Tier 1 & 2 | ✅ Complete | 350+ |
| Builtins | ✅ Complete | 189 |
| Math Module | ✅ Complete | 305 |
| Sys Module | ✅ Complete | 172 |
| OS Module | 🚧 In Progress | — |

**Total test coverage: 1600+ tests**

### Roadmap

- [ ] Exception system with zero-cost try blocks
- [ ] Generator/async support with minimal-overhead state machines
- [ ] ARM64 backend
- [ ] Extended standard library coverage
- [ ] Package import system

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

<p align="center">
  <sub>Built with Rust 🦀</sub>
</p>
