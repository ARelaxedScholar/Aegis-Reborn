# Contributing to Golden Aegis

Thank you for your interest in contributing! This document outlines the process for setting up a development environment, running tests, and submitting changes.

## Development Environment

### Rust Toolchain

The project uses the latest stable Rust (1.85.0 or higher). If you don't have Rust installed, we recommend using [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Nix (Optional)

If you use [Nix](https://nixos.org/), you can enter a fully‑contained development shell with all dependencies pre‑installed:

```bash
nix develop
```

The development shell includes:
- `cargo` and `rustc`
- `rustfmt` and `clippy` for code formatting and linting
- `rust‑analyzer` for IDE support
- `cargo‑edit` for managing dependencies

## Building the Project

```bash
cargo build          # Debug build
cargo build --release  # Optimized release build
```

## Running Tests

The test suite includes unit tests for the core components (data loading, VM, grammar, mapper, etc.):

```bash
cargo test          # Run all tests
cargo test -- --nocapture  # Show output from passing tests
```

## Benchmarks

Performance benchmarks are defined in the `benches/` directory. To run them:

```bash
cargo bench
```

## Code Style & Linting

We enforce consistent code style using `rustfmt` and `clippy`. Please run these tools before submitting changes:

```bash
cargo fmt          # Format all Rust code
cargo clippy       # Run the linter (warnings must be addressed)
```

## Submitting Changes

1. **Fork the repository** and create a feature branch.
2. **Write tests** for any new functionality.
3. **Ensure all existing tests pass** (`cargo test`).
4. **Run `cargo fmt` and `cargo clippy`** to maintain code quality.
5. **Update documentation** (README, inline docs, etc.) as needed.
6. **Submit a pull request** with a clear description of the changes and the motivation behind them.

## Project Structure

- `src/` – main source code
  - `config.rs` – configuration loading and validation
  - `data/` – CSV loading and OHLCV data structures
  - `evolution/` – grammatical evolution engine, grammar, and mapper
  - `vm/` – bytecode virtual machine and operations
  - `evaluation/` – backtesting, walk‑forward analysis, block bootstrapping, and final gauntlet
- `benches/` – performance benchmarks
- `data/` – example CSV data (BTC‑USD)
- `output_examples/` – sample JSON reports from previous runs
- `grammar.bnf` – the BNF grammar that defines the strategy language

## Adding New Indicators

To extend the grammar with new technical indicators:

1. Add the indicator to `grammar.bnf` (see the file for examples).
2. Implement the indicator computation in `src/evaluation/indicators.rs`.
3. Add the corresponding bytecode operation in `src/vm/op.rs` and implement its execution in `src/vm/engine.rs`.
4. Update the mapper (`src/evolution/mapper.rs`) to handle the new terminal symbol.
5. Write tests to verify the indicator works correctly.

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue on GitHub with:

- A clear, descriptive title
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Relevant logs or error messages
- Your environment (Rust version, OS, etc.)

## License

By contributing to Golden Aegis, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.