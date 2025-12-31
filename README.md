### **Golden Aegis: An Alchemical Journey into Strategy Evolution**

Golden Aegis is a framework for **evolving** trading strategies from scratch using Grammatical Evolution in Rust.

We don't tell the machine how to trade. We give it a basic language—the building blocks of technical analysis—and it uses a process modeled on Darwinian evolution to discover its own strategies. Good strategies survive and breed. Bad ones are discarded.

This project is an experiment in discovering genuine, data-driven alpha by combining a high-performance backtesting engine with the creative, unconstrained search of an evolutionary algorithm.



### Prerequisites

- **Rust** (via [rustup](https://rustup.rs/)) – the latest stable version (1.85.0 or higher)
- **Git** – to clone the repository
- A terminal and basic command-line familiarity

### Installation

Clone the repository (replace `<repository-url>` with your fork or the original URL):

```bash
git clone <repository-url>
cd golden-aegis
cargo build --release
```

### Quick Start

1. Copy the example configuration:
   ```bash
   cp config.example.toml config.toml
   ```
2. Ensure the default data file `data/BTC-USD.csv` exists (it is included in the repository).
3. Run the evolution with default settings:
   ```bash
   cargo run --release
   ```
   To use a custom configuration file:
   ```bash
   cargo run --release -- --config my_strategy.toml
   ```
4. View the generated reports in the `output_examples/` directory.

### Data Format

Golden Aegis expects CSV files with the following columns (case-insensitive):
- `Date` – in `YYYY-MM-DD` format (e.g., `2014-09-17`)
- `Open`, `High`, `Low`, `Close` – price values (floating-point)
- `Adj Close` – adjusted close price (used by default; falls back to `Close` if absent)
- `Volume` – trading volume (floating-point)

The CSV loader automatically detects common column name variations (`open`, `OPEN`, `Open`, etc.).  
Data from Yahoo Finance, Binance, or other sources that follow this convention should work without modification.

### Usage

- **Logging**: Set the `RUST_LOG` environment variable to control log output:
  ```bash
  RUST_LOG=info cargo run --release
  ```
- **Command-line options**:
  ```bash
  cargo run --release -- --help            # Show all available options
  cargo run --release -- --config custom.toml  # Use a custom configuration file
  ```
- **Output**: After each run, a JSON report is written to `output_examples/gauntlet_report_<timestamp>.json` containing the performance metrics of the top‑evolved strategies.

### Results / Case Study

During the **Crawl** phase, the system discovered the following strategy:

> **Entry**: `Close` is 1 % above the 100‑period Simple Moving Average (SMA)  
> **Exit**: 14‑period Relative Strength Index (RSI) ≤ 50

This strategy achieved a **smoothed Calmar ratio of 2.7709** and a **Sharpe ratio of 0.667** on the hold‑out set, demonstrating that the evolutionary process can generate semantically meaningful, profitable rules without human intervention.

![Equity curve of the discovered strategy](https://github.com/user-attachments/assets/e90a4b6f-db74-47ff-a1bd-7a9e708af898)

### Advanced Installation (Nix)

If you use [Nix](https://nixos.org/), you can enter a fully‑contained development environment with all dependencies pre‑installed:

```bash
nix develop          # Start a development shell (cargo, rustfmt, rust‑analyzer, etc.)
nix build .#golden-aegis  # Build the binary using the Nix‑managed toolchain
```

The project includes a `flake.nix` and `Cargo.nix` for reproducible builds.

---

### The Philosophy: Crawl, Walk, Run

We are building this system in three distinct phases to manage complexity and ensure a robust foundation.

*   ✅ **CRAWL (Mission Accomplished):** The mission is to build a flawless, end-to-end engine for a single, focused problem: discovering a **long-only** strategy for a **single asset**. This phase is about proving the core architecture is sound. And hurray, we accomplished it! The algorithm even generated a known strategy through the process despite the relatively simple grammar!
  

*   ⏳ **WALK (We Are Here!!!):** Once the core is proven, we will teach it nuance. This phase will introduce more complex logic and risk management, such as nested `IF/THEN/ELSE` expressions, evolvable **Stop-Loss** and **Take-Profit** levels, and dynamic **Position Sizing**.

*   🚀 **RUN (The Final Form):** The ultimate vision. This phase will expand the system to a full-featured research platform, introducing **shorting capabilities** and tackling the holy grail of quantitative finance: **multi-asset portfolio optimization**.

### Current Progress & Next Steps

The **Crawl** phase is now complete!!! We have successfully built and tested:
*   A high-performance **Bytecode Virtual Machine (VM)** for strategy execution.
*   A robust **Data Pipeline** and **Backtesting Harness**.
*   The **Grammar-to-Bytecode Compiler (`Mapper`)**.
*   The core **`EvolutionEngine`**, which is already discovering non-trivial, logical strategies.

The final and most critical task of the Crawl phase remains:
The crucible is implemented and we even got a semantically meaningful candidate quite quickly!
```if the ClosePercent is 1% above the SMA(100) [can be understood as the Close price is 1% above the SMA of 100 periods, which indicate strong upward trend, get in the trade; If the RSI(14) <= 50  you leave, 50 is generally considered neutral (neither oversold, nor overbought.) So leave when the direction of the market is uncertain, this makes sense.```
Gave a smoothed Calmar ratio of 2.7709 (quite impressive) and 0.667 Sharpe Ratio on the Hold Out set (nothing crazy, but still quite happy the algo generated a semantically meaningful strategy.)
<img width="2036" height="1088" alt="image" src="https://github.com/user-attachments/assets/e90a4b6f-db74-47ff-a1bd-7a9e708af898" />

---

### **Project Roadmap: Golden Aegis**

#### **Phase 1: CRAWL - The Signal Discovery Engine (MVP)**
*   **Objective:** Prove that the core architecture can discover a robust, long-only alpha signal for a single asset.
*   **Milestones:**
    *   `[✅]` **Step 0: Language Definition:** Finalize the Minimal Viable Grammar (BNF) and Bytecode (`Op`) language.
    *   `[✅]` **Step 1: Core Executor:** Build and unit-test the Virtual Machine (VM).
    *   `[✅]` **Step 2: Data & Harness:** Implement a production-grade data loader and a minimal viable backtesting harness (MVB).
    *   `[✅]` **Step 3: The Factory:** Build the `GrammarBasedMapper` to compile genomes into bytecode strategies.
    *   `[✅]` **Step 4: The Evolution Engine:** Implement the full Genetic Algorithm loop with a Calmar-based fitness function.
    *   `[✅]` **Step 5: The Professional Gauntlet:**
        *  `[ ✅]` Implement the **Walk-Forward Analysis** engine and integrate it as the primary fitness function.
        *  `[✅]` Implement the post-evolution **Block Bootstrapping** stress test.
        *  `[✅]` Implement the final **Hold-Out Set** validation and reporting.
*   **Exit Criteria:** The system can autonomously run, from config file to final report, and produce a "Council of Champions" that have been rigorously validated against our full gauntlet. Actually accomplished.

#### **Phase 2: WALK - The Sophisticated Trader**
*   **Objective:** Enhance the engine to evolve more complex, realistic, and risk-managed strategies.
*   **Key Features / Milestones:**
    *   **Grammar Expansion I (Advanced Logic):**
        *   Implement nested `IF/THEN/ELSE` expressions.
        *   Upgrade the `Mapper` with the **back-patching** algorithm to handle forward jumps.
    *   **Grammar Expansion II (Risk Management):**
        *   Introduce `<arithmetic_expr>`-based programs for **Stop-Loss** and **Take-Profit**.
        *   Upgrade the `Backtester` to handle and track these dynamic price-level targets.
    *   **Grammar Expansion III (Capital Allocation):**
        *   Introduce an evolvable **Position Sizing** program.
        *   Upgrade the `Portfolio` model to handle variable trade sizes.
    *   **Evolutionary Tuning:**
        *   Implement more advanced **genetic operators** (e.g., chunk mutation, program-level crossover).
        *   Implement **opcode-weighted parsimony pressure** and other advanced diagnostics (e.g., diversity monitoring).
*   **Exit Criteria:** The system can evolve strategies that include dynamic stops, take-profits, and position sizing, demonstrating a clear improvement in risk-adjusted performance over the "Crawl" phase champions.

#### **Phase 3: RUN - The Full-Scale Quant Platform**
*   **Objective:** Expand the system into a multi-asset research platform capable of handling both long and short strategies.
*   **Key Features / Milestones:**
    *   **Introduce Shorting:**
        *   Upgrade the `Backtester` and `Portfolio` model to handle the asymmetrical risk of short positions.
        *   Expand the grammar to allow for short-specific entry/exit logic.
    *   **Multi-Asset Framework:**
        *   Upgrade the data layer to handle a universe of assets.
        *   Design and evolve **portfolio allocation** models.
    *   **The "Meta-Block" (Research):**
        *   Implement the co-evolutionary feedback loop where strategies and portfolio allocations evolve in tandem.
*   **Exit Criteria:** The system is a feature-complete framework capable of discovering and validating complex, multi-asset, long/short trading portfolios.
