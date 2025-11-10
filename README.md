# Aegis Reborn: The Path of the Modern Alchemist 🏛️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built with Nix](https://img.shields.io/badge/Built%20with-Nix-5277C3.svg?logo=nixos&labelColor=777777)](https://nixos.org)

Aegis Reborn is a framework for **evolving** trading strategies from scratch using Grammatical Evolution in Rust. This is a tool for research, not live execution.

We don't tell the machine how to trade. We give it a basic language—the building blocks of technical analysis—and it uses a process modeled on Darwinian evolution to discover its own strategies. Good strategies survive and breed; bad ones are discarded. The final output is a statistically rigorous report, not a trade signal, designed to answer one question: "Is this strategy genuinely robust, or just a fluke of the data?"

Aegis Reborn aims to makes that first step, Alpha discovery simple. 

## A Glimpse of Emergent Logic

The **"Crawl"** phase of this project is complete, and the system has already demonstrated its capability by autonomously evolving a logically sound strategy from a simple grammar.

The strategy that emerged can be translated into human-readable logic:
> **Entry:** If the `Close` price is 1% above the `SMA(100)`, enter a long position. *(This identifies a strong upward trend.)*
>
> **Exit:** If the `RSI(14)` is less than or equal to 50, exit the position. *(This exits when momentum becomes neutral or bearish, protecting profits.)*

On the hold-out set, this simple, evolved strategy produced a **Sharpe Ratio of 0.67** and a **Smoothed Calmar Ratio of 2.77**. While not a world-beating result, its logical coherence is a powerful proof of concept for the evolutionary process.

![Gauntlet Report Screenshot](https://github.com/user-attachments/assets/e90a4b6f-db74-47ff-a1bd-7a9e708af898)

## Getting Started

### Prerequisites

1.  **Git:** To clone the repository.
2.  **Rust Toolchain:** For standard builds via `cargo`. If you don't have it, install it from [rustup.rs](https://rustup.rs/).
3.  **Nix (Optional):** For fully reproducible builds. Install it from [nixos.org](https://nixos.org/).

### Installation & Configuration

First, clone the repository to your local machine:```bash
git clone https://github.com/your-username/aegis-reborn.git
cd aegis-reborn
```
Next, set up your configuration:
```bash
# Copy the example configuration to a new file that you will edit
cp config.example.toml config.toml
```
Now, **open `config.toml`** in a text editor and modify the `grammar_file` and `data.file_path` to point to your `.bnf` grammar and your OHLCV data CSV file.

## Usage

You can run the application using one of the two methods below.

---

### Method 1: Using Cargo (Standard)
1.  **Build and Run:**
    ```bash
    cargo run --release
    ```
    This compiles the project in release mode and runs the full evolution and validation process. The first run will take some time; subsequent runs will be much faster.

---

### Method 2: Using Nix (For Reproducible Builds)

This method uses Nix to create a fully reproducible build environment. This guarantees the Aegis Reborn binary itself is identical for every build, though the stochastic nature of the genetic algorithm means each run will explore a different evolutionary path.

1.  **Enter the Development Shell:**
    ```bash
    nix develop
    ```
2.  **Use Cargo as Normal:**
    ```bash
    # (You are now inside the nix-shell)
    cargo run --release
    ```
3.  **One-Step Run:**
    ```bash
    nix run .
    ```

---

## The Philosophy: Crawl, Walk, Run

We are building this system in three distinct phases to manage complexity and ensure a robust foundation.

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

*   ⏳ **WALK (In Progress):** Now that the core is proven, we will teach it nuance. This phase will introduce more complex logic and risk management, such as nested `IF/THEN/ELSE` expressions, evolvable **Stop-Loss** and **Take-Profit** levels, and dynamic **Position Sizing**.

*   🚀 **RUN (The Final Form):** The ultimate vision. This phase will expand the system to a full-featured research platform, introducing **shorting capabilities** and tackling **multi-asset portfolio optimization**.

## Project Roadmap

#### Phase 1: CRAWL - The Signal Discovery Engine (MVP)
*   **Objective:** Prove that the core architecture can discover a robust, long-only alpha signal for a single asset.
*   **Milestones:**
    *   `[✅]` **Step 0: Language Definition:** Finalize the Minimal Viable Grammar (BNF) and Bytecode (`Op`) language.
    *   `[✅]` **Step 1: Core Executor:** Build and unit-test the Virtual Machine (VM).
    *   `[✅]` **Step 2: Data & Harness:** Implement a production-grade data loader and a minimal viable backtesting harness.
    *   `[✅]` **Step 3: The Factory:** Build the `GrammarBasedMapper` to compile genomes into bytecode strategies.
    *   `[✅]` **Step 4: The Evolution Engine:** Implement the full Genetic Algorithm loop with a Calmar-based fitness function.
    *   `[✅]` **Step 5: The Professional Gauntlet:**
        *  `[✅]` Implement the **Walk-Forward Analysis** engine.
        *  `[✅]` Implement the post-evolution **Block Bootstrapping** stress test.
        *  `[✅]` Implement the final **Hold-Out Set** validation and reporting.
*   **Exit Criteria:** The system can autonomously run, from config file to final report, and produce a "Council of Champions" that have been rigorously validated.

#### Phase 2: WALK - The Sophisticated Trader
*   **Objective:** Enhance the engine to evolve more complex, realistic, and risk-managed strategies.
*   **Key Features / Milestones:**
    *   **Grammar Expansion I (Advanced Logic):** Implement nested `IF/THEN/ELSE` expressions and upgrade the `Mapper` with back-patching.
    *   **Grammar Expansion II (Risk Management):** Introduce evolvable programs for **Stop-Loss** and **Take-Profit**.
    *   **Grammar Expansion III (Capital Allocation):** Introduce an evolvable **Position Sizing** program.
    *   **Evolutionary Tuning:** Implement more advanced **genetic operators** and **opcode-weighted parsimony pressure**.
*   **Exit Criteria:** The system can evolve strategies that include dynamic stops, take-profits, and position sizing.

#### Phase 3: RUN - The Full-Scale Quant Platform
*   **Objective:** Expand the system into a multi-asset research platform capable of handling both long and short strategies.
*   **Key Features / Milestones:**
    *   **Introduce Shorting:** Upgrade the backtester and grammar to handle short positions.
    *   **Multi-Asset Framework:** Evolve **portfolio allocation** models alongside alpha signals.
    *   **Co-evolution:** Implement a feedback loop where strategies and portfolio allocations evolve in tandem.
*   **Exit Criteria:** The system is a feature-complete framework for discovering and validating complex, multi-asset, long/short trading portfolios.

## Contributing

This project is under active development. Contributions, feature suggestions, and bug reports are welcome! Please open an issue to discuss your ideas.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
