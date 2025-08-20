### **Aegis Reborn: An Alchemical Journey into Strategy Evolution**

Aegis Reborn is a framework for **evolving** trading strategies from scratch using Grammatical Evolution in Rust.

We don't tell the machine how to trade. We give it a basic language—the building blocks of technical analysis—and it uses a process modeled on Darwinian evolution to discover its own strategies. Good strategies survive and breed. Bad ones are discarded.

This project is an experiment in discovering genuine, data-driven alpha by combining a high-performance backtesting engine with the creative, unconstrained search of an evolutionary algorithm.

### The Philosophy: Crawl, Walk, Run

We are building this system in three distinct phases to manage complexity and ensure a robust foundation.

*   ✅ **CRAWL (We Are Here):** The mission is to build a flawless, end-to-end engine for a single, focused problem: discovering a **long-only** strategy for a **single asset**. This phase is about proving the core architecture is sound.

*   ⏳ **WALK (The Road Ahead):** Once the core is proven, we will teach it nuance. This phase will introduce more complex logic and risk management, such as nested `IF/THEN/ELSE` expressions, evolvable **Stop-Loss** and **Take-Profit** levels, and dynamic **Position Sizing**.

*   🚀 **RUN (The Final Form):** The ultimate vision. This phase will expand the system to a full-featured research platform, introducing **shorting capabilities** and tackling the holy grail of quantitative finance: **multi-asset portfolio optimization** (At which point the previous Aegis, which will be renamed Athena, will join back combining alpha-generation with data-driven portfolio allocations.)

### Current Progress & Next Steps

The **Crawl** phase is nearly complete. We have successfully built and tested:
*   A high-performance **Bytecode Virtual Machine (VM)** for strategy execution.
*   A robust **Data Pipeline** and **Backtesting Harness**.
*   The **Grammar-to-Bytecode Compiler (`Mapper`)**.
*   The core **`EvolutionEngine`**, which is already discovering non-trivial, logical strategies.

The final and most critical task of the Crawl phase remains:

*   **Implement the Professional Evaluation Gauntlet (Step 5):** We must replace our current simple backtest with the full, three-stage validation suite: **Walk-Forward Analysis**, **Block Bootstrapping**, and a final, unbiased **Hold-Out Set** test. This is the crucible that will forge our first true champions.

---
---

### **Project Roadmap: Aegis Reborn**

#### **Phase 1: CRAWL - The Signal Discovery Engine (MVP)**
*   **Objective:** Prove that the core architecture can discover a robust, long-only alpha signal for a single asset.
*   **Milestones:**
    *   `[✅]` **Step 0: Language Definition:** Finalize the Minimal Viable Grammar (BNF) and Bytecode (`Op`) language.
    *   `[✅]` **Step 1: Core Executor:** Build and unit-test the Virtual Machine (VM).
    *   `[✅]` **Step 2: Data & Harness:** Implement a production-grade data loader and a minimal viable backtesting harness (MVB).
    *   `[✅]` **Step 3: The Factory:** Build the `GrammarBasedMapper` to compile genomes into bytecode strategies.
    *   `[✅]` **Step 4: The Evolution Engine:** Implement the full Genetic Algorithm loop with a Calmar-based fitness function.
    *   `[⏳]` **Step 5: The Professional Gauntlet:**
        *  `[ ✅]` Implement the **Walk-Forward Analysis** engine and integrate it as the primary fitness function.
        *  `[⏳]` Implement the post-evolution **Block Bootstrapping** stress test.
        *  `[⏳]` Implement the final **Hold-Out Set** validation and reporting.
*   **Exit Criteria:** The system can autonomously run, from config file to final report, and produce a "Council of Champions" that have been rigorously validated against our full gauntlet.

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
