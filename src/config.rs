use log::warn;
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// This struct encapsulates the configuration information
/// for metrics computation. Note they are NOT necessarily essential for the core grammatical evolution component
/// (might be tangentially involved), but they are helpful for tracking.
#[derive(Clone, Copy, Deserialize, Debug)]
pub struct MetricsConfig {
    /// Starting amount of money
    pub initial_cash: f64,
    /// Number of candles in a year (for a daily chart would be 252.0)
    pub annualization_rate: f64,
    /// Expected rate of return for a risk-free investment
    pub risk_free_rate: f64,
    /// Number of bootstrap runs to do for the final gauntlet (at this point evolution is over)
    pub bootstrap_runs: usize,
    /// Transaction cost percentage per side (e.g., 0.001 for 0.1% per trade)
    #[serde(default)]
    pub transaction_cost_pct: f64,
}

/// This struct encapsulates the logic related to taking CSV data from the user and preparing it
/// for the evolution
#[derive(Deserialize, Debug)]
pub struct DataConfig {
    /// File to the CSV file containing the candles
    pub file_path: String,
    /// Percentage of the data (between 0.0 and 1.0) to keep out for tests
    pub hold_out_split: f64,
}

/// This struct encapsulates all of the information related to the actual
/// core evolutionary process.
#[derive(Clone, Copy, Deserialize, Debug, Default)]
pub struct GaConfig {
    /// Size of the population (no shit, Sherlock)
    pub population_size: usize,
    /// Number of generations to run the evolution process
    pub num_generations: usize,
    /// Per-Gene Mutation Rate (chance that any given gene may be mutated)
    pub mutation_rate: f64,
    /// Rate at which "reproduction" between successful individuals successfully occurs
    pub crossover_rate: f64,
    /// For tournament selection, represents the number of individuals in a tournament
    pub tournament_size: usize,
    pub initial_genome_length: usize,
    /// Max number of tokens allowed in a given program for it to be considered valid
    pub max_program_tokens: usize, // meant to punish needlessly long strategies, and avoid bugs
    // where ultra-specific strategies find an edge in the data
    // based on random noise.
    /// Max recursion depth allowed during the mapping process before failing a given `Strategy`
    pub max_recursion_depth: u32,
    /// The strength of the penalty assigned to a bigger number of tokens
    pub parsimony_penalty: f64, // In the future, implementing something which takes into account
    // the type of tokens used (Indicators are more complex than
    // constants) is in the works. But need more data to justify the
    // weights.
    /// Smoothing constant used in the denominator of our fitness function "Smoothed Calmar Ratio"
    /// to prevent solutions with near zero volatility (very likely a fluke) to dominate the pool
    /// (and remove the kink at 0 entirely)
    pub alpha: f64,
    /// Number of candles in the scenarios used during the testing (final gauntlet) period
    pub test_window_size: usize,
    /// Number of top solutions which are kept for further evaluation in the final gauntlet
    pub size_of_council: usize,
}

/// This struct combines the previous struct in
/// one struct
#[derive(Deserialize, Debug)]
pub struct Config {
    /// Path to the .bnf file containing the user-defined grammar
    pub grammar_file: String,
    /// The `DataConfig` struct
    pub data: DataConfig,
    /// The `GaConfig` struct
    pub ga: GaConfig,
    /// The `MetricsConfig` struct
    pub metrics: MetricsConfig,
}

impl Config {
    /// Parses and loads the user-defined config
    ///
    /// # Arguments
    /// * `path` - Reference to a `Path` struct representing the path to the the user-specifed
    /// config
    ///
    /// # Returns
    /// * `Result<Self, std::error::Error>`
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Validates the config
    ///
    /// # Arguments
    /// * `&self`
    ///
    /// # Returns
    /// * `Result<Self, std::error::Error>`
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.data.hold_out_split) {
            return Err("data.hold_out_split must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&self.ga.mutation_rate) {
            return Err("ga.mutation_rate must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&self.ga.crossover_rate) {
            return Err("ga.crossover_rate must be between 0.0 and 1.0".to_string());
        }
        if self.ga.population_size == 0 {
            return Err("ga.population_size must be greater than 0".to_string());
        }
        if self.ga.tournament_size < 2 {
            return Err("ga.tournament_size must be at least 2".to_string());
        }
        if self.ga.test_window_size < 2 {
            return Err(
                "The test window size must be at least 2 for meaningful analysis"
                    .to_string(),
            );
        }
        if self.ga.size_of_council == 0 {
            return Err("ga.size_of_council must be at least 2".to_string());
        }
        if self.ga.size_of_council > self.ga.population_size {
            return Err("ga.size_of_council cannot be bigger than population size".to_string());
        }
        if self.metrics.bootstrap_runs < 1 {
            return Err("bootstrap_runs must be at least 1, a good default is 1_000, 2000, you might observe diminising returns beyond that".to_string());
        }
        if self.metrics.initial_cash <= 0.0 {
            return Err("initial_cash must be positive".to_string());
        }
        if self.metrics.annualization_rate <= 0.0 {
            return Err("annualization_rate must be positive".to_string());
        }
        if !self.metrics.transaction_cost_pct.is_finite() {
            return Err("transaction_cost_pct must be a finite number".to_string());
        }
        if self.metrics.transaction_cost_pct < 0.0 {
            return Err("transaction_cost_pct must be non-negative".to_string());
        }
        if self.metrics.transaction_cost_pct > 1.0 {
            return Err("transaction_cost_pct must be <= 1.0 (100%)".to_string());
        }
        if self.metrics.transaction_cost_pct > 0.5 {
            warn!("transaction_cost_pct is unusually high (>50%). This may be intentional for stress testing.");
        }
        Ok(())
    }
}
