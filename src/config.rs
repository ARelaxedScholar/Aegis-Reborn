use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize, Debug)]
pub struct MetricsConfig {
    pub risk_free_rate: f64,
}

#[derive(Deserialize, Debug)]
pub struct DataConfig {
    pub file_path: String,
    pub hold_out_split: f64,
}

#[derive(Deserialize, Debug)]
pub struct GaConfig {
    pub population_size: usize,
    pub num_generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub tournament_size: usize,
    pub initial_genome_length: usize,
    pub max_program_tokens: usize,
    pub max_recursion_depth: u32,
    pub parsimony_penalty: f64,
    pub alpha: f64,
    pub training_window_size: usize,
    pub test_window_size: usize,
}

#[derive(Deserialize, Debug)]
pub struct Config {
    pub grammar_file: String,
    pub data: DataConfig,
    pub ga: GaConfig,
    pub metrics: MetricsConfig,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

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
        if self.ga.training_window_size < 2 || self.ga.test_window_size < 2 {
            return Err(
                "The training and test window sizes must be at least 2 for meaningful analysis"
                    .to_string(),
            );
        }
        Ok(())
    }
}
