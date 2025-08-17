use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize, Debug)]
pub struct Config {
    pub grammar_file: String,
    pub data_file: String,
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub num_generations: usize,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
