//! Champion export module for persisting evolved strategies and their metadata.
//!
//! This module provides functionality to export the council of champions with
//! complete metadata (configuration, grammar, strategies) for post-hoc
//! transpilation and reproducibility.

use crate::config::{DataConfig, GaConfig, MetricsConfig, TranspilerConfig};
use crate::evolution::Individual;
use crate::strategy::Strategy;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// Comprehensive export of council champions with full metadata for reproducibility.
#[derive(Serialize, Deserialize)]
pub struct ChampionExport {
    /// Schema version for forward/backward compatibility
    pub schema_version: String,
    /// Unix timestamp when export was generated
    pub generated_at: u64,
    /// Snapshot of evolution configuration
    pub evolution_config: ExportConfig,
    /// Full content of the grammar file
    pub grammar_content: String,
    /// Hash of grammar content for verification
    pub grammar_hash: String,
    /// Transpiler configuration (if present in original config)
    pub transpiler_config: Option<TranspilerConfig>,
    /// Council champions (top `size_of_council` individuals)
    pub champions: Vec<ChampionData>,
}

/// Subset of configuration relevant for reproducibility
#[derive(Serialize, Deserialize, Clone)]
pub struct ExportConfig {
    /// Genetic algorithm configuration
    pub ga: GaConfig,
    /// Metrics configuration
    pub metrics: MetricsConfig,
    /// Data configuration
    pub data: DataConfig,
}

/// Champion data including genome, fitness, and mapped strategy
#[derive(Serialize, Deserialize)]
pub struct ChampionData {
    /// Rank within council (1 = best)
    pub rank: usize,
    /// Fitness score from evolution
    pub fitness: f64,
    /// Raw genome (sequence of u32)
    pub genome: Vec<u32>,
    /// Mapped strategy (programs)
    pub strategy: Strategy,
}

impl ChampionExport {
    /// Creates a new champion export from evolution results.
    ///
    /// # Arguments
    /// * `council` - Council champions (already sorted by fitness, best first)
    /// * `evolution_config` - Full evolution configuration
    /// * `grammar_content` - Content of the grammar file
    /// * `transpiler_config` - Optional transpiler configuration
    ///
    /// # Returns
    /// A new `ChampionExport` instance ready for serialization.
    pub fn new(
        council: Vec<Individual>,
        evolution_config: ExportConfig,
        grammar_content: String,
        transpiler_config: Option<TranspilerConfig>,
    ) -> Self {
        let grammar_hash = compute_grammar_hash(&grammar_content);
        let champions = council
            .into_iter()
            .enumerate()
            .map(|(i, ind)| ChampionData {
                rank: i + 1,
                fitness: ind.fitness,
                genome: ind.genome,
                strategy: Strategy::new(), // Placeholder - will be set by caller
            })
            .collect();

        Self {
            schema_version: "1.0.0".to_string(),
            generated_at: chrono::Utc::now().timestamp() as u64,
            evolution_config,
            grammar_content,
            grammar_hash,
            transpiler_config,
            champions,
        }
    }

    /// Updates champion strategies after mapping genomes to programs.
    ///
    /// # Arguments
    /// * `strategies` - Vector of strategies in same order as champions
    pub fn update_strategies(&mut self, strategies: Vec<Strategy>) {
        assert_eq!(self.champions.len(), strategies.len());
        for (champion, strategy) in self.champions.iter_mut().zip(strategies) {
            champion.strategy = strategy;
        }
    }

    /// Validates that grammar hash matches grammar content.
    pub fn validate_grammar_hash(&self) -> bool {
        compute_grammar_hash(&self.grammar_content) == self.grammar_hash
    }
}

/// Computes a hash of grammar content for reproducibility.
fn compute_grammar_hash(grammar_content: &str) -> String {
    let mut hasher = DefaultHasher::new();
    grammar_content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Writes champion export to a JSON file.
pub fn write_export_to_json(
    export: &ChampionExport,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(export)?;
    std::fs::write(output_path, json)?;
    Ok(())
}

/// Reads champion export from a JSON file.
pub fn read_export_from_json(
    input_path: &Path,
) -> Result<ChampionExport, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(input_path)?;
    let export: ChampionExport = serde_json::from_str(&content)?;
    Ok(export)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DataConfig, GaConfig, MetricsConfig};
    use tempfile::NamedTempFile;

    fn create_test_config() -> ExportConfig {
        ExportConfig {
            ga: GaConfig {
                population_size: 100,
                num_generations: 10,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
                tournament_size: 3,
                initial_genome_length: 20,
                max_program_tokens: 50,
                max_recursion_depth: 256,
                parsimony_penalty: 0.01,
                alpha: 0.05,
                test_window_size: 21,
                size_of_council: 5,
            },
            metrics: MetricsConfig {
                initial_cash: 10000.0,
                annualization_rate: 252.0,
                risk_free_rate: 0.02,
                bootstrap_runs: 100,
                transaction_cost_pct: 0.001,
                slippage_pct: 0.001,
            },
            data: DataConfig {
                file_path: "test.csv".to_string(),
                hold_out_split: 0.3,
            },
        }
    }

    fn create_test_individuals() -> Vec<Individual> {
        vec![
            Individual {
                genome: vec![1, 2, 3, 4, 5],
                fitness: 0.8,
            },
            Individual {
                genome: vec![6, 7, 8, 9, 10],
                fitness: 0.6,
            },
        ]
    }

    #[test]
    fn test_export_creation() {
        let config = create_test_config();
        let individuals = create_test_individuals();
        let grammar = "<start> ::= ENTRY 1.0".to_string();

        let export = ChampionExport::new(individuals, config, grammar.clone(), None);

        assert_eq!(export.schema_version, "1.0.0");
        assert_eq!(export.champions.len(), 2);
        assert_eq!(export.champions[0].rank, 1);
        assert_eq!(export.champions[0].fitness, 0.8);
        assert_eq!(export.champions[1].rank, 2);
        assert_eq!(export.champions[1].fitness, 0.6);
        assert!(export.validate_grammar_hash());
    }

    #[test]
    fn test_export_serialization() {
        let config = create_test_config();
        let individuals = create_test_individuals();
        let grammar = "<start> ::= ENTRY 1.0".to_string();

        let export = ChampionExport::new(individuals, config, grammar, None);

        let temp_file = NamedTempFile::new().unwrap();
        write_export_to_json(&export, temp_file.path()).unwrap();

        let loaded = read_export_from_json(temp_file.path()).unwrap();
        assert_eq!(loaded.schema_version, export.schema_version);
        assert_eq!(loaded.champions.len(), export.champions.len());
        assert_eq!(loaded.champions[0].rank, 1);
        assert_eq!(loaded.champions[0].fitness, 0.8);
        assert!(loaded.validate_grammar_hash());
    }

    #[test]
    fn test_grammar_hash() {
        let grammar1 = "<start> ::= ENTRY 1.0";
        let grammar2 = "<start> ::= ENTRY 1.0";
        let grammar3 = "<start> ::= ENTRY 2.0";

        let hash1 = compute_grammar_hash(grammar1);
        let hash2 = compute_grammar_hash(grammar2);
        let hash3 = compute_grammar_hash(grammar3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
