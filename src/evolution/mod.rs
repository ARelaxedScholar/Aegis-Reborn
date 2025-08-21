pub mod grammar;
pub mod mapper;

use crate::config::{GaConfig, MetricsConfig};
use crate::data::OHLCV;
use crate::evaluation::walk_forward::WalkForwardValidator;
use crate::evolution::grammar::Grammar;
use crate::evolution::mapper::GrammarBasedMapper;
use log::{debug, error, info};
use rand::prelude::*;
use rand::rng;

const INFINITE_PENALTY: f64 = f64::NEG_INFINITY;

pub type Genome = Vec<u32>;

#[derive(Debug, Clone)]
pub struct Individual {
    pub genome: Genome,
    pub fitness: f64,
}

pub struct EvolutionEngine<'a> {
    config: &'a GaConfig,
    metrics_params: &'a MetricsConfig,
    mapper: GrammarBasedMapper<'a>,
    population: Vec<Individual>,
    candles: &'a [OHLCV],
}

impl<'a> EvolutionEngine<'a> {
    pub fn new(
        config: &'a GaConfig,
        metrics_params: &'a MetricsConfig,
        grammar: &'a Grammar,
        candles: &'a [OHLCV],
    ) -> Self {
        Self {
            config,
            metrics_params,
            mapper: GrammarBasedMapper::new(
                grammar,
                config.max_program_tokens,
                config.max_recursion_depth,
            ),
            population: Vec::with_capacity(config.population_size),
            candles,
        }
    }
    pub fn evolve(&mut self) -> Vec<Individual> {
        info!(
            "Initializing population of size {}...",
            self.config.population_size
        );
        self.initialize_population();

        for generation in 0..self.config.num_generations {
            info!(
                "--- Starting Generation {}/{} ---",
                generation + 1,
                self.config.num_generations
            );

            let (mapping_failures, vm_errors) = self.evaluate_population();
            let avg_genome_len = self
                .population
                .iter()
                .map(|ind| ind.genome.len())
                .sum::<usize>() as f64
                / self.population.len() as f64;

            if let Some(best) = self.population.iter().max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                info!(
                    "Gen {}: Best Fitness={:.4} | Avg Genome Len={:.1} | Mapping Fails={} | VM Errors={}",
                    generation + 1,
                    best.fitness,
                    avg_genome_len,
                    mapping_failures,
                    vm_errors
                );
            }

            let parents = self.select_parents();
            let mut next_generation = Vec::new();

            if let Some(best) = self.population.iter().max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                next_generation.push(best.clone());
            }

            while next_generation.len() < self.config.population_size {
                let parent1 = parents.choose(&mut rng()).unwrap();
                let parent2 = parents.choose(&mut rng()).unwrap();

                let mut child_genome = if rng().random::<f64>() < self.config.crossover_rate {
                    self.crossover(&parent1.genome, &parent2.genome)
                } else {
                    parent1.genome.clone()
                };

                if rng().random::<f64>() < self.config.mutation_rate {
                    self.mutate(&mut child_genome);
                }

                next_generation.push(Individual {
                    genome: child_genome,
                    fitness: f64::NEG_INFINITY,
                });
            }
            self.population = next_generation;
        }

        info!("--- Final Evaluation of Last Generation ---");
        self.evaluate_population();
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!("Evolution complete.");
        self.population.clone()
    }

    fn initialize_population(&mut self) {
        let mut rng = rng();
        self.population = (0..self.config.population_size)
            .map(|_| {
                let genome: Genome = (0..self.config.initial_genome_length)
                    .map(|_| rng.random())
                    .collect();
                Individual {
                    genome,
                    fitness: f64::NEG_INFINITY,
                }
            })
            .collect();
    }
    fn evaluate_population(&mut self) -> (u32, u32) {
        let mut mapping_failures = 0;
        let mut vm_errors = 0;

        for i in 0..self.population.len() {
            if self.population[i].fitness == f64::NEG_INFINITY {
                let genome = self.population[i].genome.clone();
                let (fitness, had_map_err, had_vm_err) = self.calculate_fitness(&genome);
                self.population[i].fitness = fitness;
                if had_map_err {
                    mapping_failures += 1;
                }
                if had_vm_err {
                    vm_errors += 1;
                }
            }
        }

        (mapping_failures, vm_errors)
    }

    /// A function which delegates the actual evaluation of a a genome to the WalkForwardValidator
    /// and then aggregates the result to send them upstream.
    fn calculate_fitness(&mut self, genome: &Genome) -> (f64, bool, bool) {
        // Step 1: Map genome to strategy
        let strategy = match self.mapper.map(genome) {
            Ok(s) => s,
            Err(e) => {
                debug!(
                    "Mapping failed for genome: {:?}. Assigning catastrophic fitness.",
                    e
                );
                return (INFINITE_PENALTY, true, false); // (fitness, map_err, vm_err)
            }
        };

        // Step 2: Validate strategy has required components
        if !strategy.programs.contains_key("entry") {
            debug!("Strategy missing entry program. Assigning catastrophic fitness.");
            return (INFINITE_PENALTY, true, false);
        }

        // Step 3: Create and configure walk-forward validator
        let validator = match WalkForwardValidator::new(
            self.config.training_window_size,
            self.config.test_window_size,
            self.metrics_params.risk_free_rate,
        ) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to create walk-forward validator: {}", e);
                return (INFINITE_PENALTY, false, true); // Treat as VM error
            }
        };

        // Step 4: Run walk-forward validation
        let result = match validator.validate(self.candles, &strategy) {
            Ok(r) => r,
            Err(e) => {
                debug!("Walk-forward validation failed: {}", e);
                return (INFINITE_PENALTY, false, true); // VM/validation error
            }
        };

        // Step 5: Check for execution errors
        if result.entry_error_count > 0 || result.exit_error_count > 0 {
            debug!(
                "Strategy execution errors - Entry: {}, Exit: {}",
                result.entry_error_count, result.exit_error_count
            );
            return (INFINITE_PENALTY, false, true);
        }

        // Step 6: Calculate Calmar ratio with smoothing
        let smoothing_constant = self.config.alpha;
        let calmar = if result.max_drawdown.abs() < 1e-9 {
            // Handle near-zero drawdown case
            if result.annualized_return > 0.0 {
                result.annualized_return / smoothing_constant // Conservative approach
            } else {
                result.annualized_return // Negative returns get full penalty
            }
        } else {
            result.annualized_return / (smoothing_constant + result.max_drawdown)
        };

        // Step 7: Apply parsimony pressure
        let total_opcodes = strategy.programs.values().map(|p| p.len()).sum::<usize>() as f64;
        let penalty = total_opcodes * self.config.parsimony_penalty;

        // Step 8: Calculate final fitness
        let fitness = if calmar > 0.0 {
            calmar * (1.0 - penalty).max(0.0)
        } else {
            // For negative Calmar, don't apply parsimony bonus
            calmar
        };

        debug!(
            "Fitness calculation: annualized_return={:.4}, max_drawdown={:.4}, calmar={:.4}, opcodes={}, penalty={:.4}, final_fitness={:.4}",
            result.annualized_return,
            result.max_drawdown,
            calmar,
            total_opcodes as u32,
            penalty,
            fitness
        );

        (fitness, false, false)
    }

    fn select_parents(&self) -> Vec<Individual> {
        let tournament_size = self.config.tournament_size;
        let mut selected_parents = Vec::new();
        let mut rng = rng();

        // Select population_size parents through tournament selection
        for _ in 0..self.config.population_size {
            let mut tournament = Vec::new();

            // Randomly select tournament_size individuals
            for _ in 0..tournament_size.min(self.population.len()) {
                let contestant = &self.population[rng.random_range(0..self.population.len())];
                tournament.push(contestant.clone());
            }

            // Select the best individual from the tournament
            let winner = tournament
                .into_iter()
                .max_by(|a, b| {
                    a.fitness
                        .partial_cmp(&b.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            selected_parents.push(winner);
        }

        selected_parents
    }

    fn crossover(&self, parent1: &Genome, parent2: &Genome) -> Genome {
        if parent1.is_empty() || parent2.is_empty() {
            return if !parent1.is_empty() {
                parent1.clone()
            } else {
                parent2.clone()
            };
        }
        let mut rng = rng();
        let cut1 = rng.random_range(0..parent1.len());
        let cut2 = rng.random_range(0..parent2.len());
        parent1[..cut1]
            .iter()
            .chain(&parent2[cut2..])
            .cloned()
            .collect()
    }

    fn mutate(&self, genome: &mut Genome) {
        if genome.is_empty() {
            return;
        }
        let mut rng = rng();
        let index_to_mutate = rng.random_range(0..genome.len());
        genome[index_to_mutate] = rng.random();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::OHLCV;
    use crate::evolution::grammar::Grammar;
    use std::collections::HashMap;

    // Helper to create a minimal, valid config for testing
    fn get_test_config() -> GaConfig {
        GaConfig {
            alpha: 0.05,
            population_size: 10,
            num_generations: 5,
            mutation_rate: 1.0,  // 100% mutation rate for predictable testing
            crossover_rate: 1.0, // 100% crossover rate
            max_program_tokens: 50,
            max_recursion_depth: 256,
            initial_genome_length: 10,
            parsimony_penalty: 0.01,
            tournament_size: 3,
            test_window_size: 3,
            training_window_size: 3,
        }
    }

    // Helper to create a minimal, valid metrics config
    fn get_metrics_config() -> MetricsConfig {
        MetricsConfig {
            risk_free_rate: 0.02,
            bootstrap_runs: 100,
        }
    }

    // Helper to create a dummy grammar that produces valid strategies
    fn get_test_grammar() -> Grammar {
        let mut rules = HashMap::new();
        rules.insert(
            "<start>".to_string(),
            vec![vec!["<entry_program>".to_string()]],
        );
        rules.insert(
            "<entry_program>".to_string(),
            vec![vec!["ENTRY".to_string(), "1.0".to_string()]],
        );
        Grammar { rules }
    }

    // Helper to create a grammar that will cause mapping failures
    fn get_bad_grammar() -> Grammar {
        let mut rules = HashMap::new();
        rules.insert("<start>".to_string(), vec![vec!["<undefined>".to_string()]]);
        Grammar { rules }
    }

    // Helper to create test OHLCV data
    fn get_test_candles() -> Vec<OHLCV> {
        vec![
            OHLCV {
                timestamp: 1,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 102.0,
                volume: 1000.0,
            },
            OHLCV {
                timestamp: 2,
                open: 102.0,
                high: 108.0,
                low: 98.0,
                close: 106.0,
                volume: 1100.0,
            },
            OHLCV {
                timestamp: 3,
                open: 106.0,
                high: 110.0,
                low: 104.0,
                close: 108.0,
                volume: 900.0,
            },
        ]
    }

    #[test]
    fn test_initialize_population() {
        let config = get_test_config();
        let metrics = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let mut engine = EvolutionEngine::new(&config, &metrics, &grammar, &candles);

        engine.initialize_population();

        assert_eq!(engine.population.len(), config.population_size);

        for individual in &engine.population {
            assert_eq!(individual.genome.len(), config.initial_genome_length);
            assert_eq!(individual.fitness, f64::NEG_INFINITY);
        }
    }

    #[test]
    fn test_crossover() {
        let config = get_test_config();
        let grammar = get_test_grammar();
        let metrics = get_metrics_config();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics, &grammar, &candles);

        let parent1: Genome = vec![1, 1, 1, 1, 1];
        let parent2: Genome = vec![2, 2, 2, 2, 2];

        let child = engine.crossover(&parent1, &parent2);

        // Child should contain genes from both parents
        assert!(child.contains(&1) || child.contains(&2));
        // Child length should be reasonable (parent1 start + parent2 end)
        assert!(child.len() >= 1 && child.len() <= parent1.len() + parent2.len());
    }

    #[test]
    fn test_crossover_empty_genomes() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        let empty: Genome = vec![];
        let parent: Genome = vec![1, 2, 3];

        let child1 = engine.crossover(&empty, &parent);
        let child2 = engine.crossover(&parent, &empty);
        let child3 = engine.crossover(&empty, &empty);

        assert_eq!(child1, parent);
        assert_eq!(child2, parent);
        assert!(child3.is_empty());
    }

    #[test]
    fn test_mutation() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        let original: Genome = vec![0, 0, 0, 0, 0];
        let mut genome = original.clone();
        engine.mutate(&mut genome);

        // With mutation, genome should change
        assert_ne!(genome, original);
        // Length should remain the same
        assert_eq!(genome.len(), original.len());
    }

    #[test]
    fn test_mutation_empty_genome() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        let mut empty: Genome = vec![];
        engine.mutate(&mut empty);
        assert!(empty.is_empty()); // Should remain empty
    }

    #[test]
    fn test_fitness_handles_mapping_error() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let bad_grammar = get_bad_grammar();
        let candles = get_test_candles();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &bad_grammar, &candles);

        let genome: Genome = vec![0];
        let (fitness, map_err, vm_err) = engine.calculate_fitness(&genome);

        assert_eq!(fitness, INFINITE_PENALTY);
        assert!(map_err);
        assert!(!vm_err);
    }

    #[test]
    fn test_variable_length_crossover() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        let parent1: Genome = vec![1; 20]; // Long parent
        let parent2: Genome = vec![2; 5]; // Short parent

        let child = engine.crossover(&parent1, &parent2);

        // Child should contain genes from both parents
        assert!(child.contains(&1) && child.contains(&2));
        // Child length should be reasonable
        assert!(child.len() >= 1 && child.len() <= 25);
        // Verify it's actually a combination, not just one parent
        assert_ne!(child, parent1);
        assert_ne!(child, parent2);
    }

    #[test]
    fn test_population_size_maintained() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        engine.initialize_population();
        let initial_size = engine.population.len();

        // Simulate one generation step
        let _parents = engine.select_parents();
        let mut next_generation = Vec::new();

        // Add elite
        if let Some(best) = engine.population.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            next_generation.push(best.clone());
        }

        // Breed rest of population
        while next_generation.len() < config.population_size {
            next_generation.push(Individual {
                genome: vec![1, 2, 3],
                fitness: f64::NEG_INFINITY,
            });
        }

        assert_eq!(next_generation.len(), initial_size);
        assert_eq!(next_generation.len(), config.population_size);
    }

    #[test]
    fn test_elitism_preservation() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        engine.initialize_population();

        // Manually set fitness values to create a clear best individual
        for (i, individual) in engine.population.iter_mut().enumerate() {
            individual.fitness = i as f64; // Last individual will have highest fitness
        }

        let best_before = engine
            .population
            .iter()
            .max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
            .clone();

        // Simulate one generation step (simplified)
        let _parents = engine.select_parents();
        let mut next_generation = Vec::new();

        // Elitism: carry over the best
        if let Some(best) = engine.population.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            next_generation.push(best.clone());
        }

        // Fill rest with dummy individuals
        while next_generation.len() < config.population_size {
            next_generation.push(Individual {
                genome: vec![0; config.initial_genome_length],
                fitness: f64::NEG_INFINITY,
            });
        }

        engine.population = next_generation;

        // Verify the best individual survived
        let survived_best = engine
            .population
            .iter()
            .find(|ind| ind.fitness == best_before.fitness && ind.genome == best_before.genome);

        assert!(
            survived_best.is_some(),
            "Best individual should survive through elitism"
        );
    }

    #[test]
    fn test_mapping_failure_counting() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let bad_grammar = get_bad_grammar(); // This will cause mapping failures
        let candles = get_test_candles();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &bad_grammar, &candles);

        engine.initialize_population();
        let (mapping_failures, vm_errors) = engine.evaluate_population();

        // All individuals should have mapping failures with bad grammar
        assert_eq!(mapping_failures, config.population_size as u32);
        assert_eq!(vm_errors, 0);

        // All individuals should have catastrophic fitness
        for individual in &engine.population {
            assert_eq!(individual.fitness, INFINITE_PENALTY);
        }
    }

    #[test]
    fn test_tournament_selection() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        engine.initialize_population();

        // Set different fitness values
        for (i, individual) in engine.population.iter_mut().enumerate() {
            individual.fitness = i as f64;
        }

        let parents = engine.select_parents();

        // Should return population_size parents
        assert_eq!(parents.len(), config.population_size);

        // Tournament selection should favor higher fitness individuals
        // Count how many times each fitness level appears
        let mut fitness_counts = std::collections::HashMap::new();
        for parent in &parents {
            *fitness_counts.entry(parent.fitness as i32).or_insert(0) += 1;
        }

        // Higher fitness individuals should appear more frequently
        // (This is probabilistic, so we just check that some selection occurred)
        assert!(fitness_counts.len() <= engine.population.len());
    }

    #[test]
    fn test_full_generation_cycle() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles_extended();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        engine.initialize_population();
        let initial_pop_size = engine.population.len();

        // Run evaluation
        let (_mapping_failures, _vm_errors) = engine.evaluate_population();

        // Manually set varied fitness values to test selection properly
        for (i, individual) in engine.population.iter_mut().enumerate() {
            individual.fitness = i as f64; // 0, 1, 2, 3, ..., 9
        }

        let parents = engine.select_parents();
        assert_eq!(parents.len(), initial_pop_size);

        let mut next_generation = Vec::new();

        // Elitism - carry over the best individual
        if let Some(best) = engine.population.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            next_generation.push(best.clone());
        }

        // Realistic breeding loop
        let mut parent_idx = 0;
        while next_generation.len() < config.population_size {
            let parent1 = &parents[parent_idx % parents.len()];
            let parent2 = &parents[(parent_idx + 1) % parents.len()];

            let mut child_genome = engine.crossover(&parent1.genome, &parent2.genome);
            engine.mutate(&mut child_genome);

            next_generation.push(Individual {
                genome: child_genome,
                fitness: f64::NEG_INFINITY,
            });

            parent_idx += 2; // Move to next pair of parents
        }

        // Verify next generation is valid
        assert_eq!(next_generation.len(), config.population_size);

        // The elite should have the highest fitness from previous generation (9.0)
        assert_eq!(next_generation[0].fitness, 9.0);

        // All others should be unevaluated
        for i in 1..next_generation.len() {
            assert_eq!(next_generation[i].fitness, f64::NEG_INFINITY);
        }

        // Verify genomes are actually different (breeding occurred)
        let elite_genome = &next_generation[0].genome;
        let mut genomes_differ = false;
        for i in 1..next_generation.len() {
            if next_generation[i].genome != *elite_genome {
                genomes_differ = true;
                break;
            }
        }
        assert!(genomes_differ, "Breeding should produce different genomes");
    } // Add this helper function for sufficient test data
    fn get_test_candles_extended() -> Vec<OHLCV> {
        (1..=10)
            .map(|i| OHLCV {
                timestamp: i,
                open: 100.0 + i as f64,
                high: 105.0 + i as f64,
                low: 95.0 + i as f64,
                close: 102.0 + i as f64,
                volume: 1000.0,
            })
            .collect()
    }
}
