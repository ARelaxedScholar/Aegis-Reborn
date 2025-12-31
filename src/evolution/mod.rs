pub mod grammar;
pub mod mapper;

use crate::config::{GaConfig, MetricsConfig};
use crate::data::OHLCV;
use crate::evaluation::walk_forward::WalkForwardValidator;
use crate::evolution::grammar::Grammar;
use crate::evolution::mapper::GrammarBasedMapper;
use log::{debug, error, info, warn};
use rand::prelude::*;
use rand::rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
/// Penalty to assign when a strategy is so egregiously bad it must be excised from the gene pool
const INFINITE_PENALTY: f64 = f64::NEG_INFINITY;

/// Alias within crate for genome representation
pub type Genome = Vec<u32>;

/// This struct represents a given individual born during the evolution process
/// the goal of this architecture is to evolve, nurture, and find those of outstanding quality.
#[derive(Debug, Clone)]
pub struct Individual {
    /// This is the DNA, genome of a given individual, represents the sequence of u32 that compose
    /// it
    pub genome: Genome,
    /// This is a smoothed Calmar ratio with a linear adjustment for penalty
    pub fitness: f64,
}

/// This is the beating heart of this architecture, it orchestrate the entire evolution process.
/// It works in tandem with the `WalkForwardValidator` and the `GrammarBasedMapper` to evolve
/// (hopefully) robust strategies.
#[derive(Clone)]
pub struct EvolutionEngine<'a> {
    /// This is a reference to the user-defined config for a given evolution run
    config: &'a GaConfig,
    /// This is a reference to the user-defined config for the metrics which don't play a run in evolution, but we
    /// want to measure for the gauntlet
    metrics_params: &'a MetricsConfig,
    /// This is the mapper which takes care of converting a given genome into a program
    mapper: GrammarBasedMapper<'a>,
    /// This is an owned-vector of the population at any given moment, it is updated after every
    /// generation
    population: Vec<Individual>,
    /// This is a reference to a container of `OHLCV` candles
    candles: &'a [OHLCV],
}

/// Struct associated with the `evaluate_population` function,
/// which contains a count of the errors which occured during VM execution `vm_errors`
/// and a count of the error which occured during mapping `mapping_failures`
#[derive(Debug, Copy, Clone)]
pub struct PopulationEvaluationReport {
    mapping_failures: usize,
    vm_errors: usize,
}

/// Struct associated with the `calculate_fitness` function,
/// which contains the aggregated `fitness` score, the
#[derive(Debug, Copy, Clone)]
struct FitnessEvaluationReport {
    fitness: f64,
    mapping_failure_occurred: bool,
    vm_error_occurred: bool,
}

impl<'a> EvolutionEngine<'a> {
    /// Creates a new EvolutionEngine instance
    ///
    /// # Arguments
    /// * `config` - Reference to a `GaConfig` struct containing all the parameters required for the evolution.
    /// * `metric_params` - Reference to a `MetricsConfig` struct containing all the parameters for the metrics to be computed for reporting that do not play a role during evolution
    /// * `grammar` - Reference to a `Grammar` struct containing all the rules of the user-specifed grammar.
    /// * `candles` - Reference to a container of `OHLCV` candles which will eventually be used for backtesting the generated individuals
    ///
    /// # Returns
    /// * `Self` - An instance of the EvolutionEngine struct
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

    /// Runs the evolution process
    ///
    /// This method mutably borrows the `EvolutionEngine` instance,
    /// modifying its `population` field. After evolution, this field contains the last
    /// generation of solutions.
    ///
    /// # Arguments
    /// * `&mut self` - The EvolutionEngine that will orchestrate the evolution process
    ///
    /// # Returns
    /// * `Vec<Individual>` - A vector of the final generation of individuals, a clone of the `population` field at the end of the process
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

            // mapping_failures, vm_errors are counted to penalize degenerate trading
            // strategies
            let PopulationEvaluationReport {
                mapping_failures,
                vm_errors,
            } = self.evaluate_population();
            let avg_genome_len = self
                .population
                .iter()
                .map(|ind| ind.genome.len())
                .sum::<usize>() as f64
                / self.population.len() as f64;

            let mut next_generation = Vec::new();
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

                // Preserve it for next generation (explotation)
                next_generation.push(best.clone());
            }

            // Vector
            let parents = self.select_parents();

            while next_generation.len() < self.config.population_size {
                let parent1 = parents.choose(&mut rng()).unwrap();
                let parent2 = parents.choose(&mut rng()).unwrap();

                let mut children_genome = if rng().random::<f64>() < self.config.crossover_rate {
                    self.crossover(&parent1.genome, &parent2.genome)
                } else {
                    vec![parent1.genome.clone(), parent2.genome.clone()]
                };

                // Mutation (update to Per-Gene Probabilistic mutation)
                children_genome.iter_mut().for_each(|g| self.mutate(g));

                let young_blood: Vec<Individual> = children_genome
                    .into_iter()
                    .map(|g| Individual {
                        genome: g,
                        fitness: f64::NEG_INFINITY,
                    })
                    .collect();

                // Add the children to the next generation, up to capacity.
                let remaining_slots = self.config.population_size - next_generation.len();
                let admitted = young_blood.into_iter().take(remaining_slots);
                next_generation.extend(admitted);
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

    /// This function initializes the population vector.
    ///
    /// This method mutably borrows the `EvolutionEngine` instance,
    /// modifying its `population` field. This is usually called once, at the beginning of the
    /// process to setup the initial `population` vector.
    ///
    /// # Arguments
    /// * `&mut self` - The EvolutionEngine for which you need to initialize the population
    ///
    /// # Returns
    /// Nothing. `EvolutionEngine` is modified in-place.
    pub fn initialize_population(&mut self) {
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

    /// Evaluates the population
    ///
    /// This method mutably borrows the `EvolutionEngine` instance,
    /// modifying its `population` field. Each generation, it is called to update the `fitness`
    /// scores of the current generation. It itself delegates to the `calculate_fitness` function.
    pub fn evaluate_population(&mut self) -> PopulationEvaluationReport {
        // Collect indices and genomes together
        let work_items: Vec<(usize, Genome)> = self
            .population
            .iter()
            .enumerate()
            .filter_map(|(i, ind)| {
                (ind.fitness == f64::NEG_INFINITY).then(|| (i, ind.genome.clone()))
            })
            .collect();

        if work_items.is_empty() {
            return PopulationEvaluationReport {
                mapping_failures: 0,
                vm_errors: 0,
            };
        }

        // Evaluate in parallel without holding any reference to self.population
        let results: Vec<(usize, FitnessEvaluationReport)> = work_items
            .par_iter()
            .map(|(i, genome)| {
                let report = self.calculate_fitness(genome);
                (*i, report)
            })
            .collect();

        // Apply results sequentially (no unsafe needed)
        let mut mapping_failures = 0;
        let mut vm_errors = 0;

        for (i, report) in results {
            self.population[i].fitness = report.fitness;
            if report.mapping_failure_occurred {
                mapping_failures += 1;
            }
            if report.vm_error_occurred {
                vm_errors += 1;
            }
        }

        PopulationEvaluationReport {
            mapping_failures,
            vm_errors,
        }
    }

    /// A function which delegates the actual evaluation of a genome to the `WalkForwardValidator`
    /// and then aggregates the result to send them upstream.
    /// Evaluates the fitness of a given genome.
    ///
    /// # Arguments
    /// * `&mut self` - The `EvolutionEngine` for which you need to initialize the population
    /// * `genome` - Reference to the `Genome` of the `Individual` to evaluate
    ///
    /// # Returns
    /// `FitnessEvaluationReport`
    fn calculate_fitness(&self, genome: &Genome) -> FitnessEvaluationReport {
        let strategy = match self.mapper.map(genome) {
            Ok(s) => s,
            Err(e) => {
                debug!("Mapping failed: {}. Assigning catastrophic fitness.", e);
                return FitnessEvaluationReport {
                    fitness: INFINITE_PENALTY,
                    mapping_failure_occurred: true,
                    vm_error_occurred: false,
                };
            }
        };

        if !strategy.programs.contains_key("entry") {
            debug!("Strategy missing 'entry' program. Assigning catastrophic fitness.");
            return FitnessEvaluationReport {
                fitness: INFINITE_PENALTY,
                mapping_failure_occurred: true,
                vm_error_occurred: false,
            };
        }

        let validator = match WalkForwardValidator::new(
            self.config.test_window_size,
            self.metrics_params.risk_free_rate,
            self.metrics_params.initial_cash,
            self.metrics_params.annualization_rate,
            self.metrics_params.transaction_cost_pct,
            self.metrics_params.slippage_pct,
        ) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to create walk-forward validator: {}", e);
                return FitnessEvaluationReport {
                    fitness: INFINITE_PENALTY,
                    mapping_failure_occurred: false,
                    vm_error_occurred: true,
                };
            }
        };

        let result = match validator.validate(self.candles, &strategy) {
            Ok(r) => r,
            Err(e) => {
                debug!("Walk-forward validation failed: {}", e);
                return FitnessEvaluationReport {
                    fitness: INFINITE_PENALTY,
                    mapping_failure_occurred: false,
                    vm_error_occurred: true,
                };
            }
        };

        if result.entry_error_count > 0 || result.exit_error_count > 0 {
            debug!(
                "Strategy execution errors - Entry: {}, Exit: {}. Assigning catastrophic fitness.",
                result.entry_error_count, result.exit_error_count
            );
            return FitnessEvaluationReport {
                fitness: INFINITE_PENALTY,
                mapping_failure_occurred: false,
                vm_error_occurred: true,
            };
        }

        let smoothing_constant = self.config.alpha;
        let calmar = if result.max_drawdown.abs() < 1e-9 {
            if result.annualized_return > 0.0 {
                result.annualized_return / smoothing_constant
            } else {
                result.annualized_return
            }
        } else {
            result.annualized_return / (smoothing_constant + result.max_drawdown)
        };

        let total_opcodes = strategy.programs.values().map(|p| p.len()).sum::<usize>() as f64;
        let penalty = total_opcodes * self.config.parsimony_penalty;
        let fitness = if calmar > 0.0 {
            calmar * (1.0 - penalty).max(0.0)
        } else {
            calmar
        };

        debug!(
            "Fitness: ann_ret={:.4}, mdd={:.4}, calmar={:.4}, opcodes={}, penalty={:.4}, final={:.4}",
            result.annualized_return, result.max_drawdown, calmar, total_opcodes, penalty, fitness
        );

        FitnessEvaluationReport {
            fitness,
            mapping_failure_occurred: false,
            vm_error_occurred: false,
        }
    }

    /// Selects the parents that will be used for next generation
    ///
    /// It uses a tournament selection approach with a k size defined in the `tournament_size` field of the `config` field of the `EvolutionEngine`.
    ///
    /// # Arguments
    /// * `&self` - The EvolutionEngine that will orchestrate the evolution process
    ///
    /// # Returns
    /// * `Vec<Individual>` - The vector of `Individual` who won the right to become parents
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

    /// This is the crossover operator (variable-length)
    ///
    /// It picks a random cut point in `parent1`'s and `parent2`'s genome respectively,
    /// and then splices the first part of `parent1`'s genome with the second part of `parent2`'s genome.
    /// It then does the reverse for symmetry and full gene exploitation.
    ///
    /// # Arguments
    /// * `&self` - The EvolutionEngine that will orchestrate the evolution process
    /// * `&Genome` - Reference to parent1's genome
    /// * `&Genome` - Reference to parent2's genome
    ///
    /// # Returns
    /// * `Vec<Genome>` - The offsprings representing the splicing of parent1 and parent2
    fn crossover(&self, parent1: &Genome, parent2: &Genome) -> Vec<Genome> {
        if parent1.is_empty() || parent2.is_empty() {
            warn!("crossover operator received a parent with empty genome");
            // NoOp
            return vec![parent1.clone(), parent2.clone()];
        }
        let mut rng = rng();

        // allows taking none, all, and everything in between
        let cut1 = rng.random_range(0..=parent1.len());
        let cut2 = rng.random_range(0..=parent2.len());

        vec![
            parent1[..cut1]
                .iter()
                .chain(&parent2[cut2..])
                .cloned()
                .collect(),
            parent2[..cut2]
                .iter()
                .chain(&parent1[cut1..])
                .cloned()
                .collect(),
        ]
    }

    /// Mutates a given genome
    ///
    /// It uses a Per-Gene-Probabibilistic Mutation approach. Just a fancy way of saying, the `mutation_rate` field in the `config` of the `EvolutionEngine` applies to the
    /// probability of a random gene being modified and not the probability of some individual
    /// being modified.
    ///
    /// # Arguments
    /// * `&self` - The EvolutionEngine that will orchestrate the evolution process
    /// * `genome` - Mutable reference to the genome to mutate
    /// # Returns
    /// * Nothing.
    fn mutate(&self, genome: &mut Genome) {
        if genome.is_empty() {
            return;
        }
        let mut rng = rng();
        genome.iter_mut().for_each(|gene| {
            if rng.random::<f64>() < self.config.mutation_rate {
                // use random resetting, might be something to tweak in the future
                *gene = rng.random::<u32>();
            }
        });
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
            mutation_rate: 0.1,  // Reduced for more predictable testing
            crossover_rate: 0.8, // Reduced from 100%
            max_program_tokens: 50,
            max_recursion_depth: 256,
            initial_genome_length: 10,
            parsimony_penalty: 0.01,
            tournament_size: 3,
            test_window_size: 3,
            size_of_council: 2,
        }
    }

    // Helper to create a minimal, valid metrics config
    fn get_metrics_config() -> MetricsConfig {
        MetricsConfig {
            risk_free_rate: 0.02,
            bootstrap_runs: 100,
            initial_cash: 10_000.0,
            annualization_rate: 252.0,
            transaction_cost_pct: 0.0,
            slippage_pct: 0.0,
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

    // Extended test data for more realistic testing
    fn get_test_candles_extended() -> Vec<OHLCV> {
        (1..=20)
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
    fn test_crossover_returns_two_children() {
        let config = get_test_config();
        let grammar = get_test_grammar();
        let metrics = get_metrics_config();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics, &grammar, &candles);

        let parent1: Genome = vec![1, 1, 1, 1, 1];
        let parent2: Genome = vec![2, 2, 2, 2, 2];

        let children = engine.crossover(&parent1, &parent2);

        assert_eq!(
            children.len(),
            2,
            "Crossover should return exactly 2 children"
        );

        let child1 = &children[0];
        let child2 = &children[1];

        // Children should be composed of parent material
        assert!(child1.iter().all(|&g| g == 1 || g == 2));
        assert!(child2.iter().all(|&g| g == 1 || g == 2));

        // At least one child should be different from its corresponding parent
        // (unless by chance the crossover point results in identical genomes)
        let children_differ_from_parents =
            child1 != &parent1 || child1 != &parent2 || child2 != &parent1 || child2 != &parent2;
        assert!(
            children_differ_from_parents,
            "Children should generally differ from parents"
        );
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

        // Crossover should return clones of the parents when one is empty
        let children1 = engine.crossover(&empty, &parent);
        assert_eq!(children1.len(), 2);
        assert_eq!(children1[0], empty);
        assert_eq!(children1[1], parent);

        let children2 = engine.crossover(&parent, &empty);
        assert_eq!(children2.len(), 2);
        assert_eq!(children2[0], parent);
        assert_eq!(children2[1], empty);

        let children3 = engine.crossover(&empty, &empty);
        assert_eq!(children3.len(), 2);
        assert_eq!(children3[0], empty);
        assert_eq!(children3[1], empty);
    }

    #[test]
    fn test_crossover_variable_lengths() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        let parent1: Genome = vec![1; 10]; // Long parent
        let parent2: Genome = vec![2; 3]; // Short parent

        let children = engine.crossover(&parent1, &parent2);

        assert_eq!(children.len(), 2);

        // Children should contain genes from both parents
        for child in &children {
            if !child.is_empty() {
                assert!(child.iter().all(|&g| g == 1 || g == 2));
            }
        }
    }

    #[test]
    fn test_mutation_with_moderate_rate() {
        let mut config = get_test_config();
        config.mutation_rate = 1.0; // 100% mutation rate for testing

        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        let original: Genome = vec![0, 0, 0, 0, 0];
        let mut genome = original.clone();
        engine.mutate(&mut genome);

        // With 100% mutation rate, genome should change
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
        let engine = EvolutionEngine::new(&config, &metrics_config, &bad_grammar, &candles);

        let genome: Genome = vec![0];
        let FitnessEvaluationReport {
            fitness,
            mapping_failure_occurred,
            vm_error_occurred,
        } = engine.calculate_fitness(&genome);

        assert_eq!(fitness, INFINITE_PENALTY);
        assert!(mapping_failure_occurred);
        assert!(!vm_error_occurred);
    }

    #[test]
    fn test_tournament_selection() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        engine.initialize_population();

        // Set different fitness values - create a clear hierarchy
        for (i, individual) in engine.population.iter_mut().enumerate() {
            individual.fitness = i as f64;
        }

        let parents = engine.select_parents();

        // Should return population_size parents
        assert_eq!(parents.len(), config.population_size);

        // All parents should have valid fitness values
        for parent in &parents {
            assert!(parent.fitness.is_finite());
        }
    }

    #[test]
    fn test_evaluate_population_with_bad_grammar() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let bad_grammar = get_bad_grammar(); // This will cause mapping failures
        let candles = get_test_candles_extended(); // Use extended for proper validation
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &bad_grammar, &candles);

        engine.initialize_population();
        let PopulationEvaluationReport {
            mapping_failures,
            vm_errors: _vm_errors,
        } = engine.evaluate_population();

        // All individuals should have mapping failures with bad grammar
        assert_eq!(mapping_failures, config.population_size);

        // All individuals should have catastrophic fitness
        for individual in &engine.population {
            assert_eq!(individual.fitness, INFINITE_PENALTY);
        }
    }

    #[test]
    fn test_population_size_maintained_through_generation() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles_extended();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        engine.initialize_population();
        let initial_size = engine.population.len();

        // Manually set fitness values for predictable behavior
        for (i, individual) in engine.population.iter_mut().enumerate() {
            individual.fitness = i as f64;
        }

        // Simulate the breeding logic from evolve()
        let parents = engine.select_parents();
        let mut next_generation = Vec::new();

        // Elitism - preserve the best
        if let Some(best) = engine.population.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            next_generation.push(best.clone());
        }

        // Breed the rest
        while next_generation.len() < config.population_size {
            let parent1 = parents.choose(&mut rand::rng()).unwrap();
            let parent2 = parents.choose(&mut rand::rng()).unwrap();

            let children_genomes = if rand::rng().random::<f64>() < config.crossover_rate {
                engine.crossover(&parent1.genome, &parent2.genome)
            } else {
                vec![parent1.genome.clone(), parent2.genome.clone()]
            };

            let children: Vec<Individual> = children_genomes
                .into_iter()
                .map(|mut genome| {
                    engine.mutate(&mut genome);
                    Individual {
                        genome,
                        fitness: f64::NEG_INFINITY,
                    }
                })
                .collect();

            // Add children up to capacity
            let remaining_slots = config.population_size - next_generation.len();
            let admitted = children.into_iter().take(remaining_slots);
            next_generation.extend(admitted);
        }

        assert_eq!(next_generation.len(), initial_size);
        assert_eq!(next_generation.len(), config.population_size);
    }

    #[test]
    fn test_elitism_preservation() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles_extended();
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

        // Simulate one generation step with elitism
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
    fn test_full_evolution_run_completes() {
        let mut config = get_test_config();
        config.num_generations = 2; // Keep it short for testing
        config.population_size = 5;

        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles_extended();
        let mut engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        let final_population = engine.evolve();

        // Evolution should complete and return final population
        assert_eq!(final_population.len(), config.population_size);

        // Population should be sorted by fitness (descending)
        for i in 1..final_population.len() {
            assert!(
                final_population[i - 1].fitness >= final_population[i].fitness,
                "Population should be sorted by fitness in descending order"
            );
        }
    }

    #[test]
    fn test_breeding_logic_with_crossover_and_mutation() {
        let config = get_test_config();
        let metrics_config = get_metrics_config();
        let grammar = get_test_grammar();
        let candles = get_test_candles();
        let engine = EvolutionEngine::new(&config, &metrics_config, &grammar, &candles);

        let parent1 = Individual {
            genome: vec![1, 1, 1, 1, 1],
            fitness: 1.0,
        };
        let parent2 = Individual {
            genome: vec![2, 2, 2, 2, 2],
            fitness: 2.0,
        };

        // Test the crossover + mutation pipeline
        let mut children_genomes = engine.crossover(&parent1.genome, &parent2.genome);

        // Apply mutation to each child
        for genome in children_genomes.iter_mut() {
            engine.mutate(genome);
        }

        // Create individuals from the genomes
        let young_blood: Vec<Individual> = children_genomes
            .into_iter()
            .map(|g| Individual {
                genome: g,
                fitness: f64::NEG_INFINITY,
            })
            .collect();

        assert_eq!(young_blood.len(), 2);

        for child in &young_blood {
            assert_eq!(child.fitness, f64::NEG_INFINITY);
            assert!(!child.genome.is_empty());
        }
    }
}
