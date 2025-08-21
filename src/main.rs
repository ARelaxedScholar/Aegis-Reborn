use aegis_reborn::config::{Config, DataConfig};
use aegis_reborn::data::{OHLCV, load_csv};
use aegis_reborn::evolution::EvolutionEngine;
use aegis_reborn::evolution::grammar::Grammar;
use aegis_reborn::evolution::mapper::GrammarBasedMapper;
use std::path::Path;
use std::process;

/// Loads OHLCV data from a CSV file and splits it into training and hold-out sets.
///
/// # Arguments
/// * `data_config` - Reference to a `DataConfig` struct containing the file path and hold-out split ratio.
///
/// # Returns
/// * `Ok((Vec<OHLCV>, Vec<OHLCV>))` - Tuple containing the training data and hold-out data as vectors.
/// * `Err(String)` - Error message if loading or partitioning fails.
///
/// # Errors
/// Returns an error if:
/// - The data file cannot be loaded.
/// - The data file contains no candle data.
/// - There is not enough data for a meaningful train/hold-out split.
fn prepare_data(data_config: &DataConfig) -> Result<(Vec<OHLCV>, Vec<OHLCV>), String> {
    log::info!("Loading data from '{}'...", data_config.file_path);
    let all_candles = load_csv(Path::new(&data_config.file_path))
        .map_err(|e| format!("Failed to load data: {}", e))?;

    if all_candles.is_empty() {
        return Err("Data file contains no candle data.".to_string());
    }

    let split_index =
        (all_candles.len() as f64 * (1.0 - data_config.hold_out_split)).floor() as usize;

    // Sanity check the split indices
    if split_index < 100 || all_candles.len() - split_index < 10 {
        return Err("Not enough data for a meaningful train/hold-out split.".to_string());
    }

    let (training_data, hold_out_data) = all_candles.split_at(split_index);
    log::info!(
        "Data partitioned: {} candles for Training/Validation, {} for Hold-Out.",
        training_data.len(),
        hold_out_data.len()
    );
    Ok((training_data.to_vec(), hold_out_data.to_vec()))
}

fn main() {
    env_logger::init();
    log::info!("Booting Aegis Reborn...");

    // 1. Load and Validate Configuration
    let config = match Config::load(Path::new("config.toml")) {
        Ok(c) => c,
        Err(e) => {
            log::error!("Failed to load configuration: {}", e);
            process::exit(1);
        }
    };

    if let Err(e) = config.validate() {
        log::error!("Invalid configuration: {}", e);
        process::exit(1);
    }
    log::info!("Configuration loaded and validated.");

    // 2. Load Grammar
    let grammar = match Grammar::new(Path::new(&config.grammar_file)) {
        Ok(g) => g,
        Err(e) => {
            log::error!("Failed to load grammar: {}", e);
            process::exit(1);
        }
    };
    log::info!("Grammar '{}' loaded successfully.", config.grammar_file);

    // 3. Prepare Data
    let (training_validation_data, hold_out_data) = match prepare_data(&config.data) {
        Ok(data) => data,
        Err(e) => {
            log::error!("Data preparation failed: {}", e);
            process::exit(1);
        }
    };

    // 4. Run the Evolution ONLY on the Training/Validation Set
    log::info!("--- Starting Evolution ---");
    let mut engine = EvolutionEngine::new(
        &config.ga,
        &config.metrics,
        &grammar,
        &training_validation_data,
    );
    let champions = engine.evolve();

    // 5. Final Gauntlet (To be implemented in Step 5)
    log::info!("--- Evolution Complete: Preparing for Final Gauntlet ---");
    log::info!("Top 5 Champions (Council):");
    let mapper = GrammarBasedMapper::new(
        &grammar,
        config.ga.max_program_tokens,
        config.ga.max_recursion_depth,
    );
    for (i, champion) in champions.iter().take(5).enumerate() {
        println!("\n[Rank {}] Fitness: {:.4}", i + 1, champion.fitness);
        match mapper.map(&champion.genome) {
            Ok(strategy) => println!("{:#?}", strategy),
            Err(e) => println!("  - Failed to map champion genome: {:?}", e),
        }
    }
    log::info!(
        "Next step: Run Council of Champions through Block Bootstrapping and on the Hold-Out set of {} candles.",
        hold_out_data.len()
    );
    // run_gauntlet(
    //    &champions,
    //    training_validation_data,
    //    &hold_out_data,
    //    &config.metrics, // Pass the metrics config
    //);
}
