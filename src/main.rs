use chrono::Utc;
use clap::Parser;
use golden_aegis::config::{Config, DataConfig, TranspilerConfig};
use golden_aegis::data::{load_csv, OHLCV};
use golden_aegis::evaluation::gauntlet::{run_gauntlet, write_reports_to_json};
use golden_aegis::evolution::grammar::Grammar;
use golden_aegis::evolution::mapper::GrammarBasedMapper;
use golden_aegis::evolution::EvolutionEngine;
use golden_aegis::export::{
    read_export_from_json, write_export_to_json, ChampionExport, ExportConfig,
};
use golden_aegis::strategy::Strategy;
use golden_aegis::transpiler::TranspilerEngine;
use std::fs;
use std::path::Path;
use std::process;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, subcommand_required = false)]
enum Args {
    /// Run the evolution process
    Run {
        /// Path to the configuration file
        #[arg(short, long, default_value = "config.toml")]
        config: String,

        /// Export council champions with full metadata for post-hoc transpilation
        #[arg(long, default_value = "false")]
        export_champions: bool,
    },

    /// Transpile exported champions to target platform
    Transpile {
        /// Path to the champion export JSON file
        #[arg(short, long)]
        export_file: String,

        /// Output directory for transpiled algorithms
        #[arg(short, long, default_value = "transpiled")]
        output_dir: String,

        /// Target platform (only 'quantconnect' supported currently)
        #[arg(short, long, default_value = "quantconnect")]
        target: String,

        /// Language for QuantConnect algorithms (python or csharp)
        #[arg(short, long, default_value = "python")]
        language: String,

        /// Override symbol from export configuration
        #[arg(long)]
        symbol: Option<String>,

        /// Override resolution from export configuration
        #[arg(long)]
        resolution: Option<String>,

        /// Override market from export configuration
        #[arg(long)]
        market: Option<String>,
    },
}

impl Default for Args {
    fn default() -> Self {
        Self::Run {
            config: "config.toml".to_string(),
            export_champions: false,
        }
    }
}

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

fn run_evolution(
    config_path: &str,
    export_champions: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Booting Golden Aegis...");

    // 1. Load and Validate Configuration
    let config = Config::load(Path::new(config_path)).unwrap_or_else(|e| {
        log::error!("Failed to load configuration: {}", e);
        process::exit(1);
    });

    config.validate().unwrap_or_else(|e| {
        log::error!("Invalid configuration: {}", e);
        process::exit(1);
    });
    log::info!("Configuration loaded and validated.");

    // Prepare export configuration
    let export_config = ExportConfig {
        ga: config.ga,
        metrics: config.metrics,
        data: config.data.clone(),
    };
    let transpiler_config = config.get_transpiler_config();

    // 2. Load Grammar
    let grammar_content = fs::read_to_string(&config.grammar_file).unwrap_or_else(|e| {
        log::error!("Failed to read grammar file: {}", e);
        process::exit(1);
    });
    let grammar = Grammar::new(Path::new(&config.grammar_file)).unwrap_or_else(|e| {
        log::error!("Failed to load grammar: {}", e);
        process::exit(1);
    });
    log::info!("Grammar '{}' loaded successfully.", config.grammar_file);

    // 3. Prepare Data
    let (training_validation_data, hold_out_data) =
        prepare_data(&config.data).unwrap_or_else(|e| {
            log::error!("Data preparation failed: {}", e);
            process::exit(1);
        });

    // 4. Run the Evolution ONLY on the Training/Validation Set
    log::info!("--- Starting Evolution ---");
    let mut engine = EvolutionEngine::new(
        &config.ga,
        &config.metrics,
        &grammar,
        &training_validation_data,
    );
    let champions = engine.evolve();

    // 5. Prepare council champions for final gauntlet and export
    let size_of_council = config.ga.size_of_council;
    log::info!("--- Evolution Complete: Preparing for Final Gauntlet ---");
    log::info!("Top {size_of_council} Champions (Council):");
    let mapper = GrammarBasedMapper::new(
        &grammar,
        config.ga.max_program_tokens,
        config.ga.max_recursion_depth,
    );

    // Map council champions to strategies
    let mut council_strategies = Vec::with_capacity(size_of_council);
    let council_champions: Vec<_> = champions.iter().take(size_of_council).cloned().collect();

    for (i, champion) in council_champions.iter().enumerate() {
        println!("\n[Rank {}] Fitness: {:.4}", i + 1, champion.fitness);
        match mapper.map(&champion.genome) {
            Ok(strategy) => {
                println!("{:#?}", strategy);
                council_strategies.push(strategy);
            }
            Err(e) => {
                println!("  - Failed to map champion genome: {:?}", e);
                // Push placeholder strategy for export consistency
                council_strategies.push(Strategy::new());
            }
        }
    }

    // Export champions if requested
    if export_champions {
        log::info!("Exporting council champions with full metadata...");
        let mut export = ChampionExport::new(
            council_champions.clone(),
            export_config,
            grammar_content.clone(),
            Some(transpiler_config.clone()),
        );
        export.update_strategies(council_strategies);

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let export_filename = format!("champions_export_{}.json", timestamp);
        let export_path = Path::new(&export_filename);

        match write_export_to_json(&export, export_path) {
            Ok(_) => log::info!("Champion export written to {}", export_filename),
            Err(e) => log::error!("Failed to write champion export: {}", e),
        }
    }
    log::info!(
        "Next step: Run Council of Champions through Block Bootstrapping and on the Hold-Out set of {} candles.",
        hold_out_data.len()
    );

    let final_reports = run_gauntlet(
        &council_champions,
        &training_validation_data,
        &hold_out_data,
        &grammar,
        &config.ga,
        &config.metrics,
    )?;

    write_reports_to_json(&final_reports)?;

    log::info!("--- Golden Aegis Run Complete ---");
    Ok(())
}

fn transpile_champions(
    export_file: &str,
    output_dir: &str,
    target: &str,
    language: &str,
    symbol: Option<&str>,
    resolution: Option<&str>,
    market: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!(
        "Transpiling champions from {} to {} (target: {}, language: {})",
        export_file,
        output_dir,
        target,
        language
    );

    // Validate target
    if target != "quantconnect" {
        return Err("Only 'quantconnect' target is currently supported".into());
    }

    // Load champion export
    let export = read_export_from_json(Path::new(export_file))?;
    log::info!(
        "Loaded export with {} champions (schema version {})",
        export.champions.len(),
        export.schema_version
    );

    // Determine transpiler configuration
    let mut transpiler_config = export.transpiler_config.unwrap_or_else(|| {
        // Create default config using metrics from export
        TranspilerConfig {
            symbol: "SPY".to_string(),
            resolution: "Daily".to_string(),
            market: "usa".to_string(),
            initial_cash: Some(export.evolution_config.metrics.initial_cash),
            transaction_cost_pct: Some(export.evolution_config.metrics.transaction_cost_pct),
            slippage_pct: Some(export.evolution_config.metrics.slippage_pct),
        }
    });

    // Apply overrides from CLI
    if let Some(sym) = symbol {
        transpiler_config.symbol = sym.to_string();
    }
    if let Some(res) = resolution {
        transpiler_config.resolution = res.to_string();
    }
    if let Some(mkt) = market {
        transpiler_config.market = mkt.to_string();
    }

    // Create transpiler engine
    let engine = TranspilerEngine::new(transpiler_config);

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Transpile each champion
    for champion in &export.champions {
        let rank = champion.rank;
        let filename = match language.to_lowercase().as_str() {
            "python" => format!("champion_{}.py", rank),
            "csharp" => format!("champion_{}.cs", rank),
            _ => return Err(format!("Unsupported language: {}", language).into()),
        };
        let output_path = Path::new(output_dir).join(filename);

        log::debug!("Transpiling champion rank {}...", rank);
        let code = match language.to_lowercase().as_str() {
            "python" => engine.to_python(&champion.strategy)?,
            "csharp" => engine.to_c_sharp(&champion.strategy)?,
            _ => unreachable!(),
        };

        std::fs::write(&output_path, &code)?;
        log::info!(
            "Wrote {} ({} lines)",
            output_path.display(),
            code.lines().count()
        );
    }

    log::info!(
        "Successfully transpiled {} champions to {}",
        export.champions.len(),
        output_dir
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    match args {
        Args::Run {
            config,
            export_champions,
        } => run_evolution(&config, export_champions),
        Args::Transpile {
            export_file,
            output_dir,
            target,
            language,
            symbol,
            resolution,
            market,
        } => transpile_champions(
            &export_file,
            &output_dir,
            &target,
            &language,
            symbol.as_deref(),
            resolution.as_deref(),
            market.as_deref(),
        ),
    }
}
