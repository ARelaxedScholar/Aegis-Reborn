use crate::config::{GaConfig, MetricsConfig};
use crate::data::OHLCV;
use crate::evaluation::backtester::{Backtester, BacktestResult, INITIAL_CASH};
use crate::evolution::{grammar::Grammar, mapper::GrammarBasedMapper, Individual};
use log::{info, warn, error, debug};
use rand::prelude::IndexedRandom;
use rand::rng;
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::fs::File;
use serde_json;
use serde::Serialize;
use std::io::Write;

/// Custom error types for gauntlet validation
#[derive(Debug, Clone)]
pub enum GauntletError {
    InsufficientData(String),
    MappingFailed(String),
    BootstrapFailed(String),
    CalculationError(String),
}

impl fmt::Display for GauntletError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GauntletError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            GauntletError::MappingFailed(msg) => write!(f, "Strategy mapping failed: {}", msg),
            GauntletError::BootstrapFailed(msg) => write!(f, "Bootstrap failed: {}", msg),
            GauntletError::CalculationError(msg) => write!(f, "Calculation error: {}", msg),
        }
    }
}

impl Error for GauntletError {}

/// Comprehensive report for a single champion's gauntlet performance
#[derive(Debug, Serialize)]
pub struct GauntletReport {
    pub rank: usize,
    pub original_fitness: f64,
    pub hold_out_result: BacktestResult,
    pub bootstrap_stats: BootstrapStats,
}

#[derive(Debug, Serialize)]
pub struct BootstrapStats {
    pub avg_equity: f64,
    pub median_equity: f64,
    pub worst_equity: f64,
    pub best_equity: f64,
    pub profitable_percentage: f64,
    pub confidence_interval_95: (f64, f64),
    pub volatility: f64,
}

/// Block bootstrap configuration
#[derive(Debug)]
struct BootstrapConfig {
    block_size: usize,
    overlap_ratio: f64,
    min_blocks_required: usize,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            block_size: 21,        // ~1 month for daily data
            overlap_ratio: 0.5,    // 50% overlap between blocks
            min_blocks_required: 10, // Minimum blocks needed for valid bootstrap
        }
    }
}

/// Production-ready gauntlet runner with comprehensive validation and error handling
pub fn run_gauntlet(
    council_of_champions: &[Individual],
    training_data: &[OHLCV],
    hold_out_data: &[OHLCV],
    grammar: &Grammar,
    ga_config: &GaConfig,
    metrics_config: &MetricsConfig,
) -> Result<Vec<GauntletReport>, GauntletError> {
    info!("--- Commencing Final Gauntlet on Top {} Champions ---", council_of_champions.len());
    
    // Validate inputs
    validate_gauntlet_inputs(council_of_champions, training_data, hold_out_data, metrics_config)?;
    
    let mapper = GrammarBasedMapper::new(
        grammar, 
        ga_config.max_program_tokens, 
        ga_config.max_recursion_depth
    );
    let bootstrap_config = BootstrapConfig::default();
    let mut reports = Vec::new();

    for (i, champion) in council_of_champions.iter().enumerate() {
        let rank = i + 1;
        info!("Evaluating champion {} of {}", rank, council_of_champions.len());
        
        match process_champion(
            champion, 
            rank,
            training_data, 
            hold_out_data, 
            &mapper, 
            metrics_config,
            &bootstrap_config
        ) {
            Ok(report) => {
                debug!("Champion {} passed gauntlet evaluation", rank);
                reports.push(report);
            }
            Err(e) => {
                error!("Champion {} failed gauntlet: {}", rank, e);
                // Continue with other champions rather than failing entirely
                continue;
            }
        }
    }

    if reports.is_empty() {
        return Err(GauntletError::CalculationError(
            "No champions successfully completed the gauntlet".to_string()
        ));
    }

    // Sort reports by hold-out performance (or other criteria)
    reports.sort_by(|a, b| {
        b.hold_out_result.final_equity
            .partial_cmp(&a.hold_out_result.final_equity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    print_gauntlet_results(&reports, metrics_config);
    Ok(reports)
}

/// Validates inputs to the gauntlet
fn validate_gauntlet_inputs(
    champions: &[Individual],
    training_data: &[OHLCV],
    hold_out_data: &[OHLCV],
    metrics_config: &MetricsConfig,
) -> Result<(), GauntletError> {
    if champions.is_empty() {
        return Err(GauntletError::InsufficientData(
            "No champions provided for gauntlet evaluation".to_string()
        ));
    }

    if training_data.len() < 50 {
        return Err(GauntletError::InsufficientData(format!(
            "Training data too small: {} candles (need at least 50)", training_data.len()
        )));
    }

    if hold_out_data.len() < 20 {
        return Err(GauntletError::InsufficientData(format!(
            "Hold-out data too small: {} candles (need at least 20)", hold_out_data.len()
        )));
    }

    if metrics_config.bootstrap_runs == 0 {
        return Err(GauntletError::BootstrapFailed(
            "Bootstrap runs cannot be zero".to_string()
        ));
    }

    if !metrics_config.risk_free_rate.is_finite() {
        return Err(GauntletError::CalculationError(
            "Risk-free rate must be finite".to_string()
        ));
    }

    // Validate data integrity
    for (i, candle) in training_data.iter().enumerate() {
        if !is_valid_candle(candle) {
            return Err(GauntletError::InsufficientData(format!(
                "Invalid training data at index {}", i
            )));
        }
    }

    for (i, candle) in hold_out_data.iter().enumerate() {
        if !is_valid_candle(candle) {
            return Err(GauntletError::InsufficientData(format!(
                "Invalid hold-out data at index {}", i
            )));
        }
    }

    Ok(())
}

/// Validates individual OHLCV candle
fn is_valid_candle(candle: &OHLCV) -> bool {
    candle.open.is_finite() && candle.open > 0.0 &&
    candle.high.is_finite() && candle.high > 0.0 &&
    candle.low.is_finite() && candle.low > 0.0 &&
    candle.close.is_finite() && candle.close > 0.0 &&
    candle.volume >= 0.0 && candle.volume.is_finite() &&
    candle.high >= candle.low &&
    candle.high >= candle.open &&
    candle.high >= candle.close &&
    candle.low <= candle.open &&
    candle.low <= candle.close
}

/// Processes a single champion through the gauntlet
fn process_champion(
    champion: &Individual,
    rank: usize,
    training_data: &[OHLCV],
    hold_out_data: &[OHLCV],
    mapper: &GrammarBasedMapper,
    metrics_config: &MetricsConfig,
    bootstrap_config: &BootstrapConfig,
) -> Result<GauntletReport, GauntletError> {
    // Map strategy
    let strategy = mapper.map(&champion.genome).map_err(|e| {
        GauntletError::MappingFailed(format!("Rank {}: {}", rank, e))
    })?;

    // Run hold-out test
    let mut backtester = Backtester::new();
    let hold_out_result = backtester.run(hold_out_data, &strategy, metrics_config.risk_free_rate);

    // Validate hold-out result
    if !hold_out_result.final_equity.is_finite() {
        return Err(GauntletError::CalculationError(format!(
            "Rank {}: Hold-out test produced invalid final equity", rank
        )));
    }

    // Run bootstrap analysis
    let bootstrap_stats = run_block_bootstrap(
        training_data, 
        &strategy, 
        metrics_config, 
        bootstrap_config
    ).map_err(|e| {
        GauntletError::BootstrapFailed(format!("Rank {}: {}", rank, e))
    })?;

    Ok(GauntletReport {
        rank,
        original_fitness: champion.fitness,
        hold_out_result,
        bootstrap_stats,
    })
}

/// Enhanced block bootstrap with proper overlapping blocks and error handling
fn run_block_bootstrap(
    training_data: &[OHLCV],
    strategy: &crate::strategy::Strategy,
    metrics_config: &MetricsConfig,
    config: &BootstrapConfig,
) -> Result<BootstrapStats, String> {
    // Calculate log returns for better statistical properties
    let log_returns = calculate_log_returns(training_data)?;
    
    // Create overlapping blocks
    let blocks = create_overlapping_blocks(&log_returns, config)?;
    
    if blocks.len() < config.min_blocks_required {
        return Err(format!(
            "Insufficient blocks for bootstrap: {} (need at least {})", 
            blocks.len(), config.min_blocks_required
        ));
    }

    let mut bootstrap_results = Vec::with_capacity(metrics_config.bootstrap_runs);
    let mut rng = rng();

    for run in 0..metrics_config.bootstrap_runs {
        if run % 500 == 0 {
            debug!("Bootstrap progress: {}/{}", run, metrics_config.bootstrap_runs);
        }

        // Generate synthetic return series
        let synthetic_returns = resample_blocks(&blocks, log_returns.len(), &mut rng)?;
        
        // Convert back to price series
        let synthetic_candles = generate_synthetic_candles(training_data, &synthetic_returns)?;
        
        // Run backtest on synthetic data
        let mut bootstrap_backtester = Backtester::new();
        let result = bootstrap_backtester.run(&synthetic_candles, strategy, metrics_config.risk_free_rate);
        
        // Validate result
        if result.final_equity.is_finite() && result.final_equity > 0.0 {
            bootstrap_results.push(result);
        } else {
            warn!("Bootstrap run {} produced invalid result, skipping", run);
        }
    }

    if bootstrap_results.is_empty() {
        return Err("All bootstrap runs failed validation".to_string());
    }

    if bootstrap_results.len() < metrics_config.bootstrap_runs / 2 {
        warn!("Only {}/{} bootstrap runs were valid", bootstrap_results.len(), metrics_config.bootstrap_runs);
    }

    calculate_bootstrap_statistics(&bootstrap_results)
}

/// Calculate log returns with proper error handling
fn calculate_log_returns(data: &[OHLCV]) -> Result<Vec<f64>, String> {
    if data.len() < 2 {
        return Err("Need at least 2 data points for returns".to_string());
    }

    let mut returns = Vec::with_capacity(data.len() - 1);
    
    for window in data.windows(2) {
        let prev_price = window[0].close;
        let curr_price = window[1].close;
        
        if prev_price <= 0.0 || curr_price <= 0.0 {
            return Err("Prices must be positive for log returns".to_string());
        }
        
        let log_return = (curr_price / prev_price).ln();
        
        if !log_return.is_finite() {
            return Err("Invalid log return calculated".to_string());
        }
        
        returns.push(log_return);
    }
    
    Ok(returns)
}

/// Create overlapping blocks for bootstrap
fn create_overlapping_blocks(returns: &[f64], config: &BootstrapConfig) -> Result<Vec<Vec<f64>>, String> {
    if returns.len() < config.block_size {
        return Err(format!(
            "Data too small for block size: {} returns, {} block size", 
            returns.len(), config.block_size
        ));
    }

    let step_size = ((config.block_size as f64) * (1.0 - config.overlap_ratio)).ceil() as usize;
    let step_size = step_size.max(1); // Ensure at least step of 1
    
    let mut blocks = Vec::new();
    let mut start = 0;
    
    while start + config.block_size <= returns.len() {
        blocks.push(returns[start..start + config.block_size].to_vec());
        start += step_size;
    }
    
    debug!("Created {} overlapping blocks of size {}", blocks.len(), config.block_size);
    Ok(blocks)
}

/// Resample blocks to create synthetic return series
fn resample_blocks(blocks: &[Vec<f64>], target_length: usize, rng: &mut impl rand::Rng) -> Result<Vec<f64>, String> {
    if blocks.is_empty() {
        return Err("No blocks available for resampling".to_string());
    }

    let mut resampled = Vec::with_capacity(target_length);
    
    while resampled.len() < target_length {
        let block = blocks.choose(rng)
            .ok_or("Failed to choose random block")?;
        
        let remaining = target_length - resampled.len();
        if remaining >= block.len() {
            resampled.extend_from_slice(block);
        } else {
            resampled.extend_from_slice(&block[..remaining]);
        }
    }
    
    Ok(resampled)
}

/// Generate synthetic OHLCV data from log returns
fn generate_synthetic_candles(original_data: &[OHLCV], log_returns: &[f64]) -> Result<Vec<OHLCV>, String> {
    if original_data.is_empty() {
        return Err("Original data is empty".to_string());
    }

    if log_returns.len() + 1 > original_data.len() {
        return Err("Too many returns for original data length".to_string());
    }

    let mut synthetic_candles = Vec::with_capacity(log_returns.len() + 1);
    
    // Start with first original candle
    synthetic_candles.push(original_data[0]);
    
    let mut current_price = original_data[0].close;
    
    for (i, &log_return) in log_returns.iter().enumerate() {
        let next_price = current_price * log_return.exp();
        
        if !next_price.is_finite() || next_price <= 0.0 {
            return Err(format!("Invalid synthetic price at index {}: {}", i, next_price));
        }
        
        // Create realistic OHLC from price movement
        let price_change = next_price - current_price;
        let volatility_factor = 1.02; // Small random variation for H/L
        
        let (high, low) = if price_change >= 0.0 {
            // Price went up
            (next_price * volatility_factor, current_price / volatility_factor)
        } else {
            // Price went down
            (current_price * volatility_factor, next_price / volatility_factor)
        };
        
        // Use original timestamp pattern and volume
        let original_candle = &original_data[(i + 1).min(original_data.len() - 1)];
        
        synthetic_candles.push(OHLCV {
            timestamp: original_candle.timestamp,
            open: current_price,
            high,
            low,
            close: next_price,
            volume: original_candle.volume,
        });
        
        current_price = next_price;
    }
    
    Ok(synthetic_candles)
}

/// Calculate comprehensive bootstrap statistics
fn calculate_bootstrap_statistics(results: &[BacktestResult]) -> Result<BootstrapStats, String> {
    if results.is_empty() {
        return Err("No bootstrap results to analyze".to_string());
    }

    let mut equities: Vec<f64> = results.iter().map(|r| r.final_equity).collect();
    equities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = equities.len();
    let avg_equity = equities.iter().sum::<f64>() / n as f64;
    let median_equity = if n % 2 == 0 {
        (equities[n / 2 - 1] + equities[n / 2]) / 2.0
    } else {
        equities[n / 2]
    };

    let worst_equity = equities[0];
    let best_equity = equities[n - 1];
    
    let profitable_count = equities.iter().filter(|&&e| e > INITIAL_CASH).count();
    let profitable_percentage = (profitable_count as f64 / n as f64) * 100.0;

    // 95% confidence interval
    let ci_lower_idx = ((n as f64) * 0.025).floor() as usize;
    let ci_upper_idx = (((n as f64) * 0.975).ceil() as usize).min(n - 1);
    let confidence_interval_95 = (equities[ci_lower_idx], equities[ci_upper_idx]);

    // Calculate volatility of equity outcomes
    let variance = equities.iter()
        .map(|&e| (e - avg_equity).powi(2))
        .sum::<f64>() / n as f64;
    let volatility = variance.sqrt();

    Ok(BootstrapStats {
        avg_equity,
        median_equity,
        worst_equity,
        best_equity,
        profitable_percentage,
        confidence_interval_95,
        volatility,
    })
}

/// Print formatted gauntlet results
fn print_gauntlet_results(reports: &[GauntletReport], metrics_config: &MetricsConfig) {
    println!("\n\n╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           🏛️  THE GAUNTLET RESULTS  🏛️                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");
    
    println!("\n📊 HOLD-OUT PERFORMANCE SUMMARY:");
    println!("{:<5} | {:<12} | {:<15} | {:<12} | {:<12} | {:<12}",
        "Rank", "Orig.Fitness", "Hold-Out Equity", "Sharpe Ratio", "Max DD (%)", "Ann.Return (%)");
    println!("{}", "─".repeat(85));
    
    for report in reports {
        println!("{:<5} | {:<12.4} | ${:<14.2} | {:<12.3} | {:<11.2} | {:<13.2}",
            report.rank,
            report.original_fitness,
            report.hold_out_result.final_equity,
            report.hold_out_result.sharpe_ratio,
            report.hold_out_result.max_drawdown * 100.0,
            report.hold_out_result.annualized_return * 100.0,
        );
    }

    println!("\n🔬 BOOTSTRAP ANALYSIS ({} runs per champion):", metrics_config.bootstrap_runs);
    println!("{:<5} | {:<15} | {:<15} | {:<15} | {:<12} | {:<20}",
        "Rank", "Bootstrap Avg", "Median", "Worst Case", "Profitable%", "95% CI Range");
    println!("{}", "─".repeat(100));
    
    for report in reports {
        let ci_range = report.bootstrap_stats.confidence_interval_95.1 - 
                      report.bootstrap_stats.confidence_interval_95.0;
        println!("{:<5} | ${:<14.2} | ${:<14.2} | ${:<14.2} | {:<11.1} | ${:<19.2}",
            report.rank,
            report.bootstrap_stats.avg_equity,
            report.bootstrap_stats.median_equity,
            report.bootstrap_stats.worst_equity,
            report.bootstrap_stats.profitable_percentage,
            ci_range,
        );
    }

    println!("\n🏆 CHAMPION SUMMARY:");
    if let Some(best) = reports.first() {
        println!("   • Best Hold-Out Performer: Rank {} (${:.2} final equity)", 
            best.rank, best.hold_out_result.final_equity);
        println!("   • Bootstrap Success Rate: {:.1}% profitable scenarios", 
            best.bootstrap_stats.profitable_percentage);
        println!("   • Risk Assessment: Worst case ${:.2}, Best case ${:.2}", 
            best.bootstrap_stats.worst_equity, best.bootstrap_stats.best_equity);
    }
}


/// Writes the final gauntlet reports to a timestamped JSON file.
pub fn write_reports_to_json(reports: &[GauntletReport]) -> Result<(), std::io::Error> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let filename = format!("gauntlet_report_{}.json", timestamp);
    let path = Path::new(&filename);
    
    info!("Writing final gauntlet report to '{}'", path.display());
    
    let json_string = serde_json::to_string_pretty(reports)?;
    
    let mut file = File::create(path)?;
    file.write_all(json_string.as_bytes())?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(count: usize, start_price: f64) -> Vec<OHLCV> {
        (0..count).map(|i| {
            let price = start_price * (1.0 + 0.01 * (i as f64 - count as f64 / 2.0) / count as f64);
            OHLCV {
                timestamp: i as i64,
                open: price,
                high: price * 1.02,
                low: price * 0.98,
                close: price,
                volume: 1000.0,
            }
        }).collect()
    }

    #[test]
    fn test_log_returns_calculation() {
        let candles = create_test_candles(10, 100.0);
        let returns = calculate_log_returns(&candles).unwrap();
        
        assert_eq!(returns.len(), 9);
        assert!(returns.iter().all(|&r| r.is_finite()));
    }

    #[test]
    fn test_block_creation() {
        let returns = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let config = BootstrapConfig { block_size: 3, overlap_ratio: 0.5, min_blocks_required: 1 };
        
        let blocks = create_overlapping_blocks(&returns, &config).unwrap();
        
        assert!(blocks.len() > 0);
        assert_eq!(blocks[0].len(), 3);
    }

    #[test]
    fn test_candle_validation() {
        let valid_candle = OHLCV {
            timestamp: 1,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        };
        assert!(is_valid_candle(&valid_candle));

        let invalid_candle = OHLCV {
            timestamp: 1,
            open: 100.0,
            high: 90.0, // High < Open (invalid)
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        };
        assert!(!is_valid_candle(&invalid_candle));
    }
}
