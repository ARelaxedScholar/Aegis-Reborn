use crate::config::{GaConfig, MetricsConfig};
use crate::data::OHLCV;
use crate::evaluation::backtester::{BacktestResult, Backtester};
use crate::evolution::{grammar::Grammar, mapper::GrammarBasedMapper, Individual};
use log::{debug, error, info, warn};
use rand::prelude::IndexedRandom;
use serde::Serialize;
use serde_json;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::Path;

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
    /// Ordinal rank of position within council (lower is better)
    pub rank: usize,
    /// Original fitness, the fitness which made it qualified to enter the council (fitness before
    /// final gauntlet)
    pub original_fitness: f64,
    /// The performance on the hold-out set
    pub hold_out_result: BacktestResult,
    /// Market-path bootstrap (synthetic price timelines)
    pub market_bootstrap_stats: BootstrapStats,
    /// Strategy-return bootstrap (resampled realized PnL/returns)
    pub pnl_bootstrap_stats: Option<BootstrapStats>, // None if returns not available
}

#[derive(Debug, Serialize)]
pub struct BootstrapStats {
    /// Equity-based stats
    pub avg_equity: f64,
    pub median_equity: f64,
    pub worst_equity: f64,
    pub best_equity: f64,
    pub profitable_percentage: f64,
    pub confidence_interval_95: (f64, f64),
    pub volatility: f64,

    /// Sharpe Ratio
    pub avg_sharpe: f64,
    pub median_sharpe: f64,
    pub sharpe_ci_95: (f64, f64),

    /// Compound Annual Growth Rate
    pub avg_cagr: f64,
    pub median_cagr: f64,
    pub cagr_ci_95: (f64, f64),

    /// Drawdown
    pub avg_max_dd: f64,
    pub median_max_dd: f64,
    pub max_dd_ci_95: (f64, f64),
}

/// Block bootstrap configuration
#[derive(Debug)]
struct BootstrapConfig {
    /// Number of candles for one block
    block_size: usize,
    /// What percentage of overlap do we want for the sliding window of blocks
    overlap_ratio: f64,
    /// Minimum number of blocks we require for bootstrapping
    min_blocks_required: usize,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            block_size: 21,          // ~1 month for daily data
            overlap_ratio: 0.5,      // 50% overlap between blocks
            min_blocks_required: 10, // Minimum blocks needed for valid bootstrap
        }
    }
}

/// Production-ready gauntlet runner with comprehensive validation and error handling
///
///
/// # Arguments
/// * `council_of_champions` - Reference to a container of the top `Individual` for this run
/// * `training_data` - Data meant for the evolution of the top strategies
/// * `hold_out_data` - Data meant for the final evaluation of the champions
/// * `grammar` - Reference to the `Grammar` to be used for the evolutionary process
/// * `ga_config` - Reference to the `GaConfig` that was used for defining the evolutionary logic
/// * `metrics_config` - Reference to the `MetricsConfig` which defines the parameters which are
/// not crucial for the evolutionary process, but are still important for tracking
///
/// # Returns
/// `Result<Vec<GauntletReport>, GauntletError>`
pub fn run_gauntlet(
    council_of_champions: &[Individual],
    training_data: &[OHLCV],
    hold_out_data: &[OHLCV],
    grammar: &Grammar,
    ga_config: &GaConfig,
    metrics_config: &MetricsConfig,
) -> Result<Vec<GauntletReport>, GauntletError> {
    info!(
        "--- Commencing Final Gauntlet on Top {} Champions ---",
        council_of_champions.len()
    );

    // Validate inputs
    validate_gauntlet_inputs(
        council_of_champions,
        training_data,
        hold_out_data,
        metrics_config,
    )?;

    let mapper = GrammarBasedMapper::new(
        grammar,
        ga_config.max_program_tokens,
        ga_config.max_recursion_depth,
    );
    let bootstrap_config = BootstrapConfig::default();
    let mut reports = Vec::new();

    for (i, champion) in council_of_champions.iter().enumerate() {
        let rank = i + 1;
        info!(
            "Evaluating champion {} of {}",
            rank,
            council_of_champions.len()
        );

        match process_champion(
            champion,
            rank,
            training_data,
            hold_out_data,
            &mapper,
            metrics_config,
            &bootstrap_config,
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
            "No champions successfully completed the gauntlet".to_string(),
        ));
    }

    // Sort reports by hold-out performance
    reports.sort_by(|a, b| {
        b.hold_out_result
            .final_equity
            .partial_cmp(&a.hold_out_result.final_equity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    print_gauntlet_results(&reports, metrics_config);
    Ok(reports)
}

/// Validates inputs to the gauntlet
///
///
/// # Arguments
/// * `champions` - Reference to a container of the top `Individual` for this run
/// * `training_data` - Data meant for the evolution of the top strategies
/// * `hold_out_data` - Data meant for the final evaluation of the champions
/// * `metrics_config` - Reference to the `MetricsConfig` which defines the parameters which are
/// not crucial for the evolutionary process, but are still important for tracking
///
/// # Returns
/// `Result<()>, GauntletError>`
fn validate_gauntlet_inputs(
    champions: &[Individual],
    training_data: &[OHLCV],
    hold_out_data: &[OHLCV],
    metrics_config: &MetricsConfig,
) -> Result<(), GauntletError> {
    if champions.is_empty() {
        return Err(GauntletError::InsufficientData(
            "No champions provided for gauntlet evaluation".to_string(),
        ));
    }

    if training_data.len() < 50 {
        return Err(GauntletError::InsufficientData(format!(
            "Training data too small: {} candles (need at least 50)",
            training_data.len()
        )));
    }

    if hold_out_data.len() < 20 {
        return Err(GauntletError::InsufficientData(format!(
            "Hold-out data too small: {} candles (need at least 20)",
            hold_out_data.len()
        )));
    }

    if metrics_config.bootstrap_runs == 0 {
        return Err(GauntletError::BootstrapFailed(
            "Bootstrap runs cannot be zero".to_string(),
        ));
    }

    if !metrics_config.risk_free_rate.is_finite() {
        return Err(GauntletError::CalculationError(
            "Risk-free rate must be finite".to_string(),
        ));
    }

    // Validate data integrity
    for (i, candle) in training_data.iter().enumerate() {
        if !is_valid_candle(candle) {
            return Err(GauntletError::InsufficientData(format!(
                "Invalid training data at index {}",
                i
            )));
        }
    }

    for (i, candle) in hold_out_data.iter().enumerate() {
        if !is_valid_candle(candle) {
            return Err(GauntletError::InsufficientData(format!(
                "Invalid hold-out data at index {}",
                i
            )));
        }
    }

    Ok(())
}

/// Validates individual OHLCV candle
///
/// # Arguments
/// * `candle` - Reference to a candle
///
/// # Returns
/// `bool`
fn is_valid_candle(candle: &OHLCV) -> bool {
    if !(candle.open.is_finite()
        && candle.high.is_finite()
        && candle.low.is_finite()
        && candle.close.is_finite()
        && candle.volume.is_finite())
    {
        return false;
    }
    if candle.open <= 0.0
        || candle.high <= 0.0
        || candle.low <= 0.0
        || candle.close <= 0.0
        || candle.volume < 0.0
    {
        return false;
    }
    // High must be at least as large as both open/close; Low must be no greater than both.
    let oc_max = candle.open.max(candle.close);
    let oc_min = candle.open.min(candle.close);
    candle.high >= oc_max && candle.low <= oc_min && candle.high >= candle.low
}

/// Processes a single champion through the gauntlet
///
/// # Arguments
/// * `champion` - Reference to an outstanding `Individual` from the council
/// * `rank` - Rank of that situation among the `population_size` other individuals (lower is
/// better)
/// * `training_data` - Data meant for the evolution of the top strategies
/// * `hold_out_data` - Data meant for the final evaluation of the champions
/// * `mapper` - Reference to the mapper which mapped the `Op` to a `Grammar`
/// * `metrics_config` - Reference to the `MetricsConfig` which defines the parameters which are
/// * `bootstrap_config` - Reference to the `BootstrapConfig` for the
///
/// # Returns
/// `Result<GauntletReport, GauntletError>`
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
    let strategy = mapper
        .map(&champion.genome)
        .map_err(|e| GauntletError::MappingFailed(format!("Rank {}: {}", rank, e)))?;

    // Run hold-out test
    let mut backtester = Backtester::new();
    let hold_out_result = backtester.run(
        hold_out_data,
        &strategy,
        metrics_config.risk_free_rate,
        metrics_config.initial_cash,
        metrics_config.annualization_rate,
    );

    if !hold_out_result.final_equity.is_finite() {
        return Err(GauntletError::CalculationError(format!(
            "Rank {}: Hold-out test produced invalid final equity",
            rank
        )));
    }

    // 1) Market bootstrap (always)
    let market_bootstrap_stats =
        run_market_bootstrap(training_data, &strategy, metrics_config, bootstrap_config)
            .map_err(|e| GauntletError::BootstrapFailed(format!("Rank {}: {}", rank, e)))?;

    // 2) PnL bootstrap (only if we can extract a return stream)
    let pnl_bootstrap_stats = match extract_strategy_returns(&hold_out_result) {
        Ok(rets) => {
            let start_eq = metrics_config.initial_cash;
            match run_pnl_bootstrap(start_eq, &rets, metrics_config, bootstrap_config) {
                Ok(stats) => Some(stats),
                Err(e) => {
                    warn!("Rank {}: PnL bootstrap skipped: {}", rank, e);
                    None
                }
            }
        }
        Err(e) => {
            warn!("Rank {}: PnL bootstrap unavailable: {}", rank, e);
            None
        }
    };

    Ok(GauntletReport {
        rank,
        original_fitness: champion.fitness,
        hold_out_result,
        market_bootstrap_stats,
        pnl_bootstrap_stats,
    })
}

/// Enhanced market-path bootstrap: resample asset log-returns, rebuild OHLCV realistically,
/// rerun the strategy on each synthetic path, and summarize by terminal equity.
/// This answers: "Would the rules work on alternative-but-similar price paths?"
///
/// # Arguments
/// better)
/// * `training_data` - Data meant for the evolution of the top strategies
/// * `strategy` - Data meant for the final evaluation of the champions
/// * `metrics_config` - Reference to the `MetricsConfig` which defines the parameters which are
/// * `config` - Reference to the `BootstrapConfig` for the
///
/// # Returns
/// `Result<BootstrapStats, String>`
fn run_market_bootstrap(
    training_data: &[OHLCV],
    strategy: &crate::strategy::Strategy,
    metrics_config: &MetricsConfig,
    config: &BootstrapConfig,
) -> Result<BootstrapStats, String> {
    // 1) Compute log returns (close-to-close)
    let log_returns = calculate_log_returns(training_data)?;

    // 2) Build overlapping blocks
    let blocks = create_overlapping_blocks(&log_returns, config)?;
    if blocks.len() < config.min_blocks_required {
        return Err(format!(
            "Insufficient blocks for market bootstrap: {} (need at least {})",
            blocks.len(),
            config.min_blocks_required
        ));
    }

    // 3) Precompute empirical intrabar deltas to avoid synthetic volatility drift
    let intrabar_samples = build_intrabar_samples(training_data)?;

    // 4) Resample, generate synthetic candles, run strategy
    let mut results = Vec::with_capacity(metrics_config.bootstrap_runs);
    let mut rng = rand::rng();

    for run in 0..metrics_config.bootstrap_runs {
        if run % 500 == 0 {
            debug!(
                "Market bootstrap progress: {}/{}",
                run, metrics_config.bootstrap_runs
            );
        }
        let synthetic_returns = resample_blocks(&blocks, log_returns.len(), &mut rng)?;
        let synthetic_candles = generate_synthetic_candles_from_samples(
            training_data,
            &synthetic_returns,
            &intrabar_samples,
            &mut rng,
        )?;

        let mut tester = Backtester::new();
        let result = tester.run(
            &synthetic_candles,
            strategy,
            metrics_config.risk_free_rate,
            metrics_config.initial_cash,
            metrics_config.annualization_rate,
        );
        if result.final_equity.is_finite() && result.final_equity > 0.0 {
            results.push(result);
        } else {
            warn!(
                "Market bootstrap run {} produced invalid result, skipping",
                run
            );
        }
    }

    if results.is_empty() {
        return Err("All market bootstrap runs failed validation".to_string());
    }

    calculate_bootstrap_statistics(&results, metrics_config.initial_cash)
}

/// Intrabar distribution built from historical bars: positive deltas for high-close (upside wick)
/// and close-low (downside wick). Sampling these preserves realistic bar shapes.
#[derive(Clone, Copy, Debug)]
struct IntrabarDelta {
    up: f64,   // high - close
    down: f64, // close - low
}

/// Function to create intrabar-delta samples to allow realistic samples
/// for the evaluation runs
///
/// # Arguments
/// * `data` - Reference to a container of `OHLCV` candles
///
/// # Returns
/// `Result<Vec<IntrabarDelta>>, String>`
fn build_intrabar_samples(data: &[OHLCV]) -> Result<Vec<IntrabarDelta>, String> {
    if data.len() < 10 {
        return Err("Not enough data for intrabar sampling".to_string());
    }
    let mut v = Vec::with_capacity(data.len());
    for c in data.iter() {
        let up = (c.high - c.close).max(0.0);
        let down = (c.close - c.low).max(0.0);
        if up.is_finite() && down.is_finite() {
            v.push(IntrabarDelta { up, down });
        }
    }
    if v.is_empty() {
        return Err("Failed to collect intrabar deltas".to_string());
    }
    Ok(v)
}

/// Generate synthetic candles from resampled log-returns and sampled intrabar deltas.
/// Avoids multiplicative volatility drift and respects candle validity constraints.
///
/// # Arguments
/// * `original_data` - Reference to the original container of `OHLCV` candles
/// * `log_returns` - Reference to the log returns
/// * `intrabar` - The `IntrabarDelta` samples
/// * `rng` - The `rand::rng` object to do samples
/// # Returns
/// `Result<Vec<OHLCV>>, String>`
fn generate_synthetic_candles_from_samples(
    original_data: &[OHLCV],
    log_returns: &[f64],
    intrabar: &[IntrabarDelta],
    rng: &mut impl rand::Rng,
) -> Result<Vec<OHLCV>, String> {
    if original_data.is_empty() {
        return Err("Original data is empty".to_string());
    }
    if log_returns.len() + 1 > original_data.len() {
        return Err("Too many returns for original data length".to_string());
    }

    let mut out = Vec::with_capacity(log_returns.len() + 1);
    out.push(original_data[0]); // seed the series with first bar as-is

    let mut close = original_data[0].close;
    for (i, &lr) in log_returns.iter().enumerate() {
        let next_close = close * lr.exp();
        if !next_close.is_finite() || next_close <= 0.0 {
            return Err(format!(
                "Invalid synthetic close at index {}: {}",
                i, next_close
            ));
        }

        // Open = previous close (simple and consistent)
        let open = close;

        // Sample an intrabar shape from history
        let s = intrabar[rng.random_range(0..intrabar.len())];
        let high = next_close + s.up;
        let low = (next_close - s.down).max(1e-12);

        // Use original timestamp cadence & volume
        let src = &original_data[(i + 1).min(original_data.len() - 1)];
        let candle = OHLCV {
            timestamp: src.timestamp,
            open,
            high: high.max(open).max(next_close), // ensure high ≥ max(open, close)
            low: low.min(open).min(next_close),   // ensure low ≤ min(open, close)
            close: next_close,
            volume: src.volume, // preserve typical volume scale
        };
        if !is_valid_candle(&candle) {
            return Err(format!("Synthetic candle invalid at {}", i));
        }

        out.push(candle);
        close = next_close;
    }
    Ok(out)
}

/// Extracts per-period strategy returns from the backtest result using equity curve diffs.
/// Returns *net* returns (e.g., +0.01 = +1%).
///
/// # Arguments
/// * `result` - The `BacktestResult` to use (in this module, from the Hold-Out set)
///
/// # Returns
/// `Result<Vec<f64>, String>`
fn extract_strategy_returns(result: &BacktestResult) -> Result<Vec<f64>, String> {
    let eq = result.equity_curve.as_slice();
    if eq.len() >= 2 && eq.iter().all(|x| x.is_finite()) {
        let mut rets = Vec::with_capacity(eq.len() - 1);
        for w in eq.windows(2) {
            if w[0] == 0.0 {
                return Err("Cannot calculate return: equity value is zero".to_string());
            }
            let r = (w[1] / w[0]) - 1.0;
            rets.push(r);
        }
        Ok(rets)
    } else {
        Err("Insufficient or invalid equity data".to_string())
    }
}

/// Block-bootstrap the *strategy's* realized return stream.
/// This answers: "How sensitive are my metrics to sequencing noise in the observed PnL?"
///
/// # Arguments
/// * `start_equity` - Equity at the start of the run
/// * `strategy_returns` - The returns that this strategy showed on the Hold-out set
/// * `metrics_config` - Reference to the `MetricsConfig` defining the parameters to use for our
/// metrics
/// * `config` - Reference to the `BootstrapConfig` defining the number of bootstrap runs, etc.
///
/// # Returns
/// `Result<Vec<f64>, String>`
fn run_pnl_bootstrap(
    start_equity: f64,
    strategy_returns: &[f64],
    metrics_config: &MetricsConfig,
    config: &BootstrapConfig,
) -> Result<BootstrapStats, String> {
    if !(start_equity.is_finite() && start_equity > 0.0) {
        return Err("Invalid start equity".to_string());
    }
    if strategy_returns.len() < config.block_size {
        return Err(format!(
            "Too few strategy returns ({}) for block size {}",
            strategy_returns.len(),
            config.block_size
        ));
    }

    // Build overlapping blocks over strategy returns
    let blocks = create_overlapping_blocks(strategy_returns, config)?;
    if blocks.len() < config.min_blocks_required {
        return Err(format!(
            "Insufficient blocks for PnL bootstrap: {} (need at least {})",
            blocks.len(),
            config.min_blocks_required
        ));
    }

    let mut finals: Vec<f64> = Vec::with_capacity(metrics_config.bootstrap_runs);
    let mut rng = rand::rng();

    for run in 0..metrics_config.bootstrap_runs {
        if run % 500 == 0 {
            debug!(
                "PnL bootstrap progress: {}/{}",
                run, metrics_config.bootstrap_runs
            );
        }
        let seq = resample_blocks(&blocks, strategy_returns.len(), &mut rng)?;
        // Rebuild equity by compounding returns
        let mut eq = start_equity;
        for r in seq {
            eq *= 1.0 + r;
            if !eq.is_finite() || eq <= 0.0 {
                // Skip pathological sequences (extreme leverage/fees would show here)
                warn!("PnL bootstrap path blew up; skipping run {}", run);
                eq = f64::NAN;
                break;
            }
        }
        if eq.is_finite() && eq > 0.0 {
            finals.push(eq);
        }
    }

    if finals.is_empty() {
        return Err("All PnL bootstrap runs failed".to_string());
    }

    // Build fake BacktestResult list so we can reuse your existing statistics routine
    let results: Vec<BacktestResult> = finals
        .into_iter()
        .map(|fe| BacktestResult {
            final_equity: fe,
            // These fields aren't used by calculate_bootstrap_statistics (it looks only at final_equity).
            // Populate with safe defaults to satisfy the struct.
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            annualized_return: 0.0,
            // If your BacktestResult has more fields, fill them here or add Default/.. syntax if available.
            ..Default::default()
        })
        .collect();

    calculate_bootstrap_statistics(&results, metrics_config.initial_cash)
}

/// Creates the overlapping blocks used in the Block-Bootstrapping process
///
/// # Arguments
/// * `series` - A slice to a container of a `Copy` type T  
/// * `config` - Reference to the `BootstrapConfig` defining the number of bootstrap runs, etc.
///
/// # Returns
/// `Result<Vec<f64>, String>`
fn create_overlapping_blocks<T: Copy>(
    series: &[T],
    config: &BootstrapConfig,
) -> Result<Vec<Vec<T>>, String> {
    if series.len() < config.block_size {
        return Err(format!(
            "Data too small for block size: {} elements, {} block size",
            series.len(),
            config.block_size
        ));
    }
    let step_size = ((config.block_size as f64) * (1.0 - config.overlap_ratio)).ceil() as usize;
    let step_size = step_size.max(1);
    if step_size == 1 {
        warn!(
            "Block bootstrapping will proceed with step_size of 1. statistical inference may be unreliable."
        );
    }
    let mut blocks = Vec::new();
    let mut start = 0;
    while start + config.block_size <= series.len() {
        blocks.push(series[start..start + config.block_size].to_vec());
        start += step_size;
    }
    debug!(
        "Created {} overlapping blocks of size {}",
        blocks.len(),
        config.block_size
    );
    Ok(blocks)
}

/// Calculate log returns
///
/// # Arguments
/// * `data` - A slice containing the `OHLCV` data to convert to log-returns
///
/// # Returns
/// `Result<Vec<f64>, String>`
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

/// Resample blocks to create synthetic return series
///
/// # Arguments
/// * `blocks` - A slice containing the blocks to resample (i.e. return blocks)
/// * `target_length` - Number of blocks to sample
/// * `rng` - The `rand::rng` instance to sample with
///
/// # Returns
/// `Result<Vec<f64>, String>`
fn resample_blocks(
    blocks: &[Vec<f64>],
    target_length: usize,
    rng: &mut impl rand::Rng,
) -> Result<Vec<f64>, String> {
    if blocks.is_empty() {
        return Err("No blocks available for resampling".to_string());
    }

    let mut resampled = Vec::with_capacity(target_length);

    while resampled.len() < target_length {
        let block = blocks.choose(rng).ok_or("Failed to choose random block")?;

        let remaining = target_length - resampled.len();
        if remaining >= block.len() {
            resampled.extend_from_slice(block);
        } else {
            resampled.extend_from_slice(&block[..remaining]);
        }
    }

    Ok(resampled)
}

/// Calculate comprehensive bootstrap statistics
///
/// # Arguments
/// * `results` - Slice of the `BacktestResult` for a given strategy (obtained from Bootstrap)
/// * `initial_cash` - The initial cash amount... (I think you didn't need me commenting this)
///
/// # Returns
/// `Result<BootstrapStats, String>`, `BootstrapStats` is a summary of the performance of a
/// strategy
fn calculate_bootstrap_statistics(
    results: &[BacktestResult],
    initial_cash: f64,
) -> Result<BootstrapStats, String> {
    if results.is_empty() {
        return Err("No bootstrap results to analyze".to_string());
    }

    // Collect valid metrics from results
    let mut equities = Vec::new();
    let mut sharpes = Vec::new();
    let mut cagrs = Vec::new();
    let mut max_drawdowns = Vec::new();

    for result in results {
        if result.final_equity.is_finite() {
            equities.push(result.final_equity);
        }
        if result.sharpe_ratio.is_finite() {
            sharpes.push(result.sharpe_ratio);
        }
        if result.annualized_return.is_finite() {
            cagrs.push(result.annualized_return);
        }
        if result.max_drawdown.is_finite() {
            max_drawdowns.push(result.max_drawdown);
        }
    }

    if equities.is_empty() {
        return Err("No valid final equity values found in bootstrap results".to_string());
    }

    // Sort equities for statistics
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

    let profitable_count = equities
        .iter()
        .filter(|&&equity| equity > initial_cash)
        .count();
    let profitable_percentage = (profitable_count as f64 / n as f64) * 100.0;

    // Sample standard deviation for volatility
    let variance = if n > 1 {
        equities
            .iter()
            .map(|&e| (e - avg_equity).powi(2))
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };
    let volatility = variance.sqrt();

    // 95% confidence interval
    let ci_lower_idx = ((n - 1) as f64 * 0.025).round() as usize;
    let ci_upper_idx = ((n - 1) as f64 * 0.975).round() as usize;
    let confidence_interval_95 = (equities[ci_lower_idx], equities[ci_upper_idx]);

    let calculate_metric_stats = |mut values: Vec<f64>| -> (f64, f64, (f64, f64)) {
        if values.is_empty() {
            return (f64::NAN, f64::NAN, (f64::NAN, f64::NAN));
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = values.len();
        let mean = values.iter().sum::<f64>() / len as f64;
        let median = if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        };
        let ci_low = values[((len - 1) as f64 * 0.025).round() as usize];
        let ci_high = values[((len - 1) as f64 * 0.975).round() as usize];
        (mean, median, (ci_low, ci_high))
    };

    let (avg_sharpe, median_sharpe, sharpe_ci_95) = calculate_metric_stats(sharpes);
    let (avg_cagr, median_cagr, cagr_ci_95) = calculate_metric_stats(cagrs);
    let (avg_max_dd, median_max_dd, max_dd_ci_95) = calculate_metric_stats(max_drawdowns);

    Ok(BootstrapStats {
        avg_equity,
        median_equity,
        worst_equity,
        best_equity,
        profitable_percentage,
        confidence_interval_95,
        volatility,
        avg_sharpe,
        median_sharpe,
        sharpe_ci_95,
        avg_cagr,
        median_cagr,
        cagr_ci_95,
        avg_max_dd,
        median_max_dd,
        max_dd_ci_95,
    })
}

/// Prints the results of the gauntlet to the console
///
/// # Arguments
/// * `reports` - Slice of the `GauntletReport` for each strategy (frauds will hopefully be
/// exposed)
/// * `metrics_config` - The `MetricsConfig` for the tracking parameters
///
/// # Returns
/// Nothing. Simply prints to the console
fn print_gauntlet_results(reports: &[GauntletReport], metrics_config: &MetricsConfig) {
    println!(
        "\n\n╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║                           🏛️  THE GAUNTLET RESULTS  🏛️                            ║");
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    );

    println!("\n📊 HOLD-OUT PERFORMANCE SUMMARY:");
    println!(
        "{:<5} | {:<12} | {:<15} | {:<12} | {:<12} | {:<12}",
        "Rank", "Orig.Fitness", "Hold-Out Equity", "Sharpe Ratio", "Max DD (%)", "Ann.Return (%)"
    );
    println!("{}", "─".repeat(85));
    for r in reports {
        println!(
            "{:<5} | {:<12.4} | ${:<14.2} | {:<12.3} | {:<11.2} | {:<13.2}",
            r.rank,
            r.original_fitness,
            r.hold_out_result.final_equity,
            r.hold_out_result.sharpe_ratio,
            r.hold_out_result.max_drawdown * 100.0,
            r.hold_out_result.annualized_return * 100.0,
        );
    }

    println!(
        "\n🔬 BOOTSTRAP ANALYSIS — Market Path ({} runs):",
        metrics_config.bootstrap_runs
    );
    println!(
        "{:<5} | {:<15} | {:<15} | {:<15} | {:<12} | {:<20}",
        "Rank", "Bootstrap Avg", "Median", "Worst Case", "Profitable%", "95% CI Range"
    );
    println!("{}", "─".repeat(100));
    for r in reports {
        let s = &r.market_bootstrap_stats;
        let ci_range = s.confidence_interval_95.1 - s.confidence_interval_95.0;
        println!(
            "{:<5} | ${:<14.2} | ${:<14.2} | ${:<14.2} | {:<11.1} | ${:<19.2}",
            r.rank,
            s.avg_equity,
            s.median_equity,
            s.worst_equity,
            s.profitable_percentage,
            ci_range
        );
    }

    println!(
        "\n🔁 BOOTSTRAP ANALYSIS — Strategy PnL ({} runs):",
        metrics_config.bootstrap_runs
    );
    println!(
        "{:<5} | {:<15} | {:<15} | {:<15} | {:<12} | {:<20}",
        "Rank", "Bootstrap Avg", "Median", "Worst Case", "Profitable%", "95% CI Range"
    );
    println!("{}", "─".repeat(100));
    for r in reports {
        if let Some(s) = &r.pnl_bootstrap_stats {
            let ci_range = s.confidence_interval_95.1 - s.confidence_interval_95.0;
            println!(
                "{:<5} | ${:<14.2} | ${:<14.2} | ${:<14.2} | {:<11.1} | ${:<19.2}",
                r.rank,
                s.avg_equity,
                s.median_equity,
                s.worst_equity,
                s.profitable_percentage,
                ci_range
            );
        } else {
            println!(
                "{:<5} | {:<15} | {:<15} | {:<15} | {:<12} | {:<20}",
                r.rank, "n/a", "n/a", "n/a", "n/a", "n/a"
            );
        }
    }

    println!("\n🏆 CHAMPION SUMMARY:");
    if let Some(best) = reports.first() {
        println!(
            "   • Best Hold-Out Performer: Rank {} (${:.2} final equity)",
            best.rank, best.hold_out_result.final_equity
        );
        println!(
            "   • Market Bootstrap Success Rate: {:.1}% profitable",
            best.market_bootstrap_stats.profitable_percentage
        );
        if let Some(pnl) = &best.pnl_bootstrap_stats {
            println!(
                "   • PnL Bootstrap Success Rate: {:.1}% profitable",
                pnl.profitable_percentage
            );
        }
        println!(
            "   • Risk (Market bootstrap): Worst ${:.2}, Best ${:.2}",
            best.market_bootstrap_stats.worst_equity, best.market_bootstrap_stats.best_equity
        );
        if let Some(pnl) = &best.pnl_bootstrap_stats {
            println!(
                "   • Risk (PnL bootstrap): Worst ${:.2}, Best ${:.2}",
                pnl.worst_equity, pnl.best_equity
            );
        }
    }
}

/// Serializable template for the strategies
/// of the council to simplify writing to file
#[derive(Debug, Serialize)]
struct GauntletFile<'a> {
    /// Tracks version of schema
    schema_version: &'static str,
    /// Timestamp for when the file was generated
    generated_at: u64,
    /// The `GauntletReport`s for each `Individual` in the council
    reports: &'a [GauntletReport],
}

/// Writes the reports to a json file
///
/// # Arguments
/// * `reports` - Slice of the `GauntletReport` for each strategy (frauds will hopefully be
/// exposed)
///
/// # Returns
/// Result<(), std::io::Error>, if everything goes well return the unit, else return the error
/// which occurred at file creation
pub fn write_reports_to_json(reports: &[GauntletReport]) -> Result<(), std::io::Error> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let filename = format!("gauntlet_report_{}.json", timestamp);
    let path = Path::new(&filename);

    info!("Writing final gauntlet report to '{}'", path.display());

    let file_struct = GauntletFile {
        schema_version: "1.1.0", // bump when adding new fields
        generated_at: timestamp,
        reports,
    };

    let json_string = serde_json::to_string_pretty(&file_struct)?;
    let mut file = File::create(path)?;
    file.write_all(json_string.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::backtester::BacktestResult;
    use crate::evolution::Individual;

    // Test data helpers
    fn create_test_candles(count: usize, start_price: f64) -> Vec<OHLCV> {
        (0..count)
            .map(|i| {
                let price =
                    start_price * (1.0 + 0.01 * (i as f64 - count as f64 / 2.0) / count as f64);
                OHLCV {
                    timestamp: i as i64,
                    open: price,
                    high: price * 1.02,
                    low: price * 0.98,
                    close: price,
                    volume: 1000.0,
                }
            })
            .collect()
    }

    fn create_volatile_candles(count: usize, start_price: f64) -> Vec<OHLCV> {
        use rand::prelude::*;
        let mut rng = rand::rng();
        let mut price = start_price;

        (0..count)
            .map(|i| {
                let change = rng.random_range(-0.05..0.05); // ±5% daily moves
                price *= 1.0 + change;

                let high = price * rng.random_range(1.001..1.03);
                let low = price * rng.random_range(0.97..0.999);

                OHLCV {
                    timestamp: i as i64,
                    open: price,
                    high,
                    low,
                    close: price,
                    volume: rng.random_range(500.0..2000.0),
                }
            })
            .collect()
    }

    fn create_test_individual(fitness: f64) -> Individual {
        Individual {
            genome: vec![0, 1, 2, 3, 4], // Simple test genome
            fitness,
        }
    }

    fn create_test_config() -> (GaConfig, MetricsConfig) {
        let ga_config = GaConfig {
            max_program_tokens: 100,
            max_recursion_depth: 5,
            ..Default::default()
        };

        let metrics_config = MetricsConfig {
            bootstrap_runs: 10, // Small number for testing
            risk_free_rate: 0.02,
            initial_cash: 10000.0,
            annualization_rate: 252.0,
        };

        (ga_config, metrics_config)
    }

    // Unit Tests for Utility Functions
    #[test]
    fn test_log_returns_calculation_normal() {
        let candles = create_test_candles(10, 100.0);
        let returns = calculate_log_returns(&candles).unwrap();

        assert_eq!(returns.len(), 9);
        assert!(returns.iter().all(|&r| r.is_finite()));

        // Verify mathematical correctness
        for (i, &ret) in returns.iter().enumerate() {
            let expected = (candles[i + 1].close / candles[i].close).ln();
            assert!((ret - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_log_returns_empty_data() {
        let candles = vec![];
        let result = calculate_log_returns(&candles);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Need at least 2 data points"));
    }

    #[test]
    fn test_log_returns_single_candle() {
        let candles = create_test_candles(1, 100.0);
        let result = calculate_log_returns(&candles);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Need at least 2 data points"));
    }

    #[test]
    fn test_log_returns_zero_price() {
        let mut candles = create_test_candles(5, 100.0);
        candles[2].close = 0.0; // Invalid price

        let result = calculate_log_returns(&candles);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Prices must be positive"));
    }

    #[test]
    fn test_log_returns_negative_price() {
        let mut candles = create_test_candles(5, 100.0);
        candles[2].close = -50.0; // Invalid price

        let result = calculate_log_returns(&candles);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Prices must be positive"));
    }

    // Block Bootstrap Tests

    #[test]
    fn test_overlapping_blocks_normal() {
        let returns = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let config = BootstrapConfig {
            block_size: 3,
            overlap_ratio: 0.5,
            min_blocks_required: 1,
        };

        let blocks = create_overlapping_blocks(&returns, &config).unwrap();

        assert!(!blocks.is_empty());
        assert_eq!(blocks[0].len(), 3);

        // Verify overlap calculation: step_size = ceil(3 * (1-0.5)) = ceil(1.5) = 2
        let step_size = ((3.0_f64 * (1.0 - 0.5)).ceil() as usize).max(1);
        assert_eq!(step_size, 2);

        // Check first few blocks for correct overlap
        assert_eq!(blocks[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(blocks[1], vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_overlapping_blocks_insufficient_data() {
        let returns = vec![1.0, 2.0]; // Only 2 elements
        let config = BootstrapConfig {
            block_size: 5,
            overlap_ratio: 0.5,
            min_blocks_required: 1,
        };

        let result = create_overlapping_blocks(&returns, &config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Data too small for block size"));
    }

    #[test]
    fn test_overlapping_blocks_zero_overlap() {
        let returns = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let config = BootstrapConfig {
            block_size: 2,
            overlap_ratio: 0.0, // No overlap
            min_blocks_required: 1,
        };

        let blocks = create_overlapping_blocks(&returns, &config).unwrap();

        assert_eq!(blocks.len(), 3); // 6 elements / 2 block size = 3 blocks
        assert_eq!(blocks[0], vec![1.0, 2.0]);
        assert_eq!(blocks[1], vec![3.0, 4.0]);
        assert_eq!(blocks[2], vec![5.0, 6.0]);
    }

    // Candle Validation Tests

    #[test]
    fn test_candle_validation_normal() {
        let valid_candle = OHLCV {
            timestamp: 1,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        };
        assert!(is_valid_candle(&valid_candle));
    }

    #[test]
    fn test_candle_validation_high_too_low() {
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

    #[test]
    fn test_candle_validation_low_too_high() {
        let invalid_candle = OHLCV {
            timestamp: 1,
            open: 100.0,
            high: 105.0,
            low: 110.0, // Low > High (invalid)
            close: 102.0,
            volume: 1000.0,
        };
        assert!(!is_valid_candle(&invalid_candle));
    }

    #[test]
    fn test_candle_validation_negative_prices() {
        let invalid_candle = OHLCV {
            timestamp: 1,
            open: -100.0, // Negative price
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        };
        assert!(!is_valid_candle(&invalid_candle));
    }

    #[test]
    fn test_candle_validation_infinite_values() {
        let invalid_candle = OHLCV {
            timestamp: 1,
            open: f64::INFINITY, // Infinite value
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        };
        assert!(!is_valid_candle(&invalid_candle));
    }

    #[test]
    fn test_candle_validation_nan_values() {
        let invalid_candle = OHLCV {
            timestamp: 1,
            open: 100.0,
            high: f64::NAN, // NaN value
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        };
        assert!(!is_valid_candle(&invalid_candle));
    }

    // Input Validation Tests

    #[test]
    fn test_validate_inputs_normal() {
        let champions = vec![create_test_individual(0.8)];
        let training_data = create_test_candles(100, 100.0);
        let hold_out_data = create_test_candles(50, 100.0);
        let (_, metrics_config) = create_test_config();

        let result =
            validate_gauntlet_inputs(&champions, &training_data, &hold_out_data, &metrics_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_inputs_no_champions() {
        let champions = vec![];
        let training_data = create_test_candles(100, 100.0);
        let hold_out_data = create_test_candles(50, 100.0);
        let (_, metrics_config) = create_test_config();

        let result =
            validate_gauntlet_inputs(&champions, &training_data, &hold_out_data, &metrics_config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GauntletError::InsufficientData(_)
        ));
    }

    #[test]
    fn test_validate_inputs_insufficient_training_data() {
        let champions = vec![create_test_individual(0.8)];
        let training_data = create_test_candles(10, 100.0); // Too small
        let hold_out_data = create_test_candles(50, 100.0);
        let (_, metrics_config) = create_test_config();

        let result =
            validate_gauntlet_inputs(&champions, &training_data, &hold_out_data, &metrics_config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GauntletError::InsufficientData(_)
        ));
    }

    #[test]
    fn test_validate_inputs_insufficient_holdout_data() {
        let champions = vec![create_test_individual(0.8)];
        let training_data = create_test_candles(100, 100.0);
        let hold_out_data = create_test_candles(5, 100.0); // Too small
        let (_, metrics_config) = create_test_config();

        let result =
            validate_gauntlet_inputs(&champions, &training_data, &hold_out_data, &metrics_config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GauntletError::InsufficientData(_)
        ));
    }

    #[test]
    fn test_validate_inputs_zero_bootstrap_runs() {
        let champions = vec![create_test_individual(0.8)];
        let training_data = create_test_candles(100, 100.0);
        let hold_out_data = create_test_candles(50, 100.0);
        let metrics_config = MetricsConfig {
            bootstrap_runs: 0, // Invalid
            risk_free_rate: 0.02,
            initial_cash: 10000.0,
            annualization_rate: 252.0,
        };

        let result =
            validate_gauntlet_inputs(&champions, &training_data, &hold_out_data, &metrics_config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GauntletError::BootstrapFailed(_)
        ));
    }

    #[test]
    fn test_validate_inputs_invalid_risk_free_rate() {
        let champions = vec![create_test_individual(0.8)];
        let training_data = create_test_candles(100, 100.0);
        let hold_out_data = create_test_candles(50, 100.0);
        let metrics_config = MetricsConfig {
            bootstrap_runs: 10,
            risk_free_rate: f64::NAN, // Invalid
            initial_cash: 10000.0,
            annualization_rate: 252.0,
        };

        let result =
            validate_gauntlet_inputs(&champions, &training_data, &hold_out_data, &metrics_config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GauntletError::CalculationError(_)
        ));
    }

    #[test]
    fn test_validate_inputs_corrupted_training_data() {
        let champions = vec![create_test_individual(0.8)];
        let mut training_data = create_test_candles(100, 100.0);
        training_data[50].high = training_data[50].low - 10.0; // Invalid: high < low
        let hold_out_data = create_test_candles(50, 100.0);
        let (_, metrics_config) = create_test_config();

        let result =
            validate_gauntlet_inputs(&champions, &training_data, &hold_out_data, &metrics_config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GauntletError::InsufficientData(_)
        ));
    }

    // Synthetic Data Generation Tests

    #[test]
    fn test_generate_synthetic_candles_normal() {
        let original_data = create_test_candles(10, 100.0);
        let log_returns = calculate_log_returns(&original_data).unwrap();
        let intrabar = build_intrabar_samples(&original_data).unwrap();
        let mut rng = rand::rng();

        let synthetic = generate_synthetic_candles_from_samples(
            &original_data,
            &log_returns,
            &intrabar,
            &mut rng,
        )
        .unwrap();

        assert_eq!(synthetic.len(), original_data.len());
        assert_eq!(synthetic[0], original_data[0]); // First candle should be identical

        // Verify all synthetic candles are valid
        for candle in &synthetic {
            assert!(is_valid_candle(candle));
        }
    }

    #[test]
    fn test_generate_synthetic_candles_empty_data() {
        let original_data = vec![];
        let log_returns = vec![];
        let intrabar = vec![];
        let mut rng = rand::rng();

        let result = generate_synthetic_candles_from_samples(
            &original_data,
            &log_returns,
            &intrabar,
            &mut rng,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Original data is empty"));
    }

    #[test]
    fn test_build_intrabar_samples_normal() {
        let data = create_test_candles(50, 100.0);
        let samples = build_intrabar_samples(&data).unwrap();

        assert_eq!(samples.len(), data.len());

        // Verify all samples are non-negative and finite
        for sample in &samples {
            assert!(sample.up >= 0.0 && sample.up.is_finite());
            assert!(sample.down >= 0.0 && sample.down.is_finite());
        }
    }

    #[test]
    fn test_build_intrabar_samples_insufficient_data() {
        let data = create_test_candles(5, 100.0); // Less than minimum 10
        let result = build_intrabar_samples(&data);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Not enough data for intrabar sampling"));
    }

    // Block Resampling Tests

    #[test]
    fn test_resample_blocks_normal() {
        let blocks = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mut rng = rand::rng();

        let resampled = resample_blocks(&blocks, 10, &mut rng).unwrap();

        assert_eq!(resampled.len(), 10);
        assert!(resampled.iter().all(|&x| x >= 1.0 && x <= 9.0)); // All values from blocks
    }

    #[test]
    fn test_resample_blocks_exact_length() {
        let blocks = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mut rng = rand::rng();

        let resampled = resample_blocks(&blocks, 6, &mut rng).unwrap();

        assert_eq!(resampled.len(), 6);
    }

    #[test]
    fn test_resample_blocks_empty_blocks() {
        let blocks = vec![];
        let mut rng = rand::rng();

        let result = resample_blocks(&blocks, 10, &mut rng);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No blocks available"));
    }

    // Strategy Return Extraction Tests
    #[test]
    fn test_extract_strategy_returns_with_equity_curve() {
        let backtest_result = BacktestResult {
            final_equity: 120.0,
            equity_curve: vec![100.0, 101.0, 102.02, 103.0404],
            sharpe_ratio: 1.5,
            max_drawdown: 0.05,
            annualized_return: 0.15,
            entry_error_count: 0,
            exit_error_count: 0,
        };
        let returns = extract_strategy_returns(&backtest_result).unwrap();

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.01).abs() < 1e-10); // 1%
        assert!((returns[1] - 0.01009901).abs() < 1e-6); // ~1.0099%
        assert!((returns[2] - 0.01000196).abs() < 1e-6); // ~1.0002%
    }

    #[test]
    fn test_extract_strategy_returns_no_data() {
        let backtest_result = BacktestResult {
            final_equity: 120.0,
            equity_curve: vec![],
            sharpe_ratio: 1.5,
            max_drawdown: 0.05,
            annualized_return: 0.15,
            entry_error_count: 0,
            exit_error_count: 0,
        };

        let result = extract_strategy_returns(&backtest_result);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Insufficient"));
    }

    // Bootstrap Statistics Tests

    #[test]
    fn test_calculate_bootstrap_statistics_normal() {
        let results = vec![
            BacktestResult {
                final_equity: 100.0,
                sharpe_ratio: 1.0,
                max_drawdown: 0.1,
                annualized_return: 0.1,
                ..Default::default()
            },
            BacktestResult {
                final_equity: 110.0,
                sharpe_ratio: 1.2,
                max_drawdown: 0.08,
                annualized_return: 0.12,
                ..Default::default()
            },
            BacktestResult {
                final_equity: 90.0,
                sharpe_ratio: 0.8,
                max_drawdown: 0.15,
                annualized_return: 0.08,
                ..Default::default()
            },
            BacktestResult {
                final_equity: 105.0,
                sharpe_ratio: 1.1,
                max_drawdown: 0.12,
                annualized_return: 0.11,
                ..Default::default()
            },
        ];

        let stats = calculate_bootstrap_statistics(&results, 100.0).unwrap();

        assert_eq!(stats.avg_equity, 101.25); // (100+110+90+105)/4
        assert_eq!(stats.worst_equity, 90.0);
        assert_eq!(stats.best_equity, 110.0);
        assert!(stats.volatility > 0.0);
        assert!(stats.confidence_interval_95.0 <= stats.confidence_interval_95.1);
    }

    #[test]
    fn test_calculate_bootstrap_statistics_empty() {
        let results = vec![];
        let result = calculate_bootstrap_statistics(&results, 100.0);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No bootstrap results"));
    }

    #[test]
    fn test_calculate_bootstrap_statistics_single_result() {
        let results = vec![BacktestResult {
            final_equity: 105.0,
            sharpe_ratio: 1.5,
            max_drawdown: 0.1,
            annualized_return: 0.12,
            ..Default::default()
        }];

        let stats = calculate_bootstrap_statistics(&results, 100.0).unwrap();

        assert_eq!(stats.avg_equity, 105.0);
        assert_eq!(stats.median_equity, 105.0);
        assert_eq!(stats.worst_equity, 105.0);
        assert_eq!(stats.best_equity, 105.0);
        assert_eq!(stats.volatility, 0.0);
    }

    // Error Type Tests

    #[test]
    fn test_gauntlet_error_display() {
        let errors = vec![
            GauntletError::InsufficientData("test data".to_string()),
            GauntletError::MappingFailed("test mapping".to_string()),
            GauntletError::BootstrapFailed("test bootstrap".to_string()),
            GauntletError::CalculationError("test calc".to_string()),
        ];

        for error in errors {
            let display_str = format!("{}", error);
            assert!(!display_str.is_empty());
            assert!(display_str.contains("test"));
        }
    }

    // Integration Tests

    #[test]
    fn test_bootstrap_config_defaults() {
        let config = BootstrapConfig::default();

        assert_eq!(config.block_size, 21);
        assert_eq!(config.overlap_ratio, 0.5);
        assert_eq!(config.min_blocks_required, 10);
    }

    #[test]
    fn test_end_to_end_log_returns_to_synthetic() {
        let original_data = create_volatile_candles(100, 100.0);

        // Calculate returns
        let log_returns = calculate_log_returns(&original_data).unwrap();

        // Create blocks
        let config = BootstrapConfig::default();
        let blocks = create_overlapping_blocks(&log_returns, &config).unwrap();

        // Resample
        let mut rng = rand::rng();
        let resampled = resample_blocks(&blocks, log_returns.len(), &mut rng).unwrap();

        // Build intrabar samples
        let intrabar = build_intrabar_samples(&original_data).unwrap();

        // Generate synthetic candles
        let synthetic = generate_synthetic_candles_from_samples(
            &original_data,
            &resampled,
            &intrabar,
            &mut rng,
        )
        .unwrap();

        assert_eq!(synthetic.len(), original_data.len());
        assert!(synthetic.iter().all(|c| is_valid_candle(c)));

        // First candle should be identical
        assert_eq!(synthetic[0], original_data[0]);
    }

    // Performance Tests (with smaller data for CI)

    #[test]
    fn test_bootstrap_performance_small() {
        let data = create_volatile_candles(50, 100.0);
        let returns = calculate_log_returns(&data).unwrap();
        let config = BootstrapConfig {
            block_size: 5,
            overlap_ratio: 0.5,
            min_blocks_required: 3,
        };

        let start = std::time::Instant::now();
        let blocks = create_overlapping_blocks(&returns, &config).unwrap();
        let duration = start.elapsed();

        assert!(duration.as_millis() < 100); // Should be very fast
        assert!(blocks.len() >= config.min_blocks_required);
    }

    // Edge Case Tests

    #[test]
    fn test_extreme_price_movements() {
        let mut candles = create_test_candles(10, 100.0);

        // Simulate extreme crash
        candles[5].close = candles[4].close * 0.1; // 90% drop
        candles[6].close = candles[5].close * 10.0; // 1000% gain

        let returns = calculate_log_returns(&candles).unwrap();

        assert!(returns.iter().all(|&r| r.is_finite()));
        assert!(returns[4] < -2.0); // Should be very negative for crash
        assert!(returns[5] > 2.0); // Should be very positive for recovery
    }

    #[test]
    fn test_minimal_valid_dataset() {
        let training_data = create_test_candles(50, 100.0); // Exactly minimum

        let returns = calculate_log_returns(&training_data).unwrap();
        assert_eq!(returns.len(), 49);

        let config = BootstrapConfig {
            block_size: 5,
            overlap_ratio: 0.0,
            min_blocks_required: 5,
        };

        let blocks = create_overlapping_blocks(&returns, &config).unwrap();
        assert!(blocks.len() >= config.min_blocks_required);
    }

    #[test]
    fn test_block_bootstrap_with_high_overlap() {
        let returns = (1..=20).map(|x| x as f64).collect::<Vec<_>>();
        let config = BootstrapConfig {
            block_size: 5,
            overlap_ratio: 0.9, // Very high overlap
            min_blocks_required: 1,
        };
        let blocks = create_overlapping_blocks(&returns, &config).unwrap();

        // With 90% overlap, step size should be ceil(5 * 0.1) = 1
        let expected_step = ((5.0_f64 * (1.0 - 0.9)).ceil() as usize).max(1);
        assert_eq!(expected_step, 1);

        // Should have many overlapping blocks
        assert!(blocks.len() > 10);

        // Verify consecutive blocks have high overlap
        if blocks.len() > 1 {
            // With step size 1: block[0] = [1,2,3,4,5], block[1] = [2,3,4,5,6]
            // So elements at indices 1,2,3,4 of block[0] should match
            // elements at indices 0,1,2,3 of block[1]
            let overlap_count = (0..4).filter(|&i| blocks[0][i + 1] == blocks[1][i]).count();
            assert_eq!(overlap_count, 4); // 4 out of 5 elements should overlap
        }
    }

    #[test]
    fn test_block_bootstrap_with_high_overlap_explicit() {
        let returns = (1..=20).map(|x| x as f64).collect::<Vec<_>>();
        let config = BootstrapConfig {
            block_size: 5,
            overlap_ratio: 0.9,
            min_blocks_required: 1,
        };
        let blocks = create_overlapping_blocks(&returns, &config).unwrap();

        // Verify step size calculation
        let step_size =
            ((config.block_size as f64 * (1.0 - config.overlap_ratio)).ceil() as usize).max(1);
        assert_eq!(step_size, 1);

        // Should create approximately (returns.len() - block_size) / step_size + 1 blocks
        let expected_blocks = (returns.len() - config.block_size) / step_size + 1;
        assert_eq!(blocks.len(), expected_blocks);

        // Verify the actual overlap between consecutive blocks
        for i in 0..blocks.len().min(3) {
            // Only check first few to avoid long test runs
            if i + 1 < blocks.len() {
                let current_block = &blocks[i];
                let next_block = &blocks[i + 1];

                // With step_size = 1, the overlap should be block_size - 1 = 4 elements
                let expected_overlap = config.block_size - step_size;
                let actual_overlap = (0..expected_overlap)
                    .filter(|&j| current_block[j + step_size] == next_block[j])
                    .count();

                assert_eq!(
                    actual_overlap,
                    expected_overlap,
                    "Block {} and {} don't have expected overlap",
                    i,
                    i + 1
                );
            }
        }

        // Verify first few blocks have expected values
        assert_eq!(blocks[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(blocks[1], vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(blocks[2], vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_extract_strategy_returns_zero_equity() {
        let backtest_result = BacktestResult {
            final_equity: 0.0,
            equity_curve: vec![100.0, 0.0, 50.0], // Drops to zero in the middle
            sharpe_ratio: -2.0,
            max_drawdown: 1.0,
            annualized_return: -1.0,
            entry_error_count: 0,
            exit_error_count: 0,
        };

        let result = extract_strategy_returns(&backtest_result);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Cannot calculate return: equity value is zero"));
    }

    #[test]
    fn test_profitable_percentage_calculation() {
        let results = vec![
            BacktestResult {
                final_equity: 12000.0, // Profitable
                ..Default::default()
            },
            BacktestResult {
                final_equity: 8000.0, // Loss
                ..Default::default()
            },
            BacktestResult {
                final_equity: 11000.0, // Profitable
                ..Default::default()
            },
            BacktestResult {
                final_equity: 9000.0, // Loss
                ..Default::default()
            },
        ];

        let initial_cash = 10000.0;
        let stats = calculate_bootstrap_statistics(&results, initial_cash).unwrap();

        // 2 out of 4 results are profitable (> 10000)
        assert_eq!(stats.profitable_percentage, 50.0);
    }

    #[test]
    fn test_confidence_interval_calculation() {
        // Create a predictable dataset for CI testing
        let mut results = Vec::new();
        for i in 1..=100 {
            results.push(BacktestResult {
                final_equity: i as f64 * 100.0, // 100, 200, ..., 10000
                ..Default::default()
            });
        }

        let stats = calculate_bootstrap_statistics(&results, 5000.0).unwrap();

        // With 100 evenly spaced values from 100 to 10000:
        // 2.5th percentile should be around the 3rd value (300)
        // 97.5th percentile should be around the 98th value (9800)
        assert!(stats.confidence_interval_95.0 >= 200.0 && stats.confidence_interval_95.0 <= 400.0);
        assert!(
            stats.confidence_interval_95.1 >= 9600.0 && stats.confidence_interval_95.1 <= 10000.0
        );
    }

    #[test]
    fn test_nan_and_infinite_filtering() {
        let results = vec![
            BacktestResult {
                final_equity: 100.0,
                sharpe_ratio: 1.0,
                max_drawdown: 0.1,
                annualized_return: 0.1,
                ..Default::default()
            },
            BacktestResult {
                final_equity: f64::NAN,      // Should be filtered out
                sharpe_ratio: f64::INFINITY, // Should be filtered out
                max_drawdown: 0.1,
                annualized_return: 0.1,
                ..Default::default()
            },
            BacktestResult {
                final_equity: 110.0,
                sharpe_ratio: 1.2,
                max_drawdown: f64::NAN,           // Should be filtered out
                annualized_return: f64::INFINITY, // Should be filtered out
                ..Default::default()
            },
        ];

        let stats = calculate_bootstrap_statistics(&results, 100.0).unwrap();

        // Only valid final_equity values should be included
        assert_eq!(stats.avg_equity, 105.0); // (100 + 110) / 2
        assert_eq!(stats.median_equity, 105.0);

        // NaN values should result in NaN statistics for those metrics
        assert_eq!(stats.avg_max_dd, 0.1);
        assert_eq!(stats.avg_cagr, 0.1);
    }
    #[test]
    fn test_all_invalid_metrics() {
        let results = vec![
            BacktestResult {
                final_equity: 100.0,         // Valid
                sharpe_ratio: f64::NAN,      // Invalid
                max_drawdown: f64::INFINITY, // Invalid
                annualized_return: f64::NAN, // Invalid
                ..Default::default()
            },
            BacktestResult {
                final_equity: 110.0,              // Valid
                sharpe_ratio: f64::INFINITY,      // Invalid
                max_drawdown: f64::NAN,           // Invalid
                annualized_return: f64::INFINITY, // Invalid
                ..Default::default()
            },
        ];

        let stats = calculate_bootstrap_statistics(&results, 100.0).unwrap();

        // Equity stats should work (valid values exist)
        assert_eq!(stats.avg_equity, 105.0);

        // These should be NaN (no valid values)
        assert!(stats.avg_sharpe.is_nan());
        assert!(stats.avg_max_dd.is_nan());
        assert!(stats.avg_cagr.is_nan());
    }
    #[test]
    fn test_single_equity_volatility() {
        let results = vec![BacktestResult {
            final_equity: 100.0,
            ..Default::default()
        }];

        let stats = calculate_bootstrap_statistics(&results, 90.0).unwrap();

        // With only one data point, volatility should be 0
        assert_eq!(stats.volatility, 0.0);
        assert_eq!(stats.avg_equity, 100.0);
        assert_eq!(stats.median_equity, 100.0);
        assert_eq!(stats.confidence_interval_95.0, 100.0);
        assert_eq!(stats.confidence_interval_95.1, 100.0);
    }
}
