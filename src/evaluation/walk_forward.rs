use crate::data::OHLCV;
use crate::evaluation::backtester::{
    calculate_annualized_return, calculate_max_drawdown, calculate_sharpe_ratio, BacktestResult,
    Backtester,
};
use crate::strategy::Strategy;
use log::{debug, warn};
use std::error::Error;
use std::fmt;

/// Minimum equity threshold to avoid division by zero in rebasing calculations
const MIN_EQUITY_THRESHOLD: f64 = 1e-9;

/// Minimum number of data points required for meaningful validation
const MIN_DATA_POINTS: usize = 2;

/// Custom error types for walk-forward validation
#[derive(Debug, Clone)]
pub enum WalkForwardError {
    InvalidWindowSize(String),
    InsufficientData(String),
    ValidationFailed(String),
    CalculationError(String),
}

impl fmt::Display for WalkForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WalkForwardError::InvalidWindowSize(msg) => write!(f, "Invalid window size: {}", msg),
            WalkForwardError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            WalkForwardError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            WalkForwardError::CalculationError(msg) => write!(f, "Calculation error: {}", msg),
        }
    }
}

impl Error for WalkForwardError {}

/// Walk-forward validator for backtesting strategies with time-series cross-validation
///
/// This validator tests a fixed strategy on sequential out-of-sample windows using
/// walk-forward validation. It splits historical data into multiple training/testing
/// windows, where each test window follows its training window in time. The same
/// strategy is tested on each out-of-sample period to measure its robustness over
/// time. This provides a more robust evaluation than simple backtesting by
/// simulating how a strategy would perform across different market regimes.
///
/// # Example
/// ```rust
/// use aegis_reborn::evaluation::walk_forward::WalkForwardValidator;
/// use aegis_reborn::strategy::Strategy;
/// use aegis_reborn::data::OHLCV;
///
/// let candles : Vec<OHLCV> = vec![
///     OHLCV {
///          timestamp: 1672531200,
///          open: 100.0,
///          high: 105.0,
///          low: 95.0,
///          close: 102.0,
///          volume: 1000.0,
///     } ; 300];
/// let strategy = Strategy::new(); // in practice, would be handled by the mapper
/// let validator = WalkForwardValidator::new(252, 21, 0.02, 10_000.0, 252.0, 0.001).unwrap(); // 1 year train, 1 month test, 2% risk-free rate, 10_000.0$ initial cash, 252.0 annualization rate, 0.1% transaction cost
/// let result = validator.validate(&candles, &strategy).unwrap();
/// println!("Annualized Return: {:.2}%", result.annualized_return * 100.0);
/// ```
pub struct WalkForwardValidator {
    test_window_size: usize,
    risk_free_rate: f64,
    initial_cash: f64,
    annualization_rate: f64,
    transaction_cost_pct: f64,
}

impl WalkForwardValidator {
    /// Creates a new walk-forward validator
    ///
    /// # Arguments
    /// * `test_window_size` - Number of periods for testing (must be > 0)  
    /// * `risk_free_rate` - Annual risk-free rate for Sharpe ratio calculation (must be finite)
    ///
    /// # Returns
    /// * `Result<WalkForwardValidator, WalkForwardError>` - The validator or an error
    ///
    /// # Errors
    /// * `InvalidWindowSize` - If window sizes are zero or invalid
    /// * `ValidationFailed` - If risk-free rate is not finite
    pub fn new(
        test_window_size: usize,
        risk_free_rate: f64,
        initial_cash: f64,
        annualization_rate: f64,
        transaction_cost_pct: f64,
    ) -> Result<Self, WalkForwardError> {
        // Validate window size
        if test_window_size == 0 {
            return Err(WalkForwardError::InvalidWindowSize(
                "Test window size must be greater than zero".to_string(),
            ));
        }

        // Check for potential overflow in window calculations
        if test_window_size > usize::MAX / 2 {
            return Err(WalkForwardError::InvalidWindowSize(
                "Test window size is too large and may cause overflow".to_string(),
            ));
        }

        // Validate risk-free rate
        if !risk_free_rate.is_finite() {
            return Err(WalkForwardError::ValidationFailed(
                "Risk-free rate must be a finite number".to_string(),
            ));
        }

        if initial_cash < 0.0 {
            return Err(WalkForwardError::ValidationFailed(
                "Initial cash passed is negative".to_string(),
            ));
        }

        if annualization_rate < 0.0 {
            return Err(WalkForwardError::ValidationFailed(
                "Annualization rate cannot be negative.".to_string(),
            ));
        }
        if !transaction_cost_pct.is_finite() {
            return Err(WalkForwardError::ValidationFailed(
                "transaction_cost_pct must be a finite number".to_string(),
            ));
        }
        if transaction_cost_pct < 0.0 {
            return Err(WalkForwardError::ValidationFailed(
                "transaction_cost_pct must be non-negative".to_string(),
            ));
        }
        if transaction_cost_pct > 1.0 {
            return Err(WalkForwardError::ValidationFailed(
                "transaction_cost_pct must be <= 1.0 (100%)".to_string(),
            ));
        }

        debug!(
            "Created WalkForwardValidator with test_window: {}, risk_free_rate: {:.4}",
            test_window_size, risk_free_rate
        );

        Ok(Self {
            test_window_size,
            risk_free_rate,
            initial_cash,
            annualization_rate,
            transaction_cost_pct,
        })
    }

    /// Validates a fixed strategy using walk-forward analysis (sequential out-of-sample testing).
    /// The strategy is not retrained between windows; this measures the robustness of a single
    /// strategy across different time periods.
    ///
    /// # Arguments
    /// * `candles` - Historical price data (OHLCV)
    /// * `strategy` - Trading strategy to validate
    ///
    /// # Returns
    /// * `Result<BacktestResult, WalkForwardError>` - Validation results or error
    ///
    /// # Errors
    /// * `InsufficientData` - If not enough data for at least one complete window
    /// * `ValidationFailed` - If validation process fails
    /// * `CalculationError` - If metric calculations fail
    pub fn validate(
        &self,
        candles: &[OHLCV],
        strategy: &Strategy,
    ) -> Result<BacktestResult, WalkForwardError> {
        // Validate input data
        self.validate_input_data(candles)?;

        let mut backtester = Backtester::new();
        let mut composite_equity_curve: Vec<f64> = Vec::new();
        let mut total_entry_errors = 0;
        let mut total_exit_errors = 0;
        let mut successful_windows = 0;

        // Calculate number of possible windows
        let min_required_data = self.test_window_size;
        let available_data = candles.len();

        if available_data < min_required_data {
            return Err(WalkForwardError::InsufficientData(format!(
                "Need at least {} data points, but only {} available",
                min_required_data, available_data
            )));
        }

        let max_possible_windows = available_data.saturating_div(self.test_window_size);

        if max_possible_windows == 0 {
            return Err(WalkForwardError::InsufficientData(
                "Cannot create any complete windows with the given parameters".to_string(),
            ));
        }

        debug!(
            "Executing walk-forward validation with up to {} windows on {} data points.",
            max_possible_windows, available_data
        );

        // Execute walk-forward validation
        for window_idx in 0..max_possible_windows {
            let train_end = window_idx * self.test_window_size;
            let test_end = (train_end + self.test_window_size).min(candles.len());

            // Ensure we have a valid test window
            if train_end >= test_end || train_end >= candles.len() {
                debug!(
                    "Stopping at window {} due to insufficient remaining data",
                    window_idx
                );
                break;
            }

            let test_slice = &candles[train_end..test_end];

            // Skip if test slice is too small to be meaningful
            if test_slice.len() < MIN_DATA_POINTS {
                warn!(
                    "Skipping window {} due to insufficient test data ({} points)",
                    window_idx,
                    test_slice.len()
                );
                continue;
            }

            // Run backtest on this window
            match self.run_window_backtest(&mut backtester, test_slice, strategy, window_idx) {
                Ok(result) => {
                    // Process successful backtest result
                    if let Err(e) =
                        self.process_window_result(&result, &mut composite_equity_curve, window_idx)
                    {
                        warn!("Failed to process window {} result: {}", window_idx, e);
                        continue;
                    }

                    total_entry_errors += result.entry_error_count;
                    total_exit_errors += result.exit_error_count;
                    successful_windows += 1;
                }
                Err(e) => {
                    warn!("Window {} backtest failed: {}", window_idx, e);
                    continue;
                }
            }
        }

        // Validate that we had at least one successful window
        if successful_windows == 0 {
            return Err(WalkForwardError::ValidationFailed(
                "No windows completed successfully".to_string(),
            ));
        }

        // Handle edge case where composite curve is empty (shouldn't happen with successful windows)
        if composite_equity_curve.is_empty() {
            warn!("Composite equity curve is empty, initializing with initial cash");
            composite_equity_curve.push(self.initial_cash);
        }

        // Calculate final metrics
        self.calculate_final_result(
            composite_equity_curve,
            total_entry_errors,
            total_exit_errors,
            successful_windows,
        )
    }

    /// Validates input data for the walk-forward analysis
    fn validate_input_data(&self, candles: &[OHLCV]) -> Result<(), WalkForwardError> {
        if candles.is_empty() {
            return Err(WalkForwardError::InsufficientData(
                "Cannot validate with empty data".to_string(),
            ));
        }

        if candles.len() < MIN_DATA_POINTS {
            return Err(WalkForwardError::InsufficientData(format!(
                "Need at least {} data points for validation, got {}",
                MIN_DATA_POINTS,
                candles.len()
            )));
        }

        // Check for basic data integrity
        for (i, candle) in candles.iter().enumerate() {
            if !candle.close.is_finite()
                || !candle.high.is_finite()
                || !candle.low.is_finite()
                || !candle.open.is_finite()
            {
                return Err(WalkForwardError::ValidationFailed(format!(
                    "Invalid price data at index {}: OHLC values must be finite",
                    i
                )));
            }

            if candle.volume < 0.0 {
                return Err(WalkForwardError::ValidationFailed(format!(
                    "Invalid volume data at index {}: volume cannot be negative",
                    i
                )));
            }
        }

        Ok(())
    }

    /// Runs backtest for a single window
    fn run_window_backtest(
        &self,
        backtester: &mut Backtester,
        test_slice: &[OHLCV],
        strategy: &Strategy,
        window_idx: usize,
    ) -> Result<BacktestResult, WalkForwardError> {
        debug!(
            "Running backtest for window {} with {} data points",
            window_idx,
            test_slice.len()
        );

        let result = backtester.run(
            test_slice,
            strategy,
            self.risk_free_rate,
            self.initial_cash,
            self.annualization_rate,
            self.transaction_cost_pct,
        );

        // Validate backtest result
        if result.equity_curve.is_empty() {
            return Err(WalkForwardError::ValidationFailed(format!(
                "Window {} produced empty equity curve",
                window_idx
            )));
        }

        // Check for reasonable final equity (not NaN or infinite)
        if !result.final_equity.is_finite() || result.final_equity < 0.0 {
            return Err(WalkForwardError::ValidationFailed(format!(
                "Window {} produced invalid final equity: {}",
                window_idx, result.final_equity
            )));
        }

        Ok(result)
    }

    /// Processes and combines results from a single window
    fn process_window_result(
        &self,
        result: &BacktestResult,
        composite_equity_curve: &mut Vec<f64>,
        window_idx: usize,
    ) -> Result<(), WalkForwardError> {
        if window_idx == 0 {
            // For the first window, use the full equity curve
            *composite_equity_curve = result.equity_curve.clone();
            debug!(
                "Initialized composite curve with {} points from first window",
                composite_equity_curve.len()
            );
        } else {
            // For subsequent windows, rebase and append
            let last_equity = composite_equity_curve.last().copied().ok_or_else(|| {
                WalkForwardError::CalculationError(
                    "Composite equity curve is unexpectedly empty".to_string(),
                )
            })?;

            let initial_test_equity = result.equity_curve.first().copied().ok_or_else(|| {
                WalkForwardError::CalculationError(
                    "Window equity curve is unexpectedly empty".to_string(),
                )
            })?;

            // Check for division by zero or very small values
            if initial_test_equity.abs() < MIN_EQUITY_THRESHOLD {
                warn!(
                    "Window {} has very small initial equity ({}), using last equity as fallback",
                    window_idx, initial_test_equity
                );
                // Simply extend with the last equity value for each point
                composite_equity_curve.extend(std::iter::repeat_n(
                    last_equity,
                    result.equity_curve.len().saturating_sub(1),
                ));
            } else {
                // Normal rebasing: scale the window results relative to where we left off
                for equity_point in result.equity_curve.iter().skip(1) {
                    let scaling_factor = equity_point / initial_test_equity;
                    if !scaling_factor.is_finite() {
                        return Err(WalkForwardError::CalculationError(format!(
                            "Invalid scaling factor in window {}: {} / {} = {}",
                            window_idx, equity_point, initial_test_equity, scaling_factor
                        )));
                    }

                    let rebased_point = last_equity * scaling_factor;
                    if !rebased_point.is_finite() {
                        return Err(WalkForwardError::CalculationError(format!(
                            "Invalid rebased equity in window {}: {}",
                            window_idx, rebased_point
                        )));
                    }

                    composite_equity_curve.push(rebased_point);
                }
            }

            debug!(
                "Processed window {} results, composite curve now has {} points",
                window_idx,
                composite_equity_curve.len()
            );
        }

        Ok(())
    }

    /// Calculates final validation results and metrics
    fn calculate_final_result(
        &self,
        composite_equity_curve: Vec<f64>,
        total_entry_errors: u32,
        total_exit_errors: u32,
        successful_windows: usize,
    ) -> Result<BacktestResult, WalkForwardError> {
        let final_equity = composite_equity_curve.last().copied().ok_or_else(|| {
            WalkForwardError::CalculationError(
                "Cannot calculate final equity from empty curve".to_string(),
            )
        })?;

        // Calculate total test periods for proper annualization
        // Use successful_windows instead of theoretical max to be accurate
        let total_test_periods = successful_windows * self.test_window_size;

        if total_test_periods == 0 {
            return Err(WalkForwardError::CalculationError(
                "Cannot calculate metrics with zero test periods".to_string(),
            ));
        }

        // Calculate performance metrics with error handling
        let annualized_return = match calculate_annualized_return(
            &composite_equity_curve,
            total_test_periods,
            self.annualization_rate,
        ) {
            return_val if return_val.is_finite() => return_val,
            _ => {
                warn!("Annualized return calculation produced invalid result, using 0.0");
                0.0
            }
        };

        let max_drawdown = match calculate_max_drawdown(&composite_equity_curve) {
            dd if dd.is_finite() => dd,
            _ => {
                warn!("Max drawdown calculation produced invalid result, using 0.0");
                0.0
            }
        };

        let sharpe_ratio =
            match calculate_sharpe_ratio(&composite_equity_curve, self.risk_free_rate, self.annualization_rate) {
                ratio if ratio.is_finite() => ratio,
                _ => {
                    warn!("Sharpe ratio calculation produced invalid result, using 0.0");
                    0.0
                }
            };

        debug!(
            "Walk-forward validation completed successfully: {} windows, final_equity: {:.2}, \
             annualized_return: {:.4}, max_drawdown: {:.4}, sharpe_ratio: {:.4}",
            successful_windows, final_equity, annualized_return, max_drawdown, sharpe_ratio
        );

        Ok(BacktestResult {
            final_equity,
            entry_error_count: total_entry_errors,
            exit_error_count: total_exit_errors,
            annualized_return,
            max_drawdown,
            sharpe_ratio,
            equity_curve: composite_equity_curve,
        })
    }

    /// Returns the test window size  
    pub fn test_window_size(&self) -> usize {
        self.test_window_size
    }

    /// Returns the risk-free rate
    pub fn risk_free_rate(&self) -> f64 {
        self.risk_free_rate
    }

    /// Estimates the number of windows that will be created for given data length
    pub fn estimate_windows(&self, data_length: usize) -> usize {
        data_length.saturating_div(self.test_window_size)
    }

    /// Validates parameters without creating the validator instance
    pub fn validate_parameters(
        test_window_size: usize,
        risk_free_rate: f64,
        initial_cash: f64,
        annualization_rate: f64,
        transaction_cost_pct: f64,
        data_length: usize,
    ) -> Result<usize, WalkForwardError> {
        // Create temporary validator to check parameters
        let validator = Self::new(
            test_window_size,
            risk_free_rate,
            initial_cash,
            annualization_rate,
            transaction_cost_pct,
        )?;

        let estimated_windows = validator.estimate_windows(data_length);
        if estimated_windows == 0 {
            return Err(WalkForwardError::InsufficientData(format!(
                "Parameters would produce 0 windows for {} data points. Need at least {} points.",
                data_length,
                test_window_size
            )));
        }

        Ok(estimated_windows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        // Valid parameters
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, 0.0).is_ok());

        // Invalid parameters
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, 0.0).is_ok());
        assert!(WalkForwardValidator::new(0, 0.05, 10000.0, 252.0, 0.0).is_err());
        assert!(WalkForwardValidator::new(20, f64::NAN, 10000.0, 252.0, 0.0).is_err());
        assert!(WalkForwardValidator::new(20, f64::INFINITY, 10000.0, 252.0, 0.0).is_err());
        assert!(WalkForwardValidator::new(20, 0.05, -1000.0, 252.0, 0.0).is_err()); // Negative initial cash
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, -252.0, 0.0).is_err());
        // Negative annualization rate
    }

    #[test]
    fn test_transaction_cost_validation() {
        // Valid values (including high costs)
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, 0.0).is_ok());
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, 0.6).is_ok());
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, 1.0).is_ok());
        
        // Invalid values
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, -0.1).is_err());
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, f64::NAN).is_err());
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, f64::INFINITY).is_err());
        assert!(WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, 1.1).is_err());
    }

    #[test]
    fn test_parameter_validation() {
        // Valid case
        assert!(
            WalkForwardValidator::validate_parameters(20, 0.05, 10000.0, 252.0, 0.0, 200).is_ok()
        );

        // Insufficient data
        assert!(
            WalkForwardValidator::validate_parameters(20, 0.05, 10000.0, 252.0, 0.0, 10).is_err()
        );

        // Invalid parameters
        assert!(
            WalkForwardValidator::validate_parameters(0, 0.05, 10000.0, 252.0, 0.0, 200).is_err()
        );
    }

    #[test]
    fn test_window_estimation() {
        let validator = WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, 0.0).unwrap();

        assert_eq!(validator.estimate_windows(200), 10); // 200/20 = 10
        assert_eq!(validator.estimate_windows(100), 5); // 100/20 = 5
        assert_eq!(validator.estimate_windows(119), 5); // 119/20 = 5
        assert_eq!(validator.estimate_windows(120), 6); // 120/20 = 6
    }

    #[test]
    fn test_getters() {
        let validator = WalkForwardValidator::new(20, 0.05, 10000.0, 252.0, 0.0).unwrap();

        assert_eq!(validator.test_window_size(), 20);
        assert_eq!(validator.risk_free_rate(), 0.05);
    }

    #[test]
    fn test_input_data_validation() {
        let validator = WalkForwardValidator::new(1, 0.05, 10000.0, 252.0, 0.0).unwrap();

        // Empty data
        let empty_candles: Vec<OHLCV> = vec![];
        assert!(validator.validate_input_data(&empty_candles).is_err());

        // Valid data
        let valid_candles = vec![
            OHLCV {
                timestamp: 1672531200,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 102.0,
                volume: 1000.0,
            };
            10
        ];
        assert!(validator.validate_input_data(&valid_candles).is_ok());

        // Invalid price data (NaN)
        let mut invalid_candles = valid_candles.clone();
        invalid_candles[0].close = f64::NAN;
        assert!(validator.validate_input_data(&invalid_candles).is_err());

        // Invalid volume data (negative)
        let mut invalid_volume_candles = valid_candles.clone();
        invalid_volume_candles[0].volume = -100.0;
        assert!(validator
            .validate_input_data(&invalid_volume_candles)
            .is_err());
    }

    #[test]
    fn test_edge_cases() {
        // Test very large window sizes (should fail due to overflow protection)
        assert!(WalkForwardValidator::new(usize::MAX, 0.05, 10000.0, 252.0, 0.0).is_err());

        // Test boundary values
        assert!(WalkForwardValidator::new(1, 0.0, 0.0, 0.0, 0.0).is_ok());

        // Test reasonable financial parameters
        let validator = WalkForwardValidator::new(21, 0.02, 100000.0, 252.0, 0.0).unwrap();
        assert_eq!(validator.test_window_size(), 21); // 1 month testing
        assert_eq!(validator.risk_free_rate(), 0.02); // 2% risk-free rate
    }
}
