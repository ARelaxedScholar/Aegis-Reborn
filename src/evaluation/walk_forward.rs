use crate::data::OHLCV;
use crate::evaluation::backtester::{
    BacktestResult, Backtester, calculate_annualized_return, calculate_max_drawdown,
    calculate_sharpe_ratio,
};
use crate::strategy::Strategy;
use log::debug;

pub struct WalkForwardValidator {
    training_window_size: usize,
    test_window_size: usize,
    risk_free_rate: f64,
}

impl WalkForwardValidator {
    pub fn new(training_window_size: usize, test_window_size: usize, risk_free_rate: f64) -> Self {
        Self {
            training_window_size,
            test_window_size,
            risk_free_rate,
        }
    }

    pub fn validate(&self, candles: &[OHLCV], strategy: &Strategy) -> BacktestResult {
        let mut backtester = Backtester::new();
        let mut composite_equity_curve: Vec<f64> = Vec::new();
        let mut total_entry_errors = 0;
        let mut total_exit_errors = 0;
        let risk_free_rate = self.risk_free_rate;

        let num_windows =
            (candles.len().saturating_sub(self.training_window_size)) / self.test_window_size;
        debug!(
            "Executing walk-forward validation with {} windows.",
            num_windows
        );

        for i in 0..num_windows {
            let train_end = i * self.test_window_size + self.training_window_size;
            let test_end = (train_end + self.test_window_size).min(candles.len());

            if train_end >= test_end {
                break;
            }

            let test_slice = &candles[train_end..test_end];
            let result = backtester.run(test_slice, strategy, risk_free_rate);

            if i == 0 {
                // For the very first window, initialize the composite curve with its full result.
                composite_equity_curve = result.equity_curve;
            } else {
                // For subsequent windows, re-base and append.
                if let Some(&last_equity) = composite_equity_curve.last() {
                    let initial_test_equity =
                        result.equity_curve.first().cloned().unwrap_or(last_equity);
                    // Avoid division by zero if a test slice results in immediate bankruptcy
                    if initial_test_equity > 1e-9 {
                        for &equity_point in result.equity_curve.iter().skip(1) {
                            let rebased_point = last_equity * (equity_point / initial_test_equity);
                            composite_equity_curve.push(rebased_point);
                        }
                    }
                }
            }

            total_entry_errors += result.entry_error_count;
            total_exit_errors += result.exit_error_count;
        }

        // Handle case where no windows could be run
        if composite_equity_curve.is_empty() {
            // Use the initial cash value from the backtester's constant
            composite_equity_curve.push(crate::evaluation::backtester::INITIAL_CASH);
        }

        let final_equity = composite_equity_curve.last().cloned().unwrap_or(0.0);
        // Calculate metrics on the composite, out-of-sample-only equity curve.
        // We use the total number of candles in all *test slices* for accurate annualization.
        let total_test_candles = num_windows * self.test_window_size;
        let annualized_return =
            calculate_annualized_return(&composite_equity_curve, total_test_candles);
        let max_drawdown = calculate_max_drawdown(&composite_equity_curve);
        let sharpe_ratio = calculate_sharpe_ratio(&composite_equity_curve, risk_free_rate);

        BacktestResult {
            final_equity,
            entry_error_count: total_entry_errors,
            exit_error_count: total_exit_errors,
            annualized_return,
            max_drawdown,
            sharpe_ratio,
            equity_curve: composite_equity_curve,
        }
    }
}
