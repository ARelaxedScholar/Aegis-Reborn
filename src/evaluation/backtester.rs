use crate::data::OHLCV;
use crate::evaluation::indicators::IndicatorManager;
use crate::strategy::Strategy;
use crate::vm::engine::{VirtualMachine, VmContext};
use crate::vm::op::{IndicatorType, Op};
use log::warn;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

const ANNUALIZATION_FACTOR: f64 = 252.0; // Assuming 252 trading days in a year

#[derive(Debug, PartialEq, Clone, Copy)]
enum PositionState {
    Flat,
    Long,
}

#[derive(Debug)]
struct Portfolio {
    cash: f64,
    position_size: f64,
    state: PositionState,
    equity_curve: Vec<f64>,
}

impl Portfolio {
    fn new(initial_cash: f64) -> Self {
        Self {
            cash: initial_cash,
            position_size: 0.0,
            state: PositionState::Flat,
            equity_curve: vec![initial_cash],
        }
    }
    /// Updates the equity curve based on the current portfolio value.
    fn update_equity(&mut self, current_price: f64) {
        let equity = self.cash + self.position_size * current_price;
        self.equity_curve.push(equity);
    }
}

/// The complete result of a backtest run, with key performance metrics.
#[derive(Debug, Default, Clone, Serialize)]
pub struct BacktestResult {
    pub final_equity: f64,
    pub entry_error_count: u32,
    pub exit_error_count: u32,
    pub annualized_return: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub equity_curve: Vec<f64>, // Expose the equity curve for aggregation
}

/// Scans all programs in a strategy to find which indicators are required.
pub fn get_required_indicators(strategy: &Strategy) -> Vec<IndicatorType> {
    let mut indicators = HashSet::new();
    for program in strategy.programs.values() {
        for op in program {
            if let Op::PushIndicator(indicator_type) = op {
                indicators.insert(indicator_type.clone());
            }
        }
    }
    indicators.into_iter().collect()
}

/// Calculates the annualized return based on the equity curve.
pub fn calculate_annualized_return(equity_curve: &[f64], num_candles: usize) -> f64 {
    if equity_curve.len() <= 1 || num_candles == 0 {
        return 0.0;
    }
    let total_return = equity_curve.last().unwrap() / equity_curve[0] - 1.0;
    let years = num_candles as f64 / ANNUALIZATION_FACTOR;
    if years <= 0.0 {
        return 0.0;
    }
    (1.0 + total_return).powf(1.0 / years) - 1.0
}

/// Calculates Sharpe Ratio
pub fn calculate_sharpe_ratio(equity_curve: &[f64], risk_free_rate: f64) -> f64 {
    if equity_curve.len() < 20 {
        return 0.0;
    } // Not enough data for meaningful stats

    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] / w[0]) - 1.0)
        .collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

    let std_dev = (returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64)
        .sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    // Convert annual risk-free rate to daily rate
    let daily_risk_free_rate = risk_free_rate / 252.0;

    // Calculate excess return over risk-free rate (daily)
    let excess_return = mean_return - daily_risk_free_rate;

    // Annualize the Sharpe ratio
    (excess_return / std_dev) * (252.0f64.sqrt())
}

/// Calculates the maximum drawdown from an equity curve.
pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }
    let mut max_drawdown = 0.0;
    let mut peak = equity_curve[0];
    for &equity in equity_curve.iter().skip(1) {
        peak = peak.max(equity);
        let drawdown = (peak - equity) / peak; // As a percentage
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    max_drawdown
}

pub struct Backtester {
    vm: VirtualMachine,
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new()
    }
}

impl Backtester {
    pub fn new() -> Self {
        Self {
            vm: VirtualMachine::new(),
        }
    }

    pub fn run(
        &mut self,
        candles: &[OHLCV],
        strategy: &Strategy,
        risk_free_rate: f64,
        initial_cash: f64,
    ) -> BacktestResult {
        let mut portfolio = Portfolio::new(initial_cash);
        let mut entry_error_count = 0;
        let mut exit_error_count = 0;

        if candles.is_empty() {
            return BacktestResult {
                final_equity: initial_cash,
                ..Default::default()
            };
        }

        let required_indicators = get_required_indicators(strategy);
        let mut indicator_manager = IndicatorManager::new(&required_indicators);

        for (i, candle) in candles.iter().enumerate() {
            indicator_manager.next(candle);
            portfolio.update_equity(candle.close);

            let mut context = VmContext {
                open: candle.open,
                high: candle.high,
                low: candle.low,
                close: candle.close,
                indicators: HashMap::new(),
            };
            indicator_manager.populate_context(&mut context);

            if portfolio.state == PositionState::Flat {
                if let Some(entry_program) = strategy.programs.get("entry") {
                    match self.vm.execute(entry_program, &context) {
                        Ok(signal) if signal > 0.0 => {
                            if candle.close > 0.0 {
                                let position_cost = portfolio.cash;
                                portfolio.position_size = position_cost / candle.close;
                                portfolio.cash = 0.0;
                                portfolio.state = PositionState::Long;
                            }
                        }
                        Ok(_) => (),
                        Err(e) => {
                            warn!("VM error on ENTRY program at candle {}: {:?}.", i, e);
                            entry_error_count += 1;
                        }
                    }
                }
            } else {
                // portfolio.state == PositionState::Long
                if let Some(exit_program) = strategy.programs.get("exit") {
                    match self.vm.execute(exit_program, &context) {
                        Ok(signal) if signal > 0.0 => {
                            portfolio.cash = portfolio.position_size * candle.close;
                            portfolio.position_size = 0.0;
                            portfolio.state = PositionState::Flat;
                        }
                        Ok(_) => (),
                        Err(e) => {
                            warn!("VM error on EXIT program at candle {}: {:?}.", i, e);
                            exit_error_count += 1;
                        }
                    }
                }
            }
        }

        // Liquidate any open position at the end of the backtest
        if let Some(last_candle) = candles.last() {
            if portfolio.state == PositionState::Long {
                portfolio.cash = portfolio.position_size * last_candle.close;
                portfolio.position_size = 0.0;
            }
            portfolio.update_equity(last_candle.close);
        }

        let final_equity = portfolio.cash;
        let annualized_return = calculate_annualized_return(&portfolio.equity_curve, candles.len());
        let max_drawdown = calculate_max_drawdown(&portfolio.equity_curve);
        let sharpe_ratio = calculate_sharpe_ratio(&portfolio.equity_curve, risk_free_rate);

        BacktestResult {
            final_equity,
            entry_error_count,
            exit_error_count,
            annualized_return,
            max_drawdown,
            sharpe_ratio,
            equity_curve: portfolio.equity_curve,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::op::Op::*;

    #[test]
    fn test_entry_only_strategy_holds_forever() {
        let candles = vec![
            OHLCV {
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                close: 110.0,
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]); // always enter
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02);

        // Expected: Buys at 100, holds at 110, liquidated at 110 -> 11k final equity.
        assert_eq!(result.final_equity, 11_000.0);
        assert_eq!(result.entry_error_count, 0);

        // Equity curve: [10000 (initial), 10000 (candle 1 before trade), 11000 (candle 2)]
        // Peak is always rising or flat, so drawdown is 0.
        assert_eq!(result.max_drawdown, 0.0);
    }

    #[test]
    fn test_exit_only_strategy_does_nothing() {
        let candles = vec![
            OHLCV {
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                close: 110.0,
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        let mut programs = HashMap::new();
        programs.insert("exit".to_string(), vec![PushConstant(1.0)]); // always exit
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02);

        // Expected: Never enters, so cash remains initial. All metrics are neutral.
        assert_eq!(result.final_equity, initial_cash);
        assert_eq!(result.entry_error_count, 0);
        assert_eq!(result.exit_error_count, 0);
        assert_eq!(result.max_drawdown, 0.0);
        assert_eq!(result.annualized_return, 0.0);
    }

    #[test]
    fn test_vm_error_on_entry_is_logged() {
        let candles = vec![OHLCV {
            close: 100.0,
            ..Default::default()
        }];
        let mut backtester = Backtester::new();
        let mut programs = HashMap::new();
        // This program will cause a stack underflow
        programs.insert("entry".to_string(), vec![Add]);
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02);
        assert_eq!(result.final_equity, initial_cash);
        assert_eq!(result.entry_error_count, 1);
        assert_eq!(result.exit_error_count, 0);
        assert_eq!(result.max_drawdown, 0.0);
        assert_eq!(result.annualized_return, 0.0);
    }

    #[test]
    fn test_drawdown_calculation() {
        // This test validates the helper function directly, as it's simpler
        // than crafting a VM program to guarantee this exact equity curve.
        let equity_curve = vec![
            10000.0, // Initial
            10000.0, // Candle 1 (close=100), equity is 10k cash.
            8000.0,  // Candle 2 (close=80), equity is 100 shares * 80 = 8k.
            12000.0, // Candle 3 (close=120), equity is 100 shares * 120 = 12k.
        ];

        // The peak is 10000. The trough is 8000.
        // Drawdown = (10000 - 8000) / 10000 = 0.2
        let max_drawdown = calculate_max_drawdown(&equity_curve);
        assert_eq!(max_drawdown, 0.2);
    }
}
