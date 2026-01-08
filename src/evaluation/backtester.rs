use crate::data::OHLCV;
use crate::evaluation::indicators::IndicatorManager;
use crate::evaluation::rolling::RollingWindowManager;
use crate::strategy::Strategy;
use crate::vm::engine::{VirtualMachine, VmContext};
use crate::vm::op::{IndicatorType, Op, PriceType};
use log::warn;
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};

/// Struct representing our current position
/// at any given tiem during the backtest
#[derive(Debug, PartialEq, Clone, Copy)]
enum PositionState {
    /// Out of the market, not invested
    Flat,
    /// Hoping for growth in the market, invested
    Long,
}

/// A convenience struct containing all the data
/// needed to compute metrics, while keeping track
/// of the performance of our portfolio under a given
/// strategy
#[derive(Debug)]
struct Portfolio {
    /// Amount of cash available to buy new holdings
    cash: f64,
    /// Factor to use to annualize our metrics (number of candles in year)
    annualization_factor: f64,
    /// Number of shares for our asset
    position_size: f64, // Currently a single asset; may become a Vec<f64> or a HashMap
    /// Current `PositionState`
    state: PositionState,
    /// Tracks our equity (cash + value of holdings) during the backtesting process
    equity_curve: Vec<f64>,
}

impl Portfolio {
    /// Creates a new Portfolio with the given initial cash and annualization factor.
    ///
    /// # Arguments
    /// * `initial_cash` - Starting cash amount
    /// * `annualization_factor` - Number of candles in a year, used for annualizing metrics
    ///
    /// # Returns
    /// A new `Portfolio` instance
    fn new(initial_cash: f64, annualization_factor: f64) -> Self {
        Self {
            cash: initial_cash,
            annualization_factor: annualization_factor,
            position_size: 0.0,
            state: PositionState::Flat,
            equity_curve: vec![initial_cash],
        }
    }

    /// Updates the equity curve based on the current portfolio value.
    ///
    /// This method mutably borrows the `Portfolio` instance,
    /// modifying its `equity_curve` field.
    ///
    /// # Arguments
    /// * `&mut self` - The `Portfolio` instance to update
    /// * `current_price` - Current price of the asset
    ///
    /// # Returns
    /// `()`
    fn update_equity(&mut self, current_price: f64) {
        let equity = self.cash + self.position_size * current_price;
        self.equity_curve.push(equity);
    }
}

/// The complete result of a backtest run, with key performance metrics.
#[derive(Debug, Default, Clone, Serialize)]
pub struct BacktestResult {
    /// Equity after last candle
    pub final_equity: f64,
    /// Number of errors due to the "entry" program
    pub entry_error_count: u32,
    /// Number of errors due to the "exit" program
    pub exit_error_count: u32,
    /// Annualized return... (like yeah)
    pub annualized_return: f64,
    /// Maximum drawdown observed during the backtest (As a percentage)
    pub max_drawdown: f64,
    /// Sharpe Ratio for the `Strategy`
    pub sharpe_ratio: f64,
    /// Equity curves associated with a given `Strategy`
    pub equity_curve: Vec<f64>, // Expose the equity curve for aggregation
}

/// Scans all programs in a strategy to find which indicators are required.
///
/// # Arguments
/// * `strategy` - A reference to a `Strategy` struct that we'll scan to prepare the indicators
///
/// # Returns
/// `Vec<IndicatorType>`, a `Vec` containing our wrapper type `IndicatorType`
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

/// Scans all programs in a strategy to find the maximum historical lookback required.
///
/// # Arguments
/// * `strategy` - A reference to a `Strategy` struct that we'll scan to prepare history buffer
///
/// # Returns
/// `usize` representing the maximum offset/period needed for historical operations
pub fn get_required_history_lookback(strategy: &Strategy) -> usize {
    let mut max_lookback = 0;
    for program in strategy.programs.values() {
        for op in program {
            match op {
                Op::PushPrevious(_, offset) => {
                    max_lookback = max_lookback.max(*offset as usize);
                }
                Op::PushRollingSum(_, period) => {
                    if *period > 0 {
                        // Need period candles: indices 0..period-1
                        max_lookback = max_lookback.max((*period as usize) - 1);
                    }
                }
                _ => {}
            }
        }
    }
    max_lookback
}

/// Scans all programs in a strategy to find which rolling sums are required.
pub fn get_required_rolling_sums(strategy: &Strategy) -> Vec<(PriceType, u16)> {
    let mut sums = HashSet::new();
    for program in strategy.programs.values() {
        for op in program {
            if let Op::PushRollingSum(price_type, period) = op {
                sums.insert((price_type.clone(), *period));
            }
        }
    }
    sums.into_iter().collect()
}

/// Calculates the annualized return based on the equity curve.
///
/// # Arguments
/// * `equity_curve` - A slice representing the equity curve of the strategy
/// * `num_candles` - Number of candles in the period
/// * `annualization_factor` - Number of candles in a year
///
/// # Returns
/// Annualized return as `f64`
pub fn calculate_annualized_return(
    equity_curve: &[f64],
    num_candles: usize,
    annualization_factor: f64,
) -> f64 {
    if equity_curve.len() <= 1 || num_candles == 0 {
        return 0.0;
    }
    let growth = equity_curve.last().unwrap() / equity_curve[0];
    let years = num_candles as f64 / annualization_factor;
    if years <= 0.0 {
        return 0.0;
    }
    growth.powf(1.0 / years) - 1.0
}

/// Calculates Sharpe Ratio
///
/// # Arguments
/// * `equity_curve` - A slice representing the equity curve of the strategy
/// * `risk_free_rate` - Rate of return expected from a risk-free asset
/// * `annualization_factor` - Number of periods in a year (annualization factor)
///
/// # Returns
/// Sharpe ratio as `f64`
pub fn calculate_sharpe_ratio(
    equity_curve: &[f64],
    risk_free_rate: f64,
    annualization_factor: f64,
) -> f64 {
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

    // Convert annual risk-free rate to per-period rate
    let period_risk_free_rate = risk_free_rate / annualization_factor;

    // Calculate excess return over risk-free rate (per-period)
    let excess_return = mean_return - period_risk_free_rate;

    // Annualize the Sharpe ratio
    (excess_return / std_dev) * (annualization_factor.sqrt())
}

/// Calculates the maximum drawdown from an equity curve.
///
/// # Arguments
/// * `equity_curve` - A reference to a container representing the equity curve of some strategy
///
/// # Returns
/// `f64`, Max drawdown (as a percentage)
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

/// The struct that acts as a goon. Only know about the virtual machine which will
/// be used to run the program.
pub struct Backtester {
    /// The `VirtualMachine` which defines the meaning of the operations defined in the programs
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

    /// Runs the actual experiment for a given `Strategy`, this function encodes the logic (and our
    /// assumptions) for how a strategy should be used. Based on current `PositionState` a
    /// different program will be fetched and executed. The `risk_free_rate`,
    /// `initial_cash` and `annualization_factor` are relevant for metrics.
    ///
    /// This method mutably borrows the `Backtester` instance because its `vm` field requires
    /// mut to use its methods.
    /// Runs the backtest with "IS-Lite" (Implementation Shortfall Lite).
    /// Now includes `transaction_cost_pct` to penalize high-frequency turnover.
    ///
    /// # Arguments
    /// * `&mut self` - The `Backtester` instance with the `vm` to use
    /// * `candles` - Reference to a container of `OHLCV` candles
    /// * `strategy` - Reference to the `Strategy` for which we are running the backtest
    /// * `risk_free_rate` - A `f64` representing the expected returns generated by a risk-free investment
    /// * `initial_cash` - A `f64` representing the starting cash amount
    /// * `annualization_factor` - A `f64` representing the number of candles in a year (can technically be fractional)
    /// * `transaction_cost_pct` - A `f64` representing the transaction cost percentage per side (e.g., 0.001 for 0.1%)
    /// * `slippage_pct` - A `f64` representing the slippage percentage per side (e.g., 0.001 for 0.1% price impact)
    ///
    /// # Returns
    /// `BacktestResult`
    pub fn run(
        &mut self,
        candles: &[OHLCV],
        strategy: &Strategy,
        risk_free_rate: f64,
        initial_cash: f64,
        annualization_factor: f64,
        transaction_cost_pct: f64,
        slippage_pct: f64,
    ) -> BacktestResult {
        let mut portfolio = Portfolio::new(initial_cash, annualization_factor);
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

        let required_rolling_sums = get_required_rolling_sums(strategy);
        let mut rolling_manager = RollingWindowManager::new(&required_rolling_sums);

        let max_lookback = get_required_history_lookback(strategy);
        let mut history_buffer = VecDeque::with_capacity(max_lookback + 1);

        for (i, candle) in candles.iter().enumerate() {
            indicator_manager.next(candle);
            rolling_manager.next(candle);
            portfolio.update_equity(candle.close);

            // Update history buffer incrementally (index 0 = current, 1 = previous, etc.)
            history_buffer.push_front(*candle);
            if history_buffer.len() > max_lookback + 1 {
                history_buffer.pop_back();
            }

            let mut context = VmContext {
                open: candle.open,
                high: candle.high,
                low: candle.low,
                close: candle.close,
                indicators: HashMap::new(),
                rolling_sums: HashMap::new(),
                history: history_buffer.clone(),
            };
            indicator_manager.populate_context(&mut context);
            rolling_manager.populate_context(&mut context);

            if portfolio.state == PositionState::Flat {
                if let Some(entry_program) = strategy.programs.get("entry") {
                    match self.vm.execute(entry_program, &context) {
                        Ok(signal) if signal > 0.0 => {
                            if candle.close > 0.0 {
                                // --- IS-LITE: APPLY ENTRY FRICTION ---
                                // We cannot invest 100% of cash; we must save some for the fee/slippage.
                                // Formula: Investable = Cash * (1 - cost_pct)
                                let investable_cash = portfolio.cash * (1.0 - transaction_cost_pct);
                                let entry_price = candle.close * (1.0 + slippage_pct);

                                portfolio.position_size = investable_cash / entry_price;
                                portfolio.cash = 0.0;
                                portfolio.state = PositionState::Long;
                            } else {
                                warn!("Observed OHLCV with negative close price");
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
                            // --- IS-LITE: APPLY EXIT FRICTION ---
                            // Gross Proceeds = Size * Price
                            // Net Cash = Gross Proceeds * (1 - cost_pct)
                            let exit_price = candle.close * (1.0 - slippage_pct);
                            let gross_proceeds = portfolio.position_size * exit_price;
                            portfolio.cash = gross_proceeds * (1.0 - transaction_cost_pct);

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
                // --- IS-LITE: APPLY LIQUIDATION FRICTION ---
                let exit_price = last_candle.close * (1.0 - slippage_pct);
                let gross_proceeds = portfolio.position_size * exit_price;
                portfolio.cash = gross_proceeds * (1.0 - transaction_cost_pct);

                portfolio.position_size = 0.0;
            }
            portfolio.update_equity(last_candle.close);
        }

        let final_equity = portfolio.cash;
        let annualized_return = calculate_annualized_return(
            &portfolio.equity_curve,
            candles.len(),
            portfolio.annualization_factor,
        );
        let max_drawdown = calculate_max_drawdown(&portfolio.equity_curve);
        let sharpe_ratio = calculate_sharpe_ratio(
            &portfolio.equity_curve,
            risk_free_rate,
            portfolio.annualization_factor,
        );

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

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);

        // Expected: Buys at 100 with 10k cash = 100 shares, liquidated at 110 = 11k final equity
        assert_eq!(result.final_equity, 11_000.0);
        assert_eq!(result.entry_error_count, 0);

        // Equity curve: [10000 (initial), 10000 (after candle 1, before trade), 11000 (after candle 2, liquidated)]
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

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);

        // Expected: Never enters, so cash remains initial
        assert_eq!(result.final_equity, 10000.0);
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
        // This program will cause a stack underflow since Add expects TWO operands and we pass none.
        programs.insert("entry".to_string(), vec![Add]);
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);
        assert_eq!(result.final_equity, 10000.0);
        assert_eq!(result.entry_error_count, 1);
        assert_eq!(result.exit_error_count, 0);
        assert_eq!(result.max_drawdown, 0.0);
        assert_eq!(result.annualized_return, 0.0);
    }

    #[test]
    fn test_vm_error_on_exit_is_logged() {
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
        programs.insert("exit".to_string(), vec![Add]); // This will cause stack underflow (same as previous test)
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);
        // Should enter on first candle, then have VM error on exit attempt
        assert_eq!(result.final_equity, 11_000.0); // Liquidated at end
        assert_eq!(result.entry_error_count, 0);
        assert_eq!(result.exit_error_count, 1);
    }

    #[test]
    fn test_drawdown_calculation() {
        // This test validates the helper function directly
        let equity_curve = vec![
            10000.0, // Initial
            10000.0, // Candle 1, no change
            8000.0,  // Drawdown to 8000
            12000.0, // Recovery to 12000 (new peak)
        ];

        // Peak is 10000, trough is 8000
        // Drawdown = (10000 - 8000) / 10000 = 0.2
        let max_drawdown = calculate_max_drawdown(&equity_curve);
        assert_eq!(max_drawdown, 0.2);
    }

    #[test]
    fn test_complete_buy_sell_cycle() {
        let candles = vec![
            OHLCV {
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                close: 110.0,
                ..Default::default()
            },
            OHLCV {
                close: 120.0,
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]); // always enter
        programs.insert("exit".to_string(), vec![PushConstant(1.0)]); // always exit
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);

        // Should buy at 100, sell at 110, then buy again at 110, liquidate at 120
        // First cycle: 10000 -> 11000 (10% gain)
        // Second cycle: 11000 -> 12000 (9.09% gain on the 11000)
        // Third we get in with 91.6... parts, which means we still have the same equity.
        assert_eq!(result.final_equity, 11000.0);
        assert_eq!(result.entry_error_count, 0);
        assert_eq!(result.exit_error_count, 0);
    }

    #[test]
    fn test_annualized_return_calculation() {
        // Test the helper function directly
        let equity_curve = vec![10000.0, 11000.0]; // 10% return over 1 candle
        let _annualized_return = calculate_annualized_return(&equity_curve, 1, 252.0);

        // Should be (1.1)^252 - 1 which is a very large number
        // Let's test a more reasonable scenario
        let equity_curve_year = vec![10000.0, 11000.0]; // 10% return over 252 candles (1 year)
        let annualized_return_year = calculate_annualized_return(&equity_curve_year, 252, 252.0);

        // Should be approximately 10%
        assert!((annualized_return_year - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_sharpe_ratio_with_insufficient_data() {
        let equity_curve = vec![10000.0, 10100.0]; // Only 2 points
        let sharpe = calculate_sharpe_ratio(&equity_curve, 0.02, 252.0);
        assert_eq!(sharpe, 0.0); // Should return 0 for insufficient data
    }

    #[test]
    fn test_empty_candles_returns_initial_cash() {
        let candles = vec![];
        let mut backtester = Backtester::new();
        let strategy = Strategy {
            programs: HashMap::new(),
        };

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);

        assert_eq!(result.final_equity, 10000.0);
        assert_eq!(result.entry_error_count, 0);
        assert_eq!(result.exit_error_count, 0);
        assert_eq!(result.annualized_return, 0.0);
        assert_eq!(result.max_drawdown, 0.0);
        assert_eq!(result.sharpe_ratio, 0.0);
    }

    #[test]
    fn test_slippage_impact() {
        // Create candles with constant close price to isolate slippage effect
        let candles = vec![
            OHLCV {
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                close: 100.0,
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]); // always enter
        programs.insert("exit".to_string(), vec![PushConstant(1.0)]); // always exit
        let strategy = Strategy { programs };

        // 10% slippage each side
        let slippage_pct = 0.1;
        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, slippage_pct);

        // With slippage: entry price = 100 * (1 + 0.1) = 110, exit price = 100 * (1 - 0.1) = 90
        // Shares bought = (cash) / entry price = 10000 / 110 ≈ 90.9090909
        // Proceeds = shares * exit price = 90.9090909 * 90 = 8181.818181
        // No transaction costs, so final equity = 8181.818181
        let expected_equity = 10000.0 * (1.0 - slippage_pct) / (1.0 + slippage_pct);
        assert!((result.final_equity - expected_equity).abs() < 1e-9);
        assert_eq!(result.entry_error_count, 0);
        assert_eq!(result.exit_error_count, 0);
        // Max drawdown should reflect the loss due to slippage
        let expected_max_drawdown = 1.0 - (1.0 - slippage_pct) / (1.0 + slippage_pct);
        assert!((result.max_drawdown - expected_max_drawdown).abs() < 1e-9);
    }

    #[test]
    fn test_transaction_cost_impact() {
        // Create candles with constant close price to isolate transaction cost effect
        let candles = vec![
            OHLCV {
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                close: 100.0,
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]); // always enter
        programs.insert("exit".to_string(), vec![PushConstant(1.0)]); // always exit
        let strategy = Strategy { programs };

        // 10% transaction cost each side
        let transaction_cost_pct = 0.1;
        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, transaction_cost_pct, 0.0);

        // With transaction cost: entry cash = 10000 * (1 - 0.1) = 9000 used to buy shares
        // Shares bought = 9000 / 100 = 90
        // Exit proceeds = shares * 100 = 9000, minus transaction cost = 9000 * (1 - 0.1) = 8100
        // Final equity = 8100
        let expected_equity = 10000.0 * (1.0 - transaction_cost_pct).powi(2);
        assert!((result.final_equity - expected_equity).abs() < 1e-9);
        assert_eq!(result.entry_error_count, 0);
        assert_eq!(result.exit_error_count, 0);
        // Max drawdown should reflect the loss due to transaction cost
        let expected_max_drawdown = 1.0 - (1.0 - transaction_cost_pct).powi(2);
        assert!((result.max_drawdown - expected_max_drawdown).abs() < 1e-9);
    }
}
