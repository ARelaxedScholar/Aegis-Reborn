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
    /// Stop Loss Price (Absolute)
    sl_price: Option<f64>,
    /// Take Profit Price (Absolute)
    tp_price: Option<f64>,
}

/// Struct to hold pending order details calculated at signal generation time
#[derive(Debug, Clone, Copy)]
struct PendingOrder {
    size_pct: f64,
    sl_dist: Option<f64>,
    tp_dist: Option<f64>,
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
            sl_price: None,
            tp_price: None,
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
    if equity_curve.is_empty() {
        return 0.0;
    }
    let growth = equity_curve.last().copied().unwrap_or(0.0) / equity_curve[0];
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

        // State to track signals generated in the PREVIOUS bar for execution in the CURRENT bar
        let mut pending_entry_order: Option<PendingOrder> = None;
        let mut pending_exit_signal = false;

        for (i, candle) in candles.iter().enumerate() {
            // 1. EXECUTION PHASE (At the Open of the current bar)
            // We execute orders generated by the signal from the *previous* Close.

            if portfolio.state == PositionState::Flat {
                if let Some(order) = pending_entry_order {
                    // Execute BUY at Open
                    if candle.open > 0.0 {
                        // --- IS-LITE: APPLY ENTRY FRICTION ---
                        // Use size_pct from the pending order (default to 1.0 if not specified or invalid)
                        let size_pct = order.size_pct;
                        let investable_cash =
                            portfolio.cash * size_pct * (1.0 - transaction_cost_pct);
                        let entry_price = candle.open * (1.0 + slippage_pct);

                        portfolio.position_size = investable_cash / entry_price;
                        portfolio.cash -= investable_cash / (1.0 - transaction_cost_pct); // Deduct total cash used including cost
                                                                                          // Fix: The previous logic set cash to 0.0, which assumes 100% allocation.
                                                                                          // Now we only deduct what was used.

                        portfolio.state = PositionState::Long;

                        // Set SL/TP Levels based on Entry Price
                        if let Some(dist) = order.sl_dist {
                            portfolio.sl_price = Some(entry_price * (1.0 - dist));
                        } else {
                            portfolio.sl_price = None;
                        }

                        if let Some(dist) = order.tp_dist {
                            portfolio.tp_price = Some(entry_price * (1.0 + dist));
                        } else {
                            portfolio.tp_price = None;
                        }

                        // Reset signal after execution
                        pending_entry_order = None;
                    } else {
                        warn!(
                            "Observed OHLCV with negative/zero open price at index {}",
                            i
                        );
                    }
                }
            }

            if portfolio.state == PositionState::Long {
                // Check for SL/TP Hits (Intraday)
                let mut sl_hit = false;
                let mut tp_hit = false;

                if let Some(sl) = portfolio.sl_price {
                    if candle.low <= sl {
                        sl_hit = true;
                    }
                }

                if let Some(tp) = portfolio.tp_price {
                    if candle.high >= tp {
                        tp_hit = true;
                    }
                }

                // Prioritize SL over TP if both hit (conservative)
                if sl_hit {
                    // SL Exit
                    // Gap logic: If Open < SL, we exit at Open. Else at SL.
                    let sl_price = portfolio.sl_price.unwrap_or(0.0); // Should be safe due to sl_hit check
                    let raw_exit_price = if candle.open < sl_price {
                        candle.open
                    } else {
                        sl_price
                    };

                    let exit_price = raw_exit_price * (1.0 - slippage_pct);
                    let gross_proceeds = portfolio.position_size * exit_price;
                    portfolio.cash += gross_proceeds * (1.0 - transaction_cost_pct);
                    portfolio.position_size = 0.0;
                    portfolio.state = PositionState::Flat;
                    portfolio.sl_price = None;
                    portfolio.tp_price = None;
                    pending_exit_signal = false; // Cancel any pending standard exit
                } else if tp_hit {
                    // TP Exit
                    // Gap logic: If Open > TP, we exit at Open. Else at TP.
                    let tp_price = portfolio.tp_price.unwrap_or(f64::MAX); // Should be safe due to tp_hit check
                    let raw_exit_price = if candle.open > tp_price {
                        candle.open
                    } else {
                        tp_price
                    };

                    let exit_price = raw_exit_price * (1.0 - slippage_pct);
                    let gross_proceeds = portfolio.position_size * exit_price;
                    portfolio.cash += gross_proceeds * (1.0 - transaction_cost_pct);
                    portfolio.position_size = 0.0;
                    portfolio.state = PositionState::Flat;
                    portfolio.sl_price = None;
                    portfolio.tp_price = None;
                    pending_exit_signal = false;
                } else if pending_exit_signal {
                    // Execute Standard SELL at Open
                    if candle.open > 0.0 {
                        // --- IS-LITE: APPLY EXIT FRICTION ---
                        let exit_price = candle.open * (1.0 - slippage_pct);
                        let gross_proceeds = portfolio.position_size * exit_price;
                        portfolio.cash += gross_proceeds * (1.0 - transaction_cost_pct);

                        portfolio.position_size = 0.0;
                        portfolio.state = PositionState::Flat;
                        portfolio.sl_price = None;
                        portfolio.tp_price = None;

                        // Reset signal after execution
                        pending_exit_signal = false;
                    } else {
                        warn!(
                            "Observed OHLCV with negative/zero open price at index {}",
                            i
                        );
                    }
                }
            }

            // 2. UPDATE PHASE (Mark-to-Market)
            // Update indicators and portfolio equity based on the Close of this bar.
            indicator_manager.next(candle);
            rolling_manager.next(candle);

            // Note: We use Close for daily mark-to-market equity tracking, even though we trade at Open.
            portfolio.update_equity(candle.close);

            // Update history buffer
            history_buffer.push_front(*candle);
            if history_buffer.len() > max_lookback + 1 {
                history_buffer.pop_back();
            }

            // 3. SIGNAL GENERATION PHASE (At the Close of the current bar)
            // We run the VM on the full data of this bar to decide what to do *tomorrow* (Next Open).

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
                // We are Flat, look for Entry signal
                if let Some(entry_program) = strategy.programs.get("entry") {
                    match self.vm.execute(entry_program, &context) {
                        Ok(signal) if signal > 0.0 => {
                            // Calculate Size
                            let size_pct = if let Some(size_prog) =
                                strategy.programs.get("position_sizing")
                            {
                                let raw_size = self.vm.execute(size_prog, &context).unwrap_or(1.0);
                                raw_size.clamp(0.0, 1.0)
                            } else {
                                1.0
                            };

                            // Calculate SL
                            let sl_dist = if let Some(sl_prog) = strategy.programs.get("stop_loss")
                            {
                                self.vm.execute(sl_prog, &context).ok()
                            } else {
                                None
                            };

                            // Calculate TP
                            let tp_dist =
                                if let Some(tp_prog) = strategy.programs.get("take_profit") {
                                    self.vm.execute(tp_prog, &context).ok()
                                } else {
                                    None
                                };

                            pending_entry_order = Some(PendingOrder {
                                size_pct,
                                sl_dist,
                                tp_dist,
                            });
                        }
                        Ok(_) => {
                            pending_entry_order = None;
                        }
                        Err(e) => {
                            warn!("VM error on ENTRY program at candle {}: {:?}.", i, e);
                            entry_error_count += 1;
                            pending_entry_order = None;
                        }
                    }
                }
            } else {
                // We are Long, look for Exit signal
                // Note: We check for exit signal even if we just entered in this same loop iteration?
                // No, because portfolio.state was updated in step 1.
                // If we entered at Open (Step 1), we are now Long.
                // We can generate an Exit signal at Close (Step 3) to sell at *next* Open.
                // This is valid: Day trade or 1-day hold.

                if let Some(exit_program) = strategy.programs.get("exit") {
                    match self.vm.execute(exit_program, &context) {
                        Ok(signal) if signal > 0.0 => {
                            pending_exit_signal = true;
                        }
                        Ok(_) => {
                            pending_exit_signal = false;
                        }
                        Err(e) => {
                            warn!("VM error on EXIT program at candle {}: {:?}.", i, e);
                            exit_error_count += 1;
                            pending_exit_signal = false;
                        }
                    }
                }
            }
        }

        // Liquidate any open position at the end of the backtest
        // We execute this at the Close of the last candle to finalize the simulation.
        if let Some(last_candle) = candles.last() {
            if portfolio.state == PositionState::Long {
                // --- IS-LITE: APPLY LIQUIDATION FRICTION ---
                let exit_price = last_candle.close * (1.0 - slippage_pct);
                let gross_proceeds = portfolio.position_size * exit_price;
                portfolio.cash += gross_proceeds * (1.0 - transaction_cost_pct);

                portfolio.position_size = 0.0;
            }
            // Final equity update
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
                open: 100.0,
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                open: 105.0, // Entry happens here (Next Open)
                close: 110.0,
                ..Default::default()
            },
            OHLCV {
                open: 110.0,
                close: 120.0, // Liquidated here
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]); // always enter
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);

        // Candle 0: Signal generated at Close (100).
        // Candle 1: Executed at Open (105). Cash 10000 -> Shares = 10000 / 105 = 95.238...
        // Candle 2: Liquidated at Close (120). Equity = 95.238... * 120 = 11428.57...

        let expected_shares = 10000.0 / 105.0;
        let expected_equity = expected_shares * 120.0;

        assert!((result.final_equity - expected_equity).abs() < 1e-2);
        assert_eq!(result.entry_error_count, 0);
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
    }

    #[test]
    fn test_vm_error_on_exit_is_logged() {
        let candles = vec![
            OHLCV {
                open: 100.0,
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                open: 100.0,  // Enter here
                close: 110.0, // Exit signal generated here (with error)
                ..Default::default()
            },
            OHLCV {
                open: 110.0,
                close: 120.0,
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();
        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]); // always enter
        programs.insert("exit".to_string(), vec![Add]); // This will cause stack underflow
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);
        // Candle 0: Entry Signal.
        // Candle 1: Enter at Open. Exit Signal calculation fails (Error 1).
        // Candle 2: Still Long. Exit Signal calculation fails (Error 2). Liquidated at Close.

        assert_eq!(result.final_equity, 12_000.0);
        assert_eq!(result.entry_error_count, 0);
        assert_eq!(result.exit_error_count, 2); // Error logged twice
    }

    #[test]
    fn test_complete_buy_sell_cycle() {
        let candles = vec![
            OHLCV {
                open: 100.0,
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                open: 100.0,  // Buy executed here (Signal from Candle 0)
                close: 110.0, // Sell Signal generated here
                ..Default::default()
            },
            OHLCV {
                open: 110.0,  // Sell executed here
                close: 120.0, // Buy Signal generated here
                ..Default::default()
            },
            OHLCV {
                open: 120.0,  // Buy executed here
                close: 130.0, // Liquidated here
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]); // always enter
        programs.insert("exit".to_string(), vec![PushConstant(1.0)]); // always exit
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);

        // Cycle 1:
        // C0 Close: Buy Signal.
        // C1 Open: Buy at 100. Shares = 100.
        // C1 Close: Sell Signal.
        // C2 Open: Sell at 110. Cash = 11000.

        // Cycle 2:
        // C2 Close: Buy Signal.
        // C3 Open: Buy at 120. Shares = 11000 / 120 = 91.666...
        // C3 Close: Liquidated at 130. Cash = 91.666... * 130 = 11916.66...

        let expected_final = (10000.0 / 100.0 * 110.0) / 120.0 * 130.0;
        assert!((result.final_equity - expected_final).abs() < 1e-2);
    }

    #[test]
    fn test_gap_risk_reality() {
        // This test verifies that we pay the price for gaps.
        // If we signaled BUY at Close=100, but next Open is 110 (Gap Up), we buy at 110.
        // If we signaled BUY at Close=100, but next Open is 90 (Gap Down), we buy at 90.

        let candles = vec![
            OHLCV {
                open: 100.0,
                close: 100.0, // Signal BUY
                ..Default::default()
            },
            OHLCV {
                open: 110.0,  // Execution at 110 (Gap Up - worse entry)
                close: 120.0, // Liquidate
                ..Default::default()
            },
        ];

        let mut backtester = Backtester::new();
        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]);
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.02, 10000.0, 252.0, 0.0, 0.0);

        // We expect to buy at 110, not 100.
        // Shares = 10000 / 110 = 90.909...
        // Final = 90.909... * 120 = 10909.09...
        // If we had bought at 100, Final would be 12000.

        let expected_equity = 10000.0 / 110.0 * 120.0;
        assert!((result.final_equity - expected_equity).abs() < 1e-2);
        assert!(result.final_equity < 12000.0); // Confirm we didn't get the "cheat" price
    }

    #[test]
    fn test_position_sizing_clamping() {
        let candles = vec![
            OHLCV {
                open: 100.0,
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                open: 100.0,
                close: 110.0,
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        // Case 1: Huge Position Size (should be clamped to 1.0)
        let mut programs_huge = HashMap::new();
        programs_huge.insert("entry".to_string(), vec![PushConstant(1.0)]);
        programs_huge.insert("position_sizing".to_string(), vec![PushConstant(300.0)]); // 30000%
        let strategy_huge = Strategy {
            programs: programs_huge,
        };

        let result_huge = backtester.run(&candles, &strategy_huge, 0.02, 10000.0, 252.0, 0.0, 0.0);

        // Should invest max 100% of cash.
        // Shares = 10000 / 100 = 100.
        // Final Equity = 100 * 110 = 11000.
        // If it used 300.0, it would be 30000 / 100 = 300 shares -> 33000 equity.
        assert!(
            (result_huge.final_equity - 11000.0).abs() < 1e-2,
            "Huge size failed: expected 11000, got {}",
            result_huge.final_equity
        );

        // Case 2: Zero Position Size (should be clamped to 0.0, currently 0.01)
        let mut programs_zero = HashMap::new();
        programs_zero.insert("entry".to_string(), vec![PushConstant(1.0)]);
        programs_zero.insert("position_sizing".to_string(), vec![PushConstant(0.0)]);
        let strategy_zero = Strategy {
            programs: programs_zero,
        };

        let result_zero = backtester.run(&candles, &strategy_zero, 0.02, 10000.0, 252.0, 0.0, 0.0);

        // Should invest 0% of cash.
        // Final Equity = 10000.
        // Current buggy behavior: clamps to 0.01 -> invests 100 -> shares = 1 -> final = 10000 - 100 + 110 = 10010.
        assert!(
            (result_zero.final_equity - 10000.0).abs() < 1e-2,
            "Zero size failed: expected 10000, got {}",
            result_zero.final_equity
        );
    }

    #[test]
    fn test_slippage_impact() {
        let candles = vec![
            OHLCV {
                open: 100.0,
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                open: 100.0,  // Enter here
                close: 100.0, // Exit signal
                ..Default::default()
            },
            OHLCV {
                open: 100.0, // Exit here
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

        // Entry at C1 Open (100). Price with slippage = 110.
        // Shares = 10000 / 110 = 90.909...
        // Exit at C2 Open (100). Price with slippage = 90.
        // Cash = 90.909... * 90 = 8181.81...

        let expected_equity = 10000.0 * (1.0 - slippage_pct) / (1.0 + slippage_pct);
        assert!((result.final_equity - expected_equity).abs() < 1e-2);
    }

    #[test]
    fn test_transaction_cost_impact() {
        let candles = vec![
            OHLCV {
                open: 100.0,
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                open: 100.0,  // Enter here
                close: 100.0, // Exit signal
                ..Default::default()
            },
            OHLCV {
                open: 100.0, // Exit here
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
        let result = backtester.run(
            &candles,
            &strategy,
            0.02,
            10000.0,
            252.0,
            transaction_cost_pct,
            0.0,
        );

        // Entry at C1 Open (100). Investable = 10000 * 0.9 = 9000.
        // Shares = 9000 / 100 = 90.
        // Exit at C2 Open (100). Gross = 90 * 100 = 9000.
        // Net = 9000 * 0.9 = 8100.

        let expected_equity = 10000.0 * (1.0 - transaction_cost_pct).powi(2);
        assert!((result.final_equity - expected_equity).abs() < 1e-2);
    }

    #[test]
    fn test_sl_tp_sizing() {
        let candles = vec![
            OHLCV {
                open: 100.0,
                close: 100.0,
                ..Default::default()
            },
            OHLCV {
                open: 100.0, // Entry at 100.
                high: 105.0,
                low: 85.0, // SL (90) hit here!
                close: 95.0,
                ..Default::default()
            },
        ];
        let mut backtester = Backtester::new();

        let mut programs = HashMap::new();
        programs.insert("entry".to_string(), vec![PushConstant(1.0)]); // always enter

        // Size: 50%
        programs.insert("position_sizing".to_string(), vec![PushConstant(0.5)]);

        // SL: 10% (Price 90)
        programs.insert("stop_loss".to_string(), vec![PushConstant(0.1)]);

        // TP: 20% (Price 120)
        programs.insert("take_profit".to_string(), vec![PushConstant(0.2)]);

        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.0, 10000.0, 252.0, 0.0, 0.0);

        // Candle 1 Entry:
        // Cash used = 10000 * 0.5 = 5000.
        // Shares = 5000 / 100 = 50.
        // Remaining Cash = 5000.

        // Candle 1 Intraday:
        // Low (85) <= SL (90). SL Triggered.
        // Exit Price = 90 (since Open 100 > SL 90).
        // Proceeds = 50 * 90 = 4500.
        // Total Cash = 5000 + 4500 = 9500.

        assert_eq!(result.final_equity, 9500.0);
    }
    #[test]
    fn test_complex_indicators() {
        // Create enough candles for indicators to warm up (20 periods)
        let mut candles = Vec::new();
        for i in 0..50 {
            candles.push(OHLCV {
                open: 100.0 + i as f64,
                high: 105.0 + i as f64,
                low: 95.0 + i as f64,
                close: 100.0 + i as f64,
                volume: 1000.0,
                timestamp: chrono::Utc::now().timestamp(),
            });
        }

        let mut backtester = Backtester::new();
        let mut programs = HashMap::new();

        // Entry: Close > BB_UPPER(20, 2) AND Close > DC_MIDDLE(20)
        programs.insert(
            "entry".to_string(),
            vec![
                Op::PushPrice(PriceType::Close),
                Op::PushIndicator(IndicatorType::BbUpper(20, 2)),
                Op::GreaterThan,
                Op::PushPrice(PriceType::Close),
                Op::PushIndicator(IndicatorType::DcMiddle(20)),
                Op::GreaterThan,
                Op::And,
            ],
        );

        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy, 0.0, 10000.0, 252.0, 0.0, 0.0);

        // We just want to ensure no runtime errors occurred during indicator evaluation
        assert_eq!(result.entry_error_count, 0);
        assert_eq!(result.exit_error_count, 0);
    }
}
