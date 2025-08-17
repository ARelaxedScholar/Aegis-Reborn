use crate::data::OHLCV;
use crate::evaluation::indicators::IndicatorManager;
use crate::strategy::Strategy;
use crate::vm::engine::{VirtualMachine, VmContext};
use crate::vm::op::{IndicatorType, Op};
use log::warn;
use std::collections::{HashMap, HashSet};

const INITIAL_CASH: f64 = 10_000.0;

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
}
impl Portfolio {
    fn new() -> Self {
        Self {
            cash: INITIAL_CASH,
            position_size: 0.0,
            state: PositionState::Flat,
        }
    }
}

/// The complete result of a backtest run, with enhanced error tracking.
#[derive(Debug, Default, PartialEq)]
pub struct BacktestResult {
    pub final_equity: f64,
    pub entry_error_count: u32,
    pub exit_error_count: u32,
}

pub struct Backtester {
    vm: VirtualMachine,
}

/// Scans all programs in a strategy to find which indicators are required.
fn get_required_indicators(strategy: &Strategy) -> Vec<IndicatorType> {
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

impl Backtester {
    pub fn new() -> Self {
        Self {
            vm: VirtualMachine::new(),
        }
    }

    pub fn run(&mut self, candles: &[OHLCV], strategy: &Strategy) -> BacktestResult {
        let mut portfolio = Portfolio::new();
        let mut entry_error_count = 0;
        let mut exit_error_count = 0;

        if candles.is_empty() {
            return BacktestResult {
                final_equity: INITIAL_CASH,
                ..Default::default()
            };
        }

        let required_indicators = get_required_indicators(strategy);
        let mut indicator_manager = IndicatorManager::new(&required_indicators);

        for (i, candle) in candles.iter().enumerate() {
            indicator_manager.next(candle);

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
                            let position_cost = portfolio.cash;
                            portfolio.position_size = position_cost / candle.close;
                            portfolio.cash = 0.0;
                            portfolio.state = PositionState::Long;
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

        if let Some(last_candle) = candles.last() {
            if portfolio.state == PositionState::Long {
                portfolio.cash = portfolio.position_size * last_candle.close;
                portfolio.position_size = 0.0;
            }
        }

        BacktestResult {
            final_equity: portfolio.cash,
            entry_error_count,
            exit_error_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::op::Op::*;
    use std::collections::HashMap;

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
        // No exit program
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy);
        // Expected: Buys at 100, holds at 110, liquidated at 110 -> 11k final equity.
        assert_eq!(result.final_equity, 11_000.0);
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
        // No entry program
        let strategy = Strategy { programs };

        let result = backtester.run(&candles, &strategy);
        // Expected: Never enters, so cash remains initial.
        assert_eq!(result.final_equity, INITIAL_CASH);
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

        let result = backtester.run(&candles, &strategy);
        assert_eq!(result.final_equity, INITIAL_CASH);
        assert_eq!(result.entry_error_count, 1);
        assert_eq!(result.exit_error_count, 0);
    }
}
