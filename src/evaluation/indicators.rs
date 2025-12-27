use crate::data::OHLCV;
use crate::vm::engine::VmContext;
use crate::vm::op::IndicatorType;
use rayon::prelude::*;
use std::collections::HashMap;
use ta::indicators::{RelativeStrengthIndex, SimpleMovingAverage};
use ta::Next;

/// Manages the state and calculation of all required indicators for a backtest.
/// This acts as an abstraction layer over the `ta` crate.
pub struct IndicatorManager {
    smas: HashMap<u16, SimpleMovingAverage>,
    rsis: HashMap<u16, RelativeStrengthIndex>,
    last_values: HashMap<IndicatorType, f64>,
}

impl IndicatorManager {
    /// Creates a new manager for a specific, pre-analyzed set of indicators.
    pub fn new(required_indicators: &[IndicatorType]) -> Self {
        let mut smas = HashMap::new();
        let mut rsis = HashMap::new();

        for indicator in required_indicators {
            match *indicator {
                IndicatorType::Sma(period) => {
                    smas.entry(period)
                        .or_insert_with(|| SimpleMovingAverage::new(period.into()).unwrap());
                }
                IndicatorType::Rsi(period) => {
                    rsis.entry(period)
                        .or_insert_with(|| RelativeStrengthIndex::new(period.into()).unwrap());
                }
            }
        }
        Self {
            smas,
            rsis,
            last_values: HashMap::new(),
        }
    }

    /// Updates all managed indicators with the next candle's data and stores the results.
    pub fn next(&mut self, candle: &OHLCV) {
        let close = candle.close;
        
        // Process SMAs and RSIs in parallel using rayon::join
        let (sma_results, rsi_results) = rayon::join(
            || {
                self.smas
                    .par_iter_mut()
                    .map(|(period, sma)| (IndicatorType::Sma(*period), sma.next(close)))
                    .collect::<Vec<_>>()
            },
            || {
                self.rsis
                    .par_iter_mut()
                    .map(|(period, rsi)| (IndicatorType::Rsi(*period), rsi.next(close)))
                    .collect::<Vec<_>>()
            },
        );
        
        // Insert all results into the shared HashMap
        self.last_values.extend(sma_results);
        self.last_values.extend(rsi_results);
    }

    /// Fills the VmContext with the latest calculated indicator values.
    /// Values may be NaN during the indicator's warmup period.
    pub fn populate_context(&self, context: &mut VmContext) {
        for (indicator_type, value) in &self.last_values {
            context.indicators.insert(indicator_type.clone(), *value);
        }
    }
}
