use crate::data::OHLCV;
use crate::vm::engine::VmContext;
use crate::vm::op::IndicatorType;
use std::collections::HashMap;
use ta::Next;
use ta::indicators::{RelativeStrengthIndex, SimpleMovingAverage};

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
        for (period, sma) in self.smas.iter_mut() {
            let val = sma.next(candle.close);
            self.last_values.insert(IndicatorType::Sma(*period), val);
        }
        for (period, rsi) in self.rsis.iter_mut() {
            let val = rsi.next(candle.close);
            self.last_values.insert(IndicatorType::Rsi(*period), val);
        }
    }

    /// Fills the VmContext with the latest calculated indicator values.
    /// Values may be NaN during the indicator's warmup period.
    pub fn populate_context(&self, context: &mut VmContext) {
        for (indicator_type, value) in &self.last_values {
            context.indicators.insert(indicator_type.clone(), *value);
        }
    }
}
