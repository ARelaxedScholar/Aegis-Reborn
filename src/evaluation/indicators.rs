use crate::data::OHLCV;
use crate::vm::engine::VmContext;
use crate::vm::op::IndicatorType;
use std::collections::HashMap;
use ta::indicators::{ExponentialMovingAverage, RelativeStrengthIndex, SimpleMovingAverage};
use ta::Next;

/// Trait for technical indicators that can be updated with new price data.
pub trait Indicator: Send + Sync {
    /// Update the indicator with the next closing price and return the new value.
    fn next(&mut self, close: f64) -> f64;
    /// Get the current value of the indicator (after the last call to `next`).
    fn current(&self) -> f64;
}

/// Wrapper for Simple Moving Average.
pub struct SmaIndicator {
    inner: SimpleMovingAverage,
    last_value: f64,
}

impl SmaIndicator {
    pub fn new(period: u16) -> Self {
        let inner = SimpleMovingAverage::new(period.into()).unwrap();
        Self {
            inner,
            last_value: 0.0,
        }
    }
}

impl Indicator for SmaIndicator {
    fn next(&mut self, close: f64) -> f64 {
        self.last_value = self.inner.next(close);
        self.last_value
    }

    fn current(&self) -> f64 {
        self.last_value
    }
}

/// Wrapper for Relative Strength Index.
pub struct RsiIndicator {
    inner: RelativeStrengthIndex,
    last_value: f64,
}

impl RsiIndicator {
    pub fn new(period: u16) -> Self {
        let inner = RelativeStrengthIndex::new(period.into()).unwrap();
        Self {
            inner,
            last_value: 0.0,
        }
    }
}

impl Indicator for RsiIndicator {
    fn next(&mut self, close: f64) -> f64 {
        self.last_value = self.inner.next(close);
        self.last_value
    }

    fn current(&self) -> f64 {
        self.last_value
    }
}

/// Wrapper for Exponential Moving Average.
pub struct EmaIndicator {
    inner: ExponentialMovingAverage,
    last_value: f64,
}

impl EmaIndicator {
    pub fn new(period: u16) -> Self {
        let inner = ExponentialMovingAverage::new(period.into()).unwrap();
        Self {
            inner,
            last_value: 0.0,
        }
    }
}

impl Indicator for EmaIndicator {
    fn next(&mut self, close: f64) -> f64 {
        self.last_value = self.inner.next(close);
        self.last_value
    }

    fn current(&self) -> f64 {
        self.last_value
    }
}



/// Manages the state and calculation of all required indicators for a backtest.
/// This acts as an abstraction layer over the `ta` crate.
pub struct IndicatorManager {
    indicators: HashMap<IndicatorType, Box<dyn Indicator>>,
    last_values: HashMap<IndicatorType, f64>,
}

impl IndicatorManager {
    /// Creates a new manager for a specific, pre-analyzed set of indicators.
    pub fn new(required_indicators: &[IndicatorType]) -> Self {
        let mut indicators = HashMap::new();
        let mut last_values = HashMap::new();

        for indicator_type in required_indicators {
            let indicator: Box<dyn Indicator> = match indicator_type {
                IndicatorType::Sma(period) => Box::new(SmaIndicator::new(*period)),
                IndicatorType::Rsi(period) => Box::new(RsiIndicator::new(*period)),
                IndicatorType::Ema(period) => Box::new(EmaIndicator::new(*period)),
            };
            indicators.insert(indicator_type.clone(), indicator);
            last_values.insert(indicator_type.clone(), 0.0);
        }

        Self {
            indicators,
            last_values,
        }
    }

    /// Updates all managed indicators with the next candle's data and stores the results.
    pub fn next(&mut self, candle: &OHLCV) {
        for (indicator_type, indicator) in self.indicators.iter_mut() {
            let val = indicator.next(candle.close);
            self.last_values.insert(indicator_type.clone(), val);
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
