use crate::data::OHLCV;
use crate::vm::engine::VmContext;
use crate::vm::op::IndicatorType;
use std::collections::HashMap;
use ta::indicators::{
    BollingerBands, ExponentialMovingAverage, RelativeStrengthIndex, SimpleMovingAverage,
};
use ta::Next;

/// Simple Donchian Channel implementation
struct DonchianChannel {
    period: usize,
    history: std::collections::VecDeque<f64>,
}

impl DonchianChannel {
    fn new(period: usize) -> Self {
        Self {
            period,
            history: std::collections::VecDeque::with_capacity(period),
        }
    }

    fn next(&mut self, val: f64) -> (f64, f64, f64) {
        if self.history.len() >= self.period {
            self.history.pop_front();
        }
        self.history.push_back(val);

        let max = self.history.iter().fold(f64::MIN, |a, &b| a.max(b));
        let min = self.history.iter().fold(f64::MAX, |a, &b| a.min(b));
        let mid = (max + min) / 2.0;
        (max, min, mid)
    }
}

/// Manages the state and calculation of all required indicators for a backtest.
/// Optimized to share indicator instances (e.g. BB Upper/Lower share the same calculation).
pub struct IndicatorManager {
    smas: HashMap<u16, SimpleMovingAverage>,
    emas: HashMap<u16, ExponentialMovingAverage>,
    rsis: HashMap<u16, RelativeStrengthIndex>,
    bbs: HashMap<(u16, u8), BollingerBands>,
    dcs: HashMap<u16, DonchianChannel>,
    last_values: HashMap<IndicatorType, f64>,
}

impl IndicatorManager {
    /// Creates a new manager for a specific, pre-analyzed set of indicators.
    pub fn new(required_indicators: &[IndicatorType]) -> Self {
        let mut smas = HashMap::new();
        let mut emas = HashMap::new();
        let mut rsis = HashMap::new();
        let mut bbs = HashMap::new();
        let mut dcs = HashMap::new();
        let mut last_values = HashMap::new();

        for ind in required_indicators {
            match ind {
                IndicatorType::Sma(p) => {
                    smas.entry(*p)
                        .or_insert_with(|| SimpleMovingAverage::new(*p as usize).unwrap());
                }
                IndicatorType::Ema(p) => {
                    emas.entry(*p)
                        .or_insert_with(|| ExponentialMovingAverage::new(*p as usize).unwrap());
                }
                IndicatorType::Rsi(p) => {
                    rsis.entry(*p)
                        .or_insert_with(|| RelativeStrengthIndex::new(*p as usize).unwrap());
                }
                IndicatorType::BbUpper(p, s) | IndicatorType::BbLower(p, s) => {
                    bbs.entry((*p, *s))
                        .or_insert_with(|| BollingerBands::new(*p as usize, *s as f64).unwrap());
                }
                IndicatorType::DcUpper(p)
                | IndicatorType::DcLower(p)
                | IndicatorType::DcMiddle(p) => {
                    dcs.entry(*p)
                        .or_insert_with(|| DonchianChannel::new(*p as usize));
                }
            }
            last_values.insert(ind.clone(), 0.0);
        }

        Self {
            smas,
            emas,
            rsis,
            bbs,
            dcs,
            last_values,
        }
    }

    /// Updates all managed indicators with the next candle's data and stores the results.
    pub fn next(&mut self, candle: &OHLCV) {
        let close = candle.close;

        for (p, sma) in &mut self.smas {
            let val = sma.next(close);
            self.last_values.insert(IndicatorType::Sma(*p), val);
        }
        for (p, ema) in &mut self.emas {
            let val = ema.next(close);
            self.last_values.insert(IndicatorType::Ema(*p), val);
        }
        for (p, rsi) in &mut self.rsis {
            let val = rsi.next(close);
            self.last_values.insert(IndicatorType::Rsi(*p), val);
        }
        for ((p, s), bb) in &mut self.bbs {
            let out = bb.next(close);
            self.last_values
                .insert(IndicatorType::BbUpper(*p, *s), out.upper);
            self.last_values
                .insert(IndicatorType::BbLower(*p, *s), out.lower);
        }
        for (p, dc) in &mut self.dcs {
            let (max, min, mid) = dc.next(close);
            self.last_values.insert(IndicatorType::DcUpper(*p), max);
            self.last_values.insert(IndicatorType::DcLower(*p), min);
            self.last_values.insert(IndicatorType::DcMiddle(*p), mid);
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
