use crate::data::OHLCV;
use crate::vm::engine::VmContext;
use crate::vm::op::PriceType;
use std::collections::{HashMap, VecDeque};

/// A rolling window that maintains the last N values and their sum with O(1) updates.
pub struct RollingWindow {
    values: VecDeque<f64>,
    sum: f64,
    capacity: usize,
}

impl RollingWindow {
    pub fn new(period: u16) -> Self {
        Self {
            values: VecDeque::with_capacity(period as usize),
            sum: 0.0,
            capacity: period as usize,
        }
    }

    /// Adds a new value to the window, removing the oldest if window is full.
    /// Returns the current sum after the update.
    pub fn push(&mut self, value: f64) -> f64 {
        self.values.push_front(value);
        self.sum += value;
        
        if self.values.len() > self.capacity {
            if let Some(removed) = self.values.pop_back() {
                self.sum -= removed;
            }
        }
        self.sum
    }

    /// Returns the current sum of values in the window.
    pub fn current_sum(&self) -> f64 {
        self.sum
    }

    /// Returns true if window has reached its capacity.
    pub fn is_full(&self) -> bool {
        self.values.len() >= self.capacity
    }

    /// Returns the number of values currently in the window.
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// Manages rolling sum calculations for multiple price types and periods.
pub struct RollingWindowManager {
    windows: HashMap<(PriceType, u16), RollingWindow>,
    last_values: HashMap<(PriceType, u16), f64>,
}

impl RollingWindowManager {
    /// Creates a new manager for a specific set of required rolling sums.
    pub fn new(required_sums: &[(PriceType, u16)]) -> Self {
        let mut windows = HashMap::new();
        let mut last_values = HashMap::new();

        for (price_type, period) in required_sums.iter() {
            windows.insert((price_type.clone(), *period), RollingWindow::new(*period));
            last_values.insert((price_type.clone(), *period), f64::NAN);
        }

        Self { windows, last_values }
    }

    /// Updates all managed rolling windows with the next candle's data.
    pub fn next(&mut self, candle: &OHLCV) {
        for ((price_type, period), window) in self.windows.iter_mut() {
            let price = match price_type {
                PriceType::Open => candle.open,
                PriceType::High => candle.high,
                PriceType::Low => candle.low,
                PriceType::Close => candle.close,
            };
            let sum = window.push(price);
            // Store NaN until window is full (matching VM behavior)
            let value = if window.is_full() { sum } else { f64::NAN };
            self.last_values.insert((price_type.clone(), *period), value);
        }
    }

    /// Fills the VmContext with the latest calculated rolling sum values.
    pub fn populate_context(&self, context: &mut VmContext) {
        for (key, value) in &self.last_values {
            context.rolling_sums.insert(key.clone(), *value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use crate::vm::engine::VmContext;
    
    #[test]
    fn test_rolling_window_basic() {
        let mut window = RollingWindow::new(3);
        assert_eq!(window.len(), 0);
        assert!(!window.is_full());
        
        window.push(1.0);
        assert_eq!(window.current_sum(), 1.0);
        assert_eq!(window.len(), 1);
        
        window.push(2.0);
        assert_eq!(window.current_sum(), 3.0);
        
        window.push(3.0);
        assert_eq!(window.current_sum(), 6.0);
        assert!(window.is_full());
        
        window.push(4.0);
        assert_eq!(window.current_sum(), 9.0); // 2+3+4
        assert_eq!(window.len(), 3);
    }
    
    #[test]
    fn test_rolling_window_manager() {
        use crate::vm::op::PriceType;
        
        let required_sums = vec![
            (PriceType::Close, 3),
            (PriceType::Open, 2),
        ];
        let mut manager = RollingWindowManager::new(&required_sums);
        
        // Create a simple candle
        let candle1 = OHLCV {
            timestamp: 0,
            open: 10.0,
            high: 15.0,
            low: 5.0,
            close: 12.0,
            volume: 100.0,
        };
        let candle2 = OHLCV {
            timestamp: 1,
            open: 11.0,
            high: 16.0,
            low: 6.0,
            close: 13.0,
            volume: 110.0,
        };
        let candle3 = OHLCV {
            timestamp: 2,
            open: 12.0,
            high: 17.0,
            low: 7.0,
            close: 14.0,
            volume: 120.0,
        };
        
        // Update with first candle
        manager.next(&candle1);
        // Windows not full yet, values should be NaN
        let mut context = VmContext {
            open: 0.0,
            high: 0.0,
            low: 0.0,
            close: 0.0,
            indicators: HashMap::new(),
            rolling_sums: HashMap::new(),
            history: VecDeque::new(),
        };
        manager.populate_context(&mut context);
        assert!(context.rolling_sums.get(&(PriceType::Close, 3)).unwrap().is_nan());
        assert!(context.rolling_sums.get(&(PriceType::Open, 2)).unwrap().is_nan());
        
        // Second candle
        manager.next(&candle2);
        manager.populate_context(&mut context);
        // Open window of period 2 should now be full: 10.0 + 11.0 = 21.0
        assert_eq!(context.rolling_sums.get(&(PriceType::Open, 2)).unwrap(), &21.0);
        // Close window period 3 still not full
        assert!(context.rolling_sums.get(&(PriceType::Close, 3)).unwrap().is_nan());
        
        // Third candle
        manager.next(&candle3);
        manager.populate_context(&mut context);
        // Close window now full: 12 + 13 + 14 = 39.0
        assert_eq!(context.rolling_sums.get(&(PriceType::Close, 3)).unwrap(), &39.0);
        // Open window now contains 11 + 12 = 23.0 (shifted)
        assert_eq!(context.rolling_sums.get(&(PriceType::Open, 2)).unwrap(), &23.0);
    }
}