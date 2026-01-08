use crate::data::OHLCV;
use crate::vm::op::{DynamicConstant, IndicatorType, Op, PriceType};
use std::collections::HashMap;

const STACK_CAPACITY: usize = 256;
const MEMORY_SIZE: usize = 16;

pub struct VmContext {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub indicators: HashMap<IndicatorType, f64>,
    /// Historical OHLCV data for the last N periods (including current).
    /// Index 0 is current candle, index 1 is previous candle, etc.
    /// Length should be at least max(required historical lookback).
    pub history: Vec<OHLCV>,
}

#[derive(Debug, PartialEq)]
pub enum VmError {
    StackOverflow,
    StackUnderflow,
    MemoryOutOfBounds(u8),
    InvalidProgram,
}

pub struct VirtualMachine {
    stack: Vec<f64>,
    memory: [f64; MEMORY_SIZE],
}

impl Default for VirtualMachine {
    fn default() -> Self {
        Self::new()
    }
}
impl VirtualMachine {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(STACK_CAPACITY),
            memory: [0.0; MEMORY_SIZE],
        }
    }

    #[cfg(test)]
    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }

    pub fn execute(&mut self, program: &[Op], context: &VmContext) -> Result<f64, VmError> {
        self.stack.clear();
        let mut pc = 0;

        while pc < program.len() {
            let op = &program[pc];
            pc += 1;
            match op {
                Op::EntryMarker | Op::ExitMarker => {
                    // Does nothing, only used for delimitation
                }
                Op::PushConstant(val) => self.push(*val)?,
                Op::PushPrice(price_type) => {
                    let val = match price_type {
                        PriceType::Open => context.open,
                        PriceType::High => context.high,
                        PriceType::Low => context.low,
                        PriceType::Close => context.close,
                    };
                    self.push(val)?;
                }
                Op::PushDynamic(dynamic_const) => {
                    let base_value = match dynamic_const {
                        DynamicConstant::ClosePercent(_) => context.close,
                        DynamicConstant::SmaPercent(period, _) => context
                            .indicators
                            .get(&IndicatorType::Sma(*period))
                            .copied()
                            .unwrap_or(f64::NAN),
                    };

                    let calculated_value = match dynamic_const {
                        DynamicConstant::ClosePercent(pct) => {
                            base_value * (1.0 + (*pct as f64 / 100.0))
                        }
                        DynamicConstant::SmaPercent(_, pct) => {
                            base_value * (1.0 + (*pct as f64 / 100.0))
                        }
                    };
                    self.push(calculated_value)?;
                }
                Op::Store(idx) => {
                    let val = self.pop()?;
                    if let Some(mem_slot) = self.memory.get_mut(*idx as usize) {
                        *mem_slot = val;
                    } else {
                        return Err(VmError::MemoryOutOfBounds(*idx));
                    }
                }
                Op::Load(idx) => {
                    if let Some(val) = self.memory.get(*idx as usize) {
                        self.push(*val)?;
                    } else {
                        return Err(VmError::MemoryOutOfBounds(*idx));
                    }
                }
                Op::Add => self.apply_binary_op(|a, b| a + b)?,
                Op::Subtract => self.apply_binary_op(|a, b| a - b)?,
                Op::Multiply => self.apply_binary_op(|a, b| a * b)?,
                Op::Divide => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    if b.abs() < 1e-9 {
                        self.push(0.0)?;
                    } else {
                        self.push(a / b)?;
                    }
                }
                Op::GreaterThan => self.apply_binary_op(|a, b| (a > b) as i32 as f64)?,
                Op::LessThan => self.apply_binary_op(|a, b| (a < b) as i32 as f64)?,
                Op::GreaterThanOrEqual => self.apply_binary_op(|a, b| (a >= b) as i32 as f64)?,
                Op::LessThanOrEqual => self.apply_binary_op(|a, b| (a <= b) as i32 as f64)?,
                Op::Equal => self.apply_binary_op(|a, b| (a == b) as i32 as f64)?,
                Op::And => self.apply_binary_op(|a, b| ((a > 0.0) && (b > 0.0)) as i32 as f64)?,
                Op::Or => self.apply_binary_op(|a, b| ((a > 0.0) || (b > 0.0)) as i32 as f64)?,
                Op::Not => {
                    let val = self.pop()?;
                    self.push((val == 0.0) as i32 as f64)?;
                }
                Op::PushIndicator(indicator_type) => {
                    let val = context
                        .indicators
                        .get(indicator_type)
                        .copied()
                        .unwrap_or(f64::NAN);
                    self.push(val)?;
                }
                Op::PushPrevious(price_type, offset) => {
                    let val = if *offset == 0 {
                        match price_type {
                            PriceType::Open => context.open,
                            PriceType::High => context.high,
                            PriceType::Low => context.low,
                            PriceType::Close => context.close,
                        }
                    } else if let Some(candle) = context.history.get(*offset as usize) {
                        match price_type {
                            PriceType::Open => candle.open,
                            PriceType::High => candle.high,
                            PriceType::Low => candle.low,
                            PriceType::Close => candle.close,
                        }
                    } else {
                        f64::NAN
                    };
                    self.push(val)?;
                }
                Op::PushRollingSum(price_type, period) => {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for i in 0..*period {
                        if let Some(candle) = context.history.get(i as usize) {
                            sum += match price_type {
                                PriceType::Open => candle.open,
                                PriceType::High => candle.high,
                                PriceType::Low => candle.low,
                                PriceType::Close => candle.close,
                            };
                            count += 1;
                        }
                    }
                    // If we don't have enough history, return NaN
                    let val = if count >= *period as usize { sum } else { f64::NAN };
                    self.push(val)?;
                }
                Op::PushRollingMean(price_type, period) => {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for i in 0..*period {
                        if let Some(candle) = context.history.get(i as usize) {
                            sum += match price_type {
                                PriceType::Open => candle.open,
                                PriceType::High => candle.high,
                                PriceType::Low => candle.low,
                                PriceType::Close => candle.close,
                            };
                            count += 1;
                        }
                    }
                    // If we don't have enough history, return NaN
                    let val = if count >= *period as usize { sum / (*period as f64) } else { f64::NAN };
                    self.push(val)?;
                }
                Op::JumpIfFalse(target) => {
                    let condition = self.pop()?;
                    if condition == 0.0 {
                        // target is absolute program index
                        pc = *target;
                    }
                }
                Op::Jump(target) => {
                    // unconditional jump
                    pc = *target;
                }
                Op::Return => break,
            }
        }
        self.pop().or(Err(VmError::InvalidProgram))
    }

    #[inline]
    fn push(&mut self, val: f64) -> Result<(), VmError> {
        if self.stack.len() < STACK_CAPACITY {
            self.stack.push(val);
            Ok(())
        } else {
            Err(VmError::StackOverflow)
        }
    }

    #[inline]
    fn pop(&mut self) -> Result<f64, VmError> {
        self.stack.pop().ok_or(VmError::StackUnderflow)
    }

    #[inline]
    fn apply_binary_op<F>(&mut self, op: F) -> Result<(), VmError>
    where
        F: Fn(f64, f64) -> f64,
    {
        let b = self.pop()?;
        let a = self.pop()?;
        self.push(op(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::op::Op::*;

    fn dummy_context() -> VmContext {
        VmContext {
            open: 95.0,
            high: 105.0,
            low: 90.0,
            close: 100.0,
            indicators: HashMap::new(),
            history: vec![
                OHLCV {
                    timestamp: 0,
                    open: 95.0,
                    high: 105.0,
                    low: 90.0,
                    close: 100.0,
                    volume: 0.0,
                },
            ],
        }
    }

    #[test]
    fn test_basic_arithmetic() {
        let mut vm = VirtualMachine::new();
        let program = vec![PushConstant(5.0), PushConstant(10.0), Add];
        assert_eq!(vm.execute(&program, &dummy_context()).unwrap(), 15.0);
    }

    #[test]
    fn test_rpn_expression() {
        let mut vm = VirtualMachine::new();
        let program = vec![
            PushConstant(5.0),
            PushConstant(10.0),
            Add,
            PushConstant(2.0),
            Multiply,
        ];
        assert_eq!(vm.execute(&program, &dummy_context()).unwrap(), 30.0);
    }

    #[test]
    fn test_division_by_zero_safety() {
        let mut vm = VirtualMachine::new();
        let program = vec![PushConstant(10.0), PushConstant(0.0), Divide];
        assert_eq!(vm.execute(&program, &dummy_context()).unwrap(), 0.0);
    }

    #[test]
    fn test_comparison_operators() {
        let mut vm = VirtualMachine::new();
        let context = dummy_context();

        let program_gt_false = vec![PushConstant(5.0), PushConstant(10.0), GreaterThan];
        assert_eq!(vm.execute(&program_gt_false, &context).unwrap(), 0.0);

        let program_gt_true = vec![PushConstant(10.0), PushConstant(5.0), GreaterThan];
        assert_eq!(vm.execute(&program_gt_true, &context).unwrap(), 1.0);

        let program_eq_true = vec![PushConstant(10.0), PushConstant(10.0), Equal];
        assert_eq!(vm.execute(&program_eq_true, &context).unwrap(), 1.0);
    }

    #[test]
    fn test_logical_operators() {
        let mut vm = VirtualMachine::new();
        let context = dummy_context();

        let program = vec![PushConstant(1.0), PushConstant(0.0), And];
        assert_eq!(vm.execute(&program, &context).unwrap(), 0.0);

        let program_not = vec![PushConstant(1.0), Not];
        assert_eq!(vm.execute(&program_not, &context).unwrap(), 0.0);
    }

    #[test]
    fn test_state_management_store_and_load() {
        let mut vm = VirtualMachine::new();
        let program = vec![
            PushConstant(99.0),
            Store(5),
            Load(5),
            PushConstant(1.0),
            Add,
        ];
        assert_eq!(vm.execute(&program, &dummy_context()).unwrap(), 100.0);
        assert_eq!(vm.memory[5], 99.0);
    }

    #[test]
    fn test_stack_underflow() {
        let mut vm = VirtualMachine::new();
        let program = vec![Add];
        assert_eq!(
            vm.execute(&program, &dummy_context()),
            Err(VmError::StackUnderflow)
        );
    }

    #[test]
    fn test_memory_out_of_bounds() {
        let mut vm = VirtualMachine::new();
        let program = vec![PushConstant(1.0), Store(MEMORY_SIZE as u8)];
        assert_eq!(
            vm.execute(&program, &dummy_context()),
            Err(VmError::MemoryOutOfBounds(MEMORY_SIZE as u8))
        );
    }

    #[test]
    fn test_stack_overflow() {
        let mut vm = VirtualMachine::new();
        let mut program = vec![];
        for i in 0..257 {
            // Exceeds STACK_CAPACITY
            program.push(PushConstant(i as f64));
        }
        assert_eq!(
            vm.execute(&program, &dummy_context()),
            Err(VmError::StackOverflow)
        );
    }

    #[test]
    fn test_invalid_program_empty() {
        let mut vm = VirtualMachine::new();
        let program = vec![];
        assert_eq!(
            vm.execute(&program, &dummy_context()),
            Err(VmError::InvalidProgram)
        );
    }

    #[test]
    fn test_stack_isolation_between_executions() {
        let mut vm = VirtualMachine::new();
        let context = dummy_context();

        let prog1 = vec![PushConstant(5.0), PushConstant(99.0)];
        assert_eq!(vm.execute(&prog1, &context).unwrap(), 99.0);

        let prog2 = vec![PushConstant(1.0)];
        assert_eq!(vm.execute(&prog2, &context).unwrap(), 1.0);
    }

    #[test]
    fn test_price_access() {
        let mut vm = VirtualMachine::new();
        let context = VmContext {
            open: 95.0,
            high: 105.0,
            low: 90.0,
            close: 100.0,
            indicators: HashMap::new(),
            history: vec![
                OHLCV {
                    timestamp: 0,
                    open: 95.0,
                    high: 105.0,
                    low: 90.0,
                    close: 100.0,
                    volume: 0.0,
                },
            ],
        };

        // Test accessing close price
        let program_close = vec![PushPrice(PriceType::Close)];
        assert_eq!(vm.execute(&program_close, &context).unwrap(), 100.0);

        // Test accessing open price
        let program_open = vec![PushPrice(PriceType::Open)];
        assert_eq!(vm.execute(&program_open, &context).unwrap(), 95.0);

        // Test accessing high price
        let program_high = vec![PushPrice(PriceType::High)];
        assert_eq!(vm.execute(&program_high, &context).unwrap(), 105.0);

        // Test accessing low price
        let program_low = vec![PushPrice(PriceType::Low)];
        assert_eq!(vm.execute(&program_low, &context).unwrap(), 90.0);
    }

    #[test]
    fn test_ohlc_calculation() {
        let mut vm = VirtualMachine::new();
        let context = VmContext {
            open: 95.0,
            high: 105.0,
            low: 90.0,
            close: 100.0,
            indicators: HashMap::new(),
            history: vec![
                OHLCV {
                    timestamp: 0,
                    open: 95.0,
                    high: 105.0,
                    low: 90.0,
                    close: 100.0,
                    volume: 0.0,
                },
            ],
        };

        // Calculate typical price: (high + low + close) / 3
        let program = vec![
            PushPrice(PriceType::High),
            PushPrice(PriceType::Low),
            Add,
            PushPrice(PriceType::Close),
            Add,
            PushConstant(3.0),
            Divide,
        ];

        let result = vm.execute(&program, &context).unwrap();
        let expected = (105.0 + 90.0 + 100.0) / 3.0;
        assert!((result - expected).abs() < 1e-9);
    }

    #[test]
    fn test_return_statement() {
        let mut vm = VirtualMachine::new();
        let context = dummy_context();

        // Program with return in the middle - should stop execution
        let program = vec![
            PushConstant(42.0),
            Return,
            PushConstant(99.0), // This should not execute
        ];

        assert_eq!(vm.execute(&program, &context).unwrap(), 42.0);
    }

    #[test]
    fn test_memory_persistence_across_operations() {
        let mut vm = VirtualMachine::new();
        let context = dummy_context();

        // Store values in different memory locations
        let setup_program = vec![
            PushConstant(10.0),
            Store(0),
            PushConstant(20.0),
            Store(1),
            PushConstant(30.0),
            Store(2),
            PushConstant(0.0), // Final result placeholder
        ];
        vm.execute(&setup_program, &context).unwrap();

        // Load and compute with stored values
        let compute_program = vec![Load(0), Load(1), Add, Load(2), Multiply];

        let result = vm.execute(&compute_program, &context).unwrap();
        assert_eq!(result, (10.0 + 20.0) * 30.0);
    }

    #[test]
    fn test_push_dynamic_constant() {
        let mut vm = VirtualMachine::new();
        let mut context = VmContext {
            open: 0.0,
            high: 0.0,
            low: 0.0,
            close: 20000.0,
            indicators: HashMap::new(),
            history: vec![
                OHLCV {
                    timestamp: 0,
                    open: 0.0,
                    high: 0.0,
                    low: 0.0,
                    close: 20000.0,
                    volume: 0.0,
                },
            ],
        };
        context.indicators.insert(IndicatorType::Sma(20), 19500.0);

        // Test CLOSE +1%
        let prog1 = vec![PushDynamic(DynamicConstant::ClosePercent(1))];
        assert_eq!(vm.execute(&prog1, &context).unwrap(), 20200.0);

        // Test CLOSE -2%
        let prog2 = vec![PushDynamic(DynamicConstant::ClosePercent(-2))];
        assert_eq!(vm.execute(&prog2, &context).unwrap(), 19600.0);

        // Test SMA(20) +2%
        let prog3 = vec![PushDynamic(DynamicConstant::SmaPercent(20, 2))];
        assert!((vm.execute(&prog3, &context).unwrap() - 19890.0).abs() < 1e-9);
    }

    #[test]
    fn test_jump_debug() {
        let mut vm = VirtualMachine::new();
        let context = dummy_context();
        // Simple program: push 5, return
        let program = vec![PushConstant(5.0), Return];
        let result = vm.execute(&program, &context);
        dbg!(&result);
        assert_eq!(result.unwrap(), 5.0);
        // Program with jump over push 10
        let program2 = vec![
            PushConstant(5.0),
            Jump(3), // should jump to index 3 (PushConstant(20))
            PushConstant(10.0),
            PushConstant(20.0),
            Return,
        ];
        let result2 = vm.execute(&program2, &context);
        dbg!(&result2);
        assert_eq!(result2.unwrap(), 20.0);
        // JumpIfFalse false condition
        let program3 = vec![
            PushConstant(0.0),
            JumpIfFalse(3),
            PushConstant(30.0),
            PushConstant(40.0),
            Return,
        ];
        let result3 = vm.execute(&program3, &context);
        dbg!(&result3);
        assert_eq!(result3.unwrap(), 40.0);
        // JumpIfFalse true condition
        let program4 = vec![
            PushConstant(1.0),
            JumpIfFalse(3),
            PushConstant(50.0),
            PushConstant(60.0),
            Return,
        ];
        let result4 = vm.execute(&program4, &context);
        dbg!(&result4);
        assert_eq!(result4.unwrap(), 60.0);
    }

    #[test]
    fn test_jump_and_jump_if_false() {
        let mut vm = VirtualMachine::new();
        let context = dummy_context();

        // Test unconditional jump
        // Program: push 5, jump over push 10, push 20, return
        // Expected result: 20
        let program = vec![
            PushConstant(5.0),
            Jump(3), // jump to index 3 (PushConstant(20.0))
            PushConstant(10.0), // skipped
            PushConstant(20.0),
            Return,
        ];
        let result = vm.execute(&program, &context);
        dbg!(&result);
        assert_eq!(result.unwrap(), 20.0);

        // Test JumpIfFalse with false condition (0.0)
        // Program: push 0, jump if false over push 30, push 40, return
        // Expected: 40
        let program2 = vec![
            PushConstant(0.0),
            JumpIfFalse(3),
            PushConstant(30.0), // skipped
            PushConstant(40.0),
            Return,
        ];
        let result2 = vm.execute(&program2, &context);
        dbg!(&result2);
        assert_eq!(result2.unwrap(), 40.0);

        // Test JumpIfFalse with true condition (1.0)
        // Program: push 1, jump if false over push 50, push 60, return
        // Expected: 60 (jump not taken)
        let program3 = vec![
            PushConstant(1.0),
            JumpIfFalse(3),
            PushConstant(50.0), // executed
            PushConstant(60.0),
            Return,
        ];
        let result3 = vm.execute(&program3, &context);
        dbg!(&result3);
        assert_eq!(result3.unwrap(), 60.0);

        // Test nested jumps
        let program4 = vec![
            PushConstant(0.0), // condition false
            JumpIfFalse(4), // if false jump to push 100
            PushConstant(200.0), // true branch
            Jump(5), // skip false branch
            PushConstant(100.0), // false branch
            Return,
        ];
        let result4 = vm.execute(&program4, &context);
        dbg!(&result4);
        assert_eq!(result4.unwrap(), 100.0);
    }

    #[test]
    fn test_historical_operations() {
        use crate::vm::op::Op::*;
        use crate::vm::op::PriceType::*;
        
        // Create a context with 5 candles of history
        let mut vm = VirtualMachine::new();
        let context = VmContext {
            open: 50.0,
            high: 55.0,
            low: 45.0,
            close: 52.0,
            indicators: HashMap::new(),
            history: vec![
                OHLCV { timestamp: 4, open: 50.0, high: 55.0, low: 45.0, close: 52.0, volume: 0.0 }, // current (index 0)
                OHLCV { timestamp: 3, open: 48.0, high: 53.0, low: 43.0, close: 50.0, volume: 0.0 }, // previous 1
                OHLCV { timestamp: 2, open: 46.0, high: 51.0, low: 41.0, close: 48.0, volume: 0.0 }, // previous 2
                OHLCV { timestamp: 1, open: 44.0, high: 49.0, low: 39.0, close: 46.0, volume: 0.0 }, // previous 3
                OHLCV { timestamp: 0, open: 42.0, high: 47.0, low: 37.0, close: 44.0, volume: 0.0 }, // previous 4
            ],
        };

        // Test PushPrevious with offset 0 (current close)
        let program = vec![PushPrevious(Close, 0)];
        assert_eq!(vm.execute(&program, &context).unwrap(), 52.0);

        // Test PushPrevious with offset 1 (previous close)
        let program = vec![PushPrevious(Close, 1)];
        assert_eq!(vm.execute(&program, &context).unwrap(), 50.0);

        // Test PushPrevious with offset 4 (oldest close)
        let program = vec![PushPrevious(Close, 4)];
        assert_eq!(vm.execute(&program, &context).unwrap(), 44.0);

        // Test PushPrevious with offset 5 (beyond history) -> NaN
        let program = vec![PushPrevious(Close, 5)];
        let result = vm.execute(&program, &context).unwrap();
        assert!(result.is_nan(), "Expected NaN for offset beyond history, got {}", result);

        // Test PushRollingSum with period 3 (sum of last 3 closes: 52 + 50 + 48 = 150)
        let program = vec![PushRollingSum(Close, 3)];
        assert_eq!(vm.execute(&program, &context).unwrap(), 150.0);

        // Test PushRollingSum with period 5 (sum of all 5 closes: 52+50+48+46+44 = 240)
        let program = vec![PushRollingSum(Close, 5)];
        assert_eq!(vm.execute(&program, &context).unwrap(), 240.0);

        // Test PushRollingSum with period 6 (beyond history) -> NaN
        let program = vec![PushRollingSum(Close, 6)];
        let result = vm.execute(&program, &context).unwrap();
        assert!(result.is_nan(), "Expected NaN for period beyond history, got {}", result);

        // Test PushRollingMean with period 3 (mean of last 3 closes: 150/3 = 50)
        let program = vec![PushRollingMean(Close, 3)];
        assert_eq!(vm.execute(&program, &context).unwrap(), 50.0);

        // Test PushRollingMean with period 5 (mean of all 5 closes: 240/5 = 48)
        let program = vec![PushRollingMean(Close, 5)];
        assert_eq!(vm.execute(&program, &context).unwrap(), 48.0);

        // Test different price types (Open)
        let program = vec![PushPrevious(Open, 1)];
        assert_eq!(vm.execute(&program, &context).unwrap(), 48.0);

        // Test rolling sum with Open
        let program = vec![PushRollingSum(Open, 2)]; // 50 + 48 = 98
        assert_eq!(vm.execute(&program, &context).unwrap(), 98.0);
    }
}
