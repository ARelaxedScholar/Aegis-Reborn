use crate::config::TranspilerConfig;
use crate::strategy::Strategy;
use crate::vm::op::{DynamicConstant, IndicatorType, Op, PriceType};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct TranspilerEngine {
    config: TranspilerConfig,
}

#[derive(Debug, thiserror::Error)]
pub enum TranspilerError {
    #[error("Unsupported opcode: {0}")]
    UnsupportedOp(String),
    #[error("Invalid jump target: {0}")]
    InvalidJumpTarget(usize),
    #[error("Stack underflow during transpilation")]
    StackUnderflow,
    #[error("Invalid program structure (missing entry program)")]
    InvalidProgram,
}

impl TranspilerEngine {
    /// Creates a new transpiler engine with the given configuration
    pub fn new(config: TranspilerConfig) -> Self {
        Self { config }
    }

    /// Transpiles a strategy to QuantConnect Python algorithm
    pub fn to_python(&self, strategy: &Strategy) -> Result<String, TranspilerError> {
        let dependencies = Self::analyze_dependencies(strategy);
        let entry_code = if let Some(entry_ops) = strategy.programs.get("entry") {
            self.compile_program_to_python(entry_ops, "entry")?
        } else {
            return Err(TranspilerError::InvalidProgram);
        };

        let exit_code = if let Some(exit_ops) = strategy.programs.get("exit") {
            Some(self.compile_program_to_python(exit_ops, "exit")?)
        } else {
            None
        };

        Ok(self.generate_python_template(&dependencies, &entry_code, exit_code.as_deref()))
    }

    /// Transpiles a strategy to QuantConnect C# algorithm
    pub fn to_c_sharp(&self, strategy: &Strategy) -> Result<String, TranspilerError> {
        let dependencies = Self::analyze_dependencies(strategy);
        let entry_code = if let Some(entry_ops) = strategy.programs.get("entry") {
            self.compile_program_to_csharp(entry_ops, "entry")?
        } else {
            return Err(TranspilerError::InvalidProgram);
        };

        let exit_code = if let Some(exit_ops) = strategy.programs.get("exit") {
            Some(self.compile_program_to_csharp(exit_ops, "exit")?)
        } else {
            None
        };

        Ok(self.generate_csharp_template(&dependencies, &entry_code, exit_code.as_deref()))
    }

    /// Analyzes a strategy and returns the set of indicator dependencies
    fn analyze_dependencies(strategy: &Strategy) -> HashSet<IndicatorType> {
        let mut indicators = HashSet::new();
        for ops in strategy.programs.values() {
            for op in ops {
                match op {
                    Op::PushIndicator(indicator_type) => {
                        indicators.insert(indicator_type.clone());
                    }
                    Op::PushDynamic(DynamicConstant::SmaPercent(period, _)) => {
                        indicators.insert(IndicatorType::Sma(*period));
                    }
                    _ => {}
                }
            }
        }
        indicators
    }

    /// Compiles a single program (entry or exit) to Python expression
    fn compile_program_to_python(&self, ops: &[Op], _program_name: &str) -> Result<String, TranspilerError> {
        let mut compiler = PythonCompiler::new();
        compiler.compile(ops)
    }

    /// Compiles a single program (entry or exit) to C# expression
    fn compile_program_to_csharp(&self, ops: &[Op], _program_name: &str) -> Result<String, TranspilerError> {
        let mut compiler = CSharpCompiler::new();
        compiler.compile(ops)
    }

    /// Generates the complete Python algorithm template
    fn generate_python_template(
        &self,
        dependencies: &HashSet<IndicatorType>,
        entry_code: &str,
        exit_code: Option<&str>,
    ) -> String {
        let indicators_init = self.generate_python_indicators_init(dependencies);
        let warm_up_period = self.calculate_warm_up_period(dependencies);

        format!(r#"from AlgorithmImports import *

class GoldenAegisAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_cash({initial_cash})
        self.set_brokerage_model(BrokerageName.InteractiveBrokersBrokerage)
        self._symbol = self.add_equity("{symbol}", Resolution.{resolution}).Symbol
        self.set_warm_up({warm_up_period}, Resolution.{resolution})

{indicators_init}

    def evaluate_entry(self, data):
        {entry_code}

    {exit_eval}

    def on_data(self, data):
        if self.is_warming_up:
            return

        if not self.Portfolio.invested and self.evaluate_entry(data):
            self.SetHoldings(self.symbol, 1.0)
        elif self.Portfolio.invested{exit_condition}:
            self.Liquidate(self.symbol)
"#,
            initial_cash = self.config.initial_cash.unwrap_or(10000.0),
            symbol = self.config.symbol,
            resolution = self.config.resolution,
            warm_up_period = warm_up_period,
            indicators_init = indicators_init,
            entry_code = entry_code,
            exit_eval = if let Some(exit_code) = exit_code {
                format!("def evaluate_exit(self, data):\n        {}", exit_code)
            } else {
                "".to_string()
            },
            exit_condition = if exit_code.is_some() {
                " and self.evaluate_exit(data)"
            } else {
                ""
            }
        )
    }

    /// Generates the complete C# algorithm template
    fn generate_csharp_template(
        &self,
        dependencies: &HashSet<IndicatorType>,
        entry_code: &str,
        exit_code: Option<&str>,
    ) -> String {
        let (indicator_fields, indicator_init) = self.generate_csharp_indicator_code(dependencies);
        let warm_up_period = self.calculate_warm_up_period(dependencies);

        format!(r#"using System;
using QuantConnect.Algorithm;
using QuantConnect.Data;
using QuantConnect.Indicators;

namespace QuantConnect.Algorithm.CSharp {{
    public class GoldenAegisAlgorithm : QCAlgorithm
    {{
        {indicator_fields}

        public override void Initialize()
        {{
            SetCash({initial_cash});
            SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage);
            symbol = AddEquity("{symbol}", Resolution.{resolution}).Symbol;
            SetWarmUp({warm_up_period}, Resolution.{resolution});

{indicator_init}
        }}

        private bool EvaluateEntry(Slice data)
        {{
            {entry_code}
        }}

        {exit_eval}

        public override void OnData(Slice data)
        {{
            if (IsWarmingUp) return;

            if (!Portfolio.Invested && EvaluateEntry(data))
            {{
                SetHoldings(symbol, 1.0);
            }}
            else if (Portfolio.Invested{exit_condition})
            {{
                Liquidate(symbol);
            }}
        }}
    }}
}}
"#,
            initial_cash = self.config.initial_cash.unwrap_or(10000.0),
            symbol = self.config.symbol,
            resolution = self.config.resolution,
            warm_up_period = warm_up_period,
            indicator_fields = indicator_fields,
            indicator_init = indicator_init,
            entry_code = entry_code,
            exit_eval = if let Some(exit_code) = exit_code {
                format!("private bool EvaluateExit(Slice data)\n        {{\n            {}\n        }}", exit_code)
            } else {
                "".to_string()
            },
            exit_condition = if exit_code.is_some() {
                " && EvaluateExit(data)"
            } else {
                ""
            }
        )
    }

    /// Generates Python indicator initialization code
    fn generate_python_indicators_init(&self, dependencies: &HashSet<IndicatorType>) -> String {
        let mut lines = Vec::new();
        for indicator in dependencies {
            match indicator {
                IndicatorType::Sma(period) => {
                    lines.push(format!("        self.sma{} = self.SMA(self.symbol, {})", period, period));
                }
                IndicatorType::Rsi(period) => {
                    lines.push(format!("        self.rsi{} = self.RSI(self.symbol, {})", period, period));
                }
                IndicatorType::Ema(period) => {
                    lines.push(format!("        self.ema{} = self.EMA(self.symbol, {})", period, period));
                }
            }
        }
        lines.join("\n")
    }

    /// Generates C# indicator fields and initialization code
    fn generate_csharp_indicator_code(&self, dependencies: &HashSet<IndicatorType>) -> (String, String) {
        let mut fields = Vec::new();
        let mut init_lines = Vec::new();

        fields.push("        private Symbol symbol;".to_string());

        for indicator in dependencies {
            match indicator {
                IndicatorType::Sma(period) => {
                    fields.push(format!("        private SimpleMovingAverage sma{};", period));
                    init_lines.push(format!("            sma{} = SMA(symbol, {});", period, period));
                }
                IndicatorType::Rsi(period) => {
                    fields.push(format!("        private RelativeStrengthIndex rsi{};", period));
                    init_lines.push(format!("            rsi{} = RSI(symbol, {});", period, period));
                }
                IndicatorType::Ema(period) => {
                    fields.push(format!("        private ExponentialMovingAverage ema{};", period));
                    init_lines.push(format!("            ema{} = EMA(symbol, {});", period, period));
                }
            }
        }

        if !fields.is_empty() && fields.len() > 1 {
            fields.insert(1, "".to_string()); // Empty line after symbol field
        }

        let fields_str = fields.join("\n");
        let init_str = init_lines.join("\n");
        (fields_str, init_str)
    }

    /// Calculates the required warm-up period based on indicator dependencies
    fn calculate_warm_up_period(&self, dependencies: &HashSet<IndicatorType>) -> usize {
        let max_period = dependencies.iter()
            .map(|indicator| match indicator {
                IndicatorType::Sma(period) => *period as usize,
                IndicatorType::Rsi(period) => *period as usize,
                IndicatorType::Ema(period) => *period as usize,
            })
            .max()
            .unwrap_or(20);
        // Add a safety margin
        max_period + 20
    }
}

/// Python-specific compiler
struct PythonCompiler {}

impl PythonCompiler {
    fn new() -> Self {
        Self {}
    }

    fn compile(&mut self, ops: &[Op]) -> Result<String, TranspilerError> {
        let mut statements = Vec::new();
        let mut stack: Vec<String> = Vec::new();
        let mut memory: HashMap<u8, String> = HashMap::new();

        for op in ops {
            match op {
                Op::EntryMarker | Op::ExitMarker => continue,
                Op::PushConstant(val) => {
                    stack.push(val.to_string());
                }
                Op::PushPrice(price_type) => {
                    let expr = match price_type {
                        PriceType::Open => "data[self.symbol].open".to_string(),
                        PriceType::High => "data[self.symbol].high".to_string(),
                        PriceType::Low => "data[self.symbol].low".to_string(),
                        PriceType::Close => "data[self.symbol].close".to_string(),
                    };
                    stack.push(expr);
                }
                Op::PushIndicator(indicator_type) => {
                    let expr = match indicator_type {
                        IndicatorType::Sma(period) => format!("self.sma{}.current.value", period),
                        IndicatorType::Rsi(period) => format!("self.rsi{}.current.value", period),
                        IndicatorType::Ema(period) => format!("self.ema{}.current.value", period),
                    };
                    stack.push(expr);
                }
                Op::PushDynamic(dynamic_const) => {
                    let expr = self.translate_dynamic_constant(dynamic_const)?;
                    stack.push(expr);
                }
                Op::Store(idx) => {
                    let val = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
                    let var_name = format!("mem{}", idx);
                    statements.push(format!("{} = {}", var_name, val));
                    memory.insert(*idx, var_name);
                }
                Op::Load(idx) => {
                    let var_name = memory.get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("mem{}", idx));
                    stack.push(var_name);
                }
                Op::Add => self.apply_arithmetic_op(&mut stack, " + ")?,
                Op::Subtract => self.apply_arithmetic_op(&mut stack, " - ")?,
                Op::Multiply => self.apply_arithmetic_op(&mut stack, " * ")?,
                Op::Divide => self.apply_arithmetic_op(&mut stack, " / ")?,
                Op::GreaterThan => self.apply_comparison_op(&mut stack, " > ")?,
                Op::LessThan => self.apply_comparison_op(&mut stack, " < ")?,
                Op::GreaterThanOrEqual => self.apply_comparison_op(&mut stack, " >= ")?,
                Op::LessThanOrEqual => self.apply_comparison_op(&mut stack, " <= ")?,
                Op::Equal => self.apply_comparison_op(&mut stack, " == ")?,
                Op::And => self.apply_logical_and(&mut stack)?,
                Op::Or => self.apply_logical_or(&mut stack)?,
                Op::Not => {
                    let val = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
                    stack.push(format!("(1.0 if {} == 0.0 else 0.0)", val));
                }
                Op::JumpIfFalse(_) | Op::Jump(_) | Op::Return => {
                    return Err(TranspilerError::UnsupportedOp(
                        format!("Control flow op {:?} not yet supported", op)
                    ));
                }
            }
        }

        // Generate memory variable declarations at the start
        let mut memory_decls = Vec::new();
        for idx in 0..16 {
            memory_decls.push(format!("mem{} = 0.0", idx));
        }

        let mut all_statements = memory_decls;
        all_statements.extend(statements);

        if stack.is_empty() {
            return Err(TranspilerError::StackUnderflow);
        }

        let result = stack.pop().unwrap();
        all_statements.push(format!("return {}", result));

        Ok(all_statements.join("\n        "))
    }

    fn apply_arithmetic_op(
        &self,
        stack: &mut Vec<String>,
        op: &str,
    ) -> Result<(), TranspilerError> {
        let b = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        let a = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        stack.push(format!("({}{}{})", a, op, b));
        Ok(())
    }

    fn apply_comparison_op(
        &self,
        stack: &mut Vec<String>,
        op: &str,
    ) -> Result<(), TranspilerError> {
        let b = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        let a = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        stack.push(format!("(1.0 if ({}{}{}) else 0.0)", a, op, b));
        Ok(())
    }

    fn apply_logical_and(
        &self,
        stack: &mut Vec<String>,
    ) -> Result<(), TranspilerError> {
        let b = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        let a = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        stack.push(format!("(1.0 if ({} > 0.0) and ({} > 0.0) else 0.0)", a, b));
        Ok(())
    }

    fn apply_logical_or(
        &self,
        stack: &mut Vec<String>,
    ) -> Result<(), TranspilerError> {
        let b = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        let a = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        stack.push(format!("(1.0 if ({} > 0.0) or ({} > 0.0) else 0.0)", a, b));
        Ok(())
    }

    fn translate_dynamic_constant(&self, dynamic_const: &DynamicConstant) -> Result<String, TranspilerError> {
        match dynamic_const {
            DynamicConstant::ClosePercent(pct) => {
                let factor = 1.0 + (*pct as f64 / 100.0);
                Ok(format!("data[self.symbol].close * {}", factor))
            }
            DynamicConstant::SmaPercent(period, pct) => {
                let factor = 1.0 + (*pct as f64 / 100.0);
                Ok(format!("self.sma{}.current.value * {}", period, factor))
            }
        }
    }
}

/// C#-specific compiler
struct CSharpCompiler {}

impl CSharpCompiler {
    fn new() -> Self {
        Self {}
    }

    fn compile(&mut self, ops: &[Op]) -> Result<String, TranspilerError> {
        let mut statements = Vec::new();
        let mut stack: Vec<String> = Vec::new();
        let mut memory: HashMap<u8, String> = HashMap::new();

        for op in ops {
            match op {
                Op::EntryMarker | Op::ExitMarker => continue,
                Op::PushConstant(val) => {
                    stack.push(val.to_string());
                }
                Op::PushPrice(price_type) => {
                    let expr = match price_type {
                        PriceType::Open => "data[symbol].Open".to_string(),
                        PriceType::High => "data[symbol].High".to_string(),
                        PriceType::Low => "data[symbol].Low".to_string(),
                        PriceType::Close => "data[symbol].Close".to_string(),
                    };
                    stack.push(expr);
                }
                Op::PushIndicator(indicator_type) => {
                    let expr = match indicator_type {
                        IndicatorType::Sma(period) => format!("sma{}.Current.Value", period),
                        IndicatorType::Rsi(period) => format!("rsi{}.Current.Value", period),
                        IndicatorType::Ema(period) => format!("ema{}.Current.Value", period),
                    };
                    stack.push(expr);
                }
                Op::PushDynamic(dynamic_const) => {
                    let expr = self.translate_dynamic_constant(dynamic_const)?;
                    stack.push(expr);
                }
                Op::Store(idx) => {
                    let val = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
                    let var_name = format!("mem{}", idx);
                    statements.push(format!("var {} = {};", var_name, val));
                    memory.insert(*idx, var_name);
                }
                Op::Load(idx) => {
                    let var_name = memory.get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("mem{}", idx));
                    stack.push(var_name);
                }
                Op::Add => self.apply_arithmetic_op(&mut stack, " + ")?,
                Op::Subtract => self.apply_arithmetic_op(&mut stack, " - ")?,
                Op::Multiply => self.apply_arithmetic_op(&mut stack, " * ")?,
                Op::Divide => self.apply_arithmetic_op(&mut stack, " / ")?,
                Op::GreaterThan => self.apply_comparison_op(&mut stack, " > ")?,
                Op::LessThan => self.apply_comparison_op(&mut stack, " < ")?,
                Op::GreaterThanOrEqual => self.apply_comparison_op(&mut stack, " >= ")?,
                Op::LessThanOrEqual => self.apply_comparison_op(&mut stack, " <= ")?,
                Op::Equal => self.apply_comparison_op(&mut stack, " == ")?,
                Op::And => self.apply_logical_and(&mut stack)?,
                Op::Or => self.apply_logical_or(&mut stack)?,
                Op::Not => {
                    let val = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
                    stack.push(format!("({} == 0.0 ? 1.0 : 0.0)", val));
                }
                Op::JumpIfFalse(_) | Op::Jump(_) | Op::Return => {
                    return Err(TranspilerError::UnsupportedOp(
                        format!("Control flow op {:?} not yet supported", op)
                    ));
                }
            }
        }

        // Generate memory variable declarations at the start
        let mut memory_decls = Vec::new();
        for idx in 0..16 {
            memory_decls.push(format!("double mem{} = 0.0;", idx));
        }

        let mut all_statements = memory_decls;
        all_statements.extend(statements);

        if stack.is_empty() {
            return Err(TranspilerError::StackUnderflow);
        }

        let result = stack.pop().unwrap();
        all_statements.push(format!("return {};", result));

        Ok(all_statements.join("\n            "))
    }

    fn apply_arithmetic_op(
        &self,
        stack: &mut Vec<String>,
        op: &str,
    ) -> Result<(), TranspilerError> {
        let b = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        let a = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        stack.push(format!("({}{}{})", a, op, b));
        Ok(())
    }

    fn apply_comparison_op(
        &self,
        stack: &mut Vec<String>,
        op: &str,
    ) -> Result<(), TranspilerError> {
        let b = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        let a = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        stack.push(format!("({}{}{} ? 1.0 : 0.0)", a, op, b));
        Ok(())
    }

    fn apply_logical_and(&self, stack: &mut Vec<String>) -> Result<(), TranspilerError> {
        let b = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        let a = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        stack.push(format!("(({} > 0.0) && ({} > 0.0) ? 1.0 : 0.0)", a, b));
        Ok(())
    }

    fn apply_logical_or(&self, stack: &mut Vec<String>) -> Result<(), TranspilerError> {
        let b = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        let a = stack.pop().ok_or(TranspilerError::StackUnderflow)?;
        stack.push(format!("(({} > 0.0) || ({} > 0.0) ? 1.0 : 0.0)", a, b));
        Ok(())
    }

    fn translate_dynamic_constant(&self, dynamic_const: &DynamicConstant) -> Result<String, TranspilerError> {
        match dynamic_const {
            DynamicConstant::ClosePercent(pct) => {
                let factor = 1.0 + (*pct as f64 / 100.0);
                Ok(format!("data[symbol].Close * {}", factor))
            }
            DynamicConstant::SmaPercent(period, pct) => {
                let factor = 1.0 + (*pct as f64 / 100.0);
                Ok(format!("sma{}.Current.Value * {}", period, factor))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::Strategy;
    use std::collections::HashMap;

    fn create_test_strategy() -> Strategy {
        let mut programs = HashMap::new();
        // Simple strategy: Close > SMA20
        programs.insert(
            "entry".to_string(),
            vec![
                Op::PushPrice(PriceType::Close),
                Op::PushIndicator(IndicatorType::Sma(20)),
                Op::GreaterThan,
            ],
        );
        Strategy { programs }
    }

    #[test]
    fn test_python_transpilation() {
        let config = TranspilerConfig {
            symbol: "SPY".to_string(),
            resolution: "Daily".to_string(),
            market: "usa".to_string(),
            initial_cash: Some(10000.0),
            transaction_cost_pct: Some(0.001),
            slippage_pct: Some(0.001),
        };

        let engine = TranspilerEngine::new(config);
        let strategy = create_test_strategy();

        let result = engine.to_python(&strategy);
        assert!(result.is_ok());
        let python_code = result.unwrap();

        // Basic sanity checks
        assert!(python_code.contains("class GoldenAegisAlgorithm"));
        assert!(python_code.contains("def evaluate_entry"));
        assert!(python_code.contains("self.sma20 = self.SMA"));
        assert!(python_code.contains("data[self.symbol].close"));
        assert!(python_code.contains("self.sma20.current.value"));
    }

    #[test]
    fn test_csharp_transpilation() {
        let config = TranspilerConfig {
            symbol: "SPY".to_string(),
            resolution: "Daily".to_string(),
            market: "usa".to_string(),
            initial_cash: Some(10000.0),
            transaction_cost_pct: Some(0.001),
            slippage_pct: Some(0.001),
        };

        let engine = TranspilerEngine::new(config);
        let strategy = create_test_strategy();

        let result = engine.to_c_sharp(&strategy);
        assert!(result.is_ok());
        let csharp_code = result.unwrap();

        // Basic sanity checks
        assert!(csharp_code.contains("class GoldenAegisAlgorithm"));
        assert!(csharp_code.contains("EvaluateEntry"));
        assert!(csharp_code.contains("SimpleMovingAverage sma20"));
        assert!(csharp_code.contains("data[symbol].Close"));
        assert!(csharp_code.contains("sma20.Current.Value"));
    }

    #[test]
    fn test_complex_expression() {
        let mut programs = HashMap::new();
        // Complex expression: (Close + SMA20 * 1.02) > (RSI14 - 30)
        programs.insert(
            "entry".to_string(),
            vec![
                Op::PushPrice(PriceType::Close),
                Op::PushIndicator(IndicatorType::Sma(20)),
                Op::PushConstant(1.02),
                Op::Multiply,
                Op::Add,
                Op::PushIndicator(IndicatorType::Rsi(14)),
                Op::PushConstant(30.0),
                Op::Subtract,
                Op::GreaterThan,
            ],
        );
        let strategy = Strategy { programs };

        let config = TranspilerConfig {
            symbol: "AAPL".to_string(),
            resolution: "Hour".to_string(),
            market: "usa".to_string(),
            initial_cash: Some(50000.0),
            transaction_cost_pct: None,
            slippage_pct: None,
        };

        let engine = TranspilerEngine::new(config);

        // Test Python
        let python_result = engine.to_python(&strategy);
        assert!(python_result.is_ok());
        let python_code = python_result.unwrap();
        assert!(python_code.contains("self.sma20"));
        assert!(python_code.contains("self.rsi14"));
        assert!(python_code.contains("data[self.symbol].close"));

        // Test C#
        let csharp_result = engine.to_c_sharp(&strategy);
        assert!(csharp_result.is_ok());
        let csharp_code = csharp_result.unwrap();
        assert!(csharp_code.contains("sma20"));
        assert!(csharp_code.contains("rsi14"));
        assert!(csharp_code.contains("data[symbol].Close"));
    }

    #[test]
    fn test_clean_expression_output() {
        // Test that simple expressions generate clean, readable code without temp variables
        let mut programs = HashMap::new();
        // Simple expression: Close > SMA20
        programs.insert(
            "entry".to_string(),
            vec![
                Op::PushPrice(PriceType::Close),
                Op::PushIndicator(IndicatorType::Sma(20)),
                Op::GreaterThan,
            ],
        );
        let strategy = Strategy { programs };

        let config = TranspilerConfig {
            symbol: "SPY".to_string(),
            resolution: "Daily".to_string(),
            market: "usa".to_string(),
            initial_cash: Some(10000.0),
            transaction_cost_pct: None,
            slippage_pct: None,
        };

        let engine = TranspilerEngine::new(config);

        // Test Python
        let python_result = engine.to_python(&strategy);
        assert!(python_result.is_ok());
        let python_code = python_result.unwrap();

        // Check that evaluate_entry method contains clean expression
        let entry_method_start = python_code.find("def evaluate_entry").unwrap();
        let entry_method_end = python_code[entry_method_start..].find("\n    def").unwrap_or(python_code[entry_method_start..].len());
        let entry_method = &python_code[entry_method_start..entry_method_start + entry_method_end];

        // Should not contain temp variable assignments
        assert!(!entry_method.contains("temp"), "Python output contains temp variables: {}", entry_method);
        // Should contain the clean comparison
        assert!(entry_method.contains("data[self.symbol].close"));
        assert!(entry_method.contains("self.sma20.current.value"));

        // Test C#
        let csharp_result = engine.to_c_sharp(&strategy);
        assert!(csharp_result.is_ok());
        let csharp_code = csharp_result.unwrap();

        // Check that EvaluateEntry method contains clean expression
        let entry_method_start = csharp_code.find("private bool EvaluateEntry").unwrap();
        let entry_method_end = csharp_code[entry_method_start..].find("\n        private").unwrap_or(csharp_code[entry_method_start..].len());
        let entry_method = &csharp_code[entry_method_start..entry_method_start + entry_method_end];

        // Should not contain temp variable declarations
        assert!(!entry_method.contains("var temp"), "C# output contains temp variables: {}", entry_method);
        // Should contain the clean comparison with proper ternary
        assert!(entry_method.contains("data[symbol].Close"));
        assert!(entry_method.contains("sma20.Current.Value"));
        assert!(entry_method.contains("? 1.0 : 0.0"));
    }

    #[test]
    fn test_memory_operations() {
        // Test that Store/Load ops work correctly
        let mut programs = HashMap::new();
        // Store 5 in mem0, load it, add 3, store in mem1
        programs.insert(
            "entry".to_string(),
            vec![
                Op::PushConstant(5.0),
                Op::Store(0),
                Op::Load(0),
                Op::PushConstant(3.0),
                Op::Add,
                Op::Store(1),
                Op::Load(1),
                Op::PushConstant(2.0),
                Op::GreaterThan,
            ],
        );
        let strategy = Strategy { programs };

        let config = TranspilerConfig {
            symbol: "SPY".to_string(),
            resolution: "Daily".to_string(),
            market: "usa".to_string(),
            initial_cash: Some(10000.0),
            transaction_cost_pct: None,
            slippage_pct: None,
        };

        let engine = TranspilerEngine::new(config);

        // Both languages should handle memory ops
        let python_result = engine.to_python(&strategy);
        assert!(python_result.is_ok());
        let python_code = python_result.unwrap();
        // Note: 5.0.to_string() = "5", 3.0.to_string() = "3"
        assert!(python_code.contains("mem0 = 5"), "Python code missing mem0 assignment: {}", python_code);
        assert!(python_code.contains("mem1 = (mem0 + 3)"), "Python code missing mem1 assignment: {}", python_code);

        let csharp_result = engine.to_c_sharp(&strategy);
        assert!(csharp_result.is_ok());
        let csharp_code = csharp_result.unwrap();
        assert!(csharp_code.contains("mem0 = 5;"), "C# code missing mem0 assignment: {}", csharp_code);
        assert!(csharp_code.contains("mem1 = (mem0 + 3);"), "C# code missing mem1 assignment: {}", csharp_code);
    }
}
