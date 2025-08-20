use serde::{Deserialize, Serialize};

/// The Op codes our VM actually supports
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// The Op codes related to numbers
    PushConstant(f64),
    PushPrice(PriceType),
    PushIndicator(IndicatorType),
    PushDynamic(DynamicConstant),
    Store(u8),
    Load(u8),

    /// The Operators and Comparators
    Add,
    Subtract,
    Multiply,
    Divide,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    And,
    Or,
    Not,

    /// Conditionals
    JumpIfFalse(usize),
    Jump(usize),
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PriceType {
    Open,
    High,
    Low,
    Close,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndicatorType {
    Sma(u16),
    Rsi(u16),
}

/// Defines dynamic, relative constants calculated at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DynamicConstant {
    /// Represents `CLOSE * (1 + percent/100)`. `percent` can be negative.
    ClosePercent(i8),
    /// Represents `SMA(period) * (1 + percent/100)`.
    SmaPercent(u16, i8),
}
