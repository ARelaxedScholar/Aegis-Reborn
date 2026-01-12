use serde::{Deserialize, Serialize};

/// The Op codes our VM actually supports
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// The Program Delimiters
    EntryMarker,
    ExitMarker,

    /// The Op codes related to numbers
    PushConstant(f64),
    PushPrice(PriceType),
    PushIndicator(IndicatorType),
    PushDynamic(DynamicConstant),
    Store(u8),
    Load(u8),

    /// Historical price operations
    PushPrevious(PriceType, u16),      // Push price from N periods ago (0 = current, 1 = previous, etc.)
    PushRollingSum(PriceType, u16),    // Push sum of last N periods of given price type

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

    /// Phase 2 Markers
    StopLossMarker,
    TakeProfitMarker,
    SizeMarker,
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
    Ema(u16),
    BbUpper(u16, u8),
    BbLower(u16, u8),
    DcUpper(u16),
    DcLower(u16),
    DcMiddle(u16),
}

/// Defines dynamic, relative constants calculated at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DynamicConstant {
    /// Represents `CLOSE * (1 + percent/100)`. `percent` can be negative.
    ClosePercent(i8),
    /// Represents `SMA(period) * (1 + percent/100)`.
    SmaPercent(u16, i8),
}
