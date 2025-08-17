#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // --- Data Loading & State Management ---
    PushConstant(f64),
    PushPrice(PriceType),
    PushIndicator(IndicatorType),
    Store(u8),
    Load(u8),

    // --- Operators ---
    Add,
    Subtract,
    Multiply,
    Divide, // Mandate: The VM's implementation of this MUST be safe.
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    And,
    Or,
    Not,

    // --- Control Flow (Reserved for "Walk" phase IF implementation) ---
    JumpIfFalse(usize),
    Jump(usize),
    Return,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PriceType {
    Close,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndicatorType {
    Sma(u16),
    Rsi(u16),
}
