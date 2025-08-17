use crate::vm::op::Op;
use std::collections::HashMap;

/// Represents a complete, evolved trading strategy.
///
/// It is a collection of named programs (bytecode), adhering to the
/// "Flexible Toolkit" design principle. The backtester will query this
/// map for the specific programs it needs (e.g., "entry", "exit").
#[derive(Debug, Clone, Default)]
pub struct Strategy {
    pub programs: HashMap<String, Vec<Op>>,
}

impl Strategy {
    pub fn new() -> Self {
        Self::default()
    }
}
