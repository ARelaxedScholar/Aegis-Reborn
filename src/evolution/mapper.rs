use crate::strategy::Strategy;
use crate::evolution::Genome;
use std::collections::HashMap;

// Placeholder for the real grammar parser we will build in Step 3.
pub struct Grammar {
    pub rules: HashMap<String, Vec<String>>,
}

#[derive(Debug)]
pub enum MappingError {
    InvalidGenome,
}

/// The GrammarBasedMapper is responsible for translating a Genome into a Strategy.
pub struct GrammarBasedMapper<'a> {
    grammar: &'a Grammar,
}

impl<'a> GrammarBasedMapper<'a> {
    pub fn new(grammar: &'a Grammar) -> Self {
        Self { grammar }
    }

    /// The core function that will be implemented in Step 3.
    pub fn map(&self, _genome: &Genome) -> Result<Strategy, MappingError> {
        unimplemented!("Mapper logic to be implemented in Step 3.")
    }
}
