use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug, PartialEq)] // Added PartialEq for easier testing
pub enum GrammarError {
    #[error("Failed to read grammar file: {0}")]
    FileReadError(#[from] std::io::Error),
    #[error("Failed to parse rule on line: '{0}'")]
    ParseError(String),
    #[error("Start symbol '<start>' not found in grammar")]
    MissingStartSymbol,
    #[error("Undefined non-terminal referenced in grammar: '{0}'")]
    UndefinedNonTerminal(String),
}

#[derive(Debug, Clone)]
pub struct Grammar {
    pub rules: HashMap<String, Vec<Vec<String>>>,
}

impl Grammar {
    pub fn new(path: &Path) -> Result<Self, GrammarError> {
        let content = fs::read_to_string(path)?;
        let mut rules = HashMap::new();

        for line in content.lines() {
            if line.trim().starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split("::=").collect();
            if parts.len() != 2 {
                return Err(GrammarError::ParseError(line.to_string()));
            }
            let non_terminal = parts[0].trim().to_string();
            
            // CORRECTED: Gracefully handle empty parts from split
            let productions = parts[1].split('|').map(|p| {
                p.trim().split_whitespace().map(String::from).collect()
            }).collect();

            rules.insert(non_terminal, productions);
        }

        if !rules.contains_key("<start>") {
            return Err(GrammarError::MissingStartSymbol);
        }

        let grammar = Self { rules };
        grammar.validate()?; // Validate the grammar's integrity.
        Ok(grammar)
    }

    /// Validates that all referenced non-terminals are defined within the grammar.
    pub fn validate(&self) -> Result<(), GrammarError> {
        for productions in self.rules.values() {
            for production in productions {
                for symbol in production {
                    if self.is_non_terminal(symbol) && !self.rules.contains_key(symbol) {
                        return Err(GrammarError::UndefinedNonTerminal(symbol.clone()));
                    }
                }
            }
        }
        Ok(())
    }

    /// Checks if a given symbol is a non-terminal.
    pub fn is_non_terminal(&self, symbol: &str) -> bool {
        symbol.starts_with('<') && symbol.ends_with('>') && symbol.len() > 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;
    use std::fs::File;

    #[test]
    fn test_undefined_non_terminal_error() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("bad_grammar.bnf");
        let mut file = File::create(&file_path).unwrap();
        
        let grammar_str = "
            <start> ::= <entry>
            <entry> ::= <undefined_symbol>
        ";
        write!(file, "{}", grammar_str).unwrap();

        let result = Grammar::new(&file_path);
        assert!(matches!(result, Err(GrammarError::UndefinedNonTerminal(s)) if s == "<undefined_symbol>"));
    }

    #[test]
    fn test_malformed_non_terminal_is_not_detected() {
        let grammar = Grammar { rules: HashMap::new() };
        assert!(!grammar.is_non_terminal("<>"));
        assert!(!grammar.is_non_terminal("a"));
        assert!(grammar.is_non_terminal("<valid>"));
    }
}
