use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GrammarError {
    #[error("Failed to read grammar file: {0}")]
    FileReadError(#[from] std::io::Error),
    #[error("Failed to parse rule on line: '{0}'")]
    ParseError(String),
    #[error("Start symbol '<start>' not found in grammar")]
    MissingStartSymbol,
    #[error("Undefined non-terminal referenced in grammar: '{0}'")]
    UndefinedNonTerminal(String),
    #[error("Unreachable rule in grammar: {0}")]
    UnreachableRule(String),
    #[error("Non-terminating rule in grammar: {0}")]
    NonTerminatingRule(String),
}

/// Represents a parsed BNF grammar, validated for logical consistency.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub rules: HashMap<String, Vec<Vec<String>>>,
}

impl Grammar {
    /// Parses and validates a grammar from a `.bnf` file when creating a `Grammar` instance
    ///
    /// # Arguments
    /// * `path` - Reference to a `Path` struct representing the path to the the user-specifed grammar.
    ///
    /// # Returns
    /// * `Result<Self, GrammarError>` - A Result enum returning `Grammar` in the happy case, and
    /// `GrammarError` otherwise
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

            let productions = parts[1]
                .split('|')
                .map(|p| p.split_whitespace().map(String::from).collect())
                .collect();

            rules.insert(non_terminal, productions);
        }

        if !rules.contains_key("<start>") {
            return Err(GrammarError::MissingStartSymbol);
        }

        let grammar = Self { rules };
        grammar.validate()?; // Mandated validation pass
        Ok(grammar)
    }

    /// Validates that all referenced non-terminals are defined.
    /// Decides if the `Grammar` is properly formed.
    ///
    /// # Arguments
    /// * `&self` - Reference to the `Grammar` struct to validate
    ///
    /// # Returns
    /// * `Result<(), GrammarError>` - A Result enum returning a unit in the happy case, and
    /// `GrammarError` otherwise
    fn validate(&self) -> Result<(), GrammarError> {
        use std::collections::{HashSet, VecDeque};

        // 1. Check for undefined non-terminals
        for productions in self.rules.values() {
            for production in productions {
                for symbol in production {
                    if self.is_non_terminal(symbol) && !self.rules.contains_key(symbol) {
                        return Err(GrammarError::UndefinedNonTerminal(symbol.clone()));
                    }
                }
            }
        }

        // 2. Check for unreachable rules
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        // we already checked <start> exists
        // Does BFS to check if each rule is reachable from <start>
        queue.push_back("<start>".to_string());
        reachable.insert("<start>".to_string());
        while let Some(current) = queue.pop_front() {
            if let Some(productions) = self.rules.get(&current) {
                for production in productions {
                    for symbol in production {
                        if self.is_non_terminal(symbol)
                            && !reachable.contains(symbol)
                            && self.rules.contains_key(symbol)
                        {
                            reachable.insert(symbol.clone());
                            queue.push_back(symbol.clone());
                        }
                    }
                }
            }
        }
        for rule in self.rules.keys() {
            if !reachable.contains(rule) {
                return Err(GrammarError::UnreachableRule(rule.clone()));
            }
        }

        // 3. Check for non-terminating rules
        let mut terminating = HashSet::new();
        // Initially, rules with at least one production of only terminals are terminating
        let mut changed = true;
        while changed {
            changed = false;
            for (lhs, productions) in &self.rules {
                if terminating.contains(lhs) {
                    continue;
                }
                for production in productions {
                    if production
                        .iter()
                        .all(|s| !self.is_non_terminal(s) || terminating.contains(s))
                    {
                        terminating.insert(lhs.clone());
                        changed = true;
                        break;
                    }
                }
            }
        }
        for rule in self.rules.keys() {
            if !terminating.contains(rule) {
                return Err(GrammarError::NonTerminatingRule(rule.clone()));
            }
        }

        Ok(())
    }

    /// Checks if a given symbol is a non-terminal.
    ///
    /// # Arguments
    /// * `&self` - Reference to the `Grammar` struct in question
    /// * `symbol` - String slice of the symbol to check
    /// # Returns
    /// * `bool` - Returns `true` if is a non-terminal, `false` otherwise
    pub fn is_non_terminal(&self, symbol: &str) -> bool {
        // Check we have the proper format <NON-TERMINAL>,
        // and that it is not a degenerate one <>
        symbol.starts_with('<') && symbol.ends_with('>') && symbol.len() > 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_valid_grammar_loads() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("good.bnf");
        let mut file = File::create(&file_path).unwrap();
        write!(file, "<start> ::= <A> | <B>\n<A> ::= 'a'\n<B> ::= 'b'").unwrap();
        let result = Grammar::new(&file_path);
        assert!(result.is_ok(), "Error is {:?}", result.unwrap_err());
    }

    #[test]
    fn test_undefined_non_terminal_error() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("bad.bnf");
        let mut file = File::create(&file_path).unwrap();
        write!(file, "<start> ::= <undefined>").unwrap();
        let result = Grammar::new(&file_path);
        assert!(matches!(result, Err(GrammarError::UndefinedNonTerminal(s)) if s == "<undefined>"));
    }

    #[test]
    fn test_default_grammar_contains_dynamic_periods() {
        // Load the project's default grammar.bnf file
        let grammar_path = Path::new("grammar.bnf");
        let grammar = Grammar::new(grammar_path).expect("Failed to load default grammar.bnf");
        
        // Ensure dynamic period productions exist
        assert!(grammar.rules.contains_key("<sma_indicator>"));
        assert!(grammar.rules.contains_key("<ema_indicator>"));
        assert!(grammar.rules.contains_key("<rsi_indicator>"));
        assert!(grammar.rules.contains_key("<period>"));
        
        // Check that <period> productions include values 10..=200 step 10
        let period_productions = grammar.rules.get("<period>").unwrap();
        let expected_periods: Vec<String> = (10..=200).step_by(10).map(|p| p.to_string()).collect();
        for expected in expected_periods {
            assert!(period_productions.iter().any(|prod| prod.len() == 1 && prod[0] == expected),
                "Period {} not found in grammar", expected);
        }
        
        // Check that <indicator> includes both old and new styles
        let indicator_productions = grammar.rules.get("<indicator>").unwrap();
        // Should contain at least SMA20, SMA50, SMA100, SMA200, EMA20, EMA50, EMA100, EMA200, RSI14, RSI30
        let expected_old = vec!["SMA20", "SMA50", "SMA100", "SMA200", "EMA20", "EMA50", "EMA100", "EMA200", "RSI14", "RSI30"];
        for old in expected_old {
            assert!(indicator_productions.iter().any(|prod| prod.len() == 1 && prod[0] == old),
                "Old-style indicator {} not found in grammar", old);
        }
        // Should contain references to dynamic productions
        assert!(indicator_productions.iter().any(|prod| prod.len() == 1 && prod[0] == "<sma_indicator>"));
        assert!(indicator_productions.iter().any(|prod| prod.len() == 1 && prod[0] == "<ema_indicator>"));
        assert!(indicator_productions.iter().any(|prod| prod.len() == 1 && prod[0] == "<rsi_indicator>"));
    }
}
