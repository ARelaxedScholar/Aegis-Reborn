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
    fn test_walk_phase_grammar_structure() {
        // Load the project's default grammar.bnf file
        let grammar_path = Path::new("grammar.bnf");
        let grammar = Grammar::new(grammar_path).expect("Failed to load default grammar.bnf");

        // Ensure new Walk Phase constructs exist
        assert!(grammar.rules.contains_key("<signal_logic>"));
        assert!(grammar.rules.contains_key("<position_sizing>"));
        assert!(grammar.rules.contains_key("<stop_loss>"));
        assert!(grammar.rules.contains_key("<take_profit>"));

        // Check that <period> productions match the new restricted set
        let period_productions = grammar.rules.get("<period>").unwrap();
        let expected_periods = vec!["10", "14", "20", "50", "100", "200"];
        for expected in expected_periods {
            assert!(
                period_productions
                    .iter()
                    .any(|prod| prod.len() == 1 && prod[0] == expected),
                "Period {} not found in grammar",
                expected
            );
        }

        // Check that <std_dev> productions include 1, 2, 3
        let std_dev_productions = grammar.rules.get("<std_dev>").unwrap();
        let expected_std_devs = vec!["1", "2", "3"];
        for expected in expected_std_devs {
            assert!(
                std_dev_productions
                    .iter()
                    .any(|prod| prod.len() == 1 && prod[0] == expected),
                "StdDev {} not found in grammar",
                expected
            );
        }

        // Check that specific indicator categories exist
        assert!(grammar.rules.contains_key("<moving_average>"));
        assert!(grammar.rules.contains_key("<bands_indicator>"));
        assert!(grammar.rules.contains_key("<rsi_indicator>"));

        // Check <moving_average> productions
        let ma_productions = grammar.rules.get("<moving_average>").unwrap();
        // SMA( <period> ) -> ["SMA(", "<period>", ")"]
        assert!(ma_productions.iter().any(|prod| prod.len() == 3
            && prod[0] == "SMA("
            && prod[1] == "<period>"
            && prod[2] == ")"));
        assert!(ma_productions.iter().any(|prod| prod.len() == 3
            && prod[0] == "EMA("
            && prod[1] == "<period>"
            && prod[2] == ")"));

        // Check <bands_indicator> productions
        let bands_productions = grammar.rules.get("<bands_indicator>").unwrap();
        // BB_UPPER( <period> , <std_dev> ) -> ["BB_UPPER(", "<period>", ",", "<std_dev>", ")"]
        assert!(bands_productions.iter().any(|prod| prod.len() == 5
            && prod[0] == "BB_UPPER("
            && prod[1] == "<period>"
            && prod[2] == ","
            && prod[3] == "<std_dev>"
            && prod[4] == ")"));
        assert!(bands_productions.iter().any(|prod| prod.len() == 3
            && prod[0] == "DC_UPPER("
            && prod[1] == "<period>"
            && prod[2] == ")"));
    }
}
