use crate::evolution::grammar::Grammar;
use crate::evolution::indicator_parser::parse_indicator_terminal;
use crate::evolution::Genome;
use crate::strategy::Strategy;
use crate::vm::op::{DynamicConstant, Op, PriceType};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum MappingError {
    #[error(
        "Invalid grammar: non-terminal '{non_terminal}' has no production rules. Genome: {genome:?}"
    )]
    MissingProduction {
        non_terminal: String,
        genome: Genome,
    },
    #[error("Grammar recursion depth limit ({limit}) exceeded. Genome: {genome:?}")]
    DepthLimitExceeded { limit: u32, genome: Genome },
    #[error("Program exceeded max token limit of {limit}. Genome: {genome:?}")]
    TokenLimitExceeded { limit: usize, genome: Genome },
}

/// Struct that keeps information about the information
/// relating to the mapping of a given genome
struct MappingContext<'a> {
    /// Reference to the `Genome` to map
    genome: &'a Genome,
    /// Current index to be map for `Strategy`
    codon_idx: usize,
    /// Current number of tokens in the `Strategy`
    token_count: usize,
    /// Current recursion depth
    recursion_depth: u32,
    /// Buffer for building multi-token indicators (e.g., "SMA(", "20", ")" -> "SMA(20)")
    indicator_buffer: Option<String>,
    /// Stack for tracking jump targets during control flow expansion (IF/THEN/ELSE)
    jump_stack: Vec<usize>,
}

impl<'a> MappingContext<'a> {
    /// Creates a new `MappingContext`
    ///
    /// # Arguments
    /// * `genome` - Reference to a `Genome` struct representing the genome to map to a `Strategy`
    ///
    /// # Returns
    /// `MappingContext`
    fn new(genome: &'a Genome) -> Self {
        Self {
            genome,
            codon_idx: 0,
            token_count: 0,
            recursion_depth: 0,
            indicator_buffer: None,
            jump_stack: Vec::new(),
        }
    }

    /// This function returns the next codon to map, with wrapping if the genome is not long
    /// enough.
    ///
    /// This method mutably borrows the `MappingContext` instance,
    /// modifying its `codon_idx` field after each call to the function.
    ///
    /// # Arguments
    /// * `&mut self` - Reference to a `Genome` struct representing the genome to map to a `Strategy`
    ///
    /// # Returns
    /// `u32` - the `u32` codon to be mapped
    fn next_codon(&mut self) -> u32 {
        if self.genome.is_empty() {
            return 0;
        }
        let codon = self.genome[self.codon_idx];
        self.codon_idx = (self.codon_idx + 1) % self.genome.len();
        codon
    }

    /// Process a terminal symbol for indicator buffering.
    /// Returns Some(combined_terminal) if a complete indicator was formed,
    /// None otherwise (meaning the terminal was consumed into the buffer).
    pub(crate) fn process_indicator_token(&mut self, terminal: &str) -> Option<String> {
        // Check if we're already building an indicator
        if let Some(ref mut buf) = self.indicator_buffer {
            // Check if this is the closing parenthesis
            if terminal == ")" {
                // Complete the indicator
                buf.push(')');
                let complete = buf.clone();
                self.indicator_buffer = None;
                return Some(complete);
            } else {
                // Assume this is the period number, comma, or std_dev
                // Add to buffer (no space between tokens)
                buf.push_str(terminal);
                return None;
            }
        }

        // Not currently building an indicator, check if this starts one
        if terminal.ends_with('(') && (
            terminal.starts_with("SMA") || 
            terminal.starts_with("EMA") || 
            terminal.starts_with("RSI") ||
            terminal.starts_with("BB_UPPER") ||
            terminal.starts_with("BB_LOWER") ||
            terminal.starts_with("DC_UPPER") ||
            terminal.starts_with("DC_LOWER") ||
            terminal.starts_with("DC_MIDDLE")
        ) {
            // Start building an indicator
            self.indicator_buffer = Some(terminal.to_string());
            None
        } else {
            // Not an indicator component, pass through as is
            Some(terminal.to_string())
        }
    }
}

/// This struct maps (you don't say) the genome into a `Strategy` struct.
/// The `max_tokens` allowed for a given genome, and the `max_recursion_depth` to allow when
/// mapping.
#[derive(Clone)]
pub struct GrammarBasedMapper<'a> {
    /// The `Grammar` to be used for the mapping
    grammar: &'a Grammar,
    /// Max allowable token count for a given strategy
    max_tokens: usize,
    /// Max allowable recursion depth (to avoid infinite recursions)
    max_recursion_depth: u32,
}

impl<'a> GrammarBasedMapper<'a> {
    /// Creates a new `GrammarBasedMapper`
    ///
    /// # Arguments
    /// * `grammar` - Reference to the `Grammar` that will be used for the mapping
    /// * `max_tokens` - Maximum number of tokens allowed for a given genome
    /// * `max_recursion_depth` - Maximum recursion depth allowed during the mapping process
    ///
    /// # Returns
    /// * `Self` - An instance of the EvolutionEngine struct
    pub fn new(grammar: &'a Grammar, max_tokens: usize, max_recursion_depth: u32) -> Self {
        Self {
            grammar,
            max_tokens,
            max_recursion_depth,
        }
    }

    /// The main orchestrator for the `GrammarBasedMapper`
    /// Delegates actual expansion (token mapping) to `expand`, but it handles the rest
    ///
    /// # Arguments
    /// * `&self` - Reference to `GrammarBasedMapper`
    /// * `genome` - The genome that must be turned into an actual program
    ///
    /// # Returns
    /// * `Result<Strategy, MappingError>`
    pub fn map(&self, genome: &Genome) -> Result<Strategy, MappingError> {
        let mut context = MappingContext::new(genome);
        let mut programs = HashMap::new();

        let mut bytecode = Vec::new();
        self.expand("<start>", &mut context, &mut bytecode)?;

        // Split the bytecode stream into separate programs based on markers
        let mut current_program_name: Option<String> = None;
        let mut current_program_code: Vec<Op> = Vec::new();

        for op in bytecode {
            match op {
                Op::EntryMarker => {
                    self.save_program(&mut programs, &mut current_program_name, &mut current_program_code);
                    current_program_name = Some("entry".to_string());
                }
                Op::ExitMarker => {
                    self.save_program(&mut programs, &mut current_program_name, &mut current_program_code);
                    current_program_name = Some("exit".to_string());
                }
                Op::StopLossMarker => {
                    self.save_program(&mut programs, &mut current_program_name, &mut current_program_code);
                    current_program_name = Some("stop_loss".to_string());
                }
                Op::TakeProfitMarker => {
                    self.save_program(&mut programs, &mut current_program_name, &mut current_program_code);
                    current_program_name = Some("take_profit".to_string());
                }
                Op::SizeMarker => {
                    self.save_program(&mut programs, &mut current_program_name, &mut current_program_code);
                    current_program_name = Some("position_sizing".to_string());
                }
                _ => {
                    current_program_code.push(op);
                }
            }
        }
        
        self.save_program(&mut programs, &mut current_program_name, &mut current_program_code);

        Ok(Strategy { programs })
    }

    fn save_program(&self, programs: &mut HashMap<String, Vec<Op>>, name: &mut Option<String>, code: &mut Vec<Op>) {
        if let Some(n) = name.take() {
            if !code.is_empty() {
                programs.insert(n, std::mem::take(code));
            }
        }
    }

    /// This function takes care of transforming a symbol into
    /// either some other token or a non-terminal.
    fn expand(
        &self,
        symbol: &str,
        context: &mut MappingContext,
        bytecode: &mut Vec<Op>,
    ) -> Result<(), MappingError> {
        if context.token_count >= self.max_tokens {
            return Err(MappingError::TokenLimitExceeded {
                limit: self.max_tokens,
                genome: context.genome.clone(),
            });
        }
        if context.recursion_depth >= self.max_recursion_depth {
            return Err(MappingError::DepthLimitExceeded {
                limit: self.max_recursion_depth,
                genome: context.genome.clone(),
            });
        }

        context.recursion_depth += 1;

        if self.grammar.is_non_terminal(symbol) {
            let productions =
                self.grammar
                    .rules
                    .get(symbol)
                    .ok_or_else(|| MappingError::MissingProduction {
                        non_terminal: symbol.to_string(),
                        genome: context.genome.clone(),
                    })?;

            let choice = context.next_codon() as usize % productions.len();
            let expansion = &productions[choice];

            for s in expansion {
                self.expand(s, context, bytecode)?;
            }
        } else {
            // Translate terminal into an Op or a placeholder
            match symbol {
                "ENTRY" => bytecode.push(Op::EntryMarker),
                "EXIT" => bytecode.push(Op::ExitMarker),
                "SL" => bytecode.push(Op::StopLossMarker),
                "TP" => bytecode.push(Op::TakeProfitMarker),
                "SIZE" => bytecode.push(Op::SizeMarker),
                "IF" => {
                    // Start of IF block. No-op in bytecode.
                }
                "THEN" => {
                    // Condition is on stack. Jump to ELSE if false.
                    bytecode.push(Op::JumpIfFalse(0));
                    context.jump_stack.push(bytecode.len() - 1);
                    context.token_count += 1;
                }
                "ELSE" => {
                    // End of THEN block. Jump to END_IF (skip ELSE block).
                    bytecode.push(Op::Jump(0));
                    let jump_idx = bytecode.len() - 1;
                    context.token_count += 1;

                    // Patch the THEN jump (JumpIfFalse) to point here + 1 (start of ELSE block)
                    if let Some(then_jump_idx) = context.jump_stack.pop() {
                        let target = bytecode.len(); 
                        if let Some(Op::JumpIfFalse(t)) = bytecode.get_mut(then_jump_idx) {
                            *t = target;
                        }
                    }
                    
                    // Push the ELSE jump index to stack so END_IF can patch it
                    context.jump_stack.push(jump_idx);
                }
                "END_IF" => {
                    // End of ELSE block.
                    // Patch the ELSE jump (Jump) to point here (start of next instruction)
                    if let Some(else_jump_idx) = context.jump_stack.pop() {
                        let target = bytecode.len();
                        if let Some(Op::Jump(t)) = bytecode.get_mut(else_jump_idx) {
                            *t = target;
                        }
                    }
                }
                _ => {
                    // Handle indicator buffering for dynamic period indicators
                    let terminal_to_process = context.process_indicator_token(symbol);
                    
                    if let Some(processed_terminal) = terminal_to_process {
                        if let Some(op) = self.terminal_to_op(&processed_terminal) {
                            bytecode.push(op);
                            context.token_count += 1;
                        }
                    }
                }
            }
        }

        context.recursion_depth -= 1;
        Ok(())
    }

    /// This function is just a huge match statement which takes care
    /// of associating given (terminal) symbols to their equivalent `vm::Op` code
    fn terminal_to_op(&self, terminal: &str) -> Option<Op> {
        match terminal {
            "ADD" => Some(Op::Add),
            "SUB" => Some(Op::Subtract),
            "MUL" => Some(Op::Multiply),
            "DIV_SAFE" => Some(Op::Divide),
            "GT" => Some(Op::GreaterThan),
            "LT" => Some(Op::LessThan),
            "GE" => Some(Op::GreaterThanOrEqual),
            "LE" => Some(Op::LessThanOrEqual),
            "EQ" => Some(Op::Equal),
            "AND" => Some(Op::And),
            "OR" => Some(Op::Or),
            "NOT" => Some(Op::Not),
            "CLOSE" => Some(Op::PushPrice(PriceType::Close)),
            "OPEN" => Some(Op::PushPrice(PriceType::Open)),
            "HIGH" => Some(Op::PushPrice(PriceType::High)),
            "LOW" => Some(Op::PushPrice(PriceType::Low)),
            "CLOSE_P1" => Some(Op::PushDynamic(DynamicConstant::ClosePercent(1))),
            "CLOSE_M1" => Some(Op::PushDynamic(DynamicConstant::ClosePercent(-1))),
            "SMA20_P2" => Some(Op::PushDynamic(DynamicConstant::SmaPercent(20, 2))),
            "SMA20_M2" => Some(Op::PushDynamic(DynamicConstant::SmaPercent(20, -2))),
            c => {
                // Try to parse as indicator first (supports SMA(20), SMA20, EMA(14), etc.)
                if let Some(indicator_type) = parse_indicator_terminal(c) {
                    return Some(Op::PushIndicator(indicator_type));
                }
                // Fall back to numeric constant
                c.parse::<f64>().ok().map(Op::PushConstant)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::grammar::Grammar;
    use crate::vm::op::IndicatorType;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn grammar_from_str(grammar_str: &str) -> Grammar {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.bnf");
        let mut file = File::create(&file_path).unwrap();
        write!(file, "{}", grammar_str).unwrap();
        Grammar::new(&file_path).unwrap()
    }

    #[test]
    fn test_optional_exit_is_respected() {
        let grammar = grammar_from_str(
            "
            <start> ::= <entry_program> <optional_exit>
            <entry_program> ::= ENTRY 1.0
            <optional_exit> ::= | <exit_program>
            <exit_program> ::= EXIT 0.0
        ",
        );
        let mapper = GrammarBasedMapper::new(&grammar, 50, 256);

        // Codon consumption:
        // 1st codon for <start> (trivial choice)
        // 2nd codon for <entry_program> (trivial choice)
        // 3rd codon for <optional_exit> (choice 0 -> empty, choice 1 -> <exit_program>)

        // To get no exit program, the 3rd codon consumed must be 0 (or an even number).
        // The original `vec![0, 0]` works because the 3rd access wraps around to the first element.
        let genome_no_exit: Genome = vec![0, 0];
        let strategy_no_exit = mapper.map(&genome_no_exit).unwrap();
        assert!(strategy_no_exit.programs.contains_key("entry"));
        assert!(!strategy_no_exit.programs.contains_key("exit"));

        // To get an exit program, the 3rd codon consumed must be 1 (or an odd number).
        // The original `vec![0, 1]` failed because the 3rd codon was `0` (due to wrap-around).
        // This new genome ensures the choice for <optional_exit> is 1.
        let genome_with_exit: Genome = vec![0, 0, 1];
        let strategy_with_exit = mapper.map(&genome_with_exit).unwrap();
        assert!(strategy_with_exit.programs.contains_key("entry"));
        assert!(strategy_with_exit.programs.contains_key("exit"));
        assert_eq!(
            strategy_with_exit.programs.get("exit").unwrap(),
            &vec![Op::PushConstant(0.0)]
        );
    }

    #[test]
    fn test_token_limit_enforced() {
        let grammar = grammar_from_str(
            "
            <start> ::= <entry_program>
            <entry_program> ::= ENTRY <long_expr>
            <long_expr> ::= 1.0 1.0 ADD 1.0 ADD 1.0 ADD 1.0 ADD
        ",
        );
        let mapper = GrammarBasedMapper::new(&grammar, 3, 256); // Very small limit

        let genome: Genome = vec![0, 0];
        let result = mapper.map(&genome);

        assert!(matches!(
            result,
            Err(MappingError::TokenLimitExceeded { .. })
        ));
        if let Err(MappingError::TokenLimitExceeded {
            limit,
            genome: err_genome,
        }) = result
        {
            assert_eq!(limit, 3);
            assert_eq!(err_genome, genome);
        }
    }

    #[test]
    fn test_dynamic_period_indicator() {
        // Grammar that uses dynamic period indicators
        let grammar = grammar_from_str(
            "
            <start> ::= <entry_program>
            <entry_program> ::= ENTRY <indicator>
            <indicator> ::= <sma_indicator>
            <sma_indicator> ::= SMA( <period> )
            <period> ::= 20 | 50 | 100
        ",
        );
        let mapper = GrammarBasedMapper::new(&grammar, 50, 256);

        // Test mapping with different genomes that should produce SMA(20), SMA(50), SMA(100)
        // The genome codons determine which period is chosen.
        // Since there are 3 period options, codon % 3 selects index.
        // We'll test each possible period.
        
        // Number of non-terminal expansions: <start>, <entry_program>, <indicator>, <sma_indicator>, <period>
        // That's 5 codons consumed.
        // Codon indices 0-3 correspond to trivial choices (only one production).
        // Codon index 4 selects period (0->20, 1->50, 2->100).
        
        // Period 20 (index 0)
        let genome_sma20: Genome = vec![0, 0, 0, 0, 0];
        let strategy_sma20 = mapper.map(&genome_sma20).unwrap();
        assert!(strategy_sma20.programs.contains_key("entry"));
        let entry_code = strategy_sma20.programs.get("entry").unwrap();
        assert_eq!(entry_code.len(), 1);
        match entry_code[0] {
            Op::PushIndicator(IndicatorType::Sma(20)) => (),
            _ => panic!("Expected SMA(20), got {:?}", entry_code[0]),
        }

        // Period 50 (index 1)
        let genome_sma50: Genome = vec![0, 0, 0, 0, 1];
        let strategy_sma50 = mapper.map(&genome_sma50).unwrap();
        let entry_code = strategy_sma50.programs.get("entry").unwrap();
        match entry_code[0] {
            Op::PushIndicator(IndicatorType::Sma(50)) => (),
            _ => panic!("Expected SMA(50), got {:?}", entry_code[0]),
        }

        // Period 100 (index 2)
        let genome_sma100: Genome = vec![0, 0, 0, 0, 2];
        let strategy_sma100 = mapper.map(&genome_sma100).unwrap();
        let entry_code = strategy_sma100.programs.get("entry").unwrap();
        match entry_code[0] {
            Op::PushIndicator(IndicatorType::Sma(100)) => (),
            _ => panic!("Expected SMA(100), got {:?}", entry_code[0]),
        }
    }

    #[test]
    fn test_dynamic_period_indicator_with_expression() {
        // More complex grammar using dynamic indicator in a comparison
        let grammar = grammar_from_str(
            "
            <start> ::= <entry_program>
            <entry_program> ::= ENTRY <boolean_expr>
            <boolean_expr> ::= <indicator> CLOSE GT
            <indicator> ::= <sma_indicator>
            <sma_indicator> ::= SMA( <period> )
            <period> ::= 20 | 50
        ",
        );
        let mapper = GrammarBasedMapper::new(&grammar, 50, 256);

        // Genome sequence:
        // Non-terminal expansions: <start>, <entry_program>, <boolean_expr>, <indicator>, <sma_indicator>, <period>
        // That's 6 codons consumed.
        // Codon indices 0-4 correspond to trivial choices (only one production each).
        // Codon index 5 selects period (0->20, 1->50).
        // Let's pick period 20 (index 0)
        let genome: Genome = vec![0, 0, 0, 0, 0, 0];
        let strategy = mapper.map(&genome).unwrap();
        let entry_code = strategy.programs.get("entry").unwrap();
        // Expected bytecode: SMA(20), Close, GT
        assert_eq!(entry_code.len(), 3);
        match entry_code[0] {
            Op::PushIndicator(IndicatorType::Sma(20)) => (),
            _ => panic!("Expected SMA(20), got {:?}", entry_code[0]),
        }
        assert_eq!(entry_code[1], Op::PushPrice(PriceType::Close));
        assert_eq!(entry_code[2], Op::GreaterThan);
    }

    #[test]
    fn test_process_indicator_token() {
        let genome = vec![];
        let mut ctx = MappingContext::new(&genome);
        
        // Start SMA indicator
        assert_eq!(ctx.process_indicator_token("SMA("), None);
        // Add period
        assert_eq!(ctx.process_indicator_token("20"), None);
        // Close parenthesis
        assert_eq!(ctx.process_indicator_token(")"), Some("SMA(20)".to_string()));
        
        // Buffer should be cleared
        // Note: cannot directly access private field, but we can test by sending another token
        // and ensuring it's not buffered.
        assert_eq!(ctx.process_indicator_token("CLOSE"), Some("CLOSE".to_string()));
        
        // Test EMA
        assert_eq!(ctx.process_indicator_token("EMA("), None);
        assert_eq!(ctx.process_indicator_token("100"), None);
        assert_eq!(ctx.process_indicator_token(")"), Some("EMA(100)".to_string()));
        
        // Test RSI
        assert_eq!(ctx.process_indicator_token("RSI("), None);
        assert_eq!(ctx.process_indicator_token("14"), None);
        assert_eq!(ctx.process_indicator_token(")"), Some("RSI(14)".to_string()));
        
        // Old style indicator passes through
        assert_eq!(ctx.process_indicator_token("SMA20"), Some("SMA20".to_string()));
        assert_eq!(ctx.process_indicator_token("RSI14"), Some("RSI14".to_string()));
        
        // Non-indicator terminals pass through
        assert_eq!(ctx.process_indicator_token("ADD"), Some("ADD".to_string()));
        assert_eq!(ctx.process_indicator_token("123.45"), Some("123.45".to_string()));
    }

    #[test]
    fn test_old_style_indicator_mapping() {
        let grammar = grammar_from_str(
            "
            <start> ::= <entry_program>
            <entry_program> ::= ENTRY <indicator>
            <indicator> ::= SMA20 | SMA50 | SMA100
        ",
        );
        let mapper = GrammarBasedMapper::new(&grammar, 50, 256);
        
        // SMA20 (choice 0)
        let genome = vec![0, 0, 0];
        let strategy = mapper.map(&genome).unwrap();
        let entry_code = strategy.programs.get("entry").unwrap();
        match entry_code[0] {
            Op::PushIndicator(IndicatorType::Sma(20)) => (),
            _ => panic!("Expected SMA(20)"),
        }

        // SMA50 (choice 1)
        let genome = vec![0, 0, 1];
        let strategy = mapper.map(&genome).unwrap();
        let entry_code = strategy.programs.get("entry").unwrap();
        match entry_code[0] {
            Op::PushIndicator(IndicatorType::Sma(50)) => (),
            _ => panic!("Expected SMA(50)"),
        }

        // SMA100 (choice 2)
        let genome = vec![0, 0, 2];
        let strategy = mapper.map(&genome).unwrap();
        let entry_code = strategy.programs.get("entry").unwrap();
        match entry_code[0] {
            Op::PushIndicator(IndicatorType::Sma(100)) => (),
            _ => panic!("Expected SMA(100)"),
        }
    }
}
