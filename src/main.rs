use aegis_reborn::config::Config;
use aegis_reborn::evolution::Genome;
use aegis_reborn::evolution::grammar::Grammar;
use aegis_reborn::evolution::mapper::GrammarBasedMapper;
use rand::Rng;
use std::path::Path;

fn main() {
    env_logger::init();
    log::info!("Booting Aegis Reborn...");

    // --- 1. Load Configuration ---
    let config = match Config::load(Path::new("config.toml")) {
        Ok(c) => {
            log::info!("Configuration loaded successfully.");
            c
        }
        Err(e) => {
            log::error!("Failed to load configuration: {}", e);
            return;
        }
    };

    // --- 2. Load Grammar ---
    let grammar = match Grammar::new(Path::new(&config.grammar_file)) {
        Ok(g) => {
            log::info!("Grammar '{}' loaded and validated.", config.grammar_file);
            g
        }
        Err(e) => {
            log::error!("Failed to load grammar: {}", e);
            return;
        }
    };

    // --- 3. Create Mapper ---
    let mapper = GrammarBasedMapper::new(
        &grammar,
        config.max_program_tokens,
        config.max_recursion_depth,
    );
    log::info!("GrammarBasedMapper initialized.");

    // --- 4. Perform Multiple Test Mappings ---
    log::info!("--- Starting Mapper Smoke Test ---");
    let mut rng = rand::rng();

    for i in 0..3 {
        println!(); // Add a blank line for readability

        // --- 4a. Generate a Random Genome ---
        let genome: Genome = (0..50).map(|_| rng.random_range(0..=u32::MAX)).collect();
        log::info!(
            "[Run {}/3] Generated test genome (first 5 codons): {:?}",
            i + 1,
            &genome[..5.min(genome.len())]
        );

        // --- 4b. Perform a Test Mapping ---
        match mapper.map(&genome) {
            Ok(strategy) => {
                log::info!("[Run {}/3] Successfully mapped genome to strategy!", i + 1);

                // --- 4c. Basic Strategy Validation ---
                let program_keys: Vec<String> = strategy.programs.keys().cloned().collect();
                log::info!("[Run {}/3] Programs generated: {:?}", i + 1, program_keys);

                if strategy.programs.contains_key("entry") {
                    log::info!("[Run {}/3] ✓ Entry program found.", i + 1);
                } else {
                    log::warn!(
                        "[Run {}/3] ⚠ CRITICAL: No entry program was generated. This strategy is invalid.",
                        i + 1
                    );
                }

                println!("[Run {}/3] Full Strategy Details:\n{:#?}", i + 1, strategy);
            }
            Err(e) => {
                log::error!("[Run {}/3] Failed to map genome: {}", i + 1, e);
            }
        }
    }

    log::info!("--- Mapper Smoke Test Complete ---");
}
