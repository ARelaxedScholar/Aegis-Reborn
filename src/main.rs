use aegis_reborn;
use aegis_reborn::config::Config;
use std::path::Path;

fn main() {
    // Initialize the logger. Control verbosity with `RUST_LOG=info` env var.
    env_logger::init();

    log::info!("Booting Aegis Reborn...");

    let config = match Config::load(Path::new("config.toml")) {
        Ok(c) => {
            log::info!("Aegis Reborn - A new era of strategy evolution.");
            log::info!("Configuration loaded successfully.");
            c
        }
        Err(e) => {
            log::error!("Failed to load configuration file 'config.toml': {}", e);
            return;
        }
    };

    log::debug!("Running with config: {:?}", config);
    // The main application logic (e.g., starting the evolution) will be called from here.
}
