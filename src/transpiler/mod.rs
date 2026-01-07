//! Transpiler module for converting evolved strategies to production trading algorithms.
//!
//! This module provides functionality to transpile `Strategy` objects (VM bytecode)
//! into QuantConnect Python and C# algorithms ready for deployment.
//!
//! # Example
//! ```no_run
//! use golden_aegis::config::{Config, TranspilerConfig};
//! use golden_aegis::transpiler::TranspilerEngine;
//! use golden_aegis::strategy::Strategy;
//! use std::path::Path;
//!
//! // Load configuration
//! let config = Config::load(Path::new("config.toml")).unwrap();
//! let transpiler_config = config.get_transpiler_config();
//!
//! // Create transpiler engine
//! let engine = TranspilerEngine::new(transpiler_config);
//!
//! // Transpile a strategy (obtained from evolution)
//! let strategy = Strategy::new(); // In practice, obtained from evolution
//! let python_code = engine.to_python(&strategy);
//! let csharp_code = engine.to_c_sharp(&strategy);
//! // Results can be unwrapped or handled appropriately
//! ```
//!
//! The generated algorithms include proper indicator initialization, warm-up periods,
//! and trading logic that matches the evolved strategy's behavior.

pub mod quantconnect;

pub use quantconnect::{TranspilerEngine, TranspilerError};
