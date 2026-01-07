//! Transpiler module for converting evolved strategies to production trading algorithms.
//!
//! This module provides functionality to transpile `Strategy` objects (VM bytecode)
//! into QuantConnect Python and C# algorithms ready for deployment.
//!
//! # Example
//! ```
//! use golden_aegis::config::{Config, TranspilerConfig};
//! use golden_aegis::transpiler::TranspilerEngine;
//! use golden_aegis::strategy::Strategy;
//!
//! // Load configuration
//! let config = Config::load("config.toml").unwrap();
//! let transpiler_config = config.get_transpiler_config();
//!
//! // Create transpiler engine
//! let engine = TranspilerEngine::new(transpiler_config);
//!
//! // Transpile a strategy
//! let strategy: Strategy = ...; // obtained from evolution
//! let python_code = engine.to_python(&strategy).unwrap();
//! let csharp_code = engine.to_c_sharp(&strategy).unwrap();
//! ```
//!
//! The generated algorithms include proper indicator initialization, warm-up periods,
//! and trading logic that matches the evolved strategy's behavior.

pub mod quantconnect;

pub use quantconnect::{TranspilerEngine, TranspilerError};
