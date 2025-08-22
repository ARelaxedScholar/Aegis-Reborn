use log::warn;
use polars::prelude::*;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("Failed to read or parse CSV file: {0}")]
    CsvError(#[from] PolarsError),
    #[error("CSV file is missing required columns: '{0}'")]
    MissingColumns(String),
    #[error("Invalid OHLCV data at row {row}: {reason}")]
    ValidationError { row: usize, reason: String },
    #[error("Found {count} null values in critical columns")]
    NullDataError { count: usize },
    #[error("Invalid date at row {row}: '{date}' (expected YYYY-MM-DD format)")]
    InvalidDate { row: usize, date: String },
    #[error("Invalid timestamp at row {row}: {timestamp} (outside reasonable range)")]
    InvalidTimestamp { row: usize, timestamp: i64 },
}

/// Configuration for data loading and validation
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Whether to fail on null values or use defaults
    pub fail_on_nulls: bool,
    /// Whether to validate OHLCV constraints (high >= low, etc.)
    pub validate_ohlcv_constraints: bool,
    /// Whether to validate timestamp ranges
    pub validate_timestamps: bool,
    /// Minimum valid timestamp (Unix epoch)
    pub min_timestamp: i64,
    /// Maximum valid timestamp (Unix epoch)
    pub max_timestamp: i64,
    /// Use adjusted close instead of close price
    pub use_adjusted_close: bool,
    /// Expected date column name
    pub date_column: String,
    /// Expected volume column name  
    pub volume_column: String,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            fail_on_nulls: true,
            validate_ohlcv_constraints: true,
            validate_timestamps: true,
            min_timestamp: 946684800,  // 2000-01-01
            max_timestamp: 4102444800, // 2100-01-01
            use_adjusted_close: true,
            date_column: "Date".to_string(),
            volume_column: "Volume".to_string(),
        }
    }
}

/// Represents a single OHLCV candle.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OHLCV {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OHLCV {
    /// Validates OHLCV constraints
    pub fn validate(&self) -> Result<(), String> {
        if self.high < self.low {
            return Err(format!(
                "High ({}) cannot be less than low ({})",
                self.high, self.low
            ));
        }

        if self.high < self.open || self.high < self.close {
            return Err(format!(
                "High ({}) cannot be less than open ({}) or close ({})",
                self.high, self.open, self.close
            ));
        }

        if self.low > self.open || self.low > self.close {
            return Err(format!(
                "Low ({}) cannot be greater than open ({}) or close ({})",
                self.low, self.open, self.close
            ));
        }

        if self.volume < 0.0 {
            return Err(format!("Volume ({}) cannot be negative", self.volume));
        }

        // Check for reasonable price values (not NaN, not infinite)
        let prices = [self.open, self.high, self.low, self.close];
        for price in &prices {
            if !price.is_finite() {
                return Err(format!("Invalid price value: {}", price));
            }
            if *price < 0.0 {
                return Err(format!("Price ({}) cannot be negative", price));
            }
        }

        Ok(())
    }
}

/// Converts a date string (YYYY-MM-DD) to Unix timestamp
fn parse_date_to_timestamp(date_str: &str) -> Result<i64, String> {
    // Simple parsing for YYYY-MM-DD format without external dependencies
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        return Err(format!(
            "Invalid date format: {}, expected YYYY-MM-DD",
            date_str
        ));
    }

    let year: i32 = parts[0]
        .parse()
        .map_err(|_| format!("Invalid year in date: {}", date_str))?;
    let month: u32 = parts[1]
        .parse()
        .map_err(|_| format!("Invalid month in date: {}", date_str))?;
    let day: u32 = parts[2]
        .parse()
        .map_err(|_| format!("Invalid day in date: {}", date_str))?;

    // Basic validation
    if month < 1 || month > 12 {
        return Err(format!("Invalid month {} in date: {}", month, date_str));
    }
    if day < 1 || day > 31 {
        return Err(format!("Invalid day {} in date: {}", day, date_str));
    }

    // Convert to Unix timestamp (days since 1970-01-01)
    // This is a simplified calculation - for production use a proper date library
    let days_since_1970 = days_since_epoch(year, month, day);
    Ok(days_since_1970 * 24 * 60 * 60) // Convert days to seconds
}

/// Calculate days since Unix epoch (1970-01-01)
/// Simplified calculation - doesn't handle all edge cases perfectly
fn days_since_epoch(year: i32, month: u32, day: u32) -> i64 {
    let mut days = 0i64;

    // Add days for complete years since 1970
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }

    // Add days for complete months in the current year
    let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        days += days_in_month[(m - 1) as usize] as i64;
        if m == 2 && is_leap_year(year) {
            days += 1; // Add leap day
        }
    }

    // Add remaining days
    days + (day - 1) as i64
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Detects the column mapping for the CSV file
fn detect_columns(
    df: &DataFrame,
) -> Result<(String, String, String, String, String, String), DataError> {
    let columns: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Map common column name variations
    let date_col = [
        "Date",
        "date",
        "DATE",
        "timestamp",
        "Timestamp",
        "TIMESTAMP",
    ]
    .iter()
    .find(|&&col| columns.iter().any(|c| c == col))
    .map(|s| s.to_string())
    .ok_or_else(|| DataError::MissingColumns("Date/timestamp column".to_string()))?;

    let open_col = ["Open", "open", "OPEN"]
        .iter()
        .find(|&&col| columns.iter().any(|c| c == col))
        .map(|s| s.to_string())
        .ok_or_else(|| DataError::MissingColumns("Open column".to_string()))?;

    let high_col = ["High", "high", "HIGH"]
        .iter()
        .find(|&&col| columns.iter().any(|c| c == col))
        .map(|s| s.to_string())
        .ok_or_else(|| DataError::MissingColumns("High column".to_string()))?;

    let low_col = ["Low", "low", "LOW"]
        .iter()
        .find(|&&col| columns.iter().any(|c| c == col))
        .map(|s| s.to_string())
        .ok_or_else(|| DataError::MissingColumns("Low column".to_string()))?;

    // For close, prefer Adj Close if available, otherwise use Close
    let close_col = if columns.iter().any(|c| c == "Adj Close") {
        "Adj Close".to_string()
    } else if columns.iter().any(|c| c == "Close") {
        "Close".to_string()
    } else if columns.iter().any(|c| c == "close") {
        "close".to_string()
    } else if columns.iter().any(|c| c == "CLOSE") {
        "CLOSE".to_string()
    } else {
        return Err(DataError::MissingColumns(
            "Close/Adj Close column".to_string(),
        ));
    };

    let volume_col = ["Volume", "volume", "VOLUME", "Vol", "vol"]
        .iter()
        .find(|&&col| columns.iter().any(|c| c == col))
        .map(|s| s.to_string())
        .ok_or_else(|| DataError::MissingColumns("Volume column".to_string()))?;

    Ok((date_col, open_col, high_col, low_col, close_col, volume_col))
}

/// Loads OHLCV data from a CSV file into a Vec with enhanced validation and performance.
pub fn load_csv(file_path: &Path) -> Result<Vec<OHLCV>, DataError> {
    load_csv_with_config(file_path, LoadConfig::default())
}

/// Loads OHLCV data with custom configuration
pub fn load_csv_with_config(file_path: &Path, config: LoadConfig) -> Result<Vec<OHLCV>, DataError> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    // Auto-detect column names
    let (date_col, open_col, high_col, low_col, close_col, volume_col) = detect_columns(&df)?;

    // Extract columns - handle both string dates and numeric timestamps
    let date_series = df.column(&date_col)?;
    let open_col_data = df.column(&open_col)?.f64()?;
    let high_col_data = df.column(&high_col)?.f64()?;
    let low_col_data = df.column(&low_col)?.f64()?;
    let close_col_data = df.column(&close_col)?.f64()?;
    let binding = df.column(&volume_col)?.cast(&DataType::Float64)?;
    let volume_col_data = binding.f64()?;

    // Check for null values if configured to fail on them
    if config.fail_on_nulls {
        let null_count = date_series.null_count()
            + open_col_data.null_count()
            + high_col_data.null_count()
            + low_col_data.null_count()
            + close_col_data.null_count()
            + volume_col_data.null_count();

        if null_count > 0 {
            return Err(DataError::NullDataError { count: null_count });
        }
    }

    let mut candles = Vec::with_capacity(df.height());

    // Extract data using iterators for better performance
    let opens: Vec<Option<f64>> = open_col_data.iter().collect();
    let highs: Vec<Option<f64>> = high_col_data.iter().collect();
    let lows: Vec<Option<f64>> = low_col_data.iter().collect();
    let closes: Vec<Option<f64>> = close_col_data.iter().collect();
    let volumes: Vec<Option<f64>> = volume_col_data.iter().collect();

    for i in 0..df.height() {
        // Handle timestamp conversion - support both string dates and numeric timestamps
        let timestamp = if let Ok(date_str_series) = date_series.str() {
            // Handle string dates like "2014-09-17"
            let date_opt = date_str_series.get(i);
            match date_opt {
                Some(date_str) => {
                    parse_date_to_timestamp(date_str).map_err(|_| DataError::InvalidDate {
                        row: i,
                        date: date_str.to_string(),
                    })?
                }
                None => {
                    if config.fail_on_nulls {
                        return Err(DataError::NullDataError { count: 1 });
                    } else {
                        warn!("Null date at row {}, using timestamp 0", i);
                        0
                    }
                }
            }
        } else if let Ok(timestamp_series) = date_series.i64() {
            // Handle numeric timestamps
            timestamp_series.get(i).unwrap_or_else(|| {
                if !config.fail_on_nulls {
                    warn!("Null timestamp at row {}, using 0", i);
                }
                0
            })
        } else {
            return Err(DataError::CsvError(PolarsError::ComputeError(
                "Date column is neither string nor numeric".into(),
            )));
        };

        let open = opens[i].unwrap_or_else(|| {
            if !config.fail_on_nulls {
                warn!("Null open price at row {}, using 0.0", i);
            }
            0.0
        });

        let high = highs[i].unwrap_or_else(|| {
            if !config.fail_on_nulls {
                warn!("Null high price at row {}, using 0.0", i);
            }
            0.0
        });

        let low = lows[i].unwrap_or_else(|| {
            if !config.fail_on_nulls {
                warn!("Null low price at row {}, using 0.0", i);
            }
            0.0
        });

        let close = closes[i].unwrap_or_else(|| {
            if !config.fail_on_nulls {
                warn!("Null close price at row {}, using 0.0", i);
            }
            0.0
        });

        let volume = volumes[i].unwrap_or_else(|| {
            if !config.fail_on_nulls {
                warn!("Null volume at row {}, using 0.0", i);
            }
            0.0
        });

        // Validate timestamp range if configured
        if config.validate_timestamps {
            if timestamp < config.min_timestamp || timestamp > config.max_timestamp {
                return Err(DataError::InvalidTimestamp { row: i, timestamp });
            }
        }

        let candle = OHLCV {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        };

        // Validate OHLCV constraints if configured
        if config.validate_ohlcv_constraints {
            if let Err(reason) = candle.validate() {
                return Err(DataError::ValidationError { row: i, reason });
            }
        }

        candles.push(candle);
    }

    Ok(candles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_load_csv_with_date_strings() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "Date,Open,High,Low,Close,Adj Close,Volume").unwrap();
        writeln!(
            file,
            "2014-09-17,465.864014,468.174011,452.421997,457.334015,457.334015,21056800"
        )
        .unwrap();
        writeln!(
            file,
            "2014-09-18,456.859985,456.859985,413.104004,424.440002,424.440002,34483200"
        )
        .unwrap();

        let candles = load_csv(&file_path).unwrap();

        assert_eq!(candles.len(), 2);
        // Verify timestamp conversion (2014-09-17 should be around 1410912000)
        assert!(candles[0].timestamp > 1410000000 && candles[0].timestamp < 1420000000);
        assert_eq!(candles[0].close, 457.334015); // Should use Adj Close
        assert_eq!(candles[0].volume, 21056800.0);
    }

    #[test]
    fn test_load_csv_case_insensitive_columns() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lowercase_data.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "date,open,high,low,close,volume").unwrap();
        writeln!(file, "2014-09-17,100.0,105.0,99.0,101.0,1000").unwrap();

        let candles = load_csv(&file_path).unwrap();
        assert_eq!(candles.len(), 1);
        assert_eq!(candles[0].close, 101.0);
    }

    #[test]
    fn test_prefer_adj_close() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adj_close_data.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "Date,Open,High,Low,Close,Adj Close,Volume").unwrap();
        writeln!(file, "2014-09-17,100.0,105.0,99.0,101.0,102.0,1000").unwrap();

        let candles = load_csv(&file_path).unwrap();
        assert_eq!(candles[0].close, 102.0); // Should prefer Adj Close (102.0) over Close (101.0)
    }

    #[test]
    fn test_missing_columns_detection() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("missing_data.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "Date,Open,High,Low,Volume").unwrap(); // Missing Close/Adj Close
        writeln!(file, "2014-09-17,100.0,105.0,99.0,1000").unwrap();

        let result = load_csv(&file_path);
        assert!(matches!(result, Err(DataError::MissingColumns(_))));
    }

    #[test]
    fn test_invalid_date_format() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("invalid_date.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "Date,Open,High,Low,Close,Volume").unwrap();
        writeln!(file, "not-a-date,100.0,105.0,99.0,101.0,1000").unwrap();

        let result = load_csv(&file_path);
        assert!(matches!(result, Err(DataError::InvalidDate { .. })));
    }

    #[test]
    fn test_numeric_timestamp_fallback() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("numeric_timestamp.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,Open,High,Low,Close,Volume").unwrap();
        writeln!(file, "1672531200,100.0,105.0,99.0,101.0,1000").unwrap();

        let candles = load_csv(&file_path).unwrap();
        assert_eq!(candles[0].timestamp, 1672531200);
    }

    #[test]
    fn test_date_to_timestamp_conversion() {
        assert!(parse_date_to_timestamp("2014-09-17").is_ok());
        assert!(parse_date_to_timestamp("invalid-date").is_err());

        let timestamp = parse_date_to_timestamp("2014-09-17").unwrap();
        // Should be around September 17, 2014 (1410912000 is roughly correct)
        assert!(timestamp > 1410000000 && timestamp < 1420000000);
    }

    #[test]
    fn test_ohlcv_validation() {
        let valid_candle = OHLCV {
            timestamp: 1672531200,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        };
        assert!(valid_candle.validate().is_ok());

        let invalid_candle = OHLCV {
            timestamp: 1672531200,
            open: 100.0,
            high: 90.0, // Invalid: high < low
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        };
        assert!(invalid_candle.validate().is_err());
    }

    #[test]
    fn test_real_world_data_format() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("real_data.csv");
        let mut file = File::create(&file_path).unwrap();

        // Copy your exact data format
        writeln!(file, "Date,Open,High,Low,Close,Adj Close,Volume").unwrap();
        writeln!(
            file,
            "2014-09-17,465.864014,468.174011,452.421997,457.334015,457.334015,21056800"
        )
        .unwrap();
        writeln!(
            file,
            "2014-09-18,456.859985,456.859985,413.104004,424.440002,424.440002,34483200"
        )
        .unwrap();
        writeln!(
            file,
            "2014-09-19,424.102997,427.834991,384.532013,394.795990,394.795990,37919700"
        )
        .unwrap();

        let candles = load_csv(&file_path).unwrap();

        assert_eq!(candles.len(), 3);
        assert_eq!(candles[0].open, 465.864014);
        assert_eq!(candles[0].high, 468.174011);
        assert_eq!(candles[0].low, 452.421997);
        assert_eq!(candles[0].close, 457.334015); // Uses Adj Close
        assert_eq!(candles[0].volume, 21056800.0);

        // Verify dates are converted properly (should be in chronological order)
        assert!(candles[0].timestamp < candles[1].timestamp);
        assert!(candles[1].timestamp < candles[2].timestamp);
    }

    #[test]
    fn test_config_options() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("config_test.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "Date,Open,High,Low,Close,Adj Close,Volume").unwrap();
        writeln!(file, "2014-09-17,100.0,90.0,95.0,101.0,102.0,1000").unwrap(); // Invalid OHLCV

        // Test with validation disabled
        let config = LoadConfig {
            validate_ohlcv_constraints: false,
            ..LoadConfig::default()
        };

        let result = load_csv_with_config(&file_path, config);
        assert!(result.is_ok());

        // Test with validation enabled (default)
        let result = load_csv(&file_path);
        assert!(matches!(result, Err(DataError::ValidationError { .. })));
    }
}
