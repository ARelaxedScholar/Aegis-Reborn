use polars::prelude::*;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("Failed to read or parse CSV file: {0}")]
    CsvError(#[from] PolarsError),
    #[error("CSV file is missing required column: '{0}'")]
    MissingColumn(String),
    #[error("CSV file is missing many required columns: '{0}'")]
    MissingColumns(String),
}

/// Represents a single OHLCV candle.
#[derive(Debug, Clone, Copy, Default)]
pub struct OHLCV {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Loads OHLCV data from a CSV file into a Vec.
///
/// This version is robust, validates all columns, and uses an efficient
/// approach to build the final Vec.
///
/// # Arguments
/// * `file_path` - The path to the CSV file.
///
/// # Expected Columns
/// "timestamp", "open", "high", "low", "close", "volume"
pub fn load_csv(file_path: &Path) -> Result<Vec<OHLCV>, DataError> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    // Validate that all required columns exist before proceeding.
    let required_cols = ["timestamp", "open", "high", "low", "close", "volume"];
    let mut missing_columns: Vec<&str> = Vec::new();

    for col_name in required_cols.iter() {
        if df.column(col_name).is_err() {
            missing_columns.push(col_name);
        }
    }

    if missing_columns.len() >= 2 {
        return Err(DataError::MissingColumns(missing_columns.join(", ")));
    } else if !missing_columns.is_empty() {
        return Err(DataError::MissingColumn(missing_columns[0].to_string()));
    }

    // Extract columns and convert to Vec<OHLCV>
    let timestamp_col = df.column("timestamp")?.i64()?;
    let open_col = df.column("open")?.f64()?;
    let high_col = df.column("high")?.f64()?;
    let low_col = df.column("low")?.f64()?;
    let close_col = df.column("close")?.f64()?;
    let volume_col = df.column("volume")?.f64()?;

    let mut candles = Vec::with_capacity(df.height());

    for i in 0..df.height() {
        let candle = OHLCV {
            timestamp: timestamp_col.get(i).unwrap_or(0),
            open: open_col.get(i).unwrap_or(0.0),
            high: high_col.get(i).unwrap_or(0.0),
            low: low_col.get(i).unwrap_or(0.0),
            close: close_col.get(i).unwrap_or(0.0),
            volume: volume_col.get(i).unwrap_or(0.0),
        };
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
    fn test_load_csv_successfully() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(file, "1672531200,100.0,105.0,99.0,101.0,1000.0").unwrap();
        writeln!(file, "1672534800,101.0,106.0,100.0,102.0,1200.0").unwrap();

        let candles = load_csv(&file_path).unwrap();

        assert_eq!(candles.len(), 2);
        assert_eq!(candles[0].timestamp, 1672531200);
        assert_eq!(candles[0].close, 101.0);
        assert_eq!(candles[1].open, 101.0);
    }

    #[test]
    fn test_load_csv_missing_column() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("bad_data.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,open,high,low,volume").unwrap(); // Missing 'close'
        writeln!(file, "1672531200,100.0,105.0,99.0,1000.0").unwrap();

        let result = load_csv(&file_path);
        assert!(matches!(result, Err(DataError::MissingColumn(ref c)) if c == "close"));
    }

    #[test]
    fn test_load_csv_with_invalid_data_types() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("invalid_types.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(file, "not_a_number,100.0,105.0,99.0,101.0,1000.0").unwrap();

        let result = load_csv(&file_path);
        assert!(matches!(result, Err(DataError::CsvError(_))));
    }

    #[test]
    fn test_load_csv_with_empty_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("empty.csv");
        File::create(&file_path).unwrap(); // Create empty file

        let result = load_csv(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_csv_header_only() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("header_only.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,open,high,low,close,volume").unwrap();
        // No data rows

        let candles = load_csv(&file_path).unwrap();
        assert_eq!(candles.len(), 0);
    }

    #[test]
    fn test_load_csv_multiple_missing_columns() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("multi_missing.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,open").unwrap(); // Missing 4 columns
        writeln!(file, "1672531200,100.0").unwrap();

        let result = load_csv(&file_path);
        assert!(matches!(result, Err(DataError::MissingColumns(_))));
    }

    #[test]
    fn test_load_csv_with_negative_values() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("negative_data.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(file, "1672531200,-100.0,105.0,99.0,101.0,1000.0").unwrap();

        let candles = load_csv(&file_path).unwrap();
        assert_eq!(candles[0].open, -100.0);
    }

    #[test]
    fn test_load_csv_with_zero_volume() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("zero_volume.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(file, "1672531200,100.0,105.0,99.0,101.0,0.0").unwrap();

        let candles = load_csv(&file_path).unwrap();
        assert_eq!(candles[0].volume, 0.0);
    }

    #[test]
    fn test_load_csv_file_not_found() {
        let result = load_csv(Path::new("nonexistent_file.csv"));
        assert!(matches!(result, Err(DataError::CsvError(_))));
    }

    #[test]
    fn test_load_csv_large_dataset() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("large_data.csv");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, "timestamp,open,high,low,close,volume").unwrap();

        // Generate 1000 rows
        for i in 0..1000 {
            writeln!(
                file,
                "{},100.0,105.0,99.0,101.0,1000.0",
                1672531200 + i * 3600
            )
            .unwrap();
        }

        let candles = load_csv(&file_path).unwrap();
        assert_eq!(candles.len(), 1000);
    }
}
