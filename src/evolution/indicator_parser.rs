use crate::vm::op::IndicatorType;
use log::warn;
use regex::Regex;

/// Parses indicator terminals like "SMA20", "SMA(20)", "RSI14", "RSI(14)".
/// Supports both old style (no parentheses) and new style (with parentheses).
/// Returns None if the terminal cannot be parsed as an indicator.
/// Emits deprecation warning when old style is used.
pub fn parse_indicator_terminal(terminal: &str) -> Option<IndicatorType> {
    lazy_static::lazy_static! {
        static ref INDICATOR_REGEX: Regex = Regex::new(r"^(?P<name>SMA|RSI|EMA|DC_UPPER|DC_LOWER|DC_MIDDLE)\((?P<period>\d+)\)$").unwrap();
        static ref BB_REGEX: Regex = Regex::new(r"^(?P<name>BB_UPPER|BB_LOWER)\((?P<period>\d+),(?P<std_dev>\d+)\)$").unwrap();
        static ref OLD_STYLE_REGEX: Regex = Regex::new(r"^(?P<name>SMA|RSI|EMA)(?P<period>\d+)$").unwrap();
    }

    // Try BB style first (two args)
    if let Some(caps) = BB_REGEX.captures(terminal) {
        let name = caps.name("name").unwrap().as_str();
        let period = caps.name("period").unwrap().as_str().parse::<u16>().ok()?;
        let std_dev = caps.name("std_dev").unwrap().as_str().parse::<u8>().ok()?;
        return match name {
            "BB_UPPER" => Some(IndicatorType::BbUpper(period, std_dev)),
            "BB_LOWER" => Some(IndicatorType::BbLower(period, std_dev)),
            _ => None,
        };
    }

    // Try standard single-arg style
    if let Some(caps) = INDICATOR_REGEX.captures(terminal) {
        let name = caps.name("name").unwrap().as_str();
        let period = caps.name("period").unwrap().as_str().parse::<u16>().ok()?;
        return match name {
            "SMA" => Some(IndicatorType::Sma(period)),
            "RSI" => Some(IndicatorType::Rsi(period)),
            "EMA" => Some(IndicatorType::Ema(period)),
            "DC_UPPER" => Some(IndicatorType::DcUpper(period)),
            "DC_LOWER" => Some(IndicatorType::DcLower(period)),
            "DC_MIDDLE" => Some(IndicatorType::DcMiddle(period)),
            _ => None,
        };
    }

    // Try old style
    if let Some(caps) = OLD_STYLE_REGEX.captures(terminal) {
        let name = caps.name("name").unwrap().as_str();
        let period = caps.name("period").unwrap().as_str().parse::<u16>().ok()?;
        warn!(
            "Deprecated indicator syntax '{}'. Please use '{}({})' instead.",
            terminal, name, period
        );
        return match name {
            "SMA" => Some(IndicatorType::Sma(period)),
            "RSI" => Some(IndicatorType::Rsi(period)),
            "EMA" => Some(IndicatorType::Ema(period)),
            _ => None,
        };
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_new_style_sma() {
        assert_eq!(
            parse_indicator_terminal("SMA(20)"),
            Some(IndicatorType::Sma(20))
        );
        assert_eq!(
            parse_indicator_terminal("SMA(100)"),
            Some(IndicatorType::Sma(100))
        );
        assert_eq!(
            parse_indicator_terminal("SMA(14)"),
            Some(IndicatorType::Sma(14))
        );
    }

    #[test]
    fn test_parse_new_style_rsi() {
        assert_eq!(
            parse_indicator_terminal("RSI(14)"),
            Some(IndicatorType::Rsi(14))
        );
        assert_eq!(
            parse_indicator_terminal("RSI(30)"),
            Some(IndicatorType::Rsi(30))
        );
    }

    #[test]
    fn test_parse_old_style_sma() {
        assert_eq!(
            parse_indicator_terminal("SMA20"),
            Some(IndicatorType::Sma(20))
        );
        assert_eq!(
            parse_indicator_terminal("SMA100"),
            Some(IndicatorType::Sma(100))
        );
    }

    #[test]
    fn test_parse_old_style_rsi() {
        assert_eq!(
            parse_indicator_terminal("RSI14"),
            Some(IndicatorType::Rsi(14))
        );
    }

    #[test]
    fn test_parse_ema_new_style() {
        assert_eq!(
            parse_indicator_terminal("EMA(14)"),
            Some(IndicatorType::Ema(14))
        );
        assert_eq!(
            parse_indicator_terminal("EMA(30)"),
            Some(IndicatorType::Ema(30))
        );
    }

    #[test]
    fn test_parse_ema_old_style() {
        assert_eq!(
            parse_indicator_terminal("EMA14"),
            Some(IndicatorType::Ema(14))
        );
        assert_eq!(
            parse_indicator_terminal("EMA30"),
            Some(IndicatorType::Ema(30))
        );
    }

    #[test]
    fn test_invalid_formats() {
        assert_eq!(parse_indicator_terminal("SMA"), None);
        assert_eq!(parse_indicator_terminal("SMA()"), None);
        assert_eq!(parse_indicator_terminal("SMA(20"), None);
        assert_eq!(parse_indicator_terminal("SMA20)"), None);
        assert_eq!(parse_indicator_terminal("SMA(20) extra"), None);
        assert_eq!(parse_indicator_terminal("MACD(12,26,9)"), None);
        assert_eq!(parse_indicator_terminal(""), None);
        assert_eq!(parse_indicator_terminal("123"), None);
    }
}
