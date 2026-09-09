//! Shared helpers for decoding a tool call's JSON-encoded arguments, so the
//! "failed to parse" / "missing argument" errors have one spelling.

use serde_json::Value;

use crate::error::{Error, Result};

/// Parses the JSON argument string an LLM attached to a tool call.
pub fn parse_args(args: &str) -> Result<Value> {
    serde_json::from_str(args)
        .map_err(|e| Error::ParseError(format!("failed to parse arguments: {e}")))
}

/// Returns the string under `key`, or a `ParseError` naming the missing
/// argument (the message the LLM sees, so it can correct the call).
pub fn require_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args[key]
        .as_str()
        .ok_or_else(|| Error::ParseError(format!("missing '{key}' argument")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_rejects_malformed_json() {
        let err = parse_args("not json").unwrap_err().to_string();
        assert!(err.contains("failed to parse arguments"), "{err}");
    }

    #[test]
    fn require_str_names_the_missing_key() {
        let v = parse_args(r#"{"path": "a", "n": 1}"#).unwrap();
        assert_eq!(require_str(&v, "path").unwrap(), "a");
        let err = require_str(&v, "content").unwrap_err().to_string();
        assert!(err.contains("missing 'content' argument"), "{err}");
        // A present-but-wrong-type value is reported the same way.
        assert!(require_str(&v, "n").is_err());
    }
}
