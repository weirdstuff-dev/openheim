//! Shared helpers for decoding a tool call's JSON-encoded arguments, so the
//! "failed to parse" / "missing argument" errors have one spelling.
//!
//! Built-in tools decode into a `#[derive(Deserialize)]` struct via
//! [`parse`]; [`parse_args`]/[`require_str`] remain for custom tools that
//! prefer working with a raw [`Value`].

use std::ops::Deref;

use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use crate::error::{Error, Result};

/// Decodes a tool call's JSON arguments into `T`. On failure the error names
/// what was wrong (`missing field `path``, `invalid type: …`), which is the
/// message the LLM sees, so it can correct the call.
pub fn parse<T: DeserializeOwned>(args: &str) -> Result<T> {
    serde_json::from_str(args)
        .map_err(|e| Error::ParseError(format!("failed to parse arguments: {e}")))
}

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

/// A string argument that must contain non-whitespace text; stored trimmed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonEmptyString(String);

impl NonEmptyString {
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl Deref for NonEmptyString {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for NonEmptyString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(D::Error::custom(
                "expected non-empty text, got an empty string",
            ));
        }
        Ok(Self(trimmed.to_string()))
    }
}

/// Test helper keeping a tool's hand-written JSON schema and its args struct
/// `T` in sync. `example` must set every property the schema declares. Checks
/// that the example parses, that the schema and the example name the same
/// properties, and that dropping a property fails to parse exactly when the
/// schema lists it under `required`.
#[cfg(test)]
pub(crate) fn assert_args_match_schema<T: DeserializeOwned>(
    tool: &dyn super::ToolHandler,
    example: Value,
) {
    let definition = tool.definition();
    let name = definition.function.name;
    let schema = definition.function.parameters;
    let example = example.as_object().expect("example must be a JSON object");

    let mut declared: Vec<&String> = schema["properties"]
        .as_object()
        .map(|p| p.keys().collect())
        .unwrap_or_default();
    let mut given: Vec<&String> = example.keys().collect();
    declared.sort();
    given.sort();
    assert_eq!(
        declared, given,
        "{name}: example must set exactly the schema's properties"
    );

    let required: Vec<&str> = schema["required"]
        .as_array()
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let full = Value::Object(example.clone()).to_string();
    if let Err(e) = parse::<T>(&full) {
        panic!("{name}: complete example should parse: {e}");
    }

    for key in example.keys() {
        let mut without = example.clone();
        without.remove(key);
        let parsed = parse::<T>(&Value::Object(without).to_string());
        let is_required = required.contains(&key.as_str());
        assert_eq!(
            parsed.is_err(),
            is_required,
            "{name}: dropping '{key}' should {} parsing, per the schema's `required`",
            if is_required { "fail" } else { "not fail" },
        );
    }
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

    #[derive(Deserialize)]
    struct Sample {
        path: String,
        #[serde(default)]
        recursive: bool,
    }

    #[test]
    fn parse_names_the_missing_field() {
        let err = parse::<Sample>(r#"{"recursive": true}"#)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("missing field `path`"), "{err}");
        let ok: Sample = parse(r#"{"path": "a"}"#).unwrap();
        assert_eq!(ok.path, "a");
        assert!(!ok.recursive);
    }

    #[test]
    fn parse_rejects_malformed_json() {
        let err = parse::<Sample>("not json").err().unwrap().to_string();
        assert!(err.contains("failed to parse arguments"), "{err}");
    }

    #[test]
    fn non_empty_string_trims_and_rejects_blank() {
        #[derive(Deserialize)]
        struct Note {
            content: NonEmptyString,
        }
        let note: Note = parse(r#"{"content": "  hi  "}"#).unwrap();
        assert_eq!(&*note.content, "hi");
        let err = parse::<Note>(r#"{"content": "   "}"#)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("non-empty"), "{err}");
    }
}
