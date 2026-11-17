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

#[cfg(test)]
mod tests {
    use super::*;

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
