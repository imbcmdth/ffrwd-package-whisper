//! The row both exports emit: a cue, which is a piece of text and the seconds
//! it runs between.
//!
//! The two differ only in how much of the stream one row covers - a stretch of
//! speech for `transcribe`, a single word for `transcribe_words` - so the
//! columns, their order and the schema that declares them are the same for
//! both, and a query can read either the same way.

use serde::Serialize;

/// The schema both exports publish for their rows.
pub const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"text":{"type":"string"},"start_t":{"type":"number"},"end_t":{"type":"number"}},"required":["text","start_t","end_t"],"additionalProperties":false}"#;

/// One cue, as the row it becomes. `start_t` and `end_t` are seconds from the
/// start of the stream, not of the window that produced it.
#[derive(Serialize)]
pub struct Cue {
    pub text: String,
    pub start_t: f64,
    pub end_t: f64,
}

impl Cue {
    /// The NDJSON line this cue leaves as.
    pub fn row(&self) -> String {
        serde_json::to_string(self).expect("row serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cue_is_the_three_columns_it_declares_and_nothing_else() {
        let written = Cue {
            text: "hola".to_string(),
            start_t: 31.5,
            end_t: 32.25,
        }
        .row();
        assert_eq!(written, r#"{"text":"hola","start_t":31.5,"end_t":32.25}"#);
    }

    #[test]
    fn the_schema_requires_exactly_the_columns_a_cue_carries() {
        let schema: serde_json::Value = serde_json::from_str(ROWS_SCHEMA).expect("valid json");
        assert_eq!(
            schema["required"],
            serde_json::json!(["text", "start_t", "end_t"])
        );
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
    }
}
