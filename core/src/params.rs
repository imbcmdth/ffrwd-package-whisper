//! The parameters a decode is steered by, which both exports take: the
//! language, the task, and what the cues come out in.
//!
//! Both modules take the same list, so a query can swap one for the other
//! without rewriting its arguments, and they refuse the same things for the
//! same reasons. `module` is the name the refusal opens on, and the only thing
//! that differs between them.

use serde::Deserialize;

use crate::tokens::{self, Task};

/// The schema both exports publish for their parameters.
pub const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"language":{"type":["string","null"]},"task":{"type":"string","enum":["transcribe","translate"],"default":"transcribe"},"language_out":{"type":["string","null"]}},"additionalProperties":false}"#;

fn default_task() -> String {
    "transcribe".to_string()
}

/// The parameters as they arrive: what a caller wrote, before any of it is
/// known to name anything.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Written {
    #[serde(default)]
    language: Option<String>,
    #[serde(default = "default_task")]
    task: String,
    #[serde(default)]
    language_out: Option<String>,
}

impl Default for Written {
    fn default() -> Written {
        Written {
            language: None,
            task: default_task(),
            language_out: None,
        }
    }
}

/// The parameters as the decode uses them.
#[derive(Debug)]
pub struct Params {
    /// The forced language token, or None to detect one per window.
    pub language: Option<u32>,
    pub task: Task,
}

/// Parses and validates params, shared by `init` and `set_params`.
pub fn parse_params(module: &str, params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let written: Written = if trimmed.is_empty() {
        Written::default()
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("{module}: bad params: {e}"))?
    };

    let task = Task::named(&written.task).ok_or_else(|| {
        format!(
            "{module}: '{}' is no task; whisper does 'transcribe' or 'translate'",
            written.task
        )
    })?;

    let language = match written.language.as_deref() {
        None => None,
        Some(code) => Some(tokens::language_token(code).ok_or_else(|| {
            format!("{module}: whisper was not trained on the language '{code}'")
        })?),
    };

    // `language_out` says what the CUES are in, which is what tags the track
    // the rows mint. Translation always hands back English; transcription
    // hands back what it heard, so naming a third thing there would tag the
    // track with a language nothing in the query produces.
    if let Some(out) = written.language_out.as_deref() {
        if tokens::language_token(out).is_none() {
            return Err(format!(
                "{module}: whisper was not trained on the language '{out}'"
            ));
        }
        match task {
            Task::Translate if out != "en" => {
                return Err(format!(
                    "{module}: task 'translate' hands back English, and \
                     language_out names '{out}'"
                ));
            }
            Task::Transcribe => {
                if let Some(heard) = written.language.as_deref() {
                    if out != heard {
                        return Err(format!(
                            "{module}: task 'transcribe' hands back what it heard, \
                             and language is '{heard}' while language_out is '{out}'"
                        ));
                    }
                }
            }
            Task::Translate => {}
        }
    }

    Ok(Params { language, task })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusals name whichever export was asked, so the message a query
    /// sees is about the function it actually called.
    #[test]
    fn a_refusal_opens_on_the_name_of_the_export_that_refused() {
        let one = parse_params("transcribe", r#"{"language":"klingon"}"#).unwrap_err();
        let other = parse_params("transcribe_words", r#"{"language":"klingon"}"#).unwrap_err();
        assert_eq!(
            one,
            "transcribe: whisper was not trained on the language 'klingon'"
        );
        assert_eq!(
            other,
            "transcribe_words: whisper was not trained on the language 'klingon'"
        );
    }

    #[test]
    fn the_schema_names_the_three_parameters_and_no_others() {
        let schema: serde_json::Value = serde_json::from_str(PARAMS_SCHEMA).expect("valid json");
        let properties = schema["properties"].as_object().expect("an object");
        let mut named: Vec<&String> = properties.keys().collect();
        named.sort();
        assert_eq!(named, ["language", "language_out", "task"]);
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
    }
}
