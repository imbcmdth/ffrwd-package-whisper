//! What was said: the audio passes through untouched and every stretch of
//! speech leaves as a cue with the words in it.
//!
//! The graph is whisper, exported so that the encoder, the decoder loop and
//! the beam search all sit INSIDE it - one `compute` per pass, with no decoder
//! state crossing the sandbox. The module never opens a file: the host binds
//! the graph to a name with `-nn transcribe=<path>` and this module asks for
//! that name and nothing else.
//!
//! # The window
//!
//! 30 s, stride the same, which is the model's fixed 3000-frame input. The
//! windows are disjoint and tile the stream, so the samples pass through as
//! the very ones that arrived, and a cue's times are the model's own offset by
//! the second the window starts at.
//!
//! # The two passes
//!
//! A decode is steered by the ids it is forced to open on. Given only
//! `<|startoftranscript|>` the model answers with the language it heard - and
//! a window with no speech in it answers with no language at all, which is the
//! only thing in the graph that can tell dialogue from a music bed. That is
//! the first pass, and it is cut short after a few tokens, so it costs a
//! fraction of a decode.
//!
//! The second pass names the language and the task, and ends on the first
//! timestamp, which is what puts the model in timestamp mode: without that id
//! it emits `<|notimestamps|>` itself and the answer carries no times at all.
//!
//! # The rows an upstream detector sends
//!
//! A window a voice detector already said holds speech needs no first pass
//! when the caller named the language: the question the first pass answers has
//! been answered. Rows arriving on a window are kept rather than read and
//! dropped, because a detector closes a span AFTER the speech in it - so a
//! span's row may arrive one window late, and it still counts for the window
//! it covers. Nothing is ever skipped on a row's absence: a span still open
//! upstream has emitted nothing yet, so silence is the model's own call and
//! never the detector's.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:whisper/transcribe",
    generate_all,
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};
use whisper_core::cue::{Cue, ROWS_SCHEMA};
use whisper_core::mel::{
    self, Plan, N_FRAMES, N_MELS, SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_SECONDS,
};
use whisper_core::params::{parse_params as parse_shared, Params, PARAMS_SCHEMA};
use whisper_core::tensor::{f32_bytes, i32_bytes, to_i32};
use whisper_core::tokens::{self, Segment, Vocab};
use whisper_core::{seconds, Speech};

/// The name the host binds the graph to. `-nn transcribe=<path>`.
const MODEL: &str = "transcribe";

/// What this export is called, which is what its refusals open on.
const ME: &str = "transcribe";

/// The graph's own names for the tensors it takes and returns.
const FEATURES: &str = "input_features";
const MAX_LENGTH: &str = "max_length";
const MIN_LENGTH: &str = "min_length";
const NUM_BEAMS: &str = "num_beams";
const NUM_RETURN_SEQUENCES: &str = "num_return_sequences";
const LENGTH_PENALTY: &str = "length_penalty";
const REPETITION_PENALTY: &str = "repetition_penalty";
const DECODER_INPUT_IDS: &str = "decoder_input_ids";
const SEQUENCES: &str = "sequences";

/// Beams the transcribing pass searches over. Whisper's own default, and what
/// the quality of a noisy window rests on; the module holds it rather than
/// publishing it, since a caller trading text for time is better served by a
/// smaller model than by a narrower search.
const BEAMS: i32 = 5;

/// The longest answer a transcribing pass may produce, which is half the
/// decoder's positions - whisper's own limit on one window.
const TRANSCRIBE_LENGTH: i32 = 224;

/// The longest answer the detecting pass may produce. The language is the
/// token after the prefix, so this is the prefix, the answer, and room for the
/// model to end.
const DETECT_LENGTH: i32 = 4;

/// Samples one call is handed.
const WINDOW: u32 = WINDOW_SAMPLES as u32;

/// Parses and validates params, shared by `init` and `set_params`. The list
/// and the refusals are `transcribe_words`' too, so they live in `core`.
fn parse_params(params: &str) -> Result<Params, String> {
    parse_shared(ME, params)
}

/// The spec's spelling of an error code, so a message says what actually went
/// wrong rather than how this module happens to format things. The mapping is
/// this crate's own because the enum is: each module's bindings mint a
/// separate copy of `ErrorCode`.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    whisper_core::tensor::failed(ME, what, code, &error.data())
}

/// A scalar the beam search takes as a one-element tensor.
fn scalar_i32(name: &str, value: i32) -> (String, Tensor) {
    (
        name.to_string(),
        Tensor::new(&[1], TensorType::I32, &i32_bytes(&[value])),
    )
}

fn scalar_f32(name: &str, value: f32) -> (String, Tensor) {
    (
        name.to_string(),
        Tensor::new(&[1], TensorType::Fp32, &f32_bytes(&[value])),
    )
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    /// The unit this stream's timestamps are counted in.
    time_base: (i32, i32),
    params: Params,
    plan: Plan,
    vocab: Vocab,
    /// What an upstream detector said about where the speech is.
    speech: Speech,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per window.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

impl Opened {
    /// One pass through the graph: the answer's token ids, with the forced
    /// prefix still on the front.
    fn pass(
        &self,
        mel: &[f32],
        prefix: &[i32],
        max_length: i32,
        beams: i32,
    ) -> Result<Vec<u32>, String> {
        let outputs = self
            .context
            .compute(vec![
                (
                    FEATURES.to_string(),
                    Tensor::new(
                        &[1, N_MELS as u32, N_FRAMES as u32],
                        TensorType::Fp32,
                        &f32_bytes(mel),
                    ),
                ),
                scalar_i32(MAX_LENGTH, max_length),
                scalar_i32(MIN_LENGTH, 1),
                scalar_i32(NUM_BEAMS, beams),
                scalar_i32(NUM_RETURN_SEQUENCES, 1),
                scalar_f32(LENGTH_PENALTY, 1.0),
                scalar_f32(REPETITION_PENALTY, 1.0),
                (
                    DECODER_INPUT_IDS.to_string(),
                    Tensor::new(
                        &[1, prefix.len() as u32],
                        TensorType::I32,
                        &i32_bytes(prefix),
                    ),
                ),
            ])
            .map_err(|e| failed("compute", &e))?;

        let sequences = outputs
            .iter()
            .find(|(name, _)| name == SEQUENCES)
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| format!("transcribe: the graph returned no tensor named {SEQUENCES}"))?;

        // The first returned sequence, cut at the end-of-text the beam search
        // pads the rest of the row out with.
        let ids = to_i32(&sequences.data());
        Ok(ids
            .iter()
            .take_while(|id| **id >= 0 && **id != tokens::EOT as i32)
            .map(|id| *id as u32)
            .collect())
    }

    /// The segments one window decodes to, empty when nothing was said in it.
    ///
    /// `known_speech` is an upstream detector's word that this window holds
    /// some; it stands in for the detecting pass only when the caller already
    /// named the language, since otherwise the pass is what supplies it.
    fn window(&self, samples: &[f32], known_speech: bool) -> Result<Vec<Segment>, String> {
        let mel = self.plan.spectrogram(samples);

        let language = match self.params.language {
            Some(named) if known_speech => Some(named),
            Some(named) => {
                let heard = self.pass(
                    &mel,
                    &tokens::prefix(None, self.params.task, false),
                    DETECT_LENGTH,
                    1,
                )?;
                tokens::detected(&heard).map(|_| named)
            }
            None => {
                let heard = self.pass(
                    &mel,
                    &tokens::prefix(None, self.params.task, false),
                    DETECT_LENGTH,
                    1,
                )?;
                tokens::detected(&heard).and_then(tokens::language_token)
            }
        };
        let Some(language) = language else {
            return Ok(Vec::new());
        };

        let said = self.pass(
            &mel,
            &tokens::prefix(Some(language), self.params.task, true),
            TRANSCRIBE_LENGTH,
            BEAMS,
        )?;
        Ok(self.vocab.segments(&said, WINDOW_SECONDS))
    }
}

struct Transcribe;

impl Guest for Transcribe {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "transcribe".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                // An audio module, so it names no pixel formats.
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string()],
                // What the model was trained on, and the host conforms to it.
                sample_rates: vec![SAMPLE_RATE],
                channel_counts: vec![1],
                // What the cues are in: the language they were turned into
                // when one was asked for, else the one they were heard in.
                rows_language: vec!["language_out".to_string(), "language".to_string()],
            },
            window: WINDOW,
            stride: WINDOW,
            // The spans an upstream detector sent carry from one call to the
            // next, since a span's row may arrive after the window it covers.
            pure: false,
            // The samples pass through as they arrived.
            one_to_one: true,
            reads_rows: true,
            // What leaves is this module's own cues and nothing else.
            forwards_rows: false,
            // One stream in: the audio it listens to.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Audio(audio) = format else {
            return Err("transcribe listens to samples, and this stream is video".to_string());
        };
        if audio.sample_fmt != "f32" {
            return Err(format!(
                "transcribe does not accept sample format {}",
                audio.sample_fmt
            ));
        }
        if audio.sample_rate != SAMPLE_RATE {
            return Err(format!(
                "transcribe listens at {SAMPLE_RATE} Hz, and this instance is {} Hz",
                audio.sample_rate
            ));
        }
        if audio.channels != 1 {
            return Err(format!(
                "transcribe listens in mono, and this instance has {} channels",
                audio.channels
            ));
        }
        let parsed = parse_params(&params)?;
        let vocab = Vocab::whisper().map_err(|e| format!("transcribe: {e}"))?;

        // The graph is loaded once per instance, and the session built once:
        // the first pass is what a provider picks its kernels on, and every
        // pass after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                time_base: (stream_info.time_base.num, stream_info.time_base.den),
                params: parsed,
                plan: Plan::new(),
                vocab,
                speech: Speech::new(),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.params = parsed;
            }
        });
        Ok(())
    }

    fn process(window: &InWindow, trailing: Vec<String>, _last: bool) -> Processed {
        OPENED.with(|o| {
            let mut borrowed = o.borrow_mut();
            let opened = borrowed
                .as_mut()
                .expect("init loads the graph before any audio arrives");

            // Whatever an upstream detector had nothing left to put its rows
            // on still says where the speech was.
            for row in &trailing {
                opened.speech.push(row);
            }

            let mut out: Vec<OutFrame> = Vec::with_capacity(window.len() as usize);
            let mut samples: Vec<f32> = Vec::new();
            let mut start = 0f64;
            for index in 0..window.len() {
                for row in window.rows(index) {
                    opened.speech.push(&row);
                }
                let pts = window.pts(index);
                if index == 0 {
                    start = seconds(pts, opened.time_base.0, opened.time_base.1);
                }
                let payload = window.fetch(index);
                let (bytes, _) = payload.as_chunks::<4>();
                samples.extend(bytes.iter().copied().map(f32::from_le_bytes));
                out.push(OutFrame {
                    pts,
                    frame: FramePayload::Same,
                    rows: Vec::new(),
                });
            }

            let end = start + samples.len() as f64 / f64::from(SAMPLE_RATE);
            let known = opened.speech.covers(start, end);
            // `process` has no way to say no, so a graph that failed
            // mid-stream stops the run rather than reporting silence.
            let said = opened
                .window(&samples, known)
                .unwrap_or_else(|message| panic!("{message}"));
            opened.speech.forget(end);

            // The cues ride the window that produced them, at the seconds that
            // window starts at.
            let rows: Vec<String> = said
                .into_iter()
                .map(|segment| row(segment, start))
                .collect();
            let mut trailing = Vec::new();
            match out.last_mut() {
                Some(frame) => frame.rows = rows,
                None => trailing = rows,
            }
            Processed {
                frames: out,
                trailing,
            }
        })
    }
}

/// One segment as the NDJSON line it leaves as, its times moved onto the
/// stream's own clock.
fn row(segment: Segment, start: f64) -> String {
    Cue {
        text: segment.text,
        start_t: start + segment.start,
        end_t: start + segment.end.min(mel::WINDOW_SECONDS),
    }
    .row()
}

export!(Transcribe);

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_core::tokens::Task;

    #[test]
    fn no_params_at_all_are_the_defaults() {
        for written in ["", "{}", "  "] {
            let parsed = parse_params(written).expect("the defaults");
            assert_eq!(parsed.language, None, "the model detects one per window");
            assert_eq!(parsed.task, Task::Transcribe);
        }
    }

    #[test]
    fn a_named_language_becomes_the_token_that_forces_it() {
        let parsed = parse_params(r#"{"language":"es"}"#).expect("spanish");
        assert_eq!(parsed.language, tokens::language_token("es"));
    }

    #[test]
    fn a_null_language_is_the_same_as_naming_none() {
        let parsed = parse_params(r#"{"language":null}"#).expect("no language");
        assert_eq!(parsed.language, None);
    }

    #[test]
    fn a_language_whisper_was_not_trained_on_is_refused() {
        assert!(parse_params(r#"{"language":"klingon"}"#).is_err());
        assert!(parse_params(r#"{"language":"ES"}"#).is_err());
    }

    #[test]
    fn translating_is_a_task_and_anything_else_is_refused() {
        assert_eq!(
            parse_params(r#"{"task":"translate"}"#)
                .expect("translate")
                .task,
            Task::Translate
        );
        assert!(parse_params(r#"{"task":"summarize"}"#).is_err());
        assert!(parse_params(r#"{"task":"Transcribe"}"#).is_err());
    }

    #[test]
    fn translating_hands_back_english_and_says_so_or_says_nothing() {
        assert!(parse_params(r#"{"task":"translate","language_out":"en"}"#).is_ok());
        assert!(parse_params(r#"{"task":"translate","language":"es"}"#).is_ok());
        assert!(
            parse_params(r#"{"task":"translate","language_out":"fr"}"#).is_err(),
            "whisper translates into English alone"
        );
    }

    #[test]
    fn transcribing_cannot_claim_to_hand_back_a_language_it_did_not_hear() {
        assert!(parse_params(r#"{"language":"es","language_out":"es"}"#).is_ok());
        assert!(parse_params(r#"{"language_out":"es"}"#).is_ok());
        assert!(parse_params(r#"{"language":"es","language_out":"en"}"#).is_err());
    }

    #[test]
    fn a_parameter_this_module_does_not_have_is_refused() {
        assert!(parse_params(r#"{"languge":"es"}"#).is_err());
        assert!(parse_params(r#"{"beams":3}"#).is_err());
    }

    #[test]
    fn a_segment_becomes_a_cue_shaped_row_on_the_streams_own_clock() {
        let written = row(
            Segment {
                start: 1.5,
                end: 2.25,
                text: "hola".to_string(),
            },
            30.0,
        );
        assert_eq!(
            written, r#"{"text":"hola","start_t":31.5,"end_t":32.25}"#,
            "the three columns a cue declares, and nothing else"
        );
    }

    #[test]
    fn a_cue_never_runs_past_the_window_that_produced_it() {
        let written = row(
            Segment {
                start: 29.0,
                end: 44.0,
                text: "hola".to_string(),
            },
            0.0,
        );
        assert!(written.contains(r#""end_t":30.0"#), "{written}");
    }

    #[test]
    fn a_refusal_still_opens_on_this_modules_own_name() {
        let refused = parse_params(r#"{"task":"summarize"}"#).expect_err("no such task");
        assert!(refused.starts_with("transcribe: "), "{refused}");
    }
}
