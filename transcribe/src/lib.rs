//! What was said: every stretch of speech leaves as a cue with the words in
//! it.
//!
//! The graph is whisper, exported so that the encoder, the decoder loop and
//! the beam search all sit INSIDE it - one `compute` per pass, with no decoder
//! state crossing the sandbox. The module never opens a file: the host binds
//! the graph to a name with `-nn transcribe=<path>` and this module asks for
//! that name and nothing else.
//!
//! # The window
//!
//! 30 s, stride the same, which is the model's fixed 3000-frame input; the
//! ports are `core`'s, shared with `transcribe_words`. A cue's times are the
//! model's own offset by the second the window starts at, and held inside the
//! audio the window carried.
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
//! They arrive with the window they fall in, all of them. A window with some
//! needs no first pass when the caller named the language, since the question
//! that pass answers has been answered. A window with none is decoded all the
//! same: a detector can miss speech the model hears, and the first pass is
//! what has the last word on whether anything was said.

// `generate_all`: the world's interfaces are wasi:nn's, a package of its own,
// and without it bindgen expects them to have been generated somewhere else.
wit_bindgen::generate!({
    path: "wit-world",
    world: "ffrwd:whisper/transcribe",
    generate_all,
});

use ffrwd_node::{Bound, Init, Node, Out, Rational, Result, Shape, Tick};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};
use whisper_core::mel::{Plan, N_FRAMES, N_MELS, WINDOW_SECONDS};
use whisper_core::node::{self, Window};
use whisper_core::params::{parse_params as parse_shared, Params, PARAMS_SCHEMA};
use whisper_core::tensor::{f32_bytes, i32_bytes, to_i32};
use whisper_core::tokens::{self, Segment, Vocab};

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

struct Transcribe {
    a: u32,
    speech: Option<u32>,
    time_base: Rational,
    params: Params,
    plan: Plan,
    vocab: Vocab,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per window.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

impl Transcribe {
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

impl Node for Transcribe {
    const NAME: &'static str = "transcribe";
    const VERSION: &'static str = "0.2.0";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    const ROWS_LANGUAGE: &'static [&'static str] = node::ROWS_LANGUAGE;
    type Params = serde_json::Value;

    fn shape(_: &serde_json::Value, _: &Bound) -> Result<Shape> {
        Ok(node::shape())
    }

    fn init(params: serde_json::Value, init: &Init) -> Result<Transcribe> {
        let parsed = parse_params(&params.to_string())?;
        let vocab = Vocab::whisper().map_err(|e| format!("transcribe: {e}"))?;
        let a = init.stream("a")?;

        // The graph is loaded once per instance, and the session built once:
        // the first pass is what a provider picks its kernels on, and every
        // pass after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        Ok(Transcribe {
            a: a.id,
            speech: init.optional("speech").map(|speech| speech.id),
            time_base: a.info.time_base,
            params: parsed,
            plan: Plan::new(),
            vocab,
            context,
            _graph: graph,
        })
    }

    fn set_params(&mut self, params: serde_json::Value) -> Result<()> {
        self.params = parse_params(&params.to_string())?;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(window) = Window::of(tick, self.a, self.speech, self.time_base)? else {
            return Ok(());
        };
        let said = self.window(&window.samples, window.known())?;
        Ok(node::emit(out, cues(&window, said))?)
    }
}

/// The window's segments as cues on the stream's own clock.
fn cues(window: &Window, said: Vec<Segment>) -> Vec<whisper_core::cue::Cue> {
    said.into_iter()
        .map(|segment| window.cue(segment.text, segment.start, segment.end))
        .collect()
}

ffrwd_node::export!(Transcribe);

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

    fn window(start: f64, seconds: f64) -> Window {
        let samples = vec![0.0; (seconds * 16_000.0) as usize];
        Window {
            start,
            length: seconds,
            samples,
            speech: whisper_core::Speech::new(),
        }
    }

    fn segment(start: f64, end: f64, text: &str) -> Segment {
        Segment {
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn a_segment_becomes_a_cue_on_the_streams_own_clock() {
        let cue = &cues(&window(30.0, 30.0), vec![segment(1.5, 2.25, "hola")])[0];
        assert_eq!(
            serde_json::to_string(cue).expect("json"),
            r#"{"text":"hola","start_t":31.5,"end_t":32.25}"#,
            "the three columns a cue declares, and nothing else"
        );
    }

    #[test]
    fn a_cue_never_runs_past_the_window_that_produced_it() {
        let cue = &cues(&window(0.0, 30.0), vec![segment(29.0, 44.0, "hola")])[0];
        assert_eq!(cue.end_t, 30.0);
    }

    #[test]
    fn the_last_cue_of_a_stream_ends_with_its_audio() {
        // 75 s of audio: the last window carries 15 s, padded to 30 for the
        // model, and the segment the model ran on to its padding stops at 75.
        let cue = &cues(&window(60.0, 15.0), vec![segment(8.4, 30.0, "goodbye")])[0];
        assert_eq!((cue.start_t, cue.end_t), (68.4, 75.0));
    }

    #[test]
    fn the_ports_are_the_audio_the_detectors_rows_and_the_cues() {
        let bound = |ports: &[&str]| {
            let ports: Vec<String> = ports.iter().map(|p| p.to_string()).collect();
            ffrwd_node::Runner::<Transcribe>::shape("", &ports).expect("a shape")
        };
        let alone = bound(&["a"]);
        assert_eq!(alone.clock_input(), Some("a"));
        assert_eq!(alone.outputs[0].name, "words");
        assert!(alone.pure);
        let beside = bound(&["a", "speech"]);
        assert_eq!(beside.inputs.len(), 2);
        assert!(
            ffrwd_node::Runner::<Transcribe>::shape(r#"{"words":true}"#, &[]).is_err(),
            "a param this module does not have"
        );
    }

    #[test]
    fn a_refusal_still_opens_on_this_modules_own_name() {
        let refused = parse_params(r#"{"task":"summarize"}"#).expect_err("no such task");
        assert!(refused.starts_with("transcribe: "), "{refused}");
    }
}
