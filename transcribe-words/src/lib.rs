//! What was said and exactly when: every WORD leaves as a cue of its own.
//!
//! This is `transcribe`'s decode read a second way. The graph is the same
//! fused whisper - encoder, decode loop and beam search all inside one
//! `compute` - re-exported so that it also hands back the decoder's
//! cross-attention, and the times come from warping that attention against the
//! encoder's 20 ms frames. The windowing, the features, the prompts, the
//! tokenizer and the shape of a cue are all `core`'s, shared with `transcribe`
//! rather than written twice; what is only here is the alignment, in `align`,
//! and the one extra tensor that feeds it.
//!
//! # What differs from `transcribe`, and why
//!
//! **One beam.** `transcribe` searches five. onnxruntime does not re-order the
//! collected cross-attention when the beam search swaps its hypotheses
//! around, so with more than one beam whole stretches of rows belong to a
//! sequence that was discarded, and the word error triples. The aligned pass
//! is therefore greedy, and its text can differ slightly from what
//! `transcribe` reads out of the same audio. That is the trade this export
//! makes: a word you can cut on, out of a marginally worse transcript.
//!
//! **CUDA only.** Collecting the cross-attention is a GPU-only decoder op, so
//! the graph runs on the CUDA execution provider and nowhere else. It loads
//! anywhere and then fails inside the node at the first inference, so the
//! refusal is raised there - see [`Opened::pass`].
//!
//! # The window
//!
//! 30 s, stride the same, disjoint, exactly as `transcribe` tiles them: the
//! ports are `core`'s, shared with it. A word's times are the alignment's
//! own, offset by the second its window starts at and held inside the audio
//! the window carried.
//!
//! # The leading silence
//!
//! A window's first word absorbs whatever silence runs ahead of it: the warp
//! has to start the path at the first frame, so a word that is really spoken
//! ten seconds in is reported as starting at the top of the window. Where an
//! upstream voice detector has told this module where the speech is, that
//! word's start is pulled forward to the onset inside it. A window with no
//! spans keeps the raw start, because nothing else in the graph knows any
//! better.

// `generate_all`: the world's interfaces are wasi:nn's, a package of its own,
// and without it bindgen expects them to have been generated somewhere else.
wit_bindgen::generate!({
    path: "wit-world",
    world: "ffrwd:whisper-words/transcribe-words",
    generate_all,
});

mod align;

use align::{CrossQk, FirstRow, Word, ALIGNMENT_HEADS};
use ffrwd_node::{Bound, Init, Node, Out, Rational, Result, Shape, Tick};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};
use whisper_core::cue::Cue;
use whisper_core::mel::{self, Plan, N_FRAMES, N_MELS};
use whisper_core::node::{self, Window};
use whisper_core::params::{parse_params as parse_shared, Params};
use whisper_core::tensor::{f32_bytes, i32_bytes, to_f32, to_i32};
use whisper_core::tokens::{self, Vocab};
use whisper_core::Speech;

/// The name the host binds the graph to. `-nn transcribe_words=<path>`.
const MODEL: &str = "transcribe_words";

/// What this export is called, which is what its refusals open on.
const ME: &str = "transcribe_words";

/// The graph's own names for the tensors it takes and returns. The first eight
/// are `transcribe`'s; the last two are what the cross-QK export adds.
const FEATURES: &str = "input_features";
const MAX_LENGTH: &str = "max_length";
const MIN_LENGTH: &str = "min_length";
const NUM_BEAMS: &str = "num_beams";
const NUM_RETURN_SEQUENCES: &str = "num_return_sequences";
const LENGTH_PENALTY: &str = "length_penalty";
const REPETITION_PENALTY: &str = "repetition_penalty";
const DECODER_INPUT_IDS: &str = "decoder_input_ids";
const SEQUENCES: &str = "sequences";
/// The (layer, head) pairs whose cross-attention the graph is to keep, int32
/// `[num_layer_head, 2]`. Naming the six alignment heads here is what keeps
/// the returned tensor to megabytes rather than hundreds of them.
const CROSS_QK_LAYER_HEAD: &str = "cross_qk_layer_head";
/// The collected attention, float32
/// `[batch, num_return_sequences, num_layer_head, decoded_length, frames]`.
/// It is fp32 even out of an int8 export.
const CROSS_QK: &str = "cross_qk";

/// Beams the transcribing pass searches over: one, which is the whole of why
/// this export exists separately. See the note at the top of the file.
const BEAMS: i32 = 1;

/// The longest answer a transcribing pass may produce, which is half the
/// decoder's positions - whisper's own limit on one window.
const TRANSCRIBE_LENGTH: i32 = 224;

/// The longest answer the detecting pass may produce. The language is the
/// token after the prefix, so this is the prefix, the answer, and room for the
/// model to end.
const DETECT_LENGTH: i32 = 4;

/// Why a graph that will not run is not this module's to fix, and what to do
/// instead.
///
/// This is appended wherever the graph can give out from being on the wrong
/// provider: loading it, and the first pass of an instance. Which of the two
/// it is depends on the host and on the onnxruntime build - a session that
/// refuses the graph outright fails at the load, and one that accepts it fails
/// inside the node the first time it is asked to infer.
///
/// A host that binds `-nn` eagerly, before this module is instantiated, will
/// report its own load failure and never reach either of these. The SQL
/// header says the same thing for that case, and `not_on` in the manifest
/// keeps the weights off the accelerators that can be named - CPU cannot be,
/// since it is the fallback every provider list ends in.
const NEEDS_CUDA: &str = "word timestamps need the CUDA execution provider: the \
     cross-attention they are warped from is collected by a decoder op that \
     exists only on the GPU, so this graph runs nowhere else. Use \
     `transcribe`, which is the export that runs anywhere.";

/// This module's parameters: the three `transcribe` takes, and `strip`.
#[derive(Debug)]
struct WordParams {
    /// The language, the task and what the cues come out in, as shared.
    decode: Params,
    /// Whether a word's cue carries the bare word. The model attaches
    /// punctuation to words - a comma after `damn`, a quote before it - and
    /// a mask matching typed words against cues wants `damn`, not `damn,`.
    /// Off, the text is what the model wrote, as it always was.
    strip: bool,
}

/// The name of this module's own parameter.
const STRIP: &str = "strip";

/// The schema this module publishes: the shared one with `strip` added.
const WORDS_PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"language":{"type":["string","null"]},"task":{"type":"string","enum":["transcribe","translate"],"default":"transcribe"},"language_out":{"type":["string","null"]},"strip":{"type":"boolean","default":false}},"additionalProperties":false}"#;

/// Parses and validates params, shared by `init` and `set_params`. The list is
/// `transcribe`'s plus `strip`, so a query can swap one export for the other
/// and only lose the stripping.
fn parse_params(params: &str) -> Result<WordParams, String> {
    let trimmed = params.trim();
    if trimmed.is_empty() {
        return Ok(WordParams {
            decode: parse_shared(ME, "")?,
            strip: false,
        });
    }
    let mut written: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|e| format!("{ME}: bad params: {e}"))?;
    let strip = match written.as_object_mut().and_then(|o| o.remove(STRIP)) {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(strip)) => strip,
        Some(other) => {
            return Err(format!("{ME}: {STRIP} is true or false, not {other}"));
        }
    };
    Ok(WordParams {
        decode: parse_shared(ME, &written.to_string())?,
        strip,
    })
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

/// The alignment heads flattened the way the graph's input wants them: one row
/// of (layer, head) per head, in the order the returned tensor's head axis
/// will be in.
fn layer_head_pairs() -> Vec<i32> {
    ALIGNMENT_HEADS
        .iter()
        .flat_map(|(layer, head)| [*layer, *head])
        .collect()
}

/// Those pairs as the graph's own input tensor.
fn cross_qk_layer_head() -> (String, Tensor) {
    (
        CROSS_QK_LAYER_HEAD.to_string(),
        Tensor::new(
            &[ALIGNMENT_HEADS.len() as u32, 2],
            TensorType::I32,
            &i32_bytes(&layer_head_pairs()),
        ),
    )
}

/// What one pass through the graph produced.
struct Decoded {
    /// What the model added to the prompt, end-of-text INCLUDED - the token
    /// the last collected row is the evidence for.
    generated: Vec<u32>,
    /// The collected attention, flattened head-major, with its dimensions.
    cross_qk: Option<(Vec<f32>, usize, usize, usize)>,
}

struct TranscribeWords {
    a: u32,
    speech: Option<u32>,
    time_base: Rational,
    params: WordParams,
    plan: Plan,
    vocab: Vocab,
    /// Passes this instance has run. A graph on the wrong provider gives out
    /// on the first one, so that is the failure worth explaining.
    passes: usize,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per window.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

impl TranscribeWords {
    /// One pass through the graph: the answer's generated ids, and the
    /// cross-attention when it was asked for.
    ///
    /// `cross_qk_layer_head` is fed on every pass, wanted or not: it is a
    /// graph INPUT of this export and the session will not run without it.
    /// When the attention is not wanted the answer is simply dropped, which is
    /// cheaper than a second session on a second copy of the weights.
    fn pass(
        &mut self,
        mel: &[f32],
        prefix: &[i32],
        max_length: i32,
        beams: i32,
        want_qk: bool,
    ) -> Result<Decoded, String> {
        let first = self.passes == 0;
        self.passes += 1;
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
                cross_qk_layer_head(),
            ])
            // The graph runs only on CUDA, and it says so by failing inside
            // the node the first time it is asked to infer. Any other first
            // failure reads the same way, which is the right trade: this is by
            // far the likeliest reason a machine that installed the model
            // cannot run it.
            .map_err(|e| match first {
                true => format!("{}: {NEEDS_CUDA}", failed("compute", &e)),
                false => failed("compute", &e),
            })?;

        let named = |wanted: &str| {
            outputs
                .iter()
                .find(|(name, _)| name == wanted)
                .map(|(_, tensor)| tensor)
        };

        let sequences = named(SEQUENCES)
            .ok_or_else(|| format!("{ME}: the graph returned no tensor named {SEQUENCES}"))?;

        // The whole answer, prompt included, cut at the end-of-text the beam
        // search pads the rest of the row out with - but KEEPING that token.
        // It is the last thing the decode decided, so the last collected row
        // is its evidence, and dropping it here would leave the rows and the
        // tokens off by one.
        let ids = to_i32(&sequences.data());
        let mut tokens: Vec<u32> = Vec::with_capacity(ids.len());
        for id in ids {
            if id < 0 {
                break;
            }
            tokens.push(id as u32);
            if id == tokens::EOT as i32 && tokens.len() > prefix.len() {
                break;
            }
        }
        let generated = tokens.split_off(prefix.len().min(tokens.len()));

        let cross_qk = match want_qk {
            false => None,
            true => {
                let tensor = named(CROSS_QK).ok_or_else(|| {
                    format!(
                        "{ME}: the graph returned no tensor named {CROSS_QK}; the model pinned \
                         for this export is whisper re-exported with --collect_cross_qk, and \
                         this one was not"
                    )
                })?;
                // [batch, num_return_sequences, heads, rows, frames]. Only the
                // one returned sequence is asked for, so the first two are 1.
                let dims = tensor.dimensions();
                let [.., heads, rows, frames] = dims[..] else {
                    return Err(format!(
                        "{ME}: {CROSS_QK} came back with dimensions {dims:?}, and the alignment \
                         needs at least a head, a row and a frame axis"
                    ));
                };
                Some((
                    to_f32(&tensor.data()),
                    heads as usize,
                    rows as usize,
                    frames as usize,
                ))
            }
        };

        Ok(Decoded {
            generated,
            cross_qk,
        })
    }

    /// The words one window decodes to, empty when nothing was said in it.
    ///
    /// `known_speech` is an upstream detector's word that this window holds
    /// some; it stands in for the detecting pass only when the caller already
    /// named the language, since otherwise the pass is what supplies it.
    fn window(&mut self, samples: &[f32], known_speech: bool) -> Result<Vec<Word>, String> {
        let mel = self.plan.spectrogram(samples);
        let content_frames = mel::content_frames(samples.len());

        let detect = tokens::prefix(None, self.params.decode.task, false);
        let language = match self.params.decode.language {
            Some(named) if known_speech => Some(named),
            Some(named) => {
                let heard = self.pass(&mel, &detect, DETECT_LENGTH, 1, false)?;
                tokens::detected(&heard.generated).map(|_| named)
            }
            None => {
                let heard = self.pass(&mel, &detect, DETECT_LENGTH, 1, false)?;
                tokens::detected(&heard.generated).and_then(tokens::language_token)
            }
        };
        let Some(language) = language else {
            return Ok(Vec::new());
        };

        let said = self.pass(
            &mel,
            &tokens::prefix(Some(language), self.params.decode.task, true),
            TRANSCRIBE_LENGTH,
            BEAMS,
            true,
        )?;
        let Some((data, heads, rows, frames)) = said.cross_qk else {
            return Ok(Vec::new());
        };
        Ok(align::align(
            &CrossQk {
                data: &data,
                heads,
                rows,
                frames,
            },
            &said.generated,
            content_frames,
            &self.vocab,
            FirstRow::Duplicated,
        ))
    }
}

impl Node for TranscribeWords {
    const NAME: &'static str = "transcribe_words";
    const VERSION: &'static str = "0.2.0";
    const PARAMS_SCHEMA: &'static str = WORDS_PARAMS_SCHEMA;
    const ROWS_LANGUAGE: &'static [&'static str] = node::ROWS_LANGUAGE;
    type Params = serde_json::Value;

    fn shape(_: &serde_json::Value, _: &Bound) -> Result<Shape> {
        Ok(node::shape())
    }

    fn init(params: serde_json::Value, init: &Init) -> Result<TranscribeWords> {
        let parsed = parse_params(&params.to_string())?;
        let vocab = Vocab::whisper().map_err(|e| format!("{ME}: {e}"))?;
        let a = init.stream("a")?;

        // The graph is loaded once per instance, and the session built once:
        // the first pass is what a provider picks its kernels on, and every
        // pass after it reuses them.
        //
        // Either of these two steps is where a session that refuses this
        // graph outright gives out, so both say why.
        let graph = load_by_name(MODEL).map_err(|e| {
            format!(
                "{}: {NEEDS_CUDA}",
                failed(&format!("load-by-name({MODEL:?})"), &e)
            )
        })?;
        let context = graph
            .init_execution_context()
            .map_err(|e| format!("{}: {NEEDS_CUDA}", failed("init-execution-context", &e)))?;

        Ok(TranscribeWords {
            a: a.id,
            speech: init.optional("speech").map(|speech| speech.id),
            time_base: a.info.time_base,
            params: parsed,
            plan: Plan::new(),
            vocab,
            passes: 0,
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
        let words = clamped(said, window.start, &window.speech);
        Ok(node::emit(out, cues(&window, words, self.params.strip))?)
    }
}

/// The window's words with the first one's start pulled forward onto the
/// nearest speech onset it covers.
///
/// The warp has to open the token path on the first frame of the window, so
/// the first word swallows every quiet frame ahead of it - on a window that
/// opens on ten seconds of room tone, a word really spoken at 10.3 s is
/// reported from 0. Nothing in the attention can say otherwise: there is no
/// row before the first one to put the silence on.
///
/// An upstream voice detector does know. Its onsets are in `speech`, and the
/// latest one that falls inside the word is where the sound starts, so that is
/// where the word is moved to. The move is forward only and never past the
/// word's own end, so a word is never stretched and never leaves its window.
///
/// A first word with no onset inside it - no detector bound, or a span that
/// began in the window before - keeps the raw start. A start that is too early
/// is worse than no answer for cutting audio, but inventing an onset would be
/// worse still.
fn clamped(mut words: Vec<Word>, start: f64, speech: &Speech) -> Vec<Word> {
    if let Some(first) = words.first_mut() {
        let from = start + first.start;
        let to = start + first.end;
        if let Some(onset) = speech.onset_before(to, from) {
            first.start = onset - start;
        }
    }
    words
}

/// The window's words as cues on the stream's own clock.
///
/// With `strip`, the text is the bare word: the punctuation the model attached
/// to it comes off both ends, and a word that was only punctuation - a dash
/// standing alone - leaves no cue at all. Without it, the text is what the
/// model wrote, trimmed of the space it opens on.
fn cues(window: &Window, words: Vec<Word>, strip: bool) -> Vec<Cue> {
    words
        .into_iter()
        .filter_map(|word| {
            let text = if strip {
                word.text.trim_matches(|c: char| !c.is_alphanumeric())
            } else {
                word.text.trim()
            };
            (!text.is_empty()).then(|| window.cue(text.to_string(), word.start, word.end))
        })
        .collect()
}

ffrwd_node::export!(TranscribeWords);

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_core::mel::WINDOW_SECONDS;
    use whisper_core::tokens::Task;

    // ------------------------------------------------------------ the params

    #[test]
    fn no_params_at_all_are_the_defaults() {
        for written in ["", "{}", "  "] {
            let parsed = parse_params(written).expect("the defaults");
            assert_eq!(
                parsed.decode.language, None,
                "the model detects one per window"
            );
            assert_eq!(parsed.decode.task, Task::Transcribe);
            assert!(!parsed.strip, "the text is what the model wrote");
        }
    }

    #[test]
    fn the_parameter_list_is_the_one_transcribe_takes() {
        let parsed = parse_params(r#"{"language":"es","task":"translate","language_out":"en"}"#)
            .expect("all three");
        assert_eq!(parsed.decode.language, tokens::language_token("es"));
        assert_eq!(parsed.decode.task, Task::Translate);
    }

    #[test]
    fn strip_is_this_modules_own_and_is_true_or_false() {
        assert!(parse_params(r#"{"strip":true}"#).expect("on").strip);
        assert!(!parse_params(r#"{"strip":false}"#).expect("off").strip);
        assert!(!parse_params(r#"{"strip":null}"#).expect("unset").strip);
        assert!(parse_params(r#"{"strip":"yes"}"#).is_err());
        assert!(
            parse_params(r#"{"strop":true}"#).is_err(),
            "unknown names still refuse"
        );
        let parsed = parse_params(r#"{"language":"es","strip":true}"#).expect("both");
        assert_eq!(parsed.decode.language, tokens::language_token("es"));
        assert!(parsed.strip);
    }

    #[test]
    fn the_schema_is_the_shared_one_with_strip_added() {
        let schema: serde_json::Value =
            serde_json::from_str(WORDS_PARAMS_SCHEMA).expect("valid json");
        let mut shared: serde_json::Value =
            serde_json::from_str(whisper_core::params::PARAMS_SCHEMA).expect("valid json");
        shared["properties"][STRIP] = serde_json::json!({"type": "boolean", "default": false});
        assert_eq!(schema, shared);
        let properties = schema["properties"].as_object().expect("an object");
        let mut named: Vec<&String> = properties.keys().collect();
        named.sort();
        assert_eq!(named, ["language", "language_out", "strip", "task"]);
        assert_eq!(
            schema["properties"]["strip"]["type"],
            serde_json::json!("boolean")
        );
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
    }

    #[test]
    fn a_language_whisper_was_not_trained_on_is_refused() {
        assert!(parse_params(r#"{"language":"klingon"}"#).is_err());
        assert!(parse_params(r#"{"language":"ES"}"#).is_err());
    }

    #[test]
    fn anything_that_is_not_one_of_whispers_two_tasks_is_refused() {
        assert!(parse_params(r#"{"task":"summarize"}"#).is_err());
        assert!(parse_params(r#"{"task":"Transcribe"}"#).is_err());
    }

    #[test]
    fn translating_hands_back_english_and_says_so_or_says_nothing() {
        assert!(parse_params(r#"{"task":"translate","language_out":"en"}"#).is_ok());
        assert!(
            parse_params(r#"{"task":"translate","language_out":"fr"}"#).is_err(),
            "whisper translates into English alone"
        );
    }

    #[test]
    fn a_parameter_this_module_does_not_have_is_refused() {
        assert!(parse_params(r#"{"beams":3}"#).is_err());
        assert!(
            parse_params(r#"{"languge":"es"}"#).is_err(),
            "including a misspelt one"
        );
    }

    #[test]
    fn a_refusal_opens_on_this_exports_own_name() {
        let refused = parse_params(r#"{"task":"summarize"}"#).expect_err("no such task");
        assert!(refused.starts_with("transcribe_words: "), "{refused}");
    }

    #[test]
    fn the_refusal_for_the_wrong_provider_names_the_cause_and_the_way_out() {
        assert!(NEEDS_CUDA.contains("CUDA execution provider"));
        assert!(
            NEEDS_CUDA.contains("`transcribe`"),
            "and points at the export that runs anywhere"
        );
    }

    // ------------------------------------------------------------ the decode

    #[test]
    fn this_export_decodes_greedily() {
        assert_eq!(
            BEAMS, 1,
            "onnxruntime does not re-order the collected attention across beams"
        );
    }

    /// The six pairs `openai-whisper`'s own base85 table decodes to for
    /// medium. Asking for these and nothing else is what keeps the returned
    /// tensor to megabytes; asking for the wrong ones would align against
    /// heads that carry no timing and would not fail, just be wrong.
    #[test]
    fn the_six_alignment_heads_are_asked_for_as_layer_head_pairs() {
        assert_eq!(
            layer_head_pairs(),
            vec![13, 15, 15, 4, 15, 15, 16, 1, 20, 0, 23, 4]
        );
        assert_eq!(ALIGNMENT_HEADS.len(), 6);
        for (layer, head) in ALIGNMENT_HEADS {
            assert!((0..24).contains(&layer), "medium has 24 decoder layers");
            assert!((0..16).contains(&head), "and 16 heads in each");
        }
    }

    // ------------------------------------------------------------- the clamp

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn a_first_word_that_swallowed_the_silence_is_pulled_onto_the_onset() {
        let mut speech = Speech::new();
        speech.heard(10.26);
        let words = clamped(
            vec![word(" You", 0.0, 10.42), word(" said", 10.42, 10.6)],
            0.0,
            &speech,
        );
        assert!(
            (words[0].start - 10.26).abs() < 1e-12,
            "{:?}",
            words[0].start
        );
        assert_eq!(words[0].end, 10.42, "and its end is untouched");
        assert_eq!(words[1].start, 10.42, "and so is every word after it");
    }

    #[test]
    fn a_window_with_no_spans_keeps_the_raw_start() {
        let words = clamped(
            vec![word(" You", 0.0, 10.42), word(" said", 10.42, 10.6)],
            0.0,
            &Speech::new(),
        );
        assert_eq!(words[0].start, 0.0, "nothing knew better, so nothing moved");
    }

    #[test]
    fn an_onset_outside_the_first_word_moves_nothing() {
        let mut speech = Speech::new();
        // The detector heard speech, but not inside the word's own span.
        speech.heard(20.0);
        let words = clamped(vec![word(" You", 0.0, 10.42)], 0.0, &speech);
        assert_eq!(words[0].start, 0.0);
    }

    #[test]
    fn the_clamp_reads_the_onset_on_the_streams_clock_not_the_windows() {
        let mut speech = Speech::new();
        // A window three in: the spans are in stream seconds, the words in
        // window seconds, and the clamp has to cross between them.
        speech.heard(95.5);
        let words = clamped(vec![word(" because", 0.0, 6.0)], 90.0, &speech);
        assert!((words[0].start - 5.5).abs() < 1e-12, "{:?}", words[0].start);
    }

    #[test]
    fn a_window_with_no_words_at_all_clamps_nothing() {
        assert!(clamped(Vec::new(), 0.0, &Speech::new()).is_empty());
    }

    // --------------------------------------------------------------- the row

    /// One word through `cues` on a window of `length` seconds at `start`.
    fn cue_of(word: Word, start: f64, length: f64, strip: bool) -> Option<Cue> {
        let window = Window {
            start,
            length,
            samples: Vec::new(),
            speech: Speech::new(),
        };
        cues(&window, vec![word], strip).into_iter().next()
    }

    /// One word on a whole window at `start`, as the JSON it leaves as.
    fn row(word: Word, start: f64, strip: bool) -> Option<String> {
        cue_of(word, start, WINDOW_SECONDS, strip)
            .map(|cue| serde_json::to_string(&cue).expect("json"))
    }

    #[test]
    fn the_last_word_of_a_stream_ends_with_its_audio() {
        let cue = cue_of(word(" goodbye", 10.1, 30.0), 60.0, 10.6, false).expect("a cue");
        assert_eq!((cue.start_t, cue.end_t), (70.1, 70.6));
    }

    #[test]
    fn a_word_becomes_a_cue_shaped_row_on_the_streams_own_clock() {
        assert_eq!(
            row(word(" hola", 1.5, 2.25), 30.0, false).expect("a row"),
            r#"{"text":"hola","start_t":31.5,"end_t":32.25}"#,
            "the three columns a cue declares, trimmed, and nothing else"
        );
    }

    #[test]
    fn a_word_never_runs_past_the_window_that_produced_it() {
        let written = row(word(" hola", 29.5, 44.0), 0.0, false).expect("a row");
        assert!(written.contains(r#""end_t":30.0"#), "{written}");
    }

    #[test]
    fn stripping_leaves_the_bare_word_and_keeps_what_is_inside_it() {
        let text = |written: &str| {
            let cue: serde_json::Value =
                serde_json::from_str(&row(word(written, 0.0, 1.0), 0.0, true).expect("a row"))
                    .expect("json");
            cue["text"].as_str().expect("text").to_string()
        };
        assert_eq!(text(" damn,"), "damn");
        assert_eq!(text(" \"damn.\""), "damn");
        assert_eq!(text(" -hello"), "hello");
        assert_eq!(
            text(" o'clock"),
            "o'clock",
            "punctuation inside a word stays"
        );
        assert_eq!(
            text(" \u{201c}\u{ff2f}\u{3002}"),
            "\u{ff2f}",
            "any script, any width"
        );
        assert_eq!(
            row(word(" damn,", 0.0, 1.0), 0.0, false).expect("a row"),
            r#"{"text":"damn,","start_t":0.0,"end_t":1.0}"#,
            "off, the text is what the model wrote"
        );
    }

    #[test]
    fn a_word_that_was_only_punctuation_leaves_no_row_when_stripped() {
        assert!(row(word(" -", 0.0, 0.2), 0.0, true).is_none());
        assert!(row(word(" -", 0.0, 0.2), 0.0, false).is_some());
    }

    // ---------------------------------------------- the alignment, end to end

    /// One real window's collected attention, its token ids and the word
    /// spans the Python spike produces for it - the feasibility work this
    /// module is a port of. Captured from the very model the manifest pins,
    /// on CUDA, with one beam.
    const WINDOW_QK: &[u8] = include_bytes!("../fixtures/window.qk");
    const WINDOW_JSON: &str = include_str!("../fixtures/window.json");

    struct Fixture {
        qk: Vec<f32>,
        heads: usize,
        rows: usize,
        frames: usize,
        content_frames: usize,
        generated: Vec<u32>,
        words: Vec<(String, f64, f64)>,
        words_wrong_row_offset: Vec<(String, f64, f64)>,
    }

    fn fixture() -> Fixture {
        let meta: serde_json::Value = serde_json::from_str(WINDOW_JSON).expect("the fixture");
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .expect("a shape")
            .iter()
            .map(|n| n.as_u64().expect("a size") as usize)
            .collect();
        let spans = |key: &str| -> Vec<(String, f64, f64)> {
            meta[key]
                .as_array()
                .expect("a word table")
                .iter()
                .map(|w| {
                    (
                        w["text"].as_str().expect("text").to_string(),
                        w["start"].as_f64().expect("start"),
                        w["end"].as_f64().expect("end"),
                    )
                })
                .collect()
        };
        Fixture {
            qk: to_f32(WINDOW_QK),
            heads: shape[0],
            rows: shape[1],
            frames: shape[2],
            content_frames: meta["content_frames"].as_u64().expect("frames") as usize,
            generated: meta["generated"]
                .as_array()
                .expect("the ids")
                .iter()
                .map(|n| n.as_u64().expect("an id") as u32)
                .collect(),
            words: spans("words"),
            words_wrong_row_offset: spans("words_wrong_row_offset"),
        }
    }

    fn align_fixture(f: &Fixture, first_row: FirstRow) -> Vec<Word> {
        align::align(
            &CrossQk {
                data: &f.qk,
                heads: f.heads,
                rows: f.rows,
                frames: f.frames,
            },
            &f.generated,
            f.content_frames,
            &Vocab::whisper().expect("the vocabulary parses"),
            first_row,
        )
    }

    /// One 20 ms encoder frame, which is the finest this alignment can be.
    const FRAME: f64 = 0.02;

    #[test]
    fn the_alignment_reproduces_the_spikes_word_spans_for_a_real_window() {
        let f = fixture();
        let mine = align_fixture(&f, FirstRow::Duplicated);
        assert_eq!(
            mine.len(),
            f.words.len(),
            "words: {:?}",
            mine.iter().map(|w| &w.text).collect::<Vec<_>>()
        );
        for (got, (text, start, end)) in mine.iter().zip(&f.words) {
            assert_eq!(&got.text, text);
            assert!(
                (got.start - start).abs() <= FRAME + 1e-9,
                "{text:?} starts at {} and the spike says {start}",
                got.start
            );
            assert!(
                (got.end - end).abs() <= FRAME + 1e-9,
                "{text:?} ends at {} and the spike says {end}",
                got.end
            );
        }
    }

    #[test]
    fn the_spans_are_in_fact_exactly_the_spikes_and_not_merely_near_them() {
        // The tolerance above is what the module PROMISES; this is what it
        // actually does, and a change that starts costing a frame should have
        // to say so here.
        let f = fixture();
        for (got, (text, start, end)) in
            align_fixture(&f, FirstRow::Duplicated).iter().zip(&f.words)
        {
            assert!(
                (got.start - start).abs() < 1e-9 && (got.end - end).abs() < 1e-9,
                "{text:?}: {}..{} against the spike's {start}..{end}",
                got.start,
                got.end
            );
        }
    }

    #[test]
    fn the_row_a_token_reads_is_the_one_before_it_and_not_its_own() {
        // The whole of the bookkeeping, in one assertion: reading row k as the
        // evidence for generated[k] instead of generated[k + 1] shifts the
        // table by a token, and on this window it moves nearly every word.
        let f = fixture();
        let right = align_fixture(&f, FirstRow::Duplicated);
        let wrong = align_fixture(&f, FirstRow::At(0));
        assert_eq!(wrong.len(), f.words_wrong_row_offset.len());

        let moved = right
            .iter()
            .zip(&wrong)
            .filter(|(a, b)| (a.start - b.start).abs() > FRAME)
            .count();
        assert!(
            moved * 4 > right.len() * 3,
            "only {moved} of {} words moved, so this fixture cannot tell the \
             two offsets apart",
            right.len()
        );
        // And the wrong one is wrong in the direction the docs claim: a token
        // late, so a word starts where the NEXT one should have.
        assert!(
            wrong[0].start > right[0].start - 1e-9,
            "the shift runs forwards"
        );
    }

    #[test]
    fn the_graphs_rows_are_one_fewer_than_the_tokens_they_are_evidence_for() {
        // The invariant the offset rests on: the init subgraph decides the
        // first token and emits no row for it, so a window of n generated
        // tokens comes back with n - 1 rows.
        let f = fixture();
        assert_eq!(f.rows, f.generated.len() - 1);
        assert_eq!(f.heads, 6, "the six alignment heads and no others");
        assert_eq!(f.frames, 1500, "a 30 s window's encoder frames");
    }

    #[test]
    fn a_timestamp_takes_its_own_row_and_is_gone_from_the_words() {
        let f = fixture();
        assert!(
            f.generated.iter().any(|id| *id >= tokens::TIMESTAMP),
            "the fixture window ends on a timestamp, which is the case worth \
             testing"
        );
        for word in align_fixture(&f, FirstRow::Duplicated) {
            assert!(!word.text.contains("<|"), "{:?} is a special", word.text);
        }
    }

    #[test]
    fn punctuation_arrives_joined_to_its_word() {
        let f = fixture();
        let texts: Vec<String> = align_fixture(&f, FirstRow::Duplicated)
            .into_iter()
            .map(|w| w.text)
            .collect();
        assert!(texts.contains(&" book.".to_string()), "{texts:?}");
        assert!(
            !texts.iter().any(|t| t.trim() == "."),
            "and never as a word of its own: {texts:?}"
        );
    }

    #[test]
    fn every_word_runs_forwards_and_stays_inside_its_window() {
        let f = fixture();
        let words = align_fixture(&f, FirstRow::Duplicated);
        assert!(!words.is_empty());
        for word in &words {
            assert!(word.end >= word.start, "{word:?}");
            assert!(word.start >= 0.0 && word.end <= WINDOW_SECONDS, "{word:?}");
        }
        for pair in words.windows(2) {
            assert!(
                pair[1].start >= pair[0].start - 1e-9,
                "{:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn a_window_that_decoded_nothing_aligns_to_nothing() {
        let vocab = Vocab::whisper().expect("the vocabulary parses");
        let empty = CrossQk {
            data: &[],
            heads: 0,
            rows: 0,
            frames: 0,
        };
        assert!(align::align(&empty, &[], 3000, &vocab, FirstRow::Duplicated).is_empty());
        // A tensor whose payload does not match its dimensions is refused
        // rather than read past the end of.
        let ragged = CrossQk {
            data: &[0.0; 10],
            heads: 6,
            rows: 3,
            frames: 1500,
        };
        assert!(align::align(&ragged, &[1, 2], 3000, &vocab, FirstRow::Duplicated).is_empty());
    }
}
