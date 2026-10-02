//! What both exports are as nodes: the same ports, the same window, and the
//! same way a window's segments or words become rows.
//!
//! `a` is the clock, 30 s of 16 kHz mono a tick, which is the model's fixed
//! input. `speech` is an upstream detector's rows, optional, paired by time:
//! every row stamped inside the window reaches it with the window. A window
//! with none is still decoded: a detector misses speech a transcriber hears,
//! so the model's own pass is what decides there was nothing to say. `words`
//! is a cue a segment or a word, stamped at its start. The window is the
//! delay, which the host reads off the shape, so the output declares none.
//!
//! The node is pure: nothing carries from one window to the next, so windows
//! may be decoded on several workers at once.

use ffrwd_node::{Input, Out, Output, Rational, Shape, Tick};
use serde::Deserialize;

use crate::cue::{Cue, ROWS_SCHEMA};
use crate::mel::{SAMPLE_RATE, WINDOW_SAMPLES};
use crate::tensor::to_f32;
use crate::Speech;

/// The output the cues leave on.
pub const WORDS: &str = "words";

/// The language a node's rows are in: the one they were turned into when one
/// was asked for, else the one they were heard in.
pub const ROWS_LANGUAGE: &[&str] = &["language_out", "language"];

/// What a detector's row has to carry: the second its span began.
const SPEECH_SCHEMA: &str =
    r#"{"type":"object","properties":{"start_t":{"type":"number"}},"required":["start_t"]}"#;

#[derive(Deserialize)]
struct Onset {
    start_t: f64,
}

/// The ports of both exports.
pub fn shape() -> Shape {
    Shape::new()
        .input(
            Input::audio("a")
                .clock()
                .window(WINDOW_SAMPLES as u32, WINDOW_SAMPLES as u32)
                .sample_formats(&["f32"])
                .sample_rates(&[SAMPLE_RATE])
                .channel_counts(&[1]),
        )
        .input(
            Input::rows("speech")
                .optional()
                .interval()
                .schema_json(SPEECH_SCHEMA),
        )
        .output(Output::rows(WORDS).schema_json(ROWS_SCHEMA))
        .pure()
}

/// One tick: the samples, where they start, how long they run, and what a
/// detector said about them.
pub struct Window {
    /// The second the window starts at.
    pub start: f64,
    /// Seconds of audio in it: 30, or less on the last window of a stream.
    pub length: f64,
    pub samples: Vec<f32>,
    /// What a detector said about the window; nothing when none is bound.
    pub speech: Speech,
}

impl Window {
    /// The window tick hands on stream `a`, with the rows of `speech` when
    /// the call binds it. None when the tick hands no audio.
    pub fn of(
        tick: &Tick,
        a: u32,
        speech: Option<u32>,
        time_base: Rational,
    ) -> Result<Option<Window>, String> {
        let Some(frame) = tick.frame(a) else {
            return Ok(None);
        };
        let samples = to_f32(&tick.fetch(a, frame.index));
        let mut heard = Speech::new();
        if let Some(id) = speech {
            for onset in tick.rows::<Onset>(id)? {
                heard.heard(onset.start_t);
            }
        }
        Ok(Some(Window {
            start: time_base.seconds(frame.pts),
            length: samples.len() as f64 / f64::from(SAMPLE_RATE),
            samples,
            speech: heard,
        }))
    }

    /// A detector said this window holds speech, which stands in for the pass
    /// that would otherwise establish it.
    pub fn known(&self) -> bool {
        self.speech.any()
    }

    /// A segment or a word as its cue: its times, seconds into the window,
    /// moved onto the stream's own clock and held inside the audio the window
    /// carried, which on the last window of a stream stops short of the 30 s
    /// the model was padded to.
    pub fn cue(&self, text: String, from: f64, to: f64) -> Cue {
        Cue {
            text,
            start_t: self.start + from.min(self.length),
            end_t: self.start + to.min(self.length),
        }
    }
}

/// `cues` on `words`, in the order they start, each stamped at its start.
pub fn emit(out: &mut Out, mut cues: Vec<Cue>) -> Result<(), String> {
    cues.sort_by(|a, b| a.start_t.total_cmp(&b.start_t));
    for cue in cues {
        let pts = out.pts(WORDS, cue.start_t)?;
        out.row(WORDS, pts, &cue)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start: f64, samples: usize) -> Window {
        Window {
            start,
            length: samples as f64 / f64::from(SAMPLE_RATE),
            samples: vec![0.0; samples],
            speech: Speech::new(),
        }
    }

    #[test]
    fn a_cue_is_on_the_streams_clock() {
        let cue = window(30.0, WINDOW_SAMPLES).cue("hola".into(), 1.5, 2.25);
        assert_eq!((cue.start_t, cue.end_t), (31.5, 32.25));
    }

    #[test]
    fn a_cue_never_runs_past_the_window_that_produced_it() {
        let cue = window(0.0, WINDOW_SAMPLES).cue("hola".into(), 29.0, 44.0);
        assert_eq!(cue.end_t, 30.0);
    }

    #[test]
    fn nor_past_the_audio_the_last_window_carried() {
        // 15 s left of a 75 s stream, padded to 30 for the model.
        let cue = window(60.0, 240_000).cue("adios".into(), 14.0, 22.0);
        assert_eq!((cue.start_t, cue.end_t), (74.0, 75.0));
        let past = window(60.0, 240_000).cue("".into(), 18.0, 19.0);
        assert_eq!((past.start_t, past.end_t), (75.0, 75.0));
    }

    #[test]
    fn a_window_is_known_to_hold_speech_only_once_a_detector_says_so() {
        let mut window = window(0.0, 10);
        assert!(!window.known());
        window.speech.heard(3.0);
        assert!(window.known());
    }

    #[test]
    fn the_shape_is_thirty_seconds_in_and_cues_out() {
        let shape = shape();
        let a = shape.find_input("a").expect("a");
        assert_eq!((a.window, a.stride), (480_000, 480_000));
        assert_eq!(a.accepts.sample_rates, [16_000]);
        let speech = shape.find_input("speech").expect("speech");
        assert!(!speech.required);
        assert!(matches!(
            speech.pairing,
            ffrwd_node::Pairing::Interval(ffrwd_node::Interval { latency: None, .. })
        ));
        let words = shape.find_output(WORDS).expect("words");
        assert_eq!(words.latency, 0.0, "the window is the delay");
        assert!(shape.pure);
    }
}
