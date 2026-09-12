//! The arithmetic the modules are built on: the spectrogram whisper reads, the
//! token ids it is steered with, the parameters a decode is steered by, the
//! rows it leaves as, and the spans an upstream detector's rows carry. None of
//! it touches the model, so all of it is tested on the host.
//!
//! Both exports - `transcribe` and `transcribe_words` - are the same decode
//! read two different ways, so everything either of them would otherwise have
//! had its own copy of lives here. What is not here is what only one of them
//! does: the alignment, which only `transcribe_words` runs, sits in that
//! crate beside the graph plumbing that feeds it.

pub mod cue;
pub mod mel;
pub mod params;
pub mod tensor;
pub mod tokens;

/// A timestamp in the stream's own unit, as seconds. `den` is always positive,
/// so a negative timestamp stays negative.
pub fn seconds(ticks: i64, num: i32, den: i32) -> f64 {
    ticks as f64 * f64::from(num) / f64::from(den)
}

/// The spans an upstream detector said hold speech, in seconds from the start
/// of the stream.
///
/// Rows arrive as NDJSON on the payload the detector emitted them with, which
/// for a span-closing detector is a little AFTER the span itself. They are
/// kept rather than read and dropped, so a span still counts for the window it
/// covers once it arrives.
#[derive(Default)]
pub struct Speech {
    spans: Vec<(f64, f64)>,
    /// Whether any row has ever arrived, which is what tells a wired-up
    /// detector from none at all.
    heard: bool,
}

impl Speech {
    pub fn new() -> Speech {
        Speech::default()
    }

    /// One NDJSON row. Anything without both times is passed over: a row from
    /// a module this one was not written for is not an error.
    pub fn push(&mut self, row: &str) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(row) else {
            return;
        };
        let at = |name: &str| value.get(name).and_then(serde_json::Value::as_f64);
        if let (Some(start), Some(end)) = (at("start_t"), at("end_t")) {
            self.spans.push((start, end));
            self.heard = true;
        }
    }

    /// Whether any row has arrived at all.
    pub fn any(&self) -> bool {
        self.heard
    }

    /// Whether a known span overlaps `[start, end)`.
    pub fn covers(&self, start: f64, end: f64) -> bool {
        self.spans
            .iter()
            .any(|(from, to)| *from < end && *to > start)
    }

    /// Forgets the spans that end before `before`, which are the ones no
    /// window still to come can overlap.
    pub fn forget(&mut self, before: f64) {
        self.spans.retain(|(_, to)| *to >= before);
    }

    /// The latest onset at or before `at` that is no earlier than `floor`, or
    /// None when no known span opens in that reach.
    ///
    /// This is what a word's start is pulled forward to when the model put it
    /// in the silence ahead of the speech: a detector's onset is the place the
    /// sound actually begins, and the alignment has no other way to know it.
    /// The window's own start is the floor, so a word is never dragged back
    /// into a window that has already gone by.
    pub fn onset_before(&self, at: f64, floor: f64) -> Option<f64> {
        self.spans
            .iter()
            .map(|(from, _)| *from)
            .filter(|from| *from <= at && *from >= floor)
            .fold(None, |best: Option<f64>, from| {
                Some(match best {
                    Some(had) if had >= from => had,
                    _ => from,
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timestamp_becomes_seconds_in_the_streams_own_unit() {
        assert_eq!(seconds(16_000, 1, 16_000), 1.0);
        assert_eq!(seconds(48_000, 1, 48_000), 1.0);
        assert_eq!(seconds(0, 1, 16_000), 0.0);
    }

    #[test]
    fn a_window_is_thirty_seconds_of_sixteen_kilohertz_mono() {
        assert_eq!(mel::WINDOW_SAMPLES, 480_000);
        assert!((mel::WINDOW_SECONDS - 30.0).abs() < 1e-12);
    }

    #[test]
    fn nothing_is_covered_before_a_row_arrives() {
        let speech = Speech::new();
        assert!(!speech.any());
        assert!(!speech.covers(0.0, 30.0));
    }

    #[test]
    fn a_span_covers_the_windows_it_overlaps_and_no_others() {
        let mut speech = Speech::new();
        speech.push(r#"{"text":"speech","start_t":28.0,"end_t":41.5}"#);
        assert!(speech.any());
        assert!(
            speech.covers(0.0, 30.0),
            "it reaches back into the first window"
        );
        assert!(speech.covers(30.0, 60.0));
        assert!(!speech.covers(60.0, 90.0));
    }

    #[test]
    fn a_span_touching_a_window_only_at_its_edge_does_not_cover_it() {
        let mut speech = Speech::new();
        speech.push(r#"{"start_t":10.0,"end_t":30.0}"#);
        assert!(speech.covers(0.0, 30.0));
        assert!(
            !speech.covers(30.0, 60.0),
            "it ends exactly where the window opens"
        );
    }

    #[test]
    fn a_row_from_a_module_this_one_was_not_written_for_is_passed_over() {
        let mut speech = Speech::new();
        for row in [
            r#"{"class":"person","score":0.9}"#,
            r#"{"start_t":1.0}"#,
            "not json at all",
            "",
        ] {
            speech.push(row);
        }
        assert!(!speech.any(), "none of them said anything about a span");
    }

    #[test]
    fn the_onset_a_word_is_pulled_forward_to_is_the_latest_one_it_reaches() {
        let mut speech = Speech::new();
        speech.push(r#"{"start_t":10.2,"end_t":14.0}"#);
        speech.push(r#"{"start_t":16.0,"end_t":19.0}"#);
        // A word the model stretched from the top of the window back to the
        // speech: the onset inside it is where the sound starts.
        assert_eq!(speech.onset_before(10.4, 0.0), Some(10.2));
        // Two onsets in reach: the later one, which is the nearer.
        assert_eq!(speech.onset_before(17.0, 0.0), Some(16.0));
        // A word wholly before any onset is left where it is.
        assert_eq!(speech.onset_before(5.0, 0.0), None);
    }

    #[test]
    fn a_word_is_never_pulled_back_past_where_it_already_starts() {
        let mut speech = Speech::new();
        speech.push(r#"{"start_t":10.2,"end_t":14.0}"#);
        assert_eq!(
            speech.onset_before(13.0, 12.0),
            None,
            "the onset is behind the word's own start, so it says nothing"
        );
    }

    #[test]
    fn spans_no_window_can_reach_any_more_are_forgotten() {
        let mut speech = Speech::new();
        speech.push(r#"{"start_t":1.0,"end_t":2.0}"#);
        speech.push(r#"{"start_t":40.0,"end_t":50.0}"#);
        speech.forget(30.0);
        assert!(!speech.covers(0.0, 30.0), "the early span is gone");
        assert!(speech.covers(30.0, 60.0), "the late one is still there");
        assert!(speech.any(), "and a detector is still known to be wired in");
    }
}
