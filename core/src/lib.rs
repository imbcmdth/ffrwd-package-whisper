//! The arithmetic the modules are built on: the spectrogram whisper reads, the
//! token ids it is steered with, the parameters a decode is steered by, the
//! rows it leaves as, and the onsets an upstream detector's rows carry. None
//! of it touches the model, so all of it is tested on the host.
//!
//! Both exports - `transcribe` and `transcribe_words` - are the same decode
//! read two different ways, so everything either of them would otherwise have
//! had its own copy of lives here. What is not here is what only one of them
//! does: the alignment, which only `transcribe_words` runs, sits in that
//! crate beside the graph plumbing that feeds it.

pub mod cue;
pub mod mel;
pub mod node;
pub mod params;
pub mod tensor;
pub mod tokens;

/// Where an upstream detector heard speech in one window: the second each of
/// its rows says a span began.
///
/// The rows reach a window by their time, and every row stamped inside it is
/// there when the window is, so this is the whole of what the detector said
/// about the window and nothing carries over to the next.
#[derive(Default)]
pub struct Speech {
    onsets: Vec<f64>,
}

impl Speech {
    pub fn new() -> Speech {
        Speech::default()
    }

    /// One row: the second the span it belongs to began.
    pub fn heard(&mut self, start_t: f64) {
        if !self.onsets.contains(&start_t) {
            self.onsets.push(start_t);
        }
    }

    /// Whether any row arrived for the window.
    pub fn any(&self) -> bool {
        !self.onsets.is_empty()
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
        self.onsets
            .iter()
            .copied()
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
    fn a_window_is_thirty_seconds_of_sixteen_kilohertz_mono() {
        assert_eq!(mel::WINDOW_SAMPLES, 480_000);
        assert!((mel::WINDOW_SECONDS - 30.0).abs() < 1e-12);
    }

    #[test]
    fn nothing_is_heard_before_a_row_arrives() {
        let mut speech = Speech::new();
        assert!(!speech.any());
        speech.heard(28.0);
        assert!(speech.any());
    }

    #[test]
    fn the_rows_of_one_span_are_one_onset() {
        let mut speech = Speech::new();
        for _ in 0..40 {
            speech.heard(10.2);
        }
        assert_eq!(speech.onsets, [10.2]);
    }

    #[test]
    fn the_onset_a_word_is_pulled_forward_to_is_the_latest_one_it_reaches() {
        let mut speech = Speech::new();
        speech.heard(10.2);
        speech.heard(16.0);
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
        speech.heard(10.2);
        assert_eq!(
            speech.onset_before(13.0, 12.0),
            None,
            "the onset is behind the word's own start, so it says nothing"
        );
    }

    #[test]
    fn a_span_begun_in_the_window_before_gives_this_one_no_onset() {
        // Its rows inside this window still say the window holds speech.
        let mut speech = Speech::new();
        speech.heard(29.92);
        assert!(speech.any());
        assert_eq!(speech.onset_before(30.74, 30.0), None);
    }
}
