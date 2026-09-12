//! Whisper's token ids: the prefix a decode is forced to open on, and the
//! text and times its answer is read back as.
//!
//! The multilingual vocabulary runs 0 to 50256; everything above it is a
//! special token at a fixed id, and everything from 50364 up is a timestamp
//! two hundredths of a second apart. Nothing here asks the graph what any of
//! that is - the ids are the model's own and fixed for its whole family.

use std::collections::HashMap;

/// Whisper's multilingual vocabulary, compiled in: about a megabyte of JSON,
/// which is the whole of what the guest needs to read an answer back. The
/// merges beside it in the model repository are for turning text INTO tokens,
/// and nothing here ever does that.
pub const VOCAB_JSON: &[u8] = include_bytes!("../vocab.json");

/// The last id that is a piece of text. Everything at or above it is special.
pub const SPECIALS: u32 = 50257;

/// End of text, which is where a decode stops.
pub const EOT: u32 = 50257;

/// Start of transcript, which every decode opens on.
pub const SOT: u32 = 50258;

/// `<|en|>`, and the 98 languages after it in `LANGUAGES` order.
pub const LANGUAGE: u32 = 50259;

/// `<|translate|>` and `<|transcribe|>`, which say which job the decode does.
pub const TRANSLATE: u32 = 50358;
pub const TRANSCRIBE: u32 = 50359;

/// `<|0.00|>`, the first timestamp. Forcing it as the last id of the prefix is
/// what puts the decode in timestamp mode; without it the model emits
/// `<|notimestamps|>` itself and the answer carries no times.
pub const TIMESTAMP: u32 = 50364;

/// Seconds one timestamp id is worth.
pub const TIMESTAMP_STEP: f64 = 0.02;

/// The languages whisper was trained on, as ISO 639-1 codes, in the order
/// their tokens sit in: `LANGUAGES[i]` is `LANGUAGE + i`.
pub const LANGUAGES: [&str; 99] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

/// Which of whisper's two jobs a decode does. Translation is always into
/// English; the model knows no other direction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Task {
    Transcribe,
    Translate,
}

impl Task {
    /// The task named, or None for a word that is neither.
    pub fn named(written: &str) -> Option<Task> {
        match written {
            "transcribe" => Some(Task::Transcribe),
            "translate" => Some(Task::Translate),
            _ => None,
        }
    }

    pub fn token(self) -> u32 {
        match self {
            Task::Transcribe => TRANSCRIBE,
            Task::Translate => TRANSLATE,
        }
    }
}

/// The token for a language code, or None for one whisper does not know.
pub fn language_token(code: &str) -> Option<u32> {
    LANGUAGES
        .iter()
        .position(|known| *known == code)
        .map(|index| LANGUAGE + index as u32)
}

/// The language code a token names, or None for a token that is not one.
pub fn language_of(token: u32) -> Option<&'static str> {
    token
        .checked_sub(LANGUAGE)
        .and_then(|index| LANGUAGES.get(index as usize))
        .copied()
}

/// The ids a decode is forced to open on.
///
/// `language` unset asks the model to detect one and say so as its first
/// token, which is also the only way to hear that a window holds no speech at
/// all: a silent window comes back with no language token in it. A prefix that
/// names one goes on to the task and, when times are wanted, the first
/// timestamp.
pub fn prefix(language: Option<u32>, task: Task, timestamps: bool) -> Vec<i32> {
    let mut ids = vec![SOT as i32];
    let Some(language) = language else {
        return ids;
    };
    ids.push(language as i32);
    ids.push(task.token() as i32);
    if timestamps {
        ids.push(TIMESTAMP as i32);
    }
    ids
}

/// The first language token in an answer, which is what the model detected.
/// None means it heard no speech.
pub fn detected(sequence: &[u32]) -> Option<&'static str> {
    sequence.iter().copied().find_map(language_of)
}

/// One stretch of speech the model cut out of a window, in seconds from the
/// start of that window.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Whisper's byte-level vocabulary, as the bytes each id stands for.
///
/// The tokens are written as text in `vocab.json`, each character standing for
/// one byte through the same table GPT-2 uses: the printable bytes stand for
/// themselves and the other 68 are moved up past 255, so a token is a string
/// even when the bytes it holds are half a character.
pub struct Vocab {
    pieces: Vec<Vec<u8>>,
}

/// Whether a byte stands for itself in the byte-level alphabet.
fn prints(byte: u8) -> bool {
    matches!(byte, 33..=126 | 161..=172 | 174..=255)
}

/// The byte each character of a token stands for, indexed by code point.
fn byte_of() -> Vec<Option<u8>> {
    // The moved bytes take 256 upwards, in the order they are passed over.
    let mut table = vec![None; 324];
    let mut moved = 256usize;
    for byte in 0..=255u8 {
        let at = if prints(byte) {
            usize::from(byte)
        } else {
            let at = moved;
            moved += 1;
            at
        };
        table[at] = Some(byte);
    }
    table
}

impl Vocab {
    /// Whisper's own vocabulary, the one compiled in.
    pub fn whisper() -> Result<Vocab, String> {
        Vocab::parse(VOCAB_JSON)
    }

    /// Reads `vocab.json`: a map from token text to id.
    pub fn parse(json: &[u8]) -> Result<Vocab, String> {
        let read: HashMap<String, u32> = serde_json::from_slice(json)
            .map_err(|e| format!("the vocabulary does not parse: {e}"))?;
        let table = byte_of();
        let mut pieces = vec![Vec::new(); SPECIALS as usize];
        for (text, id) in read {
            let Some(slot) = pieces.get_mut(id as usize) else {
                continue; // a special token, which is not a piece of text
            };
            let mut bytes = Vec::with_capacity(text.len());
            for character in text.chars() {
                let byte = table
                    .get(character as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| {
                        format!("the vocabulary holds {character:?}, which stands for no byte")
                    })?;
                bytes.push(byte);
            }
            *slot = bytes;
        }
        Ok(Vocab { pieces })
    }

    /// The bytes one id stands for, empty for a special token and for an id
    /// the vocabulary does not name.
    ///
    /// A word boundary is a property of the BYTES, not of the text: whisper
    /// opens a new word on a piece that begins with a space, and a piece can
    /// be half a character, so the alignment walks the pieces itself rather
    /// than decoding a run and trying to cut the string back up.
    pub fn piece(&self, id: u32) -> &[u8] {
        self.pieces
            .get(id as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// The text a run of ids spells. Special tokens carry no text and are
    /// passed over; bytes that do not spell valid UTF-8 - which is what half a
    /// character at the end of a run looks like - come back as the
    /// replacement character rather than stopping the decode.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(piece) = self.pieces.get(*id as usize) {
                bytes.extend_from_slice(piece);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The segments one answer cuts into, at the timestamps the model put in
    /// it. A run of words the model never closed runs to `window_seconds`.
    ///
    /// The prefix the decode was forced to open on is in the answer too, and
    /// is skipped along with every other special token.
    pub fn segments(&self, sequence: &[u32], window_seconds: f64) -> Vec<Segment> {
        let mut segments = Vec::new();
        let mut pending: Vec<u32> = Vec::new();
        let mut start = 0f64;

        for id in sequence.iter().copied() {
            if id >= TIMESTAMP {
                let at = f64::from(id - TIMESTAMP) * TIMESTAMP_STEP;
                self.flush(&mut pending, start, at, &mut segments);
                start = at;
            } else if id >= SPECIALS {
                continue;
            } else {
                pending.push(id);
            }
        }
        self.flush(&mut pending, start, window_seconds, &mut segments);
        segments
    }

    /// The words gathered so far as one segment, dropped when it is empty or
    /// its times do not run forwards.
    fn flush(&self, pending: &mut Vec<u32>, start: f64, end: f64, out: &mut Vec<Segment>) {
        if pending.is_empty() {
            return;
        }
        let text = self.decode(pending).trim().to_string();
        pending.clear();
        if !text.is_empty() && end > start {
            out.push(Segment { start, end, text });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary spelling one ASCII letter per id, which is enough to read
    /// a segmenting back without carrying a megabyte of JSON into a test.
    fn letters() -> Vocab {
        let mut pieces = vec![Vec::new(); SPECIALS as usize];
        for (id, slot) in pieces.iter_mut().enumerate().take(26) {
            *slot = vec![b'a' + id as u8];
        }
        // A space, so a segment's text can be trimmed.
        pieces[26] = vec![b' '];
        Vocab { pieces }
    }

    #[test]
    fn every_language_whisper_knows_is_a_two_or_three_letter_code() {
        assert_eq!(LANGUAGES.len(), 99);
        for code in LANGUAGES {
            assert!(
                (2..=3).contains(&code.len()) && code.chars().all(|c| c.is_ascii_lowercase()),
                "{code} is not a language code"
            );
        }
    }

    #[test]
    fn the_language_tokens_run_from_english_up_to_the_task_tokens() {
        assert_eq!(language_token("en"), Some(LANGUAGE));
        assert_eq!(language_token("es"), Some(LANGUAGE + 3));
        assert_eq!(language_token("su"), Some(LANGUAGE + 98));
        assert_eq!(language_token("klingon"), None);
        assert_eq!(
            LANGUAGE + LANGUAGES.len() as u32,
            TRANSLATE,
            "the last language sits right below <|translate|>"
        );
    }

    #[test]
    fn a_language_token_reads_back_as_the_code_it_names() {
        for (index, code) in LANGUAGES.iter().enumerate() {
            assert_eq!(language_of(LANGUAGE + index as u32), Some(*code));
        }
        assert_eq!(language_of(SOT), None);
        assert_eq!(language_of(TRANSLATE), None);
        assert_eq!(language_of(TIMESTAMP), None);
    }

    #[test]
    fn a_prefix_that_names_no_language_asks_the_model_for_one() {
        assert_eq!(prefix(None, Task::Transcribe, true), vec![SOT as i32]);
    }

    #[test]
    fn a_prefix_ends_on_a_timestamp_only_when_times_are_wanted() {
        let spanish = language_token("es").expect("spanish");
        assert_eq!(
            prefix(Some(spanish), Task::Transcribe, true),
            vec![
                SOT as i32,
                spanish as i32,
                TRANSCRIBE as i32,
                TIMESTAMP as i32
            ]
        );
        assert_eq!(
            prefix(Some(spanish), Task::Transcribe, false),
            vec![SOT as i32, spanish as i32, TRANSCRIBE as i32]
        );
    }

    #[test]
    fn translating_swaps_one_token_and_nothing_else() {
        let spanish = language_token("es").expect("spanish");
        let transcribing = prefix(Some(spanish), Task::Transcribe, true);
        let translating = prefix(Some(spanish), Task::Translate, true);
        let differ: Vec<usize> = transcribing
            .iter()
            .zip(&translating)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(at, _)| at)
            .collect();
        assert_eq!(differ, vec![2]);
    }

    #[test]
    fn a_task_is_one_of_two_words() {
        assert_eq!(Task::named("transcribe"), Some(Task::Transcribe));
        assert_eq!(Task::named("translate"), Some(Task::Translate));
        assert_eq!(Task::named("Translate"), None);
        assert_eq!(Task::named("summarize"), None);
    }

    #[test]
    fn the_detected_language_is_the_first_language_token_in_the_answer() {
        let spanish = language_token("es").expect("spanish");
        assert_eq!(detected(&[SOT, spanish, TRANSCRIBE, 1, 2]), Some("es"));
    }

    #[test]
    fn an_answer_with_no_language_token_is_a_window_with_no_speech() {
        assert_eq!(detected(&[SOT, EOT]), None);
        assert_eq!(detected(&[]), None);
    }

    #[test]
    fn every_byte_stands_for_exactly_one_character() {
        let table = byte_of();
        let mut seen = [false; 256];
        for byte in table.iter().flatten() {
            assert!(!seen[usize::from(*byte)], "{byte} twice");
            seen[usize::from(*byte)] = true;
        }
        assert!(seen.iter().all(|had| *had), "every byte is spelled");
    }

    #[test]
    fn a_printable_byte_stands_for_itself_and_a_space_does_not() {
        let table = byte_of();
        assert_eq!(table[usize::from(b'a')], Some(b'a'));
        assert_eq!(table[usize::from(b'~')], Some(b'~'));
        assert_eq!(table[usize::from(b' ')], None, "a space is moved up");
        // The first byte moved is 0, and a space is the 33rd.
        assert_eq!(table[256], Some(0));
        assert_eq!(table[256 + 32], Some(b' '));
    }

    #[test]
    fn a_token_written_as_text_decodes_to_the_bytes_it_holds() {
        // "Ġde" is a space and two letters, which is how whisper spells a word
        // that follows one.
        let json = r#"{"Ġde": 0, "ma": 1}"#;
        let vocab = Vocab::parse(json.as_bytes()).expect("the vocabulary parses");
        assert_eq!(vocab.decode(&[0, 1]), " dema");
    }

    #[test]
    fn whispers_own_vocabulary_names_every_id_up_to_the_end_of_text() {
        let read: HashMap<String, u32> =
            serde_json::from_slice(VOCAB_JSON).expect("the vocabulary parses");
        let mut ids: Vec<u32> = read.values().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.first(), Some(&0));
        assert_eq!(ids.last(), Some(&EOT));
        assert_eq!(
            ids.len(),
            EOT as usize + 1,
            "and nothing in between is missing"
        );

        let vocab = Vocab::whisper().expect("the vocabulary parses");
        assert_eq!(vocab.pieces.len(), SPECIALS as usize);
        // A word that follows a space, spelled the way the byte alphabet
        // spells one, read back out of the real file.
        let de = read
            .get("\u{0120}de")
            .copied()
            .expect("whisper spells ' de'");
        assert_eq!(vocab.decode(&[de]), " de");
    }

    #[test]
    fn a_special_token_carries_no_text() {
        let vocab = letters();
        assert_eq!(vocab.decode(&[0, SOT, 1, EOT, 2]), "abc");
    }

    #[test]
    fn a_piece_is_the_bytes_an_id_stands_for_and_a_special_stands_for_none() {
        let vocab = letters();
        assert_eq!(vocab.piece(0), b"a");
        assert_eq!(vocab.piece(26), b" ");
        assert!(vocab.piece(EOT).is_empty(), "end-of-text is not text");
        assert!(vocab.piece(SOT).is_empty());
        assert!(vocab.piece(TIMESTAMP).is_empty());
        assert!(vocab.piece(u32::MAX).is_empty(), "and neither is nothing");
    }

    #[test]
    fn whispers_own_pieces_carry_the_leading_space_that_opens_a_word() {
        let vocab = Vocab::whisper().expect("the vocabulary parses");
        // 848 is " said" and 5186 " yesterday": the ids the fixture window
        // decoded, and the space is the whole of what makes them new words.
        assert_eq!(vocab.piece(848), b" said");
        assert_eq!(vocab.piece(5186), b" yesterday");
        // 13 is a bare full stop, which opens a word of its own.
        assert_eq!(vocab.piece(13), b".");
    }

    #[test]
    fn a_run_between_two_timestamps_is_one_segment_at_those_times() {
        let vocab = letters();
        let found = vocab.segments(&[SOT, TIMESTAMP, 0, 1, TIMESTAMP + 350], 30.0);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "ab");
        assert!(found[0].start.abs() < 1e-12);
        assert!(
            (found[0].end - 7.0).abs() < 1e-12,
            "350 steps is seven seconds"
        );
    }

    #[test]
    fn the_segments_of_a_window_run_in_order_and_do_not_overlap() {
        let vocab = letters();
        let found = vocab.segments(
            &[
                SOT,
                TIMESTAMP,
                0,
                TIMESTAMP + 350,
                TIMESTAMP + 350,
                1,
                TIMESTAMP + 450,
                TIMESTAMP + 450,
                2,
                TIMESTAMP + 700,
                EOT,
            ],
            30.0,
        );
        assert_eq!(found.len(), 3);
        for pair in found.windows(2) {
            assert!(pair[0].end <= pair[1].start, "{pair:?} overlap");
        }
        assert_eq!(found[2].text, "c");
        assert!((found[2].start - 9.0).abs() < 1e-12);
        assert!((found[2].end - 14.0).abs() < 1e-12);
    }

    #[test]
    fn words_the_model_never_closed_run_to_the_end_of_the_window() {
        let vocab = letters();
        let found = vocab.segments(&[SOT, TIMESTAMP + 100, 0, 1, EOT], 30.0);
        assert_eq!(found.len(), 1);
        assert!((found[0].start - 2.0).abs() < 1e-12);
        assert!((found[0].end - 30.0).abs() < 1e-12);
    }

    #[test]
    fn words_before_the_first_timestamp_start_at_the_windows_own_beginning() {
        let vocab = letters();
        let found = vocab.segments(&[SOT, 0, 1, TIMESTAMP + 50], 30.0);
        assert_eq!(found.len(), 1);
        assert!(found[0].start.abs() < 1e-12);
        assert!((found[0].end - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_empty_span_produces_no_segment() {
        let vocab = letters();
        // Two timestamps with nothing between them, and a span of no length.
        assert!(vocab
            .segments(&[SOT, TIMESTAMP, TIMESTAMP + 100], 30.0)
            .is_empty());
        assert!(vocab
            .segments(&[SOT, TIMESTAMP + 100, 26, TIMESTAMP + 100], 30.0)
            .is_empty());
    }

    #[test]
    fn an_answer_that_is_only_the_prefix_produces_nothing() {
        let vocab = letters();
        assert!(vocab.segments(&[SOT, EOT], 30.0).is_empty());
    }
}
