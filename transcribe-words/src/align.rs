//! Word times out of the decoder's cross-attention.
//!
//! The graph hands back `cross_qk`: the raw, already-scaled Q*K of the
//! decoder's cross-attention for the (layer, head) pairs it was asked for -
//! one row per decoder step, one column per encoder frame, 1500 of them for a
//! 30 s window, so 20 ms each. Where a token's attention sits in that row is
//! where in the audio the token was said, and the job here is to turn a
//! sequence of those rows into a sequence of spans.
//!
//! Every step mirrors `whisper/timing.py` from `openai-whisper`, in its order
//! and with its arithmetic, so the times are the ones that model produces and
//! not a second opinion: slice to the frames that are audio, softmax over
//! frames, normalise over the token axis, median-filter width 7 over frames,
//! average the alignment heads, negate, dynamic-time-warp the tokens against
//! the frames, then cut the token path on the tokenizer's word boundaries and
//! let punctuation ride the word it belongs to.
//!
//! # The row every token reads
//!
//! This is the one piece of bookkeeping worth getting right, and getting it
//! wrong is a clean two-hundred-millisecond bias rather than noise.
//!
//! A decoder step's query is the token ALREADY decided, so its cross-attention
//! is the evidence for the token that step produces - the NEXT one. The first
//! token of a window is not produced by the decode loop at all: the fused
//! graph decides it inside its encoder/init subgraph, which emits no row. So
//! the graph returns exactly one row fewer than it returned tokens, and
//!
//!     cross_qk row k  is the evidence for  generated[k + 1]
//!
//! leaving `generated[0]` with no row of its own. Row 0 is read twice for it,
//! which puts the first token's start at the first row's frame - see
//! [`FirstRow`] for the other arrangement, which an export carrying
//! `extra_decoding_ids` would have.

use whisper_core::tokens::{Vocab, SPECIALS};

/// The six (layer, head) pairs of whisper medium whose cross-attention carries
/// the alignment, 0-indexed, as `openai-whisper`'s own table decodes them.
///
/// Asking for these six and nothing else is what keeps the returned tensor
/// small: six heads are about 3.5 MiB for a window and cost no measurable
/// decode time, where all 24 x 16 of them are 222 MiB and five gigabytes of
/// device memory.
pub const ALIGNMENT_HEADS: [(i32, i32); 6] =
    [(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)];

/// Encoder frames in a second. The encoder halves the mel frames, so one
/// column of `cross_qk` is 20 ms.
pub const TOKENS_PER_SECOND: f64 = 50.0;

/// The width of the median filter over the frame axis.
pub const MEDFILT_WIDTH: usize = 7;

/// ASCII punctuation, in the order Python's `string.punctuation` spells it.
///
/// The order matters: `openai-whisper` asks whether a piece's stripped text is
/// `in string.punctuation`, which for Python is containment of a SUBSTRING, not
/// of a character. `"()"` is therefore punctuation to it and `".."` is not, and
/// `str::contains` on this constant answers the same question the same way.
const PUNCTUATION: &str = r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##;

/// Punctuation that rides the word AFTER it, so an opening quote is part of
/// the word it opens. Asked of with `contains`, for the reason above.
const PREPENDED: &str = "\"'\u{201c}\u{bf}([{-";

/// Punctuation that rides the word BEFORE it, so a full stop is part of the
/// word it ends.
const APPENDED: &str = "\"'.\u{3002},\u{ff0c}!\u{ff01}?\u{ff1f}:\u{ff1a}\u{201d})]}\u{3001}";

/// Which row of `cross_qk` the first generated token reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirstRow {
    /// Row 0, read a second time. This is the arrangement of the export the
    /// package ships: the first token was decided in the init subgraph, which
    /// emits no row, so there is nothing else to read for it.
    Duplicated,
    /// Row `n`, which is the row whose query is the last token of the prompt.
    /// An export carrying `extra_decoding_ids` pushes the prompt through the
    /// decode loop and so does have a row for every generated token; this
    /// module does not ask for that export.
    ///
    /// Nothing in the shipped path constructs this: it is here so that the
    /// offset above is a choice with an alternative rather than an accident,
    /// and the tests use `At(0)` - which is exactly the off-by-one - to show
    /// that reading the rows the other way moves nearly every word.
    #[allow(dead_code)]
    At(usize),
}

/// One word, in seconds from the start of the window that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// The `cross_qk` tensor as the graph returns it, flattened: head-major, then
/// step, then frame.
pub struct CrossQk<'a> {
    pub data: &'a [f32],
    pub heads: usize,
    pub rows: usize,
    pub frames: usize,
}

impl CrossQk<'_> {
    /// Whether the tensor's dimensions and payload agree with each other.
    pub fn is_whole(&self) -> bool {
        self.data.len() == self.heads * self.rows * self.frames
    }
}

/// A run of tokens that will become one word, or one special token standing on
/// its own so that it takes its own slice of the path and can be dropped after.
struct Group {
    text: String,
    tokens: usize,
    special: bool,
}

/// The word spans a window's tokens and their cross-attention spell.
///
/// `generated` is what the graph added to the prompt, end-of-text included;
/// `content_frames` is how many of the window's 3000 mel frames were audio
/// rather than padding, which is what bounds the frames a word may land on.
pub fn align(
    qk: &CrossQk,
    generated: &[u32],
    content_frames: usize,
    vocab: &Vocab,
    first_row: FirstRow,
) -> Vec<Word> {
    if qk.rows == 0 || qk.heads == 0 || generated.is_empty() || !qk.is_whole() {
        return Vec::new();
    }

    // Only the frames that held audio: the encoder saw the padding too, and
    // its attention on the padding is not evidence of anything.
    let frames = (content_frames / 2).clamp(1, qk.frames);

    // One step per token, each reading the row that is the evidence for it.
    let reach = match first_row {
        FirstRow::Duplicated => qk.rows + 1,
        FirstRow::At(n) => qk.rows.saturating_sub(n),
    };
    let steps = reach.min(generated.len());
    if steps == 0 {
        return Vec::new();
    }

    let mut weights = vec![0f32; qk.heads * steps * frames];
    for head in 0..qk.heads {
        for step in 0..steps {
            let row = match first_row {
                FirstRow::Duplicated => step.saturating_sub(1),
                FirstRow::At(n) => n + step,
            };
            let from = (head * qk.rows + row) * qk.frames;
            let into = (head * steps + step) * frames;
            weights[into..into + frames].copy_from_slice(&qk.data[from..from + frames]);
        }
    }

    softmax_over_frames(&mut weights, frames);
    normalise_over_tokens(&mut weights, qk.heads, steps, frames);
    let filtered = median_filter(&weights, frames, MEDFILT_WIDTH);
    let matrix = mean_over_heads(&filtered, qk.heads, steps, frames);

    // The warp wants a COST, and a high attention is a good match, so the
    // averaged weights are negated on the way in.
    let path = dtw(&matrix, steps, frames, |w| -f64::from(w));

    // The frame each token first lands on, in seconds. The path's token index
    // rises by at most one per cell, so this has exactly one entry per step.
    let mut jump_times: Vec<f64> = Vec::with_capacity(steps);
    let mut last_token: Option<usize> = None;
    for (token, frame) in path {
        if last_token != Some(token) {
            jump_times.push(frame as f64 / TOKENS_PER_SECOND);
            last_token = Some(token);
        }
    }
    if jump_times.is_empty() {
        return Vec::new();
    }
    let last = jump_times.len() - 1;

    let groups = groups(vocab, &generated[..steps]);
    let mut words: Vec<Word> = Vec::with_capacity(groups.len());
    let mut at = 0usize;
    for group in groups {
        let to = at + group.tokens;
        if !group.special {
            words.push(Word {
                text: group.text,
                start: jump_times[at.min(last)],
                end: jump_times[to.min(last)],
            });
        }
        at = to;
    }

    merge_punctuations(words)
}

// ------------------------------------------------------------------ the maths

/// Softmax along the frame axis, in place.
fn softmax_over_frames(weights: &mut [f32], frames: usize) {
    for row in weights.chunks_exact_mut(frames) {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0f32;
        for value in row.iter_mut() {
            *value = (*value - top).exp();
            total += *value;
        }
        if total > 0.0 {
            for value in row.iter_mut() {
                *value /= total;
            }
        }
    }
}

/// Standardise each (head, frame) column over the token axis, in place. The
/// deviation is the biased one, which is what `torch.std_mean(unbiased=False)`
/// hands `openai-whisper`.
fn normalise_over_tokens(weights: &mut [f32], heads: usize, steps: usize, frames: usize) {
    let count = steps as f32;
    for head in 0..heads {
        let base = head * steps * frames;
        for frame in 0..frames {
            let mut sum = 0f32;
            for step in 0..steps {
                sum += weights[base + step * frames + frame];
            }
            let mean = sum / count;
            let mut squares = 0f32;
            for step in 0..steps {
                let off = weights[base + step * frames + frame] - mean;
                squares += off * off;
            }
            let deviation = (squares / count).sqrt().max(1e-10);
            for step in 0..steps {
                let at = base + step * frames + frame;
                weights[at] = (weights[at] - mean) / deviation;
            }
        }
    }
}

/// Median of a sliding window of `width` along the frame axis, edges
/// reflected. A row no longer than the padding is handed back untouched, which
/// is what `openai-whisper`'s `median_filter` does with one.
fn median_filter(weights: &[f32], frames: usize, width: usize) -> Vec<f32> {
    debug_assert!(width % 2 == 1);
    let pad = width / 2;
    if frames <= pad {
        return weights.to_vec();
    }
    let mut out = vec![0f32; weights.len()];
    let mut window = vec![0f32; width];
    for (row, into) in weights
        .chunks_exact(frames)
        .zip(out.chunks_exact_mut(frames))
    {
        for (frame, middle) in into.iter_mut().enumerate() {
            for (slot, offset) in window.iter_mut().zip(0..width) {
                // Reflected, not edge-repeated: index -1 is row[1] and index
                // `frames` is row[frames - 2], the way numpy's "reflect" pads.
                let wanted = frame as isize + offset as isize - pad as isize;
                *slot = row[reflect(wanted, frames)];
            }
            window.sort_by(|a, b| a.partial_cmp(b).expect("the weights are finite"));
            *middle = window[pad];
        }
    }
    out
}

/// An index reflected back inside `0..len`, the way numpy's `reflect` padding
/// reflects one: the edge element is not repeated.
fn reflect(at: isize, len: usize) -> usize {
    let last = len as isize - 1;
    let mut at = at;
    // A loop rather than one step, so a window wider than the row still lands
    // somewhere real.
    while at < 0 || at > last {
        if at < 0 {
            at = -at;
        }
        if at > last {
            at = 2 * last - at;
        }
    }
    at as usize
}

/// The alignment heads averaged into one (token, frame) matrix.
fn mean_over_heads(weights: &[f32], heads: usize, steps: usize, frames: usize) -> Vec<f32> {
    let mut out = vec![0f32; steps * frames];
    for head in 0..heads {
        let base = head * steps * frames;
        for (at, value) in out.iter_mut().enumerate() {
            *value += weights[base + at];
        }
    }
    let count = heads as f32;
    for value in out.iter_mut() {
        *value /= count;
    }
    out
}

/// Dynamic time warping over a cost matrix of shape (tokens, frames), as the
/// path of (token, frame) cells that runs from the first of each to the last.
///
/// The recurrence and the tie-breaking are `openai-whisper`'s `dtw_cpu`: a
/// diagonal step wins only when it is strictly cheaper than both others, a
/// token step only when it is strictly cheaper than both, and every tie goes
/// to the frame step. That is what makes this trace the same trace, and not
/// merely a trace of the same length.
fn dtw(matrix: &[f32], tokens: usize, frames: usize, cost_of: impl Fn(f32) -> f64) -> Vec<(usize, usize)> {
    const DIAGONAL: i8 = 0;
    const TOKEN_STEP: i8 = 1;
    const FRAME_STEP: i8 = 2;

    let stride = frames + 1;
    let mut cost = vec![f64::INFINITY; (tokens + 1) * stride];
    let mut trace = vec![-1i8; (tokens + 1) * stride];
    cost[0] = 0.0;

    for i in 1..=tokens {
        for j in 1..=frames {
            let diagonal = cost[(i - 1) * stride + (j - 1)];
            let token_step = cost[(i - 1) * stride + j];
            let frame_step = cost[i * stride + (j - 1)];
            let step = if diagonal < token_step && diagonal < frame_step {
                DIAGONAL
            } else if token_step < diagonal && token_step < frame_step {
                TOKEN_STEP
            } else {
                FRAME_STEP
            };
            let cheapest = match step {
                DIAGONAL => diagonal,
                TOKEN_STEP => token_step,
                _ => frame_step,
            };
            cost[i * stride + j] = cost_of(matrix[(i - 1) * frames + (j - 1)]) + cheapest;
            trace[i * stride + j] = step;
        }
    }

    // The edges the backtrace runs out along. The first column is written
    // second, so the origin ends up a token step - which is what
    // `openai-whisper` leaves there, and which the walk below never reads.
    for slot in trace[..stride].iter_mut() {
        *slot = FRAME_STEP;
    }
    for i in 0..=tokens {
        trace[i * stride] = TOKEN_STEP;
    }

    let mut path = Vec::with_capacity(tokens + frames);
    let (mut i, mut j) = (tokens, frames);
    while i > 0 || j > 0 {
        // The trace can leave the first row or column only at the origin: for
        // anything but cost[0, 0] those cells are infinite, so the backtrace
        // reaches (1, 1) and steps diagonally out of it. Neither index is ever
        // zero here.
        debug_assert!(i > 0 && j > 0, "the path left the matrix at ({i}, {j})");
        path.push((i.saturating_sub(1), j.saturating_sub(1)));
        match trace[i * stride + j] {
            DIAGONAL => {
                i -= 1;
                j -= 1;
            }
            TOKEN_STEP => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path
}

// ------------------------------------------------------------- words from ids

/// The token ids of one window cut into the groups the path is sliced on: one
/// per word, and one per special token so that a timestamp takes a row of its
/// own and can be dropped without moving anything else.
fn groups(vocab: &Vocab, tokens: &[u32]) -> Vec<Group> {
    let mut out: Vec<Group> = Vec::new();
    let mut run: Vec<u32> = Vec::new();
    for id in tokens.iter().copied() {
        if id >= SPECIALS {
            out.extend(words_of(vocab, &run));
            run.clear();
            out.push(Group {
                text: String::new(),
                tokens: 1,
                special: true,
            });
        } else {
            run.push(id);
        }
    }
    out.extend(words_of(vocab, &run));
    out
}

/// A run of text tokens cut into words on the tokenizer's own boundaries.
///
/// Whisper's rule, which is a rule about the BYTES a piece carries: a piece
/// that opens on a space, or is bare punctuation, starts a new word, and
/// anything else joins the word before it. A piece whose bytes are half a
/// character is held until a later piece completes it, so a word is never cut
/// through the middle of one.
fn words_of(vocab: &Vocab, run: &[u32]) -> Vec<Group> {
    // Pieces gathered into the smallest runs that are whole UTF-8.
    let mut pieces: Vec<(String, usize)> = Vec::new();
    let mut held: Vec<u8> = Vec::new();
    let mut counted = 0usize;
    for id in run.iter().copied() {
        held.extend_from_slice(vocab.piece(id));
        counted += 1;
        if let Ok(whole) = std::str::from_utf8(&held) {
            pieces.push((whole.to_string(), counted));
            held.clear();
            counted = 0;
        }
    }
    if counted > 0 {
        // Bytes the window ended in the middle of. They are still tokens and
        // still took their rows, so they become a piece rather than vanishing.
        pieces.push((String::from_utf8_lossy(&held).into_owned(), counted));
    }

    let mut out: Vec<Group> = Vec::new();
    for (text, tokens) in pieces {
        let opens_a_word = text.starts_with(' ')
            || PUNCTUATION.contains(text.trim())
            || out.is_empty();
        if opens_a_word {
            out.push(Group {
                text,
                tokens,
                special: false,
            });
        } else if let Some(last) = out.last_mut() {
            last.text.push_str(&text);
            last.tokens += tokens;
        }
    }
    out
}

/// Punctuation joined to the word it belongs to: a leading quote rides the
/// word after it, a trailing comma the word before it.
///
/// `openai-whisper`'s `merge_punctuations`, including that the merged word
/// keeps the times of the word rather than of the punctuation - the quote's
/// own span is discarded, not unioned in.
pub fn merge_punctuations(mut words: Vec<Word>) -> Vec<Word> {
    if words.len() < 2 {
        return words.into_iter().filter(|w| !w.text.is_empty()).collect();
    }

    // Backwards, so a run of opening punctuation all lands on the same word.
    let mut j = words.len() - 1;
    for i in (0..words.len() - 1).rev() {
        let rides_on = words[i].text.starts_with(' ') && PREPENDED.contains(words[i].text.trim());
        if rides_on {
            let carried = std::mem::take(&mut words[i].text);
            words[j].text = carried + &words[j].text;
        } else {
            j = i;
        }
    }

    // Forwards, for the closing half. The piece is NOT trimmed here, so a
    // stop that arrived with a space in front of it is a word of its own.
    let mut i = 0usize;
    for j in 1..words.len() {
        let rides_on = !words[i].text.ends_with(' ') && APPENDED.contains(words[j].text.as_str());
        if rides_on {
            let carried = std::mem::take(&mut words[j].text);
            words[i].text.push_str(&carried);
        } else {
            i = j;
        }
    }

    words.retain(|word| !word.text.is_empty());
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whisper() -> Vocab {
        Vocab::whisper().expect("the vocabulary parses")
    }

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
        }
    }

    // ------------------------------------------------------------ the filter

    #[test]
    fn the_median_filter_takes_the_middle_of_seven_reflected() {
        // Hand-computed. The row is nine long, so every window is full.
        let row: Vec<f32> = vec![5.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0];
        let out = median_filter(&row, 9, 7);
        // frame 0: indices -3..3 reflect to 3,2,1,0,1,2,3 -> values
        // 2,9,1,5,1,9,2 -> sorted 1,1,2,2,5,9,9 -> middle 2
        assert_eq!(out[0], 2.0);
        // frame 3: indices 0..6 -> 5,1,9,2,8,3,7 -> sorted 1,2,3,5,7,8,9 -> 5
        assert_eq!(out[3], 5.0);
        // frame 4: indices 1..7 -> 1,9,2,8,3,7,4 -> sorted 1,2,3,4,7,8,9 -> 4
        assert_eq!(out[4], 4.0);
        // frame 8: indices 5..11 reflect to 5,6,7,8,7,6,5 -> 3,7,4,6,4,7,3
        // -> sorted 3,3,4,4,6,7,7 -> 4
        assert_eq!(out[8], 4.0);
    }

    #[test]
    fn a_row_no_longer_than_the_padding_is_handed_back_untouched() {
        let row: Vec<f32> = vec![3.0, 1.0, 2.0];
        assert_eq!(median_filter(&row, 3, 7), row);
    }

    #[test]
    fn an_index_off_the_end_reflects_without_repeating_the_edge() {
        // numpy: np.pad([0,1,2,3,4], (3,3), "reflect")
        //     -> [3,2,1, 0,1,2,3,4, 3,2,1]
        assert_eq!(
            (-3..8).map(|at| reflect(at, 5)).collect::<Vec<_>>(),
            vec![3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]
        );
    }

    #[test]
    fn the_filter_runs_over_frames_and_never_across_a_row_boundary() {
        // Two rows of five. A spike at the end of the first must not reach the
        // start of the second.
        let rows: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0];
        let out = median_filter(&rows, 5, 3);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
    }

    // --------------------------------------------------------------- the dtw

    #[test]
    fn the_warp_follows_the_cheap_diagonal_of_a_small_matrix() {
        // Three tokens, three frames, one obvious diagonal of zeros.
        let matrix: Vec<f32> = vec![
            0.0, 9.0, 9.0, //
            9.0, 0.0, 9.0, //
            9.0, 9.0, 0.0,
        ];
        let path = dtw(&matrix, 3, 3, f64::from);
        assert_eq!(path, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn the_warp_holds_a_token_across_the_frames_it_was_said_over() {
        // Token 1 is cheap on frames 1, 2 and 3; token 0 on frame 0 and token
        // 2 on frame 4.
        let matrix: Vec<f32> = vec![
            0.0, 9.0, 9.0, 9.0, 9.0, //
            9.0, 0.0, 0.0, 0.0, 9.0, //
            9.0, 9.0, 9.0, 9.0, 0.0,
        ];
        let path = dtw(&matrix, 3, 5, f64::from);
        assert_eq!(
            path,
            vec![(0, 0), (1, 1), (1, 2), (1, 3), (2, 4)],
            "the middle token holds for three frames"
        );
    }

    #[test]
    fn the_warp_runs_corner_to_corner_and_steps_by_one() {
        // A flat matrix: the path is decided entirely by the tie-breaking.
        // The expected path is `whisper.timing.dtw_cpu`'s own answer for it.
        let (tokens, frames) = (4usize, 7usize);
        let matrix = vec![1.0f32; tokens * frames];
        let path = dtw(&matrix, tokens, frames, f64::from);
        assert_eq!(
            path,
            vec![(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (3, 5), (3, 6)]
        );
        for pair in path.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(
                b.0 == a.0 || b.0 == a.0 + 1,
                "the token index rises by at most one: {a:?} -> {b:?}"
            );
            assert!(b.1 == a.1 || b.1 == a.1 + 1, "and so does the frame");
            assert!(b != a, "and a cell is never repeated");
        }
        // Every token is on the path, which is what makes one jump time per
        // token.
        let touched: std::collections::BTreeSet<usize> = path.iter().map(|(t, _)| *t).collect();
        assert_eq!(touched.len(), tokens);
    }

    #[test]
    fn a_tie_takes_the_diagonal_while_it_can_and_then_holds_the_last_token() {
        // Two tokens, three frames, all equal. A tie goes to the frame step,
        // but only once the diagonal has stopped being strictly cheaper - so
        // the path runs diagonally out of the corner and the LAST token, not
        // the first, is the one that absorbs the spare frames. Again
        // `dtw_cpu`'s own answer.
        let matrix = vec![1.0f32; 6];
        assert_eq!(dtw(&matrix, 2, 3, f64::from), vec![(0, 0), (1, 1), (1, 2)]);
    }

    // ------------------------------------------------------- word boundaries

    fn texts(groups: &[Group]) -> Vec<String> {
        groups.iter().map(|g| g.text.clone()).collect()
    }

    #[test]
    fn a_piece_that_opens_on_a_space_starts_a_word_and_others_join_one() {
        let vocab = whisper();
        // " comparing" is one piece; "yester" + "day" would join.
        let split = words_of(&vocab, &[848, 5186, 321]); // " said yesterday we"
        assert_eq!(texts(&split), vec![" said", " yesterday", " we"]);
        for group in &split {
            assert_eq!(group.tokens, 1);
            assert!(!group.special);
        }
    }

    #[test]
    fn bare_punctuation_is_a_word_of_its_own_and_a_space_before_it_is_not() {
        let vocab = whisper();
        // 1446 is " book", 13 a bare full stop, 400 " And".
        let split = words_of(&vocab, &[1446, 13, 400]);
        assert_eq!(texts(&split), vec![" book", ".", " And"]);
    }

    #[test]
    fn a_piece_that_is_neither_joins_the_word_before_it() {
        let vocab = whisper();
        // A name whisper spells in two pieces: " Cop" + "olo". The second
        // opens on no space and is not punctuation, so it is not a word.
        let split = words_of(&vocab, &[11579, 7902]);
        assert_eq!(texts(&split), vec![" Copolo"]);
        assert_eq!(split[0].tokens, 2, "and it carries both tokens");
    }

    #[test]
    fn a_character_spread_over_two_tokens_is_held_until_it_completes() {
        let vocab = whisper();
        // Whisper's vocabulary is byte-level, so a character outside ASCII can
        // arrive in pieces: 940 is the first two bytes of U+4E2D and 255 the
        // third. Neither is text on its own.
        assert_eq!(vocab.piece(940), &[0xe4, 0xb8]);
        assert_eq!(vocab.piece(255), &[0xad]);
        assert!(std::str::from_utf8(vocab.piece(940)).is_err());

        // " said" then the two halves of one character.
        let split = words_of(&vocab, &[848, 940, 255]);
        assert_eq!(
            texts(&split),
            vec![" said\u{4e2d}"],
            "the halves never became a word, or half a word, of their own"
        );
        assert_eq!(split[0].tokens, 3, "and all three tokens are accounted for");
    }

    #[test]
    fn bytes_the_window_ended_in_the_middle_of_still_take_their_rows() {
        let vocab = whisper();
        // The window stopped after the first half of a character, which is
        // what a decode cut short by `max_length` looks like.
        let split = words_of(&vocab, &[848, 940]);
        // The half character cannot be decoded and comes back as the
        // replacement, but its token is still counted - that is what keeps the
        // rest of the path lined up with the rows.
        assert_eq!(split.iter().map(|g| g.tokens).sum::<usize>(), 2);
        assert!(split.last().expect("a word").text.contains('\u{fffd}'));
    }

    #[test]
    fn a_special_token_takes_a_group_of_its_own_and_is_marked_as_one() {
        let vocab = whisper();
        let split = groups(&vocab, &[848, 5186, 51064, 50257]);
        assert_eq!(split.len(), 4);
        assert_eq!(texts(&split)[..2], [" said".to_string(), " yesterday".to_string()]);
        assert!(!split[0].special && !split[1].special);
        assert!(split[2].special && split[3].special, "a timestamp and EOT");
        assert_eq!(split[2].tokens, 1);
        assert_eq!(
            split.iter().map(|g| g.tokens).sum::<usize>(),
            4,
            "every token is accounted for exactly once"
        );
    }

    // --------------------------------------------------------- the merging

    #[test]
    fn a_trailing_full_stop_rides_the_word_before_it() {
        let merged = merge_punctuations(vec![
            word(" book", 1.0, 2.0),
            word(".", 2.0, 2.6),
            word(" And", 2.6, 2.8),
        ]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].text, " book.");
        assert_eq!(
            (merged[0].start, merged[0].end),
            (1.0, 2.0),
            "and the word keeps its own times, not the stop's"
        );
        assert_eq!(merged[1].text, " And");
    }

    #[test]
    fn a_leading_quote_rides_the_word_after_it() {
        let merged = merge_punctuations(vec![
            word(" said", 1.0, 1.4), //
            word(" \"", 1.4, 1.5),
            word("Really", 1.5, 2.0),
        ]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].text, " said");
        assert_eq!(merged[1].text, " \"Really");
        assert_eq!(
            (merged[1].start, merged[1].end),
            (1.5, 2.0),
            "the quote's own span is discarded, not unioned in"
        );
    }

    #[test]
    fn a_quote_and_a_stop_on_the_same_word_both_ride_it() {
        let merged = merge_punctuations(vec![
            word(" \"", 1.0, 1.1),
            word("Really", 1.1, 1.8),
            word("?", 1.8, 1.9),
            word("\"", 1.9, 2.0),
            word(" said", 2.0, 2.3),
        ]);
        assert_eq!(
            merged.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            vec![" \"Really?\"", " said"]
        );
        assert_eq!((merged[0].start, merged[0].end), (1.1, 1.8));
    }

    #[test]
    fn punctuation_that_rides_nothing_is_left_as_its_own_word() {
        // A dash is prepended punctuation, but with no word after it there is
        // nothing for it to ride.
        let merged = merge_punctuations(vec![word(" hello", 0.0, 1.0), word(" -", 1.0, 1.2)]);
        assert_eq!(
            merged.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            vec![" hello", " -"]
        );
    }

    #[test]
    fn a_stop_after_a_word_that_ends_in_a_space_stays_its_own_word() {
        let merged = merge_punctuations(vec![word(" hello ", 0.0, 1.0), word(".", 1.0, 1.2)]);
        assert_eq!(merged.len(), 2, "{merged:?}");
    }

    #[test]
    fn one_word_or_none_comes_back_as_it_went_in() {
        assert!(merge_punctuations(vec![]).is_empty());
        let one = merge_punctuations(vec![word(" hello", 0.0, 1.0)]);
        assert_eq!(one.len(), 1);
    }

    // -------------------------------------------------- python's punctuation

    #[test]
    fn punctuation_is_the_question_python_asks_of_string_punctuation() {
        // A single mark, which is the case that matters.
        for mark in [".", ",", "?", "!", "\"", "-", "("] {
            assert!(PUNCTUATION.contains(mark), "{mark}");
        }
        // A run that IS contiguous in string.punctuation, which Python calls
        // punctuation and so does this.
        assert!(PUNCTUATION.contains("()"));
        // A run that is not.
        assert!(!PUNCTUATION.contains(".."));
        assert!(!PUNCTUATION.contains("hello"));
        // The empty string, which Python also calls punctuation; a piece that
        // strips to nothing is whitespace, and whitespace opens a word anyway.
        assert!(PUNCTUATION.contains(""));
    }
}
