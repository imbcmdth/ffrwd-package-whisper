-- The same model, read a second way, as its own wasm module. The weights are
-- pinned in the manifest and land beside the module at install; they are a
-- second export of whisper medium and a second gigabyte, not the one
-- `transcribe` uses.
--
-- `transcribe_words` hands the audio back untouched with one cue per WORD
-- beside it: `text` is the single word and `start_t`/`end_t` are the seconds
-- it runs between. That is the whole of the difference from `transcribe`,
-- whose cues are a stretch of speech each - so a mask built from these cues
-- covers the word it named and not the sentence around it. The audio is read
-- 30 seconds at a time, which is the model's own input, and each window is
-- decoded and aligned on its own.
--
-- The times do not come from the timestamps the model writes into its answer.
-- They come from the decoder's cross-attention: where a token attended in the
-- encoder's 20 ms frames is where in the audio it was said, and the tokens are
-- warped against those frames to give every word an edge of its own.
--
-- Two things follow from that, and both are load-bearing.
--
-- It needs the CUDA execution provider. Collecting the cross-attention is a
-- GPU-only decoder op, so this graph loads anywhere and runs only there; asked
-- to run elsewhere it refuses and says so. `transcribe` is the export that
-- runs anywhere.
--
-- It decodes greedily, one beam where `transcribe` searches five, because
-- onnxruntime does not re-order the collected attention when the beam search
-- swaps its hypotheses about - and rows belonging to a discarded sequence
-- triple the word error. So the TEXT here can differ slightly from
-- `transcribe`'s on the same audio. Read words from this one and prose from
-- that one; do not expect them to agree token for token.
--
-- `speech` is where an upstream voice detector's spans arrive. It is optional,
-- and it does more here than it does for `transcribe`: besides sparing a
-- window the pass that establishes there is speech in it, its onsets are what
-- keep a window's FIRST word honest. The warp has to open its path on the
-- first frame, so that word otherwise swallows any silence ahead of it and can
-- be reported seconds early; with spans, its start is pulled forward onto the
-- onset inside it. Without them it keeps the raw start, which is the one place
-- this export is worth distrusting.
--
-- `language` is what the dialogue is in. Left unset the model detects it per
-- window, which is what to do when the clip changes language or when nobody
-- knows. `task` is 'transcribe' or 'translate'; translation is always into
-- English, since that is the only direction whisper knows. `language_out` says
-- what the cues themselves are in, and is what tags a track minted from them:
-- 'en' when translating, and otherwise the language that was heard. The list
-- is `transcribe`'s, argument for argument, so a query can swap one export for
-- the other without being rewritten.
CREATE FUNCTION transcribe_words(a audio_stream,
                                 speech cue[] DEFAULT NULL,
                                 language text DEFAULT NULL,
                                 task text DEFAULT 'transcribe',
                                 language_out text DEFAULT NULL)
RETURNS STRUCT(a audio_stream, words cue[])
  AS 'target/wasm32-wasip2/release/transcribe_words.wasm', 'transcribe_words' LANGUAGE wasm;
