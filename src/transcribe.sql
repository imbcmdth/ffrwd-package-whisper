-- The one model export, hosted as the wasm module the package ships. The
-- weights are pinned in the manifest and land beside the module at install.
--
-- `transcribe` hands the audio back untouched with one cue per stretch of
-- speech beside it: `text` is what was said and `start_t`/`end_t` are the
-- seconds it runs between. The audio is read 30 seconds at a time, which is
-- the model's own input, and each window is decoded on its own.
--
-- `speech` is where an upstream voice detector's spans arrive. It is optional:
-- with it, a window the detector already vouched for skips the pass that would
-- otherwise have to establish there is speech in it. Without it, that pass
-- runs on every window, and it is also what keeps the model from putting words
-- to a music bed.
--
-- `language` is what the dialogue is in. Left unset the model detects it per
-- window, which is what to do when the clip changes language or when nobody
-- knows. `task` is 'transcribe' or 'translate'; translation is always into
-- English, since that is the only direction whisper knows. `language_out` says
-- what the cues themselves are in, and is what tags a track minted from them:
-- 'en' when translating, and otherwise the language that was heard.
CREATE FUNCTION transcribe(a audio_stream,
                           speech cue[] DEFAULT NULL,
                           language text DEFAULT NULL,
                           task text DEFAULT 'transcribe',
                           language_out text DEFAULT NULL)
RETURNS STRUCT(a audio_stream, words cue[])
  AS 'target/wasm32-wasip2/release/transcribe.wasm', 'transcribe' LANGUAGE wasm;
