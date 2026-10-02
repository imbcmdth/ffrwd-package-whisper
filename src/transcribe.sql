-- The one model export, hosted as the wasm module the package ships. The
-- weights are pinned in the manifest and land beside the module at install.
--
-- `transcribe` returns one cue per stretch of speech: `text` is what was said
-- and `start_t`/`end_t` are the seconds it runs between. The sound is not
-- handed back; a reader takes it from the source. The audio is read 30
-- seconds at a time, which is the model's own input, so a cue leaves when the
-- window it was heard in is decoded, and each window is decoded on its own.
--
-- `speech` is where an upstream voice detector's rows arrive, given by name:
-- `speech => ffrwd.vad.speech(a)`. It is optional. With it, every row the
-- detector wrote for a window reaches that window, and a window the detector
-- vouched for skips the pass that would otherwise have to establish there is
-- speech in it when the language is named. Every other window takes that
-- pass, and it is also what keeps the model from putting words to a music
-- bed.
--
-- `language` is what the dialogue is in. Left unset the model detects it per
-- window, which is what to do when the clip changes language or when nobody
-- knows. `task` is 'transcribe' or 'translate'; translation is always into
-- English, since that is the only direction whisper knows. `language_out` says
-- what the cues themselves are in, and is what tags a track minted from them:
-- 'en' when translating, and otherwise the language that was heard.
CREATE FUNCTION transcribe(a audio_stream,
                           speech STRUCT(start_t number)[] DEFAULT NULL,
                           language text DEFAULT NULL,
                           task text DEFAULT 'transcribe',
                           language_out text DEFAULT NULL)
RETURNS cue[]
  AS 'target/wasm32-wasip2/release/transcribe.wasm', 'transcribe' LANGUAGE wasm;
