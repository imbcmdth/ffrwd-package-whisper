# ffrwd/whisper

Speech into text pipeline. `transcribe` runs whisper over an audio
stream, hands the audio back untouched, and leaves one cue per stretch
of speech beside it with the words in it. Written beside the clip, the
cues are a subtitle track; written alone, a transcript.

```pgsql
COPY (
  SELECT f.video[1], a,
         ffrwd.whisper.transcribe(ffrwd.vad.speech(a), 'es').words
  FROM input('film.mkv') f, unnest(f.audio) a
  WHERE a.index = 1
) TO 'subbed.mkv'
```

The audio is read thirty seconds at a time. `speech` is where a voice
detector's spans arrive. `ffrwd/vad`'s `speech` already
returns the original audio and the spans together, so its result is the whole
first argument.

`language` is what the dialogue is in; left unset, the model tries to detect
it per window.

`task` is `transcribe` or `translate`, and translation is always into English, the one direction whisper knows.

`language_out` tags the track minted from the cues: `en` when translating, otherwise the language heard.

The weights are pinned in the manifest and land beside the module at
install. The model does not run on DirectML so picks CUDA or the CPU by itself.

## Exports

- `transcribe(a audio_stream, speech cue[] DEFAULT NULL, language text
  DEFAULT NULL, task text DEFAULT 'transcribe', language_out text
  DEFAULT NULL)` returns `STRUCT(a audio_stream, words cue[])`: the
  audio as it came, and what was said.

## Recipes

- `subtitles` - the clip with a subtitle track of what is said, over
  the voice detector.
- `transcript` - everything said, as ndjson.

```
ffrwd run ffrwd/whisper:subtitles -v source=film.mkv -v dest=subbed.mkv -v language=es
```

## Building

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```
