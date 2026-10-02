# ffrwd/whisper

Speech into text pipeline. `transcribe` runs whisper over an audio
stream and writes one cue per stretch of speech with the words in it.
Written beside the clip, the cues are a subtitle track; written alone,
a transcript.

Requires ffrwd 0.29.

```pgsql
COPY (
  SELECT f.video[1], a,
         ffrwd.whisper.transcribe(a, speech => ffrwd.vad.speech(a), language => 'es')
  FROM input('film.mkv') f, unnest(f.audio) a
  WHERE a.index = 1
) TO 'subbed.mkv'
```

The audio is read thirty seconds at a time, the model's own input, so a
cue leaves when the window it was heard in has been decoded. The sound
is not handed back: the query takes it from the source, as above.

`speech` is where a voice detector's rows arrive, `ffrwd/vad`'s
`speech` among them. Every row the detector writes for a window reaches
that window, so a window with none is not decoded at all, and one with
some needs no pass to establish that somebody is talking.

`language` is what the dialogue is in; left unset, the model tries to detect
it per window.

`task` is `transcribe` or `translate`, and translation is always into English, the one direction whisper knows.

`language_out` tags the track minted from the cues: `en` when translating, otherwise the language heard.

The weights are pinned in the manifest and land beside the module at
install. The model does not run on DirectML so picks CUDA or the CPU by itself.

## One cue per word

`transcribe_words` takes the same arguments and returns the same
rows, but each cue is a single word with the seconds it runs between.
Use it when something downstream acts on the times, such as a mask that
bleeps only the words it was given: a phrase cue can be thirty seconds
long, so one word in it silences the half minute around it.

```pgsql
SELECT ffrwd.whisper.transcribe_words(a, speech => ffrwd.vad.speech(a))
FROM input('film.mkv') f, unnest(f.audio) a WHERE a.index = 1
```

The times are not the timestamps the model writes into its answer. They
come from the decoder's cross-attention, warped against the encoder's
20 ms frames, which is how each word gets an edge of its own. Two
consequences:

- **It needs CUDA.** Collecting the cross-attention is a GPU-only
  decoder op, so the graph loads anywhere and runs only there.
  `transcribe` is the export that runs anywhere.
- **It decodes greedily.** onnxruntime does not re-order the collected
  attention when a beam search swaps its hypotheses, so this export
  uses one beam where `transcribe` uses five. The text can differ
  slightly on the same audio. Read words from this one and prose from
  the other.

`speech` matters more here than it does for `transcribe`. The warp has
to open on a window's first frame, so without a detector a window's
first word absorbs the silence ahead of it and can be reported seconds
early. Given the detector's rows, its start is pulled onto the onset
inside it.

A cue's text is what the model wrote, punctuation included: a comma
rides the word before it and an opening quote the word after, so a word
comes through as `damn,` or `"damn`. With `strip => true` it hands back
the bare word instead, punctuation taken off both ends and a word that
was only punctuation dropped, which is what a mask matching typed words
wants:

```pgsql
SELECT ffrwd.whisper.transcribe_words(a, speech => ffrwd.vad.speech(a), strip => true)
FROM input('film.mkv') f, unnest(f.audio) a WHERE a.index = 1
```

The two exports pin different files of the same model, so installing
both downloads two gigabytes rather than one.

## Exports

- `transcribe(a audio_stream, speech STRUCT(start_t number)[] DEFAULT
  NULL, language text DEFAULT NULL, task text DEFAULT 'transcribe',
  language_out text DEFAULT NULL)` returns `cue[]`: what was said, a
  cue per stretch of speech.
- `transcribe_words(..., strip boolean DEFAULT false)` takes the same
  arguments plus `strip` and returns `cue[]`, a cue per word. CUDA
  only.

## Recipes

- `subtitles` - the clip with a subtitle track of what is said, over
  the voice detector.
- `transcript` - everything said, as ndjson.

```
ffrwd run ffrwd/whisper:subtitles -v source=film.mkv -v dest=subbed.mkv -v language=es
```

## Building

```
cargo build --target wasm32-wasip2 --release
```
