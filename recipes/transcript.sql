-- Everything said in a clip, written as ndjson: one line per stretch of speech, with the words and the seconds they run between.
-- variables: source (input media path), track (audio track index, defaults to the first), dest (output path, e.g. transcript.ndjson), language (what the dialogue is in, e.g. es; leave unset to let the model detect it per window), task (transcribe or translate, defaults to transcribe; translation is always into English)
-- example: ffrwd compile -f packages/ffrwd/whisper/recipes/transcript.sql -v source=interview.mp4 -v dest=transcript.ndjson
COPY (
  SELECT ffrwd.whisper.transcribe(a, :'language', COALESCE(:'task', 'transcribe')).words
  FROM input(:'source') f, unnest(f.audio) a
  WHERE a.index = COALESCE(:track, 1)
) TO :'dest'
