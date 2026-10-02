-- The clip with a subtitle track of what is said in it, over a voice detector that says where the speech is so the model spends nothing on the score.
-- variables: source (input media path), track (audio track index, defaults to the first), dest (output path), language (what the dialogue is in, e.g. es; leave unset to let the model detect it per window), task (transcribe or translate, defaults to transcribe; translation is always into English), language_out (what the cues come out in, which is what tags the track: en when translating)
-- example: ffrwd compile -f packages/ffrwd/whisper/recipes/subtitles.sql -v source=film.mkv -v dest=subbed.mkv -v language=es
COPY (
  SELECT f.video[1], a,
         ffrwd.whisper.transcribe(a,
                                  speech => ffrwd.vad.speech(a),
                                  language => :'language',
                                  task => COALESCE(:'task', 'transcribe'),
                                  language_out => :'language_out')
  FROM input(:'source') f, unnest(f.audio) a
  WHERE a.index = COALESCE(:track, 1)
) TO :'dest'
