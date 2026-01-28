# https://github.com/nvidia-riva/tutorials/blob/main/asr-speaker-diarization.ipynb
import io
import IPython.display as ipd
import riva.client
from datetime import datetime
import json

start = datetime.now()
auth = riva.client.Auth(uri='localhost:50051')
riva_asr = riva.client.ASRService(auth)
path = "./test.wav"

# Open file
with io.open(path, 'rb') as fh:
    content = fh.read()
ipd.Audio(path)

# Set up an offline/batch recognition request
config = riva.client.RecognitionConfig()
#req.config.encoding = ra.AudioEncoding.LINEAR_PCM    # Audio encoding can be detected from wav
#req.config.sample_rate_hertz = 0                     # Sample rate can be detected from wav and resampled if needed
config.language_code = "en-US"                    # Language code of the audio clip
config.max_alternatives = 1                       # How many top-N hypotheses to return
config.enable_automatic_punctuation = True        # Add punctuation when end of VAD detected
config.audio_channel_count = 1  # Mono channel
config.enable_word_time_offsets=True                  
riva.client.asr.add_speaker_diarization_to_config(config, diarization_enable=True)
response = riva_asr.offline_recognize(content, config)

# Turn into actual json array (riva object isn't json and breaks when saving to a file)
transcript = []
for result in response.results:
    audioBlock = result.alternatives[0]
    words = []
    if audioBlock.words:
        for wordInfo in audioBlock.words:
            record = {
                "word": wordInfo.word,
                "speaker": wordInfo.speaker_tag
                # "start_time":wordInfo.start_time,
                # "end_time":wordInfo.end_time
            }
            words.append(record)
    block = {
        # "transcript":audioBlock.transcript,
        "words": words
    }
    transcript.append(block)

# Save to file
with open(f"output/speaker - {datetime.now().timestamp()}.json", 'w', encoding='utf-8') as f:
    json.dump(transcript, f, ensure_ascii=False, indent=4)

# Show how long the process took 
end = datetime.now()
print(f"Completed: {path} in {end - start}")

