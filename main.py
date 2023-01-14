import os
import io

import speech_recognition as sr
import whisper
import tempfile
import pvporcupine
from pvrecorder import PvRecorder

AccessKey = os.environ.get('PORCUPINE_ACCESS_KEY')

porcupine = pvporcupine.create(
    access_key=AccessKey,
    keyword_paths=['./keyword_model.ppn']
)

model = whisper.load_model("small")
language = "russian"


def listen():
    try:
        keyword_recorder = PvRecorder(device_index=1, frame_length=porcupine.frame_length)

        print("Waiting for keyword...")
        while True:
            keyword_recorder.start()
            pcm = keyword_recorder.read()

            keyword_index = porcupine.process(pcm)
            if keyword_index == 0:
                keyword_recorder.stop()
                print("Listening...")
                listening()

    except KeyboardInterrupt:
        print('Stopping...')


def listening():
    with sr.Microphone(sample_rate=16000) as source:
        r = sr.Recognizer()
        audio = r.listen(source)
        data = io.BytesIO(audio.get_wav_data())

        transcribe(data)


def transcribe(data):
    print("Transcribing...")

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(data.read())
        f.flush()
        transcribe_with_whisper(f)


def transcribe_with_whisper(f):
    audio = whisper.load_audio(f.name)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(fp16=False, language=language)
    result = whisper.decode(model, mel, options)

    print(result.text)


if __name__ == "__main__":
    listen()
