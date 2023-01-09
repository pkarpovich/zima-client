import io
import threading

import speech_recognition as sr
import whisper
import tempfile

r = sr.Recognizer()
r.dynamic_energy_threshold = True
r.dynamic_energy_ratio = 3.0

model = whisper.load_model("small")
language = "russian"

mutex = threading.Lock()


def listen():
    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        while True:
            audio = r.record(source, duration=4)
            data = io.BytesIO(audio.get_wav_data())

            transcribe_thread = threading.Thread(target=transcribe, args=(data,))
            transcribe_thread.daemon = True
            transcribe_thread.start()


def transcribe(data):
    try:
        mutex.acquire()

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(data.read())
            f.flush()

            audio = whisper.load_audio(f.name)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            options = whisper.DecodingOptions(fp16=False, language=language)
            result = whisper.decode(model, mel, options)

            print(result.text)
    finally:
        mutex.release()


if __name__ == "__main__":
    listen()
