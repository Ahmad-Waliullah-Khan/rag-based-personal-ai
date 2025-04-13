import whisper
import pyttsx3
from app_logic import query_ai  # function to run full_query + return response

model = whisper.load_model("base")
engine = pyttsx3.init()

print("🎙️ Speak into your mic...")

import sounddevice as sd
import soundfile as sf

duration = 5
samplerate = 44100
filename = "input.wav"

mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
sd.wait()
sf.write(filename, mydata, samplerate)

result = model.transcribe(filename)
query = result["text"]
print(f"🗣️ You said: {query}")

response = query_ai(query)
engine.say(response)
engine.runAndWait()
