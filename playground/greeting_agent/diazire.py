import base64
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

def to_data_url(path: str) -> str:
  with open(path, "rb") as fh:
    return "data:audio/wav;base64," + base64.b64encode(fh.read()).decode("utf-8")

with open("/home/dedsec995/Downloads/jeff.wav", "rb") as audio_file:
  transcript = client.audio.transcriptions.create(
    model="gpt-4o-transcribe-diarize",
    file=audio_file,
    response_format="diarized_json",
    chunking_strategy="auto",
    extra_body={
      "known_speaker_names": ["agent"],
      "known_speaker_references": [to_data_url("/home/dedsec995/Downloads/jeff.wav")],
    },
  )

print(transcript)
print()
print()
print(transcript.segments)
