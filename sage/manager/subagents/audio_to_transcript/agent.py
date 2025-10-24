import base64
import os
from openai import OpenAI
from google.adk.agents import Agent
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def transcribe_audio(filepath: str) -> dict:
    """
    Transcribes an audio file and performs speaker diarization.

    Args:
        filepath (str): The local path to the audio file (e.g., mp3, wav) 
                        that needs to be transcribed.

    Returns:
        dict: A dictionary containing the 'Transcription' key with a list of 
              [start_time, end_time, speaker_id, text] segments.
    """
     
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not found in environment."}

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(filepath, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",
            )
        
        modified_output = [
            [segment['start'], segment['end'], segment['speaker'], segment['text'].strip()]
            for segment in transcript.segments
        ]

        return {'Transcription': modified_output}
    except FileNotFoundError:
        return {"error": f"Audio file not found at path: {filepath}"}
    except Exception as e:
        return {"error": f"An error occurred during transcription: {e}"}


root_agent = Agent(
    model='gemma-3-27b-it',
    name='audio_transcriber_agent',
    description="A specialized agent for processing and transcribing audio files with speaker diarization.",
    instruction=f"""
        You are an intelligent agent that converts a call audio to transcription using the tool.
        If you do not have filepath, ask from the user. Only then you must use the tool.
        You have access to the following tools:
        - transcribe_audio
        You must always ask for filepath for audio if missing from the user
        The tool takes 'filepath:str' as input and gives a structured JSON output 
        {'Transcription:' 'Transcription_object'}
        Transcription_object = [[start_time,end_time,speaker_name,actual_text]]
        IMPORTANT: Your response **MUST** be valid JSON matching this structure.
        DO NOT include any explanations or additional text outside the JSON response.
        
    """,
    tools=[transcribe_audio],
)