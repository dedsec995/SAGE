from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def transcribe_audio(tool_context: ToolContext, audio_filepath: str) -> dict:
    """ 
    Transcribes the audio file at the given filepath and returns the transcript with diarization.
    The transcript format is [[start_time, end_time, speaker_name, actual_text]].
    """
    # In the future, this tool will call the audio transcription service.
    # For now, it returns a dummy transcript.
    print(f"Transcribing audio file at: {audio_filepath}")
    dummy_transcript = [
        [0.5, 2.5, "Speaker 1", "Hello, I'm calling to check the status of my loan application."],
        [2.8, 4.2, "Speaker 2", "Certainly, I can help with that. Can I have your application ID?"],
    ]
    tool_context.state["is_audio_transcribed"] = True
    tool_context.state['transcript'] = dummy_transcript
    return {"transcript": dummy_transcript}

def transcribe_audio(tool_context: ToolContext, audio_filepath: str) -> dict:
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
        with open(audio_filepath, "rb") as audio_file:
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
        return {"error": f"Audio file not found at path: {audio_filepath}"}
    except Exception as e:
        return {"error": f"An error occurred during transcription: {e}"}


# Create the audio to transcript agent
audio_to_transcript_agent = Agent(
    name="audio_to_transcript_agent",
    model="gemini-2.0-flash",
    description="Transcribes an audio file and returns the transcript with diarization.",
    instruction="""
    You are the audio to transcript agent. Your role is to take a filepath to an audio file,
    transcribe it, and return the transcript with diarization.
    """,
    tools=[transcribe_audio],
)