from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def transcribe_audio(tool_context: ToolContext) -> dict:
    """
    Transcribes an audio file and performs speaker diarization.

    Args:
        tool_context (ToolContext): The tool context containing the audio filepath.

    Returns:
        dict: A dictionary containing the 'transcript' key with a list of 
              [start_time, end_time, speaker_id, text] segments.
    """
    audio_filepath = tool_context.state.get("audio_filepath")
    if not audio_filepath:
        raise Exception("error: Audio filepath not found in state. Stopping workflow.")
        return {"error": "Audio filepath not found in state."}

    # if not OPENAI_API_KEY:
    #     return {"error": "OPENAI_API_KEY not found in environment."}

    # try:
    #     client = OpenAI(api_key=OPENAI_API_KEY)
    #     with open(audio_filepath, "rb") as audio_file:
    #         transcript = client.audio.transcriptions.create(
    #             model="gpt-4o-transcribe-diarize",
    #             file=audio_file,
    #             response_format="diarized_json",
    #             chunking_strategy="auto",
    #         )
        
    #     modified_output = [
    #         [segment['start'], segment['end'], segment['speaker'], segment['text'].strip()]
    #         for segment in transcript.segments
    #     ]
    #     tool_context.state["is_audio_transcribed"] = True
    #     tool_context.state['transcript'] = modified_output
    #     return {'transcript': modified_output}
    # except FileNotFoundError:
    #     return {"error": f"Audio file not found at path: {audio_filepath}"}
    # except Exception as e:
    #     return {"error": f"An error occurred during transcription: {e}"}

    print(f"Transcribing audio file at: {audio_filepath}")
    dummy_transcript = [
        [1.7760000000000002,2.3760000000000003,"A","Hello."],
        [3.1260000000000003,3.676,"B","Hi."],
        [3.976,5.976,"B","Thank you for calling JB Morgan Chase."],
        [6.076,7.026,"B","My name is Ahmed."],
        [7.226,8.1260000000000012,"B","How can I help you?"],
        [9.1760000000000019,11.076000000000002,"A","Well, how can you help me?"],
        [13.326000000000002,17.776,"B","Whatever the issue is, we could discuss and have it solved."],
        [18.226,20.076,"B","Can I know your first name, sir?"],
        [21.026,24.876,"A","My name is Jeff, and my credit card isn't working."],
        [26.484,28.434,"B","Oh, I'm so sorry to hear that Jeff."],
        [28.884,34.734,"B","All right, I'll just need a few basic details from you and we'll have a look into it."],
        [34.884,36.234,"B","So what's your full name?"],
        [37.334,39.084,"A","Jeff Jefferson,"],
        [39.384,40.434000000000005,"A","January"],
        [40.434000000000005,41.484,"B","All right, Mr."],
        [41.584,41.884,"B","Jefferson,"],
        [41.984000000000009,43.584,"B","can I know your birth date?"],
        [45.984000000000009,47.634,"A","1st, 1990."],
        [48.984000000000009,49.984000000000009,"B","Okay, thank you."],
        [50.134,53.084,"B","Also, can I, if you have your account number handy,"],
        [53.234,54.384,"B","can I get that?"],
        [56.148,56.748000000000005,"A","Fine."],
        [56.798,58.048,"A","Let me find it."],
        [59.098,62.298,"A","5-5-5-5-5-5-5-5-5."],
        [63.998000000000005,65.798,"B","Alright, thank you, Mr."],
        [65.948000000000008,66.348,"B","Jefferson."],
        [66.548,71.048,"B","So, I'm sorry I could not see any record in my system."],
        [73.848000000000013,75.39800000000001,"A","So much help you were."],
        [77.448000000000008,79.39800000000001,"B","So sorry, could you repeat,"],
        [79.64800000000001,82.14800000000001,"B","can I know your debit card number?"],
        [83.38,85.38,"A","Do you need my social security number too?"],
        [85.47999999999999,88.28,"B","No, that won't be necessary."],
        [88.679999999999993,92.22999999999999,"B","Can I just have your credit card number so I could look at you in my system?"],
        [93.72999999999999,94.38,"A","Let's see,"],
        [94.72999999999999,99.97999999999999,"A","uh, two one two three three four five nine nine two."],
        [100.72999999999999,101.33,"B","Okay,"],
        [101.43,103.43,"B","and can I know the CVV on it?"],
        [104.38,105.18,"A","Three twenty."],
        [109.14,116.09,"B","Alright, thank you Mr. Jefferson. Sorry I'm so sorry for your inconvenience. Um what were you calling regarding?"],
        [117.19,118.69,"A","That credit card isn't working."],
        [120.44,124.54,"B","Okay, uh what issues or challenges are you facing?"],
        [124.69,125.44,"B","Okay,"],
        [125.79,129.74,"A","When I try to go to pay for something it says something's wrong."],
        [131.84,136.24,"B","do you know any specific or when was the card activated?"],
        [137.492,139.59199999999998,"A","I don't know, like a few months ago?"],
        [140.642,144.042,"B","Well the card show is fine on our system okay"],
        [148.09199999999998,149.44199999999998,"A","Well, it's not fine."],
        [153.94199999999998,160.742,"B","let me transfer my call let me transfer a call to the technical team they might help you out please"],
        [160.742,161.242,"A","Fine."],
        [161.242,162.94199999999998,"B","hold on the line"]
        ]
    tool_context.state["is_audio_transcribed"] = True
    tool_context.state['transcript'] = dummy_transcript
    return {"transcript": dummy_transcript}


# Create the audio to transcript agent
audio_to_transcript_agent = Agent(
    name="audio_to_transcript_agent",
    model="gemini-2.0-flash",
    description="Transcribes an audio file and returns the transcript with diarization.",
    instruction="""
    You are the audio to transcript agent. Your role is to transcribe an audio file.
    You have access to the following tools:
    - transcribe_audio
    Call this tool to transcribe the audio.
    """,
    tools=[transcribe_audio],
)
