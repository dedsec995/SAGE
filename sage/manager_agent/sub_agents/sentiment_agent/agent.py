from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
import google.generativeai as genai
import math
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash') # TODO: Need to centralize the model names

def analyze_sentiment_per_minute(tool_context: ToolContext) -> dict:
    """
    Analyzes the sentiment of the transcript per minute and saves it to the state.

    Args:
        tool_context (ToolContext): The tool context containing the transcript.

    Returns:
        dict: A dictionary containing the sentiments per minute.
    """
    transcript = tool_context.state.get("transcript")
    if not transcript:
        return {"error": "Transcript not found in state."}

    sentiments_per_minute = []
    max_time = max(segment[1] for segment in transcript)
    num_minutes = math.ceil(max_time / 60)

    for minute in range(num_minutes):
        start_time = minute * 60
        end_time = (minute + 1) * 60
        minute_transcript = [segment for segment in transcript if segment[0] >= start_time and segment[0] < end_time]

        if not minute_transcript:
            sentiments_per_minute.append("Neutral")
            continue

        full_text = " ".join(segment[3] for segment in minute_transcript)
        
        # Another Gemini Model call in mA Tool
        response = model.generate_content(
            f"Analyze the sentiment of the following text from a customer service call. Respond with only one of the following words: Positive, Negative, Disappointed, Angry, Happy, Neutral.\n\nText: {full_text}"
        )
        sentiment = response.text.strip()
        sentiments_per_minute.append(sentiment)

    tool_context.state["sentiment_state"] = sentiments_per_minute
    return {"sentiment_state": sentiments_per_minute}

sentiment_agent = Agent(
    name="sentiment_agent",
    model="gemini-2.0-flash",
    description="Analyzes the sentiment of the user in the transcript.",
    instruction="""
    You are a sentiment analysis expert specializing in customer service calls.
    Your task is to analyze the provided call transcript and identify the sentiments expressed by the speakers for each minute of the call.
    You have access to the 'analyze_sentiment_per_minute' tool. Call this tool to perform the analysis and save the sentiments to the state.
    """,
    tools=[analyze_sentiment_per_minute],
)
