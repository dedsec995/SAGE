from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
import google.generativeai as genai
import math
import os
import json
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemma-3-27b-it')

def analyze_sentiment_per_minute(tool_context: ToolContext) -> dict:
    """
    Analyzes the emotional tone and satisfaction level of the transcript per minute and saves it to the state.

    Args:
        tool_context (ToolContext): The tool context containing the transcript.

    Returns:
        dict: A dictionary containing the sentiment details per minute.
    """
    transcript = tool_context.state.get("transcript")
    if not transcript:
        return {"error": "Transcript not found in state."}

    sentiment_details_per_minute = []
    max_time = max(segment[1] for segment in transcript)
    num_minutes = math.ceil(max_time / 60)

    for minute in range(num_minutes):
        start_time = minute * 60
        end_time = (minute + 1) * 60
        minute_transcript = [segment for segment in transcript if segment[0] >= start_time and segment[0] < end_time]

        if not minute_transcript:
            sentiment_details_per_minute.append({"emotional_tone": "Neutral", "satisfaction_level": "Neutral"})
            continue

        full_text = " ".join(segment[3] for segment in minute_transcript)
        
        prompt = f"""Analyze the emotional tone and satisfaction level of the following text from a customer service call. 
        Respond with a JSON object with two keys: 'emotional_tone' and 'satisfaction_level'.
        
        Possible values for 'emotional_tone': Angry, Frustrated, Happy, Calm, Anxious, Neutral.
        Possible values for 'satisfaction_level': Very Satisfied, Satisfied, Neutral, Dissatisfied, Very Dissatisfied.
        
        Text: {full_text}
        """
        
        response = model.generate_content(prompt)
        try:
            sentiment_details = json.loads(response.text.strip())
        except json.JSONDecodeError:
            sentiment_details = {"emotional_tone": "Neutral", "satisfaction_level": "Neutral"}

        sentiment_details_per_minute.append(sentiment_details)

    tool_context.state["sentiment_state"] = sentiment_details_per_minute
    return {"sentiment": sentiment_details_per_minute}

sentiment_agent = Agent(
    name="sentiment_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Analyzes the emotional tone and satisfaction levels in the transcript.",
    instruction="""
    You are a sentiment analysis expert specializing in customer service calls.
    Your task is to analyze the provided call transcript to identify the emotional tone and satisfaction levels for each minute of the call.
    You have access to the 'analyze_sentiment_per_minute' tool. Call this tool to perform the analysis and save the results to the state.
    
    """,
    tools=[analyze_sentiment_per_minute],
)