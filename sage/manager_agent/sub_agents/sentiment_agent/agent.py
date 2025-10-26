from google.adk.agents import Agent

# Create the sentiment agent
sentiment_agent = Agent(
    name="sentiment_agent",
    model="gemini-2.0-flash",
    description="Analyzes the sentiment of the user in the transcript.",
    instruction="""
    You are the sentiment agent. Your role is to analyze the transcript and determine the user's sentiment.
    """,
    tools=[],
)