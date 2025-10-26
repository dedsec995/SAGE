from google.adk.agents import Agent

# Create the intent agent
intent_agent = Agent(
    name="intent_agent",
    model="gemini-2.0-flash",
    description="Identifies the user's intent from the transcript.",
    instruction="""
    You are the intent agent. Your role is to analyze the transcript and identify the user's intent.
    """,
    tools=[],
)