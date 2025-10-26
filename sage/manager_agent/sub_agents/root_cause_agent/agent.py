from google.adk.agents import Agent

# Create the root cause agent
root_cause_agent = Agent(
    name="root_cause_agent",
    model="gemini-2.0-flash",
    description="Identifies the root cause of the user's issue from the transcript.",
    instruction="""
    You are the root cause agent. Your role is to analyze the transcript and identify the root cause of the user's issue.
    """,
    tools=[],
)