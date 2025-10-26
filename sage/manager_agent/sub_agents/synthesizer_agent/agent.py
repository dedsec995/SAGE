from google.adk.agents import Agent

# Create the synthesizer agent
synthesizer_agent = Agent(
    name="synthesizer_agent",
    model="gemini-2.0-flash",
    description="Synthesizes the analysis from other agents into a final report.",
    instruction="""
    You are the synthesizer agent. Your role is to take the analysis from the intent, sentiment, and root cause agents and synthesize it into a final report.
    """,
    tools=[],
)