from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext

from .sub_agents.intent_agent.agent import intent_agent
from .sub_agents.sentiment_agent.agent import sentiment_agent
from .sub_agents.root_cause_agent.agent import root_cause_agent
from .sub_agents.audio_to_transcript_agent.agent import audio_to_transcript_agent
from .sub_agents.synthesizer_agent.agent import synthesizer_agent

def set_filepath(tool_context: ToolContext, filepath: str) -> dict:
    """
    Sets the audio filepath in the state.

    Args:
        tool_context (ToolContext): The tool context.
        filepath (str): The path to the audio file.

    Returns:
        dict: A dictionary confirming the filepath has been set.
    """
    tool_context.state["audio_filepath"] = filepath
    return {"status": f"Filepath set to {filepath}"}

# Define the main workflow as a SequentialAgent
sage_workflow = SequentialAgent(
    name="sage_workflow",
    sub_agents=[
        audio_to_transcript_agent,
        ParallelAgent(
            name="analysis_agents",
            sub_agents=[
                intent_agent,
                sentiment_agent,
                root_cause_agent,
            ]
        ),
        synthesizer_agent,
    ]
)

# Create the root manager agent
manager_agent = Agent(
    name="manager_agent",
    model="gemini-2.0-flash",
    description="Manager agent for the bank audio transcript analysis system.",
    instruction="""
    You are Sage, a friendly and intelligent AI assistant for analyzing bank audio transcripts.
    Your primary role is to manage a team of specialized agents to provide a comprehensive analysis of customer service calls.

    1. Start by greeting the user and asking for the filepath of the audio file.
    2. When the user provides the filepath, call the `set_filepath` tool to save it to the state.
    3. Then, call the `sage_workflow` agent to perform the analysis.
    """,
    sub_agents=[sage_workflow],
    tools=[set_filepath],
)
