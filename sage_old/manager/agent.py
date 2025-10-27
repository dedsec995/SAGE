from google.adk.agents.llm_agent import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.agents import ParallelAgent, SequentialAgent

from .subagents.intent_agent import intent_agent
from .subagents.sentiment_agent import sentiment_agent
from .subagents.root_cause_agent import root_cause_agent
from .subagents.synthesizer_agent import synthesizer_agent
from .subagents.audio_to_transcript import audio_to_transcript

parallel_agents = ParallelAgent(
    name="system_info_gatherer",
    description='A set of 3 agents intent_agent, sentiment_agent and root_cause_agent. They analyze the call and returns a breakdown of the call transcript',
    sub_agents=[intent_agent, sentiment_agent, root_cause_agent],
)

manager_agent = Agent(
    name='ManagerAgent',
    model='gemma-3-27b-it',
    description='A helpful assistant for user questions.',
    instruction="""
        You are a manager agent responsible for overseeing the work of other agents.
        Your goal is to analyze a call transcript, either by transcribing an audio file or by using a provided transcript.

        HERE'S HOW YOU SHOULD WORK:

        1. **Assess the user's input.**
           - If the user provides a file path, assume it's an audio file that needs to be transcribed.
           - If the user provides a string of text, assume it's a transcript.

        2. **Delegate tasks based on the input.**
           - **If you have an audio file:**
             1. Use the `audio_to_transcript` tool to transcribe the audio file.
             2. Once you have the transcript, pass it to the `parallel_agents` for analysis.
           - **If you have a transcript:**
             1. Pass the transcript directly to the `parallel_agents` for analysis.

        3. **Summarize the analysis.**
           - After the `parallel_agents` have completed their analysis, pass their output to the `synthesizer_agent` to get a summary.

        4. **Present the final result.**
           - The output of the `synthesizer_agent` is the final result. Present it to the user.

        **AGENT DESCRIPTIONS:**

        *   `parallel_agents`: A group of three agents that analyze a call transcript:
            *   `intent_agent`: Determines the intent of the call from a predefined list of 14 intents.
            *   `sentiment_agent`: Analyzes the transcript and returns a list of sentiments.
            *   `root_cause_agent`: Identifies the root cause of the call and returns a string.
        *   `synthesizer_agent`: Summarizes the output of the `parallel_agents`.
        *   `audio_to_transcript`: A tool that transcribes an audio file and returns a transcript.

        Always maintain a helpful and professional tone. If you're unsure how to proceed, ask clarifying questions.
    """,
    sub_agents= [parallel_agents,synthesizer_agent],
    tools = [AgentTool(audio_to_transcript)]
)