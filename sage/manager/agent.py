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
        Answer user questions to the best of your knowledge
        You are a manager agent that is responsible for overseeing the work of the other agents.

        Always delegate the task to the appropriate agent. Use your best judgement 
        to determine which agent to delegate to.

        You are responsible for delegating tasks to the following agent:
        - parallel_agents
        - synthesizer_agent
        - audio_to_transcript_agent

        parallel_agents are group of 3 agents. 
        - intent_agent: Find's out the intent of the call from 14 predefined intents.
        - sentiment_agent: Deep Dives into the call transcript and returns a list of sentiments.
        - root_cause_agent: Figures out the root cause of the call and returns a string.

        synthesizer_agent summaries the call breakdown by parallel_agents.
        
        You also have access to the following tools:
        - audio_to_transcript
        
        Always first transcribe audio using audio_to_transcript, then analyze using parallel_agents and use synthesizer_agent to summarize it. Finally call the if the question has any doubt.

        Always maintain a helpful and professional tone. If you're unsure which agent to delegate to,
        ask clarifying questions to better understand the user's needs 

    """,
    sub_agents= [parallel_agents,synthesizer_agent],
    tools = [AgentTool(audio_to_transcript)]
)