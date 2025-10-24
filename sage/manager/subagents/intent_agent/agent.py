from google.adk.agents.llm_agent import Agent

intent_agent = Agent(
    name="IntentAgent",
    model='gemma-3-27b-it',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
