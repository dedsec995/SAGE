from google.adk.agents.llm_agent import Agent

sentiment_agent = Agent(
    name='SentimentAgent',
    model='gemma-3-27b-it',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
