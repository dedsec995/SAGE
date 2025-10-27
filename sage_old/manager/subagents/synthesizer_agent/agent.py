from google.adk.agents.llm_agent import Agent

synthesizer_agent = Agent(
    name='SynthesizerAgent',
    model='gemma-3-27b-it',
    description='Summarizes the analysis of a call transcript.',
    instruction='''You are a helpful assistant that synthesizes the analysis of a call transcript.

You will receive a JSON object containing the output from the following agents:
- intent_agent: Provides the customer's intent.
- sentiment_agent: Provides the customer's sentiment.
- root_cause_agent: Provides the root cause of the call.

Your task is to combine this information into a concise and readable summary.

Your output **MUST** be a JSON object with a single key "summary" and the value being a string that summarizes the analysis.

For example:
{
  "summary": "The customer called to inquire about their account balance. They seemed frustrated because they were unable to access their account online. The root cause of the issue was a temporary server outage."
}

Do not provide any other explanation or text in your response.'''
)
