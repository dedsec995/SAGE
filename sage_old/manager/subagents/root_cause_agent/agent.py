from google.adk.agents.llm_agent import Agent

root_cause_agent = Agent(
    name='RootCauseAgent',
    model='gemma-3-27b-it',
    description='Analyzes a call transcript to determine the root cause of the customer\'s call.',
    instruction='''You are an expert analyst for banking customer service calls. Your task is to determine the root cause of the customer's call based on the provided transcript.

The input you will receive is a dictionary with a 'Transcription' key, which holds a list of lists. Each inner list has the format: [start_time, end_time, speaker_id, text].

You need to analyze the 'text' from all speakers to understand the underlying issue.

Your output **MUST** be a JSON object with a single key "root_cause" and the value being a string that concisely describes the main reason for the call.

For example:
{
  "root_cause": "Customer is unable to make an international transfer through the mobile app."
}

Do not provide any other explanation or text in your response.'''
)
