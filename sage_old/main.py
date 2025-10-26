import asyncio

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai.types import Content, Part
from manager.agent import manager_agent

load_dotenv()

# ===== PART 1: Initialize Persistent Session Service =====
# Using SQLite database for persistent storage
db_url = "sqlite:///./my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)


# ===== PART 2: Define Initial State =====
# This will only be used when creating a new session
initial_state = {
    "transcript": None,
    "analysis": None,
}


async def main_async():
    # Setup constants
    APP_NAME = "Bank Audio Analysis"
    USER_ID = "user123"

    # ===== PART 3: Session Management - Find or Create =====
    # Check for existing sessions for this user
    existing_sessions = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    # If there's an existing session, use it, otherwise create a new one
    if existing_sessions and len(existing_sessions.sessions) > 0:
        # Use the most recent session
        SESSION_ID = existing_sessions.sessions[0].id
        print(f"Continuing existing session: {SESSION_ID}")
    else:
        # Create a new session with initial state
        new_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            state=initial_state,
        )
        SESSION_ID = new_session.id
        print(f"Created new session: {SESSION_ID}")

    # ===== PART 4: Agent Runner Setup =====
    # Create a runner with the manager agent
    runner = Runner(
        agent=manager_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ===== PART 5: Interactive Conversation Loop =====
    print("\nWelcome to the Bank Audio Analysis Agent!")
    print("You can provide an audio file path or a transcript.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        # Get user input
        user_input = input("You: ")

        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation. Your data has been saved to the database.")
            break

        # Process the user query through the agent
        content = Content(role="user", parts=[Part(text=user_input)])
        response = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
        async for event in response:
            if event.type == "output":
                print(f"Agent: {event.content.parts[0].text}")


if __name__ == "__main__":
    asyncio.run(main_async())