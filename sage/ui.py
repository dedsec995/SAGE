import streamlit as st
import asyncio
import os
import uuid
from datetime import datetime
from manager_agent.agent import manager_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.events import Event
from google.genai import types
# Load environment variables
load_dotenv()

# --- Application Constants ---
APP_NAME = "Bank Audio Transcript Analyst"
USER_ID = "dedsec995"
DB_URL = "sqlite:///./my_agent_data.db"
UPLOAD_DIR = "sage/uploaded_audio"

initial_state = {
    "user_name": "Amit Luhar",
    "intent_state": None,
    "sentiment_state": None,
    "root_cause_state": None,
    "is_audio_transcribed": False,
    "audio_filepath": None,
    "transcript": [],
    "analysis_report": None,
    "interaction_history": [],
}

async def call_agent_async_ui(runner, session_id, query, chat_placeholder):
    """Call the agent asynchronously and display the response in the UI."""
    # 1. Get the session and append the user query as an event first.
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=USER_ID, session_id=session_id
    )
    session.state["interaction_history"].append({
        "action": "user_query",
        "query": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    user_query_event = Event(
        author="user",
        content=types.Content(role="user", parts=[types.Part(text=query)]),
    )
    await runner.session_service.append_event(session=session, event=user_query_event)

    # 2. Run the agent.
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = ""
    agent_name = ""
    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            if event.author:
                agent_name = event.author
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()
                chat_placeholder.markdown(final_response_text)

    except Exception as e:
        st.error(f"An error occurred during agent execution: {e}")
        return None

    # 3. Re-fetch the session and append the agent response.
    if final_response_text and agent_name:
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=USER_ID, session_id=session_id
        )
        session.state["interaction_history"].append({
            "action": "agent_response",
            "agent": agent_name,
            "response": final_response_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        agent_response_event = Event(
            author=agent_name,
            content=types.Content(
                role="model", parts=[types.Part(text=final_response_text)]
            ),
        )
        await runner.session_service.append_event(
            session=session, event=agent_response_event
        )
    return final_response_text

def home_page():
    """Renders the home page for uploading audio files."""
    st.title("SAGE - Smart Audio Guardian for Enterprises")
    st.markdown("Welcome to SAGE, your intelligent assistant for analyzing customer service calls. Upload an audio file to begin.")
    col1, col2 = st.columns([2, 1])

    with st.container():
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file (.wav, .mp3, .m4a)",
            type=["wav", "mp3", "m4a"]
        )
        if uploaded_file is not None:
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            if st.button("Analyze File"):
                st.session_state.page = "analysis"
                st.session_state.audio_path = file_path
                st.session_state.session_id = str(uuid.uuid4()) # New session for each file
                st.rerun()

def analysis_page():
    st.title("Analysis Report")
    audio_path = st.session_state.get("audio_path")
    session_id = st.session_state.get("session_id")

    if not audio_path or not session_id:
        st.warning("Please upload an audio file on the Home page first.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    # Initialize session service and runner
    session_service = DatabaseSessionService(db_url=DB_URL)
    runner = Runner(
        agent=manager_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Perform initial analysis if not already done
    if "analysis_done" not in st.session_state:
        async def run_analysis():
            # Create a new session with the audio path
            session_state = initial_state.copy()
            session_state["audio_filepath"] = audio_path
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=session_state,
            )

            with st.chat_message("assistant"):
                with st.spinner("Analyzing audio... This may take a few minutes."):
                    placeholder = st.empty()
                    analysis_report = await call_agent_async_ui(
                        runner, session_id, "Analyze the audio file", placeholder
                    )
            if analysis_report:
                st.session_state.messages.append({"role": "assistant", "content": analysis_report})
                st.session_state.analysis_done = True
        asyncio.run(run_analysis())
        st.rerun()
    # Handle follow-up questions
    if prompt := st.chat_input("Ask a follow-up question about the analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                placeholder = st.empty()
                response = asyncio.run(call_agent_async_ui(runner, session_id, prompt, placeholder))
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("Analyze Another File"):
        # Clear session state for next analysis
        for key in list(st.session_state.keys()):
            if key not in ['page']:
                del st.session_state[key]
        st.session_state.page = "home"
        st.rerun()

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="SAGE", layout="wide")
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "analysis":
        analysis_page()

if __name__ == "__main__":
    main()
