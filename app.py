import streamlit as st
import asyncio
import os
from sage.manager_agent.agent import manager_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from sage.utils import call_agent_async

# Load environment variables
load_dotenv()

# --- Configuration ---
APP_NAME = "Bank Audio Transcript Analyst"
USER_ID = "dedsec995"
DB_URL = "sqlite:///./my_agent_data.db"
UPLOAD_DIR = "uploaded_audio"

# --- Initialization ---
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

st.set_page_config(page_title=APP_NAME, layout="wide")

# Initialize session service
session_service = DatabaseSessionService(db_url=DB_URL)

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    st.title(APP_NAME)

    # Page routing
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "chat":
        chat_page()

def home_page():
    """Home page to display sessions and upload new audio."""
    st.header("Analysis Sessions")

    # --- Upload new audio ---
    with st.expander("Upload New Audio File"):
        uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            if st.button("Start Analysis"):
                filepath = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Create a new session
                initial_state = {
                    "user_name": "Amit Luhar",
                    "intent_state": None,
                    "sentiment_state": None,
                    "root_cause_state": None,
                    "is_audio_transcribed": False,
                    "transcript": [],
                    "analysis_report": None,
                    "interaction_history": [],
                    "audio_filepath": filepath
                }
                
                new_session = asyncio.run(session_service.create_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    state=initial_state,
                ))
                
                st.session_state.session_id = new_session.id
                

    # --- Display existing sessions ---
    st.subheader("Previous Analyses")
    
    sessions = asyncio.run(session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID))
    
    if not sessions.sessions:
        st.info("No previous analysis sessions found.")
    else:
        for session in sessions.sessions:
            filepath = session.state.get("audio_filepath", "Unknown File")
            report = session.state.get("analysis_report", "Not yet analyzed")

            with st.container():
                st.markdown(f"**File:** `{os.path.basename(filepath)}`")
                st.markdown(f"**Session ID:** `{session.id}`")
                
                if report != "Not yet analyzed":
                    st.success("Analysis Complete")
                else:
                    st.warning("Analysis Pending")

                if st.button("Open Chat", key=session.id):
                    st.session_state.session_id = session.id
                    st.session_state.page = "chat"
                    
                st.divider()

def chat_page():
    """Chat page for a specific session."""
    st.header("Chat and Analysis")

    if st.button("<- Back to Home"):
        st.session_state.page = "home"
        del st.session_state.session_id
        st.rerun()
        

    session_id = st.session_state.session_id
    session = asyncio.run(session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id))

    # --- Agent Runner ---
    runner = Runner(
        agent=manager_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # --- Initial Analysis ---
    if not session.state.get("analysis_report"):
        with st.spinner("Performing initial analysis... This may take a moment."):
            # The manager agent will automatically run the workflow because the filepath is set
            asyncio.run(call_agent_async(runner, USER_ID, session_id, "analyze the audio"))
        st.success("Initial analysis complete!")
        session = asyncio.run(session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id))

    # --- Display Analysis Report ---
    st.subheader("Analysis Report")
    report = session.state.get("analysis_report", "Report not available.")
    st.markdown(report)

    # --- Chat Interface ---
    st.subheader("Follow-up Questions")
    
    # Display chat history
    for message in session.state.get("interaction_history", []):
        if message.get("action") == "user_query":
            with st.chat_message("user"):
                st.markdown(message["query"])
        elif message.get("action") == "agent_response":
            with st.chat_message("assistant"):
                st.markdown(message["response"])

    # Chat input
    if prompt := st.chat_input("Ask a follow-up question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            response = asyncio.run(call_agent_async(runner, USER_ID, session_id, prompt))
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        

if __name__ == "__main__":
    main()
