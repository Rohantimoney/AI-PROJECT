# It will handle the connection to Firebase and all database operations.

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json
from langchain_core.messages import HumanMessage, AIMessage

# --- Firebase Initialization ---
@st.cache_resource
def initialize_firebase():
    """Initializes the Firebase Admin SDK using Streamlit secrets."""
    try:
        # Check if the app is already initialized
        if not firebase_admin._apps:
            # The service account info is stored as a string in secrets.toml, so we parse it
            service_account_info = json.loads(st.secrets["firebase_service_account"])
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        return None

# --- Firestore Functions ---

def get_user_interviews(db, username):
    """Fetches a list of interview IDs for a given user."""
    if not db or not username:
        return []
    try:
        interviews_ref = db.collection('users').document(username).collection('interviews').stream()
        return [interview.id for interview in interviews_ref]
    except Exception as e:
        st.error(f"Error fetching interviews: {e}")
        return []

def save_chat_history(db, username, interview_id, chat_history):
    """Saves the entire chat history to Firestore."""
    if not all([db, username, interview_id]):
        return
    try:
        interview_ref = db.collection('users').document(username).collection('interviews').document(interview_id)
        # Convert LangChain messages to a serializable format
        history_to_save = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                history_to_save.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_to_save.append({"type": "ai", "content": msg.content})
        
        # We store the history as a list of message maps in a single document
        interview_ref.set({"messages": history_to_save})
    except Exception as e:
        st.error(f"Error saving chat history: {e}")


def load_chat_history(db, username, interview_id):
    """Loads a specific chat history from Firestore."""
    if not all([db, username, interview_id]):
        return []
    try:
        interview_ref = db.collection('users').document(username).collection('interviews').document(interview_id)
        doc = interview_ref.get()
        if doc.exists:
            messages_data = doc.to_dict().get("messages", [])
            chat_history = []
            for msg_data in messages_data:
                if msg_data["type"] == "human":
                    chat_history.append(HumanMessage(content=msg_data["content"]))
                elif msg_data["type"] == "ai":
                    chat_history.append(AIMessage(content=msg_data["content"]))
            return chat_history
        return []
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []