# firebase_helpers.py

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json
import requests
from langchain_core.messages import HumanMessage, AIMessage
import os

# --- Firebase Initialization (Admin SDK for Firestore) ---
@st.cache_resource
def initialize_firebase_admin():
    """
    Initializes the Firebase Admin SDK using a service account file.
    This is a more robust method that avoids formatting errors.
    """
    try:
        if not firebase_admin._apps:
            # Get the path to the key file from the current script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            key_path = os.path.join(script_dir, '..', 'firebase_key.json') # Go up one directory to find the key

            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(
            f"Firebase initialization failed. Please ensure your 'firebase_key.json' "
            f"is in the main 'AI Interviewer' directory. Error: {e}"
        )
        return None

# --- Firebase Authentication (Client SDK for Login/Signup) ---
def get_firebase_api_key():
    """Retrieves the Web API Key directly from Streamlit secrets."""
    # This key is separate and used for client-side authentication
    return st.secrets["firebase_web_api_key"]

def sign_up(email, password):
    """Signs up a new user using Firebase Authentication."""
    api_key = get_firebase_api_key()
    signup_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    try:
        response = requests.post(signup_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_json = e.response.json()
        return {"error": error_json.get("error", {}).get("message", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}

def sign_in(email, password):
    """Signs in an existing user."""
    api_key = get_firebase_api_key()
    signin_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    try:
        response = requests.post(signin_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_json = e.response.json()
        return {"error": error_json.get("error", {}).get("message", "INVALID_LOGIN_CREDENTIALS")}
    except Exception as e:
        return {"error": str(e)}

# --- Firestore Functions (No changes needed here) ---
def get_user_interviews(db, user_id):
    if not db or not user_id: return []
    try:
        interviews_ref = db.collection('users').document(user_id).collection('interviews').stream()
        return [interview.id for interview in interviews_ref]
    except Exception as e:
        st.error(f"Error fetching interviews: {e}")
        return []

def save_chat_history(db, user_id, interview_id, chat_history):
    if not all([db, user_id, interview_id]): return
    try:
        interview_ref = db.collection('users').document(user_id).collection('interviews').document(interview_id)
        history_to_save = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                history_to_save.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_to_save.append({"type": "ai", "content": msg.content})
        interview_ref.set({"messages": history_to_save})
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def load_chat_history(db, user_id, interview_id):
    if not all([db, user_id, interview_id]): return []
    try:
        interview_ref = db.collection('users').document(user_id).collection('interviews').document(interview_id)
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