# stream.py

import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
import nest_asyncio

# Apply the patch for asyncio
nest_asyncio.apply()

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# Firebase Imports
from firebase_helpers import (
    initialize_firebase_admin, get_user_interviews, save_chat_history,
    load_chat_history, sign_up, sign_in
)

# --- 1. SETUP AND CONFIGURATION ---
load_dotenv()
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="AI Interviewer", page_icon="🤖")
st.title("🤖 Personal AI Interviewer")

# --- Initialize Firebase Admin SDK for Firestore ---
db = initialize_firebase_admin()

# --- CORE APPLICATION FUNCTIONS (CACHED) ---
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def process_and_store_resume(pdf_file):
    if not pdf_file: return None, None
    with st.spinner("Analyzing your resume..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.getvalue())
            loader = PyPDFLoader(tmpfile.name)
            documents = loader.load()
        os.remove(tmpfile.name)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=chunks, embedding=get_embeddings())
        try:
            first_page_content = documents[0].page_content
            prompt = f"From the following text, extract only the candidate's full name. Respond with nothing but the name.\n\nText: {first_page_content[:1000]}"
            name_response = get_llm().invoke(prompt)
            user_name = name_response.content.strip()
        except Exception:
            user_name = "Candidate"
    return vector_store, user_name

def get_conversational_rag_chain(_vector_store):
    llm = get_llm()
    retriever = _vector_store.as_retriever(search_kwargs={"k": 4})
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    system_prompt = """
    You are an expert AI interviewer and career coach. Your goal is to conduct a professional, dynamic interview based on the provided resume context.
    Perform two tasks in your response:
    1.  **Provide Feedback:** In one short paragraph, give constructive feedback on the user's previous answer. Start with "Feedback:". Skip this for the very first question.
    2.  **Ask the Next Question:** Ask the next relevant interview question. Vary your questions and ask probing follow-ups if answers are brief.
    If the user asks a question about their own resume, answer it first, then ask the next interview question.
    Context: {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, chain)

# --- PAGE DEFINITIONS ---

def show_auth_page():
    """Displays the login and sign-up forms."""
    st.subheader("Welcome!")
    
    auth_option = st.radio("Choose an option:", ('Login', 'Sign Up'), key="auth_option")

    email = st.text_input("Email", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")

    if auth_option == 'Login':
        if st.button("Login", key="login_button"):
            user = sign_in(email, password)
            if "error" in user:
                st.error(user["error"])
            else:
                st.session_state.user_info = user
                st.rerun()

    elif auth_option == 'Sign Up':
        if st.button("Sign Up", key="signup_button"):
            user = sign_up(email, password)
            if "error" in user:
                st.error(user["error"])
            else:
                st.success("Account created successfully! Please log in.")
                # No rerun here, user should now log in with their new credentials

def show_main_app():
    """Displays the main application after a successful login."""
    user_id = st.session_state.user_info['localId']
    
    # Initialize session state variables for the main app
    if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "interview_id" not in st.session_state: st.session_state.interview_id = None
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.write(f"Welcome, *{st.session_state.user_info['email']}*!")
        if st.button("Logout"):
            # Clear all session data on logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.divider()
        st.header("Your Interviews")
        past_interviews = get_user_interviews(db, user_id)
        interview_options = ["Start a New Interview"] + past_interviews
        selected_option = st.selectbox("Choose an interview session:", options=interview_options)

        if selected_option != "Start a New Interview":
            if st.button("Load Selected Interview"):
                st.session_state.chat_history = load_chat_history(db, user_id, selected_option)
                st.session_state.interview_id = selected_option
                st.session_state.rag_chain = None
                st.warning("Chat history loaded for review.")
                st.rerun()
        
        st.divider()
        st.header("Start a New Interview")
        resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

        if st.button("Begin Interview"):
            if resume_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.interview_id = f"interview_{timestamp}"
                vector_store, user_name_from_resume = process_and_store_resume(resume_file)
                if vector_store:
                    st.session_state.rag_chain = get_conversational_rag_chain(vector_store)
                    greeting = f"Hello {user_name_from_resume}! Let's get started."
                    initial_input = "Generate the first interview question based on my resume."
                    with st.spinner("Generating first question..."):
                        response = st.session_state.rag_chain.invoke({"input": initial_input, "chat_history": []})
                    initial_ai_message = f"{greeting}\n\n{response['answer']}"
                    st.session_state.chat_history = [AIMessage(content=initial_ai_message)]
                    save_chat_history(db, user_id, st.session_state.interview_id, st.session_state.chat_history)
                    st.success("New interview started!")
                    st.rerun()
                else:
                    st.error("Failed to process resume.")
            else:
                st.warning("Please upload a resume to start.")

    # --- MAIN CHAT INTERFACE ---
    if not st.session_state.chat_history:
        st.info("Start a new interview or load a past one from the sidebar.")
    else:
        for msg in st.session_state.chat_history:
            if isinstance(msg, AIMessage):
                st.chat_message("ai", avatar="🤖").write(msg.content)
            elif isinstance(msg, HumanMessage):
                st.chat_message("human", avatar="👤").write(msg.content)

        if st.session_state.rag_chain:
            if user_query := st.chat_input("Your answer..."):
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.chat_message("human", avatar="👤").write(user_query)
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
                    ai_response = response["answer"]
                st.session_state.chat_history.append(AIMessage(content=ai_response))
                st.chat_message("ai", avatar="🤖").write(ai_response)
                save_chat_history(db, user_id, st.session_state.interview_id, st.session_state.chat_history)
        else:
             st.info("This is a past interview. You can review the conversation.")


# --- MAIN ROUTER ---
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

if st.session_state.user_info:
    show_main_app()
else:
    show_auth_page()