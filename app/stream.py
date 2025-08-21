# stream.py

import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
import streamlit_authenticator as stauth
from datetime import datetime

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import yaml
from yaml.loader import SafeLoader

# Firebase Imports
from firebase_helpers import initialize_firebase, get_user_interviews, save_chat_history, load_chat_history

# --- 1. SETUP AND CONFIGURATION ---
load_dotenv()
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="AI Interviewer", page_icon="🤖")
st.title("🤖 Personal AI Interviewer")

# --- Initialize Firebase and Firestore client ---
db = initialize_firebase()

# --- User Authentication ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login()

# --- CORE APP LOGIC (RUNS ONLY ON SUCCESSFUL LOGIN) ---"ADDED  THIS LINE OF CODE"

if authentication_status:
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

    # --- SESSION STATE INITIALIZATION ---
    if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "interview_id" not in st.session_state: st.session_state.interview_id = None
    
    # --- SIDEBAR UI ---
    with st.sidebar:
        st.write(f"Welcome, *{name}*!")
        authenticator.logout("Logout", "sidebar")
        st.divider()

        # Interview Selection
        st.header("Your Interviews")
        past_interviews = get_user_interviews(db, username)
        
        # Add "New Interview" option
        interview_options = ["Start a New Interview"] + past_interviews
        selected_option = st.selectbox("Choose an interview session:", options=interview_options)

        if selected_option != "Start a New Interview":
            if st.button("Load Selected Interview"):
                st.session_state.chat_history = load_chat_history(db, username, selected_option)
                st.session_state.interview_id = selected_option
                # To get the RAG chain back, we would need to store the resume. 
                # For now, loading history is for review. A full implementation
                # would store resume data or re-process it.
                st.session_state.rag_chain = None # Reset chain as we don't have the resume context
                st.warning("Chat history loaded for review. To continue an interview, please re-upload the resume and start a new session.")
                st.rerun()

        st.divider()
        st.header("Start a New Interview")
        resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

        if st.button("Begin Interview"):
            if resume_file:
                # Generate a unique ID for this new interview
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.interview_id = f"interview_{timestamp}"
                
                vector_store, user_name_from_resume = process_and_store_resume(resume_file)
                if vector_store:
                    st.session_state.rag_chain = get_conversational_rag_chain(vector_store)
                    
                    greeting = f"Hello {user_name_from_resume}! I've reviewed your resume. Let's get started."
                    initial_input = "Generate the first interview question based on my resume."
                    
                    with st.spinner("Generating first question..."):
                        response = st.session_state.rag_chain.invoke({"input": initial_input, "chat_history": []})
                    
                    initial_ai_message = f"{greeting}\n\n{response['answer']}"
                    st.session_state.chat_history = [AIMessage(content=initial_ai_message)]
                    
                    # Save the very first message to Firestore
                    save_chat_history(db, username, st.session_state.interview_id, st.session_state.chat_history)
                    st.success("New interview started!")
                    st.rerun()
                else:
                    st.error("Failed to process resume.")
            else:
                st.warning("Please upload a resume to start.")


    # --- MAIN CHAT INTERFACE ---
    if st.session_state.chat_history:
        if not st.session_state.rag_chain:
            st.info("This is a past interview. You can review the conversation.")

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
                    response = st.session_state.rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    ai_response = response["answer"]
                
                st.session_state.chat_history.append(AIMessage(content=ai_response))
                st.chat_message("ai", avatar="🤖").write(ai_response)

                # Save the updated history after each turn
                save_chat_history(db, username, st.session_state.interview_id, st.session_state.chat_history)

# --- Login Error Messages ---
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')