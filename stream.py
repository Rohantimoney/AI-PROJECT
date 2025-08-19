import nest_asyncio
nest_asyncio.apply()

import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

# --- 1. SETUP AND CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()
# Explicitly set the Google API key from the loaded .env file for robustness
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Interviewer", page_icon="🤖")
st.title("🤖 Personal AI Interviewer")
st.markdown("Upload your resume to start your mock interview. Get real-time feedback on your answers!")

# --- Core Application Logic (Functions) ---

@st.cache_resource
def get_vectorstore_from_files(pdf_files):
    """Loads PDF files, splits them into chunks, and creates a vector store."""
    if not pdf_files:
        return None
    
    with st.spinner("Processing document... This may take a moment."):
        documents = []
        for pdf_file in pdf_files:
            # PyPDFLoader needs a file path, so we save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(pdf_file.getvalue())
                loader = PyPDFLoader(tmpfile.name)
                documents.extend(loader.load())
            os.remove(tmpfile.name) # Clean up the temporary file

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Initialize embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create a Chroma vector database
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store

def get_conversational_rag_chain(vector_store):
    """Creates the conversational RAG chain for the interview."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # This is the main prompt for the AI interviewer, updated to only use the resume.
    system_prompt = """
    You are an expert AI interviewer and career coach. Your goal is to conduct a professional interview and provide feedback.
    Based on the provided context from the candidate's resume and the conversation history, perform two tasks in your response:
    1.  **Provide Feedback:** In one short paragraph, give constructive feedback on the candidate's previous answer. Analyze if it was clear, relevant, and how well it connected to the skills mentioned in the context. Start this section with "Feedback:".
    2.  **Ask the Next Question:** After the feedback, ask the next relevant interview question based on the context and conversation flow.
    NOTE:- before asking the first question to the user , always greet the user according to his name given in the resume(at the very start of resume , eg - Shashwat Shahi , Rohan beri etc.) if not found 
    in the resume then leave it.
    - 
    - Ask one question at a time.
    - Your questions should be insightful and probe into the candidate's skills and experience.
    - For the very first question of the interview, skip the feedback part and just ask the opening question.
    - Keep the conversation flowing naturally. Do not repeat questions.

    Context:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 2. STREAMLIT UI AND SESSION STATE MANAGEMENT ---

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("Upload Document")
    resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

    if st.button("Start Interview"):
        if resume_file:
            vector_store = get_vectorstore_from_files([resume_file])
            if vector_store:
                st.session_state.rag_chain = get_conversational_rag_chain(vector_store)
                initial_input = "Generate the first interview question based on the candidate's most prominent skill or recent project."
                
                with st.spinner("Generating first question..."):
                    response = st.session_state.rag_chain.invoke({"input": initial_input, "chat_history": []})
                
                st.session_state.chat_history = [AIMessage(content=response["answer"])]
                st.success("Document processed! The interview is ready.")
            else:
                st.error("Failed to process document.")
        else:
            st.warning("Please upload your resume.")

# --- Main Chat Interface ---
if st.session_state.rag_chain:
    # Display chat messages from history
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            st.chat_message("ai", avatar="🤖").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("human", avatar="👤").write(msg.content)

    # Get user input from chat box
    if user_query := st.chat_input("Your answer..."):
        # Add user's message to history and display it
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.chat_message("human", avatar="👤").write(user_query)

        # Get AI's response (feedback + next question)
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            ai_response = response["answer"]
        
        # Add AI's response to history and display it
        st.session_state.chat_history.append(AIMessage(content=ai_response))
        st.chat_message("ai", avatar="🤖").write(ai_response)
else:
    st.info("Upload your resume in the sidebar and click 'Start Interview' to begin.")
