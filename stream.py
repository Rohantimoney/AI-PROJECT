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
# Use Streamlit's secrets for deployment
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback to local .env file for development
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Interviewer", page_icon="🤖")
st.title("🤖 Personal AI Interviewer")
st.markdown("Upload your resume to start a dynamic mock interview. Get real-time feedback and even ask questions about your own resume!")

# --- Core Application Logic (Functions) ---

@st.cache_resource
def get_llm():
    """Initializes and caches the Language Model."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@st.cache_resource
def get_embeddings():
    """Initializes and caches the Embedding Model."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def process_resume(pdf_file, _llm, _embeddings):
    """Loads a PDF, creates a vector store, and extracts the user's name."""
    if not pdf_file:
        return None, None
    
    with st.spinner("Analyzing your resume... This may take a moment."):
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.getvalue())
            loader = PyPDFLoader(tmpfile.name)
            documents = loader.load()
        os.remove(tmpfile.name)

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create a Chroma vector database using the cached embedding model
        vector_store = Chroma.from_documents(documents=chunks, embedding=_embeddings)

        # Extract user's name from the first page of the resume
        try:
            first_page_content = documents[0].page_content
            name_extraction_prompt = f"From the following resume text, extract only the full name of the candidate. Respond with nothing but the name.\n\nText: {first_page_content[:1000]}"
            name_response = _llm.invoke(name_extraction_prompt)
            user_name = name_response.content.strip()
        except (IndexError, Exception):
            user_name = "Candidate" # Fallback name

    return vector_store, user_name

def get_conversational_rag_chain(_vector_store, _llm):
    """Creates the conversational RAG chain for the interview."""
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(_llm, retriever, contextualize_q_prompt)
    
    system_prompt = """
    You are an expert AI interviewer and career coach. Your goal is to conduct a professional, dynamic interview.
    
    **Your Tasks:**
    1.  **Answer User Questions:** If the user's input is a question about their own resume (e.g., "What projects did I list?"), answer it directly based on the provided context. After answering, seamlessly transition back to being an interviewer.
    2.  **Provide Feedback:** After the user answers an interview question, provide one short paragraph of constructive feedback. Analyze clarity, relevance, and connection to the skills in the resume. Start with "Feedback:".
    3.  **Ask Interview Questions:** Ask the next relevant interview question. Vary your questions between technical, behavioral, and situational. If the user's answer is brief, ask a probing follow-up question.

    **Rules:**
    - For the very first turn, just ask the opening interview question (do not provide feedback).
    - If you just answered a user's question about their resume, your next response should be an interview question (without feedback).
    - Keep the conversation flowing naturally. Do not repeat questions.

    Context from Resume:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(_llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 2. STREAMLIT UI AND SESSION STATE MANAGEMENT ---

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Candidate"

# Initialize models once
llm = get_llm()
embeddings = get_embeddings()

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("Upload Document")
    resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

    if st.button("Start Interview"):
        if resume_file:
            vector_store, user_name = process_resume(resume_file, llm, embeddings)
            if vector_store:
                st.session_state.rag_chain = get_conversational_rag_chain(vector_store, llm)
                st.session_state.processed_file_id = resume_file.file_id
                st.session_state.user_name = user_name
                
                greeting = f"Hello {st.session_state.user_name}! I've reviewed your resume. Let's get started."
                initial_input = "Generate the first interview question based on my resume."
                
                with st.spinner("Generating first question..."):
                    response = st.session_state.rag_chain.invoke({"input": initial_input, "chat_history": []})
                
                initial_ai_message = f"{greeting}\n\n{response['answer']}"
                st.session_state.chat_history = [AIMessage(content=initial_ai_message)]
                st.success("Interview started!")
            else:
                st.error("Failed to process document.")
        else:
            st.warning("Please upload your resume.")

# --- Automatic Reset Logic ---
if st.session_state.rag_chain and (resume_file is None or resume_file.file_id != st.session_state.get("processed_file_id")):
    st.info("Resume file removed. Resetting the interview session...")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Clear all caches
    st.cache_resource.clear()
    st.rerun()

# --- Main Chat Interface ---
if st.session_state.rag_chain:
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            st.chat_message("ai", avatar="🤖").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("human", avatar="👤").write(msg.content)

    if user_query := st.chat_input("Your answer or question..."):
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.chat_message("human", avatar="👤").write(user_query)

        try:
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                ai_response = response["answer"]
            
            st.session_state.chat_history.append(AIMessage(content=ai_response))
            st.chat_message("ai", avatar="🤖").write(ai_response)
        except Exception as e:
            error_message = f"Sorry, I encountered an error. This can sometimes happen due to network issues or high demand. Please try rephrasing your answer or restarting the interview. (Error: {e})"
            st.session_state.chat_history.append(AIMessage(content=error_message))
            st.chat_message("ai", avatar="🤖").write(error_message)

else:
    st.info("Upload your resume in the sidebar and click 'Start Interview' to begin.")
