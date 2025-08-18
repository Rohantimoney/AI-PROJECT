# Iam creating a project using llm and chat models which is called as 
# Personal AI Interviewer using rag and llm 
# it will take your resume and job description and will ask you questions based on that
# it will be conversational and will help you prepare for interviews
# This is the main file where the application starts
# iam using gemini chat model to generate questions based on the resume and job description


import getpass
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings # This works for Gemini with an API key
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# Replace 'from langchain_openai import OpenAIEmbeddings' with this:
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

#---1) SETUP---
# Load environment variables from .env file
load_dotenv()
# Explicitly set the Google API key from the loaded .env file
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Initialize the embedding model and llm
model = init_chat_model("gemini-1.5-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


#---2) LOAD AND PROCESS THE DCUMENTS---

# Assuming your resume is named 'my_resume.pdf' and is in the same folder
resume_loader = PyPDFLoader("docs/RESUME4 (4).pdf")
# jd_loader = PyPDFLoader("docs/job_description.pdf") # Make sure you have this file
resume_docs = resume_loader.load()
# jd_docs = jd_loader.load()
# print(f"Loaded {len(documents)} document(s).")
# print(documents[0].page_content[:500]) # Print the first 500 characters of the first page

#COMBINING DOCS
documents = resume_docs
    
#splitting the document into smalled chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # The maximum size of each chunk
    chunk_overlap=200 # The number of characters to overlap between chunks
)
chunks = text_splitter.split_documents(documents)

# print(f"Split the document into {len(chunks)} chunks.")   


# Create a new Chroma database from the document chunks
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings
)

# Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# retriever = vector_store.as_retriever() 

# --- 3. CREATE CONVERSATIONAL RAG CHAIN ---
#  This prompt helps the AI reformulate the user's input to be a standalone question
# based on the chat history. This makes retrieving relevant documents more effective.
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Create a history-aware retriever chain
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

# This is the main prompt for the AI interviewer. It takes context (from the resume/JD)
# and the chat history to generate a relevant interview question.
system_prompt = """
You are an expert AI interviewer and career coach. Your goal is to conduct a professional interview and provide feedback.
Based on the provided context (resume and job description) and the conversation history, perform two tasks in your response:
1.  **Provide Feedback:** In one short paragraph, give constructive feedback on the candidate's previous answer. Analyze if it was clear, relevant, and how well it connected to the skills mentioned in the context. Start this section with "Feedback:".
2.  **Ask the Next Question:** After the feedback, ask the next relevant interview question based on the context and conversation flow.

- Ask one question at a time.
- Your questions should be insightful and probe into the candidate's skills and experience.
- For the very first question of the interview, skip the feedback part and just ask the opening question.
- Keep the conversation flowing naturally. Do not repeat questions.

Context:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the final chain that combines document retrieval and question generation
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



# --- 4. RUN THE INTERACTIVE INTERVIEW ---

def run_interview():
    """
    Manages the interactive conversation loop for the AI interview.
    """
    chat_history = []
    
    # Start the conversation with a greeting and the first question
    print("AI Interviewer: Hello! I'm your AI interviewer for today. I've reviewed your resume and the job description. Let's get started.")
    
    # Invoke the chain with an initial input to generate the first question
    initial_input = "Generate the first interview question based on the candidate's most prominent skill or recent project."
    response = rag_chain.invoke({"input": initial_input, "chat_history": chat_history})
    
    # Print the first question
    first_question = response["answer"]
    print(f"AI Interviewer: {first_question}")
    
    # Add the AI's first question to the chat history
    chat_history.append(AIMessage(content=first_question))
    
    # Start the interactive loop
    while True:
        # Get user's answer
        user_answer = input("You: ")
        
        if user_answer.lower() in ["quit", "exit", "stop"]:
            print("AI Interviewer: Thank you for your time. The interview has now ended.")
            break
            
        # Add user's answer to chat history
        chat_history.append(HumanMessage(content=user_answer))
        
        # Invoke the RAG chain to get the next question
        # The input is the user's last answer, which the AI uses as context for the next question
        response = rag_chain.invoke({"input": user_answer, "chat_history": chat_history})
        
        ai_question = response["answer"]
        print(f"AI Interviewer: {ai_question}")
        
        # Add AI's new question to chat history
        chat_history.append(AIMessage(content=ai_question))

# --- Start the application ---
if __name__ == "__main__":
    run_interview()



# 1. Define a prompt template
# prompt_template = ChatPromptTemplate.from_template("""
# Based on the following context from a candidate's resume, please generate a relevant interview question.
# Do not ask to "describe a time when," instead, ask a direct question about their experience, projects and skills.

# Context:
# {context}

# Interview Question:
# """)

# # 2. Create the "stuff documents" chain
# question_generator_chain = create_stuff_documents_chain(model, prompt_template)

# # 3. Create the final retrieval chain
# rag_chain = create_retrieval_chain(retriever, question_generator_chain)

# # --- Now you can use the chain! ---
# # Let's generate a question based on the resume's content
# response = rag_chain.invoke({"input": "Generate a technical question about the candidate's skills."})

# print(response["answer"])
