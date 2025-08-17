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

# Load environment variables from .env file
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-1.5-flash", model_provider="google_genai")

result = model.invoke("What is the capital of France?")
print(result.content) 
    