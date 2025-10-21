from click import prompt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from openai import chat
import streamlit as st
import os
import dotenv 
# Load environment variables from .env file
dotenv.load_dotenv()



# Set environment variables directly in code
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant.Please answer user's qurery"),
    ("user", "Query : {query}"),
])

#OpenApi LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = prompt | llm | StrOutputParser()
st.title("Query Answering App")



