from langchain_community.document_loaders.snowflake_loader import SnowflakeLoader
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# Load environment variables from .env file
dotenv_path = Path('/Users/sivaranjanis/Desktop/genai/AI4MentalHealth/.env')
load_dotenv(dotenv_path=dotenv_path)


snowflake_account = os.getenv('SNOWFLAKE_ACCOUNT')
snowflake_user = os.getenv('SNOWFLAKE_USER')
snowflake_password = os.getenv('SNOWFLAKE_PASSWORD')
snowflake_database = os.getenv('SNOWFLAKE_DATABASE')
snowflake_schema = os.getenv('SNOWFLAKE_SCHEMA')
snowflake_warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
snowflake_role = os.getenv('SNOWFLAKE_ROLE')
QUERY = "select * from THERAPIST_DETAILS"
snowflake_loader = SnowflakeLoader(
    query=QUERY,
    user=snowflake_user,
    password=snowflake_password,
    account=snowflake_account,
    warehouse=snowflake_warehouse,
    database=snowflake_database,
    schema=snowflake_schema,
    role=snowflake_role
)
snowflake_documents = snowflake_loader.load()



text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(snowflake_documents)
#db = Chroma.from_documents(docs, OpenAIEmbeddings())
db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="./chroma_db")
#db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

query = "Tell me about bipolar disorder and how to cope with it"
result = db.similarity_search(query)
# k is the number of chunks to retrieve
retriever = db.as_retriever(k=4)

docs = retriever.invoke("how can i battle depression?")

"""
CHATBOT SECTION
"""
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)

#from langchain_core.messages import HumanMessage

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Imagine you are a therapist, talk to the user like a friend who understands their problem and keep the answer short..end with a question:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

"""
document chain
"""
from langchain.memory import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("how can I battle depression?")

document_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
        "context": docs,
    }
)

"""
retreival chain
"""
from typing import Dict

from langchain_core.runnables import RunnablePassthrough


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)

response = retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    }
)

"""
CHATTING SECTION
"""
demo_ephemeral_chat_history.add_ai_message(response["answer"])

demo_ephemeral_chat_history.add_user_message("tell me more about that!")

response=retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    },
)

print(response['answer'])