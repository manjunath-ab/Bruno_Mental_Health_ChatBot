from langchain_community.document_loaders.snowflake_loader import SnowflakeLoader
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# Load environment variables from .env file
#dotenv_path = Path('/home/abhi/.env')
dotenv_path = Path('/Users/abhis/.env')

load_dotenv(dotenv_path=dotenv_path)

snowflake_account = os.getenv('SNOWFLAKE_ACCOUNT')
snowflake_user = os.getenv('SNOWFLAKE_USER')
snowflake_password = os.getenv('SNOWFLAKE_PASSWORD')
snowflake_database = os.getenv('SNOWFLAKE_DATABASE')
snowflake_schema = 'DBT_CHATBOT'
snowflake_warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
snowflake_role = os.getenv('SNOWFLAKE_ROLE')


def snowflake_loader(QUERY):
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
 return docs




def main():
    QUERY = "select * from KNOWLEDGE_BASE"
    QUERY1 = "select * from THERAPIST_DETAILS"
    docs = snowflake_loader(QUERY)
    docs1 = snowflake_loader(QUERY1)
    db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="../new_knowledge_db")
    db.add_documents(docs1)

if __name__ == "__main__":
    main()



