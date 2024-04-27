import snowflake.connector
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
#dotenv_path = Path('c:/Users/abhis/.env')
dotenv_path = Path('/home/abhi/.env')
load_dotenv(dotenv_path=dotenv_path)


# Snowflake connection parameters
snowflake_account = os.getenv('SNOWFLAKE_ACCOUNT')
snowflake_user = os.getenv('SNOWFLAKE_USER')
snowflake_password = os.getenv('SNOWFLAKE_PASSWORD')
snowflake_database = os.getenv('SNOWFLAKE_DATABASE')
snowflake_schema = os.getenv('SNOWFLAKE_SCHEMA')
snowflake_warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')


def create_snowflake_conn():
    # Snowflake connection
    conn = snowflake.connector.connect(
        user=snowflake_user,
        password=snowflake_password,
        account=snowflake_account,
        warehouse=snowflake_warehouse,
        database=snowflake_database,
        schema=snowflake_schema
    )
    return conn

def upload_to_stage(cursor,file_path,file_name):
    # Create a stage
    cursor.execute("CREATE STAGE IF NOT EXISTS KNOWLEDGEBASE_STAGE ")
    cursor.execute(f"PUT file:///{file_path}/{file_name} @KNOWLEDGEBASE_STAGE")
    #cursor.execute("PUT file:///C:/Users/abhis/Documents/Data_Engineering/AI4MentalHealth/staging_files/knowledge.csv @KNOWLEDGEBASE_STAGE") 
    return

def stage_to_table(cursor):
    # Copy data from stage to table
    cursor.execute("COPY INTO CHATBOT_KNOWLEDGE FROM @KNOWLEDGEBASE_STAGE FILE_FORMAT='PYTHON' ON_ERROR=CONTINUE ")
    return

"""
def main():
    conn = create_snowflake_conn()
    cursor = conn.cursor()
    upload_to_stage(cursor,Path(os.getenv('FILE_PATH')),f'knowledge-{unique_identifier}.csv')
    stage_to_table(cursor)
    cursor.close()
    conn.close()
  """
