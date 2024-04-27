import snowflake.connector
from pathlib import Path
import os
from dotenv import load_dotenv
import json

dotenv_path = Path('C:/Users/abhis/.env')

load_dotenv(dotenv_path=dotenv_path)

snowflake_account = os.getenv('SNOWFLAKE_ACCOUNT')
snowflake_user = os.getenv('SNOWFLAKE_USER')
snowflake_password = os.getenv('SNOWFLAKE_PASSWORD')
snowflake_database = os.getenv('SNOWFLAKE_DATABASE')
snowflake_schema = os.getenv('SNOWFLAKE_SCHEMA')
snowflake_warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
snowflake_role = os.getenv('SNOWFLAKE_ROLE')


def create_snowflake_conn():
 conn = snowflake.connector.connect(
    user='',
    password='',
    account='',
    warehouse='',
    database='',
    schema=''
)
 return conn



def get_availability(conn,doctor_name):

    query = f"SELECT availability FROM THERAPIST_DETAILS WHERE THERAPIST_NAME = '{doctor_name}'"
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()[0]  # Assuming there's only one row
    availability = json.loads(result)
    cursor.close()

    return availability

def insert_into_therapist_view(conn,doctor_name,patient_email,summary):
    summary=json.dumps(summary)
    summary = summary.replace("'", "")
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO THERAPIST_VIEW (THERAPIST_NAME, PATIENT_EMAIL,SUMMARY) SELECT '{doctor_name}', '{patient_email}',PARSE_JSON('{summary}')")
    cursor.close()

