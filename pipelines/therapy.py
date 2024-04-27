import streamlit as st

from snowflake_integrator import create_snowflake_conn
import pandas as pd
def snowflake_loader(conn,query):

    with conn.cursor() as cur:
            cur.execute(query)
            df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
            return df

def main():
    st.title('Therapist Dashboard')

    # Load data from Snowflake
    therapist_query = "SELECT * FROM THERAPIST_VIEW"
    conn=create_snowflake_conn()
    therapists_df = snowflake_loader(conn,therapist_query)
    

    # Visualizations and widgets
    if not therapists_df.empty:
        st.subheader('Patient Context')
        st.write(therapists_df)
        # Add more visualizations as needed
    else:
         st.write("No data available")