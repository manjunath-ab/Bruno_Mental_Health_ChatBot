import streamlit as st
from snowflake_integrator import create_snowflake_conn
#import snowflake.connector

# Connect to Snowflake



# Function to retrieve user credentials
def retrieve_credentials(conn,email):
    cursor = conn.cursor()
    cursor.execute("SELECT Password,Authority FROM UserCredentials WHERE Email = %s", (email,))
    result = cursor.fetchone()
    cursor.close()
    return result


# Function to login
def login(password, email,conn):
    result = retrieve_credentials(conn,email)
    if result is not None and result[0] == password:
        st.session_state.email=email
        st.session_state.user_type = result[1]
        st.success("Login successful!")
    else:
        st.error("Invalid credentials. Please try again.")

# Main function
def main():
    # Create a sidebar
        conn=create_snowflake_conn()
        st.subheader("Login")
        existing_email = st.text_input("Email")
        existing_password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(existing_password, existing_email,conn)


if __name__ == "__main__":
    main()