import streamlit as st
from snowflake_integrator import create_snowflake_conn

def insert_credentials(conn,password, email):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO USERCREDENTIALS (Password, Email) VALUES (%s, %s)", (password, email))
    cursor.close()
    conn.commit()



# Function to sign up
def sign_up(conn,password, email):
    insert_credentials(conn,password, email)
    st.success("Sign up successful! You can now login.")


# Main function
def main():
    # Create a sidebar

        conn=create_snowflake_conn()
   
        st.subheader("Sign Up")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            sign_up(conn,new_password, new_email)



if __name__ == "__main__":
    main()