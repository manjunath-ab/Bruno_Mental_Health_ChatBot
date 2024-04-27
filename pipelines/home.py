import streamlit as st

# Set page configuration
st.set_page_config(page_title="Mental Health Chat", page_icon=":speech_balloon:")

# Set theme colors
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(45deg, #FF7E5F, #1A4876);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(45deg, #1A4876, #FF7E5F);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# from  login import login, sign_up
import login
import agent_chat_prod
import cohere_ranker
import therapy
import dashboard

def main():
    
    # Initialize login status
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False


    option = st.sidebar.selectbox("Menu", ["Chat", "Sign Up", "Login","Therapist View","Admin Dashboard"])
    st.image("dog2.jpeg", width=300)

    # Upload an image on the sidebar
    avatar_image = "ai4mh.jpeg"
    
    # Check if an image file is uploaded
    if avatar_image is not None:
        # Display the uploaded image on the sidebar
        st.sidebar.image(avatar_image, use_column_width=True)
        st.sidebar.markdown("""
            <div style="font-family: 'Arial', sans-serif; font-size: 20px; font-style: bold;">
                Appointment Booking Section
            </div>
        """, unsafe_allow_html=True)

    if option == "Chat":
        if (st.session_state.is_logged_in and st.session_state.user_type == "Patient") or (st.session_state.is_logged_in and st.session_state.user_type == "Admin"):
            cohere_ranker.main()
        else:
            st.write("""<div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;">
                        <h3>Please log in as a patient to access the chat.</h3>
                       </div>
                    """, unsafe_allow_html=True)

    elif option == "Therapist View":
        if (st.session_state.is_logged_in and st.session_state.user_type == "Doctor") or (st.session_state.is_logged_in and st.session_state.user_type == "Admin"):
            therapy.main()
        else:
            st.write("""<div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;">
                        <h3>Please log in as a therapist to access this page.</h3>
                       </div>
                    """, unsafe_allow_html=True)

    elif option == "Admin Dashboard":
        if st.session_state.is_logged_in and st.session_state.user_type == "Admin":
            st.write("""<div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;">
                        <h3>Welcome to the Admin Dashboard</h3>
                       </div>
                    """, unsafe_allow_html=True)
            dashboard.main()
        else:
            st.write("""<div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;">
                        <h3>Please log in as an admin to access the dashboard.</h3>
                       </div>
                    """, unsafe_allow_html=True)
    
    elif option == "Login":
        st.session_state.is_logged_in = login.main()
        st.session_state.is_logged_in=True

    elif option == "Dashboard":  # Add this new condition
        if st.session_state.is_logged_in:
            dashboard.main()  # Call the main function of the therapist dashboard
        else:
            st.write("""<div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;">
                        <h3>Please log in to access the dashboard.</h3>
                       </div>
                    """, unsafe_allow_html=True)

    else:
        st.session_state.is_logged_in = login.main()
        st.session_state.is_logged_in=True

if __name__ == "__main__":
    main()
