
# Import necessary libraries
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from snowflake.connector import connect
from dotenv import load_dotenv
import os
import snowflake.connector
import seaborn as sns
import plotly.express as px
import numpy  as np
import json
import calendar
import plotly.graph_objects as go
import plotly.express as px


def main():
    # Load .env environment variables
    dotenv_path = '/Users/abhis/.env'  # Specify your dotenv path
    load_dotenv(dotenv_path=dotenv_path)

    # Set up Snowflake connection parameters
    snowflake_account = os.getenv('SNOWFLAKE_ACCOUNT')
    snowflake_user = os.getenv('SNOWFLAKE_USER')
    snowflake_password = os.getenv('SNOWFLAKE_PASSWORD')
    snowflake_database = os.getenv('SNOWFLAKE_DATABASE')
    snowflake_schema = os.getenv('SNOWFLAKE_SCHEMA')
    snowflake_warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
    snowflake_role = os.getenv('SNOWFLAKE_ROLE')

    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user=snowflake_user,
        password=snowflake_password,
        account=snowflake_account,
        warehouse=snowflake_warehouse,
        database=snowflake_database,
        schema=snowflake_schema,
        role=snowflake_role
    )

    # Query to fetch data from APPOINTMENT_DETAILS and THERAPIST_DETAILS tables
    query_appointments = "SELECT * FROM APPOINTMENT_DETAILS"
    query_therapists = "SELECT * FROM THERAPIST_DETAILS"

    # Load data into Pandas DataFrames
    appointments_df = pd.read_sql(query_appointments, conn)
    therapists_df = pd.read_sql(query_therapists, conn)

    # Close Snowflake connection
    conn.close()

    # Now you can use appointments_df and therapists_df for visualization
    st.title('Therapist Dashboard')
    # Number of Patients per Therapist (Bar Chart)
    st.header('Number of Patients per Therapist')
    patients_per_therapist = appointments_df.groupby('THERAPIST_ID')['NO_OF_PATIENTS'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='THERAPIST_ID', y='NO_OF_PATIENTS', data=patients_per_therapist)
    plt.title('Number of Patients per Therapist')
    plt.xlabel('Therapist ID')
    plt.ylabel('Number of Patients')
    st.pyplot(plt)


    # Therapist Specialization Distribution (Pie Chart)
    fig = px.pie(therapists_df, names='SPECIALIZATION', title='Therapist Specialization Distribution')
    st.plotly_chart(fig)

    # Patients vs. Specialization (Scatter Plot)
    st.header('Patients vs. Specialization')
    merged_df = pd.merge(appointments_df, therapists_df, on='THERAPIST_ID', how='inner')
    fig = px.scatter(merged_df, x='SPECIALIZATION', y='NO_OF_PATIENTS', color='SPECIALIZATION',
                    title='Patients vs. Specialization')
    st.plotly_chart(fig)

    st.header('Therapist Availability by Day and Hour')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = ['9AM', '10AM', '11AM', '12PM', '1PM', '2PM', '3PM', '4PM', '5PM']
    availability = np.random.randint(0, 10, size=(len(days), len(hours)))  # Sample data
    plt.figure(figsize=(12, 7))
    sns.heatmap(availability, annot=True, fmt="d", cmap="YlGnBu", xticklabels=hours, yticklabels=days)
    plt.title('Therapist Availability by Day and Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Day of the Week')
    st.pyplot(plt)  # Display the figure in the Streamlit app



    

    st.header('Number of Patients by Therapist Specialization')
    selected_specializations = st.multiselect('Select Specializations', therapists_df['SPECIALIZATION'].unique())

    # Filter data based on selected specializations
    filtered_df = merged_df[merged_df['SPECIALIZATION'].isin(selected_specializations)]
    patients_count = filtered_df.groupby('SPECIALIZATION')['NO_OF_PATIENTS'].sum().reset_index()

    # Create and display bar chart
    fig = go.Figure([go.Bar(x=patients_count['SPECIALIZATION'], y=patients_count['NO_OF_PATIENTS'])])
    fig.update_layout(title='Number of Patients for Selected Specializations', xaxis_title='Specialization', yaxis_title='Number of Patients')
    st.plotly_chart(fig)

  

    # Assuming therapists_df is your DataFrame containing all the details
    def display_selected_therapist_details(therapists_df):
        # Create a dropdown to select a therapist
        therapist_names = therapists_df['THERAPIST_NAME'].tolist()
        selected_therapist_name = st.selectbox('Select a Therapist:', therapist_names)
        
        # Get the details of the selected therapist
        selected_therapist_details = therapists_df[therapists_df['THERAPIST_NAME'] == selected_therapist_name].iloc[0]
        
        # Display the details of the selected therapist
        st.markdown(f"""
            **Specialization:** {selected_therapist_details['SPECIALIZATION']}  
            **Description:** {selected_therapist_details['DESCRIPTION']}  
            **Contact Email:** [Email](mailto:{selected_therapist_details['CONTACT_EMAIL']})  
            **Clinic Address:** {selected_therapist_details['CLINIC_ADDRESS']}  
            **Availability:**  
        """)
        # Convert the availability JSON string back to a dictionary
        availability = json.loads(selected_therapist_details['AVAILABILITY'])
        for day, times in availability.items():
            st.text(f"{day}: {', '.join(times)}")

        # Add interactive elements here, e.g., buttons for contacting or viewing on map
        if st.button(f'Contact {selected_therapist_details["THERAPIST_NAME"]}', key='contact'):
            # This could be replaced with actual actions like opening a contact form or sending an email
            st.write(f"Send an email to: {selected_therapist_details['CONTACT_EMAIL']}")

        # Here you could add more interactive elements like booking buttons or links to external pages

    # Call the function with your DataFrame
    display_selected_therapist_details(therapists_df)


   

    st.header('Distribution of Patient Visits')

    # Convert NO_OF_PATIENTS to numeric type if it's not already
    appointments_df['NO_OF_PATIENTS'] = pd.to_numeric(appointments_df['NO_OF_PATIENTS'], errors='coerce')

    # Create the histogram using Plotly Express
    fig = px.histogram(appointments_df, x='NO_OF_PATIENTS',
                    nbins=20,  # Starting number of bins
                    title='Distribution of Patient Visits',
                    labels={'NO_OF_PATIENTS': 'Number of Visits'},  # Customize axis labels
                    color_discrete_sequence=['blue'])  # Customize histogram color
    fig.update_layout(bargap=0.2)  # Customize gap between bars

    # Add range slider for interactivity
    fig.update_xaxes(title='Number of Visits',
                    rangeslider_visible=True)

    # Add a bin size slider in Streamlit to adjust the number of bins dynamically
    bins = st.slider('Select Number of Bins', min_value=5, max_value=100, value=20, step=5)
    fig.update_traces(xbins=dict(  # Update the number of bins based on the slider
        start=appointments_df['NO_OF_PATIENTS'].min(),
        end=appointments_df['NO_OF_PATIENTS'].max(),
        size=(appointments_df['NO_OF_PATIENTS'].max() - appointments_df['NO_OF_PATIENTS'].min()) / bins
    ))

    # Display the figure in the Streamlit app
    st.plotly_chart(fig)
