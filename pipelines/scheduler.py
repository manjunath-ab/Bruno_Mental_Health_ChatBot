import os
from dotenv import load_dotenv
import csv
from faker import Faker
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()
OPENAI_API_KEY = ''

fake = Faker()

def generate_therapists_data(num_therapists):
    therapists_data = []
    for _ in range(num_therapists):
        therapist = {
            "name": fake.name(),
            "available_slots": fake.random_number(digits=2),
            "max_patients_per_day": fake.random_number(digits=1)
        }
        therapists_data.append(therapist)
    return therapists_data

def generate_patients_data(num_patients):
    patients_data = []
    for _ in range(num_patients):
        patient = {
            "name": fake.name(),
            "preferences": fake.sentence()
        }
        patients_data.append(patient)
    return patients_data

therapists_data = generate_therapists_data(5)  # Generate 5 therapists
patients_data = generate_patients_data(10)     # Generate 10 patients

chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

system_message_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert scheduler for therapy sessions with proficiency in interpreting JSON."
)

human_message_prompt = HumanMessagePromptTemplate.from_template(
"""
The following therapists' data contains therapist names, their available time slots, and the maximum number of patients they can handle per day:
{therapists_data}

Here is the data for a specific patient, including their preferences and considerations. Your task is to schedule therapy sessions for this patient.
{patient}

The number of sessions per week for each patient must be the same as mentioned in the data. You must make sure that sessions should not overlap. If there is a conflict between the patient's preferences and the scheduling, ignore the preference. Include a brief comment at the end regarding the extent to which the preferences have been met, specifying the therapist names.

The timetable should be prepared in the following format:
PatientName:
Therapist1:
Sessions list
Therapist2:
Sessions list
Therapist3:
Sessions list
...

Remember that sessions must not overlap, you can ignore preferences when needed.
"""
)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)

for patient in patients_data:
    response = chain.run(therapists_data=therapists_data, patient=patient)
    print(response, '\n')