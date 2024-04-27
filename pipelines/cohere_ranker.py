import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from typing import Dict
from streamlit_chat import message
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import tool
from langchain.tools.base import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from dateutil import parser
import datetime
import calendar
import random
import json
from faker import Faker
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
import re
from snowflake_integrator import get_availability,create_snowflake_conn,insert_into_therapist_view
from email_patient import send_email
from create_event import create_event,get_credentials,build
fake = Faker()
today = datetime.datetime.now()
from langchain.chains import create_extraction_chain
from weekdate_converter import convert_to_iso8601
import logging 
import sys
from langchain_core.messages import HumanMessage
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain_core.runnables import RunnablePassthrough
from collections import defaultdict

hours = (9, 18)   # open hours

my_logger=logging.getLogger("__init__")
stream_handler=logging.StreamHandler(sys.stdout)
my_logger.addHandler(stream_handler)
my_logger.setLevel(logging.INFO)
def load_environment_variables():
    dotenv_path = Path('C:/Users/abhis/.env')
    load_dotenv(dotenv_path=dotenv_path)

def initialize_chat_and_db():
    chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
    gpt_chat= ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
    cohere_chat= Cohere(temperature=0)
    db = Chroma(persist_directory="../new_knowledge_db",embedding_function=OpenAIEmbeddings())
    return chat, gpt_chat,cohere_chat,db

#If the user question is not relevant to mental health or therapists details or details of the user, don't make something up and just say "I don't know":
def create_system_template():
    SYSTEM_TEMPLATE = """
    Imagine you are a human friend,remember their name and  talk to the user like a friend who understands their problem and keep the reply short and use their name if given in your repsonses.Do not diagnose the patient. Ask the user if they need suggestions on coping mechanisms, self care practices and support systems used by other people for similar mental health issue.End with a follow up question unrelated to therapy to get more information on the user's mental state. 
    * Do not bring up therapy if the user does not mention it.*
    
    <context>
    {context}
    </context>
    """
    return SYSTEM_TEMPLATE

def create_cohere_system_template():
    SYSTEM_TEMPLATE = """
Imagine you are a human friend of the user. Use their name in your response and speak to them like a caring friend who wants to catch up and see how they're doing. Keep your responses concise and friendly.
Begin the conversation by greeting them and showing interest in their life. 
*Do not suggest counselling or therapy unless the user specifically asks for it*
Ask open-ended questions about their recent experiences, hobbies, or anything positive they'd like to share.
If the user brings up any challenges or difficulties they're facing, express empathy and support. Ask if they would like to hear some coping emchanisms, self-care ideas that have helped others in similar situations. However, avoid diagnosing or making assumptions about their mental health.
If the user does not mention any specific issues or challenges, continue the conversation by asking follow-up questions about their interests, goals, or general well-being.
If the user's message is not relevant to the conversation or includes details you cannot address, gently redirect the conversation or respond with "I'm not sure, but I'm always here to listen and support you as a friend."

    
    <context>
    {context}
    </context>
    """
    return SYSTEM_TEMPLATE

def create_retriever(db):
    retriever = db.as_retriever(k=25)
    compressor = CohereRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )
    
    return compression_retriever,retriever

"""
def initialize_chat_history():
    demo_ephemeral_chat_history = ChatMessageHistory()
    return demo_ephemeral_chat_history

"""
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! I'm Bruno. You can talk to me about anything related to mental health and therapy. How can I help you today?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey"]
    
    if 'gpt_generated' not in st.session_state:
        st.session_state['gpt_generated'] = ["Hello ! this chat directly talks to ChatGPT"]

    if 'gpt_past' not in st.session_state:
        st.session_state['gpt_past'] = ["Hey"]

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = ConversationBufferMemory()

    if 'cohere_generated' not in st.session_state:
        st.session_state['cohere_generated'] = ["Hello ! this chat directly talks to Cohere"]
    if 'cohere_past' not in st.session_state:
        st.session_state['cohere_past'] = ["Hey"]

    if 'cohere_chat_history' not in st.session_state:
        st.session_state['cohere_chat_history'] = ConversationBufferMemory()
    
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))




def display_avatar_image():
    st.image("zen.jpg", width=300)

    # Upload an image on the sidebar
    avatar_image = "avatar.jpg"

    # Check if an image file is uploaded
    if avatar_image is not None:
        # Display the uploaded image on the sidebar
        st.sidebar.image(avatar_image, use_column_width=True)
        #I am Zenny! I'm here to be your virtual friend, to chat with you, and to help you find the support and resources you need. Whether you're feeling down and need someone to talk to, or you're looking for information on mental health and therapy, I'm here to listen and assist. So, let's chat and find the help you need!
        st.sidebar.markdown("""
        <div style="font-family: 'Arial', sans-serif; font-size: 20px; font-style: bold;">
            Appointment Booking Section
        </div>
    """, unsafe_allow_html=True)

def get_retriever_tool(retriever):
    tool = create_retriever_tool(
    retriever,
    "mental_health_and_therapist_knowledge_base",
    "This is a retreiver tool for which has information on the coping mechanisms,triggers, self-care routines, support systems of people dealing with anxiety,depression and bi polar disorder. It also contains therapist details ",
    )
    return tool


# create random schedule
def createSchedule(daysAhead=5, perDay=8):
    schedule = {}
    for d in range(0, daysAhead):
        date = (today + datetime.timedelta(days=d)).strftime('%m/%d/%y')
        schedule[date] = {}

        for h in range(0, perDay -d):
            hour = random.randint(hours[0], hours[1])
            if hour not in schedule[date]:
                schedule[date][hour] = fake.name()
                
    return schedule

# get available times for a date
def getAvailTimes(date, num=10):

    #schedule=createSchedule()
    if '/' not in date:
        return 'Date parameter must be in format: mm/dd/yy'
    
    
    if date in schedule:
        hoursAvail = 'Hours available on %s are ' % date
        for h in range(hours[0], hours[1] + 1):
            if str(h) not in schedule[date]:
                hoursAvail += str(h) + ':00, '
                num -= 1
                if num == 0:
                    break

        if num > 0:
            hoursAvail = hoursAvail[:-2] + ' - all other times are reserved'
        else:
            hoursAvail = hoursAvail[:-2]

        return hoursAvail
    else:
        return 'That day is entirely open. All times are available.'



# sche# schedule available time
def scheduleTime(dateTime):
    date, time = dateTime.split(',')
    
    if not date or not time:
        return "Sorry, parameters must be date and time comma-separated. For example, '12/31/23, 10:00' would be the input for Dec 31, 2023, at 10 AM."
    if date not in schedule:
        return 'No schedule available yet for this date.'

    # get hours
    if ':' in time:
        timeHour = int(time[:time.index(':')])
        print(timeHour)
        
        if timeHour not in schedule[date]:
            if timeHour >= hours[0] and timeHour <= hours[1]:
                schedule[date][timeHour] = fake.name()
                saveSchedule(schedule)  # Corrected line: passing the schedule variable
                return 'Thank you, the appointment is scheduled for %s under the name %s.' % (time, schedule[date][timeHour])
            else:
                return '%s is after hours. Please select a time during business hours.' % time
        else:
            return 'Sorry, that time (%s) on %s is not available.' % (time, date)
    else:
        return '%s is not a valid time. Time must be in the format hh:mm.' % time


# Load schedule json
def loadSchedule():
    global schedule
    
    with open('schedule.json') as json_file:
        return json.load(json_file)

    
# save schedule json
def saveSchedule(schedule):
    with open('schedule.json', 'w') as f:
        json.dump(schedule, f)
    

import os.path
# load schedule json
def loadSchedule():
    global schedule
    

    if os.path.exists('schedule.json'):
     with open('schedule.json') as json_file:
        return json.load(json_file)


# get today's date
def todayDate():
    return today.strftime('%m/%d/%y')

# get day of week for a date (or 'today')
def dayOfWeek(date):
    if date == 'today':
        return calendar.day_name[today.weekday()]
    else:
        try:
            theDate = parser.parse(date)
        except:
            return 'invalid date format, please use format: mm/dd/yy'
        
        return calendar.day_name[theDate.weekday()]
    
#########
def define_schema():
 schema = {
    "properties": {
        "mental_illness_title": {"type": "string"},
        "mental_illness_story": {"type": "string"},
        "coping_mechanism": {"type": "string"},
        "support_system": {"type": "string"},
        "triggers": {"type": "string"},      
        "self_care_practices:": {"type": "string"},
        "reflections": {"type": "string"},
    },
    "required": ["mental_illness_title","mental_illness_story","coping_mechanism","support_system","triggers","self_care_practices","reflections"],
 }
 return schema


def extract(chat,content: str, schema: dict):
    prompt = (
        f"Explore and provide detailed insights into all of the following aspects related to mental health. As you provide this information, imagine you are both a compassionate mental health therapist and a empathetic, supportive friend.\n"
        f"1. **Mental Illness Title:** Describe the specific mental health condition or challenge.\n"
        f"2. **Mental Illness Story:** Narrate a detailed and emotive story of someone navigating this mental health condition. Include their emotions, challenges, and moments of resilience.\n"
        f"3. **Coping Mechanism:** Explain the strategies and methods adopted by the individual to cope with their mental health challenges.\n"
        f"4. **Support System:** Identify and elaborate on the crucial individuals, organizations, or resources that contribute to the individual's well-being.\n"
        f"5. **Triggers:** Delve into a nuanced exploration of environmental, emotional, or situational triggers that significantly impact or exacerbate the mental health condition.\n"
        f"6. **Self-Care Practices:** Provide a detailed examination of the daily routines, rituals, and habits that actively contribute to the individual's mental well-being.\n"
        f"7. **Reflections:** acknowledging progress made and lessons learned, offering a holistic perspective on the individual's mental health experiences.\n"
        f"If you are not able to provide specific data for any of the fields, please put in N/A.\n"
        f"Make sure to keep names and any identifying information confidential.\n"
    )
    return create_extraction_chain(schema=schema, llm=chat).invoke(prompt+content)

def parse_retriever_input(params: Dict):
    print(params)
    return params["messages"][-1].content

def create_document_chain(chat, question_answering_prompt):
    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)
    return document_chain

def create_cohere_question_answering_prompt():
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                create_cohere_system_template(),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return question_answering_prompt

def main():
   load_environment_variables()
  
   conn=create_snowflake_conn()
   chat,gpt_chat,cohere_chat,db = initialize_chat_and_db()
    #question = "mental health and therapy"
   global schedule
   schedule = createSchedule()
   retriever,base_retriever= create_retriever(db)
   tools = [
    Tool(
        name = "today_date",
        func = lambda string: todayDate(),
        description="use to get today's date",
        ),
    Tool(
        name = "day_of_week",
        func = lambda string: dayOfWeek(string),
        description="use to get the day of the week, input is 'today' or date using format mm/dd/yy",
        ),
    Tool(
        name = 'available_appointments',
        func = lambda string: getAvailTimes(string),
        description="Use to check on available appointment times for a given date. The input to this tool should be a string in this format mm/dd/yy. This is the only way for you to answer questions about available appointments. This tool will reply with available times for the specified date in 24hour time, for example: 15:00 and 3pm are the same.",
        ),
    Tool(
        name = 'schedule_appointment',
        func = lambda string: scheduleTime(string),
        description="Use to schedule an appointment for a given date and time. The input to this tool should be a comma separated list of 2 strings: date and time in format: mm/dd/yy, hh:mm, convert date and time to these formats. For example, `12/31/23, 10:00` would be the input if for Dec 31'st 2023 at 10am",
        )]
   tools.append(get_retriever_tool(retriever))
   question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
             create_system_template(), 
            
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    )
   cohere_prompt=create_cohere_question_answering_prompt()
   agent = create_openai_tools_agent(chat, tools, question_answering_prompt)
   #agent_executor = AgentExecutor(agent=agent,tools=tools, verbose=True)
   cohere_document_chain = create_document_chain(cohere_chat, cohere_prompt)
   gpt_document_chain = create_document_chain(gpt_chat, cohere_prompt)
   agent_executor = create_conversational_retrieval_agent(chat, tools,system_message=question_answering_prompt,verbose=True)
   
   demo_ephemeral_chat_history= ConversationBufferMemory()
   conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history.chat_memory,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history",
    )
    #display_avatar_image()
   initialize_session_state()
   RAG_GPT_Chat, ChatGPT,RAG_COHERE_CHAT= st.tabs(["RAG GPTChat","ChatGPT", "RAG CohereChat"])
   
   with RAG_GPT_Chat:
    
    
    response_container = st.container()
    
    container = st.container()
    appointment_container = st.container()
    pattern_dr = r'Dr\. [A-Z][a-z]+ [A-Z][a-z]+'
    
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Chat:", placeholder="Talk to Bruno 👉", key='input')
            submit_button = st.form_submit_button(label='Send')
            
            
            if submit_button and user_input:
                st.success(f"User input: {user_input}")
                
              
                st.session_state['chat_history'].chat_memory.add_user_message(user_input)
                gpt_document_chain.invoke(
                    {
                        "messages": st.session_state['chat_history'].chat_memory.messages,
                        "context": retriever.invoke(user_input),
                    }
                )
                print(retriever.invoke(user_input))
                print(base_retriever.invoke(user_input))
                retrieval_chain = RunnablePassthrough.assign(
                    context=parse_retriever_input | retriever,
                ).assign(
                    answer=gpt_document_chain,
                )
                response = retrieval_chain.invoke(
                    {
                        "messages": st.session_state['chat_history'].chat_memory.messages,
                    },
                )
                output = response['answer']
                '''
                response=agent_executor.invoke(
                       {"input": user_input,
                        "chat_history":  st.session_state['chat_history'].chat_memory.messages
                        },
                       
                       {"configurable": {"session_id": "unused"}},
                        )
                
                demo_ephemeral_chat_history.chat_memory.add_user_message("My name is Brinda")
                demo_ephemeral_chat_history.chat_memory.add_ai_message(response['output'])
                print(demo_ephemeral_chat_history.chat_memory.messages)
                '''
                #add both Human and AI messages to the chat history
                st.session_state['chat_history'].chat_memory.add_ai_message(output)
                print(st.session_state['chat_history'].chat_memory.messages)
                #output = response['output']
                
                pattern_dr = r'Dr\. [A-Z][a-z]+ [A-Z][a-z]+'
                pattern_a = r'\d{1,2}:\d{2}'
                pattern_date=r'(\d{2}/\d{2}/\d{2})'

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                
        
    matches=[]
    try:
        matches=re.findall(pattern_dr, st.session_state['generated'][-1])
    except:
        pass
    
    
    with st.sidebar.form(key='appointment_form', clear_on_submit=True):
                  response=st.selectbox("Select a doctor", options=matches)
                  date=""
                  if response:
                      #st.success(f"{response} Doctor selected")
                      #extract available times for doctor and display it as a select box
                      doctor_availability = get_availability(conn,response)
                      #make a list of sets
                      intermed=[[(x,y) for y in doctor_availability[x]] for x in doctor_availability.keys()]
                      final_ops=[]
                      for x in intermed:
                          final_ops.extend(x) 
                      
                      date=st.selectbox("Select a date", options=final_ops)
                  
                  s_button = st.form_submit_button(label='Submit')
                  if s_button:
                      
                      st.session_state.date=convert_to_iso8601(date)
                      st.session_state.response=response
                      try:
                      
                       send_email(st.session_state.email,date,st.session_state.response)
                       st.success(f"Appointment booked for {st.session_state.email}")
                       
                       try:
                           my_logger.info("Getting credentials")
                           creds=get_credentials()
                       except Exception as e:
                            my_logger.error(f"An error occurred: {e}")
                       
                       service = build("calendar", "v3", credentials=creds)
                       # Create a Google Calendar event
                       create_event(service,st.session_state.email,'zenaidemo111@gmail.com',st.session_state.response,st.session_state.date)
                      
                          #insert schema into snowflake
                       schema=define_schema()
                       content=st.session_state["past"]
                       #st.write(type(content))
                       summary=extract(chat,str(content), schema)
                       #st.write(summary)
                          #insert into snowflake
                          
                       insert_into_therapist_view(conn,st.session_state.response,st.session_state.email,summary["text"][0])
                       
                      except Exception as e:
                        st.error(f"Appointment not booked, please try again later. {e}")

    
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                continue


    end_chat_button = st.button("End Chat")


    if end_chat_button:
        # Refresh the app without clearing session state
        memory=st.session_state['chat_history'].chat_memory.messages
        mem_dict=defaultdict(list)
        for i in memory:
            if type(i)==HumanMessage:
                mem_dict["human"].append(i.content)
            else:
                mem_dict["ai"].append(i.content)
        # Write dictionary to JSON file
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Create unique file name with timestamp
        file_name = f"chat_evaluation_{st.session_state.session_id}.json"
        with open(f'C:/Users/abhis/Desktop/AI4MentalHealth/pipelines/chat_evaluation/{file_name}', 'w') as json_file:
              json.dump(mem_dict, json_file)
        st.session_state['past'] = []
        st.session_state['generated'] = []
        # Refresh the app
        st.rerun()

   with ChatGPT:
       #build a normal chatbot with chatgpt
     gpt_response_container = st.container()
    
     gpt_container = st.container()
     with gpt_container:
        with st.form(key='gpt_form', clear_on_submit=True):
            user_input = st.text_input("Chat:", placeholder="Talk to Chatgpt👉", key='gpt')
            gpt_submit_button = st.form_submit_button(label='Send')
            
            
            if gpt_submit_button and user_input:
                st.success(f"User input: {user_input}")
                output=gpt_chat.invoke(
                  [
                   HumanMessage(
                   content=user_input,
                     )
                   ]
                  ).to_json()
                
                
                
                #print(type(output))
                #print(output['kwargs']['content'])
                st.session_state['gpt_past'].append(user_input)
                st.session_state['gpt_generated'].append(output['kwargs']['content'])
     if st.session_state['gpt_generated']:
        with gpt_response_container:
            for i in range(len(st.session_state['gpt_generated'])):
                message(st.session_state["gpt_past"][i], is_user=True, key=str(i) + '_gptuser', avatar_style="big-smile")
                message(st.session_state["gpt_generated"][i], key=str(i)+'_gpt', avatar_style="thumbs")
                continue


     gpt_end_chat_button = st.button("End GPT Chat")


     if gpt_end_chat_button:
        # Refresh the app without clearing session state
        st.session_state['gpt_past'] = []
        st.session_state['gpt_generated'] = []
        # Refresh the app
        st.experimental_rerun()

   with RAG_COHERE_CHAT:
    
    
    response_container = st.container()
    
    container = st.container()
    
    with container:
        with st.form(key='my_cohere_form', clear_on_submit=True):
            user_input = st.text_input("Chat:", placeholder="Talk to Cohere 👉", key='cohere_input')
            submit_button = st.form_submit_button(label='Send')
            
            
            if submit_button and user_input:
                st.success(f"User input: {user_input}")
                st.session_state['cohere_chat_history'].chat_memory.add_user_message(user_input)
                print(st.session_state['cohere_chat_history'].chat_memory.messages)
                cohere_document_chain.invoke(
                    {
                        "messages": st.session_state['cohere_chat_history'].chat_memory.messages,
                        "context": retriever.invoke(user_input),
                    }
                )
                print(retriever.invoke(user_input))
                retrieval_chain = RunnablePassthrough.assign(
                    context=parse_retriever_input | retriever,
                ).assign(
                    answer=cohere_document_chain,
                )
                response = retrieval_chain.invoke(
                    {
                        "messages": st.session_state['cohere_chat_history'].chat_memory.messages,
                    },
                )
                output = response['answer']
              
                #add both Human and AI messages to the chat history
                
                st.session_state['cohere_chat_history'].chat_memory.add_ai_message(output)
                print(st.session_state['cohere_chat_history'].chat_memory.messages)
                
                

                st.session_state['cohere_past'].append(user_input)
                st.session_state['cohere_generated'].append(output)
                

    
    if st.session_state['cohere_generated']:
        with response_container:
            for i in range(len(st.session_state['cohere_generated'])):
                message(st.session_state["cohere_past"][i], is_user=True, key=str(i) + '_cohereuser', avatar_style="big-smile")
                message(st.session_state["cohere_generated"][i], key=str(i)+'_cohere', avatar_style="thumbs")
                continue


    cohere_end_chat_button = st.button("End Cohere Chat")


    if cohere_end_chat_button:
        # Refresh the app without clearing session state
        st.session_state['cohere_past'] = []
        st.session_state['cohere_generated'] = []
        # Refresh the app
        st.experimental_rerun()


if __name__ == "__main__":
    main()
