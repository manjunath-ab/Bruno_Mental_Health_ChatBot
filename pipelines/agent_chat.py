import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import tool
from langchain.tools.base import Tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool

# Load environment variables from .env file
#dotenv_path = Path('/home/abhi/.env')
dotenv_path = Path('C:/Users/abhis/.env')
load_dotenv(dotenv_path=dotenv_path)

chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
db = Chroma(persist_directory="../knowledge_db",embedding_function=OpenAIEmbeddings())


retriever= db.as_retriever(k=4)
from langchain.tools.retriever import create_retriever_tool

def create_system_template():
    SYSTEM_TEMPLATE = """
    Imagine you are a human friend, talk to the user like a friend who understands their problem and keep the reply short.End with a follow up question. 
    If the user asks you about therapists then provide details such as the therapist's name, location, and description.
    When the user asks to book an appointment, ask about preferences such as location and preferred timings for the appointment.After user input, ask a question to keep the conversation going.
    If the user question is not relevant to mental health or therapists details, don't make something up and just say "I don't know":

    <context>
    
    </context>
    """
    return SYSTEM_TEMPLATE
tool = create_retriever_tool(
    retriever,
    "mental_health_and_therapist_knowledge_base",
    "This is a retreiver tool for therapist details and mental health journies,coping mechanisms,triggers, self care rountines and more",
)


tools = [tool]
"""
agent chat
"""




#get_word_length.invoke("abc")

llm_with_tools = chat.bind_tools(tools)
prompt = ChatPromptTemplate.from_messages(
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


agent = create_openai_tools_agent(chat, tools, prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools, verbose=True)



#agent_executor.invoke({"messages": [HumanMessage(content="suggest some therapists in Boston?")]})

demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history",
)


# Streamlit app

"""
streamlit app
"""
from streamlit_chat import message
# Set page title and theme to dark

st.title("Zen AI Chat")
# Initialize chat history
if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me about therapy, mental health, or anything you want to talk about "]

if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey"]
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Chat:", placeholder="Talk to ZEN.AI 👉 (:", key='input')
        submit_button = st.form_submit_button(label='Send')
if submit_button and user_input:
    
    response=conversational_agent_executor.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "unused"}},
    )
    output = response['output']

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
          message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
    
          message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
          continue


response=conversational_agent_executor.invoke(
        {"input": "tell me something else"},
        {"configurable": {"session_id": "unused"}},
    )

# Create a checkbox to determine if the chat should end
end_chat_checkbox = st.checkbox("I entered all the details.")

end_chat_button = st.button("End Chat")

if end_chat_button and end_chat_checkbox:

    #Extract Process to snowflake

    st.success("An appointment will be scheduled for you.")

    # Clear session state
    st.session_state['past'] = []
    st.session_state['generated'] = []

    # Refresh the app
    
elif end_chat_button:
    # Refresh the app without clearing session state
    st.session_state['past'] = []
    st.session_state['generated'] = []

    # Refresh the app
    st.experimental_rerun()
