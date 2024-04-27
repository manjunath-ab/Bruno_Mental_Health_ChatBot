import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.memory import ChatMessageHistory



# Load environment variables from .env file
#dotenv_path = Path('/home/abhi/.env')
dotenv_path = Path('C:/Users/abhis/.env')
load_dotenv(dotenv_path=dotenv_path)
#pipelines/chroma_db journey
chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
db = Chroma(persist_directory="../new_knowledge_db",embedding_function=OpenAIEmbeddings())
#If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

SYSTEM_TEMPLATE = """
Imagine you are a human friend, talk to the user like a friend who understands their problem and keep the reply short.End with a follow up question. 
If the user asks you about therapists then provide details such as the therapist's name, location, and description.
When the user asks to book an appointment, ask about preferences such as location and preferred timings for the appointment.After user input, ask a question to keep the conversation going.
If the user question is not relevant to mental health or therapists details, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""
retriever = db.as_retriever(k=4)
docs=retriever.invoke("how can i battle depression?")
#print(docs)
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)


from langchain.memory import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()


document_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
        "context": docs,
    }
)

from typing import Dict

from langchain_core.runnables import RunnablePassthrough


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)
demo_ephemeral_chat_history.add_user_message("how can I battle depression?")
response = retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    }
)


demo_ephemeral_chat_history.add_ai_message(response["answer"])

#demo_ephemeral_chat_history.add_user_message("tell me more about that!")

response=retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    },
)

# Streamlit app

from streamlit_chat import message
# Set page title and theme to dark

# Change the title formatting to white and blue
st.image("zen.jpg", width=300)

# Upload an image on the sidebar
avatar_image = "avatar.jpg"

# Check if an image file is uploaded
if avatar_image is not None:
    # Display the uploaded image on the sidebar
    st.sidebar.image(avatar_image, use_column_width=True)
    st.sidebar.markdown("""
    <div style="font-family: 'Arial', sans-serif; font-size: 20px; font-style: italic;">
        "I am Zenny! I'm here to be your virtual friend, to chat with you, and to help you find the support and resources you need. Whether you're feeling down and need someone to talk to, or you're looking for information on mental health and therapy, I'm here to listen and assist. So, let's chat and find the help you need!"
    </div>
""", unsafe_allow_html=True)
# Initialize chat history
if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! I'm Zenny.Ask me about therapy, mental health, or anything you want to talk about "]

if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey"]
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Chat:", placeholder="Talk to ZEN.AI 👉", key='input')
        submit_button = st.form_submit_button(label='Send')
if submit_button and user_input:

    demo_ephemeral_chat_history.add_user_message(user_input)

    document_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
        "context": retriever.invoke(user_input),
    }
    )

    print(retriever.invoke(user_input))
    retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
     ).assign(
    answer=document_chain,
    )

    

    response=retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    },
    )

    
    output = response['answer']

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
          message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
    
          message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
          continue



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
