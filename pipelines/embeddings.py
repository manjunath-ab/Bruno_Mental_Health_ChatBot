import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import snowflake.connector
import sqlalchemy
#from embeddings import embeddings
import pinecone
import numpy as np
from python_to_snowflake import create_snowflake_conn


# Specify the columns you want to select
columns_to_select = [
    'SUPPORT_SYSTEM',
    'SELF_CARE_PRACTICES',
    'MENTAL_ILLNESS_STORY',
    'COPING_MECHANISM',
    'MENTAL_ILLNESS_TITLE',
    'TRIGGERS',
    'REFLECTIONS'
]

# Construct the SQL query
columns_string = ', '.join(columns_to_select)
query = f"SELECT {columns_string} FROM CHATBOT_KNOWLEDGE LIMIT 5"

conn = create_snowflake_conn() 

# Extract data from Snowflake
snowflake_data = pd.read_sql_query(query, conn)

# Concatenate text data from all selected columns
text_data = snowflake_data.apply(lambda x: ' '.join(x.dropna()), axis=1).tolist()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and embed text data using BERT
embeddings = []
for text in text_data:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pooled_output = outputs.pooler_output  # You can use other outputs such as last hidden states if needed
    embeddings.append(pooled_output.numpy())

# Generate unique IDs for each embedding
ids = range(len(embeddings))

# Create a list of items with IDs and embeddings
items = [{'id': str(i), 'values': embedding.tolist()} for i, embedding in zip(ids, embeddings)]

pc = Pinecone(api_key="99162b48-9be5-4cac-b0a7-39c60e3218cc")

index = pc.Index("chatbot-knowledgebase")
# Name of the Pinecone index
index_name = "chatbot-knowledgebase"

# Connect to the Pinecone index (create if it doesn't exist)
index = pc.Index(index_name)

# Select the first 100 items from the list
items_to_insert = items[:5]


# Upsert items into the Pinecone index
index.upsert(vectors=items_to_insert)

print("First 100 items inserted into the Pinecone index successfully.")
