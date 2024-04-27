# AI4MentalHealth 💚

## Live Application Links 🌐

[![Presentation Link](https://img.shields.io/badge/Presentation_Link-808080?style=for-the-badge&logo=Google&logoColor=white)](https://docs.google.com/presentation/d/1MOpAVT97rVr3FBrZUoh3YzQ-wVk_Bs2A-gLwHbBo8go/edit?usp=sharing)

[![Demo Link](https://img.shields.io/badge/Demo_Link-808080?style=for-the-badge&logo=Loom&logoColor=white)](https://www.loom.com/share/dc6108d1c309493d9af014286ca3420a?sid=607bc134-3821-4eb7-b910-644675122170)

## Meet Bruno

![image](https://github.com/manjunath-ab/AI4MentalHealth/assets/114537365/07fe09b9-f688-4bfd-b893-9b139d48ecee)

Bruno is our friendly chatbot, designed to provide a supportive and non-judgmental space for users seeking mental health support and information.

## Architecture

![image](https://github.com/manjunath-ab/AI4MentalHealth/assets/114537365/c7f7a356-4a6b-42ae-9244-a51c7846a1f8)

The above diagram illustrates the architecture of the Mental Health Chatbot, showcasing the various components and technologies used in the project.

## Use Case 📖

The use case for this project is to develop a mental health platform that leverages conversation AI based on Retrieval-augmented generation (RAG) to offer users a supportive and non-judgmental space, fostering emotional well-being and personal introspection. 🧠

## Technologies Used 🛠️

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![dbt](https://img.shields.io/badge/dbt-FF694B?style=for-the-badge&logo=dbt&logoColor=white)](https://www.getdbt.com/)
[![LangChain](https://img.shields.io/badge/LangChain-FF9900?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Dagster](https://img.shields.io/badge/Dagster-F23FC5?style=for-the-badge&logo=dagster&logoColor=white)](https://dagster.io/)
[![Cohere](https://img.shields.io/badge/Cohere-000000?style=for-the-badge&logo=cohere&logoColor=white)](https://cohere.ai/)
[![Snowflake](https://img.shields.io/badge/Snowflake-0093F1?style=for-the-badge&logo=snowflake&logoColor=white)](https://www.snowflake.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-7289DA?style=for-the-badge&logo=chromadb&logoColor=white)](https://www.chromadb.com/)
[![DeepEval](https://img.shields.io/badge/DeepEval-000000?style=for-the-badge&logo=deepeval&logoColor=white)](https://deepeval.com/)
[![Google Calendar](https://img.shields.io/badge/Google_Calendar-4285F4?style=for-the-badge&logo=google-calendar&logoColor=white)](https://calendar.google.com/)

## Steps

1. **Knowledge Base Preparation**: The chatbot's knowledge base is populated with mental health-related information from various sources, such as blogs and websites. This data is scraped, cleaned, and processed using techniques like OpenAI's extraction chain.

2. **Embedding and Storage**: The processed knowledge chunks are embedded using OpenAI's text embedding model, and the resulting vectors are stored in a ChromaDB vector database for efficient similarity search.

3. **User Input and Knowledge Retrieval**: When a user enters a message in the chatbot interface, the message is embedded using the same OpenAI embedding model. The chatbot then queries the ChromaDB database to retrieve the most relevant knowledge chunks based on vector similarity.

4. **Reranking**: The retrieved knowledge chunks are reranked using the Cohere reranker to improve their semantic relevance to the user's query.

5. **Language Generation**: The reranked knowledge chunks and the user's input are passed to the OpenAI GPT-3.5 language model, which generates a contextual and empathetic response.

6. **Conversation Management**: The generated response, along with the user's input, is added to the conversation history managed by LangChain's memory module. This conversation history is used to provide context for future responses.

7. **Appointment Scheduling**: If the user requests to schedule an appointment, the chatbot prompts the user for available dates and times. Once confirmed, the chatbot creates a calendar event using the Google Calendar API and sends appointment details via email.

8. **Data Storage**: User chat history and relevant information, including scheduled appointments, are stored in a Snowflake database for future reference and analysis.

9. **Evaluation**: The chatbot's responses are evaluated using DeepEval from ConfidentAI, which calculates Hallucination, Bias, Answer Relevancy, and Toxicity scores. This helps assess the quality and safety of the chatbot's outputs.

10. **Dashboard**: Business Analytics dashboard is built which gives the admin a wholesome view of the number of appoitnments made, therapist availability, number of specialisations, etc.

## Project Tree 

```
📦 
├─ .gitignore
├─ LICENSE
├─ README.md
├─ analytics_engineering
│  └─ dbt
│     ├─ .gitignore
│     ├─ .gitkeep
│     ├─ README.md
│     ├─ analyses
│     │  └─ .gitkeep
│     ├─ dbt_project.yml
│     ├─ macros
│     │  └─ .gitkeep
│     ├─ models
│     │  ├─ chatbot
│     │  │  ├─ knowledge_base.sql
│     │  │  ├─ schema.yml
│     │  │  ├─ transform_blurt.sql
│     │  │  ├─ transform_chipur.sql
│     │  │  └─ transform_nat.sql
│     │  ├─ example
│     │  │  ├─ my_first_dbt_model.sql
│     │  │  ├─ my_second_dbt_model.sql
│     │  │  └─ schema.yml
│     │  └─ staging
│     │     ├─ schema.yml
│     │     └─ stg_chatbot_knowledge.sql
│     ├─ seeds
│     │  └─ .gitkeep
│     └─ snapshots
│        └─ .gitkeep
├─ chat_eval.json
├─ dagster_configuration
│  └─ dagster_ai4mentalhealth
│     ├─ dagster_ai4mentalhealth
│     │  ├─ __init__.py
│     │  ├─ blurt.py
│     │  ├─ chipur.py
│     │  ├─ nat.py
│     │  └─ python_to_snowflake.py
│     ├─ dagster_ai4mentalhealth_tests
│     │  ├─ __init__.py
│     │  └─ test_assets.py
│     ├─ pyproject.toml
│     ├─ setup.cfg
│     ├─ setup.py
│     ├─ tmp1oiwfd6j
│     │  ├─ history
│     │  │  ├─ runs.db
│     │  │  └─ runs
│     │  │     ├─ 315d980b-5792-4129-8386-b564cc6a8a96.db
│     │  │     └─ index.db
│     │  ├─ schedules
│     │  │  └─ schedules.db
│     │  └─ storage
│     │     ├─ 315d980b-5792-4129-8386-b564cc6a8a96
│     │     │  └─ compute_logs
│     │     ├─ create_df
│     │     ├─ create_snowflake_conn
│     │     ├─ define_schema
│     │     ├─ extracted_url_list
│     │     ├─ html_scrape
│     │     ├─ publish_to_snowflake
│     │     └─ threaded_url_list_pull
│     ├─ tmp5zgffo06
│     │  ├─ schedules
│     │  │  └─ schedules.db
│     │  └─ storage
│     │     ├─ 71f98b34-2b0c-4a64-b7d9-fec9ceca0704
│     │     │  └─ compute_logs
│     │     ├─ c_create_df
│     │     ├─ c_define_schema
│     │     ├─ c_extracted_url_list
│     │     ├─ c_html_scrape
│     │     ├─ c_publish_to_snowflake
│     │     └─ c_threaded_url_list_pull
│     └─ tmppterspc2
│        ├─ schedules
│        │  └─ schedules.db
│        └─ storage
│           ├─ 2cbedd3a-ed51-455b-b972-da90485c60ca
│           │  └─ compute_logs
│           ├─ b29d93e0-f0b2-4aec-a62c-bc55dafe1fc5
│           │  └─ compute_logs
│           ├─ c_create_df
│           ├─ c_define_schema
│           ├─ c_extracted_url_list
│           ├─ c_html_scrape
│           ├─ c_publish_to_snowflake
│           ├─ c_threaded_url_list_pull
│           ├─ n_create_df
│           ├─ n_define_schema
│           ├─ n_extracted_url_list
│           ├─ n_html_scrape
│           ├─ n_publish_to_snowflake
│           └─ n_threaded_url_list_pull
├─ new_knowledge_db
│  ├─ 89207a28-d694-4681-b72d-a6a21b882f04
│  └─ chroma.sqlite3
├─ pipelines
│  ├─ CREDENTIALS
│  │  ├─ credentials.json
│  │  └─ token.json
│  ├─ README.md
│  ├─ __pycache__
│  │  └─ python_to_snowflake.cpython-311.pyc
│  ├─ agent_chat.py
│  ├─ agent_chat_prod.py
│  ├─ ai4mh.jpeg
│  ├─ bipolar.py
│  ├─ blurt_etl.py
│  ├─ calendar_api.py
│  ├─ chatbot.py
│  ├─ chatbot_dev.py
│  ├─ cohere_ranker.py
│  ├─ create_event.py
│  ├─ dashboard.py
│  ├─ dog2.jpeg
│  ├─ email_patient.py
│  ├─ embeddings.py
│  ├─ home.py
│  ├─ login.py
│  ├─ python_to_snowflake.py
│  ├─ scheduler.py
│  ├─ seperator.py
│  ├─ signup.py
│  ├─ snowflake-embedded.py
│  ├─ snowflake_embedding.py
│  ├─ snowflake_integrator.py
│  ├─ staging_files
│  │  ├─ .DS_Store
│  │  ├─ CREDENTIALS
│  │  │  ├─ credentials.json
│  │  │  └─ token.json
│  │  ├─ calendar_api.py
│  │  ├─ create_event.py
│  │  ├─ test.py
│  │  └─ token.json
│  ├─ therapy.py
│  ├─ token.json
│  └─ weekdate_converter.py
├─ schedule.json
├─ test_1.py
└─ test_evaluation.py
```
©generated by [Project Tree Generator](https://woochanleee.github.io/project-tree-generator)


## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/mental-health-chat.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the necessary environment variables (e.g., OpenAI API keys, Google Calendar API key, Snowflake database credentials, Cohere API key)

4. Run the Streamlit app:
``` bash
streamlit run app.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

