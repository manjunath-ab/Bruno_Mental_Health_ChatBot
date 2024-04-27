"""
author:manjunath-ab
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.chains import create_extraction_chain
from langchain.prompts import PromptTemplate
import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time
import random

chrome_options = Options()
chrome_options.add_argument("--headless")

dotenv_path = Path('c:/Users/abhis/.env')
load_dotenv(dotenv_path=dotenv_path)



# disable the AutomationControlled feature of Blink rendering engine
chrome_options.add_argument('--disable-blink-features=AutomationControlled')

# disable pop-up blocking
chrome_options.add_argument('--disable-popup-blocking')

# start the browser window in maximized mode
chrome_options.add_argument('--start-maximized')

# disable extensions
chrome_options.add_argument('--disable-extensions')

# disable sandbox mode
chrome_options.add_argument('--no-sandbox')
# Allow all cookies, including third-party cookies
chrome_options.add_argument("--disable-web-security")
chrome_options.add_argument("--allow-running-insecure-content")
# disable shared memory usage
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)



# Create an instance of the ChatOpenAI class
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-0125')


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


def extract(content: str, schema: dict):
    prompt = (
        f"Explore and provide detailed insights into all of the following aspects related to bipolar disorder and its various manifestations. As you provide this information, imagine you are both a compassionate mental health therapist and an empathetic, supportive friend.\n"
        f"1. **Mental Illness Title:** Describe the specific type of bipolar disorder or give it a name based on the individual's experiences narrated in the Mental Illness Story.\n"
        f"2. **Mental Illness Story:** Narrate a detailed and emotive story of someone navigating this subtype of bipolar disorder or related condition. Include their emotions, challenges, and moments of resilience.\n"
        f"3. **Coping Mechanism:** Explain the strategies and methods adopted by the individual to cope with the challenges posed by this subtype of bipolar disorder or related condition.\n"
        f"4. **Support System:** Identify and elaborate on the crucial individuals, organizations, or resources that contribute to the individual's well-being while managing this subtype of bipolar disorder or related condition.\n"
        f"5. **Triggers:** Delve into a nuanced exploration of environmental, emotional, or situational triggers that significantly impact or exacerbate the symptoms of this subtype of bipolar disorder or related condition.\n"
        f"6. **Self-Care Practices:** Provide a detailed examination of the daily routines, rituals, and habits that actively contribute to the individual's mental well-being while living with this subtype of bipolar disorder or related condition.\n"
        f"7. **Reflections:** Acknowledge progress made and lessons learned, offering a holistic perspective on the individual's experiences with this subtype of bipolar disorder or related condition.\n"
        f"If specific data is not available for any of the fields, please add content more closely associated with the mental illness, creating a comprehensive and insightful narrative.\n"
        f"As you navigate this exploration, envision yourself peeling back layers to reveal a profound understanding of the diverse triggers impacting the individual's mental health while managing this subtype of bipolar disorder or related condition."
    )
    return create_extraction_chain(schema=schema, llm=llm).invoke(prompt+content)

def batching_dataframes():
   pass

#url is a list of urls batched for the sake of OPENAI API rate limits
def process_url(url, schema):
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["p"]
    )

    print("Extracting content with LLM for the URL: ", url)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=350
    )
    splits = splitter.split_documents(docs_transformed)

    try:
        extracted_content = extract(schema=schema, content=splits[0].page_content)
    except Exception as e:
        print(e)
        return None

    #extracted_content['text']['article'] = url
    pprint.pprint(extracted_content)

    # Define the expected schema with column names and their corresponding data types
    schema_dict = {
    "mental_illness_title": str,
    "mental_illness_story": str,
    "coping_mechanism": str,
    "support_system": str,
    "triggers": str,
    "self_care_practices": str,
    "reflections": str
    }

    
    if extracted_content['text'] is list:
        df=pd.DataFrame(extracted_content['text'][0])
    else:
        df=pd.DataFrame(extracted_content['text'])
    # Check if all expected columns are present in the DataFrame
    missing_columns = set(schema_dict.keys()) - set(df.columns)
    # Add missing columns with NaN values
    for column in missing_columns:
      df[column] = None
    
    return df


def html_scrape(urls, schema):
    df_list = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for i,url in enumerate(urls):
            print(f"Processing URL set {i+1} of {len(urls)}")
            # Submit each URL for processing in the thread pool
            future = executor.submit(process_url, url, schema)
            futures.append(future)

            # If we've processed a batch of 3 URL sets, wait for a minute
            if len(futures) == 3:
                for completed_future in futures:
                    result_df = completed_future.result()
                    if result_df is not None:
                        df_list.append(result_df)

                # Clear the futures list for the next batch
                print("Clearing the futures list")
                futures.clear()

                # Wait for a minute 20 before processing the next batch
                print("Waiting for 70 seconds before processing the next batch of URLs...")
                time.sleep(70)
            time.sleep(5)

        # Wait for any remaining threads to finish
        for future in futures:
            result_df = future.result()
            if result_df is not None:
                df_list.append(result_df)

    return df_list

def scrape_with_playwright(urls, schema):
  df_list=[]
  #batch the urls into groups of 10
  for i in range(0,20,10): #len(urls)
    loader = AsyncChromiumLoader(urls[i:i+10])
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["p"]
    )
    #print(docs_transformed)
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)
    #print(f'content being used is {splits[0]}')
    # Process the first split
    """
    Need to wrap the extract function in a try except block to handle the case where the content is empty
    """
    try:
        extracted_content = extract(schema=schema, content=splits[0].page_content)
        #pprint.pprint(extracted_content)
    except:
        print('content is empty')
        continue

    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)

    #convert into a dataframe
    df = pd.DataFrame(extracted_content)
    df_list.append(df)
  return df_list

def initial_fetch(url_thread):
   driver = webdriver.Chrome(options=chrome_options)
   driver.get(url_thread)
   # Find all elements with the class "read-link"
   tag_elements = driver.find_elements(By.CLASS_NAME, 'read-more-button')
   # Extract href attribute from each element and store in a list
   href_list = [tag.get_attribute("href") for tag in tag_elements]
   driver.quit()
   return href_list



def threaded_url_list_pull(base_url, num_threads=5):
    print('starting the threaded url list pull')
    url_list = []
    i = 1

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a list to hold the futures
        futures = []

        while True:
            url_thread = base_url + str(i)
            # Submit the task to the thread pool and store the future
            future = executor.submit(initial_fetch, url_thread)
            futures.append(future)

            if i > 16:
                break

            i += 1
            print(i)

        # Wait for all threads to complete before moving on
        for completed_future in as_completed(futures):
            url_list.extend(completed_future.result())

    return url_list


def button_sequence_flow(base_url):
   url_list=[]
   driver = webdriver.Chrome() #options=chrome_options
   driver.get(base_url)
   while True:
 
    link_elements = driver.find_elements(By.CLASS_NAME, 'read-link')
    # Extract href attribute from each element and store in a list
    url_list.extend([link.get_attribute("href") for link in link_elements])
    print(url_list)

    try:
        # Look for the next button and click it
        next_button = driver.find_elements(By.CLASS_NAME,'next')
        next_button.click()
    except Exception:
        # If the next button is not found, break out of the loop
        break

# Close the webdriver
   driver.quit()

def batch_url_list(url_list):
    batch_size = random.randint(3, 20)
    return [url_list[i:i+batch_size] for i in range(0, len(url_list), batch_size)]

def main():
    base_url="https://natashatracy.com/topic/bipolar-blog/"
    url_list=list(set(threaded_url_list_pull(base_url)))
    #put in a check to not repeat the same url for future runs
    print('completed the url list')
    print(url_list,len(url_list))
    schema = define_schema()
    url_list=batch_url_list(url_list)
    start_time = time.time()
    df_list = html_scrape(url_list, schema=schema)
    end_time = time.time()
    print("Time taken to process all URLs: ", end_time - start_time)
    result=pd.concat(df_list, ignore_index=True)
    result.dropna(how='all',inplace=True)
    result.to_csv('bipolar_diaries.csv',index=False)

if __name__=='__main__':
 main()