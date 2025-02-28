import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
load_dotenv()

# ----------------------------------------
# 1. Initialize Azure OpenAI Client
# ----------------------------------------
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_KEY = os.getenv("AOAI_KEY")
client = AzureOpenAI(  
    azure_endpoint=os.getenv('AOAI_ENDPOINT'),  
    api_key=os.getenv('AOAI_KEY'),  
    api_version="2024-05-01-preview",
)

# ----------------------------------------
# 2. Load system prompts from file
# ----------------------------------------
with open('extract_item_info_sys.prompt', 'r') as file:
    EXTRACT_ITM_SYS_PROMPT = file.read()
with open('/compare_items_sys.prompt', 'r') as file:
    COMPARE_ITM_SYS_PROMPT = file.read()

# ----------------------------------------
# 3. Load webpages and convert into markdown
# ----------------------------------------
urls = ["https://www.justnsnparts.com/rfq/3m-purification-inc/4330012137798/12838-21-50-0015/", "https://www.ebay.com/itm/312574229877"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs) # can filter out markdown tags: 
# https://python.langchain.com/api_reference/community/document_transformers/langchain_community.document_transformers.html2text.Html2TextTransformer.html


# ----------------------------------------
# 4. Define LLM calls to extract item info
# ----------------------------------------
def get_item_info(url, doc):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": EXTRACT_ITM_SYS_PROMPT
                }
            ]
        }
    ]
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The following information is extracted from the website: " + url + ":\n" + doc.page_content
                }
            ]
        }
    )  
    completion = client.chat.completions.create(  
        model=os.getenv("AOAI_MODEL"),
        messages=messages,
        max_tokens=800,  
        temperature=0.7,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False,
        response_format={ "type": "json_object" }
    )
    response = completion.choices[0].message.content
    return response
def get_llm_response(item_infos, user_query):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": COMPARE_ITM_SYS_PROMPT
                }
            ]
        }
    ]
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The user asks: " + user_query + "The following information is extracted: " + json.dumps(item_infos)
                }
            ]
        }
    ) 
    completion = client.chat.completions.create(  
        model=os.getenv("AOAI_MODEL"),
        messages=messages,
        max_tokens=800,  
        temperature=0.7,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False,
    )
    # print the model response
    response = completion.choices[0].message.content
    return response

# ----------------------------------------
# 5. Extract item info from each website and store in a list
# ----------------------------------------
item_infos = []
for url, doc in zip(urls, docs_transformed):
    item_info = get_item_info(url, doc)
    item_infos.append(item_info)
    print("Item info extracted from:", url)
    print(item_info)
    print("\n")

# ----------------------------------------
# 6. Summarize item info from each extracted item and form final output response
# ----------------------------------------
user_query = "FILTER-STRAINER, STEEL ELEMENT, MODEL EGS, 5' LONG, 0.0015' SPACING  AMF CUNO INC  12838-21-50-0015"
llm_response = get_llm_response(item_infos, user_query)
print("------------------------")
print("LLM Response:")
print(llm_response)