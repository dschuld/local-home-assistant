# models.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from openai import OpenAI

client = OpenAI()
import json
import codecs
import os

functions = [
        {
            "name": "add_item_to_shopping_list",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "The items to add to the shopping list",
                        "items": {
                            "type": "string",
                            "description": "A single item to add to the shopping list",
                        },
                    },
                },
                "required": ["items"],
            },
        },
        {
            "name": "trigger_alert",
            "description": "Trigger an alert based on the given severity",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["red", "yellow", "black"],
                        "description": "The items to add to the shopping list"
                    },
                },
                "required": ["severity"],
            },
        }
    ]

def gpt_4_turbo(text):
    messages = [{"role": "user", "content": text}]
    response = client.chat.completions.create(model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="auto")
    response_message = response.choices[0].message    
    if (response_message.function_call):
        return {
            "name": response_message.function_call.name,
            "parameters": json.loads(response_message.function_call.arguments)
        }  
    else:
        print("Did not find a function call.")  

def mistral_7b(text):

    os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.environ['HUGGING_FACE_API_KEY']
    functions_json = json.dumps(functions).replace("{", "{{").replace("}", "}}")
    

    template = """
    <s> [INST]
     You are an assistant that analyses a command in human language and helps extract data for an function call. Based on the text you received, you choose one of the following function calls and return data in JSON format. you are strictly not allowed to output any text that is not in JSON format.                   

The function calls are:

""" + functions_json + """
        
Example:

User Input: Hi, please add milk, rice and butter to the shopping list

you will respond with this JSON:

{{
    "name": "add_item_to_shopping_list",
    "parameters": {{
        "items": ["milk", "rice", "butter"]
    }}
    
}}
                   
                   
Response ONLY with JSON. Do not write any additional text. In case you cannot match the command to a function call, respond with: 
{{ 
    "name": "None", "parameters": [] 
}}
    . Make sure to output only JSON. 
Once again, you are strictly not allowed to output any text that is not in JSON format. 
User command:
{question} [/INST] Model answer</s>


    """

    prompt = PromptTemplate(template = template, input_variables = ["question"])

    llm_chain = LLMChain(
        prompt = prompt,
        llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        model_kwargs={
                            "temperature": 0.1,
                            "max_length": 512})
    )


    response = llm_chain.run(text).strip().replace("\\", "")
    response = codecs.decode(response, 'unicode_escape')
    response_object = json.loads(response)
    return response_object

class WhisperLocal:
    pass  # Replace with your implementation

class WhisperHF:
    pass  # Replace with your implementation