import whisper
import os
import time
import openai
import json
import requests


audio_folder = "/home/david/projects/local-home-assistant/audio/"
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
SHOPPING_LIST_ID = os.environ.get("SHOPPING_LIST_ID")


NOTION_BLOCK_ENDPOINT = "https://api.notion.com/v1/blocks/" + SHOPPING_LIST_ID + "/children"


def body(item):
    return """
    {
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "__ITEM__",
                                "link": null
                            }
                        }
                    ]
                }
            }
        ]
    }
    """.replace("__ITEM__", item)



def add_item_to_shopping_list(items):
    print("Adding item(s) to shopping list: " + ", ".join(items))
    if items:
        for item in items:
            payload = body(item)
            headers = {
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28",
                "Authorization": "Bearer " + NOTION_API_KEY
            }
            response = requests.patch(NOTION_BLOCK_ENDPOINT, data=payload, headers=headers)
            if response.status_code == 200:
                print(f"Item '{item}' added to Notion shopping list.")
            else:
                print(f"Failed to add item '{item}' to Notion shopping list.")
    else:
        print("No items to add to the shopping list.")
    


def send_to_api(text):
    print("Sending to API: " + text)
    messages = [{"role": "user", "content": text}]
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
                "required": ["item"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]    
    if (response_message.function_call) and (response_message.function_call.name == "add_item_to_shopping_list"):
        arguments = json.loads(response_message.function_call.arguments)
        add_item_to_shopping_list(arguments['items'])
    else:
        print("Did not find a function call.")    

def transcribe():
    print("Started monitoring audio folder.")
    model = whisper.load_model("base")
    # Keep the script running
    try:
        while True:
            for filename in os.listdir(audio_folder):
                print("Found audio file: " + filename)
                if filename.endswith(".wav"):
                    print("Transcribing audio file: " + filename)
                    filepath = os.path.join(audio_folder, filename)
                    result = model.transcribe(filepath)
                    text = result["text"].replace(",", "").replace(".", "").replace(":", "").replace(";", "")
                    text = " ".join(text.split()).lower()
                    if "okay computer" in text:
                        print("Found 'okay computer' in text. Sending to API.")
                        send_to_api(text.replace("hello house", "").strip())
                    else:
                        print("No command:" + text)



                    # Optionally, you can delete the transcribed file
                    os.remove(filepath)
            
            # Sleep for a certain period before checking again
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped monitoring audio folder.")



if __name__ == "__main__":
    transcribe()     




