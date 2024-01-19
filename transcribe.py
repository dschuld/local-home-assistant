import whisper
import os
import time
from models import gpt_4_turbo, mistral_7b
import requests
import argparse


audio_folder = "/home/david/projects/local-home-assistant/audio/"
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
SHOPPING_LIST_ID = os.environ.get("SHOPPING_LIST_ID")


NOTION_BLOCK_ENDPOINT = "https://api.notion.com/v1/blocks/" + SHOPPING_LIST_ID + "/children"

model_functions = {
    'gpt4': gpt_4_turbo,
    'mistral': mistral_7b,
}

# Create the parser
parser = argparse.ArgumentParser(description='Choose a model to run.')





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
    # Parse the arguments
    args = parser.parse_args()
    print(f"Sending to {args.model}: " + text)
    call = model_functions[args.model](text)
    if (call["name"] == "add_item_to_shopping_list"):
        add_item_to_shopping_list(call["parameters"]['items'])
    elif(call.name == "trigger_alert"):
        print("Triggering alert with severity: " + call["parameters"]['severity'])
    else:
        print("Unknown function call: " + call["name"])  

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
                    os.remove(filepath)
                    text = result["text"].replace(",", "").replace(".", "").replace(":", "").replace(";", "")
                    text = " ".join(text.split()).lower()
                    if "okay computer" in text:
                        print("Found 'okay computer' in text. Sending to API.")
                        send_to_api(text.replace("okay computer", "").strip())
                    else:
                        print("No command:" + text)
            
            # Sleep for a certain period before checking again
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped monitoring audio folder.")








if __name__ == "__main__":
    # Add the arguments
    parser.add_argument('--model', type=str, default='gpt4', choices=model_functions.keys(),
                    help='The model to run. Default is gpt4.')

    transcribe()   




