# https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/gpt-with-vision?tabs=python

import base64
from mimetypes import guess_type
from openai import AzureOpenAI
from credentials import *

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Example usage
image_path = '/home/nomuram/rag-from-scratch/figures/figure-3-1.jpg'


# data_url = local_image_to_data_url(image_path)

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,  
    api_version=OPENAI_API_VERSION,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment_name}",
    # base_url=AZURE_OPENAI_ENDPOINT
)


def generate_summary(image_path, azure_client=azure_client, language='English'):
    data_url = local_image_to_data_url(image_path)
    msg=[
            { "role": "system", "content": "You are a helpful assistant." if language == 'English' else "あなたは優秀なアシスタントです。" },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": "Describe this picture:" if language == 'English' else "この写真の内容を説明してください。" 
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ] } 
        ]
    response = azure_client.chat.completions.create(
        model=deployment_name,
        messages=msg,
        max_tokens=1000 #2000 
    )

    # return response
    return response.to_dict()