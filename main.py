## Image to Text Using AI
from google import genai
from google.genai import types
import os
import requests

from dotenv import load_dotenv
load_dotenv()

## Python Path: ~/csis333/project/venv/bin/python
## Pip Path: ~/csis333/project/venv/bin/pip

askAi = "What is this image? Just say 1 word (the object or whatever it is)"

client = genai.Client(api_key=os.getenv("GemApiKey"))

# Input: image_bytes
# Ouput: Text of the Image.
def sendImageToAI(image_bytes):
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            askAi, # Text Prompt
            types.Part.from_bytes(
                data=image_bytes, mime_type="image/jpeg" # Image
            )
        ]
    )
    return response.text

# Image Url To image_bytes
def imageUrlToBtyes(image_path):
    return requests.get(image_path).content

# Image From Computer To image_bytes
def imagePathToBytes(image_path):
    with open(image_path, 'rb') as f:
        return f.read()
    
def main():
    # Picure of a red apple
    image = imagePathToBytes("./exampleImage.jpg")

    # Ai's Description of the image
    aiImageDescription = sendImageToAI(image)

    print(aiImageDescription)

main()