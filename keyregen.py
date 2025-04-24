import os
from dotenv import load_dotenv
from twelvelabs import TwelveLabs

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = ""

if not api_key:
    raise ValueError("TWELVELABS_API_KEY not found in environment variables.")

client = TwelveLabs(api_key=api_key)

index = client.index.create(
    name="k;sadasdasdasdasd",
    models=[
        { "name": "marengo2.7", "options": ["visual", "audio"] }
    ]
)

print(f"Index created! ID: {index.id}, Name: {index.name}, Models: {index.models}")
