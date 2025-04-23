import os
from dotenv import load_dotenv
from twelvelabs import TwelveLabs

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = "tlk_1G36X5Q1KS4J5B26BPP8H2WJ2BHR"

if not api_key:
    raise ValueError("TWELVELABS_API_KEY not found in environment variables.")

client = TwelveLabs(api_key=api_key)

index = client.index.create(
    name="k;qefpwpqjofopqpowfoqwfjiqwf",
    models=[
        { "name": "marengo2.7", "options": ["visual", "audio"] }
    ]
)

print(f"Index created! ID: {index.id}, Name: {index.name}, Models: {index.models}")