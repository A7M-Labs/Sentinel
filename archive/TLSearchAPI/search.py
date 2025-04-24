from twelvelabs import TwelveLabs
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get environment variables
client = TwelveLabs(api_key=os.getenv('TWELVELABS_API_KEY'))

video_dict={

    "normal1.mp4":"68088a3c352908d3bc50a428",
    "normal2.mp4":"680897cc352908d3bc50a442",
    "normal3.mp4":"68089816669d2e9f3f513bb2",
    "normal4.mp4":"6808a21a352908d3bc50a45b",
    "normal5.mp4":"6808a26c02327bef162a41a8",

    "rob1.mp4":"6808a2bc02327bef162a41ad",
    "rob2.mp4":"6808a2f9045da61d81570784",
    "rob3.mp4":"6808a33702327bef162a41af",
    "rob4.mp4":"6808a37d669d2e9f3f513bc5",
    "rob5.mp4":"6808a49f669d2e9f3f513bda",

}

index_id = os.getenv('INDEX_ID')      
video_id = video_dict["normal1.mp4"]      
query_text = "man in white shirt walking"

results = client.search.query(
    index_id=index_id,
    query_text=query_text,
    options=["visual","audio"],
    group_by="video"
)

for group in results.data.root:
    for clip in group.clips.root:
        if clip.video_id == video_id:
            print(
                f"Match found in Video: {clip.video_id}, "
                f"From: {clip.start}s to {clip.end}s, "
                f"Score: {clip.score}, "
                f"Confidence: {clip.confidence}"
            )
