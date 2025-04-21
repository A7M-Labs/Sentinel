from twelvelabs import TwelveLabs
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get environment variables
client = TwelveLabs(api_key=os.getenv('TWELVELABS_API_KEY'))

video_dict={
    "normal1.mp4":"68062eaf8a51b971e4f94693",
    "normal5.mp4":"68063b3b790d2356d11264b6",
}

index_id = os.getenv('INDEX_ID')      
video_id = video_dict["normal1.mp4"]      
query_text = "woman in blue shirt acting suspicious and hiding item"

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
