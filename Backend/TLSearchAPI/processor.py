from twelvelabs import TwelveLabs
from twelvelabs.models.task import Task
import time
import sqlite3
from datetime import datetime
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def adapt_datetime(dt):
    """Convert datetime to ISO format string for SQLite storage"""
    return dt.isoformat()

def convert_datetime(s):
    """Convert ISO format string back to datetime object"""
    return datetime.fromisoformat(s)

def init_db():
    """Initialize database with proper datetime handling"""
    conn = sqlite3.connect('processing_stats.db')
    
    # Register adapter and converter for datetime
    sqlite3.register_adapter(datetime, adapt_datetime)
    sqlite3.register_converter("datetime", convert_datetime)
    
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_times (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            duration FLOAT,
            timestamp datetime,
            video_id TEXT
        )
    ''')
    conn.commit()
    return conn

def calculate_average(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT AVG(duration) FROM processing_times')
    avg = cursor.fetchone()[0]
    return avg if avg else 0

def process_video(file_path, index_id):
    start_time = time.time()
    
    client = TwelveLabs(api_key=os.getenv(('TWELVELABS_API_KEY')))
    
    task = client.task.create(
        index_id=index_id,
        file=file_path
    )
    print(f"Task id={task.id}, Video id={task.video_id}")

    def on_task_update(task: Task):
        print(f"  Status={task.status}")

    task.wait_for_done(sleep_interval=5, callback=on_task_update)

    if task.status != "ready":
        raise RuntimeError(f"Indexing failed with status {task.status}")
    
    # Calculate and store processing time
    end_time = time.time()
    processing_duration = end_time - start_time
    
    # Connect with datetime support
    conn = sqlite3.connect('processing_stats.db', 
                         detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO processing_times (duration, timestamp, video_id) VALUES (?, ?, ?)',
            (processing_duration, datetime.now(), task.video_id)
        )
        conn.commit()
        
        avg_time = calculate_average(conn)
        
        print(f"\nProcessing Statistics:")
        print(f"Current processing time: {processing_duration:.2f} seconds")
        print(f"Average processing time: {avg_time:.2f} seconds")
        
    finally:
        conn.close()
    
    return task.video_id

if __name__ == "__main__":
    # Initialize database on first run
    init_db()
    
    # Example usage
    file_path = "/home/aahil/Code/Personal/Projects/Sentinel/searchapi/trainingdata/normal5.mp4"
    index_id = os.getenv('INDEX_ID')
    video_id = process_video(file_path, index_id)
    print(f"The unique identifier of your video is {video_id}.")
