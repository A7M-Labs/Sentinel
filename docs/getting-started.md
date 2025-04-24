# Getting Started 🚀

## Prerequisites
- Python 3.8+
- TwelveLabs API Key

## Quick Start
1. Clone the repository
   ```bash
   git clone https://github.com/A7M-Labs/Sentinel.git
   cd Sentinel
   git checkout backend
   ```

2. Set up environment
   ```bash
   python -m venv sentinel-venv
   source sentinel-venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your TwelveLabs API key
   ```

4. Start the application
   ```bash
   streamlit run main.py --server.port 5001
   ```

## Directory Structure
```
Sentinel/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── keyregen.py
├── core/
│   ├── asm_funcs.py
│   ├── config.py
│   ├── db.py
│   ├── detector.py
│   ├── jobs.py
│   ├── pipeline.py
│   └── utils.py
├── databases/
│   ├── events/
│   │   └── events.db
│   └── security/
│       ├── security_system.db
│       └── security_stats.db
├── videos/
│   ├── normal1.mp4
│   ├── normal2.mp4
│   ├── normal3.mp4
│   └── … 
├── models/
│   └── yolo/
│       ├── yolov8n.pt
│       └── yolov8l.pt
├── tmp/
│   └── segments/
└── archive/
    └── TLSearchAPI/
        └── search.py
```