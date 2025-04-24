# Getting Started 🚀

## Prerequisites
- Python 3.8+
- TwelveLabs API Key

## Quick Start
1. Clone the repository
   ```bash
   git clone https://github.com/A7M-Labs/Sentinel.git
   cd Sentinel
   ```

2. Set up environment
   ```bash
   python -m venv sentinel-venv
   source sentinel-venv/bin/activate  # Linux/MacOS
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
sentinel/
├── frontend/        # React frontend application
├── backend/         # Python backend services
├── models/          # ML models and configurations
├── utils/          # Utility functions and helpers
└── config/         # Configuration files
```