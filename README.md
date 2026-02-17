# 🌿 Serenity - Mental Health Companion

Serenity is an AI-powered mental health chatbot designed to support students. It uses emotion detection and Google's Gemini AI to provide empathetic, non-medical support.

## Features

- **Emotion Detection**: Uses a HuggingFace model (`j-hartmann/emotion-english-distilroberta-base`) to understand how you're feeling.
- **Empathetic AI Chat**: Powered by Google Gemini 2.5 Flash, providing warm, short, and supportive responses.
- **Crisis Detection**: Automatically detects crisis keywords and provides helpline numbers instead of AI responses.
- **Mood Tracking**: Visualizes your mood over time with interactive charts.
- **Journaling**: A private space to write down thoughts and track daily moods.
- **Secure**: API keys are managed securely via Streamlit secrets.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create a file named `.streamlit/secrets.toml` in the project root directory and add your Google Gemini API key:

```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

> **Note**: Never commit your `secrets.toml` file to version control.

### 3. Run the Application
Start the Streamlit server:
```bash
streamlit run main.py
```

## Project Structure
- `main.py`: The main application code.
- `requirements.txt`: Python dependencies.
- `mood_log.csv`: Stores a log of detected moods (created automatically).
- `.streamlit/secrets.toml`: Configuration file for API keys.

## disclaimer
This application is a supportive tool and **not** a substitute for professional medical advice, diagnosis, or treatment. If you are in crisis, please contact emergency services or the helplines provided in the app.