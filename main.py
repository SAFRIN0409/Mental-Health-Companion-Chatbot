
import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types
from datetime import datetime
import os
import time
import random
import urllib.parse
import json
import glob

# --- Page Configuration ---
st.set_page_config(
    page_title="Serenity - Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');

    /* General App Styling */
    .stApp {
        background-color: #fdfbf7; /* Soft Beige Background */
        font-family: 'Nunito', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, .stMarkdown, p, div {
        font-family: 'Nunito', sans-serif !important;
        color: #4a4a4a;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 20px;
        color: #4a4a4a;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        padding: 0 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6e6fa !important; /* Lavender */
        color: #5d4e8c !important;
        border-bottom: none;
    }

    /* Custom Chat Bubbles */
    .user-bubble {
        background-color: #e3f2fd;
        color: #4a4a4a;
        padding: 12px 18px;
        border-radius: 20px 20px 0 20px;
        margin-bottom: 15px;
        text-align: right;
        width: fit-content;
        margin-left: auto;
        border: 1px solid #d1e7f7;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 75%;
    }
    .bot-bubble {
        background-color: #fff0f5;
        color: #4a4a4a;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 0;
        margin-bottom: 15px;
        width: fit-content;
        margin-right: auto;
        border: 1px solid #fce4ec;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 75%;
    }
    .timestamp {
        font-size: 0.7rem;
        color: #888;
        margin-top: 5px;
        display: block;
    }
    
    /* Game Buttons */
    .game-btn {
        height: 100px;
        width: 100%;
        font-size: 24px;
        border-radius: 15px;
    }

    /* Text Input Styling */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        padding: 10px;
    }

    /* Cards for Journal */
    .journal-card {
        background-color: white;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #f0f0f0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #edf2f7;
    }
    
    /* Share Button Style */
    .share-btn {
        display: inline-block;
        padding: 0.5em 1em;
        color: #ffffff;
        background-color: #ffb7b2;
        border-radius: 20px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    .share-btn:hover {
        background-color: #ff9e99;
        color: white;
    }
    
    /* History Button Style */
    .history-btn {
        width: 100%;
        text-align: left;
        padding: 8px;
        background: white;
        border: 1px solid #eee;
        border-radius: 10px;
        margin-bottom: 5px;
        cursor: pointer;
    }
    .history-btn:hover {
        background: #f0f0f0;
    }
    
    /* Mood Card */
    .mood-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #5d4e8c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .mood-stat {
        font-size: 0.9rem;
        font-weight: 700;
        color: #5d4e8c;
    }
    
    /* Crisis Banner */
    .crisis-banner {
        background-color: #ffebee;
        border: 2px solid #ef5350;
        padding: 20px;
        border-radius: 15px;
        color: #c62828;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    /* Journal Analytics */
    .journal-ana {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 10px;
        font-size: 0.85rem;
        margin-top: 10px;
        border-left: 3px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


# --- API & Model Setup ---

# 1. Load Gemini API Key
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.secrets.get("GEMINI_API_KEY") 
        if not api_key:
            st.error("🚨 Gemini API Key not found in st.secrets. Please configure .streamlit/secrets.toml")
            st.stop()
except FileNotFoundError:
    st.error("🚨 .streamlit/secrets.toml file not found.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Lightweight AI Emotion Detection (Replaces heavy Transformers/Torch)
def get_ai_emotion(text):
    """Detect emotion and confidence using Gemini instead of heavy local models."""
    try:
        prompt = f"""
        Analyze the emotion in this text: "{text}"
        Choose ONE from: joy, sadness, anger, fear, surprise, love, disgust.
        Return the result in JSON format: {{"label": "emotion", "score": 0.95}}
        """
        res = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(res.text)
        return data
    except Exception as e:
        return {"label": "neutral", "score": 1.0}

# --- Helper Functions ---

def save_mood_to_csv(mood):
    """Save detected mood to a CSV file."""
    file_path = "mood_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame({"Timestamp": [timestamp], "Mood": [mood]})
    
    if not os.path.exists(file_path):
        new_data.to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, mode='a', header=False, index=False)
        
import re

def generate_chat_title():
    """Generate a 4-8 word title from the first user message."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            text = msg["content"]
            text = re.sub(r'[^\w\s]', '', text) # remove punctuation
            words = text.split()
            title = " ".join(words[:6]) # first 6 words
            return title.capitalize()
    return "New Chat"

def save_chat_session():
    """Save the current chat session to a JSON file."""
    if not os.path.exists("chat_history"):
        os.makedirs("chat_history")
    
    if not st.session_state.messages:
        return

    if st.session_state.chat_title == "New Chat":
        st.session_state.chat_title = generate_chat_title()

    # Use the timestamp + slugified title as filename
    safe_title = "".join([c if c.isalnum() else "_" for c in st.session_state.chat_title])
    filename = f"chat_history/{st.session_state.session_id}_{safe_title}.json"
    
    # Clean up old versions of this session if they exist with different titles
    for old_file in glob.glob(f"chat_history/{st.session_state.session_id}_*.json"):
        if old_file != filename:
            try: os.remove(old_file)
            except: pass

    with open(filename, "w") as f:
        json.dump({
            "title": st.session_state.chat_title,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": st.session_state.messages
        }, f)

def load_chat_session(filename):
    """Load a chat session from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            st.session_state.messages = data.get("messages", [])
            st.session_state.chat_title = data.get("title", "New Chat")
        else:
            st.session_state.messages = data
            st.session_state.chat_title = "Past Chat"
            
    # Extract ID from filename for current session context
    st.session_state.session_id = os.path.basename(filename).split("_")[0]

def get_chat_history_files():
    """Get list of chat history files sorted by modification time."""
    if not os.path.exists("chat_history"):
        return []
    files = glob.glob("chat_history/*.json")
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def check_crisis(text):
    """Check for crisis keywords."""
    crisis_keywords = [
        "suicide", "want to die", "kill myself", "self harm", "self-harm",
        "end everything", "hopeless", "no point in living", "don't want to live",
        "better off dead", "death", "hurting myself"
    ]
    for keyword in crisis_keywords:
        if keyword in text.lower():
            return True
    return False

def get_tone(emotion):
    positive = ["joy", "love", "surprise"]
    negative = ["sadness", "anger", "fear", "disgust"]
    if emotion in positive:
        return "Positive ✨"
    elif emotion in negative:
        return "Negative 🌧️"
    return "Neutral ☁️"

def get_mood_emoji(mood):
    mood_emoji = {
        "joy": "🌻", "sadness": "🌧️", "anger": "🔥", "fear": "🍂",
        "neutral": "☁️", "surprise": "✨", "love": "❤️", "disgust": "🤢"
    }
    return mood_emoji.get(mood.lower(), "🌱")

if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")

if "chat_title" not in st.session_state:
    st.session_state.chat_title = "New Chat"

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []
if "journal_entries" not in st.session_state:
    st.session_state.journal_entries = []
if "bubble_wrap" not in st.session_state:
    st.session_state.bubble_wrap = [True] * 20
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# 4. Welcome Message (One-time)
if len(st.session_state.messages) == 0:
    name_str = f" {st.session_state.user_name}" if st.session_state.user_name else ""
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"Hi{name_str} 🌿 I’m Serenity. You can talk to me about anything. How are you feeling today?",
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div style='text-align: center; font-size: 60px;'>🧠</div>", unsafe_allow_html=True)
    
    # User Profile Section
    if not st.session_state.user_name:
        with st.expander("👤 Set Your Name", expanded=True):
            un = st.text_input("What should Serenity call you?", key="name_input")
            if st.button("Save Name"):
                st.session_state.user_name = un
                st.rerun()
    else:
        st.markdown(f"<h3 style='text-align: center;'>Welcome, {st.session_state.user_name}!</h3>", unsafe_allow_html=True)
        if st.button("Edit Name"):
            st.session_state.user_name = None
            st.rerun()

    st.markdown("<h2 style='text-align: center;'>Mood Tracker</h2>", unsafe_allow_html=True)
    
    if st.session_state.mood_history:
        # Mood Distribution Chart
        mood_counts = pd.Series(st.session_state.mood_history).value_counts()
        fig = px.pie(
            values=mood_counts.values, 
            names=mood_counts.index, 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recent Moods")
        # Show last 5 moods
        for i, mood in enumerate(reversed(st.session_state.mood_history[-5:])):
            st.markdown(f"{get_mood_emoji(mood)} **{mood.capitalize()}**")
    else:
        st.info("Start chatting to track your moods!")

    st.markdown("---")
    
    # NEW: Chat History Section
    st.markdown("### 📜 Past Conversations")
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        name_str = f" {st.session_state.user_name}" if st.session_state.user_name else ""
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Hi{name_str} 🌿 I’m Serenity. You can talk to me about anything. How are you feeling today?",
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        st.rerun()

    history_files = get_chat_history_files()
    if history_files:
        for file in history_files:
            display_name = "Past Conversation"
            date_info = ""
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        display_name = data.get("title", "Past Chat")
                        ca = data.get("created_at", "")
                        if ca:
                            dt = datetime.strptime(ca, "%Y-%m-%d %H:%M")
                            date_info = dt.strftime("%b %d • %I:%M %p")
            except:
                pass
            
            # If no stored date, use filename
            if not date_info:
                try:
                    ts_str = os.path.basename(file).split("_")[0]
                    date_info = datetime.strptime(ts_str[:8], "%Y%m%d").strftime("%b %d")
                except:
                    date_info = "Previous"

            # Cleaner Button Label with Delete Option
            col_chat, col_del = st.columns([0.85, 0.15])
            with col_chat:
                full_label = f"💭 {display_name}\n{date_info}"
                if st.button(full_label, key=file, use_container_width=True):
                    load_chat_session(file)
                    st.rerun()
            with col_del:
                # Minimalistic trash button
                if st.button("🗑️", key=f"del_{file}", help="Delete this chat"):
                    try:
                        os.remove(file)
                        # If deleted chat was the active one, clear it
                        file_base = os.path.basename(file).split("_")[0]
                        if file_base == st.session_state.session_id:
                            st.session_state.messages = []
                            st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.caption("No saved chats yet.")

    st.markdown("---")
    
    if st.button("🗑️ Clear Current Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # 5. Export & Share Chat
    st.markdown("### Options")
    # Convert chat history to string
    chat_export = ""
    for msg in st.session_state.messages:
        chat_export += f"[{msg['role'].upper()}] {msg.get('timestamp','')}: {msg['content']}\n"
    
    col_dl, col_share = st.columns(2)
    with col_dl:
        st.download_button(
            label="⬇ Download",
            data=chat_export,
            file_name="serenity_chat.txt",
            mime="text/plain"
        )
    with col_share:
        # Create a mailto link
        subject = "My Chat with Serenity 🌿"
        body = urllib.parse.quote(chat_export)
        # Limit body length for mailto compatibility
        share_link = f"mailto:?subject={subject}&body={body[:1500]}..." 
        st.markdown(f'<a href="{share_link}" target="_blank" class="share-btn">💌 Share</a>', unsafe_allow_html=True)


# --- Main Interface ---
st.title("🌿 Serenity")
st.markdown("##### *Your safe space for reflection and calm.*")

# Updated Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎮 Stress Relief", "📔 Journal"])

# --- TAB 1: Chat Interface ---
with tab1:
    # 2. Add chat scroll container
    chat_container = st.container()
    
    with chat_container:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        # Display Chat History
        for msg in st.session_state.messages:
            bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
            safe_content = msg["content"].replace("\n", "<br>")
            
            st.markdown(f"""
            <div class="{bubble_class}">
                {safe_content}
                <div class="timestamp">{msg.get("timestamp", "")}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chat Input
    if user_input := st.chat_input("How are you feeling today?"):
        timestamp = datetime.now().strftime("%I:%M %p")
        
        # 1. Update Session and UI immediately
        st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": timestamp})
        # SAVE SESSION AUTOMATICALLY
        save_chat_session()
        
        with chat_container:
            st.markdown(f"""
            <div class="user-bubble">
                {user_input}
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Farewell Check
            if user_input.lower().strip() in ["bye", "goodbye", "see you", "ok bye", "bye serenity"]:
                farewell_msg = "Take care 🌿 I'm here whenever you need to talk again."
                
                with st.empty():
                     st.markdown("💭 *Serenity is typing...*")
                     time.sleep(1)
                
                st.markdown(f"""
                <div class="bot-bubble">
                    {farewell_msg}
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": farewell_msg, "timestamp": timestamp})
                save_chat_session()
                time.sleep(1)
                st.rerun()

            else:
                # 3. Process Response
                response_text = ""
                emotion = "neutral" 
                
                # A. Crisis Detection
                if check_crisis(user_input):
                    response_text = """
                    <div class="crisis-banner">
                        <h3>🌿 Please Reach Out — You Matter.</h3>
                        <p>I'm really concerned about you. You deserve support, and you don't have to go through this alone. Please reach out to a professional or a trusted person right now.</p>
                        <hr>
                        <b>📞 KIRAN (Mental Health Helpline):</b> 1800-599-0019<br>
                        <b>📞 AASRA:</b> +91-9820466726 (24/7 Support)<br>
                        <b>💬 iCall:</b> +91 91529 87821
                    </div>
                    """
                    
                    with st.empty():
                        st.markdown("💭 *Serenity is typing...*")
                        time.sleep(0.6)
                    
                    st.markdown(f"""
                    <div class="bot-bubble">
                        {response_text}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    # B. Emotion Detection
                    with st.spinner("Analyzing Mood..."):
                        result = get_ai_emotion(user_input)
                        emotion = result["label"]
                        confidence = result["score"] * 100
                        tone = get_tone(emotion)
                        st.session_state.mood_history.append(emotion)
                        save_mood_to_csv(emotion)

                    # Display Mood Detection Card
                    st.markdown(f"""
                    <div class="mood-card">
                        <div class="mood-stat">🧠 Mood Detection Results</div>
                        <div style="font-size: 0.85rem; margin-top: 5px;">
                            <b>Detected Mood:</b> {emotion.capitalize()} {get_mood_emoji(emotion)}<br>
                            <b>Confidence:</b> {confidence:.1f}%<br>
                            <b>Emotional Tone:</b> {tone}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # C. Generate Response with History
                    # Prepare history for Gemini
                    history_for_gemini = []
                    # System instruction as the first message context
                    current_time = datetime.now().strftime("%A, %b %d, %Y, %I:%M %p")
                    user_name_part = f"The user's name is {st.session_state.user_name}." if st.session_state.user_name else ""
                    system_instruction = f"""
                    You are Serenity, a warm and supportive mental health companion for students.
                    Current date and time: {current_time}.
                    {user_name_part}
                    Current detected user emotion: {emotion}.
                    
                    DYNAMIC RESPONSE RULES:
                    - If user is SAD: Start with "I'm so sorry you're feeling this way..." or "I'm here for you."
                    - If user is ANXIOUS: Start with "Let's take a slow breath together..."
                    - If user is HAPPY: Start with "That's wonderful to hear!"
                    - If user is ANGERED: Be extremely validating and calm.
                    
                    GENERAL RULES:
                    - Never mention being an AI or language model.
                    - Speak like a caring human friend.
                    - Validate feelings first.
                    - Keep replies 2–3 sentences.
                    - Offer one gentle suggestion.
                    - Avoid medical advice or diagnosis.
                    - Maintain continuity from the conversation history.
                    """
                    
                    # Convert session messages to Gemini format (limit to last 10 for performance)
                    for msg in st.session_state.messages[-10:]:
                        role = "model" if msg["role"] == "assistant" else "user"
                        # Clean up '🌿 ' or other prefixes if any in the stored content
                        content = msg["content"].replace("🌿 ", "")
                        history_for_gemini.append({"role": role, "parts": [{"text": content}]})

                    try:
                        # Use chat session for true real-time conversation memory
                        chat = client.chats.create(
                            model="gemini-flash-latest",
                            config=types.GenerateContentConfig(system_instruction=system_instruction),
                            history=history_for_gemini[:-1] # Exclude the current message which we send next
                        )
                        
                        response_placeholder = st.empty()
                        full_text = ""
                        
                        # Send the actual user input
                        stream = chat.send_message_stream(user_input)
                        
                        for chunk in stream:
                            if chunk.text:
                                text_chunk = chunk.text
                                full_text += text_chunk
                                response_placeholder.markdown(f"🌿 {full_text} ...")
                        
                        # Set Title from first user message if it's still "New Chat"
                        if st.session_state.chat_title == "New Chat":
                            st.session_state.chat_title = generate_chat_title()
                        
                        # Final Bubble
                        response_placeholder.markdown(f"""
                        <div class="bot-bubble">
                            {full_text}
                            <div class="timestamp">{timestamp}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        response_text = full_text
                        
                    except Exception as e:
                        # Show actual error for debugging
                        st.error(f"Connection Error: {str(e)}")
                        response_text = "I'm having a little trouble connecting right now. Please check your internet or API key."

                if response_text:
                    st.session_state.messages.append({"role": "assistant", "content": response_text, "timestamp": timestamp})
                    save_chat_session()
                    st.rerun()

# --- TAB 2: Stress Relief Games ---
with tab2:
    st.header("🎮 Relax & De-stress")
    
    game_choice = st.radio("Choose a calm activity:", ["🌬️ Breathing", "🧘 Grounding", "🧼 Bubble Wrap", "😄 Jokes & Music"], horizontal=True)
    
    if game_choice == "🌬️ Breathing":
        st.markdown("### 4-4-4 Box Breathing")
        st.write("Equal parts inhale, hold, and exhale. Great for immediate anxiety relief.")
        
        if st.button("Start 4-4-4 Session"):
            progress_text = st.empty()
            bar = st.progress(0)
            
            for cycle in range(3):
                # Inhale
                progress_text.markdown("### 🌸 **Inhale... (4s)**")
                for i in range(100):
                    time.sleep(0.04) 
                    bar.progress(i + 1)
                
                # Hold
                progress_text.markdown("### 😐 **Hold... (4s)**")
                for i in range(100):
                    time.sleep(0.04)
                    bar.progress(100)
                
                # Exhale
                progress_text.markdown("### 🍃 **Exhale... (4s)**")
                for i in range(100):
                    time.sleep(0.04)
                    bar.progress(100 - i)
                
                # Pause
                progress_text.markdown("### ⏸️ **Pause... (4s)**")
                time.sleep(4)
            
            progress_text.markdown("### ✨ **You're doing great. Feel more centered?**")
            st.balloons()
            
    elif game_choice == "🧘 Grounding":
        st.markdown("### 5-4-3-2-1 Grounding Technique")
        st.write("A grounding technique to help you focus on the present moment.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("👀 **5** things you can **SEE**")
            st.info("✋ **4** things you can **TOUCH**")
            st.info("👂 **3** things you can **HEAR**")
        with col2:
            st.info("👃 **2** things you can **SMELL**")
            st.info("👅 **1** thing you can **TASTE**")
        st.write("Take a deep breath once you've acknowledged all five.")

    elif game_choice == "🧼 Bubble Wrap":
        st.markdown("### Pop the Stress Away!")
        st.write("Click the bubbles to pop them!")
        
        # Grid layout for bubbles
        cols = st.columns(5)
        for i in range(20):
            with cols[i % 5]:
                if st.session_state.bubble_wrap[i]:
                    if st.button("🔵", key=f"bubble_{i}"):
                        st.session_state.bubble_wrap[i] = False
                        st.rerun()
                else:
                    st.button("💥", key=f"popped_{i}", disabled=True)
        
        if st.button("Reset Bubbles"):
            st.session_state.bubble_wrap = [True] * 20
            st.rerun()

    elif game_choice == "😄 Jokes & Music":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Tell me a Joke")
            if st.button("😂 Laugh"):
                jokes = [
                    ("Why did the scarecrow win an award?", "Because he was outstanding in his field!"),
                    ("What do you call a fake noodle?", "An impasta!"),
                    ("Why don't scientists trust atoms?", "Because they make up everything!"),
                    ("How does a penguin build its house?", "Igloos it together!"),
                    ("Why did the math book look sad?", "It had too many problems.")
                ]
                j = random.choice(jokes)
                st.info(f"**Q:** {j[0]}\n\n**A:** {j[1]}")
        
        with c2:
            st.markdown("#### Calming Sounds")
            st.write("Choose a soundscape to focus or relax:")
            st.markdown("[🎵 Lofi Beats for Study](https://www.youtube.com/watch?v=jfKfPfyJRdk)")
            st.markdown("[🌊 Ocean Waves](https://www.youtube.com/watch?v=bn9F19Hi1Lk)")
            st.markdown("[🌧️ Thunderstorm & Rain](https://www.youtube.com/watch?v=mPZkdNFkNps)")

# --- TAB 3: Journal Interface ---
with tab3:
    # Academic Gold: Journal Analytics Summary
    if st.session_state.journal_entries:
        total = len(st.session_state.journal_entries)
        pos_count = sum(1 for e in st.session_state.journal_entries if "Positive" in e.get("analysis", {}).get("tone", ""))
        happiness_index = (pos_count / total) * 100 if total > 0 else 0
        
        st.markdown(f"""
        <div style="background: white; padding: 15px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px;">
            <h4 style="margin: 0; color: #5d4e8c;">📔 Journal Insights</h4>
            <div style="display: flex; gap: 20px; margin-top: 10px;">
                <div><b>Entries:</b> {total}</div>
                <div><b>Happiness Index:</b> {happiness_index:.0f}%</div>
                <div><b>Trend:</b> {'Improving 📈' if happiness_index > 50 else 'Needs Care 🌿'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 🖊️ New Entry")
        with st.container(border=True):
            journal_text = st.text_area("Write down your thoughts...", height=200)
            mood_tag = st.selectbox("How was your day?", ["Happy 😊", "Calm 😌", "Stressed 😫", "Sad 😢", "Productive 🚀", "Anxious 😰"], index=1)
            
            if st.button("Save Entry"):
                if journal_text:
                    # Run Sentiment Analysis for Academic Gold
                    res = get_ai_emotion(journal_text)
                    j_emotion = res["label"]
                    j_score = res["score"] * 100
                    
                    entry = {
                        "date": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
                        "text": journal_text,
                        "mood": mood_tag,
                        "analysis": {
                            "emotion": j_emotion,
                            "confidence": j_score,
                            "tone": get_tone(j_emotion)
                        }
                    }
                    st.session_state.journal_entries.append(entry)
                    st.success("Saved to your journal! 📔")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Please write something first.")
    
    with col2:
        st.markdown("### 📖 Past Entries")
        if st.session_state.journal_entries:
            for entry in reversed(st.session_state.journal_entries):
                ana = entry.get("analysis")
                ana_html = ""
                if ana:
                    ana_html = f"""
                    <div class="journal-ana">
                        <b>AI Analysis:</b> Detected {ana['emotion']} ({ana['tone']}) with {ana['confidence']:.1f}% confidence.
                    </div>
                    """
                
                st.markdown(f"""
                <div class="journal-card">
                    <div style="display: flex; justify-content: space-between; color: #888; font-size: 0.8rem;">
                        <span>{entry['date']}</span>
                        <span>{entry['mood']}</span>
                    </div>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #eee;">
                    <p style="white-space: pre-wrap;">{entry['text']}</p>
                    {ana_html}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Your journal is empty. Start writing to clear your mind.")