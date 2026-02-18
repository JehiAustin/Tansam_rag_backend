import streamlit as st
import requests
import json

# Configure page
st.set_page_config(
    page_title="AI Academic Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-size: 2.5rem;
        color: #00ff41;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    .user-message {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff41;
    }
    .bot-message {
        background-color: #262626;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff41;
    }
    .status-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .success { background-color: #1a3d1a; color: #00ff41; }
    .warning { background-color: #3d2a1a; color: #ffaa00; }
    .error { background-color: #3d1a1a; color: #ff4444; }
    .stTextInput > div > input {
        background-color: #1e1e1e;
        color: #ffffff;
        border: 1px solid #333333;
    }
    .stButton > button {
        background-color: #00ff41;
        color: #000000;
        border: none;
    }
    .stButton > button:hover {
        background-color: #00cc33;
    }
    .stSidebar {
        background-color: #1e1e1e;
        border-right: 1px solid #333333;
    }
    .streamlit-expanderHeader {
        background-color: #262626;
    }
    /* Center main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Fix sidebar width */
    .css-1l4qy8 {
        width: 300px !important;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE = "http://127.0.0.1:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=3)
        return response.status_code == 200, response.json()
    except:
        return False, {"error": "API not responding"}

def get_api_stats():
    """Get API statistics"""
    try:
        response = requests.get(f"{API_BASE}/stats", timeout=3)
        return response.status_code == 200, response.json()
    except:
        return False, {"error": "Stats not available"}

def ask_question(question, timeout=15):
    """Ask question to API"""
    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={"question": question},
            timeout=timeout
        )
        return response.status_code == 200, response.json()
    except requests.exceptions.Timeout:
        return False, {"error": "Request timeout - please try again"}
    except Exception as e:
        return False, {"error": f"Connection error: {str(e)}"}

# Header
st.markdown('<h1 class="main-header">🎓 AI Academic Advisor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔧 System Status")
    
    # Health check
    health_ok, health_data = check_api_health()
    if health_ok:
        st.markdown('<div class="status-indicator success">✅ API Online</div>', unsafe_allow_html=True)
        st.write(f"**RAG Ready:** {health_data.get('rag_ready', False)}")
        st.write(f"**LLM Ready:** {health_data.get('llm_ready', False)}")
    else:
        st.markdown('<div class="status-indicator error">❌ API Offline</div>', unsafe_allow_html=True)
        st.error("Please start the API server first!")
        st.stop()
    
    # Stats
    st.subheader("📊 Statistics")
    stats_ok, stats_data = get_api_stats()
    if stats_ok:
        st.write(f"**Documents:** {stats_data.get('documents', 0)}")
        st.write(f"**Embeddings:** {stats_data.get('cached_embeddings', 0)}")
        st.write(f"**Model:** {stats_data.get('llm_model', 'Unknown')}")
    else:
        st.write("Stats unavailable")

# Main chat interface
st.header("💬 Ask Questions")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for question preservation
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

def submit_question():
    """Handle question submission callback."""
    if st.session_state.question_input.strip():
        st.session_state.current_question = st.session_state.question_input
        st.session_state.submit_question = True

# Question input - preserves text after Enter
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Ask about academic programs, fees, admission, etc...",
        placeholder="e.g., What is fee structure?",
        key="question_input",
        label_visibility="collapsed",
        on_change=submit_question
    )
with col2:
    ask_button = st.button("🚀 Ask", type="primary", use_container_width=False)

# Handle question submission - works with both Enter key and button click
if (ask_button and question) or ('submit_question' in st.session_state and st.session_state.submit_question):
    # Clear the submit flag
    if 'submit_question' in st.session_state:
        del st.session_state.submit_question
    
    # Use the current question
    actual_question = st.session_state.current_question if 'current_question' in st.session_state else question
    
    with st.spinner("🤔 Thinking..."):
        success, result = ask_question(actual_question)
    
    if success:
        answer = result.get('answer', 'No answer received')
        status = result.get('status', 'unknown')
        
        # Add to chat history
        st.session_state.chat_history.append({
            'question': actual_question,
            'answer': answer,
            'status': status
        })
    else:
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Clear current question for next input
    st.session_state.current_question = ""

# Display chat history
if st.session_state.chat_history:
    st.subheader("📝 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        # User question
        st.markdown(f'''
        <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
            <div class="user-message" style="max-width: 70%; display: inline-block;">
                <strong>👤 You:</strong><br>
                {chat['question']}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Bot answer with status
        status_color = {
            'success': 'success',
            'cached': 'success', 
            'fallback': 'warning',
            'error': 'error'
        }.get(chat['status'], 'warning')
        
        status_icon = {
            'success': '✅',
            'cached': '⚡',
            'fallback': '⚠️',
            'error': '❌'
        }.get(chat['status'], '❓')
        
        st.markdown(f'''
        <div style="display: flex; justify-content: flex-start; margin-bottom: 1rem;">
            <div class="bot-message" style="max-width: 70%; display: inline-block;">
                <strong>🤖 AI Advisor ({status_icon} {chat['status']}):</strong><br>
                {chat['answer']}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 AI Academic Advisor | Powered by RAG + LLM</p>
    <p>💡 Tip: Repeat questions get instant cached responses!</p>
</div>
""", unsafe_allow_html=True)
