import os
import streamlit as st
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from streamlit_chat import message
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_URL = f"http://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '8000')}"
API_KEY = os.getenv("OPENAI_API_KEY", "test_key")  # Using test_key as fallback for development

# Set page configuration
st.set_page_config(
    page_title="IntelAssist - Multimodal AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{int(time.time())}"
if "use_rag" not in st.session_state:
    st.session_state.use_rag = True
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

# Helper functions
def make_api_request(endpoint, data=None, files=None, method="POST"):
    """Make a request to the API."""
    headers = {"X-API-Key": API_KEY}
    url = f"{API_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, data=data, files=files)
            else:
                response = requests.post(url, headers=headers, json=data)
        else:
            st.error(f"Unsupported method: {method}")
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def submit_feedback(response_id, rating, comments=None):
    """Submit feedback for a response."""
    data = {
        "response_id": response_id,
        "rating": rating,
        "user_id": st.session_state.user_id,
        "comments": comments
    }
    
    result = make_api_request("/api/feedback/submit", data=data)
    if result:
        st.session_state.feedback_given[response_id] = rating
        return True
    return False

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

# Sidebar
with st.sidebar:
    st.title("🧠 IntelAssist")
    st.subheader("Multimodal Self-Learning AI Assistant")
    
    # User settings
    st.header("Settings")
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id
    
    st.session_state.use_rag = st.checkbox("Use RAG (Retrieval-Augmented Generation)", value=st.session_state.use_rag)
    
    # Mode selection
    st.header("Interaction Mode")
    mode = st.radio(
        "Select Mode",
        ["Chat", "Image Analysis", "Voice Interaction", "Memory Explorer", "Feedback Dashboard"]
    )
    
    # About section
    st.header("About")
    st.markdown("""
    **IntelAssist** is an intelligent AI assistant that continuously learns from user interactions, 
    integrates text, images, and speech, and improves over time using RAG and RLHF.
    
    [View Documentation](https://github.com/yourusername/intelassist)
    """)

# Main content
st.title("IntelAssist - Multimodal AI Assistant")

# Chat Mode
if mode == "Chat":
    st.header("💬 Chat")
    
    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_msg_{i}")
        else:
            message(msg["content"], key=f"assistant_msg_{i}")
            
            # Show feedback buttons if not already given
            response_id = msg.get("id", f"msg_{i}")
            if response_id not in st.session_state.feedback_given:
                cols = st.columns(6)
                with cols[0]:
                    if st.button("👍", key=f"thumbs_up_{i}"):
                        if submit_feedback(response_id, 5):
                            st.success("Thank you for your feedback!")
                with cols[1]:
                    if st.button("👎", key=f"thumbs_down_{i}"):
                        if submit_feedback(response_id, 1):
                            st.success("Thank you for your feedback!")
    
    # Chat input
    user_input = st.text_input("Type your message here...", key="chat_input")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Prepare API request
        data = {
            "messages": st.session_state.messages,
            "use_rag": st.session_state.use_rag,
            "user_id": st.session_state.user_id
        }
        
        # Show spinner while waiting for response
        with st.spinner("Thinking..."):
            response = make_api_request("/api/text/chat", data=data)
        
        if response:
            # Generate a response ID
            response_id = f"resp_{int(time.time())}"
            
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["response"],
                "id": response_id
            })
            
            # Show sources if available
            if response.get("sources") and st.session_state.use_rag:
                with st.expander("Sources"):
                    for i, source in enumerate(response["sources"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(source["content"])
                        st.markdown("---")
            
            # Force refresh to show the new message
            st.experimental_rerun()

# Image Analysis Mode
elif mode == "Image Analysis":
    st.header("🖼️ Image Analysis")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Image analysis options
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Caption", "Detailed Analysis", "Ask About Image", "Generate Similar Image"]
        )
        
        if analysis_type == "Caption":
            if st.button("Generate Caption"):
                with st.spinner("Generating caption..."):
                    files = {"image": uploaded_file.getvalue()}
                    data = {"detailed": False}
                    response = make_api_request("/api/image/caption", data=data, files=files)
                
                if response:
                    st.subheader("Caption:")
                    st.write(response["caption"])
                    
                    st.subheader("Tags:")
                    st.write(", ".join(response["tags"]))
                    
                    st.subheader("Confidence:")
                    st.progress(response["confidence"])
        
        elif analysis_type == "Detailed Analysis":
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    files = {"image": uploaded_file.getvalue()}
                    response = make_api_request("/api/image/analyze", data={}, files=files)
                
                if response:
                    # Display analysis results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Description")
                        st.write(response["analysis"]["description"])
                        
                        st.subheader("Style")
                        st.write(response["analysis"]["style"])
                        
                        st.subheader("Mood")
                        st.write(response["analysis"]["mood"])
                        
                        st.subheader("Setting")
                        st.write(response["analysis"]["setting"])
                    
                    with col2:
                        st.subheader("Detected Objects")
                        for obj in response["objects"]:
                            st.write(f"• {obj['name']} (Confidence: {obj['confidence']:.2f})")
        
        elif analysis_type == "Ask About Image":
            question = st.text_input("Ask a question about this image")
            
            if question and st.button("Get Answer"):
                with st.spinner("Processing question..."):
                    files = {"image": uploaded_file.getvalue()}
                    data = {"text": question}
                    response = make_api_request("/api/image/multimodal", data=data, files=files)
                
                if response:
                    st.subheader("Answer:")
                    st.write(response["response"])
        
        elif analysis_type == "Generate Similar Image":
            prompt = st.text_input("Describe the image you want to generate")
            
            col1, col2 = st.columns(2)
            with col1:
                width = st.slider("Width", min_value=256, max_value=1024, value=512, step=64)
            with col2:
                height = st.slider("Height", min_value=256, max_value=1024, value=512, step=64)
            
            if prompt and st.button("Generate Image"):
                with st.spinner("Generating image..."):
                    data = {"prompt": prompt, "width": width, "height": height}
                    response = make_api_request("/api/image/text-to-image", data=data)
                
                if response and "image" in response:
                    st.subheader("Generated Image:")
                    generated_image = base64_to_image(response["image"])
                    st.image(generated_image, caption=prompt, use_column_width=True)

# Voice Interaction Mode
elif mode == "Voice Interaction":
    st.header("🎤 Voice Interaction")
    
    st.write("Upload an audio file to interact with the assistant using voice.")
    
    # Audio upload
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        # Voice interaction options
        interaction_type = st.radio(
            "Select Interaction Type",
            ["Speech to Text", "Voice Chat", "Text to Speech"]
        )
        
        if interaction_type == "Speech to Text":
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing audio..."):
                    files = {"audio": uploaded_file.getvalue()}
                    response = make_api_request("/api/speech/speech-to-text", data={}, files=files)
                
                if response:
                    st.subheader("Transcription:")
                    st.write(response["text"])
                    
                    st.subheader("Details:")
                    st.write(f"Language: {response['language']}")
                    st.write(f"Duration: {response['duration']:.2f} seconds")
                    st.write(f"Confidence: {response['confidence']:.2f}")
        
        elif interaction_type == "Voice Chat":
            if st.button("Process Voice Chat"):
                with st.spinner("Processing voice chat..."):
                    files = {"audio": uploaded_file.getvalue()}
                    data = {
                        "use_rag": st.session_state.use_rag,
                        "user_id": st.session_state.user_id
                    }
                    
                    # Add context if available
                    if st.session_state.messages:
                        data["context"] = json.dumps(st.session_state.messages)
                    
                    response = make_api_request("/api/speech/voice-chat", data=data, files=files)
                
                if response:
                    st.subheader("Text Response:")
                    st.write(response["text_response"])
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "user", "content": "Audio message"})
                    st.session_state.messages.append({"role": "assistant", "content": response["text_response"]})
                    
                    st.subheader("Audio Response:")
                    audio_bytes = base64.b64decode(response["audio_response"])
                    st.audio(audio_bytes, format=f"audio/{response['format']}")
        
        elif interaction_type == "Text to Speech":
            text = st.text_area("Enter text to convert to speech")
            voice = st.selectbox("Select Voice", ["default", "male", "female"])
            speed = st.slider("Speech Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            
            if text and st.button("Generate Speech"):
                with st.spinner("Generating speech..."):
                    data = {"text": text, "voice": voice, "speed": speed}
                    response = make_api_request("/api/speech/text-to-speech", data=data)
                
                if response and "audio" in response:
                    st.subheader("Generated Speech:")
                    audio_bytes = base64.b64decode(response["audio"])
                    st.audio(audio_bytes, format=f"audio/{response['format']}")
                    st.write(f"Duration: {response['duration']:.2f} seconds")

# Memory Explorer Mode
elif mode == "Memory Explorer":
    st.header("🧠 Memory Explorer")
    
    # Memory search
    st.subheader("Search Memory")
    query = st.text_input("Enter search query")
    
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Result Limit", min_value=1, max_value=20, value=5)
    with col2:
        min_score = st.slider("Minimum Score", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    
    user_filter = st.checkbox("Filter by current user ID")
    user_id_filter = st.session_state.user_id if user_filter else None
    
    if query and st.button("Search"):
        with st.spinner("Searching memory..."):
            data = {
                "query": query,
                "limit": limit,
                "min_score": min_score,
                "user_id": user_id_filter
            }
            response = make_api_request("/api/memory/search", data=data)
        
        if response:
            st.subheader(f"Found {response['total']} results:")
            
            for i, result in enumerate(response["results"]):
                with st.expander(f"Result {i+1} (Score: {result['score']:.2f})"):
                    st.markdown(f"**Content:**")
                    st.write(result["content"])
                    
                    st.markdown(f"**Metadata:**")
                    for key, value in result["metadata"].items():
                        if key == "timestamp" and isinstance(value, (int, float)):
                            st.write(f"- {key}: {datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            st.write(f"- {key}: {value}")
    
    # Memory statistics
    st.subheader("Memory Statistics")
    if st.button("Get Statistics"):
        with st.spinner("Fetching memory statistics..."):
            data = {"user_id": user_id_filter} if user_filter else {}
            response = make_api_request("/api/memory/stats", data=data, method="GET")
        
        if response:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Memories", response["total_memories"])
                st.metric("Total Users", response["total_users"])
                
                # Format size in KB/MB/GB
                size_bytes = response["size_bytes"]
                if size_bytes < 1024:
                    size_str = f"{size_bytes} bytes"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.2f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                
                st.metric("Total Size", size_str)
            
            with col2:
                if response.get("oldest_timestamp"):
                    oldest = datetime.fromtimestamp(response["oldest_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    newest = datetime.fromtimestamp(response["newest_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    
                    st.write(f"Oldest Memory: {oldest}")
                    st.write(f"Newest Memory: {newest}")
                else:
                    st.write("No timestamps available")
    
    # Add new memory
    st.subheader("Add New Memory")
    memory_content = st.text_area("Memory Content")
    
    col1, col2 = st.columns(2)
    with col1:
        memory_type = st.selectbox("Memory Type", ["conversation", "knowledge", "personal", "other"])
    with col2:
        memory_source = st.text_input("Source (optional)")
    
    if memory_content and st.button("Store Memory"):
        with st.spinner("Storing memory..."):
            data = {
                "content": memory_content,
                "metadata": {
                    "type": memory_type,
                    "source": memory_source,
                    "added_manually": True
                },
                "user_id": st.session_state.user_id
            }
            response = make_api_request("/api/memory/store", data=data)
        
        if response and "id" in response:
            st.success(f"Memory stored successfully with ID: {response['id']}")

# Feedback Dashboard Mode
elif mode == "Feedback Dashboard":
    st.header("📊 Feedback Dashboard")
    
    # Get feedback statistics
    with st.spinner("Loading feedback statistics..."):
        response = make_api_request("/api/feedback/stats", method="GET")
    
    if response:
        # Display overall metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", response["total_feedback"])
        
        with col2:
            st.metric("Average Rating", f"{response['average_rating']:.2f}/5.0")
        
        with col3:
            # Calculate percentage of positive ratings (4-5)
            positive = response["rating_distribution"]["4"] + response["rating_distribution"]["5"]
            total = sum(response["rating_distribution"].values())
            positive_pct = (positive / total * 100) if total > 0 else 0
            st.metric("Positive Feedback", f"{positive_pct:.1f}%")
        
        # Rating distribution chart
        st.subheader("Rating Distribution")
        
        # Prepare data for chart
        ratings = list(response["rating_distribution"].keys())
        counts = list(response["rating_distribution"].values())
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(ratings, counts, color='skyblue')
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Feedback Rating Distribution')
        ax.set_ylim(0, max(counts) * 1.2 if counts else 10)  # Add some space for labels
        
        st.pyplot(fig)
        
        # Recent trends
        st.subheader("Recent Trends")
        
        trend_data = {
            "Period": ["Last Day", "Last Week", "Last Month"],
            "Average Rating": [
                response["recent_trend"]["last_day"],
                response["recent_trend"]["last_week"],
                response["recent_trend"]["last_month"]
            ]
        }
        
        trend_df = pd.DataFrame(trend_data)
        st.table(trend_df)
    
    # RLHF Status
    st.subheader("RLHF Status")
    
    if st.button("Check RLHF Status"):
        with st.spinner("Checking RLHF status..."):
            response = make_api_request("/api/feedback/rlhf-status", method="GET")
        
        if response:
            st.write(f"Status: {response['status'].capitalize()}")
            st.write(f"RLHF Enabled: {'Yes' if response['rlhf_enabled'] else 'No'}")
            st.write(f"Unprocessed Feedback: {response['unprocessed_feedback_count']}")
            
            if response['last_run']:
                last_run = datetime.fromtimestamp(response['last_run']).strftime("%Y-%m-%d %H:%M:%S")
                st.write(f"Last Run: {last_run}")
            
            if response['status'] == 'running':
                st.progress(response['progress'])
            
            if response['error']:
                st.error(f"Error: {response['error']}")
    
    # Trigger RLHF Update
    st.subheader("Trigger RLHF Update")
    
    if st.button("Trigger RLHF Update"):
        with st.spinner("Triggering RLHF update..."):
            response = make_api_request("/api/feedback/trigger-rlhf")
        
        if response and "job_id" in response:
            st.success(f"RLHF update triggered with job ID: {response['job_id']}")
            st.info("Check the status above to monitor progress.")

# Run the app
if __name__ == "__main__":
    # This code is executed when the script is run directly
    pass 