import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import random
import os
import pandas as pd
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Le Sserafim AI Classifier",
    page_icon="🎵",
    layout="wide"
)

# Constants
MEMBERS = ['Sakura', 'Kim Chaewon', 'Huh Yunjin', 'Kazuha', 'Hong Eunchae']
PATH = Path('le_sserafim_images')

# Load Model
@st.cache_resource
def load_model():
    # PosixPath fix for cross-platform compatibility if needed
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    return load_learner('le_sserafim_model.pkl')

try:
    learn = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Helper: Get random image
def get_random_image():
    if not PATH.exists():
        return None, None
    files = get_image_files(PATH)
    if not files:
        return None, None
    f = random.choice(files)
    return f, parent_label(f)

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Member Classifier", "PK Game: You vs AI"])

# --- Mode 1: Classifier ---
if app_mode == "Member Classifier":
    st.title("🎵 Le Sserafim Member Classifier")
    st.markdown("Upload an image or pick a random one to see if the AI can identify the member!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Upload Image", "Random Test Image"])
        
        img_file = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                img_file = uploaded_file
        
        elif input_method == "Random Test Image":
            if st.button("🎲 Load Random Image"):
                f_path, label = get_random_image()
                if f_path:
                    st.session_state.random_image_path = f_path
                    st.session_state.random_image_label = label
                    # Clear previous prediction
                    if 'last_pred' in st.session_state:
                        del st.session_state.last_pred
                else:
                    st.warning("No images found in dataset folder.")
            
            if 'random_image_path' in st.session_state:
                img_file = st.session_state.random_image_path
                st.info(f"True Label (Hidden): {st.session_state.random_image_label}")

        if img_file:
            # Display Image
            # Handle both UploadedFile and Path objects
            if isinstance(img_file, (str, pathlib.Path)):
                image = PILImage.create(img_file)
                st.image(str(img_file), caption='Selected Image', use_container_width=True)
            else:
                image = PILImage.create(img_file)
                st.image(image, caption='Selected Image', use_container_width=True)
            
            # Predict Button
            if st.button('🔍 Identify Member', type="primary"):
                with st.spinner('AI is analyzing...'):
                    pred, pred_idx, probs = learn.predict(image)
                    
                    # Store result in session state to persist
                    st.session_state.last_pred = pred
                    st.session_state.last_probs = probs
                    st.session_state.last_idx = pred_idx

    with col2:
        if img_file and 'last_pred' in st.session_state:
            st.subheader("Analysis Results")
            
            pred = st.session_state.last_pred
            probs = st.session_state.last_probs
            pred_idx = st.session_state.last_idx
            
            # Display Result
            st.success(f"**Prediction: {pred}**")
            st.metric("Confidence", f"{probs[pred_idx]*100:.2f}%")
            
            # Chart
            probs_list = [float(p) for p in probs]
            labels = learn.dls.vocab
            
            df = pd.DataFrame({'Member': labels, 'Probability': probs_list})
            fig = px.bar(df, x='Member', y='Probability', 
                         title='Confidence Distribution',
                         color='Probability',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

# --- Mode 2: PK Game ---
elif app_mode == "PK Game: You vs AI":
    st.title("🎮 PK Game: Human vs AI")
    st.markdown("Can you beat the AI in recognizing Le Sserafim members?")
    
    # Scoreboard
    if 'score_user' not in st.session_state: st.session_state.score_user = 0
    if 'score_ai' not in st.session_state: st.session_state.score_ai = 0
    
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("👤 Your Score", st.session_state.score_user)
    col_s2.metric("🤖 AI Score", st.session_state.score_ai)
    
    st.divider()
    
    # Game Logic
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
        st.session_state.game_img_path = None
        st.session_state.game_true_label = None
        st.session_state.round_finished = False

    if not st.session_state.game_active:
        if st.button("🚀 Start New Round"):
            f_path, label = get_random_image()
            if f_path:
                st.session_state.game_img_path = f_path
                st.session_state.game_true_label = label
                st.session_state.game_active = True
                st.session_state.round_finished = False
                st.rerun()
            else:
                st.error("No images found to play with!")
    
    else:
        # Round is active
        col_g1, col_g2 = st.columns([1, 1])
        
        with col_g1:
            st.image(str(st.session_state.game_img_path), caption="Who is this?", use_container_width=True)
        
        with col_g2:
            if not st.session_state.round_finished:
                st.subheader("Make your guess!")
                user_guess = st.radio("Select Member:", MEMBERS)
                
                if st.button("Submit Guess"):
                    # Run AI Prediction
                    img = PILImage.create(st.session_state.game_img_path)
                    pred, pred_idx, probs = learn.predict(img)
                    
                    st.session_state.game_ai_pred = pred
                    st.session_state.game_ai_conf = probs[pred_idx]
                    st.session_state.game_user_guess = user_guess
                    st.session_state.round_finished = True
                    
                    # Update Scores
                    if user_guess == st.session_state.game_true_label:
                        st.session_state.score_user += 1
                    if pred == st.session_state.game_true_label:
                        st.session_state.score_ai += 1
                        
                    st.rerun()
            
            else:
                # Round Finished - Show Results
                st.subheader("Round Results")
                
                true_lbl = st.session_state.game_true_label
                user_lbl = st.session_state.game_user_guess
                ai_lbl = st.session_state.game_ai_pred
                
                st.info(f"**Correct Answer:** {true_lbl}")
                
                # User Result
                if user_lbl == true_lbl:
                    st.success(f"👤 You guessed: {user_lbl} (Correct!)")
                else:
                    st.error(f"👤 You guessed: {user_lbl} (Wrong!)")
                
                # AI Result
                if ai_lbl == true_lbl:
                    st.success(f"🤖 AI guessed: {ai_lbl} (Correct!)")
                else:
                    st.error(f"🤖 AI guessed: {ai_lbl} (Wrong!)")
                
                st.caption(f"AI Confidence: {st.session_state.game_ai_conf*100:.2f}%")
                
                if st.button("Next Round ➡️"):
                    st.session_state.game_active = False
                    st.rerun()
