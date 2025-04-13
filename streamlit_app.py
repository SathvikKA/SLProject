import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
import json
import re
from typing import List, Dict
import streamlit as st

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
#hello

# Set up Streamlit page config
st.set_page_config(page_title="ASL Interpreter", page_icon="👋", layout="wide")
st.title("ASL Interpreter")
st.write('''How to Use the ASL Interpreter
Upload a video of an ASL sign to identify what's being signed! The program will analyze the video you provide and tell you its best interpretation of the sign being performed.

Quick Start Guide:
Record or select a video of a single ASL sign (5 seconds or less recommended)
Upload the video using the "Upload" button below
Wait for analysis - our AI will process the hand movements and gestures
View the results showing the most likely ASL sign interpretation
For Best Results:
Ensure good lighting with minimal shadows
Position yourself against a plain background
Frame the video to show your upper body and hands clearly
Make deliberate, complete hand movements
For multi-part signs, record each component separately
Limitations:
Currently supports single signs, not full sentences or phrases
Works best with standard ASL signs from common dictionaries
May have difficulty with regional variations or personalized signing''')

# File uploader
video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
if video_file:
    st.video(video_file,loop=True,autoplay=True,muted=True)
#-----------------------------------------------------------LOAD AND SCAN VIDEO and dataset---------------------------------------------------------------
def load_asl_database(file_path: str = 'FINAL_ASL.json') -> Dict:
    """Load the ASL database from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as file:  # Explicitly specify encoding
        data = json.load(file)
    
    
#------------------------------------------------------------SIMPLIFICATION FUNCTIONS----------------------------------------------------------

def extract_two_hand_signs(file_path, number):
    """
    Extracts the words from the dataset where the 'hands' field matches the specified number.
    
    Args:
    - file_path (str): Path to the JSON file.
    - number (str or int): Number of hands (1 or 2)
    
    Returns:
    - dict: Words and their details that use the specified number of hands
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Convert number to int for comparison
    num_hands = int(number)
    
    # Create a filtered dictionary with words that match the hand count
    filtered_signs = {}
    for word, details in data.items():
        if details.get('hands') == num_hands:
            filtered_signs[word] = details
    
    return filtered_signs

def filter_by_handshape(filtered_signs, search_term):
    """
    Filters the ASL data by handshape from the already filtered dataset.
    
    Args:
    - filtered_signs (dict): Dictionary of ASL signs filtered by number of hands
    - search_term (str): Handshape to search for
    
    Returns:
    - dict: Filtered dictionary containing only signs with matching handshape
    """
    filtered = {}
    search_term = search_term.lower()

    for word, details in filtered_signs.items():
        handshape = details.get('handshape', '').lower()
        if search_term in handshape or handshape in search_term:
            filtered[word] = details

    return filtered

def extract_unique_locations(sign_data):
    """
    Extract all unique location values from a sign language JSON dictionary.
   
    Args:
        sign_data (dict): Dictionary containing sign language data where each entry
                         has a 'location' field
   
    Returns:
        set: Set of unique location values
    """
    # Use a set comprehension to extract unique locations
    # Skip any entries that might be missing the location field
    locations = {
        sign_info.get('location')
        for sign_info in sign_data.values()
        if isinstance(sign_info, dict) and 'location' in sign_info
    }
   
    # Remove None values if any entries were missing the location field
    locations.discard(None)
   
    return locations

#---------------------------------------------------------------PROPMTS AND FUNCTIONS CALLS

# Main app logic
if video_file is not None:
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Save uploaded video temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())
    
    status_text.text("Processing video frames...")
    
    # Process video
    video = cv2.VideoCapture("temp_video.mp4")
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    
    progress_bar.progress(25)
    status_text.text("Analyzing number of hands used...")

    # First API call - Number of hands
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                '''You are an ASL expert analyzing frames from a video of a sign. You can see and analyze images. The frames show the progression of a single ASL sign being performed.

        Carefully examine ALL frames to determine if one or two hands are used at ANY point during the sign.
        Return ONLY the number 1 or 2.''',
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::15]),
            ],
        },
    ]

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=PROMPT_MESSAGES,
        max_tokens=2048,
        temperature=0.5,
    )
    gpt_response_one = result.choices[0].message.content
    
    progress_bar.progress(50)
    status_text.text("Analyzing hand shapes...")

    # Second API call - Hand shapes
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                '''You are an ASL expert analyzing frames from a video of a sign. Identify the most likely hand shapes from these options:
                s-hand, open hand - fingers spread, bent hand, 1-hand, c-hand, flat palm, clawed hand, o-hand
                
                Return only the two handshapes separated by a comma, in order of confidence.''',
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::15]),
            ],
        },
    ]

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=PROMPT_MESSAGES,
        max_tokens=2048,
        temperature=0.7,
    )
    gpt_response_two = result.choices[0].message.content
    first_handshape = gpt_response_two.split(',')[0].strip()

    # Process intermediate results
    simplified = extract_two_hand_signs("FINAL_ASL.json", gpt_response_one)
    finaldata = filter_by_handshape(simplified, first_handshape)
    locations = extract_unique_locations(finaldata)
    
    progress_bar.progress(75)
    status_text.text("Analyzing hand location...")

    # Third API call - Location
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                f'''You are an ASL expert analyzing frames from a video of a sign. From these options, choose the most specific location: {locations}
                Return only the location name. If not sure, return "Not sure"''',
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
            ],
        },
    ]

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=PROMPT_MESSAGES,
        max_tokens=2048,
        temperature=0.3,
    )
    location_guess = result.choices[0].message.content

    if location_guess == "Not sure":
        known_location = st.text_input("Please enter the location of the hand relative to the body:")
    else:
        known_location = location_guess

    if known_location:
        progress_bar.progress(90)
        status_text.text("Making final prediction...")

        # Final API call
        json_string = json.dumps(finaldata)
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    f'''You are an ASL expert analyzing video frames of a single sign. Using the following key details:
                    - Handshape: {first_handshape}
                    - Full motion sequence
                    - Starting and ending positions
                    - Most likely location: {known_location}
                    From this set of possible signs: {json_string}
                    Return ONLY the top 3 most likely matching sign names in order of confidence, separated by commas. No explanation needed.
                    Example format:
                    bathroom, thank you, hello
                    ''',
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
                ],
            },
        ]

        result = client.chat.completions.create(
            model="gpt-4o",
            messages=PROMPT_MESSAGES,
            max_tokens=2048,
            temperature=0.5,
        )
        final_guess = result.choices[0].message.content
        print(final_guess)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")

        # Display results
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Hands", gpt_response_one)
        with col2:
            st.metric("Primary Hand Shape", first_handshape)
        with col3:
            st.metric("Location", known_location)

        st.subheader("Top Predictions")
        predictions = final_guess.split(',')
        print(predictions)
        for i, pred in enumerate(predictions, 1):
            st.write(f"{i}. {pred.strip()}")

        # Cleanup
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

else:
    st.info("Please upload a video to begin analysis.")