import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
import json
import re
from typing import List, Dict

client = OpenAI()


#-----------------------------------------------------------LOAD AND SCAN VIDEO and dataset---------------------------------------------------------------
video = cv2.VideoCapture("pleasevideoasl.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")


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


# Updated main code-----------------------call 1
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            '''You are an ASL expert analyzing frames from a video of a sign. You can see and analyze images. The frames show the progression of a single ASL sign being performed.

    Carefully examine ALL frames to determine if one or two hands are used at ANY point during the sign.
    - Count a sign as two-handed if the second hand is used even briefly
    - Include the non-dominant hand if it's used as a base or reference
    - Check the entire sequence
    
    Return ONLY the number 1 or 2:
    - Return 1 if only one hand is used throughout the entire sign
    - Return 2 if both hands are used at any point

''',
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::15]),
        ],
    },
]

params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 2048,
    "temperature": 0.5,
}

# Get GPT response
result = client.chat.completions.create(**params)
gpt_response_one = result.choices[0].message.content

print(gpt_response_one)

#-------------------------------------------------------------------------------------simplify
simplified = extract_two_hand_signs("FINAL_ASL.json", gpt_response_one)




#-----------------------------call 2




# Updated main code
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            '''You are an ASL expert analyzing frames from a video of a sign. You can see and analyze images. The frames show the progression of a single ASL sign being performed.

    Identify the most likely hand shapes in the sign performed and match them to these options which are the only ones available:
        s-hand
        open hand - fingers spread(return "open hand" for this option)
        bent hand
        1-hand
        c-hand
        flat palm
        clawed hand
        o-hand
        
    Return only the two handshapes separated by a comma, in order of confidence (most likely first).
    Example: "flat palm, open hand"
''',
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::15]),
        ],
    },
]

params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 2048,
    "temperature": 0.7,
}

# Get GPT response
result = client.chat.completions.create(**params)
gpt_response_two = result.choices[0].message.content
print(gpt_response_two)
# Extract first handshape
first_handshape = gpt_response_two.split(',')[0].strip()
print(first_handshape)

#-------------------------------------------------------------------------------simplifications
finaldata = filter_by_handshape(simplified, first_handshape)
# print(finaldata)
locations = extract_unique_locations(finaldata)
# print(locations)


# Updated main code
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            f'''You are an ASL expert analyzing frames from a video of a sign. You can see and analyze images. The frames show the progression of a single ASL sign being performed.

    Analyze the location where the hand makes contact or is positioned relative to the body. Pay special attention to:
    - The exact point where the hand touches the body (if any)
    - The height level (face, chest, stomach, etc.)
    - The side (left, right, center)
    - The distance from the body
    
    From these options, choose the most specific location that applies: {locations}
    
    Return only the location name from the options provided. Be very specific. Example: "chin" instead of "face"
    DO NOT GUESS OR MAKE ASSUMPTIONS. If you are not 100% sure of the location, return "Not sure"
''',
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
        ],
    },
]

params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 2048,
    "temperature": 0.3,
}

result = client.chat.completions.create(**params)
location_guess = result.choices[0].message.content

print(location_guess)

if location_guess == "Not sure":
    known_location = input("Enter the location of the hand relative to the body: ")
else:
    known_location = location_guess


#--------------------------call 3 - FINAL

#convert to string 

json_string = json.dumps(finaldata)

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            f'''

You are an ASL expert analyzing video frames of a single sign. Using the following key details:
- Known handshape: {first_handshape}
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

params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 2048,
    "temperature": 0.5,
}

# Get GPT response
result = client.chat.completions.create(**params)
final_guess = result.choices[0].message.content

print(json_string)
print(final_guess)