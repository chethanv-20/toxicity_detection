import streamlit as st
from collections import defaultdict
import pandas as pd
from model import predict_toxicity_with_context

# Load euphemism dictionary from CSV
toxic_words_path = 'toxic_words_with_variations.csv'
toxic_words_df = pd.read_csv(toxic_words_path)

# Flatten the toxic words variations into a dictionary
euphemism_dict = {}
for _, row in toxic_words_df.iterrows():
    base_word = row['Original Word']
    for col in toxic_words_df.columns[1:]:
        variation = row[col]
        if pd.notna(variation):
            euphemism_dict[variation] = base_word

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_toxic_count' not in st.session_state:
    st.session_state.user_toxic_count = defaultdict(int)
if 'flagged_users' not in st.session_state:
    st.session_state.flagged_users = set()
if 'blocked_users' not in st.session_state:
    st.session_state.blocked_users = set()

# Function to display chat history with color-coded results
def display_chat(chat_history):
    for user, msg, result, is_toxic in chat_history:
        color = "red" if is_toxic else "green"
        st.markdown(f"**{user}:** {msg} - <span style='color: {color};'>{result}</span>", unsafe_allow_html=True)

def view_toxicity_report():
    report = pd.DataFrame(list(st.session_state.user_toxic_count.items()), columns=['User', 'Toxic Message Count'])
    st.write(report)

# Design and styling
st.set_page_config(page_title="Toxicity Analysis Chat Room", page_icon=":speech_balloon:", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00FFFF;'>Toxicity Analysis in Social Media</h1>", unsafe_allow_html=True)

# Define columns for user input to align them horizontally
col1, col2, col3 = st.columns([1, 1, 1])

def handle_user_input(user, user_input):
    if user in st.session_state.blocked_users:
        st.error(f"{user} is blocked and cannot send messages.")
        return "Blocked", False

    if user_input.strip() == "":  # Check if input is empty
        st.error("Input cannot be empty!")
        return "No input provided", False
    
    result = predict_toxicity_with_context(user_input, euphemism_dict)
    
    # Debugging: Show the result for inspection
    st.write(f"Debug: Result from toxicity analysis - '{result}'")
    
    # Determine if the result indicates toxicity
    if "classified as non-toxic" in result:
        is_toxic = False
    elif "classified as toxic" in result:
        is_toxic = True
    else:
        # Handle unexpected result formats
        is_toxic = "toxic" in result.lower()
    
    # Debugging: Show whether the message is classified as toxic
    st.write(f"Debug: Is toxic - {is_toxic}")
    
    st.session_state.chat_history.append((user, user_input, result, is_toxic))
    
    # Update the count only if the message is toxic
    if is_toxic:
        st.session_state.user_toxic_count[user] += 1
        toxic_count = st.session_state.user_toxic_count[user]
        
        if toxic_count == 3:
            st.warning(f"{user}, you have sent 3 toxic messages. Please be mindful of your behavior.")
        elif toxic_count == 5:
            st.warning(f"{user}, you have sent 5 toxic messages. Further toxic behavior will result in being blocked.")
        elif toxic_count >= 6:
            st.session_state.blocked_users.add(user)
            st.warning(f"{user} has been blocked for repeated toxic behavior. you cannot send messages anymore.")
    
    return result, is_toxic

with col1:
    user1_input = st.text_input("User 1:", key="user1_input")
    if st.button("Send as User 1", key="send_user1"):
        result, is_toxic = handle_user_input("User 1", user1_input)

with col2:
    user2_input = st.text_input("User 2:", key="user2_input")
    if st.button("Send as User 2", key="send_user2"):
        result, is_toxic = handle_user_input("User 2", user2_input)

with col3:
    user3_input = st.text_input("User 3:", key="user3_input")
    if st.button("Send as User 3", key="send_user3"):
        result, is_toxic = handle_user_input("User 3", user3_input)

# Display chat history
st.markdown("<hr style='border:1px solid #00FFFF;'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #00FFFF;'>Chat History</h3>", unsafe_allow_html=True)
display_chat(st.session_state.chat_history)

# Button to view toxicity report
if st.button("View Toxicity Report"):
    view_toxicity_report()

# Check for flagged users
if st.session_state.flagged_users:
    st.markdown("<hr style='border:1px solid red;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: red;'>Flagged Users</h3>", unsafe_allow_html=True)
    for user in st.session_state.flagged_users:
        st.warning(f"{user} has been flagged for repeated toxic behavior.")

# Add some background styling
st.markdown(
    """
    <style>
    body {
        background-color: #F0F8FF;
        color: #808080;
        font-family: Arial, sans-serif;
    }
    .stTextInput>div>input {
        border-radius: 12px;
    }
    div.stButton > button {
        background-color: #000000;
        color: white;
        border-radius: 12px;
    }
    div.stButton > button:hover {
        background-color: #333333;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)
