import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Custom styling using Streamlit's markdown
st.markdown(
    """
    <style>
    .main {background-color: #F0F2F6;}
    .reportview-container {
        background: linear-gradient(135deg, #f0f2f6, #ffffff);
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffb3b3;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        color: white;
        background-color: #FF4B4B;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        background-color: #FFF8F0;
        border-radius: 5px;
        padding: 10px;
    }
    .stMultiSelect>div>div>div {
        background-color: #FFF8F0;
    }
    .stSlider>div>div>div>div {
        background-color: #ff8b8b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    training = pd.read_csv('C:/Users/My Device/Downloads/healthcare-chatbot-master/healthcare-chatbot-master/Data/Training.csv')
    testing = pd.read_csv('C:/Users/My Device/Downloads/healthcare-chatbot-master/healthcare-chatbot-master/Data/Testing.csv')
    return training, testing

# Load and prepare data
training, testing = load_data()
cols = training.columns[:-1]  # Symptoms columns
x = training[cols]
y = training['prognosis']

# Preprocessing: map strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier().fit(x_train, y_train)

# Load additional data (severity, description, precaution)
@st.cache_data
def load_master_data():
    severity_dict = {}
    description_list = {}
    precaution_dict = {}

    # Load Symptom Severity
    with open('healthcare-chatbot-master/MasterData/Symptom_severity.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    severity_dict[row[0]] = int(row[1])
                except ValueError:
                    print(f"Warning: Couldn't parse severity for {row[0]}")

    # Load Symptom Descriptions
    with open('healthcare-chatbot-master/MasterData/symptom_Description.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) >= 2:
                description_list[row[0]] = row[1]

    # Load Symptom Precautions
    with open('healthcare-chatbot-master/MasterData/symptom_precaution.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) >= 5:
                precaution_dict[row[0]] = [row[1], row[2], row[3], row[4]]

    return severity_dict, description_list, precaution_dict

severity_dict, description_list, precaution_dict = load_master_data()

# Symptom checker to find patterns based on user input
def check_pattern(dis_list, inp):
    inp = inp.replace(' ', '_').lower()  # Make it lowercase to handle case sensitivity
    pattern = f"^{inp}"
    matched = [item for item in dis_list if re.search(pattern, item.lower())]
    if matched:
        return True, matched
    return False, []

# Enhanced prediction function with multiple symptoms, severity, and days consideration
def get_disease_prediction(symptom_inputs, days):
    chk_dis = list(training.columns[:-1])  # All symptom columns except the 'prognosis'
    
    # Initialize the input vector for prediction
    symptoms_dict = {symptom: index for index, symptom in enumerate(chk_dis)}
    input_vector = np.zeros(len(symptoms_dict))  # Create an input vector of zeros
    
    valid_symptoms = []
    severity_factor = 0  # Total severity factor to consider for final decision

    for symptom_input in symptom_inputs:
        is_valid, matched_symptoms = check_pattern(chk_dis, symptom_input)
        if is_valid:
            matched_symptom = matched_symptoms[0]  # Pick the first matched symptom
            input_vector[symptoms_dict[matched_symptom]] = 1  # Set the corresponding index to 1
            valid_symptoms.append(matched_symptom)

            # Adjust severity factor for the matched symptom
            severity = severity_dict.get(matched_symptom, 1)  # Default to 1 if severity is missing
            severity_factor += severity * (days / 10)  # Scale by days (more days -> higher severity)
        else:
            return f"Symptom '{symptom_input}' not recognized. Please try again."

    if not valid_symptoms:
        return "No valid symptoms provided. Please enter recognized symptoms."

    # Predict using the trained decision tree model
    disease_prediction = clf.predict([input_vector])
    predicted_disease = le.inverse_transform(disease_prediction)[0]

    # Format the response message
    response = f"💡 **Prediction:** Based on your symptoms ({', '.join(valid_symptoms)}) for {days} days, the predicted disease is **{predicted_disease}** (severity factor: {severity_factor:.2f})."
    
    # Provide additional details
    if predicted_disease in description_list:
        response += f"\n\n📝 **Description:** {description_list[predicted_disease]}"
    if predicted_disease in precaution_dict:
        precautions = "\n".join([f"🔹 {p}" for p in precaution_dict[predicted_disease]])
        response += f"\n\n💊 **Precautions:**\n{precautions}"
    
    return response

# Streamlit UI
st.title("Healthcare Chatbot 💬")
st.write("This chatbot helps diagnose and provide suggestions based on your symptoms. 🚑")

# User Name Input
name = st.text_input("👤 What is your name?", value="", placeholder="Enter your name")
if name:
    st.write(f"Hello, {name}! Please describe your symptoms below. 👇")

# Multi-Select for Multiple Symptoms Input
symptom_inputs = st.multiselect(
    "🩺 Start typing your symptoms:",
    options=list(training.columns[:-1])  # All symptoms as options
)

# Days Slider
days = st.slider("📅 For how many days have you had these symptoms?", 1, 30, 1)

# Submit Button
if st.button("🔍 Submit"):
    if symptom_inputs:
        response = get_disease_prediction(symptom_inputs, days)
        st.write(response)
    else:
        st.write("⚠️ Please select or type your symptoms to proceed.")
