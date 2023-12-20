import streamlit as st
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch

# Function to initialize the model and tokenizer
def load_model(model_path):
    tokenizer = CamembertTokenizer.from_pretrained(model_path)
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

# Function to predict the difficulty of the text
def predict_difficulty(model, tokenizer, text):
    inputs = tokenizer.encode_plus(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.argmax().item()

# User interface
st.title('French Text Difficulty Classification')

# Loading the model
model_path = './my_model/'
model, tokenizer = load_model(model_path)

# Mapping of CEFR labels
cefr_labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# User input
user_input = st.text_area("Enter your text here: ")
if st.button('Classify'):
    if user_input:
        predicted_index = predict_difficulty(model, tokenizer, user_input)
        st.write(f"The phrase is classified as a CEFR level: {cefr_labels[predicted_index]}")
    else:
        st.write("Please enter some text before classifying.")
