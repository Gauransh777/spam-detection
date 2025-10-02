import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Initialize Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    """
    Transforms the input text by converting to lowercase, tokenizing,
    removing non-alphanumeric characters, stopwords, and punctuation,
    and applying stemming.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens and keep only alphabetic ones
    tokens = [i for i in tokens if i.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [i for i in tokens if i not in stop_words and i not in punctuation]

    # Apply stemming
    stemmed_tokens = [ps.stem(i) for i in tokens]

    # Return the cleaned text as a single string
    return " ".join(stemmed_tokens)

# --- Load Pre-trained Models ---
# Determine the absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to the model and vectorizer files
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

# Load the TF-IDF vectorizer and the prediction model
try:
    with open(vectorizer_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()


# --- Streamlit App Interface ---
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

st.title("📧 SMS Spam Classifier")
st.write("Enter a message below to determine if it's spam or not.")

# Text area for user input
input_sms = st.text_area("Enter the message:", height=150, key="input_sms")

# Predict button
if st.button('Predict', key="predict_button"):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess the input message
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize the transformed message
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict the result
        result = model.predict(vector_input)[0]
        
        # 4. Display the result
        if result == 1:
            st.error("This message is likely **Spam**.")
        else:
            st.success("This message is **Not Spam**.")
