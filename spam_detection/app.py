import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Fetch stopwords once
stop_words = set(stopwords.words('english'))

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    # Return the cleaned text as a single string
    return " ".join(text)

# Load the vectorizer and model
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Fetch stopwords once
stop_words = set(stopwords.words('english'))

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    # Return the cleaned text as a single string
    return " ".join(text)

# Load the vectorizer and model

BASE_DIR = os.path.dirname(__file__)  # folder where app.py is

tfidf = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), 'rb'))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), 'rb'))


# Streamlit app setup
st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    # Vectorize the transformed message
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict the result
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# Streamlit app setup
st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    # Vectorize the transformed message
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict the result
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

