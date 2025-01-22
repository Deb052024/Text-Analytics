import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')

nltk.download("wordnet")
nltk.download("omw-1.4")


# Load your data
df = pd.read_csv(r'E:\DigiCrome\Summer Internship NextHikes\Project-7\Streamlit\twitter_disaster.csv')

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    return text

df['cleaned_text'] = df['text'].apply(clean_text)
# Find the most frequent value in 'cleaned_keyword' and 'cleaned_location'
def clean_keyword(keyword):
    if isinstance(keyword, str):
        keyword = re.sub(r'http\S+', '', keyword)
        keyword = re.sub(r'[^\w\s]', '', keyword)
        return keyword
    else:
        return keyword
df['cleaned_keyword'] = df['keyword'].apply(clean_keyword)
def clean_location(location):
    if isinstance(location, str):
        location = re.sub(r'http\S+', '', location)
        location = re.sub(r'[^\w\s]', '', location)
        return location
    else:
        return location

df['cleaned_location'] = df['location'].apply(clean_location)
keyword_mode = df['keyword'].mode()
location_mode = df['location'].mode()

# Replace blank spaces and NaN values with the mode
df['cleaned_keyword'] = df['cleaned_keyword'].replace([' ', np.nan], keyword_mode[0])
df['cleaned_location'] = df['cleaned_location'].replace([' ', np.nan], location_mode[0])



df1 = df[['id', 'target', 'cleaned_keyword', 'cleaned_location', 'cleaned_text']]
df1 = df1.rename(columns={'cleaned_keyword': 'keyword', 'cleaned_location': 'location', 'cleaned_text': 'text'})

# Load the saved model and vectorizer
with open('disaster_tweet_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Create the Streamlit app
st.title("Disaster Tweet Classifier")

# Input text area for the user
user_input = st.text_area("Enter a tweet:", "")

# Prediction button
if st.button("Predict"):
    if user_input:
        # Vectorize the input text
        input_vec = vectorizer.transform([user_input])

        # Make a prediction
        prediction = model.predict(input_vec)[0]

        # Display the prediction
        if prediction == 1:
            st.write("This tweet is likely about a disaster.")
        else:
            st.write("This tweet is likely not about a disaster.")
    else:
        st.write("Please enter a tweet.")