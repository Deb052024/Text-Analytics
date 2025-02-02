import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from textblob import TextBlob
nltk.download('punkt')

nltk.download("wordnet")
nltk.download("omw-1.4")


# Load your data
df = pd.read_csv(r'E:\DigiCrome\Summer Internship NextHikes\Project-7\Streamlit\twitter_disaster.csv')

from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    # Classify as positive, negative, or neutral based on polarity
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['text'].apply(get_sentiment)

# Group by sentiment and get the most frequent keyword and location for each sentiment
sentiment_groups = df.groupby('sentiment')
keyword_by_sentiment = sentiment_groups['keyword'].agg(lambda x: x.mode()[0] if x.mode().size > 0 else None)  # Handle empty modes
location_by_sentiment = sentiment_groups['location'].agg(lambda x: x.mode()[0] if x.mode().size > 0 else None)

# Fill null values based on sentiment
df['keyword'] = df.apply(lambda row: keyword_by_sentiment[row['sentiment']] if pd.isnull(row['keyword']) else row['keyword'], axis=1)
df['location'] = df.apply(lambda row: location_by_sentiment[row['sentiment']] if pd.isnull(row['location']) else row['location'], axis=1)


def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    return text

df['cleaned_text'] = df['text'].apply(clean_text)
# Find the most frequent value in 'cleaned_keyword' and 'cleaned_location'

# Load the saved model and vectorizer
with open('disaster_tweet_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Set background color
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;  # Light gray background
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Bold header
st.markdown("<h1 style='text-align: center; font-weight: bold; background-color: blue'>Disaster Tweet Classifier</h1>", unsafe_allow_html=True)
# Input text area
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    textarea {
        background-color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True # Correct comma placement
)
user_input = st.text_area("Enter a tweet:", "")

# Prediction button
if st.button("Predict"):
    if user_input:
        # Vectorize the input text
        input_vec = vectorizer.transform([user_input])

        # Make a prediction
        prediction = model.predict(input_vec)[0]

        # Display the prediction with conditional color
        if prediction == 1:
            st.markdown("<p style='color: red; font-size: 20px;'>This tweet is likely about a disaster.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: green; font-size: 20px;'>This tweet is likely not about a disaster.</p>", unsafe_allow_html=True)
    else:
        st.write("Please enter a tweet.")