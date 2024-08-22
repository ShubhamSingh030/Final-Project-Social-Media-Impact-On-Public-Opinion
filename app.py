
# Create the Streamlit app
# Streamlit is an open-source app framework for machine learning and data science projects.

# We can create a Streamlit app to showcase the results of the machine learning model and provide an interactive interface for users to explore the data.

# Streamlit App
# Create a Streamlit app to showcase the results of the machine learning model and provide an interactive interface for users to explore the data.

# Import the required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import shap

# Load the data
tweets_data = pd.read_csv('Cleaned Data/tweets_cleaned.csv')
covid_data = pd.read_csv('Cleaned Data/covid_tweets_cleaned.csv')
reddit_data = pd.read_csv('Cleaned Data/reddit_data_cleaned.csv')
twitter_data = pd.read_csv('Cleaned Data/twitter_data_cleaned.csv')

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit app title and description
st.title("Social Media Influence on Public Opinion")
st.write("Predict the influence score based on social media data.")

# Input fields for user input
tweet = st.text_area("Tweet Text", "")
likes = st.number_input("Number of Likes", min_value=0, step=1)
retweets = st.number_input("Number of Retweets", min_value=0, step=1)
hour = st.slider("Hour of the Day", 0, 23, 12)

# Function to preprocess and vectorize input data
def preprocess_input(tweet, likes, retweets, hour):
    tweet_vectorized = vectorizer.transform([tweet])
    input_data = pd.DataFrame([[likes, retweets, hour]], columns=["likes", "retweets", "hour"])
    input_data = pd.concat([input_data, pd.DataFrame(tweet_vectorized.toarray())], axis=1)
    return input_data

# Predict button
if st.button("Predict"):
    input_data = preprocess_input(tweet, likes, retweets, hour)
    prediction = model.predict(input_data)
    st.write(f"Predicted Influence: {prediction[0]}")



