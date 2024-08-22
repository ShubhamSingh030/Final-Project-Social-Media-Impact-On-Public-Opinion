import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open('final_model.pkl', 'rb') as file:
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

