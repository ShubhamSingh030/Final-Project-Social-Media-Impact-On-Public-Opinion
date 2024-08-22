
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the TF-IDF Vectorizer
def load_vectorizer(vectorizer_path):
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

# Preprocess and vectorize input data
def preprocess_input(tweet, likes, retweets, hour, vectorizer):
    tweet_vectorized = vectorizer.transform([tweet])
    input_data = pd.DataFrame([[likes, retweets, hour]], columns=["likes", "retweets", "hour"])
    input_data = pd.concat([input_data, pd.DataFrame(tweet_vectorized.toarray())], axis=1)
    return input_data

# Predict influence score
def predict_influence(input_data, model):
    prediction = model.predict(input_data)
    return prediction[0]