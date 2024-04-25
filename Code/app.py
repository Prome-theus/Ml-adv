from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import sklearn as sk
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from model import train_data

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def root():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    review = request.form["review"]
    processed_text = review.lower()

    tokens = word_tokenize(processed_text)

    tokens = [word for word in tokens if word.isalnum()]

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    token = [lemmatizer.lemmatize(word) for word in tokens]

    processed_text = ''.join(tokens)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(train_data)
    input_vector = tfidf_vectorizer.transform([processed_text])

    prediction = model.predict(input_vector)

    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
