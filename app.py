import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load model
model = pickle.load(
    open("fake_news_model.pkl", "rb")
)

# Load vectorizer
vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

port_stem = PorterStemmer()

stop_words = set(
    stopwords.words('english')
)

def stemming(content):

    review = re.sub(
        '[^a-zA-Z]',
        ' ',
        str(content)
    )

    review = review.lower()

    words = review.split()

    stemmed_words = [
        port_stem.stem(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(stemmed_words)


st.title("📰 Fake News Detector")

st.write(
    "Enter any news article below and check whether it is Real or Fake."
)

news_text = st.text_area(
    "Paste News Here",
    height=250
)

if st.button("Check News"):

    cleaned_news = stemming(news_text)

    vector_news = vectorizer.transform(
        [cleaned_news]
    )

    prediction = model.predict(
        vector_news
    )

    probability = model.predict_proba(
        vector_news
    )

    confidence = round(
        max(probability[0]) * 100,
        2
    )

    if prediction[0] == 0:

        st.success(
            "✅ REAL NEWS"
        )

    else:

        st.error(
            "❌ FAKE NEWS"
        )

    st.write(
        f"Confidence Score: {confidence}%"
    )