import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentAnalyzer
from transformers import pipeline


st.set_page_config(page_title='Sentiment Analysis', page_icon = '')

st.markdown('Analysis TransformersSentiment')

def load_model():
    pipeline_model = pipeline('sentiment-analysis')
    return pipeline_model


def main():
    st.write('Enter your text, please analyze the sentiment')
    sentiment_analyzer = load_model()

    # entrer utilisateur
    user_input = st.write('Entrer un text:')
    if st.button('Analyzer sentiment'):
        if user_input:
            result = sentiment_analyzer(user_input)[0]

            # display result

            sentiment = result['label']
            confidence = result['score']

            st.write(f'Sentiment: {sentiment}')
            st.write(f'Confidence: {confidence: .2f}')

            if sentiment == 'POSITIVE':
                return 'Green'
            else:
                return 'Rouge'
        else:
            st.write('Please, enter some text')


if __name__ == "__main__":
    main()



