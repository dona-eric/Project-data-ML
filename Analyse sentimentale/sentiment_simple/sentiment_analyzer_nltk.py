import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(page_title='Analyse Sentimentale', page_icon="😊")

st.title('Analyseur de Sentiments')
st.markdown('Une application simple pour analyser les sentiments dans un texte.')

# Télécharger les ressources nécessaires de NLTK
@st.cache_resource
def load_model():
    nltk.download('vader_lexicon')
    model = SentimentIntensityAnalyzer()
    return model

# Charger le modèle
model_analyze = load_model()

# Interface utilisateur
st.subheader("Entrez un texte pour analyser son sentiment:")
user_input = st.text_area("Tapez votre texte ici:")

# Bouton pour l'analyse
if st.button('Analyser sentiment'):
    if user_input:
        # Obtenir le score de sentiment
        result = model_analyze.polarity_scores(user_input)
        score = result['compound']
        
        # Déterminer le sentiment
        if score > 0.05:
            sentiment = "POSITIF 😊"
        elif score < -0.05:
            sentiment = "NÉGATIF 😟"
        elif score == 0:
            sentiment = "NEUTRALITÉ : Le modèle n’a pas détecté d’émotion forte. Ajoutez plus de contexte."
        else:
            sentiment = "NEUTRE 😐"
        
        # Afficher les résultats
        st.write("### Résultats de l'analyse:")
        st.write(f"**Sentiment :** {sentiment}")
        st.write(f"**Score :** {score}")
    else:
        st.warning("Veuillez entrer un texte pour l'analyse.")
