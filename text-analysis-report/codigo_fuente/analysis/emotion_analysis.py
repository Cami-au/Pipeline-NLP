# codigo_fuente/analysis/emotion_analysis.py

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Descargar diccionario si falta
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except:
    nltk.download("vader_lexicon")

# Ajustes de palabras comunes en español
SPANISH_FIX = {
    "bueno": 2.0,
    "excelente": 3.0,
    "malo": -2.5,
    "horrible": -3.0,
    "pésimo": -3.2,
    "caro": -1.5,
    "lento": -1.2,
    "rápido": 1.2,
    "agradable": 1.8,
    "enojado": -2.5,
    "feliz": 2.5,
    "contento": 2.2,
    "triste": -2.3
}


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()

    # Ajustes manuales al lexicón
    for palabra, score in SPANISH_FIX.items():
        if palabra in text.lower():
            sia.lexicon[palabra] = score

    comp = sia.polarity_scores(text)["compound"]

    if comp >= 0.05:
        label = "positivo"
    elif comp <= -0.05:
        label = "negativo"
    else:
        label = "neutral"

    return label, comp


def batch_analyze(texts):
    sentiments = []
    scores = []

    for t in texts:
        label, score = analyze_sentiment(t)
        sentiments.append(label)
        scores.append(score)

    return sentiments, scores
