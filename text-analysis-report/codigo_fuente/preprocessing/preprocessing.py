# preprocessing.py — versión final corregida para UMAP, BERTopic, NER y SBERT
import re
import spacy
import nltk
import subprocess
import sys
import unicodedata
from nltk.corpus import stopwords
import emoji

from codigo_fuente.config.config_loader import load_config
config = load_config()


# ---------------------------------------------------------
# CARGA DE MODELO SPACY (con instalación automática)
# ---------------------------------------------------------
def ensure_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"[INFO] Instalando modelo spaCy: {model_name}")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        return spacy.load(model_name)


# Descargar stopwords si faltan
try:
    stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords")


# ---------------------------------------------------------
# MANEJO DE EMOJOS
# ---------------------------------------------------------
def handle_emojis(text, mode):
    if mode == "remove":
        return emoji.replace_emoji(text, "")
    elif mode == "convert":
        return emoji.demojize(text)
    elif mode == "keep":
        return text
    return text


# ---------------------------------------------------------
# PREPROCESADOR PRINCIPAL
# ---------------------------------------------------------
class TextPreprocessor:

    def __init__(self):
        cfg = config["PREPROCESSING"]

        self.language = cfg["language"]
        self.min_token_length = cfg["min_token_length"]
        self.preserve_accents = cfg["preserve_accents"]
        self.return_text_only = cfg["return_text_only"]
        self.emoji_mode = cfg["emoji_handling"]

        # Cargar modelo spaCy segun idioma
        if self.language == "es":
            self.nlp = ensure_spacy_model("es_core_news_sm")
            base_stops = set(stopwords.words("spanish"))
        else:
            self.nlp = ensure_spacy_model("en_core_web_sm")
            base_stops = set(stopwords.words("english"))

        # Stopwords spaCy + NLTK
        spacy_stops = {w.lower() for w in self.nlp.Defaults.stop_words}
        self.stopwords = base_stops.union(spacy_stops)

        # personalizadas
        custom_stops = cfg.get("custom_stopwords", [])
        self.stopwords.update([w.lower() for w in custom_stops])

    # ---------------------------------------------------------
    # NORMALIZACIÓN DE ACENTOS (segura)
    # ---------------------------------------------------------
    def strip_accents(self, text):
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    # ---------------------------------------------------------
    # LIMPIEZA DE TEXTO (segura para embeddings)
    # ---------------------------------------------------------
    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)

        # emojis
        text = handle_emojis(text, self.emoji_mode)

        # minúsculas
        text = text.lower().strip()

        # URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # normalizar acentos
        if not self.preserve_accents:
            text = self.strip_accents(text)

        # permitir letras acentuadas, números y puntuación básica
        text = re.sub(r"[^a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ\s,.!?¿¡-]", " ", text)

        # espacios múltiples
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ---------------------------------------------------------
    # TOKENIZACIÓN + LEMMATIZACIÓN
    # ---------------------------------------------------------
    def tokenize_and_lemmatize(self, cleaned):
        doc = self.nlp(cleaned)
        tokens = []

        for tok in doc:

            if tok.is_space or tok.is_punct:
                continue

            lemma = tok.lemma_.lower().strip()

            # spaCy usa "-pron-" como marcador
            if lemma == "-pron-":
                lemma = tok.text.lower()

            # remover acentos si está configurado
            if not self.preserve_accents:
                lemma = self.strip_accents(lemma)

            if (
                lemma.isalpha()
                and lemma not in self.stopwords
                and len(lemma) >= self.min_token_length
            ):
                tokens.append(lemma)

        return tokens

    # ---------------------------------------------------------
    # PIPELINE COMPLETO
    # ---------------------------------------------------------
    def preprocess(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        lemmatized = " ".join(tokens)

        if self.return_text_only:
            return lemmatized

        return {
            "clean_text": cleaned,
            "tokens": tokens,
            "lemmatized_text": lemmatized,
        }
