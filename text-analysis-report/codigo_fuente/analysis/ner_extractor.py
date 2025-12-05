# codigo_fuente/analysis/ner_extractor.py

import spacy
import pandas as pd
from collections import Counter, defaultdict

# Modelo liviano en español
# (rápido y suficiente para entidades estándar)
try:
    nlp = spacy.load("es_core_news_sm")
except:
    import os
    os.system("python -m spacy download es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")


# ============================================================
# Extraer entidades de un solo documento
# ============================================================

def extract_entities(text):
    """
    Retorna entidades en forma:
    [("Superman", "ORG"), ("Pueblo Vaquero", "LOC"), ...]
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ============================================================
# NER por tópico dominante
# ============================================================

def ner_by_topic(texts, dominant_topics):
    """
    Recibe:
      - texts: lista de textos procesados o brutos
      - dominant_topics: lista de índices de tópicos dominantes

    Devuelve:
      - DataFrame con entidades agrupadas por tópico
      - Diccionario topic_id → lista de entidades
    """

    topic_entities = defaultdict(list)

    # Extraer entidades por documento
    for idx, text in enumerate(texts):
        entities = extract_entities(text)
        tid = int(dominant_topics[idx])

        for ent_text, ent_label in entities:
            topic_entities[tid].append((ent_text, ent_label))

    # Construir tabla final
    rows = []
    for tid, ents in topic_entities.items():
        counter = Counter(ents)
        for (ent_text, ent_label), freq in counter.most_common():
            rows.append({
                "topic": tid,
                "entity": ent_text,
                "label": ent_label,
                "frequency": freq
            })

    df = pd.DataFrame(rows, columns=["topic", "entity", "label", "frequency"])
    return df, topic_entities
