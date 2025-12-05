# =============================================================
# topics.py — versión universal robusta con:
# - LDA
# - BERTopic avanzado
# - Embeddings multilingües fuertes
# - Fallback automático si BERTopic falla
# - Forzado de nr_topics con KMeans
# - Títulos GPT + resúmenes GPT
# - Etiquetas heurísticas
# =============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from pathlib import Path
from codigo_fuente.config.config_loader import load_config
from codigo_fuente.analysis.topic_summarizer import summarize_topic_gpt

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
config = load_config()

OUTPUT_DIR = Path("data/outputs/topics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# UTILIDAD: Guardar CSV
# -------------------------------------------------------------
def _save_csv(df, filename: str):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# -------------------------------------------------------------
# ETIQUETAS HEURÍSTICAS
# -------------------------------------------------------------
def auto_label_topic(keywords):
    kws = [k.lower() for k in keywords]

    def has_any(words):
        return any(w in kws for w in words)

    if has_any(["fila", "filas", "espera", "tiempo"]):
        return "Filas y tiempos de espera"
    if has_any(["comida", "restaurante", "hamburguesa", "bebida"]):
        return "Comida y restaurantes"
    if has_any(["precio", "precios", "caro", "costo"]):
        return "Precios y costos"
    if has_any(["servicio", "personal", "empleado", "atención", "trato"]):
        return "Atención y servicio"
    if has_any(["atracción", "juego", "superman", "medusa", "montaña"]):
        return "Atracciones"
    if has_any(["seguridad", "riesgo", "accidente"]):
        return "Seguridad"
    if has_any(["limpieza", "sucio", "basura"]):
        return "Limpieza"

    # fallback genérico
    if len(kws) >= 3:
        return f"Tópico sobre: {kws[0]}, {kws[1]}, {kws[2]}"
    return f"Tópico sobre: {', '.join(kws)}"


# -------------------------------------------------------------
# PREPARAR OUTPUT (GPT incluido)
# -------------------------------------------------------------
def _prepare_outputs(topic_keywords, doc_topic_matrix, representative_docs):
    """
    topic_keywords: lista de dicts {topic, keywords}
    doc_topic_matrix: np.array [n_docs x n_topics]
    representative_docs: lista de dicts {topic, representative_doc}
    """
    topic_labels = {}
    titles = {}
    summaries = {}

    for entry in topic_keywords:
        tid = entry["topic"]
        kws = entry["keywords"]

        # etiqueta heurística
        label = auto_label_topic(kws)
        entry["label"] = label
        topic_labels[tid] = label

        # RESUMEN GPT (usa topic_summarizer.py)
        title, summary = summarize_topic_gpt(kws)
        entry["title"] = title
        entry["summary"] = summary

        titles[tid] = title
        summaries[tid] = summary

    # Tabla principal de tópicos
    df_kw = pd.DataFrame(topic_keywords)
    _save_csv(df_kw, "topic_keywords.csv")

    # Documentos representativos
    df_rep = pd.DataFrame(representative_docs)
    _save_csv(df_rep, "topic_representative_docs.csv")

    # Ablación: palabras únicas por tópico
    topic_sets = [set(kw_list) for kw_list in df_kw["keywords"]]
    unique_words = []

    for i, s in enumerate(topic_sets):
        others = set.union(*(topic_sets[j] for j in range(len(topic_sets)) if j != i))
        uniq = list(s - others)
        unique_words.append(uniq)

    df_ablation = pd.DataFrame({
        "topic": list(range(len(unique_words))),
        "unique_words": unique_words
    })
    _save_csv(df_ablation, "topic_ablation.csv")

    return {
        "topic_keywords": df_kw,
        "topic_representative_docs": df_rep,
        "topic_ablation": df_ablation,
        "doc_topic_matrix": doc_topic_matrix,
        "topic_labels": topic_labels,
        "titles": titles,
        "summaries": summaries
    }


# -------------------------------------------------------------
# LDA
# -------------------------------------------------------------
def _run_lda(processed_texts):
    print("[INFO] Ejecutando LDA...")

    cfg = config["TOPICS"]

    vectorizer = CountVectorizer(
        max_features=cfg["max_features"],
        min_df=cfg["min_df"],
        max_df=cfg["max_df"]
    )

    X = vectorizer.fit_transform(processed_texts)

    lda = LatentDirichletAllocation(
        n_components=cfg["num_topics"],
        random_state=42
    ).fit(X)

    words = vectorizer.get_feature_names_out()

    topic_keywords = []
    for tid, topic in enumerate(lda.components_):
        top_ids = topic.argsort()[-10:][::-1]
        top_words = [words[i] for i in top_ids]

        topic_keywords.append({
            "topic": tid,
            "keywords": top_words
        })

    doc_topic_matrix = lda.transform(X)

    rep_docs = []
    for tid in range(cfg["num_topics"]):
        idx = int(np.argmax(doc_topic_matrix[:, tid]))
        rep_docs.append({
            "topic": tid,
            "representative_doc": processed_texts[idx]
        })

    return _prepare_outputs(topic_keywords, doc_topic_matrix, rep_docs)


# -------------------------------------------------------------
# BERTopic con fallback universal
# -------------------------------------------------------------
def _run_bertopic(processed_texts):
    print("[INFO] Ejecutando BERTopic robusto...")

    # EMBEDDINGS MULTILINGÜES (universal)
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedder.encode(processed_texts, show_progress_bar=True)

    # CONFIG UMAP + HDBSCAN
    umap_model = UMAP(
        n_neighbors=config["UMAP"]["n_neighbors"],
        min_dist=config["UMAP"]["min_dist"],
        metric=config["UMAP"]["metric"],
        random_state=config["UMAP"]["random_state"]
    )

    hdbscan_cfg = config.get("HDBSCAN", {})
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_cfg.get("min_cluster_size", 5),
        min_samples=hdbscan_cfg.get("min_samples", 1)
    )

    # Modelo BERTopic
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True
    )

    topics, probabilities = topic_model.fit_transform(processed_texts, embeddings)
    unique_topics = set(topics) - {-1}

    # -------------------------------------------------
    # FALLBACK: Si solo detectó 0 o 1 tópico
    # -------------------------------------------------
    if len(unique_topics) <= 1:
        print("[WARN] BERTopic detectó 0/1 tópico — usando fallback automático.")

        # 1) Fallback a LDA
        if config["TOPICS"].get("fallback_to_lda", True):
            print("[INFO] → Cambiando a LDA por fallback.")
            return _run_lda(processed_texts)

        # 2) Opción: forzar KMeans con k tópicos
        if config["TOPICS"].get("force_topic_count", False):
            k = config["TOPICS"]["num_topics"]
            print(f"[INFO] → Forzando {k} tópicos vía KMeans.")

            km = KMeans(n_clusters=k, random_state=42)
            forced_topics = km.fit_predict(embeddings)

            forced_prob = np.zeros((len(processed_texts), k))
            for i, t in enumerate(forced_topics):
                forced_prob[i, t] = 1.0

            groups = {t: [] for t in range(k)}
            for text, t in zip(processed_texts, forced_topics):
                groups[t].append(text)

            topic_keywords = []
            for t in range(k):
                words = " ".join(groups[t]).split()
                freq = pd.Series(words).value_counts().head(10).index.tolist()
                topic_keywords.append({"topic": t, "keywords": freq})

            rep_docs = [{"topic": t, "representative_doc": groups[t][0]} for t in range(k)]

            return _prepare_outputs(topic_keywords, forced_prob, rep_docs)

        # Si llega aquí, no hay fallback definido
        print("[WARN] Sin fallback definido; devolviendo salida vacía básica.")
        n_docs = len(processed_texts)
        doc_topic_matrix = np.ones((n_docs, 1))
        topic_keywords = [{"topic": 0, "keywords": []}]
        rep_docs = [{"topic": 0, "representative_doc": processed_texts[0] if n_docs else ""}]
        return _prepare_outputs(topic_keywords, doc_topic_matrix, rep_docs)

    # -------------------------------------------------
    # CASO NORMAL BERTopic
    # -------------------------------------------------
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info.Topic != -1]["Topic"].tolist()

    topic_keywords = []
    for tid in valid_topics:
        words = [w for w, _ in topic_model.get_topic(tid)]
        topic_keywords.append({
            "topic": int(tid),
            "keywords": words[:10]
        })

    rep_docs = []
    topics_arr = np.array(topics)
    for tid in valid_topics:
        indices = np.where(topics_arr == tid)[0]
        best_idx = int(indices[0]) if len(indices) > 0 else None
        rep_docs.append({
            "topic": int(tid),
            "representative_doc": processed_texts[best_idx] if best_idx is not None else ""
        })

    # Normalizar matriz
    doc_topic_matrix = np.array(probabilities)
    if doc_topic_matrix.ndim == 1:
        doc_topic_matrix = doc_topic_matrix.reshape(-1, 1)

    if doc_topic_matrix.shape[0] == 0:
        doc_topic_matrix = np.zeros((len(processed_texts), 1))

    row_sums = doc_topic_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    doc_topic_matrix = doc_topic_matrix / row_sums

    return _prepare_outputs(topic_keywords, doc_topic_matrix, rep_docs)


# -------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -------------------------------------------------------------
def generate_topics(processed_texts, method=None, num_topics=None):
    """
    Wrapper principal usado por el pipeline.
    method: 'lda' o 'bertopic' (si None, usa settings.json)
    num_topics: para LDA y para forzar KMeans si se usa force_topic_count.
    """
    cfg = config["TOPICS"]

    if method is not None:
        cfg["method"] = method
    if num_topics is not None:
        cfg["num_topics"] = num_topics

    method_effective = cfg.get("method", "bertopic").lower()

    if method_effective == "lda":
        return _run_lda(processed_texts)

    if method_effective == "bertopic":
        return _run_bertopic(processed_texts)

    raise ValueError("Método no válido. Usa 'lda' o 'bertopic'.")
