# ngrams.py

import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

from codigo_fuente.config.config_loader import load_config

config = load_config()


# Paletas accesibles
COLOR_PALETTES = {
    "viridis",
    "cividis",
    "plasma",
    "inferno",
    "magma"
}


def validate_palette(palette):
    if palette not in COLOR_PALETTES:
        raise ValueError(
            f"Paleta '{palette}' no válida. Opciones: {', '.join(COLOR_PALETTES)}"
        )


# ---------------------------------------------------------
# Extraer n-gramas
# ---------------------------------------------------------
def extract_ngrams(texts, n=2, top_k=10):
    """
    Extrae los n-gramas más frecuentes usando CountVectorizer.
    """
    vec = CountVectorizer(ngram_range=(n, n))
    matrix = vec.fit_transform(texts)

    counts = matrix.sum(axis=0).A1
    vocab = vec.get_feature_names_out()

    freq = list(zip(vocab, counts))
    freq_sorted = sorted(freq, key=lambda x: x[1], reverse=True)

    return freq_sorted[:top_k]


# ---------------------------------------------------------
# Graficar n-gramas
# ---------------------------------------------------------
def plot_ngrams(ngrams, n, output_path):
    """
    Grafica bigramas o trigramas según configuración.
    """

    viz_cfg = config["VISUALIZATION"]
    palette = viz_cfg["global_palette"]
    rotation = viz_cfg["ngram_label_rotation"]

    validate_palette(palette)

    if not ngrams:
        raise ValueError("No hay n-gramas para graficar.")

    ngram_labels = [x[0] for x in ngrams]
    counts = [x[1] for x in ngrams]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(12, 6))
    color_map = plt.cm.get_cmap(palette)
    plt.bar(ngram_labels, counts, color=color_map(0.7))

    plt.xlabel(f"{n}-gramas")
    plt.ylabel("Frecuencia")
    plt.title(f"Top {len(ngrams)} {n}-gramas más frecuentes")
    plt.xticks(rotation=rotation, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

    return output_path


# ---------------------------------------------------------
# Función principal: genera bigramas y trigramas
# ---------------------------------------------------------
def generate_ngrams(texts):
    """
    Genera:
        - Top N bigramas
        - Top N trigramas
        - Gráficas respectivas

    Usando parámetros desde settings.json.
    """

    viz_cfg = config["VISUALIZATION"]

    top_k = viz_cfg["ngrams_top"]

    # 1. Extraer bigramas
    bigrams = extract_ngrams(texts, n=2, top_k=top_k)
    bigram_plot = plot_ngrams(bigrams, n=2, 
                              output_path="data/outputs/bigrams.png")

    # 2. Extraer trigramas
    trigrams = extract_ngrams(texts, n=3, top_k=top_k)
    trigram_plot = plot_ngrams(trigrams, n=3, 
                               output_path="data/outputs/trigrams.png")

    return {
        "bigrams": bigrams,
        "trigrams": trigrams,
        "bigrams_plot": bigram_plot,
        "trigrams_plot": trigram_plot
    }
