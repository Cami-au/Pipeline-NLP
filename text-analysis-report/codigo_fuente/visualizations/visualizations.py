# =============================================================
# visualizations.py — versión accesible con paletas seguras,
# WordCloud real y visualizaciones interactivas limpias.
# =============================================================

import os
import pandas as pd
import altair as alt
from collections import Counter

from wordcloud import WordCloud

from codigo_fuente.config.config_loader import load_config
from codigo_fuente.visualizations.palettes import resolve_visual_palettes


# =============================================================
# CONFIGURACIONES
# =============================================================
config = load_config()
VIS_CFG = config.get("VISUALIZATION", {})

_palette_cfg = resolve_visual_palettes(VIS_CFG)

NGRAM_PALETTE = _palette_cfg["ngram_palette"]
BUBBLE_PALETTE = _palette_cfg["bubble_palette"]

OUTPUT_DIR = "data/outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# UTILIDADES
# =============================================================
def compute_word_frequencies(texts):
    words = []
    for t in texts:
        words.extend(t.split())
    return Counter(words)


# =============================================================
# WORDCLOUD (IMAGEN ESTÁTICA)
# =============================================================
def generate_wordcloud(texts, output_path="data/outputs/wordcloud.png"):
    print("[INFO] Generando WordCloud...")

    text = " ".join(texts)

    wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        colormap="viridis"
    ).generate(text)

    wc.to_file(output_path)

    print(f"[OK] WordCloud guardada en: {output_path}")
    return output_path


# =============================================================
# BUBBLE CHART
# =============================================================
def generate_bubble_chart_interactive(processed_texts):
    print("[INFO] Generando Bubble Chart interactivo...")

    counts = compute_word_frequencies(processed_texts)
    top_words = counts.most_common(80)

    df = pd.DataFrame(top_words, columns=["word", "count"])

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("count:Q", title="Frecuencia"),
            y=alt.Y("count:Q", axis=None),
            size=alt.Size("count:Q", scale=alt.Scale(range=[50, 2000])),
            color=alt.Color("count:Q", scale=alt.Scale(scheme=BUBBLE_PALETTE)),
            tooltip=["word:N", "count:Q"]
        )
        .properties(
            width=850,
            height=480,
            title="Nube de Palabras (Bubble Chart Interactivo)"
        )
        .interactive()
    )

    output = f"{OUTPUT_DIR}/bubble_chart.html"
    chart.save(output)

    print(f"[OK] Bubble chart guardado en: {output}")
    return output


# =============================================================
# TOP WORDS — BARRA INTERACTIVA
# =============================================================
def generate_top_words_interactive(processed_texts):
    print("[INFO] Generando gráfico interactivo de Top Words...")

    counts = compute_word_frequencies(processed_texts)
    top_words = counts.most_common(20)
    df = pd.DataFrame(top_words, columns=["word", "count"])

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Frecuencia"),
            y=alt.Y("word:N", sort="-x", title="Palabra"),
            color=alt.Color("word:N", scale=alt.Scale(scheme=NGRAM_PALETTE)),
            tooltip=["word:N", "count:Q"]
        )
        .properties(width=700, height=450, title="Top 20 Palabras (Interactivo)")
    )

    output = f"{OUTPUT_DIR}/top_words_interactive.html"
    chart.save(output)

    print("[OK] top_words_interactive.html generado.")
    return output


# =============================================================
# N-GRAMS (BIGRAMAS / TRIGRAMAS)
# =============================================================
def generate_ngrams_interactive(processed_texts, n=2, output_file=""):
    print(f"[INFO] Generando gráfico interactivo de {n}-gramas...")

    ngrams_list = []
    for text in processed_texts:
        tokens = text.split()
        ngrams_list.extend([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    counts = Counter(ngrams_list).most_common(15)
    df = pd.DataFrame(counts, columns=["ngram", "count"])

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q"),
            y=alt.Y("ngram:N", sort="-x"),
            color=alt.Color("ngram:N", scale=alt.Scale(scheme=NGRAM_PALETTE)),
            tooltip=["ngram:N", "count:Q"]
        )
        .properties(
            width=700,
            height=450,
            title=f"Top {n}-gramas (Interactivo)"
        )
    )

    chart.save(output_file)
    print(f"[OK] {output_file} generado.")
    return output_file


# =============================================================
# GENERADOR GLOBAL
# =============================================================
def generate_all_visualizations(processed_texts):

    print("[INFO] Generando visualizaciones interactivas...")

    # WordCloud estática
    wordcloud_path = generate_wordcloud(processed_texts)

    # Bubble Chart
    bubble_chart = generate_bubble_chart_interactive(processed_texts)

    # Top Words
    top_words_html = generate_top_words_interactive(processed_texts)

    # Bigramas
    bigrams_html = generate_ngrams_interactive(
        processed_texts,
        n=2,
        output_file=f"{OUTPUT_DIR}/bigrams_interactive.html"
    )

    # Trigramas
    trigrams_html = generate_ngrams_interactive(
        processed_texts,
        n=3,
        output_file=f"{OUTPUT_DIR}/trigrams_interactive.html"
    )

    # RETORNAR TODO PARA EL REPORT BUILDER
    return {
        "wordcloud": wordcloud_path,
        "bubble_chart": bubble_chart,
        "top_words_interactive": top_words_html,
        "bigrams_interactive": bigrams_html,
        "trigrams_interactive": trigrams_html
    }
