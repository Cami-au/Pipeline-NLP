import os
import pandas as pd
import altair as alt


def generate_scatter_emotions(df, output_html="data/outputs/scatter_emotions.html"):
    """
    Genera un UMAP coloreado por sentimiento.
    Requiere que el DataFrame ya tenga columnas:
      - UMAP1, UMAP2
      - Sentiment ("positivo", "neutral", "negativo")
      - Sentiment_Score
      - procesado (texto)
    """

    print("[INFO] Generando UMAP coloreado por emoción...")

    os.makedirs("data/outputs/", exist_ok=True)

    if "UMAP1" not in df.columns or "UMAP2" not in df.columns:
        raise ValueError("No se encontraron columnas UMAP1/UMAP2 en el DataFrame.")

    if "Sentiment" not in df.columns:
        raise ValueError("No se encontró la columna 'Sentiment'. ¿EMOTIONS.enabled = true?")

    df_plot = pd.DataFrame({
        "UMAP1": df["UMAP1"],
        "UMAP2": df["UMAP2"],
        "Sentiment": df["Sentiment"],
        "Score": df.get("Sentiment_Score", 0),
        "Text": df["procesado"]
    })

    color_scale = alt.Scale(
        domain=["negativo", "neutral", "positivo"],
        range=["#d73027", "#cccccc", "#1a9850"]
    )

    chart = (
        alt.Chart(df_plot)
        .mark_circle(size=80, opacity=0.85)
        .encode(
            x="UMAP1:Q",
            y="UMAP2:Q",
            color=alt.Color("Sentiment:N", scale=color_scale, title="Emoción"),
            tooltip=[
                alt.Tooltip("Sentiment:N", title="Emoción"),
                alt.Tooltip("Score:Q", title="Intensidad"),
                alt.Tooltip("Text:N", title="Texto")
            ]
        )
        .properties(
            title="Mapa Emocional (UMAP por Sentimiento)",
            width=700,
            height=450
        )
        .interactive()
    )

    chart.save(output_html)
    print(f"[OK] UMAP emocional guardado en: {output_html}")

    return output_html
