# codigo_fuente/visualizations/scatter_topics.py

import os
import pandas as pd
import altair as alt
import numpy as np
import umap

from codigo_fuente.config.config_loader import load_config
config = load_config()

# Config visual
vis_cfg = config.get("VISUALIZATION", {})
PALETTE = vis_cfg.get("topic_palette", "set2")
SHAPES = vis_cfg.get("shapes", ["circle", "square", "triangle", "diamond", "cross", "star"])


def generate_scatter_topics(
    df,
    feature_matrix,
    output_html="data/outputs/scatter_topics.html"
):
    print("[INFO] Generando scatter UMAP semántico con etiquetas...")

    os.makedirs("data/outputs/", exist_ok=True)

    # ================================================
    # 1. Determinar tópico dominante
    # ================================================
    if "Dominant_Topic" not in df.columns:
        raise ValueError("[ERROR] El DataFrame no contiene la columna 'Dominant_Topic'.")

    dominant_topics = df["Dominant_Topic"].astype(str).values

    # ================================================
    # 2. Calcular UMAP si aún no existe
    # ================================================
    if "UMAP1" not in df.columns or "UMAP2" not in df.columns:
        print("[INFO] Calculando UMAP...")

        reducer = umap.UMAP(
            n_neighbors=config["UMAP"]["n_neighbors"],
            min_dist=config["UMAP"]["min_dist"],
            metric=config["UMAP"]["metric"],
            random_state=config["UMAP"]["random_state"]
        )

        umap_coords = reducer.fit_transform(feature_matrix)
        df["UMAP1"] = umap_coords[:, 0]
        df["UMAP2"] = umap_coords[:, 1]

    else:
        print("[INFO] Usando UMAP existente.")
        umap_coords = df[["UMAP1", "UMAP2"]].values

    # ================================================
    # 3. Preparar DataFrame para la gráfica
    # ================================================
    df_plot = pd.DataFrame({
        "UMAP1": df["UMAP1"],
        "UMAP2": df["UMAP2"],
        "Dominant_Topic": dominant_topics,
        "Text": df["procesado"]
    })

    # ================================================
    # 4. Calcular centroides por tópico
    # ================================================
    centroids = (
        df_plot.groupby("Dominant_Topic")[["UMAP1", "UMAP2"]]
        .mean()
        .reset_index()
    )
    centroids["label"] = "Tópico " + centroids["Dominant_Topic"].astype(str)

    # ================================================
    # 5. Capa de puntos
    # ================================================
    points = (
        alt.Chart(df_plot)
        .mark_point(size=60, filled=True)
        .encode(
            x="UMAP1:Q",
            y="UMAP2:Q",
            color=alt.Color("Dominant_Topic:N", scale=alt.Scale(scheme=PALETTE)),
            shape=alt.Shape("Dominant_Topic:N", scale=alt.Scale(range=SHAPES)),
            tooltip=[
                alt.Tooltip("Dominant_Topic:N", title="Tópico"),
                alt.Tooltip("Text:N", title="Texto procesado")
            ]
        )
    )

    # ================================================
    # 6. Capa de centroides
    # ================================================
    centroid_mark = (
        alt.Chart(centroids)
        .mark_point(size=250, color="black", opacity=0.35)
        .encode(
            x="UMAP1:Q",
            y="UMAP2:Q"
        )
    )

    centroid_labels = (
        alt.Chart(centroids)
        .mark_text(
            fontSize=16,
            fontWeight="bold",
            dy=-10,
            color="black"
        )
        .encode(
            x="UMAP1:Q",
            y="UMAP2:Q",
            text="label:N"
        )
    )

    # ================================================
    # 7. Combinar capas
    # ================================================
    chart = (
        (points + centroid_mark + centroid_labels)
        .properties(
            width=750,
            height=500,
            title="Distribución UMAP semántica por Tópicos"
        )
        .interactive()
    )

    # Guardar
    chart.save(output_html)
    print(f"[OK] Scatter UMAP guardado en: {output_html}")

    # ================================================
    # 8. Devolver umap_coords para el pipeline
    # ================================================
    return {
        "html": output_html,
        "umap_coords": np.array(umap_coords)  # ← ESTA CLAVE ES ESENCIAL
    }
