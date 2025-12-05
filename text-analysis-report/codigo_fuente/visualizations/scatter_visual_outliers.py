# scatter_visual_outliers.py — versión accesible con formas y paletas

import os
import pandas as pd
import altair as alt
import numpy as np

from codigo_fuente.config.config_loader import load_config

config = load_config()

# Configuración visual desde settings.json
vis_cfg = config.get("OUTLIERS_VIS", {})

NORMAL_SHAPE = vis_cfg.get("normal_shape", "circle")
OUTLIER_SHAPE = vis_cfg.get("outlier_shape", "diamond")


def generate_scatter_outliers(
    umap_coords,
    texts,
    topics,
    df_outliers,
    output_html="data/outputs/scatter_outliers.html"
):
    """
    Scatter UMAP con outliers resaltados usando:
      - Formas diferenciadas
      - Colores accesibles
      - Tooltips configurables
    """

    print("[INFO] Generando scatter UMAP con outliers resaltados...")

    os.makedirs("data/outputs/", exist_ok=True)

    # ----------------------------
    # Construir DataFrame base
    # ----------------------------
    df = pd.DataFrame({
        "UMAP1": umap_coords[:, 0],
        "UMAP2": umap_coords[:, 1],
        "Text": texts,
        "Dominant_Topic": topics.astype(str),
        "is_outlier": False,
        "votes": 0
    })

    # ----------------------------
    # Marcar outliers
    # ----------------------------
    for _, row in df_outliers.iterrows():
        idx = int(row["idx"])
        df.loc[idx, "is_outlier"] = True
        if "votes" in df_outliers.columns:
            df.loc[idx, "votes"] = row["votes"]

    # ----------------------------
    # Tooltips dinámicos
    # ----------------------------
    tooltip_fields = []

    if vis_cfg.get("enable_tooltip", True):
        tooltip_fields.append(alt.Tooltip("Text:N", title="Texto"))
        tooltip_fields.append(alt.Tooltip("Dominant_Topic:N", title="Tópico"))

        if vis_cfg.get("show_votes_in_tooltip", True):
            tooltip_fields.append(alt.Tooltip("votes:Q", title="Coincidencia Detectores"))

    # ----------------------------
    # Capa NORMAL
    # ----------------------------
    normal_layer = (
        alt.Chart(df[df["is_outlier"] == False])
        .mark_point(
            size=vis_cfg.get("normal_point_size", 40),
            shape=NORMAL_SHAPE,
            opacity=vis_cfg.get("normal_point_opacity", 0.25),
            filled=True
        )
        .encode(
            x="UMAP1:Q",
            y="UMAP2:Q",
            color=alt.value(vis_cfg.get("normal_point_color", "#bbbbbb")),
            tooltip=tooltip_fields
        )
    )

    # ----------------------------
    # Capa OUTLIER
    # ----------------------------
    outlier_layer = (
        alt.Chart(df[df["is_outlier"] == True])
        .mark_point(
            size=vis_cfg.get("outlier_point_size", 120),
            shape=OUTLIER_SHAPE,
            opacity=1.0,
            filled=True
        )
        .encode(
            x="UMAP1:Q",
            y="UMAP2:Q",
            color=alt.value(vis_cfg.get("outlier_point_color", "#D62728")),
            tooltip=tooltip_fields
        )
    )

    # ----------------------------
    # Combinar capas
    # ----------------------------
    chart = (
        normal_layer + outlier_layer
    ).properties(
        width=700,
        height=450,
        title=vis_cfg.get("scatter_title", "Outliers resaltados en UMAP")
    ).interactive()

    # ----------------------------
    # Guardar HTML
    # ----------------------------
    chart.save(output_html)
    print(f"[OK] Scatter con outliers guardado en: {output_html}")

    return {
        "html": output_html,
        "df": df
    }
