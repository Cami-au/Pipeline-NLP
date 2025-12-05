# codigo_fuente/visualizations/palettes.py

def resolve_visual_palettes(vis_cfg: dict):
    """
    Dado el bloque VISUALIZATION de settings.json, devuelve
    los nombres de paletas a usar para ngramas, bubble chart y tópicos.
    """

    palette_id = str(vis_cfg.get("palette_id", 1))
    palette_options = vis_cfg.get("palette_options", {})

    # Opción elegida
    chosen = palette_options.get(palette_id, {})

    ngram_palette = chosen.get("ngram_palette", vis_cfg.get("ngram_palette", "tableau10"))
    bubble_palette = chosen.get("bubble_palette", vis_cfg.get("bubble_palette", "cividis"))
    topic_palette = chosen.get("topic_palette", vis_cfg.get("topic_palette", "set2"))

    return {
        "ngram_palette": ngram_palette,
        "bubble_palette": bubble_palette,
        "topic_palette": topic_palette
    }
