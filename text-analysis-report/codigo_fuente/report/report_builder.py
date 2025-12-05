# report_builder.py — versión con WordCloud integrada

from pathlib import Path
import base64
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from codigo_fuente.config.config_loader import load_config

# Directorios
TEMPLATE_DIR = Path("templates")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config = load_config()

# -----------------------------------------------------------
# Utilidad para codificar imágenes PNG a base64
# -----------------------------------------------------------
def encode_image(path: Path):
    if not path or not path.exists() or not path.is_file():
        return None
    with open(path, "rb") as img:
        return "data:image/png;base64," + base64.b64encode(img.read()).decode()


# -----------------------------------------------------------
# Cargar CSVs si existen
# -----------------------------------------------------------
def load_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


# -----------------------------------------------------------
# Generar reporte HTML
# -----------------------------------------------------------
def build_report(
        vis_paths=None,
        scatter_topics_iframe="scatter_topics.html",
        scatter_outliers_iframe="scatter_outliers.html",
        title="Reporte NLP Interactivo"
    ):

    print("[INFO] Construyendo reporte con WordCloud...")

    if vis_paths is None:
        vis_paths = {}

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report_template.html")

    # imágenes opcionales
    wordcloud_img = encode_image(Path(vis_paths.get("wordcloud", "")))

    # visualizar CSVs
    df_keywords = load_csv(OUTPUT_DIR / "topics/topic_keywords.csv")
    df_ablation = load_csv(OUTPUT_DIR / "topics/topic_ablation.csv")
    df_represent = load_csv(OUTPUT_DIR / "topics/topic_representative_docs.csv")
    df_outliers = load_csv(OUTPUT_DIR / "outliers.csv")
    df_entities = load_csv(OUTPUT_DIR / "topic_entities.csv")

    keywords_table = df_keywords.to_html(index=False) if df_keywords is not None else "<p>No disponible.</p>"
    ablation_table = df_ablation.to_html(index=False) if df_ablation is not None else "<p>No disponible.</p>"
    rep_docs_table = df_represent.to_html(index=False) if df_represent is not None else "<p>No disponible.</p>"
    outliers_table = df_outliers.to_html(index=False) if df_outliers is not None else "<p>No disponible.</p>"
    entities_table = df_entities.to_html(index=False) if df_entities is not None else "<p>No disponible.</p>"

    background_path = config["REPORT"].get("background_image")
    if background_path:
        background_b64 = encode_image(Path("static") / background_path)
    else:
        background_b64 = None

    html = template.render(
        report_title=title,

        # imágenes estáticas
        wordcloud_img=wordcloud_img,

        # tablas
        keywords_table=keywords_table,
        ablation_table=ablation_table,
        rep_docs_table=rep_docs_table,
        outliers_table=outliers_table,
        entities_table=entities_table,

        # iframes
        bubble_chart_iframe="bubble_chart.html",
        top_words_iframe="top_words_interactive.html",
        bigrams_iframe="bigrams_interactive.html",
        trigrams_iframe="trigrams_interactive.html",
        scatter_topics_iframe=scatter_topics_iframe,
        scatter_outliers_iframe=scatter_outliers_iframe,

        background_b64=background_b64
    )

    output_file = OUTPUT_DIR / "report.html"
    output_file.write_text(html, encoding="utf-8")

    print(f"[OK] Reporte generado en: {output_file}")
    return str(output_file)
