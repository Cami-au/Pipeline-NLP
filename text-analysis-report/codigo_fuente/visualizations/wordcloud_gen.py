from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_wordcloud(texts):
    full_text = " ".join(texts)

    wc = WordCloud(
        width=1600,
        height=1000,
        background_color="white",
        colormap='viridis'
    ).generate(full_text)

    out_path = OUTPUT_DIR / "wordcloud.png"
    wc.to_file(out_path)
    return out_path
