# topic_summarizer.py — GPT + fallback automático

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", None))


def summarize_topic_gpt(keywords):
    """
    Intenta generar título + resumen usando GPT.
    Si falla (sin créditos, sin API KEY, error 429, etc.),
    regresa un resumen heurístico básico.
    """

    # Si no hay API KEY → fallback inmediato
    if client.api_key is None:
        return fallback_summary(keywords)

    prompt = (
        "Genera un título corto (máx 5 palabras) y un resumen de 15 palabras "
        "que describa el siguiente conjunto de palabras clave de un tópico:\n\n"
        f"{', '.join(keywords)}\n\n"
        "Formato:\n"
        "TITULO: ...\n"
        "RESUMEN: ..."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )

        text = resp.choices[0].message["content"]

        # Extraemos título y resumen
        lines = text.split("\n")
        title = lines[0].replace("TITULO:", "").strip()
        summary = lines[1].replace("RESUMEN:", "").strip()

        return title, summary

    except Exception as e:
        print(f"[WARN] GPT falló al resumir tópico: {e}")
        return fallback_summary(keywords)


# ==========================================================
# FALLBACK LOCAL (sin IA)
# ==========================================================
def fallback_summary(keywords):
    """
    Resumen local cuando GPT no funciona.
    """

    title = f"Tema sobre {keywords[0].capitalize()}"
    summary = (
        "Tópico basado en las palabras clave más frecuentes: "
        + ", ".join(keywords[:5])
    )

    return title, summary
