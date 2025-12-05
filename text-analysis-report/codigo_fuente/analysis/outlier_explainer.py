# codigo_fuente/analysis/outlier_explainer.py

import pandas as pd
from openai import OpenAI

import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "\n[ERROR] No se encontró la variable de entorno OPENAI_API_KEY.\n"
        "Asegúrate de haber ejecutado:\n\n"
        "    setx OPENAI_API_KEY \"tu_api_key\"\n\n"
        "Luego CIERRA y vuelve a abrir la terminal o VSCode.\n"
    )

client = OpenAI(api_key=api_key)



SYSTEM_PROMPT = """
Eres un analista de datos experto en NLP. 
Tu tarea es explicar por qué un documento fue detectado como outlier.

Debes producir una explicación clara y ejecutiva, de máximo 5–6 líneas.
Evita lenguaje técnico innecesario. No inventes contenido que no aparece en el texto.
"""

def explain_outlier_gpt(text, topic_label, topic_keywords, topic_summary,
                        reason, umap_distance, entities):
    """
    Genera una explicación natural del outlier usando GPT.
    """

    prompt = f"""
Documento analizado:
{text}

Tópico dominante:
{topic_label}

Palabras clave del tópico:
{topic_keywords}

Resumen del tópico:
{topic_summary}

Entidades detectadas:
{entities}

Razón de detección del outlier:
{reason}

Distancia UMAP:
{umap_distance:.3f}

Explica en español, en un tono profesional y claro, por qué este documento fue considerado un outlier.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # ligero, rápido, barato
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"].strip()

    except Exception as e:
        return f"[ERROR al generar explicación: {e}]"
