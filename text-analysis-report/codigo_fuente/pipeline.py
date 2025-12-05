# codigo_fuente/pipeline.py

import os
from pathlib import Path
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from codigo_fuente.preprocessing.preprocessing import TextPreprocessor
from codigo_fuente.visualizations.visualizations import generate_all_visualizations
from codigo_fuente.analysis.topics import generate_topics
from codigo_fuente.visualizations.scatter_topics import generate_scatter_topics
from codigo_fuente.visualizations.scatter_visual_outliers import generate_scatter_outliers
from codigo_fuente.analysis.outliers import detect_outliers
from codigo_fuente.report.report_builder import build_report
from codigo_fuente.text_detection import detect_or_build_text
from codigo_fuente.analysis.ner_extractor import ner_by_topic
from codigo_fuente.analysis.outlier_explainer import explain_outlier_gpt


class NLPReportPipeline:

    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.df = None
        self.processed_texts = None
        self.text_source = None

        self.doc_topic_matrix = None
        self.umap_coords = None

        self.outliers_df = None
        self.df_ner = None   # tabla de entidades por tópico

    # ======================================================
    # 1. Cargar CSV con auto-encoding
    # ======================================================
    def load_csv(self):
        path = self.args.file
        encodings = ["utf-8", "latin1", "ISO-8859-1", "windows-1252"]

        for enc in encodings:
            try:
                self.df = pd.read_csv(path, encoding=enc)
                print(f"[OK] Archivo leído con encoding: {enc}")
                return
            except Exception:
                continue

        raise ValueError("No se pudo leer el CSV con ningún encoding válido.")

    # ======================================================
    # 2. Selección y preprocesado de texto
    # ======================================================
    def detect_text(self):
        raw_texts, src = detect_or_build_text(self.df, self.config["DATA_LOADING"])
        self.text_source = src
        print(f"[INFO] Columna usada para texto: {src}")

        pre = TextPreprocessor()
        self.processed_texts = [
            pre.preprocess(t)["lemmatized_text"] for t in raw_texts
        ]

        self.df["procesado"] = self.processed_texts

        # Guardar texto procesado
        os.makedirs("data/outputs/", exist_ok=True)
        self.df.to_csv("data/outputs/procesado.csv", index=False)
        print("[OK] Texto preprocesado guardado en data/outputs/procesado.csv")

        # --------------------------------------------------
        # 2.1 Análisis emocional (opcional)
        # --------------------------------------------------
        if self.config.get("EMOTIONS", {}).get("enabled", False):
            print("[INFO] Realizando análisis emocional...")
            from codigo_fuente.analysis.emotion_analysis import batch_analyze

            sentiments, scores = batch_analyze(self.processed_texts)
            self.df["Sentiment"] = sentiments
            self.df["Sentiment_Score"] = scores
            print("[OK] Análisis de emociones realizado.")
        else:
            print("[INFO] Análisis de emociones desactivado por configuración.")

    # ======================================================
    # 3. Visualizaciones interactivas
    # ======================================================
    def generate_visuals(self):
        return generate_all_visualizations(self.processed_texts)

    # ======================================================
    # 4. Modelado de tópicos
    # ======================================================
    def generate_topics(self):

        method = self.config["TOPICS"].get("method")
        num_topics = self.config["TOPICS"].get("num_topics")

        result = generate_topics(
            self.processed_texts,
            method=method,
            num_topics=num_topics
        )

        # Matriz documento–tópico (probabilidades)
        self.doc_topic_matrix = result["doc_topic_matrix"]

        # Tópico dominante por documento (argmax)
        self.df["Dominant_Topic"] = self.doc_topic_matrix.argmax(axis=1)

        # Etiquetas amigables
        topic_labels = result.get("topic_labels", {})
        self.df["Topic_Label"] = (
            self.df["Dominant_Topic"]
            .map(topic_labels)
            .fillna(self.df["Dominant_Topic"].astype(str))
        )

        # --------------------------------------------------
        # 4.5 NER por tópico (opcional)
        # --------------------------------------------------
        if self.config.get("NER", {}).get("enabled", True):
            print("[INFO] Ejecutando NER por tópico...")

            df_ner, ner_dict = ner_by_topic(
                self.df[self.text_source].astype(str).tolist(),
                self.df["Dominant_Topic"].tolist()
            )

            os.makedirs("data/outputs/", exist_ok=True)
            df_ner.to_csv("data/outputs/topic_entities.csv", index=False, encoding="utf-8")
            print("[OK] topic_entities.csv generado.")

            self.df_ner = df_ner
        else:
            print("[INFO] NER desactivado desde configuración.")

        return result

    # ======================================================
    # 5. Generación UMAP (semántico con SBERT)
    # ======================================================
    def generate_umap(self):
        print("[INFO] Generando UMAP semántico con embeddings SBERT...")

        # Embeddings SBERT a partir de texto preprocesado
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(self.processed_texts, show_progress_bar=True)

        # Generar scatter UMAP en espacio semántico
        result = generate_scatter_topics(self.df, embeddings)

        # Guardamos coordenadas para outliers
        self.umap_coords = result.get("umap_coords", None)

        if self.umap_coords is None:
            # Fallback: reconstruir a partir de columnas si existen
            if {"UMAP1", "UMAP2"}.issubset(self.df.columns):
                self.umap_coords = self.df[["UMAP1", "UMAP2"]].values
                print("[WARN] umap_coords no devuelto por la función, usando columnas UMAP1/UMAP2 del DataFrame.")
            else:
                raise ValueError("[ERROR] No se pudieron obtener las coordenadas UMAP.")

        return result

    # ======================================================
    # 6. Outliers
    # ======================================================
    def detect_outliers(self):

        if self.umap_coords.shape[0] < 5:
            print("[WARN] Muy pocos documentos para detección de outliers, creando archivo vacío.")

            empty_df = pd.DataFrame(columns=[
                "idx", "text", "dominant_topic", "votes", "score", "reason"
            ])

            empty_df.to_csv("data/outputs/outliers.csv", index=False, encoding="utf-8")
            self.outliers_df = empty_df
            return empty_df

        # Detección numérica de outliers
        self.outliers_df, _ = detect_outliers(
            self.umap_coords,
            self.doc_topic_matrix,
            self.df[self.text_source].astype(str).tolist()
        )

        os.makedirs("data/outputs/", exist_ok=True)
        self.outliers_df.to_csv("data/outputs/outliers.csv", index=False, encoding="utf-8")

        # Si no hay outliers, no intentamos explicar nada
        if self.outliers_df.empty:
            print("[INFO] No se detectaron outliers significativos. No se generan explicaciones GPT.")
            return self.outliers_df

        # --------------------------------------------------
        # Explicaciones GPT para cada outlier (con try/except)
        # --------------------------------------------------
        print("[INFO] Generando explicaciones GPT para outliers...")

        try:
            df_keywords = pd.read_csv("data/outputs/topics/topic_keywords.csv")
            topic_map = {
                row["topic"]: (row.get("label", "(sin etiqueta)"), row["keywords"])
                for _, row in df_keywords.iterrows()
            }
        except Exception:
            topic_map = {}
            print("[WARN] No se pudo cargar topic_keywords.csv; explicaciones usarán info mínima.")

        explanations = []

        for _, row in self.outliers_df.iterrows():
            topic_id = row.get("dominant_topic", 0)

            label, keywords = topic_map.get(topic_id, ("(sin etiqueta)", []))
            text = row.get("text", "")
            reason = row.get("reason", "")
            umap_distance = row.get("umap_distance", 0.0)

            # Entidades para ese tópico (si el CSV existe)
            ents = []
            try:
                if self.df_ner is not None:
                    ents = self.df_ner[self.df_ner["topic"] == topic_id]["entity"].tolist()
                else:
                    ent_df = pd.read_csv("data/outputs/topic_entities.csv")
                    ents = ent_df[ent_df["topic"] == topic_id]["entity"].tolist()
            except Exception:
                pass

            topic_summary = "(sin resumen disponible)"

            # Llamada protegida a GPT
            try:
                explanation = explain_outlier_gpt(
                    text=text,
                    topic_label=label,
                    topic_keywords=keywords,
                    topic_summary=topic_summary,
                    reason=reason,
                    umap_distance=umap_distance,
                    entities=ents
                )
            except Exception as e:
                # Si falla por cuota u otro error, no rompemos el pipeline
                print(f"[WARN] Falló explain_outlier_gpt para un outlier ({e}); usando explicación genérica.")
                explanation = (
                    f"Este texto fue marcado como outlier porque se comporta de forma distinta al resto "
                    f"en el espacio semántico y/o en los modelos de anomalía. "
                    f"Tópico aproximado: {label}. Razón técnica: {reason}."
                )

            explanations.append(explanation)

        self.outliers_df["explanation_gpt"] = explanations
        self.outliers_df.to_csv(
            "data/outputs/outliers_explanations.csv",
            index=False,
            encoding="utf-8"
        )

        print("[OK] outliers_explanations.csv generado.")
        return self.outliers_df

    # ======================================================
    # 7. Scatter de Outliers
    # ======================================================
    def plot_outliers(self):
        return generate_scatter_outliers(
            self.umap_coords,
            self.df[self.text_source],
            self.df["Dominant_Topic"],
            self.outliers_df
        )

    # ======================================================
    # 8. Construir Reporte HTML Final
    # ======================================================
    def build_report(self, visuals):
        build_report(
            vis_paths=visuals,
            scatter_topics_iframe="scatter_topics.html",
            scatter_outliers_iframe="scatter_outliers.html",
            title=self.config["REPORT"]["title"]
        )

    # ======================================================
    # ORQUESTADOR
    # ======================================================
    def run(self):
        self.load_csv()
        self.detect_text()
        visuals = self.generate_visuals()
        self.generate_topics()
        self.generate_umap()
        self.detect_outliers()
        self.plot_outliers()
        self.build_report(visuals)

        print("\n[✔] PROCESO COMPLETO. Reporte generado en data/outputs/report.html\n")
