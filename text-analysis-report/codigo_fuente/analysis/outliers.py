# outliers.py

import os
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from codigo_fuente.config.config_loader import load_config

config = load_config()

out_cfg = config["OUTLIERS"]
score_cfg = config["OUTLIERS_SCORE"]


# ======================================================
# DETECTORES INDIVIDUALES
# ======================================================

def detect_low_topic_confidence(doc_topic_matrix):
    max_prob = doc_topic_matrix.max(axis=1)
    idx = np.where(max_prob < out_cfg["topic_confidence_threshold"])[0]
    return idx, max_prob


def detect_umap_distance_outliers(umap_coords):
    d = np.linalg.norm(umap_coords - umap_coords.mean(axis=0), axis=1)
    z = (d - d.mean()) / d.std()
    idx = np.where(z > out_cfg["umap_zscore_threshold"])[0]
    return idx, d, z


def detect_dbscan(umap_coords):
    db = DBSCAN(
        eps=out_cfg["dbscan_eps"],
        min_samples=out_cfg["dbscan_min_samples"]
    ).fit(umap_coords)

    labels = db.labels_
    noise_idx = np.where(labels == -1)[0]
    return noise_idx, labels


def detect_iforest(umap_coords):
    model = IsolationForest(
        contamination=out_cfg["iforest_contamination"],
        random_state=42
    )
    labels = model.fit_predict(umap_coords)
    idx = np.where(labels == -1)[0]
    return idx, labels, model


def detect_oneclass_svm(umap_coords):
    oc = OneClassSVM(
        nu=out_cfg["ocsvm_nu"],
        kernel=out_cfg["ocsvm_kernel"]
    )
    labels = oc.fit_predict(umap_coords)
    idx = np.where(labels == -1)[0]
    return idx, labels, oc


# ======================================================
# FUNCIÓN PRINCIPAL (A + C + SISTEMA PONDERADO)
# ======================================================

def detect_outliers(umap_coords, doc_topic_matrix, texts):

    print("[INFO] Detectando outliers con sistema ponderado...")

    if umap_coords.shape[0] < 5:
        print("[WARN] Muy pocos puntos para detección de outliers. Se omite detección.")
        empty_df = pd.DataFrame(columns=[
            "idx", "text", "dominant_topic", "votes", "score", "reason"
        ])
        out_path = "data/outputs/outliers.csv"
        empty_df.to_csv(out_path, index=False, encoding="utf-8")
        return empty_df, {}

    enabled = out_cfg["enabled_methods"]
    require_methods = out_cfg["require_methods"]
    weight_map = score_cfg["weights"]
    required_score = score_cfg["required_score"]

    # Guardar detecciones individuales
    outlier_votes = {}
    detector_results = {}

    # -----------------------------
    # 1. Topic Confidence
    # -----------------------------
    if enabled["topic_confidence"]:
        idx, maxp = detect_low_topic_confidence(doc_topic_matrix)
        outlier_votes["low_topic_confidence"] = set(idx)
        detector_results["low_topic_confidence"] = (idx, maxp)

    # -----------------------------
    # 2. UMAP Distance
    # -----------------------------
    if enabled["umap_distance"]:
        idx, dist, z = detect_umap_distance_outliers(umap_coords)
        outlier_votes["high_umap_distance"] = set(idx)
        detector_results["high_umap_distance"] = (idx, dist, z)

    # -----------------------------
    # 3. DBSCAN
    # -----------------------------
    if enabled["dbscan"]:
        idx, labels = detect_dbscan(umap_coords)
        outlier_votes["dbscan_noise"] = set(idx)
        detector_results["dbscan_noise"] = (idx, labels)

    # -----------------------------
    # 4. Isolation Forest
    # -----------------------------
    if enabled["isolation_forest"]:
        idx, labels, model = detect_iforest(umap_coords)
        outlier_votes["isolation_forest"] = set(idx)
        detector_results["isolation_forest"] = (idx, labels, model)

    # -----------------------------
    # 5. One-Class SVM
    # -----------------------------
    if enabled["oneclass_svm"]:
        idx, labels, model = detect_oneclass_svm(umap_coords)
        outlier_votes["oneclass_svm"] = set(idx)
        detector_results["oneclass_svm"] = (idx, labels, model)

    # ======================================================
    # CONTAR MÉTODOS Y COMPUTAR PUNTUACIÓN
    # ======================================================
    vote_count = {}
    score_sum = {}
    method_reasons = {}

    for method, indices in outlier_votes.items():
        for doc in indices:
            vote_count[doc] = vote_count.get(doc, 0) + 1
            score_sum[doc] = score_sum.get(doc, 0) + weight_map[method]
            method_reasons.setdefault(doc, []).append(method)

    # ======================================================
    # SELECCIONAR OUTLIERS SEGÚN:
    #   1. require_methods (A)
    #   2. required_score (Ponderado)
    # ======================================================
    final_outliers = [
        doc_id for doc_id in vote_count
        if vote_count[doc_id] >= require_methods
        and score_sum[doc_id] >= required_score
    ]

    # ======================================================
    # DETERMINAR LA RAZÓN PRINCIPAL
    # ======================================================
    def determine_reason(doc):
        methods = method_reasons.get(doc, [])
        if len(methods) == 0:
            return "unknown"
        if len(methods) == 1:
            return methods[0]
        return "multi_detector"

    reasons = {doc: determine_reason(doc) for doc in final_outliers}

    # ======================================================
    # CONSTRUIR DATAFRAME FINAL
    # ======================================================
    dominant = np.argmax(doc_topic_matrix, axis=1)

    df = pd.DataFrame({
        "idx": final_outliers,
        "text": [texts[i] for i in final_outliers],
        "dominant_topic": [dominant[i] for i in final_outliers],
        "votes": [vote_count[i] for i in final_outliers],
        "score": [score_sum[i] for i in final_outliers],
        "reason": [reasons[i] for i in final_outliers]
    })

    # Añadir scores individuales
    if enabled["topic_confidence"]:
        _, maxp = detector_results["low_topic_confidence"]
        df["topic_confidence"] = [maxp[i] for i in final_outliers]

    if enabled["umap_distance"]:
        _, dist, z = detector_results["high_umap_distance"]
        df["umap_distance"] = [dist[i] for i in final_outliers]
        df["umap_zscore"] = [z[i] for i in final_outliers]

    if enabled["dbscan"]:
        idx, labels = detector_results["dbscan_noise"]
        df["dbscan_noise"] = [1 if i in idx else 0 for i in final_outliers]

    if enabled["isolation_forest"]:
        idx, labels, _ = detector_results["isolation_forest"]
        df["isolation_forest"] = [labels[i] for i in final_outliers]

    if enabled["oneclass_svm"]:
        idx, labels, _ = detector_results["oneclass_svm"]
        df["oneclass_svm_flag"] = [labels[i] for i in final_outliers]

    # Guardar archivo
    out_path = "data/outputs/outliers.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Outliers detectados: {len(df)} (sistema ponderado)")
    print(f"[OK] Guardado en: {out_path}")

    return df, detector_results
