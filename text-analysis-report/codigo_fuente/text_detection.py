# codigo_fuente/text_detection.py

def detect_or_build_text(df, settings):
    """
    Detecta automáticamente la columna de texto más narrativa
    o combina columnas si es necesario.
    """

    min_len = settings["min_text_length"]
    combine = settings["combine_columns"]
    forced_column = settings.get("text_column")

    # 1) Si el usuario pasa la columna explícitamente
    if forced_column:
        if forced_column not in df.columns:
            raise ValueError(f"La columna '{forced_column}' no existe.")
        return df[forced_column].astype(str).tolist(), forced_column

    # 2) Buscar columnas con texto narrativo auténtico
    candidates = []
    for col in df.columns:
        serie = df[col].astype(str)
        avg_len = serie.str.len().mean()
        avg_words = serie.str.split().apply(len).mean()

        if avg_len > min_len and avg_words > 3:
            candidates.append((col, avg_len))

    if candidates:
        best_col = max(candidates, key=lambda x: x[1])[0]
        print(f"[INFO] Columna narrativa detectada automáticamente: {best_col}")
        return df[best_col].astype(str).tolist(), best_col

    # 3) Fallback: combinar todas las columnas en un solo texto
    if combine:
        print("[INFO] No se encontró texto narrativo, combinando columnas...")
        combined = df.astype(str).apply(lambda row: " ".join(row.values), axis=1)
        return combined.tolist(), "combined_text"

    raise ValueError("No se encontró texto narrativo y combine_columns=False.")
