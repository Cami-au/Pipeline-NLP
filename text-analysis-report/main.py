from codigo_fuente.pipeline import NLPReportPipeline
from codigo_fuente.config.config_loader import load_config

import argparse
from pathlib import Path
import pandas as pd


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Pipeline NLP completo con tópicos, n-gramas interactivos, outliers y reporte HTML."
    )

    # -------------------------
    # Archivo CSV
    # -------------------------
    parser.add_argument(
        "--file",
        required=False,
        help="Ruta al archivo CSV de entrada. "
             "Si se omite, se buscarán automáticamente CSV dentro de data/."
    )

    # -------------------------
    # Columna de texto
    # -------------------------
    parser.add_argument(
        "--textcol",
        required=False,
        help="Nombre de la columna de texto a usar. "
             "Si no se especifica, el sistema intentará detectarla."
    )

    # -------------------------
    # Título del reporte
    # -------------------------
    parser.add_argument(
        "--title",
        required=False,
        help="Título personalizado del reporte HTML."
    )

    # -------------------------
    # Número de tópicos
    # -------------------------
    parser.add_argument(
        "--topics",
        type=int,
        required=False,
        help="Número de tópicos para LDA (por ejemplo: 5, 10, 15)."
    )

    # -------------------------
    # Método de tópicos
    # -------------------------
    parser.add_argument(
        "--topic-method",
        choices=["lda", "bertopic"],
        required=False,
        help="Método de modelado de tópicos: 'lda' (rápido) o 'bertopic' (más pesado)."
    )

    # -------------------------
    # Paleta de colores (1–5)
    # -------------------------
    parser.add_argument(
        "--palette-id",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=False,
        help="ID de paleta de colores (1–5) definida en settings.json."
    )

    return parser.parse_args()


def main():
    args = parse_cli()
    config = load_config()

    # ======================================================
    # Overrides desde CLI
    # ======================================================
    if args.textcol:
        config["DATA_LOADING"]["text_column"] = args.textcol

    if args.title:
        config["REPORT"]["title"] = args.title

    if args.topics:
        config["TOPICS"]["num_topics"] = args.topics

    if args.topic_method:
        config["TOPICS"]["method"] = args.topic_method

    if args.palette_id:
        config["VISUALIZATION"]["palette_id"] = str(args.palette_id)  # mantenerlo como string (más seguro)

    # ======================================================
    # Selección de archivo CSV
    # ======================================================
    if not args.file:
        csvs = list(Path("data").glob("*.csv"))

        if len(csvs) == 0:
            raise FileNotFoundError("No se encontraron archivos CSV dentro de data/")

        elif len(csvs) == 1:
            args.file = str(csvs[0])
            print(f"[INFO] CSV detectado automáticamente: {args.file}")

        else:
            print("\n[INFO] CSV encontrados en carpeta data/:")
            for i, f in enumerate(csvs, 1):
                print(f"{i}. {f.name}")

            idx = int(input("Selecciona el número del archivo a procesar: "))
            args.file = str(csvs[idx - 1])

    # ======================================================
    # Ejecutar Pipeline
    # ======================================================
    pipeline = NLPReportPipeline(config, args)
    pipeline.run()


if __name__ == "__main__":
    main()
