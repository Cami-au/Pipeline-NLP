from pathlib import Path
import json

def load_config():
    """
    Carga settings.json desde la carpeta /config en la raíz del proyecto,
    sin importar desde dónde se ejecute el script.
    """

    # Obtener raíz del proyecto (dos niveles arriba de este archivo)
    project_root = Path(__file__).resolve().parents[2]

    # Ruta real del settings.json
    config_path = project_root / "config" / "settings.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontró settings.json en:\n{config_path}\n"
            "Verifica que el archivo esté en /config/settings.json"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
