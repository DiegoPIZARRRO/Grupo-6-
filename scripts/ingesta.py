import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd


# Definir la ruta raíz del proyecto
PROJECT_ROOT = Path.cwd()

# Carpetas del pipeline de datos.
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Nombre del archivo CSV de entrada.
INPUT_FILENAME = "telco.csv"

# Nombre del archivo de reporte de errores 
LOG_FILENAME = "reporte_errores_validacion.csv"

# Rutas completas.
INPUT_FILE = RAW_DIR / INPUT_FILENAME
LOG_FILE = REPORTS_DIR / LOG_FILENAME
README_FILE = PROJECT_ROOT / "README.md"


def preparar_carpetas() -> None:
    """
    Crea las carpetas necesarias para ejecutar el pipeline.

    Esta función no elimina ni reemplaza archivos existentes.
    Solo asegura que las carpetas requeridas estén disponibles.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def registrar_log(
    logs: list,
    nivel: str,
    etapa: str,
    columna: str,
    descripcion: str,
    filas_afectadas: int,
    accion: str
) -> None:
    """
    Agrega un registro al listado de logs del proceso.

    Parámetros:
    logs:
        Lista donde se guardan los eventos de validación.
    nivel:
        Nivel del evento. Ejemplos: INFO, WARNING, ERROR.
    etapa:
        Nombre de la etapa donde ocurrió el evento.
    columna:
        Columna asociada al evento. Si aplica a todo el dataset, usar "TODAS".
    descripcion:
        Explicación breve del problema o transformación.
    filas_afectadas:
        Cantidad de filas afectadas por la regla aplicada.
    accion:
        Acción tomada frente al problema detectado.
    """
    logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nivel": nivel,
        "etapa": etapa,
        "columna": columna,
        "descripcion": descripcion,
        "filas_afectadas": int(filas_afectadas),
        "accion": accion
    })
    
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

def cargar_dataset(logs: list) -> pd.DataFrame:
    """
    Carga el dataset de telco desde la carpeta data/raw/.
    si no se encuentra el archivo en esa ubicación, busca en la carpeta del raiz y lo copia a data/raw/.

    Retorna:
    Un DataFrame de pandas con los datos cargados.
    """
    archivo_en_carpeta_actual = PROJECT_ROOT / INPUT_FILENAME
    if not INPUT_FILE.exists() and archivo_en_carpeta_actual.exists():
        shutil.copy(archivo_en_carpeta_actual, INPUT_FILE)
        registrar_log(
            logs=logs,
            nivel="INFO",
            etapa="Carga de datos",
            columna="TODAS",
            descripcion="El archivo estaba en la carpeta raiz y fue copiado a data/raw/.",
            filas_afectadas=0,
            accion=f"Archivo copiado a {INPUT_FILE}"
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {INPUT_FILENAME} en {RAW_DIR}. "
            "Debes ubicar el archivo en data/raw/ o en la carpeta raíz antes de ejecutar el script."
        )

    df = pd.read_csv(INPUT_FILE)

    registrar_log(
        logs=logs,
        nivel="INFO",
        etapa="Carga de datos",
        columna="TODAS",
        descripcion="Dataset cargado correctamente desde data/raw/.",
        filas_afectadas=len(df),
        accion="Lectura de archivo CSV completada exitosamente."
    )

    return df

logs = []
preparar_carpetas()
df = cargar_dataset(logs)






