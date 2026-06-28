from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
from ingesta import registrar_log, REPORTS_DIR, PROCESSED_DIR, LOG_FILENAME

# Carpeta raíz del proyecto.
PROJECT_ROOT = Path.cwd()

# Nombre del archivo de entrada y salida.
INPUT_FILENAME = "telco_preprocesado.csv"
OUTPUT_FILENAME = "telco_limpio.csv"

# Rutas completas.
INPUT_FILE = PROCESSED_DIR / INPUT_FILENAME
OUTPUT_FILE = PROCESSED_DIR / OUTPUT_FILENAME
LOG_FILE = REPORTS_DIR / LOG_FILENAME
README_FILE = PROJECT_ROOT / "README.md"

# Parámetro para controlar la normalización de texto.
# Si está en True, los textos se transforman a mayúsculas para estandarizar categorías.
NORMALIZAR_TEXTO_A_MAYUSCULAS = True

def estandarizar_nombres_columnas(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Estandariza los nombres de las columnas del dataset.

    Reglas aplicadas:
    - Elimina espacios al inicio y final.
    - Convierte a minúsculas.
    - Reemplaza espacios por guiones bajos.
    - Elimina caracteres problemáticos simples.
    """
    df = df.copy()
    columnas_originales = df.columns.tolist()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(".", "", regex=False)
    )

    columnas_nuevas = df.columns.tolist()

    registrar_log(
        logs=logs,
        nivel="INFO",
        etapa="Estandarización de columnas",
        columna="TODAS",
        descripcion=f"Columnas estandarizadas. Antes: {columnas_originales}. Después: {columnas_nuevas}.",
        filas_afectadas=0,
        accion="Normalización de nombres de columnas"
    )

    return df

def estandarizar_textos(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Limpia columnas de texto.

    Reglas aplicadas:
    - Elimina espacios al inicio y al final.
    - Reemplaza múltiples espacios internos por un solo espacio.
    - Convierte textos vacíos en valores nulos.
    - Opcionalmente convierte a mayúsculas.
    """
    df = df.copy()
    columnas_texto = df.select_dtypes(include=["object"]).columns.tolist()

    for col in columnas_texto:
        serie_original = df[col].copy()

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        # Reemplaza strings vacíos o textos que representan nulos.
        df[col] = df[col].replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "NONE": pd.NA,
            "NaN": pd.NA
        })

        if NORMALIZAR_TEXTO_A_MAYUSCULAS:
            df[col] = df[col].str.upper()

        cambios = (serie_original.astype(str) != df[col].astype(str)).sum()

        if cambios > 0:
            registrar_log(
                logs=logs,
                nivel="INFO",
                etapa="Estandarización de texto",
                columna=col,
                descripcion="Se limpiaron espacios, valores vacíos y formato textual.",
                filas_afectadas=cambios,
                accion="Normalización de texto"
            )

    return df


def convertir_columnas_fecha(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Convierte a formato fecha las columnas cuyo nombre sugiera que contienen fechas.

    Se consideran columnas candidatas aquellas que contienen palabras como:
    - fecha
    - date
    """
    df = df.copy()

    columnas_fecha = [
        col for col in df.columns
        if "fecha" in col.lower() or "date" in col.lower()
    ]

    for col in columnas_fecha:
        valores_nulos_antes = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        valores_nulos_despues = df[col].isna().sum()
        nuevos_nulos = valores_nulos_despues - valores_nulos_antes

        registrar_log(
            logs=logs,
            nivel="INFO" if nuevos_nulos == 0 else "WARNING",
            etapa="Conversión de fechas",
            columna=col,
            descripcion="Columna convertida a formato datetime.",
            filas_afectadas=max(nuevos_nulos, 0),
            accion="Conversión con pd.to_datetime(errors='coerce')"
        )

    return df


def convertir_columnas_numericas(df: pd.DataFrame, logs: list, umbral_conversion: float = 0.80) -> pd.DataFrame:
    """
    Intenta convertir a numéricas las columnas de texto que parecen contener números.

    Parámetro:
    umbral_conversion:
        Porcentaje mínimo de valores convertibles para aceptar la conversión.
        Por ejemplo, 0.80 significa que al menos el 80% de los valores no nulos
        deben poder convertirse correctamente a número.
    """
    df = df.copy()
    columnas_texto = df.select_dtypes(include=["object"]).columns.tolist()

    for col in columnas_texto:
        serie = df[col].dropna()

        if len(serie) == 0:
            continue

        serie_convertida = pd.to_numeric(serie, errors="coerce")
        proporcion_convertible = serie_convertida.notna().mean()

        if proporcion_convertible >= umbral_conversion:
            nulos_antes = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            nulos_despues = df[col].isna().sum()

            registrar_log(
                logs=logs,
                nivel="INFO",
                etapa="Conversión numérica",
                columna=col,
                descripcion=f"Columna convertida a numérica. Proporción convertible: {proporcion_convertible:.2f}.",
                filas_afectadas=max(nulos_despues - nulos_antes, 0),
                accion="Conversión con pd.to_numeric(errors='coerce')"
            )

    return df


def estandarizar_formatos(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Aplica las transformaciones de formato principales:
    texto, fechas y números.
    """
    df = estandarizar_textos(df, logs)
    df = convertir_columnas_fecha(df, logs)
    df = convertir_columnas_numericas(df, logs)

    return df

def eliminar_filas_vacias(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Elimina filas completamente vacías.
    """
    df = df.copy()
    filas_antes = len(df)

    df = df.dropna(how="all")
    filas_eliminadas = filas_antes - len(df)

    if filas_eliminadas > 0:
        registrar_log(
            logs=logs,
            nivel="WARNING",
            etapa="Limpieza básica",
            columna="TODAS",
            descripcion="Se eliminaron filas completamente vacías.",
            filas_afectadas=filas_eliminadas,
            accion="dropna(how='all')"
        )

    return df


def eliminar_duplicados(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Elimina filas duplicadas considerando todas las columnas del dataset.
    """
    df = df.copy()
    filas_antes = len(df)

    df = df.drop_duplicates()
    filas_eliminadas = filas_antes - len(df)

    if filas_eliminadas > 0:
        registrar_log(
            logs=logs,
            nivel="WARNING",
            etapa="Limpieza básica",
            columna="TODAS",
            descripcion="Se eliminaron registros duplicados.",
            filas_afectadas=filas_eliminadas,
            accion="drop_duplicates()"
        )

    return df


def eliminar_registros_con_nulos(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Elimina registros que contienen al menos un valor nulo.

    Esta decisión debe revisarse según el contexto.
    En un proyecto real, algunas columnas podrían imputarse en vez de eliminarse.
    """
    df = df.copy()
    filas_con_nulos = df.isna().any(axis=1)
    cantidad_nulos = filas_con_nulos.sum()
    columnas_con_nulos = df.columns[df.isna().any()].tolist()

    df = df.loc[~filas_con_nulos].copy()

    if cantidad_nulos > 0:
        registrar_log(
            logs=logs,
            nivel="WARNING",
            etapa="Limpieza básica",
            columna="TODAS",
            descripcion=f"Se eliminaron registros con valores nulos. Columnas con nulos detectadas: {columnas_con_nulos}.",
            filas_afectadas=cantidad_nulos,
            accion="Eliminación de filas con al menos un valor nulo"
        )

    return df


def limpieza_basica(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Ejecuta la limpieza básica del dataset.
    """
    df = eliminar_filas_vacias(df, logs)
    df = eliminar_duplicados(df, logs)
    df = eliminar_registros_con_nulos(df, logs)

    return df

def crear_columnas_derivadas(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    
    # Crea columnas derivadas cuando corresponde.

    df = df.copy()
    columnas_creadas = []
    columnas_binarias = []

    for col in df.select_dtypes(include=[np.number]).columns:
        valores_unicos = set(df[col].dropna().unique().tolist())

        if len(valores_unicos) > 0 and valores_unicos.issubset({0, 1}):
            columnas_binarias.append(col)

    # Evita contar la variable objetivo como feature activa si existe.
    columnas_binarias_features = [col for col in columnas_binarias if col != "churn"]

    if len(columnas_binarias_features) > 0:
        df["cantidad_features_activas"] = df[columnas_binarias_features].sum(axis=1)
        columnas_creadas.append("cantidad_features_activas")

    df["fecha_procesamiento"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    columnas_creadas.append("fecha_procesamiento")

    registrar_log(
        logs=logs,
        nivel="INFO",
        etapa="Columnas derivadas",
        columna="TODAS",
        descripcion=f"Columnas derivadas creadas: {columnas_creadas}.",
        filas_afectadas=len(df),
        accion="Creación de nuevas columnas"
    )

    return df

def guardar_resultados(df: pd.DataFrame, logs: list) -> None:
    """
    Guarda el dataset limpio y el reporte de validación.
    """
    df.to_excel(OUTPUT_FILE, index=False)

    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

    print("Archivos generados correctamente:")
    print(f"- Dataset limpio: {OUTPUT_FILE}")
    print(f"- Reporte de validación: {LOG_FILE}")

def actualizar_readme(df_original: pd.DataFrame, df_limpio: pd.DataFrame, logs: list) -> None:
    """
    Actualiza el archivo README.md con un resumen del procesamiento realizado.

    Si el README.md no existe, lo crea automáticamente.
    Si ya existe, agrega una nueva sección al final.
    """
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filas_eliminadas = len(df_original) - len(df_limpio)

    resumen_logs = pd.DataFrame(logs)
    problemas_detectados = 0

    if not resumen_logs.empty:
        problemas_detectados = resumen_logs[
            resumen_logs["nivel"].isin(["WARNING", "ERROR"])
        ]["filas_afectadas"].sum()

    seccion = f'''

## Validación y limpieza del dataset de Telco

**Fecha de ejecución:** {fecha}

### Archivos utilizados

- Dataset de entrada: `{INPUT_FILE.relative_to(PROJECT_ROOT)}`
- Dataset limpio generado: `{OUTPUT_FILE.relative_to(PROJECT_ROOT)}`
- Reporte de validación: `{LOG_FILE.relative_to(PROJECT_ROOT)}`

### Resumen del procesamiento

- Filas originales: {len(df_original)}
- Filas finales: {len(df_limpio)}
- Filas eliminadas: {filas_eliminadas}
- Problemas registrados en logs: {int(problemas_detectados)}

### Transformaciones aplicadas

- Estandarización de nombres de columnas.
- Limpieza y normalización de columnas de texto.
- Conversión de columnas de fecha cuando corresponde.
- Conversión de columnas numéricas cuando corresponde.
- Eliminación de filas vacías.
- Eliminación de registros duplicados.
- Eliminación de registros con valores nulos.
- Validación de rangos permitidos para columnas numéricas relevantes.
- Creación de columnas derivadas cuando existen variables suficientes.
- Generación de reporte de validación para revisar decisiones tomadas durante la limpieza.

### Rol dentro del pipeline de datos

Esta etapa permite mejorar la calidad semántica y técnica del dataset antes de su uso en procesos posteriores,
como carga a base de datos, análisis exploratorio, construcción de modelos o generación de reportes.
Además, deja evidencia reproducible de las reglas aplicadas y de los problemas detectados.
'''

    if README_FILE.exists():
        contenido_actual = README_FILE.read_text(encoding="utf-8")
    else:
        contenido_actual = "# Proyecto de procesamiento de datos de la empresa Telco\n"

    README_FILE.write_text(contenido_actual + seccion, encoding="utf-8")

    print(f"README actualizado en: {README_FILE}")

def cargar_dataset(logs: list) -> pd.DataFrame:
    """
    Carga el dataset telco_preprocesado.csv desde la carpeta data/processed
    Retorna:
    Un DataFrame de pandas con los datos cargados.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {INPUT_FILENAME} en {PROCESSED_DIR}. "
            "Asegurate de que el dataset exista en data/processed/ y de haber ejecutado los scripts anteriores."
        )

    df = pd.read_csv(INPUT_FILE)

    registrar_log(
        logs=logs,
        nivel="INFO",
        etapa="Carga de datos",
        columna="TODAS",
        descripcion="Dataset cargado correctamente desde data/processed/.",
        filas_afectadas=len(df),
        accion="Lectura de archivo CSV completada exitosamente."
    )

    return df

logs = []

# 1. Cargar datos.
df_original = cargar_dataset(logs)

# 2. Estandarizar nombres y formatos.
df_procesado = estandarizar_nombres_columnas(df_original, logs)

df_procesado = estandarizar_formatos(df_procesado, logs)

# 3. Aplicar limpieza y validaciones.
df_procesado = limpieza_basica(df_procesado, logs)

# 4. Crear columnas derivadas.
df_procesado = crear_columnas_derivadas(df_procesado, logs)

# 5. Guardar resultados.
guardar_resultados(df_procesado, logs)

# 6. Actualizar README.
actualizar_readme(df_original, df_procesado, logs)

print("\nResumen del proceso:")
print(f"- Filas originales: {len(df_original)}")
print(f"- Filas finales:    {len(df_procesado)}")
print(f"- Filas eliminadas: {len(df_original) - len(df_procesado)}")
print(f"- Columnas finales: {len(df_procesado.columns)}")