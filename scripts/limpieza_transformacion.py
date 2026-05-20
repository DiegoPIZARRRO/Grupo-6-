from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ingesta import PROCESSED_DIR, RAW_DIR

# Nombre del archivo de entrada y salida.
INPUT_FILENAME = "telco.csv"
OUTPUT_FILENAME = "telco_preprocesado.csv"

# Rutas completas.
INPUT_FILE = RAW_DIR / INPUT_FILENAME
OUTPUT_FILE = PROCESSED_DIR / OUTPUT_FILENAME

# Nombre de la variable objetivo.
TARGET_COL = "churn"

# ------------------------------------------------------------
# Parámetro principal de codificación
# ------------------------------------------------------------
# Cambia este valor según el tipo de codificación que quieras usar:
#
# "onehot" -> crea columnas binarias para cada categoría.
# "label"  -> reemplaza cada categoría por un número entero.
#
# Ejemplo:
# METODO_CODIFICACION = "onehot"
# METODO_CODIFICACION = "label"

METODO_CODIFICACION = "label"

def crear_onehot_encoder():
    """
    Crea un codificador OneHotEncoder compatible con distintas versiones de scikit-learn.

    En versiones recientes de scikit-learn se utiliza sparse_output=False.
    En versiones anteriores se utiliza sparse=False.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def cargar_dataset(ruta_archivo: Path) -> pd.DataFrame:
    """
    Lee el archivo CSV con los datos de telco y aplica las primeras limpiezas.

    Pasos realizados:
    1. Lee el archivo telco.csv.
    2. Normaliza los nombres de columnas:
       - elimina espacios al inicio y al final;
       - transforma los nombres a minúsculas.
    3. Filtra la variable churn para conservar solo valores YES y NO.
    4. Convierte churn a formato numérico:
       - SI se transforma en 1;
       - NO se transforma en 0.

    Retorna:
        DataFrame preprocesado en su etapa inicial.
    """
    if not ruta_archivo.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {ruta_archivo}. "
            "Verifica que esté en la carpeta data/raw o en la carpeta raíz."
        )

    df = pd.read_csv(ruta_archivo)
    
    # Normalizar nombres de columnas para evitar errores por mayúsculas o espacios.
    df.columns = [c.strip().lower() for c in df.columns]

    # Se covierte esta variable a númerica para evitar errores de formato   
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # La variable objetivo se limpia y se transforma a mayúsculas
    # para trabajar de forma uniforme con los valores SI y NO.
    df["churn"] = df["churn"].astype(str).str.strip().str.upper()

    # Se conservan únicamente los registros donde churn sea SI o NO.
    df = df[df["churn"].isin(["YES", "NO"])].copy()

    # Conversión de la variable objetivo a valores numéricos.
    df[TARGET_COL] = df[TARGET_COL].map({"YES": 1, "NO": 0})

    return df

def separar_variables(df: pd.DataFrame, target_col: str):
    """
    Separa el dataset en variables predictoras X y variable objetivo y.

    X contiene todas las columnas excepto la variable objetivo.
    y contiene solamente la variable objetivo.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y

def identificar_tipos_de_variables(X: pd.DataFrame):
    """
    Identifica automáticamente las columnas categóricas y numéricas.

    Las columnas categóricas son aquellas cuyo tipo de dato es object.
    Las columnas numéricas son todas las restantes.
    """
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    return categorical_features, numeric_features


def codificar_con_onehot(X: pd.DataFrame, categorical_features: list, numeric_features: list) -> pd.DataFrame:
    """
    Aplica One Hot Encoding a las variables categóricas.

    Este método crea nuevas columnas binarias para representar las categorías.
    Las variables numéricas se mantienen sin cambios mediante passthrough.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", crear_onehot_encoder(), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    X_transformado = preprocessor.fit_transform(X)

    # Obtener los nombres de las nuevas columnas generadas por OneHotEncoder.
    if categorical_features:
        cat_columns = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
    else:
        cat_columns = []

    final_columns = cat_columns + numeric_features

    X_codificado = pd.DataFrame(
        X_transformado,
        columns=final_columns,
        index=X.index
    )

    return X_codificado


def codificar_con_labelencoder(X: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """
    Aplica Label Encoding a las variables categóricas.

    Este método reemplaza cada categoría por un número entero.
    Por ejemplo, una columna con valores como Norte, Centro y Sur
    podría transformarse en 0, 1 y 2.

    Importante:
    Label Encoding no crea columnas nuevas. Mantiene la misma cantidad de columnas,
    pero convierte las categorías a números.
    """
    X_codificado = X.copy()

    for columna in categorical_features:
        encoder = LabelEncoder()

        # Se convierte a texto para evitar errores cuando existan valores faltantes.
        X_codificado[columna] = X_codificado[columna].astype(str).str.strip()

        # Cada categoría se reemplaza por un número entero.
        X_codificado[columna] = encoder.fit_transform(X_codificado[columna])

    return X_codificado

def codificar_variables(X: pd.DataFrame, metodo: str = "onehot") -> pd.DataFrame:
    """
    Codifica las variables categóricas usando el método indicado por parámetro.

    Parámetros:
        X:
            DataFrame con las variables predictoras.
        metodo:
            Tipo de codificación a utilizar.
            Valores permitidos:
            - "onehot"
            - "label"

    Retorna:
        DataFrame con las variables categóricas codificadas.
    """
    metodo = metodo.lower().strip()

    categorical_features, numeric_features = identificar_tipos_de_variables(X)

    print("Variables categóricas identificadas:")
    print(categorical_features)

    print("\nVariables numéricas identificadas:")
    print(numeric_features)

    if metodo == "onehot":
        print("\nMétodo seleccionado: One Hot Encoding")
        return codificar_con_onehot(X, categorical_features, numeric_features)

    if metodo == "label":
        print("\nMétodo seleccionado: Label Encoding")
        return codificar_con_labelencoder(X, categorical_features)

    raise ValueError(
        "Método de codificación no válido. "
        "Usa 'onehot' o 'label'."
    )

def guardar_dataset_preprocesado(X_codificado: pd.DataFrame, y: pd.Series, ruta_salida: Path) -> pd.DataFrame:
    """
    Une las variables predictoras codificadas con la variable objetivo
    y guarda el resultado en un archivo CSV.

    Retorna:
        DataFrame final preprocesado.
    """
    df_preprocesado = pd.concat(
        [
            X_codificado.reset_index(drop=True),
            y.reset_index(drop=True)
        ],
        axis=1
    )

    df_preprocesado.to_csv(ruta_salida, index=False)

    return df_preprocesado

def ejecutar_preprocesamiento(metodo_codificacion: str = "onehot") -> pd.DataFrame:
    """
    Ejecuta el flujo completo de preprocesamiento.

    Flujo:
    1. Cargar telco.csv.
    2. Limpiar columnas relevantes.
    3. Separar X e y.
    4. Codificar variables categóricas según el método elegido.
    5. Guardar telco_preprocesado.csv.

    Retorna:
        DataFrame preprocesado.
    """
    print("Iniciando preprocesamiento de datos...")

    df = cargar_dataset(INPUT_FILE)

    print(f"Dataset cargado correctamente con {len(df)} registros.")
    print(f"Columnas disponibles: {list(df.columns)}")

    X, y = separar_variables(df, TARGET_COL)

    X_codificado = codificar_variables(
        X,
        metodo=metodo_codificacion
    )

    df_preprocesado = guardar_dataset_preprocesado(
        X_codificado,
        y,
        OUTPUT_FILE
    )

    print("\nPreprocesamiento completado correctamente.")
    print(f"Archivo generado: {OUTPUT_FILE}")
    print(f"Filas del dataset final: {df_preprocesado.shape[0]}")
    print(f"Columnas del dataset final: {df_preprocesado.shape[1]}")

    return df_preprocesado

# ============================================================
# Ejecutar el preprocesamiento
# ============================================================

# Para cambiar el método de codificación, modifica el valor de
# METODO_CODIFICACION en la celda de configuración.
#
# Valores permitidos:
# - "onehot"
# - "label"

df_preprocesado = ejecutar_preprocesamiento(
    metodo_codificacion=METODO_CODIFICACION
)
